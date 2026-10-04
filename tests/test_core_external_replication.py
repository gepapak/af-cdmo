from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from af_cdmo import (
    AffineFeasibleGeometry,
    canonicalize_certificate,
    equality_gauge_scale_rewrite,
)
from scripts.core_external_replication import (
    CORE_PHYSICAL_ZONES,
    CoreSchemaError,
    canonicalize_shadow_rows,
    core_local_delivery_windows,
    global_balance_geometry,
    infer_spread_convention,
    reconstruct_centered_price_field,
)
from scripts.af_cdmo_real_extension_v13 import RealAffineGaugeDataset, RealAffineGeometry
from scripts.multizone_quotient_gauge_v6 import MultiZoneRegressor, collate_multizone
from scripts.download_jao_core_external_replication import _payload_is_causal
from scripts.benchmark_af_cdmo_core_external import MaterializedDataset


def _field() -> np.ndarray:
    values = np.linspace(-11.0, 17.0, len(CORE_PHYSICAL_ZONES))
    return values - values.mean()


def _shadow_row(*, negative: bool = False, omit_zone: str | None = None) -> dict[str, object]:
    field = _field()
    row: dict[str, object] = {
        "id": "row-1",
        "dateTimeUtc": "2025-01-02T00:00:00Z",
        "_publication_utc": "2025-01-01T12:30:00Z",
        "shadowPrice": -1.0 if negative else 1.0,
        "ram": 1000.0,
        "ramMcp": 980.0,
        "fmax": 1200.0,
        "fref": 25.0,
        "f0all": 15.0,
        "cnecName": "synthetic",
        "contName": "basecase",
    }
    for zone, value in zip(CORE_PHYSICAL_ZONES, -field, strict=True):
        if zone != omit_zone:
            row[f"hub_{zone}"] = float(value)
    return row


def _spread_row(*, break_reverse: bool = False, disconnect: str | None = None) -> dict[str, object]:
    field = _field()
    row: dict[str, object] = {"dateTimeUtc": "2025-01-02T00:00:00Z"}
    for left_index, left in enumerate(CORE_PHYSICAL_ZONES):
        for right_index, right in enumerate(CORE_PHYSICAL_ZONES):
            if left == right or left == disconnect or right == disconnect:
                continue
            # The fixture uses JAO's destination-minus-source convention.
            row[f"border_{left}_{right}"] = float(field[right_index] - field[left_index])
    if break_reverse:
        row["border_BE_AT"] = float(row["border_BE_AT"]) + 1.0
    return row


def test_core_shadow_canonicalization_preserves_physical_coordinate_order() -> None:
    frame, audit = canonicalize_shadow_rows([_shadow_row()])
    assert audit["physical_hub_count"] == 12
    assert audit["virtual_hub_count"] == 0
    unit_columns = [column for column in frame if column.startswith("unit_hub_")]
    assert unit_columns == [f"unit_hub_{zone}" for zone in CORE_PHYSICAL_ZONES]
    assert frame.loc[0, "dual_mass_eur_mwh"] > 0.0
    assert audit["causal_publication_share"] == 1.0


def test_core_shadow_schema_fails_closed() -> None:
    with pytest.raises(CoreSchemaError, match="omitted hubs"):
        canonicalize_shadow_rows([_shadow_row(omit_zone="SK")])
    with pytest.raises(CoreSchemaError, match="non-negative"):
        canonicalize_shadow_rows([_shadow_row(negative=True)])


def test_core_delivery_windows_follow_local_midnight_and_dst() -> None:
    winter = core_local_delivery_windows(
        "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"
    )
    assert winter == (
        (
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01T23:00:00Z"),
        ),
        (
            pd.Timestamp("2025-01-01T23:00:00Z"),
            pd.Timestamp("2025-01-02T00:00:00Z"),
        ),
    )
    spring = core_local_delivery_windows(
        "2025-03-29T23:00:00Z", "2025-03-30T22:00:00Z"
    )
    assert len(spring) == 1
    assert spring[0][1] - spring[0][0] == pd.Timedelta(hours=23)
    autumn = core_local_delivery_windows(
        "2025-10-25T22:00:00Z", "2025-10-26T23:00:00Z"
    )
    assert len(autumn) == 1
    assert autumn[0][1] - autumn[0][0] == pd.Timedelta(hours=25)


def test_response_level_publication_timing_fails_closed() -> None:
    causal = {
        "lastModifiedOn": "2025-01-01T12:00:00Z",
        "rows": [{"dateTimeUtc": "2025-01-02T00:00:00Z"}],
    }
    revised = {
        "lastModifiedOn": "2025-01-10T12:00:00Z",
        "rows": [{"dateTimeUtc": "2025-01-02T00:00:00Z"}],
    }
    missing = {
        "lastModifiedOn": None,
        "rows": [{"dateTimeUtc": "2025-01-02T00:00:00Z"}],
    }
    assert _payload_is_causal(causal)
    assert not _payload_is_causal(revised)
    assert not _payload_is_causal(missing)
    assert _payload_is_causal({"lastModifiedOn": None, "rows": []})


def test_price_graph_reconstructs_centered_field_and_orientation() -> None:
    spread = pd.DataFrame([_spread_row()])
    panel, diagnostics = reconstruct_centered_price_field(
        spread, convention="destination_minus_source"
    )
    recovered = panel[
        [f"price_{zone}_centered_eur_mwh" for zone in CORE_PHYSICAL_ZONES]
    ].to_numpy()[0]
    np.testing.assert_allclose(recovered, _field(), atol=1.0e-11)
    assert diagnostics.minimum_graph_rank == len(CORE_PHYSICAL_ZONES) - 1
    assert diagnostics.cycle_residual_max_eur_mwh < 1.0e-10
    assert diagnostics.reverse_antisymmetry_max_eur_mwh < 1.0e-12


def test_price_graph_rejects_bad_reverse_and_disconnection() -> None:
    with pytest.raises(CoreSchemaError, match="anti-symmetry"):
        reconstruct_centered_price_field(
            pd.DataFrame([_spread_row(break_reverse=True)]),
            convention="destination_minus_source",
        )
    with pytest.raises(CoreSchemaError, match="Disconnected"):
        reconstruct_centered_price_field(
            pd.DataFrame([_spread_row(disconnect="SK")]),
            convention="destination_minus_source",
        )


def test_sign_inference_is_decisive_on_schema_audit_prefix() -> None:
    deliveries = pd.date_range("2025-01-02", periods=24, freq="h", tz="UTC")
    shadow_rows = []
    spread_rows = []
    for index, delivery in enumerate(deliveries):
        shadow = _shadow_row()
        spread = _spread_row()
        shadow["id"] = f"row-{index}"
        shadow["dateTimeUtc"] = delivery.isoformat()
        shadow["_publication_utc"] = (delivery - pd.Timedelta(hours=12)).isoformat()
        spread["dateTimeUtc"] = delivery.isoformat()
        shadow_rows.append(shadow)
        spread_rows.append(spread)
    dual, _ = canonicalize_shadow_rows(shadow_rows)
    convention, audit, panel = infer_spread_convention(
        pd.DataFrame(spread_rows),
        dual,
        audit_end=pd.Timestamp("2025-01-03T00:00:00Z"),
        minimum_overlap=12,
    )
    assert convention == "destination_minus_source"
    assert audit["selected_to_rejected_ratio"] < 1.0e-10
    assert len(panel) == 24


def test_global_balance_quotient_is_exact_under_scale_and_gauge() -> None:
    names = tuple(CORE_PHYSICAL_ZONES) + ("DE_DK1_VH",)
    equality, tangent = global_balance_geometry(names)
    geometry = AffineFeasibleGeometry.from_equalities(equality, np.zeros(1))
    np.testing.assert_allclose(geometry.tangent_projector, tangent, atol=1.0e-13)
    rng = np.random.default_rng(7)
    normals = rng.normal(size=(5, len(names)))
    rhs = rng.normal(size=5)
    dual = rng.uniform(0.1, 3.0, size=5)
    tail = rng.normal(size=(5, 4))
    first = canonicalize_certificate(geometry, normals, rhs, dual, tail)
    changed_normal, changed_rhs, changed_dual = equality_gauge_scale_rewrite(
        geometry,
        normals,
        rhs,
        dual,
        scales=np.exp(rng.uniform(-2.0, 2.0, size=5)),
        gauge_coefficients=rng.normal(size=(5, 1)),
    )
    second = canonicalize_certificate(
        geometry, changed_normal, changed_rhs, changed_dual, tail
    )
    np.testing.assert_allclose(second.directions, first.directions, atol=2.0e-12)
    np.testing.assert_allclose(second.normalized_rhs, first.normalized_rhs, atol=2.0e-12)
    np.testing.assert_allclose(second.masses, first.masses, atol=2.0e-12)
    np.testing.assert_allclose(second.dual_current, first.dual_current, atol=2.0e-12)


def test_core_af_dataset_and_network_are_invariant_to_registered_presentations() -> None:
    hub_names = tuple(CORE_PHYSICAL_ZONES) + ("DE_DK1_VH",)
    equality, tangent = global_balance_geometry(hub_names)
    geometry = RealAffineGeometry(
        zone_names=hub_names,
        equality_names=("global_net_position_balance",),
        equality_matrix=equality,
        tangent_projector=tangent,
        rank=1,
    )
    rng = np.random.default_rng(42)
    normals = rng.normal(size=(4, len(hub_names)))
    norms = np.linalg.norm(normals, axis=1)
    shadow = rng.uniform(0.2, 2.0, size=4)
    rhs = rng.uniform(300.0, 900.0, size=4)
    raw_tail = rng.normal(size=(4, 4))
    unit = normals / norms[:, None]
    masses = shadow * norms
    physical = np.arange(len(CORE_PHYSICAL_ZONES), dtype=np.int64)
    analytic = -(masses @ unit[:, physical])
    analytic -= analytic.mean()
    sample = {
        "timestamp": pd.Timestamp("2025-01-02T00:00:00Z"),
        "target_vector": (analytic + rng.normal(0.0, 0.1, size=len(analytic))).astype(
            np.float32
        ),
        "raw": np.column_stack([shadow, normals, rhs, raw_tail]).astype(np.float32),
        "canonical": np.column_stack(
            [unit, np.arcsinh((rhs / norms)[:, None]), raw_tail]
        ).astype(np.float32),
        "weights": (masses / masses.sum()).astype(np.float32),
        "total": float(masses.sum()),
        "analytic_vector": analytic.astype(np.float32),
    }
    raw_dim = 1 + len(hub_names) + 5
    canonical_dim = len(hub_names) + 5
    datasets = {
        presentation: RealAffineGaugeDataset(
            [sample],
            raw_scale=np.ones(raw_dim),
            target_scale=1.0,
            presentation=presentation,
            representation="af",
            real_zone_indices=physical,
            geometry=geometry,
            canonical_ram_scale=1.0,
        )
        for presentation in (
            "original",
            "combined",
            "slack_AT",
            "random_equality_gauge",
        )
    }
    materialized = MaterializedDataset(datasets["original"])
    np.testing.assert_array_equal(
        materialized[0]["canonical"], datasets["original"][0]["canonical"]
    )
    np.testing.assert_array_equal(
        materialized[0]["raw_scaled"], datasets["original"][0]["raw_scaled"]
    )
    torch.manual_seed(7)
    model = MultiZoneRegressor(
        "cqdm_gauge_residual",
        raw_dim,
        canonical_dim,
        len(CORE_PHYSICAL_ZONES),
    )
    model.eval()
    predictions = {}
    with torch.no_grad():
        for presentation, dataset in datasets.items():
            batch = collate_multizone([dataset[0]])
            predictions[presentation] = model(batch, torch.device("cpu")).numpy()
    for presentation, prediction in predictions.items():
        np.testing.assert_allclose(
            prediction,
            predictions["original"],
            atol=2.0e-6,
            err_msg=presentation,
        )
