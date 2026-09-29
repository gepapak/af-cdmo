from __future__ import annotations

import numpy as np

from scripts.af_cdmo_real_extension_v13 import (
    RealAffineGaugeDataset,
    RealAffineGeometry,
    parse_presentation,
    presentation_names,
)
from scripts.multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise


ZONES = (
    "DK1",
    "DK1_CO",
    "DK1_DE",
    "DK1_KS",
    "DK1_SB",
    "DK1_SK",
    "DK2",
    "DK2_KO",
    "DK2_SB",
    "FI",
    "FI_EL",
    "FI_FS",
    "NO1",
    "NO2",
    "NO2_ND",
    "NO2_NK",
    "NO2_SK",
    "NO3",
    "NO4",
    "NO5",
    "SE1",
    "SE2",
    "SE3",
    "SE3_FS",
    "SE3_KS",
    "SE3_SWL",
    "SE4",
    "SE4_BC",
    "SE4_NB",
    "SE4_SP",
    "SE4_SWL",
)
OBSERVED = ("DK1", "DK2", "FI", "NO1", "NO2", "SE3", "SE4")


def _sample(seed: int = 7) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = 8
    normals = rng.normal(size=(rows, len(ZONES)))
    shadow = rng.lognormal(size=rows)
    rhs = rng.normal(500.0, 50.0, size=rows)
    raw_tail = rng.normal(size=(rows, 4))
    norm = np.linalg.norm(normals, axis=1)
    unit = normals / norm[:, None]
    canonical_tail = rng.normal(size=(rows, 4))
    canonical = np.column_stack([unit, np.arcsinh(rhs / norm / 500.0), canonical_tail])
    mass = shadow * norm
    indices = np.asarray([ZONES.index(zone) for zone in OBSERVED])
    analytic = center_price_field(-(mass @ unit[:, indices]))
    target = analytic + center_price_field(rng.normal(0.0, 1.0, len(OBSERVED)))
    return {
        "timestamp": 0,
        "target_vector": target.astype(np.float32),
        "raw": np.column_stack([shadow, normals, rhs, raw_tail]).astype(np.float32),
        "canonical": canonical.astype(np.float32),
        "weights": (mass / mass.sum()).astype(np.float32),
        "total": float(mass.sum()),
        "analytic_vector": analytic.astype(np.float32),
    }


def _dataset(presentation: str, representation: str) -> RealAffineGaugeDataset:
    geometry = RealAffineGeometry.loss_safe(ZONES)
    indices = np.asarray([ZONES.index(zone) for zone in OBSERVED])
    return RealAffineGaugeDataset(
        [_sample()],
        raw_scale=np.ones(1 + len(ZONES) + 5),
        target_scale=1.0,
        presentation=presentation,
        representation=representation,
        real_zone_indices=indices,
        geometry=geometry,
        canonical_ram_scale=500.0,
        analytic_tolerance=2.0e-6,
    )


def test_presentation_registry_is_parseable_and_unique() -> None:
    geometry = RealAffineGeometry.loss_safe(ZONES)
    names = presentation_names(geometry)
    assert len(names) == len(set(names))
    assert "gauge_internal_hvdc_storebaelt" in names
    assert "gauge_internal_hvdc_skagerak" not in names
    for name in names:
        assert parse_presentation(name, geometry).name == name


def test_af_features_are_invariant_to_registered_equality_gauges() -> None:
    reference = _dataset("original", "af")[0]
    for presentation in (
        "af_projected_representative",
        "equality_reflection",
        "random_equality_gauge",
        "gauge_internal_hvdc_storebaelt",
        "gauge_internal_hvdc_fennoskan",
        "gauge_internal_hvdc_kontiskan",
        "gauge_internal_hvdc_southwestlink",
    ):
        candidate = _dataset(presentation, "af")[0]
        np.testing.assert_allclose(candidate["canonical"], reference["canonical"], atol=2e-6)
        np.testing.assert_allclose(candidate["weights"], reference["weights"], atol=2e-6)
        np.testing.assert_allclose(candidate["analytic_vector"], reference["analytic_vector"], atol=2e-6)
        assert abs(candidate["total"] - reference["total"]) < 2e-5


def test_internal_hvdc_gauge_is_not_removed_by_slack_only_representation() -> None:
    reference = _dataset("original", "slack")[0]
    candidate = _dataset("gauge_internal_hvdc_storebaelt", "slack")[0]
    assert np.max(np.abs(candidate["canonical"] - reference["canonical"])) > 1e-4
    np.testing.assert_allclose(
        field_to_pairwise(candidate["analytic_vector"]),
        field_to_pairwise(reference["analytic_vector"]),
        atol=2e-6,
    )


def test_internal_hvdc_gauge_is_visible_in_ambient_representation() -> None:
    reference = _dataset("original", "ambient")[0]
    candidate = _dataset("gauge_internal_hvdc_storebaelt", "ambient")[0]
    assert np.max(np.abs(candidate["canonical"] - reference["canonical"])) > 1e-4
    np.testing.assert_allclose(
        field_to_pairwise(candidate["analytic_vector"]),
        field_to_pairwise(reference["analytic_vector"]),
        atol=2e-6,
    )
