from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.af_cdmo_real_controls_v14 import (
    HELD_OUT_RANDOM_GAUGE_AMPLITUDE,
    FixedEqualityGaugeDataset,
    GaugeAugmentedAmbientDataset,
    NonEquivalentRAMDataset,
    NonuniformRefinementAFDataset,
    UniformMassDataset,
    control_contract,
)
from scripts.af_cdmo_real_extension_v13 import RealAffineGaugeDataset, RealAffineGeometry
from scripts.benchmark_af_cdmo_real_v14 import (
    FROZEN_V11_REPORTS,
    NEURAL_METHODS,
    _conditioning_ratio,
    _frozen_v11_reproduction_contract,
    _rank_balanced_groups,
)
from scripts.multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise


ZONES = (
    "DK1", "DK1_CO", "DK1_DE", "DK1_KS", "DK1_SB", "DK1_SK",
    "DK2", "DK2_KO", "DK2_SB", "FI", "FI_EL", "FI_FS", "NO1",
    "NO2", "NO2_ND", "NO2_NK", "NO2_SK", "NO3", "NO4", "NO5",
    "SE1", "SE2", "SE3", "SE3_FS", "SE3_KS", "SE3_SWL", "SE4",
    "SE4_BC", "SE4_NB", "SE4_SP", "SE4_SWL",
)
OBSERVED = ("DK1", "DK2", "FI", "NO1", "NO2", "SE3", "SE4")


def _sample(seed: int = 7) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = 9
    normals = rng.normal(size=(rows, len(ZONES)))
    shadow = rng.lognormal(size=rows)
    rhs = rng.normal(500.0, 50.0, size=rows)
    raw_tail = rng.normal(size=(rows, 4))
    norms = np.linalg.norm(normals, axis=1)
    unit = normals / norms[:, None]
    canonical_tail = rng.normal(size=(rows, 4))
    canonical = np.column_stack(
        [unit, np.arcsinh(rhs / norms / 500.0), canonical_tail]
    )
    masses = shadow * norms
    indices = np.asarray([ZONES.index(zone) for zone in OBSERVED])
    analytic = center_price_field(-(masses @ unit[:, indices]))
    target = analytic + center_price_field(rng.normal(0.0, 1.0, len(OBSERVED)))
    return {
        "timestamp": 0,
        "target_vector": target.astype(np.float32),
        "raw": np.column_stack([shadow, normals, rhs, raw_tail]).astype(np.float32),
        "canonical": canonical.astype(np.float32),
        "weights": (masses / masses.sum()).astype(np.float32),
        "total": float(masses.sum()),
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


def test_v14_registers_both_fairness_controls() -> None:
    assert NEURAL_METHODS["af_uniform_attention_residual"] == ("af", "uniform")
    assert NEURAL_METHODS["ambient_gauge_augmented_residual"] == (
        "ambient",
        "augmented",
    )


def test_uniform_mass_changes_only_attention_weights() -> None:
    base = _dataset("original", "af")[0]
    uniform = UniformMassDataset(_dataset("original", "af"))[0]
    np.testing.assert_allclose(uniform["canonical"], base["canonical"])
    np.testing.assert_allclose(uniform["analytic_vector"], base["analytic_vector"])
    assert uniform["total"] == base["total"]
    np.testing.assert_allclose(
        uniform["weights"], np.full(len(base["weights"]), 1.0 / len(base["weights"]))
    )


def test_uniform_mass_is_not_split_merge_invariant_but_dual_mass_is() -> None:
    original_base = _dataset("original", "af")[0]
    refined_dataset = NonuniformRefinementAFDataset(_dataset("original", "af"))
    combined_base = refined_dataset[0]
    original_uniform = UniformMassDataset(_dataset("original", "af"))[0]
    combined_uniform = UniformMassDataset(refined_dataset)[0]

    original_dual_mean = original_base["weights"] @ original_base["canonical"]
    combined_dual_mean = combined_base["weights"] @ combined_base["canonical"]
    np.testing.assert_allclose(original_dual_mean, combined_dual_mean, atol=2e-6)

    original_uniform_mean = (
        original_uniform["weights"] @ original_uniform["canonical"]
    )
    combined_uniform_mean = (
        combined_uniform["weights"] @ combined_uniform["canonical"]
    )
    assert np.max(np.abs(original_uniform_mean - combined_uniform_mean)) > 1e-5


def test_gauge_augmentation_is_deterministic_causal_and_conservative() -> None:
    augmented = GaugeAugmentedAmbientDataset(_dataset("original", "ambient"), seed=19)
    augmented.set_epoch(3)
    first = augmented[0]
    second = augmented[0]
    np.testing.assert_allclose(first["canonical"], second["canonical"])

    augmented.set_epoch(4)
    changed = augmented[0]
    assert np.max(np.abs(first["canonical"] - changed["canonical"])) > 1e-5
    reference = _dataset("original", "ambient")[0]
    np.testing.assert_allclose(
        field_to_pairwise(first["analytic_vector"]),
        field_to_pairwise(reference["analytic_vector"]),
        atol=2e-6,
    )


def test_augmentation_rejects_the_held_out_evaluation_amplitude() -> None:
    with pytest.raises(ValueError, match="held-out"):
        GaugeAugmentedAmbientDataset(
            _dataset("original", "ambient"),
            amplitudes=(HELD_OUT_RANDOM_GAUGE_AMPLITUDE,),
        )


def test_fixed_gauge_is_removed_only_by_the_af_projection() -> None:
    af_reference = _dataset("original", "af")[0]
    af_gauged = FixedEqualityGaugeDataset(
        _dataset("original", "af"), amplitude=6.0
    )[0]
    np.testing.assert_allclose(af_gauged["canonical"], af_reference["canonical"], atol=2e-6)
    np.testing.assert_allclose(af_gauged["weights"], af_reference["weights"], atol=2e-6)

    ambient_reference = _dataset("original", "ambient")[0]
    ambient_gauged = FixedEqualityGaugeDataset(
        _dataset("original", "ambient"), amplitude=6.0
    )[0]
    assert np.max(
        np.abs(ambient_gauged["canonical"] - ambient_reference["canonical"])
    ) > 1e-4


def test_non_equivalent_ram_control_changes_only_the_ram_coordinate() -> None:
    base_dataset = _dataset("original", "af")
    base = base_dataset[0]
    changed = NonEquivalentRAMDataset(base_dataset, delta=0.2)[0]
    difference = changed["canonical"] - base["canonical"]
    np.testing.assert_allclose(
        difference[:, base_dataset.ptdf_count], np.full(len(difference), 0.2), atol=1e-6
    )
    other = np.delete(difference, base_dataset.ptdf_count, axis=1)
    np.testing.assert_allclose(other, 0.0, atol=1e-7)


def test_control_contract_records_held_out_families() -> None:
    contract = control_contract()
    assert contract["ambient_gauge_augmentation"][
        "held_out_random_gauge_amplitude"
    ] == HELD_OUT_RANDOM_GAUGE_AMPLITUDE
    assert "mass-conserving split/merge" in contract[
        "ambient_gauge_augmentation"
    ]["held_out_families"]


def test_rank_balanced_conditioning_groups_survive_tied_values() -> None:
    values = np.asarray([0.1] * 6 + [0.4] * 7 + [0.9] * 8)
    groups = _rank_balanced_groups(values, bins=5)
    assert len(groups) == 5
    assert all(len(group) > 0 for group in groups)
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    combined = np.concatenate(groups)
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(values)))


def test_conditioning_ratio_excludes_empty_certificates_only() -> None:
    geometry = RealAffineGeometry.loss_safe(ZONES)
    empty = {"raw": np.empty((0, 1 + len(ZONES) + 5))}
    assert _conditioning_ratio(empty, geometry, len(ZONES)) is None

    ratio = _conditioning_ratio(_sample(), geometry, len(ZONES))
    assert ratio is not None
    assert np.isfinite(ratio)
    assert 0.0 <= ratio <= 1.0 + 1.0e-12


def _reproduction_method(threshold: float) -> dict[str, object]:
    return {
        "validation_max_pair_error_alert_threshold_eur_mwh": threshold,
        "stresses": {
            "original": {
                "mean_pairwise_mae_eur_mwh": 2.5,
                "p95_timestamp_pairwise_mae_eur_mwh": 11.0,
                "mean_centered_zone_mae_eur_mwh": 2.0,
                "pairwise_rmse_eur_mwh": 10.0,
                "seed_members": [
                    {"seed": 7, "mean_pairwise_mae_eur_mwh": 2.4},
                    {"seed": 42, "mean_pairwise_mae_eur_mwh": 2.6},
                ],
            }
        },
    }


def test_frozen_reproduction_uses_strict_mixed_numerical_tolerance(
    tmp_path, monkeypatch
) -> None:
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "study_role": "development",
                    "zones": ["DK1", "DK2", "FI"],
                    "seeds": [7, 42],
                    "fit_end_exclusive": "2025-01-01T00:00:00+00:00",
                    "internal_validation_end_exclusive": "2025-02-01T00:00:00+00:00",
                    "evaluation_end_exclusive": "2025-03-01T00:00:00+00:00",
                },
                "results": {
                    "ambient_cqdm_residual": _reproduction_method(20.0),
                    "slack_qdm_residual": _reproduction_method(20.0),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(FROZEN_V11_REPORTS, "development", reference_path)
    results = {
        "ambient_cqdm_residual": _reproduction_method(20.000025),
        "slack_qdm_residual": _reproduction_method(20.0),
    }
    contract = _frozen_v11_reproduction_contract(
        study_role="development",
        zones=("DK1", "DK2", "FI"),
        seeds=[7, 42],
        fit_end=pd.Timestamp("2025-01-01T00:00:00Z"),
        validation_end=pd.Timestamp("2025-02-01T00:00:00Z"),
        evaluation_end=pd.Timestamp("2025-03-01T00:00:00Z"),
        results=results,
    )
    assert contract["status"] == "passed"
    assert 0.0 < contract["max_normalized_error"] < 1.0

    results["ambient_cqdm_residual"] = _reproduction_method(20.001)
    with pytest.raises(RuntimeError, match="max_normalized_error"):
        _frozen_v11_reproduction_contract(
            study_role="development",
            zones=("DK1", "DK2", "FI"),
            seeds=[7, 42],
            fit_end=pd.Timestamp("2025-01-01T00:00:00Z"),
            validation_end=pd.Timestamp("2025-02-01T00:00:00Z"),
            evaluation_end=pd.Timestamp("2025-03-01T00:00:00Z"),
            results=results,
        )
