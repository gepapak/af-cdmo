from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from af_cdmo_probabilistic_headroom import audit_qct_risk_temporal_robustness_v5 as temporal_audit
from af_cdmo_probabilistic_headroom import confirm_quotient_tangent_risk_v5 as confirmation
from af_cdmo_probabilistic_headroom import summarize_qct_risk_confirmation_v5 as summary


def _batches() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    atoms = torch.tensor(
        [[[1.0, 0.0, 0.2], [0.0, 1.0, -0.3]]], dtype=torch.float32
    )
    refined_atoms = torch.tensor(
        [[[1.0, 0.0, 0.2], [1.0, 0.0, 0.2], [1.0, 0.0, 0.2], [0.0, 1.0, -0.3]]],
        dtype=torch.float32,
    )

    def make(canonical: torch.Tensor, weights: list[float]) -> dict[str, torch.Tensor]:
        return {
            "canonical": canonical,
            "weights": torch.tensor([weights], dtype=torch.float32),
            "mask": torch.ones(canonical.shape[:2], dtype=torch.bool),
            "total": torch.tensor([[1.7]], dtype=torch.float32),
            "present": torch.tensor([[1.0]], dtype=torch.float32),
            "analytic": torch.tensor([[0.4, -0.1, -0.3]], dtype=torch.float32),
        }

    return make(atoms, [0.7, 0.3]), make(refined_atoms, [0.2, 0.3, 0.2, 0.3])


def test_mass_weighted_models_are_nonuniform_refinement_invariant() -> None:
    torch.manual_seed(7)
    original, refined = _batches()
    mean = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    models = (
        confirmation.MassDeepSetOperator(3, 3, mean, std),
        confirmation.qct.QuotientTangentOperator(
            3, 3, variant="full", atom_mean=mean, atom_std=std
        ),
    )
    for model in models:
        model.eval()
        with torch.no_grad():
            before = model(original, torch.device("cpu"))
            after = model(refined, torch.device("cpu"))
        assert torch.allclose(before["tail_logits"], after["tail_logits"], atol=2.0e-6)
        assert torch.allclose(before["pair_residual"], after["pair_residual"], atol=2.0e-6)


def test_qct_is_invariant_to_random_mass_refinement_with_padding() -> None:
    torch.manual_seed(123)
    rng = np.random.default_rng(20261002)
    atom_dim = 5
    zone_count = 4
    atoms = torch.tensor(rng.normal(size=(1, 3, atom_dim)), dtype=torch.float32)
    weights = np.asarray([0.17, 0.51, 0.32], dtype=np.float32)

    split = rng.dirichlet(np.ones(5)).astype(np.float32) * weights[1]
    refined_atoms = torch.cat(
        [atoms[:, :1], atoms[:, 1:2].repeat(1, 5, 1), atoms[:, 2:]], dim=1
    )
    refined_weights = np.concatenate([[weights[0]], split, [weights[2]]])

    padded_atoms = torch.cat(
        [refined_atoms, torch.tensor(rng.normal(size=(1, 2, atom_dim)), dtype=torch.float32)],
        dim=1,
    )
    padded_weights = np.concatenate([refined_weights, [0.0, 0.0]]).astype(np.float32)

    def make(
        canonical: torch.Tensor, values: np.ndarray, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "canonical": canonical,
            "weights": torch.tensor(values[None, :], dtype=torch.float32),
            "mask": mask,
            "total": torch.tensor([[2.3]], dtype=torch.float32),
            "present": torch.tensor([[1.0]], dtype=torch.float32),
            "analytic": torch.tensor([[0.4, -0.2, 0.1, -0.3]], dtype=torch.float32),
        }

    original = make(atoms, weights, torch.ones((1, 3), dtype=torch.bool))
    refined = make(
        padded_atoms,
        padded_weights,
        torch.tensor([[True] * 7 + [False, False]], dtype=torch.bool),
    )
    for variant in ("full", "no_interactions", "no_tangent"):
        model = confirmation.qct.QuotientTangentOperator(
            atom_dim,
            zone_count,
            variant=variant,
            atom_mean=np.zeros(atom_dim, dtype=np.float32),
            atom_std=np.ones(atom_dim, dtype=np.float32),
        )
        model.eval()
        with torch.no_grad():
            before = model(original, torch.device("cpu"))
            after = model(refined, torch.device("cpu"))
        assert torch.allclose(before["tail_logits"], after["tail_logits"], atol=3.0e-6)
        assert torch.allclose(before["pair_residual"], after["pair_residual"], atol=3.0e-6)


def test_mass_weighted_standardization_is_refinement_invariant() -> None:
    original, refined = _batches()

    class Items:
        def __init__(self, batch: dict[str, torch.Tensor]) -> None:
            self.item = {
                "canonical": batch["canonical"][0].numpy(),
                "weights": batch["weights"][0].numpy(),
            }

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, np.ndarray]:
            if index != 0:
                raise IndexError(index)
            return self.item

    original_mean, original_std = confirmation.qct._atom_standardization(Items(original))
    refined_mean, refined_std = confirmation.qct._atom_standardization(Items(refined))
    assert np.allclose(original_mean, refined_mean, atol=1.0e-7)
    assert np.allclose(original_std, refined_std, atol=1.0e-7)


def test_qct_responds_to_non_equivalent_mass_relocation() -> None:
    torch.manual_seed(7)
    original, relocated = _batches()
    relocated["canonical"] = original["canonical"].clone()
    relocated["weights"] = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
    relocated["mask"] = torch.ones((1, 2), dtype=torch.bool)
    model = confirmation.qct.QuotientTangentOperator(
        3,
        3,
        variant="full",
        atom_mean=np.zeros(3, dtype=np.float32),
        atom_std=np.ones(3, dtype=np.float32),
    )
    model.eval()
    with torch.no_grad():
        before = model(original, torch.device("cpu"))
        after = model(relocated, torch.device("cpu"))
    assert not torch.allclose(before["tail_logits"], after["tail_logits"], atol=1.0e-6)


def test_qct_inference_is_blind_to_price_derived_targets() -> None:
    """Residual labels may travel in a training batch but are not model inputs."""
    torch.manual_seed(19)
    original, _ = _batches()
    original.update(
        {
            "target": torch.tensor([0.0], dtype=torch.float32),
            "residual": torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float32),
            "pair_residual": torch.tensor([[0.5, 0.1, -0.4]], dtype=torch.float32),
            "magnitude": np.asarray([0.4], dtype=np.float32),
        }
    )
    altered = {
        key: value.clone() if isinstance(value, torch.Tensor) else np.copy(value)
        for key, value in original.items()
    }
    altered["target"] = torch.tensor([1.0], dtype=torch.float32)
    altered["residual"] = torch.tensor([[900.0, -400.0, -500.0]], dtype=torch.float32)
    altered["pair_residual"] = torch.tensor([[1300.0, 1400.0, 100.0]], dtype=torch.float32)
    altered["magnitude"] = np.asarray([1200.0], dtype=np.float32)

    model = confirmation.qct.QuotientTangentOperator(
        3,
        3,
        variant="full",
        atom_mean=np.zeros(3, dtype=np.float32),
        atom_std=np.ones(3, dtype=np.float32),
    )
    model.eval()
    with torch.no_grad():
        before = model(original, torch.device("cpu"))
        after = model(altered, torch.device("cpu"))

    assert torch.equal(before["tail_logits"], after["tail_logits"])
    assert torch.equal(before["residual"], after["residual"])
    assert torch.equal(before["pair_residual"], after["pair_residual"])


def test_uniform_set_transformer_is_not_refinement_invariant() -> None:
    torch.manual_seed(7)
    original, refined = _batches()
    model = confirmation.UniformSetTransformerOperator(
        3, 3, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    )
    model.eval()
    with torch.no_grad():
        before = model(original, torch.device("cpu"))
        after = model(refined, torch.device("cpu"))
    difference = torch.max(torch.abs(before["tail_logits"] - after["tail_logits"]))
    assert float(difference) > 1.0e-6


def test_risk_coverage_reports_operational_quantities() -> None:
    target = np.asarray([0.0, 0.0, 1.0, 1.0])
    probability = np.asarray([0.1, 0.2, 0.8, 0.9])
    magnitude = np.asarray([0.5, 1.0, 6.0, 8.0])
    result = confirmation._risk_coverage(target, probability, magnitude)
    half = result["operating_points"]["0.50"]
    assert half["tail_rate"] == 0.0
    assert half["tail_false_negative_fraction"] == 0.0
    assert half["escalation_rate"] == 0.5
    assert result["area_under_tail_risk_coverage"] >= 0.0


def test_decision_metrics_use_fixed_calibration_threshold() -> None:
    target = np.asarray([0.0, 1.0, 1.0, 0.0])
    probability = np.asarray([0.1, 0.9, 0.8, 0.2])
    result = confirmation._decision_metrics(target, probability, 0.75)
    assert result["flagged_rows"] == 2
    assert result["tail_recall"] == 1.0
    assert result["tail_precision"] == 1.0


def test_calendar_grid_bootstrap_preserves_feed_gaps() -> None:
    complete = pd.date_range(
        "2026-03-01 00:00:00",
        periods=5 * temporal_audit.ROWS_PER_DAY,
        freq="15min",
        tz="Europe/Copenhagen",
    )
    keep = np.ones(complete.size, dtype=bool)
    keep[[3, 17, 109, 310]] = False
    timestamps = complete[keep]
    loss_difference = np.linspace(-0.04, 0.02, timestamps.size, dtype=np.float64)

    result = temporal_audit._calendar_grid_block_bootstrap(
        loss_difference,
        timestamps,
        block_days=1,
        replicates=64,
        seed=7,
    )

    assert result["observed_rows"] == timestamps.size
    assert result["calendar_grid_slots"] == complete.size
    assert result["block_grid_slots"] == temporal_audit.ROWS_PER_DAY
    assert 0.0 < result["grid_coverage"] < 1.0
    assert np.isclose(
        result["candidate_minus_baseline_brier"], np.mean(loss_difference)
    )
    assert np.all(np.isfinite(result["block_bootstrap_95ci"]))


def test_fast_row_block_bootstrap_matches_explicit_sampling() -> None:
    values = np.asarray([-0.04, 0.02, -0.01, 0.03, -0.02, 0.01, -0.03])
    block_rows = 3
    replicates = 41
    seed = 20261002
    result = temporal_audit._circular_block_bootstrap(
        values,
        block_rows=block_rows,
        replicates=replicates,
        seed=seed,
    )

    block_count = int(np.ceil(values.size / block_rows))
    offsets = np.arange(block_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        starts = rng.integers(0, values.size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % values.size).reshape(-1)
        draws[draw] = np.mean(values[indices[: values.size]])

    assert np.isclose(result["candidate_minus_baseline_brier"], np.mean(values))
    assert np.allclose(
        result["block_bootstrap_95ci"],
        np.quantile(draws, [0.025, 0.975]),
    )
    assert np.isclose(
        result["probability_of_improvement"],
        np.mean(draws < -result["improvement_zero_tolerance"]),
    )


def test_tangent_error_is_timestamp_pairwise_mae_in_physical_units() -> None:
    target = np.asarray([[1.0, -2.0, 0.5], [0.0, 1.0, -1.0]])
    prediction = np.asarray([[0.5, -1.0, 0.0], [0.5, 0.5, -0.5]])
    values = temporal_audit._tangent_error_per_timestamp(
        target, prediction, target_scale=4.0
    )
    expected = np.mean(np.abs(prediction - target), axis=1) * 4.0
    assert np.allclose(values, expected)
    metrics = temporal_audit._absolute_error_summary(values)
    assert metrics["rows"] == 2
    assert np.isclose(metrics["mean_pairwise_mae_eur_mwh"], np.mean(expected))


def test_refinement_decision_impact_separates_flip_direction_and_tail() -> None:
    target = np.array([1.0, 1.0, 0.0, 0.0])
    magnitude = np.array([10.0, 8.0, 2.0, 1.0])
    original = np.array([0.8, 0.2, 0.7, 0.1])
    refined = np.array([0.4, 0.9, 0.6, 0.8])
    result = temporal_audit._refinement_decision_impact(
        target, magnitude, original, refined, threshold=0.5
    )
    assert result["decision_flip_count"] == 3
    assert result["accept_to_escalate_count"] == 2
    assert result["escalate_to_accept_count"] == 1
    assert result["tail_flip_count"] == 2
    assert result["tail_accept_to_escalate_count"] == 1
    assert result["tail_escalate_to_accept_count"] == 1
    assert np.isclose(result["original_tail_recall"], 0.5)
    assert np.isclose(result["refined_tail_recall"], 0.5)


def test_tangent_bootstrap_result_uses_mae_label() -> None:
    result = temporal_audit._rename_difference_key(
        temporal_audit._circular_block_bootstrap(
            np.asarray([-0.2, -0.1, 0.05, -0.15]),
            block_rows=2,
            replicates=32,
            seed=7,
        ),
        "candidate_minus_baseline_mae_eur_mwh",
    )
    assert "candidate_minus_baseline_brier" not in result
    assert np.isclose(result["candidate_minus_baseline_mae_eur_mwh"], -0.1)


def test_supplemental_contract_rejects_refinement_drift() -> None:
    report = {
        "decision": {"registered_rewrite_contract_pass": True},
        "invariance_and_sensitivity": {
            "held_out_nonuniform_mass_refinement": {
                "methods": {
                    "qct_full": {
                        "max_abs_probability_drift": 2.0e-4,
                        "decision_flip_rate": 0.0,
                    }
                }
            },
            "non_equivalent_ram_sensitivity": {
                "methods": {
                    "qct_full": {
                        "0.05": {"max_abs_probability_change": 0.03}
                    }
                }
            },
        },
    }
    review = summary._supplemental_contract_review(report)
    assert review["registered_rewrite_contract_pass"]
    assert not review["held_out_nonuniform_refinement_pass"]
    assert not review["full_registered_structural_contract_pass"]
    assert review["maximum_non_equivalent_ram_probability_response"] == 0.03


def test_certificate_only_reconstruction_review_uses_registered_metrics() -> None:
    report = {
        "confirmation_target": {"mean_residual_magnitude_eur_mwh": 4.0},
        "methods": {
            "qct_full": {
                "tangent_residual": {"mean_pairwise_mae_eur_mwh": 3.0}
            },
            "hgb_quotient": {
                "tangent_residual": {"mean_pairwise_mae_eur_mwh": 2.5}
            },
        },
    }
    review = summary._certificate_only_reconstruction_review(report)
    assert review["qct_beats_analytic_only_point_estimate"]
    assert not review["qct_beats_hgb_quotient_point_estimate"]
    assert np.isclose(review["qct_relative_mae_reduction_vs_analytic_only"], 0.25)
    assert np.isclose(review["qct_relative_mae_reduction_vs_hgb_quotient"], -0.2)
    assert review["registered_primary_gate_unchanged"]
