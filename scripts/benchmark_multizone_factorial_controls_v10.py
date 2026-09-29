"""Run matched controls for the CQDM multi-zone quotient-gauge study.

This follow-up completes the raw/quotient and gauge/pairwise comparison while
holding analytic residualization fixed.  It also projects independent pairwise
predictions onto the nearest cycle-consistent price field.  The task remains
simultaneous reconstruction from a causally published solved certificate; it is
not future-price forecasting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from scripts.benchmark_certificate_governance_v5 import (
        DEFAULT_DUAL,
        DEFAULT_EVALUATION_END,
        DEFAULT_FIT_END,
        DEFAULT_VALIDATION_END,
        _positive_scales,
        _read_certificate,
    )
    from scripts.benchmark_multizone_quotient_gauge_v6 import (
        DEFAULT_PRICE_ROOT,
        VectorCertificateDataset,
        _build_vector_samples,
        _classical_predictions,
        _load_price_panel,
        _parse_utc,
        _summarize_method,
    )
    from scripts.cqdm_utils import (
        paired_block_bootstrap,
        sha256_file,
    )
    from scripts.multizone_factorial_controls_v10 import (
        CONTROL_MODES,
        decompose_pairwise_hodge,
        fit_factorial_control,
        predict_factorial_control,
        project_pairwise_to_consistent,
    )
    from scripts.multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        MultiZoneRegressor,
        center_price_field,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
    )
except ModuleNotFoundError:
    from benchmark_certificate_governance_v5 import (
        DEFAULT_DUAL,
        DEFAULT_EVALUATION_END,
        DEFAULT_FIT_END,
        DEFAULT_VALIDATION_END,
        _positive_scales,
        _read_certificate,
    )
    from benchmark_multizone_quotient_gauge_v6 import (
        DEFAULT_PRICE_ROOT,
        VectorCertificateDataset,
        _build_vector_samples,
        _classical_predictions,
        _load_price_panel,
        _parse_utc,
        _summarize_method,
    )
    from cqdm_utils import paired_block_bootstrap, sha256_file
    from multizone_factorial_controls_v10 import (
        CONTROL_MODES,
        decompose_pairwise_hodge,
        fit_factorial_control,
        predict_factorial_control,
        project_pairwise_to_consistent,
    )
    from multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        MultiZoneRegressor,
        center_price_field,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "official_jao_dual_measure" / "multizone_factorial_controls_development_v10.json"
)
STRESSES = ("original", "combined")
REFERENCE_MODES = (
    "raw_deepset_gauge",
    "cqdm_gauge_residual",
    "cqdm_pairwise_residual",
)
ALL_NEURAL_MODES = (*REFERENCE_MODES, *CONTROL_MODES)
PROJECTED_MODE_NAMES = {
    "raw_deepset_pairwise": "projected_raw_deepset_pairwise",
    "raw_deepset_analytic_residual_pairwise": (
        "projected_raw_deepset_analytic_residual_pairwise"
    ),
    "cqdm_pairwise_residual": "projected_cqdm_pairwise_residual",
}


def _project_prediction(
    prediction: dict[str, Any], zone_count: int
) -> dict[str, Any]:
    projected = project_pairwise_to_consistent(prediction["pairwise"], zone_count)
    return {
        **projected,
        "target": prediction.get("target"),
        "timestamps": prediction.get("timestamps"),
    }


def _fit_and_predict_neural(
    mode: str,
    seeds: list[int],
    train_dataset: VectorCertificateDataset,
    validation_dataset: VectorCertificateDataset,
    evaluation_datasets: dict[str, VectorCertificateDataset],
    *,
    raw_dim: int,
    canonical_dim: int,
    zone_count: int,
    batch_size: int,
    epochs: int,
    target_scale: float,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    models: list[Any] = []
    fits: list[dict[str, Any]] = []
    for seed in seeds:
        if mode in REFERENCE_MODES:
            model, fit = fit_multizone(
                mode,
                seed,
                train_dataset,
                validation_dataset,
                raw_dim=raw_dim,
                canonical_dim=canonical_dim,
                zone_count=zone_count,
                batch_size=batch_size,
                epochs=epochs,
                device=device,
            )
        else:
            model, fit = fit_factorial_control(
                mode,
                seed,
                train_dataset,
                validation_dataset,
                raw_dim=raw_dim,
                zone_count=zone_count,
                batch_size=batch_size,
                epochs=epochs,
                device=device,
            )
        models.append(model)
        fits.append(fit)
        print(
            f"[FIT] mode={mode} seed={seed} epoch={fit['best_epoch']} "
            f"validation={fit['validation_objective']:.6f}",
            flush=True,
        )

    def predict(model: Any, dataset: VectorCertificateDataset) -> dict[str, Any]:
        if isinstance(model, MultiZoneRegressor):
            return predict_multizone(
                model,
                dataset,
                batch_size=batch_size,
                target_scale=target_scale,
                device=device,
            )
        return predict_factorial_control(
            model,
            dataset,
            batch_size=batch_size,
            target_scale=target_scale,
            device=device,
        )

    validation_members = [predict(model, validation_dataset) for model in models]
    evaluation_members = {
        stress: [predict(model, dataset) for model in models]
        for stress, dataset in evaluation_datasets.items()
    }
    return validation_members, evaluation_members, fits


def _ensemble(members: list[dict[str, Any]]) -> dict[str, np.ndarray | None]:
    fields = [member["field"] for member in members if member["field"] is not None]
    return {
        "field": np.mean(np.stack(fields), axis=0) if fields else None,
        "pairwise": np.mean(
            np.stack([np.asarray(member["pairwise"]) for member in members]), axis=0
        ),
    }


def _add_projection_variants(
    predictions: dict[str, dict[str, dict[str, np.ndarray | None]]],
    validation_predictions: dict[str, np.ndarray],
    neural_members: dict[str, dict[str, list[dict[str, Any]]]],
    validation_members: dict[str, list[dict[str, Any]]],
    zone_count: int,
) -> None:
    for source, projected_name in PROJECTED_MODE_NAMES.items():
        predictions[projected_name] = {
            stress: project_pairwise_to_consistent(
                np.asarray(predictions[source][stress]["pairwise"]), zone_count
            )
            for stress in STRESSES
        }
        validation_predictions[projected_name] = project_pairwise_to_consistent(
            validation_predictions[source], zone_count
        )["pairwise"]
        neural_members[projected_name] = {
            stress: [
                _project_prediction(member, zone_count)
                for member in neural_members[source][stress]
            ]
            for stress in STRESSES
        }
        validation_members[projected_name] = [
            _project_prediction(member, zone_count)
            for member in validation_members[source]
        ]


def _comparison(
    losses: dict[str, np.ndarray], baseline: str, candidate: str
) -> dict[str, Any]:
    return paired_block_bootstrap(
        {"timestamp_pairwise_mae": losses[baseline]},
        {"timestamp_pairwise_mae": losses[candidate]},
    )["timestamp_pairwise_mae"]


def _add_hodge_metrics(
    summary: dict[str, Any],
    target_field: np.ndarray,
    prediction: np.ndarray,
    zone_count: int,
) -> None:
    """Record the exact squared-error decomposition for a valid price-field target."""

    target_pairs = field_to_pairwise(target_field)
    decomposition = decompose_pairwise_hodge(prediction, zone_count)
    error = np.asarray(prediction, dtype=np.float64) - target_pairs
    projected_error = decomposition["gradient_pairwise"] - target_pairs
    cycle_component = decomposition["cycle_pairwise"]
    total_mse = float(np.mean(np.square(error)))
    gradient_mse = float(np.mean(np.square(projected_error)))
    cycle_mse = float(np.mean(np.square(cycle_component)))
    summary.update(
        {
            "pairwise_rmse_eur_mwh": float(np.sqrt(total_mse)),
            "hodge_gradient_error_rmse_eur_mwh": float(np.sqrt(gradient_mse)),
            "hodge_cycle_component_rmse_eur_mwh": float(np.sqrt(cycle_mse)),
            "hodge_cycle_share_of_pairwise_mse": (
                cycle_mse / total_mse if total_mse > 0.0 else 0.0
            ),
            "hodge_pythagorean_identity_abs_error": abs(
                total_mse - gradient_mse - cycle_mse
            ),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--price_root", type=Path, default=DEFAULT_PRICE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zones", nargs="+", choices=REAL_NORDIC_ZONES, default=REAL_NORDIC_ZONES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--alert_quantile", type=float, default=0.95)
    parser.add_argument("--min_price_panel_coverage", type=float, default=0.90)
    parser.add_argument(
        "--study_role",
        choices=("development", "frozen_transition", "confirmation"),
        default="development",
    )
    parser.add_argument("--fit_end", default=DEFAULT_FIT_END.isoformat())
    parser.add_argument("--validation_end", default=DEFAULT_VALIDATION_END.isoformat())
    parser.add_argument("--evaluation_end", default=DEFAULT_EVALUATION_END.isoformat())
    args = parser.parse_args()

    zones = tuple(dict.fromkeys(args.zones))
    seeds = list(args.seeds)
    if len(zones) < 3:
        raise ValueError("The factorial study requires at least three distinct zones")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if not 0.0 < args.alert_quantile < 1.0:
        raise ValueError("Alert quantile must lie strictly between zero and one")
    if not 0.0 < args.min_price_panel_coverage <= 1.0:
        raise ValueError("Coverage threshold must lie in (0, 1]")
    fit_end = _parse_utc(args.fit_end)
    validation_end = _parse_utc(args.validation_end)
    evaluation_end = _parse_utc(args.evaluation_end)
    if not fit_end < validation_end < evaluation_end:
        raise ValueError("Require fit_end < validation_end < evaluation_end")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    frame, certificate_audit = _read_certificate(args.dual.resolve(), 60.0)
    unit_columns = certificate_audit["unit_columns"]
    missing_ptdf = [zone for zone in zones if f"unit_ptdf_{zone}" not in unit_columns]
    if missing_ptdf:
        raise RuntimeError(f"Certificate omitted real-zone PTDF channels: {missing_ptdf}")
    price_panel, price_provenance = _load_price_panel(args.price_root.resolve(), zones)
    panel_start = max(frame["delivery_utc"].min(), price_panel.index.min())
    expected_grid = pd.date_range(
        start=panel_start, end=evaluation_end, freq="15min", inclusive="left"
    )
    coverage = len(price_panel.index.intersection(expected_grid)) / max(len(expected_grid), 1)
    if coverage < args.min_price_panel_coverage:
        raise RuntimeError(
            f"All-zone panel coverage {coverage:.2%} is below "
            f"{args.min_price_panel_coverage:.2%}; no price imputation is permitted"
        )
    samples = _build_vector_samples(
        frame, price_panel, unit_columns, zones, fit_end, evaluation_end
    )
    train_samples = [sample for sample in samples if sample["timestamp"] < fit_end]
    validation_samples = [
        sample for sample in samples if fit_end <= sample["timestamp"] < validation_end
    ]
    evaluation_samples = [
        sample
        for sample in samples
        if validation_end <= sample["timestamp"] < evaluation_end
    ]
    if not train_samples or not validation_samples or not evaluation_samples:
        raise RuntimeError(
            "Chronology produced an empty split: "
            f"fit={len(train_samples)}, validation={len(validation_samples)}, "
            f"evaluation={len(evaluation_samples)}"
        )

    raw_values = np.concatenate(
        [sample["raw"] for sample in train_samples if len(sample["raw"])], axis=0
    ).astype(np.float64)
    raw_scale = _positive_scales(raw_values)
    target_values = np.stack([sample["target_vector"] for sample in train_samples])
    target_scale = max(float(np.median(np.abs(target_values))), 1.0)
    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    dataset_kwargs = {
        "raw_scale": raw_scale,
        "target_scale": target_scale,
        "real_zone_indices": real_indices,
        "ptdf_count": len(unit_columns),
    }
    train_dataset = VectorCertificateDataset(train_samples, stress="original", **dataset_kwargs)
    validation_dataset = VectorCertificateDataset(
        validation_samples, stress="original", **dataset_kwargs
    )
    evaluation_datasets = {
        stress: VectorCertificateDataset(evaluation_samples, stress=stress, **dataset_kwargs)
        for stress in STRESSES
    }
    raw_dim = train_samples[0]["raw"].shape[1]
    canonical_dim = train_samples[0]["canonical"].shape[1]

    predictions: dict[str, dict[str, dict[str, np.ndarray | None]]] = {}
    neural_members: dict[str, dict[str, list[dict[str, Any]]]] = {}
    validation_members: dict[str, list[dict[str, Any]]] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    fits: dict[str, Any] = {}
    for mode in ALL_NEURAL_MODES:
        mode_validation, mode_evaluation, mode_fits = _fit_and_predict_neural(
            mode,
            seeds,
            train_dataset,
            validation_dataset,
            evaluation_datasets,
            raw_dim=raw_dim,
            canonical_dim=canonical_dim,
            zone_count=len(zones),
            batch_size=args.batch_size,
            epochs=args.epochs,
            target_scale=target_scale,
            device=device,
        )
        fits[mode] = mode_fits
        validation_members[mode] = mode_validation
        validation_predictions[mode] = _ensemble(mode_validation)["pairwise"]
        neural_members[mode] = mode_evaluation
        predictions[mode] = {
            stress: _ensemble(members) for stress, members in mode_evaluation.items()
        }

    classical_datasets = {**evaluation_datasets, "validation": validation_dataset}
    classical_predictions, classical_fit = _classical_predictions(
        train_dataset, classical_datasets
    )
    validation_classical = {
        method: by_stress.pop("validation")
        for method, by_stress in classical_predictions.items()
    }
    for method, value in validation_classical.items():
        validation_predictions[method] = np.asarray(value["pairwise"])
    predictions.update(classical_predictions)

    independent_projected_name = "projected_independent_pairwise_hgb"
    predictions[independent_projected_name] = {
        stress: project_pairwise_to_consistent(
            np.asarray(predictions["independent_pairwise_hgb"][stress]["pairwise"]),
            len(zones),
        )
        for stress in STRESSES
    }
    validation_predictions[independent_projected_name] = project_pairwise_to_consistent(
        validation_predictions["independent_pairwise_hgb"], len(zones)
    )["pairwise"]
    _add_projection_variants(
        predictions,
        validation_predictions,
        neural_members,
        validation_members,
        len(zones),
    )

    validation_target = np.stack([sample["target_vector"] for sample in validation_samples])
    validation_pairs = field_to_pairwise(validation_target)
    thresholds = {
        method: float(
            np.quantile(
                np.max(np.abs(validation_pairs - prediction), axis=1),
                args.alert_quantile,
            )
        )
        for method, prediction in validation_predictions.items()
    }
    target_field = np.stack([sample["target_vector"] for sample in evaluation_samples])
    results: dict[str, Any] = {}
    losses: dict[str, np.ndarray] = {}
    for method, by_stress in predictions.items():
        original_pairs = np.asarray(by_stress["original"]["pairwise"])
        results[method] = {
            "validation_max_pair_error_alert_threshold_eur_mwh": thresholds[method],
            "stresses": {},
        }
        for stress in STRESSES:
            summary, timestamp_losses = _summarize_method(
                target_field,
                by_stress[stress]["field"],
                np.asarray(by_stress[stress]["pairwise"]),
                original_pairs,
                thresholds[method],
                len(zones),
            )
            _add_hodge_metrics(
                summary,
                target_field,
                np.asarray(by_stress[stress]["pairwise"]),
                len(zones),
            )
            if method in neural_members:
                summary["seed_members"] = []
                for member_index, (seed, member) in enumerate(
                    zip(seeds, neural_members[method][stress])
                ):
                    member_summary, _ = _summarize_method(
                        target_field,
                        member["field"],
                        member["pairwise"],
                        neural_members[method]["original"][member_index]["pairwise"],
                        thresholds[method],
                        len(zones),
                    )
                    summary["seed_members"].append(
                        {
                            "seed": seed,
                            "mean_pairwise_mae_eur_mwh": member_summary[
                                "mean_pairwise_mae_eur_mwh"
                            ],
                            "mean_centered_zone_mae_eur_mwh": member_summary[
                                "mean_centered_zone_mae_eur_mwh"
                            ],
                            "mean_abs_cycle_inconsistency_eur_mwh": member_summary[
                                "mean_abs_cycle_inconsistency_eur_mwh"
                            ],
                            "mean_abs_pairwise_prediction_drift_eur_mwh": member_summary[
                                "mean_abs_pairwise_prediction_drift_eur_mwh"
                            ],
                        }
                    )
            results[method]["stresses"][stress] = summary
            if stress == "original":
                losses[method] = timestamp_losses

    comparison_pairs = {
        "quotient_effect_with_gauge_residual": (
            "raw_deepset_analytic_residual_gauge",
            "cqdm_gauge_residual",
        ),
        "quotient_effect_with_pairwise_residual": (
            "raw_deepset_analytic_residual_pairwise",
            "cqdm_pairwise_residual",
        ),
        "analytic_residual_effect_with_raw_gauge": (
            "raw_deepset_gauge",
            "raw_deepset_analytic_residual_gauge",
        ),
        "raw_output_geometry_gauge_minus_pairwise": (
            "raw_deepset_pairwise",
            "raw_deepset_gauge",
        ),
        "cqdm_output_geometry_gauge_minus_pairwise": (
            "cqdm_pairwise_residual",
            "cqdm_gauge_residual",
        ),
        "projection_effect_independent_hgb": (
            "independent_pairwise_hgb",
            "projected_independent_pairwise_hgb",
        ),
        "projection_effect_raw_pairwise": (
            "raw_deepset_pairwise",
            "projected_raw_deepset_pairwise",
        ),
        "cqdm_gauge_minus_analytic_kkt": (
            "analytic_gauge",
            "cqdm_gauge_residual",
        ),
    }
    comparisons = {
        name: {
            "baseline": baseline,
            "candidate": candidate,
            **_comparison(losses, baseline, candidate),
        }
        for name, (baseline, candidate) in comparison_pairs.items()
    }

    report = {
        "protocol": {
            "name": "multizone_factorial_controls_v10",
            "study_role": args.study_role,
            "task": (
                "simultaneous reconstruction of the observed centered Nordic "
                "day-ahead zonal price field from a timely solved certificate"
            ),
            "not_a_forecast": True,
            "zones": list(zones),
            "pairs": [
                f"{zones[left]}-{zones[right]}"
                for left, right in pair_indices(len(zones))
            ],
            "fit_end_exclusive": fit_end.isoformat(),
            "internal_validation_end_exclusive": validation_end.isoformat(),
            "evaluation_end_exclusive": evaluation_end.isoformat(),
            "seeds": seeds,
            "stresses": list(STRESSES),
            "all_zone_price_panel_coverage": coverage,
            "minimum_required_price_panel_coverage": args.min_price_panel_coverage,
            "primary_isolation": (
                "raw versus CQDM representation, holding analytic residual target "
                "and output geometry fixed"
            ),
            "projection": (
                "complete-graph least-squares projection onto the unique zero-sum "
                "price field"
            ),
        },
        "source": {
            "certificate": certificate_audit,
            "day_ahead_prices": price_provenance,
        },
        "rows": {
            "fit": len(train_samples),
            "internal_validation": len(validation_samples),
            "evaluation": len(evaluation_samples),
        },
        "target_scale_eur_mwh": target_scale,
        "fits": {**fits, "classical": classical_fit},
        "results": results,
        "paired_original_formulation_comparisons": comparisons,
        "interpretation_contract": {
            "quotient_effect": (
                "Compare matched raw analytic-residual controls with CQDM residual "
                "models; do not attribute raw-direct differences solely to CQDM."
            ),
            "projection_effect": (
                "Post-hoc projection enforces output geometry but cannot remove "
                "input-presentation sensitivity."
            ),
            "analytic_limit": (
                "A learned model need not beat the exact analytic KKT reconstruction "
                "to establish presentation robustness; predictive superiority is a "
                "separate empirical claim."
            ),
        },
        "reproducibility": {
            "engine": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "control_model_engine": {
                "path": str(
                    (ROOT / "scripts/multizone_factorial_controls_v10.py").resolve()
                ),
                "sha256": sha256_file(
                    ROOT / "scripts/multizone_factorial_controls_v10.py"
                ),
            },
            "frozen_v6_model_engine": {
                "path": str((ROOT / "scripts/multizone_quotient_gauge_v6.py").resolve()),
                "sha256": sha256_file(ROOT / "scripts/multizone_quotient_gauge_v6.py"),
            },
            "frozen_v6_benchmark_engine": {
                "path": str(
                    (ROOT / "scripts/benchmark_multizone_quotient_gauge_v6.py").resolve()
                ),
                "sha256": sha256_file(
                    ROOT / "scripts/benchmark_multizone_quotient_gauge_v6.py"
                ),
            },
            "dual_source_sha256": sha256_file(args.dual.resolve()),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
