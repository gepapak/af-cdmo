"""Benchmark loss-safe AF-QDM on real Nordic solved certificates.

The task reconstructs a same-delivery centered zonal day-ahead price field from
a causally published solved flow-based certificate.  It is not a future-price
forecast.  All outputs use new v13 paths and frozen v10-v12 artifacts are read
only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

try:
    from scripts.af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
        parse_presentation,
        presentation_names,
        raw_rows_for_scale,
    )
    from scripts.benchmark_certificate_governance_v5 import (
        DEFAULT_DUAL,
        DEFAULT_EVALUATION_END,
        DEFAULT_FIT_END,
        DEFAULT_VALIDATION_END,
        _positive_scales,
        _read_certificate,
    )
    from scripts.benchmark_multizone_factorial_controls_v10 import _add_hodge_metrics
    from scripts.benchmark_multizone_quotient_gauge_v6 import (
        DEFAULT_PRICE_ROOT,
        _build_vector_samples,
        _load_price_panel,
        _parse_utc,
        _summarize_method,
    )
    from scripts.cqdm_utils import paired_block_bootstrap, sha256_file
    from scripts.multizone_factorial_controls_v10 import (
        fit_factorial_control,
        predict_factorial_control,
    )
    from scripts.multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
    )
except ModuleNotFoundError:
    from af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
        parse_presentation,
        presentation_names,
        raw_rows_for_scale,
    )
    from benchmark_certificate_governance_v5 import (
        DEFAULT_DUAL,
        DEFAULT_EVALUATION_END,
        DEFAULT_FIT_END,
        DEFAULT_VALIDATION_END,
        _positive_scales,
        _read_certificate,
    )
    from benchmark_multizone_factorial_controls_v10 import _add_hodge_metrics
    from benchmark_multizone_quotient_gauge_v6 import (
        DEFAULT_PRICE_ROOT,
        _build_vector_samples,
        _load_price_panel,
        _parse_utc,
        _summarize_method,
    )
    from cqdm_utils import paired_block_bootstrap, sha256_file
    from multizone_factorial_controls_v10 import (
        fit_factorial_control,
        predict_factorial_control,
    )
    from multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
    )


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "official_jao_dual_measure"
DEFAULT_OUTPUT = REPORT_ROOT / "af_cdmo_real_development_v13.json"
DEFAULT_GEOMETRY_AUDIT = REPORT_ROOT / "af_cdmo_real_geometry_audit_v13.json"
FROZEN_V11_REPORTS = {
    "development": REPORT_ROOT / "slack_qdm_extension_development_v11.json",
    "frozen_transition": REPORT_ROOT
    / "slack_qdm_extension_market_design_shift_v11.json",
    "confirmation": REPORT_ROOT / "slack_qdm_extension_confirmation_v11.json",
}
PRIMARY_NORDIC_ZONES = (
    "DK1",
    "DK2",
    "FI",
    "NO1",
    "NO2",
    "NO3",
    "NO5",
    "SE1",
    "SE3",
    "SE4",
)

NEURAL_METHODS: dict[str, tuple[str, str]] = {
    "ambient_cqdm_residual": ("ambient", "cqdm"),
    "slack_qdm_residual": ("slack", "cqdm"),
    "af_projected_raw_residual": ("af", "raw"),
    "af_qdm_residual": ("af", "cqdm"),
}
ALL_METHODS = (*NEURAL_METHODS, "analytic_gauge")
ENGINE_PATHS = (
    Path(__file__),
    ROOT / "scripts" / "af_cdmo_real_extension_v13.py",
    ROOT / "scripts" / "audit_af_cdmo_real_geometry_v13.py",
    ROOT / "scripts" / "slack_qdm_extension_v11.py",
    ROOT / "scripts" / "benchmark_certificate_governance_v5.py",
    ROOT / "scripts" / "benchmark_multizone_quotient_gauge_v6.py",
    ROOT / "scripts" / "benchmark_multizone_factorial_controls_v10.py",
    ROOT / "scripts" / "multizone_quotient_gauge_v6.py",
    ROOT / "scripts" / "multizone_factorial_controls_v10.py",
    ROOT / "scripts" / "cqdm_utils.py",
)


def _canonical_ram_scale(frame: pd.DataFrame, fit_end: pd.Timestamp) -> float:
    values = np.abs(
        frame.loc[frame["delivery_utc"] < fit_end, "canonical_ram_mw"].to_numpy(
            dtype=np.float64
        )
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    return max(float(np.median(values)), 1.0e-6) if len(values) else 1.0


def _ensemble(members: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "field": np.mean(
            np.stack([np.asarray(member["field"]) for member in members]), axis=0
        ),
        "pairwise": np.mean(
            np.stack([np.asarray(member["pairwise"]) for member in members]), axis=0
        ),
    }


def _predict_analytic(dataset: RealAffineGaugeDataset) -> dict[str, np.ndarray]:
    field = np.stack([dataset[index]["analytic_vector"] for index in range(len(dataset))])
    return {"field": field, "pairwise": field_to_pairwise(field)}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _paired(
    baseline: np.ndarray, candidate: np.ndarray, *, key: str
) -> dict[str, Any]:
    return paired_block_bootstrap(
        {key: np.asarray(baseline)}, {key: np.asarray(candidate)}
    )[key]


def _seed_sign_test(
    baseline: np.ndarray, candidate: np.ndarray, *, tolerance: float = 1.0e-12
) -> dict[str, Any]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if baseline_values.shape != candidate_values.shape or baseline_values.ndim != 1:
        raise ValueError("Seed-level comparisons require matched one-dimensional arrays")
    improvement = baseline_values - candidate_values
    wins = int(np.sum(improvement > tolerance))
    losses = int(np.sum(improvement < -tolerance))
    ties = int(len(improvement) - wins - losses)
    effective = wins + losses
    p_value = (
        float(binomtest(wins, effective, 0.5, alternative="greater").pvalue)
        if effective
        else None
    )
    two_sided_p_value = (
        float(binomtest(wins, effective, 0.5, alternative="two-sided").pvalue)
        if effective
        else None
    )
    return {
        "baseline_seed_mae_eur_mwh": baseline_values.tolist(),
        "candidate_seed_mae_eur_mwh": candidate_values.tolist(),
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": ties,
        "effective_pairs": effective,
        "mean_baseline_minus_candidate_eur_mwh": float(improvement.mean()),
        "median_baseline_minus_candidate_eur_mwh": float(np.median(improvement)),
        "one_sided_exact_sign_test_p_value": p_value,
        "two_sided_exact_sign_test_p_value": two_sided_p_value,
    }


def _frozen_v11_reproduction_contract(
    *,
    study_role: str,
    zones: tuple[str, ...],
    seeds: list[int],
    fit_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    results: dict[str, Any],
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    if study_role == "smoke":
        return {"status": "not_applicable_to_smoke"}
    reference_path = FROZEN_V11_REPORTS[study_role].resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing frozen v11 reference: {reference_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    protocol = reference["protocol"]
    expected = {
        "study_role": study_role,
        "zones": list(zones),
        "seeds": seeds,
        "fit_end_exclusive": fit_end.isoformat(),
        "internal_validation_end_exclusive": validation_end.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(),
    }
    mismatches = {
        key: {"reference": protocol.get(key), "current": value}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "v13 chronology/panel does not match the frozen v11 reference: "
            + json.dumps(mismatches, sort_keys=True)
        )

    checked: dict[str, Any] = {}
    max_abs_difference = 0.0
    metric_names = (
        "mean_pairwise_mae_eur_mwh",
        "p95_timestamp_pairwise_mae_eur_mwh",
        "mean_centered_zone_mae_eur_mwh",
        "pairwise_rmse_eur_mwh",
    )
    for method in ("ambient_cqdm_residual", "slack_qdm_residual"):
        current_method = results[method]
        reference_method = reference["results"][method]
        differences: dict[str, float] = {}
        threshold_difference = abs(
            float(current_method["validation_max_pair_error_alert_threshold_eur_mwh"])
            - float(
                reference_method[
                    "validation_max_pair_error_alert_threshold_eur_mwh"
                ]
            )
        )
        differences["validation_alert_threshold"] = threshold_difference
        current_original = current_method["stresses"]["original"]
        reference_original = reference_method["stresses"]["original"]
        for metric in metric_names:
            differences[metric] = abs(
                float(current_original[metric]) - float(reference_original[metric])
            )
        current_members = current_original["seed_members"]
        reference_members = reference_original["seed_members"]
        if [item["seed"] for item in current_members] != [
            item["seed"] for item in reference_members
        ]:
            raise RuntimeError(f"Frozen v11 seed order changed for {method}")
        differences["max_seed_member_pairwise_mae"] = max(
            abs(
                float(current["mean_pairwise_mae_eur_mwh"])
                - float(frozen["mean_pairwise_mae_eur_mwh"])
            )
            for current, frozen in zip(current_members, reference_members, strict=True)
        )
        method_max = max(differences.values())
        max_abs_difference = max(max_abs_difference, method_max)
        checked[method] = {
            "absolute_differences": differences,
            "max_abs_difference": method_max,
        }
    if max_abs_difference > tolerance:
        raise RuntimeError(
            "AF-CDMO v13 failed to reproduce frozen v11 clean-form controls: "
            f"max_abs_difference={max_abs_difference:.6g}, tolerance={tolerance:.6g}"
        )
    return {
        "status": "passed",
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "tolerance": tolerance,
        "max_abs_difference": max_abs_difference,
        "checked": checked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--price_root", type=Path, default=DEFAULT_PRICE_ROOT)
    parser.add_argument("--geometry_audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--losses_output", type=Path)
    parser.add_argument(
        "--zones",
        nargs="+",
        choices=REAL_NORDIC_ZONES,
        default=PRIMARY_NORDIC_ZONES,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--alert_quantile", type=float, default=0.95)
    parser.add_argument("--min_price_panel_coverage", type=float, default=0.90)
    parser.add_argument(
        "--analytic_invariance_tolerance_eur_mwh", type=float, default=1.0e-3
    )
    parser.add_argument("--invariance_tolerance_eur_mwh", type=float, default=1.0e-3)
    parser.add_argument(
        "--study_role",
        choices=("development", "frozen_transition", "confirmation", "smoke"),
        default="development",
    )
    parser.add_argument("--fit_end", default=DEFAULT_FIT_END.isoformat())
    parser.add_argument("--validation_end", default=DEFAULT_VALIDATION_END.isoformat())
    parser.add_argument("--evaluation_end", default=DEFAULT_EVALUATION_END.isoformat())
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    losses_output = (
        args.losses_output.expanduser().resolve()
        if args.losses_output is not None
        else output.with_name(output.stem + "_timestamp_losses.npz")
    )
    existing = [path for path in (output, losses_output) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing v13 artifacts; use new paths or --force: "
            + ", ".join(str(path) for path in existing)
        )

    zones = tuple(dict.fromkeys(args.zones))
    seeds = list(args.seeds)
    if len(zones) < 3:
        raise ValueError("At least three output zones are required")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be non-empty and unique")
    if args.study_role != "smoke" and len(seeds) < 10:
        raise ValueError("Publication roles require ten preregistered seeds")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if not 0.0 < args.alert_quantile < 1.0:
        raise ValueError("alert_quantile must lie in (0, 1)")
    if not 0.0 < args.min_price_panel_coverage <= 1.0:
        raise ValueError("Coverage threshold must lie in (0, 1]")
    if args.invariance_tolerance_eur_mwh <= 0.0:
        raise ValueError("Invariance tolerance must be positive")
    if args.analytic_invariance_tolerance_eur_mwh <= 0.0:
        raise ValueError("Analytic invariance tolerance must be positive")

    fit_end = _parse_utc(args.fit_end)
    validation_end = _parse_utc(args.validation_end)
    evaluation_end = _parse_utc(args.evaluation_end)
    if not fit_end < validation_end < evaluation_end:
        raise ValueError("Require fit_end < validation_end < evaluation_end")

    geometry_audit_path = args.geometry_audit.expanduser().resolve()
    if not geometry_audit_path.is_file():
        raise FileNotFoundError(
            f"Missing v13 geometry audit: {geometry_audit_path}. Run "
            "scripts/audit_af_cdmo_real_geometry_v13.py first."
        )
    geometry_audit = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
    if not geometry_audit["observed_transfer_queries"]["all_tangent"]:
        raise RuntimeError("The v13 geometry audit did not certify tangent price queries")
    if geometry_audit["protocol"].get("skagerrak_excluded") is not True:
        raise RuntimeError("The v13 primary geometry must exclude Skagerrak")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dual_path = args.dual.expanduser().resolve()
    dual_hash = sha256_file(dual_path)
    audited_dual_hash = geometry_audit.get("sources", {}).get("dual_archive_sha256")
    if audited_dual_hash != dual_hash:
        raise RuntimeError(
            "Current dual archive is not the archive certified by the v13 "
            "geometry audit"
        )
    archive_projection = geometry_audit.get("archive_projection", {})
    if int(archive_projection.get("rows_at_or_below_projected_norm_tolerance", -1)) != 0:
        raise RuntimeError("The v13 geometry audit did not certify non-degenerate rows")
    if geometry_audit.get("registered_gauge_rewrite", {}).get("status") != "passed":
        raise RuntimeError("The v13 geometry audit did not pass its gauge-rewrite check")
    if (
        geometry_audit.get("registered_equality_basis_rewrite", {}).get("status")
        != "passed"
    ):
        raise RuntimeError(
            "The v13 geometry audit did not pass its equality-basis rewrite check"
        )
    frame, certificate_audit = _read_certificate(dual_path, 60.0)
    unit_columns = certificate_audit["unit_columns"]
    input_zone_names = tuple(column.removeprefix("unit_ptdf_") for column in unit_columns)
    geometry = RealAffineGeometry.loss_safe(input_zone_names)
    if list(input_zone_names) != geometry_audit["geometry"]["zone_order"]:
        raise RuntimeError("Current PTDF order differs from the v13 geometry audit")
    if geometry.rank != int(geometry_audit["geometry"]["equality_rank"]):
        raise RuntimeError("Current equality rank differs from the v13 geometry audit")
    missing_ptdf = [zone for zone in zones if f"unit_ptdf_{zone}" not in unit_columns]
    if missing_ptdf:
        raise RuntimeError(f"Certificate omitted output-zone PTDFs: {missing_ptdf}")

    price_panel, price_provenance = _load_price_panel(
        args.price_root.expanduser().resolve(), zones
    )
    panel_start = max(frame["delivery_utc"].min(), price_panel.index.min())
    expected_grid = pd.date_range(
        start=panel_start, end=evaluation_end, freq="15min", inclusive="left"
    )
    coverage = len(price_panel.index.intersection(expected_grid)) / max(len(expected_grid), 1)
    if coverage < args.min_price_panel_coverage:
        raise RuntimeError(
            f"All-zone price coverage {coverage:.2%} is below "
            f"{args.min_price_panel_coverage:.2%}; no imputation is permitted"
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

    target_values = np.stack([sample["target_vector"] for sample in train_samples])
    target_scale = max(float(np.median(np.abs(target_values))), 1.0)
    canonical_ram_scale = _canonical_ram_scale(frame, fit_end)
    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    raw_scales = {
        representation: _positive_scales(
            raw_rows_for_scale(
                train_samples,
                representation=representation,
                real_zone_indices=real_indices,
                geometry=geometry,
                canonical_ram_scale=canonical_ram_scale,
            )
        )
        for representation in ("ambient", "slack", "af")
    }

    def dataset(
        split_samples: list[dict[str, Any]], representation: str, presentation: str
    ) -> RealAffineGaugeDataset:
        return RealAffineGaugeDataset(
            split_samples,
            raw_scale=raw_scales[representation],
            target_scale=target_scale,
            presentation=presentation,
            representation=representation,
            real_zone_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
            analytic_tolerance=args.analytic_invariance_tolerance_eur_mwh,
        )

    train_datasets = {
        representation: dataset(train_samples, representation, "original")
        for representation in ("ambient", "slack", "af")
    }
    validation_datasets = {
        representation: dataset(validation_samples, representation, "original")
        for representation in ("ambient", "slack", "af")
    }
    train_af_analytic_drift = max(
        float(train_datasets["af"][index]["analytic_pair_drift_eur_mwh"])
        for index in range(len(train_datasets["af"]))
    )
    validation_af_analytic_drift = max(
        float(validation_datasets["af"][index]["analytic_pair_drift_eur_mwh"])
        for index in range(len(validation_datasets["af"]))
    )
    raw_dim = 1 + len(unit_columns) + 5
    canonical_dim = len(unit_columns) + 5

    models: dict[str, list[Any]] = {}
    fits: dict[str, list[dict[str, Any]]] = {}
    for method, (representation, kind) in NEURAL_METHODS.items():
        method_models = []
        method_fits = []
        for seed in seeds:
            if kind == "cqdm":
                model, fit = fit_multizone(
                    "cqdm_gauge_residual",
                    seed,
                    train_datasets[representation],
                    validation_datasets[representation],
                    raw_dim=raw_dim,
                    canonical_dim=canonical_dim,
                    zone_count=len(zones),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    device=device,
                )
            else:
                model, fit = fit_factorial_control(
                    "raw_deepset_analytic_residual_gauge",
                    seed,
                    train_datasets[representation],
                    validation_datasets[representation],
                    raw_dim=raw_dim,
                    zone_count=len(zones),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    device=device,
                )
            method_models.append(model)
            method_fits.append(fit)
            print(
                f"[FIT] method={method} seed={seed} epoch={fit['best_epoch']} "
                f"validation={fit['validation_objective']:.6f}",
                flush=True,
            )
        models[method] = method_models
        fits[method] = method_fits

    def predict_members(
        method: str, evaluation_dataset: RealAffineGaugeDataset
    ) -> list[dict[str, Any]]:
        _, kind = NEURAL_METHODS[method]
        predictions = []
        for model in models[method]:
            if kind == "cqdm":
                value = predict_multizone(
                    model,
                    evaluation_dataset,
                    batch_size=args.batch_size,
                    target_scale=target_scale,
                    device=device,
                )
            else:
                value = predict_factorial_control(
                    model,
                    evaluation_dataset,
                    batch_size=args.batch_size,
                    target_scale=target_scale,
                    device=device,
                )
            predictions.append(value)
        return predictions

    validation_predictions: dict[str, dict[str, np.ndarray]] = {}
    for method, (representation, _) in NEURAL_METHODS.items():
        validation_predictions[method] = _ensemble(
            predict_members(method, validation_datasets[representation])
        )
    validation_predictions["analytic_gauge"] = _predict_analytic(
        validation_datasets["af"]
    )
    validation_target = np.stack(
        [sample["target_vector"] for sample in validation_samples]
    )
    validation_pairs = field_to_pairwise(validation_target)
    thresholds = {
        method: float(
            np.quantile(
                np.max(np.abs(validation_pairs - prediction["pairwise"]), axis=1),
                args.alert_quantile,
            )
        )
        for method, prediction in validation_predictions.items()
    }

    stresses = presentation_names(geometry)
    target_field = np.stack([sample["target_vector"] for sample in evaluation_samples])
    timestamps_ns = np.asarray(
        [pd.Timestamp(sample["timestamp"]).value for sample in evaluation_samples],
        dtype=np.int64,
    )
    results: dict[str, Any] = {
        method: {
            "validation_max_pair_error_alert_threshold_eur_mwh": thresholds[method],
            "stresses": {},
        }
        for method in ALL_METHODS
    }
    losses: dict[str, dict[str, np.ndarray]] = {method: {} for method in ALL_METHODS}
    member_losses: dict[str, dict[str, np.ndarray]] = {
        method: {} for method in NEURAL_METHODS
    }
    original_predictions: dict[str, np.ndarray] = {}
    original_member_predictions: dict[str, list[np.ndarray]] = {}
    evaluation_af_analytic_drift: dict[str, float] = {}

    for stress_index, stress in enumerate(stresses, start=1):
        print(f"[EVAL] stress={stress} ({stress_index}/{len(stresses)})", flush=True)
        stress_datasets = {
            representation: dataset(evaluation_samples, representation, stress)
            for representation in ("ambient", "slack", "af")
        }
        evaluation_af_analytic_drift[stress] = max(
            float(
                stress_datasets["af"][index]["analytic_pair_drift_eur_mwh"]
            )
            for index in range(len(stress_datasets["af"]))
        )
        predictions: dict[str, dict[str, np.ndarray]] = {}
        members_by_method: dict[str, list[dict[str, Any]]] = {}
        for method, (representation, _) in NEURAL_METHODS.items():
            members = predict_members(method, stress_datasets[representation])
            members_by_method[method] = members
            predictions[method] = _ensemble(members)
        predictions["analytic_gauge"] = _predict_analytic(stress_datasets["af"])

        if stress == "original":
            original_predictions = {
                method: np.asarray(value["pairwise"]).copy()
                for method, value in predictions.items()
            }
            original_member_predictions = {
                method: [np.asarray(member["pairwise"]).copy() for member in members]
                for method, members in members_by_method.items()
            }
        if not original_predictions:
            raise RuntimeError("The original presentation must be evaluated first")

        for method, prediction in predictions.items():
            summary, timestamp_loss = _summarize_method(
                target_field,
                prediction["field"],
                np.asarray(prediction["pairwise"]),
                original_predictions[method],
                thresholds[method],
                len(zones),
            )
            _add_hodge_metrics(
                summary, target_field, np.asarray(prediction["pairwise"]), len(zones)
            )
            if method in members_by_method:
                summary["seed_members"] = []
                stress_member_losses = []
                for member_index, (seed, member) in enumerate(
                    zip(seeds, members_by_method[method], strict=True)
                ):
                    member_summary, member_timestamp_loss = _summarize_method(
                        target_field,
                        member["field"],
                        np.asarray(member["pairwise"]),
                        original_member_predictions[method][member_index],
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
                            "mean_abs_pairwise_prediction_drift_eur_mwh": member_summary[
                                "mean_abs_pairwise_prediction_drift_eur_mwh"
                            ],
                        }
                    )
                    stress_member_losses.append(
                        np.asarray(member_timestamp_loss, dtype=np.float32)
                    )
                member_losses[method][stress] = np.stack(stress_member_losses)
            results[method]["stresses"][stress] = summary
            losses[method][stress] = np.asarray(timestamp_loss, dtype=np.float32)

    max_drift = {
        method: max(
            float(metrics["max_abs_pairwise_prediction_drift_eur_mwh"])
            for metrics in results[method]["stresses"].values()
        )
        for method in ALL_METHODS
    }
    affine_only_stresses = tuple(
        stress
        for stress in stresses
        if not parse_presentation(stress, geometry).combined
    )
    affine_only_max_drift = {
        method: max(
            float(
                results[method]["stresses"][stress][
                    "max_abs_pairwise_prediction_drift_eur_mwh"
                ]
            )
            for stress in affine_only_stresses
        )
        for method in ALL_METHODS
    }
    exact_methods = ("af_qdm_residual", "analytic_gauge")
    failed = {
        method: max_drift[method]
        for method in exact_methods
        if max_drift[method] > args.invariance_tolerance_eur_mwh
    }
    if failed:
        raise RuntimeError(
            "AF-CDMO v13 exact prediction-invariance contract failed before write: "
            + json.dumps(failed, sort_keys=True)
        )
    projection_only_drift = affine_only_max_drift["af_projected_raw_residual"]
    if projection_only_drift > args.invariance_tolerance_eur_mwh:
        raise RuntimeError(
            "AF-CDMO v13 projection-only affine-invariance contract failed "
            f"before write: {projection_only_drift}"
        )

    comparisons: dict[str, Any] = {}
    for baseline in (
        "ambient_cqdm_residual",
        "slack_qdm_residual",
        "af_projected_raw_residual",
        "analytic_gauge",
    ):
        candidate = "af_qdm_residual"
        comparisons[f"original::{baseline}_to_{candidate}"] = _paired(
            losses[baseline]["original"],
            losses[candidate]["original"],
            key="timestamp_pairwise_mae",
        )
        baseline_stack = np.stack([losses[baseline][stress] for stress in stresses])
        candidate_stack = np.stack([losses[candidate][stress] for stress in stresses])
        comparisons[f"orbit_mean::{baseline}_to_{candidate}"] = _paired(
            baseline_stack.mean(axis=0),
            candidate_stack.mean(axis=0),
            key="timestamp_orbit_mean_pairwise_mae",
        )
        comparisons[f"adversarial_envelope::{baseline}_to_{candidate}"] = _paired(
            baseline_stack.max(axis=0),
            candidate_stack.max(axis=0),
            key="timestamp_adversarial_pairwise_mae",
        )
        comparisons[f"worst_fixed::{baseline}_to_{candidate}"] = {
            "baseline": baseline,
            "candidate": candidate,
            "baseline_mae_eur_mwh": max(
                float(results[baseline]["stresses"][stress]["mean_pairwise_mae_eur_mwh"])
                for stress in stresses
            ),
            "candidate_mae_eur_mwh": max(
                float(results[candidate]["stresses"][stress]["mean_pairwise_mae_eur_mwh"])
                for stress in stresses
            ),
        }
        if baseline in NEURAL_METHODS:
            baseline_members = np.stack(
                [member_losses[baseline][stress] for stress in stresses]
            )
            candidate_members = np.stack(
                [member_losses[candidate][stress] for stress in stresses]
            )
            comparisons[f"seed_level::{baseline}_to_{candidate}"] = {
                "original": _seed_sign_test(
                    member_losses[baseline]["original"].mean(axis=1),
                    member_losses[candidate]["original"].mean(axis=1),
                ),
                "orbit_mean": _seed_sign_test(
                    baseline_members.mean(axis=(0, 2)),
                    candidate_members.mean(axis=(0, 2)),
                ),
                "adversarial_envelope": _seed_sign_test(
                    baseline_members.max(axis=0).mean(axis=1),
                    candidate_members.max(axis=0).mean(axis=1),
                ),
                "worst_fixed": _seed_sign_test(
                    baseline_members.mean(axis=2).max(axis=0),
                    candidate_members.mean(axis=2).max(axis=0),
                ),
            }

    frozen_reproduction = _frozen_v11_reproduction_contract(
        study_role=args.study_role,
        zones=zones,
        seeds=seeds,
        fit_end=fit_end,
        validation_end=validation_end,
        evaluation_end=evaluation_end,
        results=results,
    )

    loss_arrays: dict[str, np.ndarray] = {"timestamps_utc_ns": timestamps_ns}
    for method in ALL_METHODS:
        for stress in stresses:
            loss_arrays[f"loss__{method}__{stress}"] = losses[method][stress]
    for method in NEURAL_METHODS:
        for stress in stresses:
            loss_arrays[f"member_loss__{method}__{stress}"] = member_losses[method][stress]
    _atomic_npz(losses_output, loss_arrays)

    report = {
        "protocol": {
            "name": "af_cdmo_real_loss_safe_v13",
            "study_role": args.study_role,
            "task": (
                "same-delivery reconstruction of the observed centered Nordic "
                "day-ahead zonal price field from a timely solved certificate"
            ),
            "not_a_forecast": True,
            "zones": list(zones),
            "pairs": [
                f"{zones[left]}-{zones[right]}"
                for left, right in pair_indices(len(zones))
            ],
            "seeds": seeds,
            "presentations": list(stresses),
            "fit_end_exclusive": fit_end.isoformat(),
            "internal_validation_end_exclusive": validation_end.isoformat(),
            "evaluation_end_exclusive": evaluation_end.isoformat(),
            "all_zone_price_panel_coverage": coverage,
            "minimum_required_price_panel_coverage": args.min_price_panel_coverage,
            "equality_rank": geometry.rank,
            "skagerrak_excluded": True,
            "analytic_invariance_tolerance_eur_mwh": (
                args.analytic_invariance_tolerance_eur_mwh
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "source": {
            "dual_archive": str(dual_path),
            "dual_archive_sha256": dual_hash,
            "geometry_audit": str(geometry_audit_path),
            "geometry_audit_sha256": sha256_file(geometry_audit_path),
            "day_ahead_prices": price_provenance,
            "certificate_audit": certificate_audit,
        },
        "rows": {
            "fit": len(train_samples),
            "internal_validation": len(validation_samples),
            "evaluation": len(evaluation_samples),
        },
        "target_scale_eur_mwh": target_scale,
        "fits": fits,
        "results": results,
        "paired_comparisons": comparisons,
        "frozen_v11_reproduction_contract": frozen_reproduction,
        "invariance_contract": {
            "tolerance_eur_mwh": args.invariance_tolerance_eur_mwh,
            "method_max_abs_prediction_drift_eur_mwh": max_drift,
            "affine_only_presentations": list(affine_only_stresses),
            "method_affine_only_max_abs_prediction_drift_eur_mwh": (
                affine_only_max_drift
            ),
            "exact_methods": list(exact_methods),
            "projection_only_exact_on_affine_only_presentations": True,
            "analytic_pair_drift_eur_mwh": {
                "fit_original": train_af_analytic_drift,
                "validation_original": validation_af_analytic_drift,
                "evaluation_by_presentation": evaluation_af_analytic_drift,
                "maximum": max(
                    train_af_analytic_drift,
                    validation_af_analytic_drift,
                    *evaluation_af_analytic_drift.values(),
                ),
                "tolerance": args.analytic_invariance_tolerance_eur_mwh,
            },
            "status": "passed",
        },
        "timestamp_loss_archive": {
            "path": str(losses_output),
            "sha256": sha256_file(losses_output),
        },
        "interpretation_contract": {
            "primary_claim": (
                "Formulation robustness under documented loss-safe affine null rewrites."
            ),
            "clean_accuracy_not_assumed": True,
            "prohibited_claims": [
                "future-price forecasting superiority",
                "operational welfare improvement",
                "universal optimization-learning superiority",
                "worldwide priority",
            ],
        },
        "reproducibility": {
            "engines": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path.resolve()),
                }
                for path in ENGINE_PATHS
            ]
        },
    }
    _atomic_json(output, report)
    print(f"[OK] wrote {output}")
    print(f"[OK] wrote {losses_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
