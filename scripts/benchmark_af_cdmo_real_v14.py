"""Confirm loss-safe AF-QDM with matched fairness controls.

The task reconstructs a same-delivery centered zonal day-ahead price field from
a causally published solved flow-based certificate.  It is not a future-price
forecast.  This isolated v14 companion adds a parameter-matched uniform-mass
ablation and an augmentation-trained ambient control without changing the
completed v13 engines or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

try:
    from scripts.af_cdmo_real_controls_v14 import (
        GaugeAugmentedAmbientDataset,
        FixedEqualityGaugeDataset,
        NonEquivalentRAMDataset,
        NonuniformRefinementAFDataset,
        UniformMassDataset,
        control_contract,
    )
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
        RawFactorialControlRegressor,
        fit_factorial_control,
        predict_factorial_control,
    )
    from scripts.multizone_quotient_gauge_v6 import (
        MultiZoneRegressor,
        REAL_NORDIC_ZONES,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
    )
except ModuleNotFoundError:
    from af_cdmo_real_controls_v14 import (
        GaugeAugmentedAmbientDataset,
        FixedEqualityGaugeDataset,
        NonEquivalentRAMDataset,
        NonuniformRefinementAFDataset,
        UniformMassDataset,
        control_contract,
    )
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
        RawFactorialControlRegressor,
        fit_factorial_control,
        predict_factorial_control,
    )
    from multizone_quotient_gauge_v6 import (
        MultiZoneRegressor,
        REAL_NORDIC_ZONES,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
    )


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "official_jao_dual_measure"
DEFAULT_OUTPUT = REPORT_ROOT / "af_cdmo_real_development_v14.json"
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
    "af_uniform_attention_residual": ("af", "uniform"),
    "ambient_gauge_augmented_residual": ("ambient", "augmented"),
}
ALL_METHODS = (*NEURAL_METHODS, "analytic_gauge")
GAUGE_AMPLITUDES = (0.125, 0.25, 0.5, 1.0, 1.5, 3.0, 6.0)
RAM_PERTURBATIONS = (-0.20, -0.05, 0.05, 0.20)
TRAINING_PROTOCOL_VERSION = "af_cdmo_v14_matched_controls_v1"
TRAINING_ENGINE_PATHS = (
    ROOT / "scripts" / "af_cdmo_real_controls_v14.py",
    ROOT / "scripts" / "af_cdmo_real_extension_v13.py",
    ROOT / "scripts" / "benchmark_certificate_governance_v5.py",
    ROOT / "scripts" / "benchmark_multizone_quotient_gauge_v6.py",
    ROOT / "scripts" / "multizone_quotient_gauge_v6.py",
    ROOT / "scripts" / "multizone_factorial_controls_v10.py",
)
ENGINE_PATHS = (
    Path(__file__),
    ROOT / "scripts" / "af_cdmo_real_controls_v14.py",
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


def _atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _stable_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


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


def _rank_balanced_groups(values: np.ndarray, bins: int = 5) -> list[np.ndarray]:
    """Partition finite scalar diagnostics into deterministic, balanced ranks.

    Numeric quantile edges can produce empty bins when a conditioning statistic
    has many ties. Stable rank partitioning preserves every observation exactly
    once and keeps bin sizes within one row. Tied values may consequently span
    adjacent bins, which is recorded in the report rather than hidden.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rank-balanced values must be a finite one-dimensional array")
    if bins < 1 or len(values) < bins:
        raise ValueError("Rank-balanced partition needs at least one row per bin")
    order = np.argsort(values, kind="stable")
    return [np.asarray(group, dtype=np.int64) for group in np.array_split(order, bins)]


def _conditioning_ratio(
    sample: dict[str, Any], geometry: RealAffineGeometry, ptdf_count: int
) -> float | None:
    """Return the weakest projected/raw row norm ratio when it is defined.

    Some valid certificates contain no active rows. Their analytic contribution
    is the zero measure, but a minimum row-wise conditioning ratio is undefined.
    They remain in every predictive metric and are excluded only from this
    row-conditioning stratification.
    """

    raw = np.asarray(sample["raw"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0:
        return None
    normals = raw[:, 1 : 1 + ptdf_count]
    if normals.shape != (raw.shape[0], ptdf_count):
        raise ValueError(
            "Certificate row width is incompatible with the PTDF geometry: "
            f"raw={raw.shape}, ptdf_count={ptdf_count}"
        )
    ambient_norm = np.linalg.norm(normals, axis=1)
    projected_norm = np.linalg.norm(geometry.project(normals), axis=1)
    valid = (
        np.isfinite(ambient_norm)
        & np.isfinite(projected_norm)
        & (ambient_norm > 0.0)
    )
    if not valid.any():
        return None
    ratio = projected_norm[valid] / ambient_norm[valid]
    if not np.isfinite(ratio).all():
        raise RuntimeError("Non-finite projected-to-ambient conditioning ratio")
    return float(np.min(ratio))


def _frozen_v11_reproduction_contract(
    *,
    study_role: str,
    zones: tuple[str, ...],
    seeds: list[int],
    fit_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    results: dict[str, Any],
    absolute_tolerance: float = 1.0e-5,
    relative_tolerance: float = 1.0e-6,
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
    max_normalized_error = 0.0
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
        allowed_differences: dict[str, float] = {}
        normalized_errors: dict[str, float] = {}

        def record(name: str, current_value: float, reference_value: float) -> None:
            difference = abs(current_value - reference_value)
            scale = max(abs(current_value), abs(reference_value))
            allowed = absolute_tolerance + relative_tolerance * scale
            differences[name] = difference
            allowed_differences[name] = allowed
            normalized_errors[name] = difference / allowed

        record(
            "validation_alert_threshold",
            float(current_method["validation_max_pair_error_alert_threshold_eur_mwh"]),
            float(
                reference_method[
                    "validation_max_pair_error_alert_threshold_eur_mwh"
                ]
            ),
        )
        current_original = current_method["stresses"]["original"]
        reference_original = reference_method["stresses"]["original"]
        for metric in metric_names:
            record(
                metric,
                float(current_original[metric]),
                float(reference_original[metric]),
            )
        current_members = current_original["seed_members"]
        reference_members = reference_original["seed_members"]
        if [item["seed"] for item in current_members] != [
            item["seed"] for item in reference_members
        ]:
            raise RuntimeError(f"Frozen v11 seed order changed for {method}")
        member_checks = []
        for current, frozen in zip(current_members, reference_members, strict=True):
            current_value = float(current["mean_pairwise_mae_eur_mwh"])
            reference_value = float(frozen["mean_pairwise_mae_eur_mwh"])
            difference = abs(current_value - reference_value)
            allowed = absolute_tolerance + relative_tolerance * max(
                abs(current_value), abs(reference_value)
            )
            member_checks.append((difference, allowed, difference / allowed))
        differences["max_seed_member_pairwise_mae"] = max(
            item[0] for item in member_checks
        )
        allowed_differences["max_seed_member_pairwise_mae"] = max(
            item[1] for item in member_checks
        )
        normalized_errors["max_seed_member_pairwise_mae"] = max(
            item[2] for item in member_checks
        )
        method_max = max(differences.values())
        method_max_normalized = max(normalized_errors.values())
        max_abs_difference = max(max_abs_difference, method_max)
        max_normalized_error = max(max_normalized_error, method_max_normalized)
        checked[method] = {
            "absolute_differences": differences,
            "allowed_differences": allowed_differences,
            "normalized_errors": normalized_errors,
            "max_abs_difference": method_max,
            "max_normalized_error": method_max_normalized,
        }
    if max_normalized_error > 1.0:
        raise RuntimeError(
            "AF-CDMO v14 failed to reproduce frozen v11 clean-form controls: "
            f"max_abs_difference={max_abs_difference:.6g}, "
            f"max_normalized_error={max_normalized_error:.6g}, "
            f"absolute_tolerance={absolute_tolerance:.6g}, "
            f"relative_tolerance={relative_tolerance:.6g}, "
            f"checked={json.dumps(checked, sort_keys=True)}"
        )
    return {
        "status": "passed",
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "tolerance": absolute_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_abs_difference": max_abs_difference,
        "max_normalized_error": max_normalized_error,
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
        "--checkpoint_dir",
        type=Path,
        help=(
            "Resumable per-method/per-seed model checkpoints. Defaults to a "
            "directory beside --output whose name ends in _checkpoints."
        ),
    )
    parser.add_argument(
        "--evaluation_progress_path",
        type=Path,
        help=(
            "Atomic resumable evaluation state. Defaults beside --output with "
            "a _evaluation_progress.pt suffix."
        ),
    )
    parser.add_argument(
        "--no_resume_evaluation_progress",
        action="store_true",
        help="Ignore compatible per-presentation evaluation progress.",
    )
    parser.add_argument(
        "--no_resume_checkpoints",
        action="store_true",
        help="Ignore compatible checkpoints and refit every model.",
    )
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
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else output.with_name(output.stem + "_checkpoints")
    )
    evaluation_progress_path = (
        args.evaluation_progress_path.expanduser().resolve()
        if args.evaluation_progress_path is not None
        else output.with_name(output.stem + "_evaluation_progress.pt")
    )
    existing = [path for path in (output, losses_output) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing v14 artifacts; use new paths or --force: "
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
    training_signature_payload = {
        "format_version": 1,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "study_role": args.study_role,
        "methods": NEURAL_METHODS,
        "zones": list(zones),
        "seeds": seeds,
        "fit_end_exclusive": fit_end.isoformat(),
        "validation_end_exclusive": validation_end.isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dual_archive_sha256": dual_hash,
        "geometry_audit_sha256": sha256_file(geometry_audit_path),
        "price_provenance": price_provenance,
        "fit_rows": len(train_samples),
        "validation_rows": len(validation_samples),
        "target_scale": target_scale,
        "canonical_ram_scale": canonical_ram_scale,
        "raw_scales": {
            key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in raw_scales.items()
        },
        "training_engines": {
            str(path.resolve()): sha256_file(path.resolve())
            for path in TRAINING_ENGINE_PATHS
        },
    }
    training_signature = _stable_digest(training_signature_payload)

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
    uniform_train_dataset = UniformMassDataset(train_datasets["af"])
    uniform_validation_dataset = UniformMassDataset(validation_datasets["af"])
    augmented_train_dataset = GaugeAugmentedAmbientDataset(
        train_datasets["ambient"]
    )
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

    def new_model(kind: str) -> torch.nn.Module:
        if kind in {"cqdm", "uniform", "augmented"}:
            return MultiZoneRegressor(
                "cqdm_gauge_residual",
                raw_dim,
                canonical_dim,
                len(zones),
            ).to(device)
        return RawFactorialControlRegressor(
            "raw_deepset_analytic_residual_gauge", raw_dim, len(zones)
        ).to(device)

    models: dict[str, list[Any]] = {}
    fits: dict[str, list[dict[str, Any]]] = {}
    for method, (representation, kind) in NEURAL_METHODS.items():
        method_train_dataset = (
            uniform_train_dataset
            if kind == "uniform"
            else augmented_train_dataset
            if kind == "augmented"
            else train_datasets[representation]
        )
        method_validation_dataset = (
            uniform_validation_dataset
            if kind == "uniform"
            else validation_datasets["ambient"]
            if kind == "augmented"
            else validation_datasets[representation]
        )
        method_models = []
        method_fits = []
        for seed in seeds:
            checkpoint_path = checkpoint_dir / f"{method}__seed{seed}.pt"
            if checkpoint_path.is_file() and not args.no_resume_checkpoints:
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                if checkpoint.get("training_signature") != training_signature:
                    raise RuntimeError(
                        "Refusing incompatible model checkpoint: "
                        f"{checkpoint_path}. Remove it or use a different "
                        "--checkpoint_dir."
                    )
                if checkpoint.get("method") != method or int(
                    checkpoint.get("seed", -1)
                ) != seed:
                    raise RuntimeError(
                        f"Checkpoint identity mismatch: {checkpoint_path}"
                    )
                model = new_model(kind)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                fit = dict(checkpoint["fit"])
                fit["checkpoint_reused"] = True
                print(
                    f"[CHECKPOINT] method={method} seed={seed} loaded "
                    f"{checkpoint_path}",
                    flush=True,
                )
            else:
                if kind in {"cqdm", "uniform", "augmented"}:
                    model, fit = fit_multizone(
                        "cqdm_gauge_residual",
                        seed,
                        method_train_dataset,
                        method_validation_dataset,
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
                fit = dict(fit)
                fit["checkpoint_reused"] = False
                _atomic_torch(
                    checkpoint_path,
                    {
                        "format_version": 1,
                        "training_signature": training_signature,
                        "training_signature_payload": training_signature_payload,
                        "method": method,
                        "kind": kind,
                        "seed": seed,
                        "fit": fit,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                    },
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
        method: str, evaluation_dataset: Any
    ) -> list[dict[str, Any]]:
        _, kind = NEURAL_METHODS[method]
        predictions = []
        for model in models[method]:
            if kind in {"cqdm", "uniform", "augmented"}:
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
    for method, (representation, kind) in NEURAL_METHODS.items():
        method_validation_dataset = (
            uniform_validation_dataset
            if kind == "uniform"
            else validation_datasets["ambient"]
            if kind == "augmented"
            else validation_datasets[representation]
        )
        validation_predictions[method] = _ensemble(
            predict_members(method, method_validation_dataset)
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
    frozen_reproduction: dict[str, Any] | None = None

    evaluation_signature_payload = {
        "format_version": 1,
        "training_signature": training_signature,
        "study_role": args.study_role,
        "zones": list(zones),
        "seeds": seeds,
        "evaluation_end_exclusive": evaluation_end.isoformat(),
        "evaluation_rows": len(evaluation_samples),
        "presentations": list(stresses),
        "alert_quantile": args.alert_quantile,
        "analytic_invariance_tolerance_eur_mwh": (
            args.analytic_invariance_tolerance_eur_mwh
        ),
        "invariance_tolerance_eur_mwh": args.invariance_tolerance_eur_mwh,
        "benchmark_sha256": sha256_file(Path(__file__).resolve()),
    }
    evaluation_signature = _stable_digest(evaluation_signature_payload)
    completed_stresses: set[str] = set()
    if (
        evaluation_progress_path.is_file()
        and not args.no_resume_evaluation_progress
    ):
        progress = torch.load(
            evaluation_progress_path, map_location="cpu", weights_only=False
        )
        if progress.get("evaluation_signature") != evaluation_signature:
            raise RuntimeError(
                "Refusing incompatible evaluation progress: "
                f"{evaluation_progress_path}. Remove it or pass "
                "--no_resume_evaluation_progress."
            )
        completed_stresses = set(progress["completed_stresses"])
        if not completed_stresses.issubset(stresses):
            raise RuntimeError("Evaluation progress contains unknown presentations")
        results = progress["results"]
        losses = progress["losses"]
        member_losses = progress["member_losses"]
        original_predictions = progress["original_predictions"]
        original_member_predictions = progress["original_member_predictions"]
        evaluation_af_analytic_drift = progress["evaluation_af_analytic_drift"]
        frozen_reproduction = progress.get("frozen_reproduction")
        print(
            f"[EVAL CHECKPOINT] loaded {len(completed_stresses)}/{len(stresses)} "
            f"presentations from {evaluation_progress_path}",
            flush=True,
        )

    for stress_index, stress in enumerate(stresses, start=1):
        if stress in completed_stresses:
            print(
                f"[EVAL CHECKPOINT] stress={stress} "
                f"({stress_index}/{len(stresses)}) loaded",
                flush=True,
            )
            continue
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
        for method, (representation, kind) in NEURAL_METHODS.items():
            method_stress_dataset = (
                UniformMassDataset(stress_datasets["af"])
                if kind == "uniform"
                else stress_datasets["ambient"]
                if kind == "augmented"
                else stress_datasets[representation]
            )
            members = predict_members(method, method_stress_dataset)
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

        if stress == "original":
            frozen_reproduction = _frozen_v11_reproduction_contract(
                study_role=args.study_role,
                zones=zones,
                seeds=seeds,
                fit_end=fit_end,
                validation_end=validation_end,
                evaluation_end=evaluation_end,
                results=results,
            )

        completed_stresses.add(stress)
        _atomic_torch(
            evaluation_progress_path,
            {
                "format_version": 1,
                "evaluation_signature": evaluation_signature,
                "evaluation_signature_payload": evaluation_signature_payload,
                "completed_stresses": [
                    name for name in stresses if name in completed_stresses
                ],
                "results": results,
                "losses": losses,
                "member_losses": member_losses,
                "original_predictions": original_predictions,
                "original_member_predictions": original_member_predictions,
                "evaluation_af_analytic_drift": evaluation_af_analytic_drift,
                "frozen_reproduction": frozen_reproduction,
            },
        )

    base_af_evaluation = dataset(evaluation_samples, "af", "original")
    refined_af_evaluation = NonuniformRefinementAFDataset(base_af_evaluation)
    mass_refinement: dict[str, Any] = {
        "name": "held_out_nonuniform_mass_conserving_refinement",
        "training_exposure": False,
        "multiplicity_range": [1, 3],
        "methods": {},
    }
    maximum_measure_moment_drift = 0.0
    maximum_total_drift = 0.0
    for index in range(len(base_af_evaluation)):
        base_item = base_af_evaluation[index]
        refined_item = refined_af_evaluation[index]
        base_moment = base_item["weights"] @ base_item["canonical"]
        refined_moment = refined_item["weights"] @ refined_item["canonical"]
        maximum_measure_moment_drift = max(
            maximum_measure_moment_drift,
            float(np.max(np.abs(base_moment - refined_moment), initial=0.0)),
        )
        maximum_total_drift = max(
            maximum_total_drift,
            abs(float(base_item["total"]) - float(refined_item["total"])),
        )
    mass_refinement["maximum_conserved_moment_drift"] = (
        maximum_measure_moment_drift
    )
    mass_refinement["maximum_total_mass_drift"] = maximum_total_drift

    for method, refined_dataset in (
        ("af_qdm_residual", refined_af_evaluation),
        (
            "af_uniform_attention_residual",
            UniformMassDataset(refined_af_evaluation),
        ),
    ):
        members = predict_members(method, refined_dataset)
        ensemble = _ensemble(members)
        summary, _ = _summarize_method(
            target_field,
            ensemble["field"],
            np.asarray(ensemble["pairwise"]),
            original_predictions[method],
            thresholds[method],
            len(zones),
        )
        member_records = []
        for member_index, (seed, member) in enumerate(
            zip(seeds, members, strict=True)
        ):
            member_summary, _ = _summarize_method(
                target_field,
                member["field"],
                np.asarray(member["pairwise"]),
                original_member_predictions[method][member_index],
                thresholds[method],
                len(zones),
            )
            member_records.append(
                {
                    "seed": seed,
                    "mean_pairwise_mae_eur_mwh": member_summary[
                        "mean_pairwise_mae_eur_mwh"
                    ],
                    "mean_abs_pairwise_prediction_drift_eur_mwh": member_summary[
                        "mean_abs_pairwise_prediction_drift_eur_mwh"
                    ],
                    "max_abs_pairwise_prediction_drift_eur_mwh": member_summary[
                        "max_abs_pairwise_prediction_drift_eur_mwh"
                    ],
                }
            )
        mass_refinement["methods"][method] = {
            "mean_pairwise_mae_eur_mwh": summary["mean_pairwise_mae_eur_mwh"],
            "mean_abs_pairwise_prediction_drift_eur_mwh": summary[
                "mean_abs_pairwise_prediction_drift_eur_mwh"
            ],
            "max_abs_pairwise_prediction_drift_eur_mwh": summary[
                "max_abs_pairwise_prediction_drift_eur_mwh"
            ],
            "seed_members": member_records,
        }
    if (
        mass_refinement["methods"]["af_qdm_residual"][
            "max_abs_pairwise_prediction_drift_eur_mwh"
        ]
        > args.invariance_tolerance_eur_mwh
    ):
        raise RuntimeError(
            "AF-QDM failed the held-out nonuniform refinement contract"
        )

    gauge_amplitude_sensitivity: dict[str, Any] = {
        "amplitudes": list(GAUGE_AMPLITUDES),
        "training_amplitudes": control_contract()["ambient_gauge_augmentation"][
            "amplitudes"
        ],
        "methods": {},
    }
    sensitivity_methods = (
        "ambient_cqdm_residual",
        "slack_qdm_residual",
        "af_qdm_residual",
        "af_uniform_attention_residual",
        "ambient_gauge_augmented_residual",
    )
    for method in sensitivity_methods:
        representation, kind = NEURAL_METHODS[method]
        method_records = []
        for amplitude in GAUGE_AMPLITUDES:
            gauge_dataset: Any = FixedEqualityGaugeDataset(
                dataset(evaluation_samples, representation, "original"),
                amplitude=amplitude,
            )
            if kind == "uniform":
                gauge_dataset = UniformMassDataset(gauge_dataset)
            members = predict_members(method, gauge_dataset)
            ensemble = _ensemble(members)
            summary, _ = _summarize_method(
                target_field,
                ensemble["field"],
                np.asarray(ensemble["pairwise"]),
                original_predictions[method],
                thresholds[method],
                len(zones),
            )
            method_records.append(
                {
                    "amplitude": amplitude,
                    "mean_pairwise_mae_eur_mwh": summary[
                        "mean_pairwise_mae_eur_mwh"
                    ],
                    "mean_abs_pairwise_prediction_drift_eur_mwh": summary[
                        "mean_abs_pairwise_prediction_drift_eur_mwh"
                    ],
                    "max_abs_pairwise_prediction_drift_eur_mwh": summary[
                        "max_abs_pairwise_prediction_drift_eur_mwh"
                    ],
                }
            )
        gauge_amplitude_sensitivity["methods"][method] = method_records
    maximum_af_amplitude_drift = max(
        record["max_abs_pairwise_prediction_drift_eur_mwh"]
        for record in gauge_amplitude_sensitivity["methods"]["af_qdm_residual"]
    )
    gauge_amplitude_sensitivity["af_qdm_maximum_drift_eur_mwh"] = (
        maximum_af_amplitude_drift
    )
    if maximum_af_amplitude_drift > args.invariance_tolerance_eur_mwh:
        raise RuntimeError("AF-QDM failed the gauge-amplitude invariance sweep")

    non_equivalent_ram_sensitivity: dict[str, Any] = {
        "interpretation": (
            "Negative control: normalized RAM changes the represented feasible "
            "domain; prediction invariance is not expected."
        ),
        "perturbations": list(RAM_PERTURBATIONS),
        "methods": {},
    }
    for method in ("af_qdm_residual", "af_uniform_attention_residual"):
        method_records = []
        for delta in RAM_PERTURBATIONS:
            ram_dataset: Any = NonEquivalentRAMDataset(
                base_af_evaluation, delta=delta
            )
            if method == "af_uniform_attention_residual":
                ram_dataset = UniformMassDataset(ram_dataset)
            members = predict_members(method, ram_dataset)
            ensemble = _ensemble(members)
            summary, _ = _summarize_method(
                target_field,
                ensemble["field"],
                np.asarray(ensemble["pairwise"]),
                original_predictions[method],
                thresholds[method],
                len(zones),
            )
            method_records.append(
                {
                    "delta_canonical_ram": delta,
                    "mean_abs_pairwise_prediction_drift_eur_mwh": summary[
                        "mean_abs_pairwise_prediction_drift_eur_mwh"
                    ],
                    "max_abs_pairwise_prediction_drift_eur_mwh": summary[
                        "max_abs_pairwise_prediction_drift_eur_mwh"
                    ],
                }
            )
        non_equivalent_ram_sensitivity["methods"][method] = method_records

    condition_values = np.full(len(evaluation_samples), np.nan, dtype=np.float64)
    excluded_conditioning_indices = []
    for sample_index, sample in enumerate(evaluation_samples):
        ratio = _conditioning_ratio(sample, geometry, len(unit_columns))
        if ratio is None:
            excluded_conditioning_indices.append(sample_index)
        else:
            condition_values[sample_index] = ratio
    valid_conditioning_indices = np.flatnonzero(np.isfinite(condition_values))
    condition_values_array = condition_values[valid_conditioning_indices]
    if len(condition_values_array) < 5:
        raise RuntimeError(
            "Conditioning sensitivity requires at least five certificates with "
            "one or more valid active rows"
        )
    condition_edges = np.quantile(condition_values_array, np.linspace(0.0, 1.0, 6))
    local_condition_groups = _rank_balanced_groups(condition_values_array, bins=5)
    condition_groups = [
        valid_conditioning_indices[group] for group in local_condition_groups
    ]
    conditioning_sensitivity: dict[str, Any] = {
        "statistic": "minimum row projected_norm / ambient_norm per certificate",
        "partition": "stable_rank_balanced_quintiles",
        "ties_may_span_adjacent_bins": True,
        "quantile_edges": condition_edges.tolist(),
        "evaluation_rows": len(evaluation_samples),
        "included_rows": int(len(valid_conditioning_indices)),
        "excluded_rows": int(len(excluded_conditioning_indices)),
        "exclusion_rule": (
            "Certificates with no active rows, or no finite positive-norm PTDF "
            "rows, have no defined row-conditioning statistic. They remain in "
            "all predictive metrics and are excluded only from this diagnostic."
        ),
        "excluded_timestamps_utc": [
            pd.Timestamp(evaluation_samples[index]["timestamp"]).isoformat()
            for index in excluded_conditioning_indices
        ],
        "bins": [],
    }
    for bin_index, indices in enumerate(condition_groups):
        selected = np.zeros(len(evaluation_samples), dtype=bool)
        selected[indices] = True
        observed = condition_values[indices]
        conditioning_sensitivity["bins"].append(
            {
                "bin": bin_index + 1,
                "nominal_quantile_lower": float(condition_edges[bin_index]),
                "nominal_quantile_upper": float(condition_edges[bin_index + 1]),
                "observed_min": float(observed.min()),
                "observed_max": float(observed.max()),
                "rows": int(len(indices)),
                "original_mean_pairwise_mae_eur_mwh": {
                    method: (
                        float(losses[method]["original"][selected].mean())
                        if selected.any()
                        else None
                    )
                    for method in ALL_METHODS
                },
            }
        )

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
            "AF-CDMO v14 exact prediction-invariance contract failed before write: "
            + json.dumps(failed, sort_keys=True)
        )
    projection_only_drift = affine_only_max_drift["af_projected_raw_residual"]
    if projection_only_drift > args.invariance_tolerance_eur_mwh:
        raise RuntimeError(
            "AF-CDMO v14 projection-only affine-invariance contract failed "
            f"before write: {projection_only_drift}"
        )
    uniform_affine_drift = affine_only_max_drift[
        "af_uniform_attention_residual"
    ]
    if uniform_affine_drift > args.invariance_tolerance_eur_mwh:
        raise RuntimeError(
            "AF-CDMO v14 uniform-mass affine-invariance contract failed "
            f"before write: {uniform_affine_drift}"
        )

    fairness = control_contract()
    for baseline, control in (
        ("af_qdm_residual", "af_uniform_attention_residual"),
        ("ambient_cqdm_residual", "ambient_gauge_augmented_residual"),
    ):
        baseline_parameters = [int(item["parameter_count"]) for item in fits[baseline]]
        control_parameters = [int(item["parameter_count"]) for item in fits[control]]
        if baseline_parameters != control_parameters:
            raise RuntimeError(
                f"Parameter-match contract failed for {baseline} and {control}"
            )
        baseline_rows = [int(item["training_rows_per_epoch"]) for item in fits[baseline]]
        control_rows = [int(item["training_rows_per_epoch"]) for item in fits[control]]
        if baseline_rows != control_rows:
            raise RuntimeError(
                f"Sample-budget contract failed for {baseline} and {control}"
            )
        fairness[f"{baseline}_to_{control}"] = {
            "parameter_count_by_seed": baseline_parameters,
            "training_rows_per_epoch_by_seed": baseline_rows,
            "configured_max_epochs": args.epochs,
            "same_optimizer_and_early_stopping_rule": True,
            "realized_update_counts_forced_equal": False,
            "realized_budget_note": (
                "Validation-selected stopping epochs may differ; selected epochs "
                "are reported below and no exact realized-update equality is claimed."
            ),
            "actual_best_epochs_baseline": [
                int(item["best_epoch"]) for item in fits[baseline]
            ],
            "actual_best_epochs_control": [
                int(item["best_epoch"]) for item in fits[control]
            ],
        }

    comparisons: dict[str, Any] = {}
    for baseline in (*NEURAL_METHODS, "analytic_gauge"):
        candidate = "af_qdm_residual"
        if baseline == candidate:
            continue
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

    if frozen_reproduction is None:
        raise RuntimeError("Frozen v11 reproduction contract was not evaluated")

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
            "name": "af_cdmo_real_loss_safe_v14",
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
            "control_freeze": control_contract(),
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
        "fairness_contract": fairness,
        "mass_refinement_contract": mass_refinement,
        "gauge_amplitude_sensitivity": gauge_amplitude_sensitivity,
        "non_equivalent_ram_sensitivity": non_equivalent_ram_sensitivity,
        "conditioning_sensitivity": conditioning_sensitivity,
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
            "uniform_mass_exact_on_affine_only_presentations": True,
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
                "Formulation robustness under documented loss-safe affine null "
                "rewrites, with parameter-matched mass and augmentation controls."
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
            "model_checkpoints": {
                "directory": str(checkpoint_dir),
                "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                "training_signature": training_signature,
                "reused_count": sum(
                    int(bool(item.get("checkpoint_reused")))
                    for method_fits in fits.values()
                    for item in method_fits
                ),
                "files": [
                    {
                        "path": str(checkpoint_dir / f"{method}__seed{seed}.pt"),
                        "sha256": sha256_file(
                            checkpoint_dir / f"{method}__seed{seed}.pt"
                        ),
                    }
                    for method in NEURAL_METHODS
                    for seed in seeds
                ],
            },
            "evaluation_progress": {
                "path": str(evaluation_progress_path),
                "sha256": sha256_file(evaluation_progress_path),
                "evaluation_signature": evaluation_signature,
                "completed_presentations": len(completed_stresses),
            },
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
