"""Post-confirmation temporal robustness audit for frozen QCT-Risk v5.

This script does not train or select a model. It reconstructs the registered
chronology, loads the frozen qct_full checkpoints and their December Platt
maps, reproduces the confirmation predictions, and reports descriptive
monthwise stability plus paired Brier intervals under several temporal block
lengths. The primary registered seven-day result remains authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

try:
    from af_cdmo_probabilistic_headroom import confirm_quotient_tangent_risk_v5 as frozen
    from scripts.af_cdmo_real_extension_v13 import RealAffineGeometry, raw_rows_for_scale
    from scripts.benchmark_af_cdmo_real_v14 import _canonical_ram_scale
    from scripts.benchmark_certificate_governance_v5 import _positive_scales
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(ROOT))
    from af_cdmo_probabilistic_headroom import confirm_quotient_tangent_risk_v5 as frozen
    from scripts.af_cdmo_real_extension_v13 import RealAffineGeometry, raw_rows_for_scale
    from scripts.benchmark_af_cdmo_real_v14 import _canonical_ram_scale
    from scripts.benchmark_certificate_governance_v5 import _positive_scales


DEFAULT_REPORT = ROOT / "af_cdmo_probabilistic_headroom" / "quotient_tangent_risk_confirmation_v5.json"
DEFAULT_OUTPUT_DIR = ROOT / "af_cdmo_probabilistic_headroom" / "qct_risk_temporal_robustness_v5"
DEFAULT_BLOCK_DAYS = (1, 7, 14, 28)
ROWS_PER_DAY = 96
DEFAULT_REPLICATES = 4000
DEFAULT_SEED = 20261002


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block_days", type=int, nargs="+", default=list(DEFAULT_BLOCK_DAYS))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_source(value: str, fallback: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.exists() else fallback.resolve()


def _verify_hash(path: Path, expected: str | None, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if expected and _sha256(path) != expected:
        raise RuntimeError(f"{label} hash differs from the frozen confirmation report: {path}")


def _safe_classification(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int | None]:
    target = np.asarray(target, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-8, 1.0 - 1.0e-8)
    both_classes = np.unique(target).size == 2
    return {
        "rows": int(target.size),
        "prevalence": float(np.mean(target)),
        "mean_probability": float(np.mean(probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0.0, 1.0])),
        "roc_auc": float(roc_auc_score(target, probability)) if both_classes else None,
        "average_precision": float(average_precision_score(target, probability)) if both_classes else None,
    }


def _decision_metrics(target: np.ndarray, probability: np.ndarray, magnitude: np.ndarray, threshold: float) -> dict[str, float | int]:
    flagged = probability > threshold
    retained = ~flagged
    tail_count = int(np.sum(target > 0.5))
    retained_tail_count = int(np.sum((target > 0.5) & retained))
    return {
        "rows": int(target.size),
        "accepted_rows": int(np.sum(retained)),
        "flagged_rows": int(np.sum(flagged)),
        "coverage": float(np.mean(retained)),
        "flag_rate": float(np.mean(flagged)),
        "tail_false_negative_fraction": float(retained_tail_count / max(tail_count, 1)),
        "retained_tail_rate": float(np.mean(target[retained])) if np.any(retained) else float("nan"),
        "retained_mean_residual_eur_mwh": float(np.mean(magnitude[retained])) if np.any(retained) else float("nan"),
        "retained_p95_residual_eur_mwh": float(np.quantile(magnitude[retained], 0.95)) if np.any(retained) else float("nan"),
    }


def _refinement_decision_impact(
    target: np.ndarray,
    magnitude: np.ndarray,
    original_probability: np.ndarray,
    refined_probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Describe action changes caused only by an equivalent presentation."""

    target_values = np.asarray(target, dtype=np.float64)
    magnitude_values = np.asarray(magnitude, dtype=np.float64)
    original_values = np.asarray(original_probability, dtype=np.float64)
    refined_values = np.asarray(refined_probability, dtype=np.float64)
    if not (
        target_values.shape
        == magnitude_values.shape
        == original_values.shape
        == refined_values.shape
    ) or target_values.ndim != 1:
        raise ValueError("refinement decision inputs must be aligned one-dimensional arrays")
    if target_values.size == 0 or not all(
        np.all(np.isfinite(values))
        for values in (target_values, magnitude_values, original_values, refined_values)
    ):
        raise ValueError("refinement decision inputs must be non-empty and finite")

    original_flagged = original_values > float(threshold)
    refined_flagged = refined_values > float(threshold)
    flipped = original_flagged != refined_flagged
    accept_to_escalate = (~original_flagged) & refined_flagged
    escalate_to_accept = original_flagged & (~refined_flagged)
    tail = target_values > 0.5
    non_tail = ~tail

    def _conditional_rate(mask: np.ndarray, condition: np.ndarray) -> float:
        denominator = int(np.sum(condition))
        return float(np.sum(mask & condition) / max(denominator, 1))

    def _precision(flagged: np.ndarray) -> float:
        denominator = int(np.sum(flagged))
        return float(np.sum(flagged & tail) / max(denominator, 1))

    return {
        "rows": int(target_values.size),
        "original_flag_rate": float(np.mean(original_flagged)),
        "refined_flag_rate": float(np.mean(refined_flagged)),
        "decision_flip_count": int(np.sum(flipped)),
        "decision_flip_rate": float(np.mean(flipped)),
        "accept_to_escalate_count": int(np.sum(accept_to_escalate)),
        "escalate_to_accept_count": int(np.sum(escalate_to_accept)),
        "tail_flip_count": int(np.sum(flipped & tail)),
        "tail_flip_rate": _conditional_rate(flipped, tail),
        "non_tail_flip_count": int(np.sum(flipped & non_tail)),
        "non_tail_flip_rate": _conditional_rate(flipped, non_tail),
        "tail_accept_to_escalate_count": int(np.sum(accept_to_escalate & tail)),
        "tail_escalate_to_accept_count": int(np.sum(escalate_to_accept & tail)),
        "original_tail_recall": _conditional_rate(original_flagged, tail),
        "refined_tail_recall": _conditional_rate(refined_flagged, tail),
        "original_tail_precision": _precision(original_flagged),
        "refined_tail_precision": _precision(refined_flagged),
        "original_accepted_tail_rate": float(np.mean(target_values[~original_flagged]))
        if np.any(~original_flagged)
        else float("nan"),
        "refined_accepted_tail_rate": float(np.mean(target_values[~refined_flagged]))
        if np.any(~refined_flagged)
        else float("nan"),
        "mean_residual_on_flips_eur_mwh": float(np.mean(magnitude_values[flipped]))
        if np.any(flipped)
        else 0.0,
        "p95_residual_on_flips_eur_mwh": float(np.quantile(magnitude_values[flipped], 0.95))
        if np.any(flipped)
        else 0.0,
    }


def _absolute_error_summary(error_eur_mwh: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(error_eur_mwh, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("error_eur_mwh must be a non-empty finite vector")
    return {
        "rows": int(values.size),
        "mean_pairwise_mae_eur_mwh": float(np.mean(values)),
        "median_timestamp_pairwise_mae_eur_mwh": float(np.median(values)),
        "p95_timestamp_pairwise_mae_eur_mwh": float(np.quantile(values, 0.95)),
    }


def _tangent_error_per_timestamp(
    target_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    target_scale: float,
) -> np.ndarray:
    target = np.asarray(target_scaled, dtype=np.float64)
    prediction = np.asarray(prediction_scaled, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target_scaled and prediction_scaled must be aligned 2-D arrays")
    return np.mean(np.abs(prediction - target), axis=1) * float(target_scale)


def _rename_difference_key(result: dict[str, Any], new_key: str) -> dict[str, Any]:
    renamed = dict(result)
    renamed[new_key] = renamed.pop("candidate_minus_baseline_brier")
    return renamed


def _circular_block_bootstrap(
    loss_difference: np.ndarray,
    *,
    block_rows: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(loss_difference, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("loss_difference must be a non-empty vector")
    block_rows = max(1, min(int(block_rows), values.size))
    block_count = int(math.ceil(values.size / block_rows))
    remainder = int(values.size - (block_count - 1) * block_rows)
    extended = np.concatenate([values, values[: block_rows - 1]])
    prefix = np.concatenate([[0.0], np.cumsum(extended)])
    full_sum = prefix[block_rows:] - prefix[:-block_rows]
    if remainder == block_rows:
        remainder_sum = full_sum
    else:
        remainder_prefix = np.concatenate(
            [[0.0], np.cumsum(np.concatenate([values, values[: remainder - 1]]))]
        )
        remainder_sum = remainder_prefix[remainder:] - remainder_prefix[:-remainder]
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        starts = rng.integers(0, values.size, size=block_count)
        total = float(np.sum(full_sum[starts[:-1]])) if block_count > 1 else 0.0
        total += float(remainder_sum[starts[-1]])
        draws[draw] = total / values.size
    zero_tolerance = float(
        32.0 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(values)))
    )
    return {
        "candidate_minus_baseline_brier": float(np.mean(values)),
        "block_rows": block_rows,
        "block_days": float(block_rows / ROWS_PER_DAY),
        "replicates": int(replicates),
        "block_bootstrap_95ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_of_improvement": float(np.mean(draws < -zero_tolerance)),
        "improvement_zero_tolerance": zero_tolerance,
    }


def _calendar_grid_block_bootstrap(
    loss_difference: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    block_days: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap fixed calendar-time blocks while retaining feed gaps."""

    values = np.asarray(loss_difference, dtype=np.float64)
    index = pd.DatetimeIndex(timestamps)
    if values.size != index.size or values.size == 0:
        raise ValueError("timestamps and loss_difference must be non-empty and aligned")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("confirmation timestamps must be unique and increasing")
    if index.tz is None:
        raise ValueError("confirmation timestamps must be timezone-aware")
    start = index[0].floor("15min")
    elapsed = np.asarray(
        (index - start) / pd.Timedelta(minutes=15), dtype=np.float64
    )
    positions = np.asarray(np.rint(elapsed), dtype=np.int64)
    if np.max(np.abs(elapsed - positions), initial=0.0) > 1.0e-8:
        raise ValueError("confirmation timestamps are not aligned to a 15-minute grid")

    grid_length = int(positions[-1] + 1)
    grid_values = np.zeros(grid_length, dtype=np.float64)
    grid_valid = np.zeros(grid_length, dtype=np.float64)
    grid_values[positions] = values
    grid_valid[positions] = 1.0
    block_slots = max(1, min(int(block_days * ROWS_PER_DAY), grid_length))

    # Precompute every circular calendar block's observed sum/count so bootstrap
    # draws are fast even when the publication feed has missing quarter-hours.
    extended_values = np.concatenate([grid_values, grid_values[: block_slots - 1]])
    extended_valid = np.concatenate([grid_valid, grid_valid[: block_slots - 1]])
    value_prefix = np.concatenate([[0.0], np.cumsum(extended_values)])
    valid_prefix = np.concatenate([[0.0], np.cumsum(extended_valid)])
    block_sum = value_prefix[block_slots:] - value_prefix[:-block_slots]
    block_count = valid_prefix[block_slots:] - valid_prefix[:-block_slots]

    sampled_block_count = int(math.ceil(grid_length / block_slots))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        starts = rng.integers(0, grid_length, size=sampled_block_count)
        denominator = float(np.sum(block_count[starts]))
        if denominator <= 0.0:
            raise RuntimeError("calendar bootstrap sampled no observed rows")
        draws[draw] = float(np.sum(block_sum[starts]) / denominator)
    zero_tolerance = float(
        32.0 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(values)))
    )
    return {
        "candidate_minus_baseline_brier": float(np.mean(values)),
        "block_calendar_days": int(block_days),
        "block_grid_slots": block_slots,
        "calendar_grid_slots": grid_length,
        "observed_rows": int(values.size),
        "grid_coverage": float(values.size / grid_length),
        "replicates": int(replicates),
        "block_bootstrap_95ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_of_improvement": float(np.mean(draws < -zero_tolerance)),
        "improvement_zero_tolerance": zero_tolerance,
    }


def _serial_dependence(target: np.ndarray) -> dict[str, Any]:
    values = np.asarray(target, dtype=np.float64)
    result: dict[str, Any] = {}
    for label, lag in (("15min", 1), ("1h", 4), ("1d", 96), ("7d", 672)):
        if values.size <= lag or np.std(values[:-lag]) == 0.0 or np.std(values[lag:]) == 0.0:
            result[label] = None
        else:
            result[label] = float(np.corrcoef(values[:-lag], values[lag:])[0, 1])
    runs: list[int] = []
    current = 0
    for value in values > 0.5:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    result["tail_run_count"] = len(runs)
    result["tail_run_mean_rows"] = float(np.mean(runs)) if runs else 0.0
    result["tail_run_p95_rows"] = float(np.quantile(runs, 0.95)) if runs else 0.0
    result["tail_run_max_rows"] = int(max(runs, default=0))
    return result


def _reconstruct(report: dict[str, Any]) -> dict[str, Any]:
    protocol = report["protocol"]
    sources = report["sources"]
    smoke = bool(protocol.get("smoke", False))

    dual_path = _resolve_source(sources["dual_archive"], frozen.qct.DEFAULT_DUAL)
    price_root = _resolve_source(sources["price_root"], frozen.qct.DEFAULT_PRICE_ROOT)
    geometry_path = _resolve_source(sources["geometry_audit"], frozen.DEFAULT_GEOMETRY_AUDIT)
    development_path = _resolve_source(sources["development_artifact"], frozen.DEVELOPMENT_ARTIFACT)
    confirmation_reference = _resolve_source(sources["confirmation_reference"], frozen.CONFIRMATION_REPORT)
    _verify_hash(dual_path, sources.get("dual_archive_sha256"), "dual archive")
    _verify_hash(geometry_path, sources.get("geometry_audit_sha256"), "geometry audit")
    _verify_hash(development_path, sources.get("development_artifact_sha256"), "development artifact")
    _verify_hash(confirmation_reference, sources.get("confirmation_reference_sha256"), "AF-CDMO confirmation reference")

    geometry_audit = json.loads(geometry_path.read_text(encoding="utf-8"))
    frame, certificate_audit = frozen.qct._read_certificate(dual_path, 60.0)
    unit_columns = certificate_audit["unit_columns"]
    input_zones = tuple(column.removeprefix("unit_ptdf_") for column in unit_columns)
    geometry = RealAffineGeometry.loss_safe(input_zones)
    if list(input_zones) != geometry_audit["geometry"]["zone_order"]:
        raise RuntimeError("PTDF order differs from the frozen geometry audit")
    zones = frozen.qct.PRIMARY_ZONES
    real_indices = np.asarray([unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64)
    price_panel, _ = frozen.qct._load_price_panel(price_root, zones)
    confirmation_end = pd.Timestamp(protocol["confirmation_end_exclusive"])
    samples = frozen.qct._build_vector_samples(
        frame, price_panel, unit_columns, zones, frozen.qct.FIT_END, confirmation_end
    )
    fit_samples = [sample for sample in samples if sample["timestamp"] < frozen.qct.FIT_END]
    calibration_samples = [
        sample
        for sample in samples
        if frozen.qct.TUNING_END <= sample["timestamp"] < frozen.qct.CALIBRATION_END
    ]
    confirmation_samples = [
        sample
        for sample in samples
        if frozen.qct.CONFIRMATION_START <= sample["timestamp"] < confirmation_end
    ]
    if smoke:
        fit_samples = fit_samples[-2400:]
        calibration_samples = calibration_samples[-1200:]
        confirmation_samples = confirmation_samples[:800]
    expected_rows = report["rows"]
    if len(fit_samples) != expected_rows["fit"] or len(calibration_samples) != expected_rows["calibration"]:
        raise RuntimeError("Reconstructed fit/calibration row count differs from the frozen report")
    if len(confirmation_samples) != expected_rows["confirmation"]:
        raise RuntimeError("Reconstructed confirmation row count differs from the frozen report")

    target_values = np.stack([sample["target_vector"] for sample in fit_samples])
    target_scale = max(float(np.median(np.abs(target_values))), 1.0)
    canonical_ram_scale = _canonical_ram_scale(frame, frozen.qct.FIT_END)
    raw_scale = _positive_scales(
        raw_rows_for_scale(
            fit_samples,
            representation="af",
            real_zone_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
        )
    )
    threshold = float(protocol["tail_thresholds_eur_mwh"][1])

    def make(rows: list[dict[str, Any]]) -> frozen.RiskDataset:
        return frozen._dataset_factory(
            rows,
            raw_scale=raw_scale,
            target_scale=target_scale,
            real_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
            threshold=threshold,
        )

    fit = make(fit_samples)
    calibration = make(calibration_samples)
    confirmation = make(confirmation_samples)
    return {
        "fit": fit,
        "calibration": calibration,
        "confirmation": confirmation,
        "confirmation_samples": confirmation_samples,
        "real_indices": real_indices,
        "raw_scale": raw_scale,
        "target_scale": target_scale,
        "canonical_ram_scale": canonical_ram_scale,
        "geometry": geometry,
        "tail_threshold": threshold,
        "atom_standardization": frozen.qct._atom_standardization(fit),
        "zone_count": len(zones),
        "canonical_dim": fit[0]["canonical"].shape[1],
    }


def _load_qct_ensemble(
    report: dict[str, Any],
    context: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
]:
    records = [record for record in report["checkpoints"] if record["method"] == "qct_full"]
    expected_seeds = [int(value) for value in report["protocol"]["seeds"]]
    if sorted(int(record["seed"]) for record in records) != sorted(expected_seeds):
        raise RuntimeError("The report does not contain exactly the registered qct_full checkpoints")
    atom_mean, atom_std = context["atom_standardization"]
    calibration_probabilities: list[np.ndarray] = []
    confirmation_probabilities: list[np.ndarray] = []
    confirmation_by_seed: dict[int, np.ndarray] = {}
    confirmation_tangent: list[np.ndarray] = []
    tangent_by_seed: dict[int, np.ndarray] = {}
    expected_tangent_target = frozen._pair_targets(context["confirmation"])
    for record in sorted(records, key=lambda value: expected_seeds.index(int(value["seed"]))):
        checkpoint_path = Path(record["path"]).expanduser().resolve()
        _verify_hash(checkpoint_path, record.get("sha256"), f"qct_full seed {record['seed']} checkpoint")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["method"] != "qct_full" or int(payload["seed"]) != int(record["seed"]):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
        model, _ = frozen._model_factory(
            "qct_full",
            context["canonical_dim"],
            context["zone_count"],
            atom_mean,
            atom_std,
        )
        model = model.to(device)
        model.load_state_dict(payload["state_dict"])
        platt = frozen.PlattMap(**payload["platt"])
        calibration_logits, _, _, _, _ = frozen._predict(
            model, context["calibration"], batch_size=batch_size, device=device
        )
        confirmation_logits, _, _, pair_prediction, pair_target = frozen._predict(
            model, context["confirmation"], batch_size=batch_size, device=device
        )
        if not np.allclose(pair_target, expected_tangent_target, atol=1.0e-6):
            raise RuntimeError("QCT tangent target differs from the reconstructed confirmation target")
        calibration_probabilities.append(platt.probability(calibration_logits))
        seed_probability = platt.probability(confirmation_logits)
        confirmation_probabilities.append(seed_probability)
        confirmation_by_seed[int(record["seed"])] = seed_probability
        confirmation_tangent.append(pair_prediction)
        tangent_by_seed[int(record["seed"])] = pair_prediction
    return (
        np.mean(np.stack(calibration_probabilities), axis=0),
        np.mean(np.stack(confirmation_probabilities), axis=0),
        confirmation_by_seed,
        np.mean(np.stack(confirmation_tangent), axis=0),
        tangent_by_seed,
    )


def _load_classifier_ensemble(
    report: dict[str, Any],
    context: dict[str, Any],
    method: str,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    list[torch.nn.Module],
    list[frozen.PlattMap],
]:
    records = [record for record in report["checkpoints"] if record["method"] == method]
    expected_seeds = [int(value) for value in report["protocol"]["seeds"]]
    if sorted(int(record["seed"]) for record in records) != sorted(expected_seeds):
        raise RuntimeError(f"The report does not contain exactly the registered {method} checkpoints")
    atom_mean, atom_std = context["atom_standardization"]
    calibration_probabilities: list[np.ndarray] = []
    confirmation_probabilities: list[np.ndarray] = []
    confirmation_by_seed: dict[int, np.ndarray] = {}
    members: list[torch.nn.Module] = []
    calibrators: list[frozen.PlattMap] = []
    for record in sorted(records, key=lambda value: expected_seeds.index(int(value["seed"]))):
        checkpoint_path = Path(record["path"]).expanduser().resolve()
        _verify_hash(checkpoint_path, record.get("sha256"), f"{method} seed {record['seed']} checkpoint")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["method"] != method or int(payload["seed"]) != int(record["seed"]):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
        model, _ = frozen._model_factory(
            method,
            context["canonical_dim"],
            context["zone_count"],
            atom_mean,
            atom_std,
        )
        model = model.to(device)
        model.load_state_dict(payload["state_dict"])
        platt = frozen.PlattMap(**payload["platt"])
        calibration_logits, _, _, _, _ = frozen._predict(
            model, context["calibration"], batch_size=batch_size, device=device
        )
        confirmation_logits, _, _, _, _ = frozen._predict(
            model, context["confirmation"], batch_size=batch_size, device=device
        )
        calibration_probabilities.append(platt.probability(calibration_logits))
        seed_probability = platt.probability(confirmation_logits)
        confirmation_probabilities.append(seed_probability)
        confirmation_by_seed[int(record["seed"])] = seed_probability
        members.append(model)
        calibrators.append(platt)
    return (
        np.mean(np.stack(calibration_probabilities), axis=0),
        np.mean(np.stack(confirmation_probabilities), axis=0),
        confirmation_by_seed,
        members,
        calibrators,
    )


def main() -> int:
    args = _parse_args()
    report_path = args.report.expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Confirmation report does not exist yet: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "confirmation_complete":
        raise RuntimeError("QCT-Risk confirmation report is incomplete")
    if "qct_full" not in report["protocol"].get("methods", []):
        raise RuntimeError("Confirmation report does not include qct_full")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    context = _reconstruct(report)
    fit = context["fit"]
    calibration = context["calibration"]
    confirmation = context["confirmation"]
    target = frozen._targets(confirmation)
    magnitude = frozen._magnitudes(confirmation)
    calibration_target = frozen._targets(calibration)
    fit_target = frozen._targets(fit)

    quotient_features = {
        "fit": frozen._fixed_quotient_features(fit, context["real_indices"]),
        "calibration": frozen._fixed_quotient_features(calibration, context["real_indices"]),
        "confirmation": frozen._fixed_quotient_features(confirmation, context["real_indices"]),
    }
    calendar_features = {
        "fit": frozen._calendar_features(fit),
        "calibration": frozen._calendar_features(calibration),
        "confirmation": frozen._calendar_features(confirmation),
    }
    hgb_quotient_calibration, hgb_quotient, _ = frozen._calibrated_hgb_probability(
        quotient_features["fit"],
        fit_target,
        quotient_features["calibration"],
        calibration_target,
        quotient_features["confirmation"],
        random_state=frozen.BOOTSTRAP_SEED,
    )
    hgb_calendar_calibration, hgb_calendar, _ = frozen._calibrated_hgb_probability(
        calendar_features["fit"],
        fit_target,
        calendar_features["calibration"],
        calibration_target,
        calendar_features["confirmation"],
        random_state=frozen.BOOTSTRAP_SEED,
    )
    unconditional = np.full(target.size, float(np.mean(fit_target)), dtype=np.float64)
    (
        qct_calibration,
        qct_probability,
        qct_probability_by_seed,
        qct_tangent,
        qct_tangent_by_seed,
    ) = _load_qct_ensemble(report, context, device=device, batch_size=args.batch_size)
    (
        qct_no_tangent_calibration,
        qct_no_tangent_probability,
        qct_no_tangent_by_seed,
        qct_no_tangent_members,
        qct_no_tangent_calibrators,
    ) = _load_classifier_ensemble(
        report,
        context,
        "qct_no_tangent",
        device=device,
        batch_size=args.batch_size,
    )
    (
        uniform_transformer_calibration,
        uniform_transformer_probability,
        _,
        uniform_transformer_members,
        uniform_transformer_calibrators,
    ) = _load_classifier_ensemble(
        report,
        context,
        "uniform_set_transformer",
        device=device,
        batch_size=args.batch_size,
    )
    reproduced_brier = float(brier_score_loss(target, qct_probability))
    reported_brier = float(report["methods"]["qct_full"]["classification"]["brier"])
    if abs(reproduced_brier - reported_brier) > 1.0e-10:
        raise RuntimeError(
            f"QCT confirmation Brier did not reproduce: {reproduced_brier} versus {reported_brier}"
        )
    for method, probability in (
        ("qct_no_tangent", qct_no_tangent_probability),
        ("uniform_set_transformer", uniform_transformer_probability),
    ):
        reproduced = float(brier_score_loss(target, probability))
        reported = float(report["methods"][method]["classification"]["brier"])
        if abs(reproduced - reported) > 1.0e-10:
            raise RuntimeError(
                f"{method} confirmation Brier did not reproduce: {reproduced} versus {reported}"
            )

    pair_target = frozen._pair_targets(confirmation)
    fit_residual = np.stack(
        [fit[index]["residual_scaled"] for index in range(len(fit))]
    )
    hgb_residual_columns = []
    for zone_index in range(context["zone_count"]):
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=3.0,
            random_state=frozen.BOOTSTRAP_SEED + zone_index,
        ).fit(quotient_features["fit"], fit_residual[:, zone_index])
        hgb_residual_columns.append(
            regressor.predict(quotient_features["confirmation"])
        )
    hgb_residual = np.column_stack(hgb_residual_columns)
    hgb_residual -= hgb_residual.mean(axis=1, keepdims=True)
    hgb_tangent = frozen.qct._numpy_pairwise(hgb_residual)

    tangent_errors = {
        "analytic_kkt_only": _tangent_error_per_timestamp(
            pair_target, np.zeros_like(pair_target), context["target_scale"]
        ),
        "hgb_quotient": _tangent_error_per_timestamp(
            pair_target, hgb_tangent, context["target_scale"]
        ),
        "qct_full": _tangent_error_per_timestamp(
            pair_target, qct_tangent, context["target_scale"]
        ),
    }
    reproduced_tangent_mae = float(np.mean(tangent_errors["qct_full"]))
    reported_tangent_mae = float(
        report["methods"]["qct_full"]["tangent_residual"][
            "mean_pairwise_mae_eur_mwh"
        ]
    )
    if abs(reproduced_tangent_mae - reported_tangent_mae) > 1.0e-8:
        raise RuntimeError(
            "QCT tangent MAE did not reproduce: "
            f"{reproduced_tangent_mae} versus {reported_tangent_mae}"
        )
    reported_hgb_tangent_mae = float(
        report["methods"]["hgb_quotient"]["tangent_residual"][
            "mean_pairwise_mae_eur_mwh"
        ]
    )
    if abs(float(np.mean(tangent_errors["hgb_quotient"])) - reported_hgb_tangent_mae) > 1.0e-8:
        raise RuntimeError("Invariant HGB tangent MAE did not reproduce")
    analytic_tangent_mae = float(np.mean(tangent_errors["analytic_kkt_only"]))
    reported_analytic_tangent_mae = float(
        report["confirmation_target"]["mean_residual_magnitude_eur_mwh"]
    )
    if abs(analytic_tangent_mae - reported_analytic_tangent_mae) > 1.0e-8:
        raise RuntimeError("Analytic KKT-only tangent MAE did not reproduce")

    timestamps = pd.DatetimeIndex(
        [pd.Timestamp(sample["timestamp"]) for sample in context["confirmation_samples"]]
    )
    calibration_methods = {
        "unconditional": np.full(
            calibration_target.size, float(np.mean(fit_target)), dtype=np.float64
        ),
        "hgb_calendar": hgb_calendar_calibration,
        "hgb_quotient": hgb_quotient_calibration,
        "qct_full": qct_calibration,
    }
    decision_thresholds = {
        name: float(np.quantile(probability, 0.80))
        for name, probability in calibration_methods.items()
    }
    decision_threshold = decision_thresholds["qct_full"]
    methods = {
        "unconditional": unconditional,
        "hgb_calendar": hgb_calendar,
        "hgb_quotient": hgb_quotient,
        "qct_full": qct_probability,
    }
    overall = {
        name: {
            **_safe_classification(target, probability),
            "decision": _decision_metrics(
                target,
                probability,
                magnitude,
                decision_thresholds[name],
            ),
        }
        for name, probability in methods.items()
    }

    seedwise_qct: dict[str, Any] = {}
    baseline_offsets = {
        "unconditional": 101,
        "hgb_calendar": 211,
        "hgb_quotient": 307,
        "uniform_set_transformer": 401,
        "qct_full": 503,
    }
    for seed, probability in sorted(qct_probability_by_seed.items()):
        record: dict[str, Any] = {
            "classification": _safe_classification(target, probability),
            "paired_brier_7d": {},
        }
        qct_seed_loss = np.square(target - probability)
        for baseline_name, baseline_probability in methods.items():
            if baseline_name == "qct_full":
                continue
            difference = qct_seed_loss - np.square(target - baseline_probability)
            record["paired_brier_7d"][baseline_name] = {
                "row_block": _circular_block_bootstrap(
                    difference,
                    block_rows=7 * ROWS_PER_DAY,
                    replicates=args.replicates,
                    seed=args.seed + 100_003 * seed + baseline_offsets[baseline_name],
                ),
                "calendar_grid_block": _calendar_grid_block_bootstrap(
                    difference,
                    timestamps,
                    block_days=7,
                    replicates=args.replicates,
                    seed=args.seed + 200_003 * seed + baseline_offsets[baseline_name],
                ),
            }
        seedwise_qct[str(seed)] = record

    tangent_overall = {
        name: _absolute_error_summary(values)
        for name, values in tangent_errors.items()
    }
    tangent_seedwise: dict[str, Any] = {}
    for seed, pair_prediction in sorted(qct_tangent_by_seed.items()):
        seed_error = _tangent_error_per_timestamp(
            pair_target, pair_prediction, context["target_scale"]
        )
        record = {"metrics": _absolute_error_summary(seed_error), "paired_mae_7d": {}}
        for baseline_name in ("analytic_kkt_only", "hgb_quotient"):
            difference = seed_error - tangent_errors[baseline_name]
            record["paired_mae_7d"][baseline_name] = {
                "row_block": _rename_difference_key(
                    _circular_block_bootstrap(
                        difference,
                        block_rows=7 * ROWS_PER_DAY,
                        replicates=args.replicates,
                        seed=args.seed + 300_007 * seed + baseline_offsets.get(baseline_name, 401),
                    ),
                    "candidate_minus_baseline_mae_eur_mwh",
                ),
                "calendar_grid_block": _rename_difference_key(
                    _calendar_grid_block_bootstrap(
                        difference,
                        timestamps,
                        block_days=7,
                        replicates=args.replicates,
                        seed=args.seed + 400_009 * seed + baseline_offsets.get(baseline_name, 503),
                    ),
                    "candidate_minus_baseline_mae_eur_mwh",
                ),
            }
        tangent_seedwise[str(seed)] = record

    secondary_classifier_probabilities = {
        "hgb_quotient": hgb_quotient,
        "uniform_set_transformer": uniform_transformer_probability,
        "qct_full": qct_probability,
        "qct_no_tangent": qct_no_tangent_probability,
    }
    secondary_classifier_calibration = {
        "hgb_quotient": hgb_quotient_calibration,
        "uniform_set_transformer": uniform_transformer_calibration,
        "qct_full": qct_calibration,
        "qct_no_tangent": qct_no_tangent_calibration,
    }
    secondary_thresholds = {
        name: float(np.quantile(probability, 0.80))
        for name, probability in secondary_classifier_calibration.items()
    }
    original_base = frozen.RealAffineGaugeDataset(
        context["confirmation_samples"],
        raw_scale=context["raw_scale"],
        target_scale=context["target_scale"],
        presentation="original",
        representation="af",
        real_zone_indices=context["real_indices"],
        geometry=context["geometry"],
        canonical_ram_scale=context["canonical_ram_scale"],
    )
    refined = frozen.RiskDataset(
        frozen.NonuniformRefinementAFDataset(original_base),
        target_scale=context["target_scale"],
        threshold=context["tail_threshold"],
    )
    refinement_records: dict[str, Any] = {}
    refined_probabilities: dict[str, np.ndarray] = {}
    for method, members, calibrators, original_probability in (
        (
            "qct_no_tangent",
            qct_no_tangent_members,
            qct_no_tangent_calibrators,
            qct_no_tangent_probability,
        ),
        (
            "uniform_set_transformer",
            uniform_transformer_members,
            uniform_transformer_calibrators,
            uniform_transformer_probability,
        ),
    ):
        refined_probability, _ = frozen._ensemble_predict(
            members,
            calibrators,
            refined,
            batch_size=args.batch_size,
            device=device,
        )
        refined_probabilities[method] = refined_probability
        probability_drift = np.abs(refined_probability - original_probability)
        refinement_records[method] = {
            "max_abs_probability_drift": float(np.max(probability_drift, initial=0.0)),
            "mean_abs_probability_drift": float(np.mean(probability_drift)),
            **_refinement_decision_impact(
                target,
                magnitude,
                original_probability,
                refined_probability,
                secondary_thresholds[method],
            ),
        }
    if (
        refinement_records["qct_no_tangent"]["max_abs_probability_drift"] > 1.0e-4
        or refinement_records["qct_no_tangent"]["decision_flip_rate"] > 0.0
    ):
        raise RuntimeError("qct_no_tangent failed its exact refinement-invariance contract")
    secondary_classifier_overall = {
        name: {
            **_safe_classification(target, probability),
            "decision": _decision_metrics(
                target,
                probability,
                magnitude,
                secondary_thresholds[name],
            ),
        }
        for name, probability in secondary_classifier_probabilities.items()
    }
    qct_no_tangent_loss = np.square(target - qct_no_tangent_probability)
    secondary_classifier_sensitivity: dict[str, Any] = {}
    for baseline_name in ("hgb_quotient", "uniform_set_transformer", "qct_full"):
        difference = qct_no_tangent_loss - np.square(
            target - secondary_classifier_probabilities[baseline_name]
        )
        secondary_classifier_sensitivity[baseline_name] = {}
        for days in args.block_days:
            secondary_classifier_sensitivity[baseline_name][f"{days}d"] = {
                "row_block": _circular_block_bootstrap(
                    difference,
                    block_rows=days * ROWS_PER_DAY,
                    replicates=args.replicates,
                    seed=args.seed + 9001 * days + baseline_offsets[baseline_name],
                ),
                "calendar_grid_block": _calendar_grid_block_bootstrap(
                    difference,
                    timestamps,
                    block_days=days,
                    replicates=args.replicates,
                    seed=args.seed + 11003 * days + baseline_offsets[baseline_name],
                ),
            }
    secondary_seedwise: dict[str, Any] = {}
    for seed, probability in sorted(qct_no_tangent_by_seed.items()):
        seed_loss = np.square(target - probability)
        record = {
            "classification": _safe_classification(target, probability),
            "paired_brier_7d": {},
        }
        for baseline_name in ("hgb_quotient", "uniform_set_transformer"):
            difference = seed_loss - np.square(
                target - secondary_classifier_probabilities[baseline_name]
            )
            record["paired_brier_7d"][baseline_name] = {
                "row_block": _circular_block_bootstrap(
                    difference,
                    block_rows=7 * ROWS_PER_DAY,
                    replicates=args.replicates,
                    seed=args.seed + 500_009 * seed + baseline_offsets[baseline_name],
                ),
                "calendar_grid_block": _calendar_grid_block_bootstrap(
                    difference,
                    timestamps,
                    block_days=7,
                    replicates=args.replicates,
                    seed=args.seed + 600_011 * seed + baseline_offsets[baseline_name],
                ),
            }
        secondary_seedwise[str(seed)] = record

    monthwise: dict[str, Any] = {}
    month_keys = timestamps.strftime("%Y-%m")
    for month in sorted(set(month_keys)):
        mask = np.asarray(month_keys == month)
        monthwise[month] = {
            "target": {
                "rows": int(np.sum(mask)),
                "prevalence": float(np.mean(target[mask])),
                "mean_residual_eur_mwh": float(np.mean(magnitude[mask])),
                "p95_residual_eur_mwh": float(np.quantile(magnitude[mask], 0.95)),
            },
            "methods": {
                name: _safe_classification(target[mask], probability[mask])
                for name, probability in methods.items()
            },
            "qct_decision": _decision_metrics(
                target[mask], qct_probability[mask], magnitude[mask], decision_threshold
            ),
            "tangent_reconstruction": {
                name: _absolute_error_summary(values[mask])
                for name, values in tangent_errors.items()
            },
        }

    qct_loss = np.square(target - qct_probability)
    temporal_sensitivity: dict[str, Any] = {}
    for baseline_name, baseline_probability in methods.items():
        if baseline_name == "qct_full":
            continue
        baseline_loss = np.square(target - baseline_probability)
        difference = qct_loss - baseline_loss
        temporal_sensitivity[baseline_name] = {}
        for days in args.block_days:
            temporal_sensitivity[baseline_name][f"{days}d"] = {
                "row_block": _circular_block_bootstrap(
                    difference,
                    block_rows=days * ROWS_PER_DAY,
                    replicates=args.replicates,
                    seed=args.seed + 1009 * days,
                ),
                "calendar_grid_block": _calendar_grid_block_bootstrap(
                    difference,
                    timestamps,
                    block_days=days,
                    replicates=args.replicates,
                    seed=args.seed + 2027 * days,
                ),
            }

    tangent_temporal_sensitivity: dict[str, Any] = {}
    for baseline_name in ("analytic_kkt_only", "hgb_quotient"):
        difference = tangent_errors["qct_full"] - tangent_errors[baseline_name]
        tangent_temporal_sensitivity[baseline_name] = {}
        for days in args.block_days:
            tangent_temporal_sensitivity[baseline_name][f"{days}d"] = {
                "row_block": _rename_difference_key(
                    _circular_block_bootstrap(
                        difference,
                        block_rows=days * ROWS_PER_DAY,
                        replicates=args.replicates,
                        seed=args.seed + 5003 * days,
                    ),
                    "candidate_minus_baseline_mae_eur_mwh",
                ),
                "calendar_grid_block": _rename_difference_key(
                    _calendar_grid_block_bootstrap(
                        difference,
                        timestamps,
                        block_days=days,
                        replicates=args.replicates,
                        seed=args.seed + 7001 * days,
                    ),
                    "candidate_minus_baseline_mae_eur_mwh",
                ),
            }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps.astype(str),
            "tail_target": target.astype(np.int8),
            "residual_magnitude_eur_mwh": magnitude,
            **{f"probability_{name}": value for name, value in methods.items()},
            "probability_uniform_set_transformer": uniform_transformer_probability,
            "probability_qct_no_tangent": qct_no_tangent_probability,
            "probability_refined_uniform_set_transformer": refined_probabilities[
                "uniform_set_transformer"
            ],
            "probability_refined_qct_no_tangent": refined_probabilities[
                "qct_no_tangent"
            ],
            **{f"tangent_mae_{name}_eur_mwh": value for name, value in tangent_errors.items()},
        }
    )
    prediction_path = output_dir / "confirmation_predictions_internal.csv.gz"
    prediction_frame.to_csv(prediction_path, index=False, compression="gzip")

    audit = {
        "status": "post_confirmation_temporal_robustness_complete",
        "interpretation": {
            "registered_primary_result_remains_authoritative": True,
            "post_confirmation_analysis_is_descriptive": True,
            "no_model_selection_or_recalibration": True,
            "internal_predictions_not_for_redistribution_without_data_permission": True,
        },
        "provenance": {
            "confirmation_report": str(report_path),
            "confirmation_report_sha256": _sha256(report_path),
            "analysis_engine": str(Path(__file__).resolve()),
            "analysis_engine_sha256": _sha256(Path(__file__).resolve()),
            "prediction_archive": str(prediction_path),
            "prediction_archive_sha256": _sha256(prediction_path),
        },
        "rows": int(target.size),
        "decision_thresholds_from_december": decision_thresholds,
        "overall": overall,
        "seedwise_qct": seedwise_qct,
        "tangent_reconstruction": {
            "interpretation": "post-confirmation certificate-only centered price-spread reconstruction audit",
            "overall": tangent_overall,
            "seedwise_qct": tangent_seedwise,
            "paired_mae_block_length_sensitivity": tangent_temporal_sensitivity,
        },
        "registered_ablation_secondary_classifier": {
            "interpretation": (
                "post-confirmation temporal audit of the preregistered exact-invariant "
                "qct_no_tangent classifier; this does not replace the qct_full primary gate"
            ),
            "overall": secondary_classifier_overall,
            "seedwise_qct_no_tangent": secondary_seedwise,
            "paired_brier_block_length_sensitivity": secondary_classifier_sensitivity,
            "held_out_nonuniform_mass_refinement": {
                "rows": int(len(refined)),
                "training_exposure": False,
                "methods": refinement_records,
            },
            "claim_boundary": {
                "registered_primary_result_unchanged": True,
                "secondary_classifier_was_preregistered": True,
                "post_confirmation_temporal_intervals_are_descriptive": True,
            },
        },
        "monthwise": monthwise,
        "paired_brier_block_length_sensitivity": temporal_sensitivity,
        "tail_serial_dependence": _serial_dependence(target),
    }
    audit_path = output_dir / "temporal_robustness_audit.json"
    _atomic_json(audit_path, audit)

    rows: list[dict[str, Any]] = []
    for month, values in monthwise.items():
        for method, metrics in values["methods"].items():
            rows.append({"month": month, "method": method, **metrics})
    pd.DataFrame(rows).to_csv(output_dir / "monthwise_metrics.csv", index=False)

    seed_rows: list[dict[str, Any]] = []
    for seed, values in seedwise_qct.items():
        classification = values["classification"]
        for baseline, estimators in values["paired_brier_7d"].items():
            for estimator, paired_values in estimators.items():
                low, high = paired_values["block_bootstrap_95ci"]
                seed_rows.append(
                    {
                        "seed": int(seed),
                        "qct_brier": classification["brier"],
                        "baseline": baseline,
                        "estimator": estimator,
                        "qct_minus_baseline_brier": paired_values[
                            "candidate_minus_baseline_brier"
                        ],
                        "ci95_low": low,
                        "ci95_high": high,
                        "probability_of_improvement": paired_values[
                            "probability_of_improvement"
                        ],
                    }
                )
    pd.DataFrame(seed_rows).to_csv(output_dir / "seedwise_qct_7d.csv", index=False)

    tangent_month_rows: list[dict[str, Any]] = []
    for month, values in monthwise.items():
        for method, metrics in values["tangent_reconstruction"].items():
            tangent_month_rows.append({"month": month, "method": method, **metrics})
    pd.DataFrame(tangent_month_rows).to_csv(
        output_dir / "monthwise_tangent_reconstruction.csv", index=False
    )

    tangent_seed_rows: list[dict[str, Any]] = []
    for seed, values in tangent_seedwise.items():
        qct_mae = values["metrics"]["mean_pairwise_mae_eur_mwh"]
        for baseline, estimators in values["paired_mae_7d"].items():
            for estimator, paired_values in estimators.items():
                low, high = paired_values["block_bootstrap_95ci"]
                tangent_seed_rows.append(
                    {
                        "seed": int(seed),
                        "qct_pairwise_mae_eur_mwh": qct_mae,
                        "baseline": baseline,
                        "estimator": estimator,
                        "qct_minus_baseline_mae_eur_mwh": paired_values[
                            "candidate_minus_baseline_mae_eur_mwh"
                        ],
                        "ci95_low": low,
                        "ci95_high": high,
                        "probability_of_improvement": paired_values[
                            "probability_of_improvement"
                        ],
                    }
                )
    pd.DataFrame(tangent_seed_rows).to_csv(
        output_dir / "seedwise_qct_tangent_7d.csv", index=False
    )

    summary_lines = [
        "# QCT-Risk v5 temporal robustness audit",
        "",
        "This is a post-confirmation descriptive robustness audit. It does not replace the registered seven-day primary test.",
        "",
        f"- Confirmation rows: {target.size:,}",
        f"- QCT Brier reproduced: {reproduced_brier:.6f}",
        f"- December-fixed accept/escalate threshold: {decision_threshold:.6f}",
        "",
        "## Overall Brier",
        "",
    ]
    for name, metrics in overall.items():
        summary_lines.append(f"- `{name}`: {metrics['brier']:.6f}")
    summary_lines.extend(["", "## QCT seedwise stability", ""])
    for seed, values in seedwise_qct.items():
        summary_lines.append(
            f"- seed `{seed}`: Brier={values['classification']['brier']:.6f}"
        )
    summary_lines.extend(["", "## Certificate-only tangent reconstruction", ""])
    for name, metrics in tangent_overall.items():
        summary_lines.append(
            f"- `{name}`: pairwise MAE={metrics['mean_pairwise_mae_eur_mwh']:.6f} EUR/MWh"
        )
    summary_lines.extend(["", "### Paired QCT minus tangent baseline MAE sensitivity", ""])
    for baseline, block_results in tangent_temporal_sensitivity.items():
        summary_lines.append(f"#### {baseline}")
        for label, estimators in block_results.items():
            for estimator_name, values in estimators.items():
                lo, hi = values["block_bootstrap_95ci"]
                summary_lines.append(
                    f"- {label} `{estimator_name}`: delta={values['candidate_minus_baseline_mae_eur_mwh']:.6f}, "
                    f"95% CI [{lo:.6f}, {hi:.6f}]"
                )
    summary_lines.extend(
        [
            "",
            "## Registered exact-invariant classifier ablation",
            "",
            "`qct_no_tangent` was a registered ablation; this secondary temporal audit does not replace the `qct_full` primary gate.",
            "",
        ]
    )
    for name, metrics in secondary_classifier_overall.items():
        summary_lines.append(f"- `{name}`: Brier={metrics['brier']:.6f}")
    summary_lines.extend(["", "### Held-out nonuniform refinement", ""])
    for name, metrics in refinement_records.items():
        summary_lines.append(
            f"- `{name}`: max probability drift={metrics['max_abs_probability_drift']:.9f}, "
            f"decision flips={metrics['decision_flip_count']:,} "
            f"({metrics['decision_flip_rate']:.6f}), "
            f"tail flips={metrics['tail_flip_count']:,}, "
            f"tail recall {metrics['original_tail_recall']:.6f} -> "
            f"{metrics['refined_tail_recall']:.6f}"
        )
    summary_lines.extend(["", "### Paired qct_no_tangent minus classifier baseline Brier", ""])
    for baseline, block_results in secondary_classifier_sensitivity.items():
        summary_lines.append(f"#### {baseline}")
        for label, estimators in block_results.items():
            for estimator_name, values in estimators.items():
                lo, hi = values["block_bootstrap_95ci"]
                summary_lines.append(
                    f"- {label} `{estimator_name}`: delta={values['candidate_minus_baseline_brier']:.6f}, "
                    f"95% CI [{lo:.6f}, {hi:.6f}]"
                )
    summary_lines.extend(["", "## Paired QCT minus baseline Brier sensitivity", ""])
    for baseline, block_results in temporal_sensitivity.items():
        summary_lines.append(f"### {baseline}")
        for label, estimators in block_results.items():
            for estimator_name, values in estimators.items():
                lo, hi = values["block_bootstrap_95ci"]
                summary_lines.append(
                    f"- {label} `{estimator_name}`: delta={values['candidate_minus_baseline_brier']:.6f}, "
                    f"95% CI [{lo:.6f}, {hi:.6f}]"
                )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {audit_path}")
    print(f"[OK] wrote {prediction_path}")
    print(f"[OK] wrote {output_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
