"""Calendar-time block-bootstrap sensitivity for AF-CDMO real v14.

The frozen CQDM utility resamples fixed-length blocks of observed rows.  This
post-processing analysis keeps those primary results intact and adds a
timestamp-aware sensitivity for an incomplete, non-imputed price panel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.cqdm_utils import sha256_file
except ModuleNotFoundError:
    from cqdm_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "official_jao_dual_measure" / "af_cdmo_real_development_v14.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "official_jao_dual_measure"
    / "af_cdmo_real_development_v14_calendar_bootstrap.json"
)
BASELINES = (
    "ambient_cqdm_residual",
    "slack_qdm_residual",
    "af_projected_raw_residual",
    "af_uniform_attention_residual",
    "ambient_gauge_augmented_residual",
    "analytic_gauge",
)
CANDIDATE = "af_qdm_residual"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _calendar_blocks(
    timestamps_ns: np.ndarray, *, block_days: float
) -> tuple[list[np.ndarray], dict[str, Any]]:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("At least two one-dimensional timestamps are required")
    gaps = np.diff(timestamps)
    if np.any(gaps <= 0):
        raise ValueError("Timestamps must be strictly increasing and unique")
    if block_days <= 0.0:
        raise ValueError("block_days must be positive")

    minute_ns = 60 * 1_000_000_000
    block_ns = int(round(block_days * 24.0 * 60.0 * minute_ns))
    typical_gap = int(np.median(gaps))
    period = int(timestamps[-1] - timestamps[0] + typical_gap)
    blocks: list[np.ndarray] = []
    row_counts = np.empty(len(timestamps), dtype=np.int64)
    for start in range(len(timestamps)):
        order = np.concatenate(
            [
                np.arange(start, len(timestamps), dtype=np.int64),
                np.arange(0, start, dtype=np.int64),
            ]
        )
        circular_time = np.concatenate(
            [timestamps[start:], timestamps[:start] + period]
        )
        count = int(np.searchsorted(circular_time, timestamps[start] + block_ns, side="left"))
        count = max(1, min(count, len(timestamps)))
        block = order[:count]
        blocks.append(block)
        row_counts[start] = count

    gap_minutes = gaps.astype(np.float64) / minute_ns
    return blocks, {
        "rows": int(len(timestamps)),
        "first_utc_ns": int(timestamps[0]),
        "last_utc_ns": int(timestamps[-1]),
        "block_days": float(block_days),
        "block_duration_ns": block_ns,
        "typical_gap_minutes": float(typical_gap / minute_ns),
        "gap_minutes": {
            "min": float(gap_minutes.min()),
            "median": float(np.median(gap_minutes)),
            "p95": float(np.quantile(gap_minutes, 0.95)),
            "max": float(gap_minutes.max()),
            "share_equal_typical": float(np.mean(gaps == typical_gap)),
        },
        "calendar_block_observed_rows": {
            "min": int(row_counts.min()),
            "median": float(np.median(row_counts)),
            "p95": float(np.quantile(row_counts, 0.95)),
            "max": int(row_counts.max()),
        },
    }


def _bootstrap_indices(
    blocks: list[np.ndarray], *, rows: int, replicates: int, seed: int
) -> np.ndarray:
    if rows < 1 or replicates < 1 or not blocks:
        raise ValueError("rows, replicates, and blocks must be non-empty")
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, rows), dtype=np.int32)
    for replicate in range(replicates):
        cursor = 0
        while cursor < rows:
            block = blocks[int(rng.integers(0, len(blocks)))]
            take = min(len(block), rows - cursor)
            output[replicate, cursor : cursor + take] = block[:take]
            cursor += take
    return output


def _paired_summary(
    baseline: np.ndarray, candidate: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if baseline_values.shape != candidate_values.shape or baseline_values.ndim != 1:
        raise ValueError("Paired losses must be matched one-dimensional arrays")
    if len(baseline_values) != indices.shape[1]:
        raise ValueError("Bootstrap index width differs from the loss arrays")
    if np.any(baseline_values < 0.0) or np.any(candidate_values < 0.0):
        raise ValueError("Loss arrays must be non-negative")

    difference = candidate_values - baseline_values
    replicate_difference = np.empty(len(indices), dtype=np.float64)
    replicate_improvement = np.empty(len(indices), dtype=np.float64)
    for row, sampled in enumerate(indices):
        delta = float(difference[sampled].mean())
        denominator = float(baseline_values[sampled].mean())
        replicate_difference[row] = delta
        replicate_improvement[row] = (
            -100.0 * delta / denominator if denominator > 0.0 else np.nan
        )

    finite = replicate_improvement[np.isfinite(replicate_improvement)]
    baseline_mean = float(baseline_values.mean())
    if baseline_mean <= 0.0 or not len(finite):
        raise RuntimeError("Calendar bootstrap encountered a zero baseline loss")
    point_difference = float(difference.mean())
    return {
        "candidate_minus_baseline_mean_loss": point_difference,
        "candidate_minus_baseline_mean_loss_calendar_block_95ci": [
            float(np.quantile(replicate_difference, 0.025)),
            float(np.quantile(replicate_difference, 0.975)),
        ],
        "improvement_percent": float(-100.0 * point_difference / baseline_mean),
        "improvement_percent_calendar_block_95ci": [
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
        ],
        "calendar_block_probability_of_improvement": float(
            np.mean(replicate_difference < 0.0)
        ),
    }


def _worst_fixed_summary(
    baseline: np.ndarray, candidate: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if baseline_values.shape != candidate_values.shape or baseline_values.ndim != 2:
        raise ValueError("Worst-fixed losses must have shape (stresses, rows)")
    if baseline_values.shape[1] != indices.shape[1]:
        raise ValueError("Bootstrap index width differs from worst-fixed losses")
    if not np.isfinite(baseline_values).all() or not np.isfinite(candidate_values).all():
        raise ValueError("Worst-fixed losses must be finite")
    if np.any(baseline_values < 0.0) or np.any(candidate_values < 0.0):
        raise ValueError("Worst-fixed losses must be non-negative")

    replicate_difference = np.empty(len(indices), dtype=np.float64)
    replicate_improvement = np.empty(len(indices), dtype=np.float64)
    for row, sampled in enumerate(indices):
        baseline_worst = float(baseline_values[:, sampled].mean(axis=1).max())
        candidate_worst = float(candidate_values[:, sampled].mean(axis=1).max())
        delta = candidate_worst - baseline_worst
        replicate_difference[row] = delta
        replicate_improvement[row] = (
            -100.0 * delta / baseline_worst if baseline_worst > 0.0 else np.nan
        )

    baseline_point = float(baseline_values.mean(axis=1).max())
    candidate_point = float(candidate_values.mean(axis=1).max())
    point_difference = candidate_point - baseline_point
    finite = replicate_improvement[np.isfinite(replicate_improvement)]
    if baseline_point <= 0.0 or not len(finite):
        raise RuntimeError("Calendar bootstrap encountered zero worst-fixed loss")
    return {
        "baseline_worst_fixed_mae_eur_mwh": baseline_point,
        "candidate_worst_fixed_mae_eur_mwh": candidate_point,
        "candidate_minus_baseline_mean_loss": point_difference,
        "candidate_minus_baseline_mean_loss_calendar_block_95ci": [
            float(np.quantile(replicate_difference, 0.025)),
            float(np.quantile(replicate_difference, 0.975)),
        ],
        "improvement_percent": float(-100.0 * point_difference / baseline_point),
        "improvement_percent_calendar_block_95ci": [
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
        ],
        "calendar_block_probability_of_improvement": float(
            np.mean(replicate_difference < 0.0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block_days", type=float, default=7.0)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --force")
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing v13 report: {report_path}")
    if args.replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol", {}).get("name") != "af_cdmo_real_loss_safe_v14":
        raise RuntimeError("Unexpected report protocol")
    loss_record = report["timestamp_loss_archive"]
    loss_path = Path(loss_record["path"]).expanduser().resolve()
    if sha256_file(loss_path) != loss_record["sha256"]:
        raise RuntimeError("Timestamp-loss archive hash mismatch")

    with np.load(loss_path) as archive:
        timestamps = np.asarray(archive["timestamps_utc_ns"], dtype=np.int64)
        blocks, time_audit = _calendar_blocks(timestamps, block_days=args.block_days)
        bootstrap_indices = _bootstrap_indices(
            blocks,
            rows=len(timestamps),
            replicates=args.replicates,
            seed=args.seed,
        )
        stresses = tuple(report["protocol"]["presentations"])
        if "original" not in stresses:
            raise RuntimeError("The registered presentation orbit omitted original")
        original_index = stresses.index("original")
        comparisons: dict[str, Any] = {}
        for baseline in BASELINES:
            baseline_stack = np.stack(
                [np.asarray(archive[f"loss__{baseline}__{stress}"]) for stress in stresses]
            )
            candidate_stack = np.stack(
                [np.asarray(archive[f"loss__{CANDIDATE}__{stress}"]) for stress in stresses]
            )
            prefix = f"{baseline}_to_{CANDIDATE}"
            comparisons[f"original::{prefix}"] = _paired_summary(
                baseline_stack[original_index],
                candidate_stack[original_index],
                bootstrap_indices,
            )
            comparisons[f"orbit_mean::{prefix}"] = _paired_summary(
                baseline_stack.mean(axis=0),
                candidate_stack.mean(axis=0),
                bootstrap_indices,
            )
            comparisons[f"adversarial_envelope::{prefix}"] = _paired_summary(
                baseline_stack.max(axis=0),
                candidate_stack.max(axis=0),
                bootstrap_indices,
            )
            comparisons[f"worst_fixed::{prefix}"] = _worst_fixed_summary(
                baseline_stack, candidate_stack, bootstrap_indices
            )

    result = {
        "protocol": {
            "name": "af_cdmo_real_v14_calendar_time_bootstrap_sensitivity",
            "study_role": report["protocol"]["study_role"],
            "primary_results_unchanged": True,
            "block_days": args.block_days,
            "replicates": args.replicates,
            "seed": args.seed,
            "sampling": (
                "paired circular moving blocks of fixed calendar duration; "
                "observed-row count varies with source-panel gaps"
            ),
        },
        "sources": {
            "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            "timestamp_losses": {
                "path": str(loss_path),
                "sha256": sha256_file(loss_path),
            },
            "engine": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "time_audit": time_audit,
        "comparisons": comparisons,
    }
    _atomic_json(output_path, result)
    print(f"[OK] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
