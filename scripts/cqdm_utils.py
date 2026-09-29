"""Shared deterministic utilities for the CQDM publication campaign."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAY_AHEAD_QUARTER_HOUR_START = pd.Timestamp("2025-09-30T22:00:00Z")
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_ROWS = 7 * 24 * 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_native_hourly_certificates(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold pre-transition hourly certificates over their four delivery MTUs."""

    pre = frame.loc[frame["delivery_utc"] < DAY_AHEAD_QUARTER_HOUR_START]
    post = frame.loc[frame["delivery_utc"] >= DAY_AHEAD_QUARTER_HOUR_START]
    if pre.empty:
        return frame
    if pre["delivery_utc"].dt.minute.ne(0).any():
        raise RuntimeError(
            "Pre-transition JAO data unexpectedly contains non-hourly delivery stamps"
        )
    expanded = []
    for offset in (0, 15, 30, 45):
        piece = pre.copy()
        piece["delivery_utc"] = piece["delivery_utc"] + pd.Timedelta(minutes=offset)
        expanded.append(piece)
    return pd.concat([*expanded, post], ignore_index=True).sort_values("delivery_utc")


def paired_block_bootstrap(
    baseline_losses: dict[str, np.ndarray],
    candidate_losses: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Paired circular-block uncertainty with a replicate-specific denominator."""

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, Any] = {}
    for metric in baseline_losses:
        baseline = np.asarray(baseline_losses[metric], dtype=np.float64)
        candidate = np.asarray(candidate_losses[metric], dtype=np.float64)
        if len(baseline) != len(candidate) or not len(baseline):
            raise RuntimeError(f"Invalid paired losses for {metric}")
        if np.any(baseline < 0.0) or np.any(candidate < 0.0):
            raise RuntimeError(f"Paired losses must be non-negative for {metric}")

        difference = candidate - baseline
        n = len(difference)
        block = min(BOOTSTRAP_BLOCK_ROWS, n)
        blocks_needed = int(np.ceil(n / block))
        bootstrap_means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        bootstrap_improvement = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        offsets = np.arange(block, dtype=np.int64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            starts = rng.integers(0, n, size=blocks_needed)
            indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
            replicate_difference = float(difference[indices].mean())
            replicate_baseline = float(baseline[indices].mean())
            bootstrap_means[replicate] = replicate_difference
            bootstrap_improvement[replicate] = (
                -100.0 * replicate_difference / replicate_baseline
                if replicate_baseline > 0.0
                else np.nan
            )

        baseline_mean = float(baseline.mean())
        if baseline_mean <= 0.0:
            raise RuntimeError(f"Baseline mean loss is zero for {metric}")
        finite_improvement = bootstrap_improvement[np.isfinite(bootstrap_improvement)]
        if not len(finite_improvement):
            raise RuntimeError(f"Bootstrap baseline loss is zero for every {metric} replicate")
        output[metric] = {
            "candidate_minus_baseline_mean_loss": float(difference.mean()),
            "candidate_minus_baseline_mean_loss_block_bootstrap_95ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "improvement_percent": float(-100.0 * difference.mean() / baseline_mean),
            "improvement_percent_block_bootstrap_95ci": [
                float(np.quantile(finite_improvement, 0.025)),
                float(np.quantile(finite_improvement, 0.975)),
            ],
            "block_bootstrap_probability_of_improvement": float(
                np.mean(bootstrap_means < 0.0)
            ),
            "bootstrap": {
                "method": "paired circular moving block",
                "replicates": BOOTSTRAP_REPLICATES,
                "block_rows": block,
                "seed": BOOTSTRAP_SEED,
                "percentage_denominator": "replicate-specific baseline mean",
            },
        }
    return output
