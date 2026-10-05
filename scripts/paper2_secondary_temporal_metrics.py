"""Standalone statistical functions used by Paper2 later-period adapters.

Copied without numerical changes from the frozen QCT chronology engine;
the original module SHA-256 is recorded below. These use circular blocks
of observed UTC days and retain the original whole-block oversampling.
The main confirmation triage uses a different complete-calendar grid.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ORIGINAL_STATISTICS_SOURCE_SHA256 = "ec91fcd9643d63abd4a07f5afd619baaf6d6569bce684690dd7cb2c591c8337c"

EPSILON = 1.0e-9


def classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    if target.shape != probability.shape:
        raise ValueError("Target and probability shapes differ")
    unique = np.unique(target)
    return {
        "rows": int(len(target)),
        "prevalence": float(target.mean()),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability)) if len(unique) == 2 else float("nan"),
        "average_precision": float(average_precision_score(target, probability)) if target.sum() else float("nan"),
        "mean_probability": float(probability.mean()),
    }


def exact_budget_flags(probability: np.ndarray, audit_budget: float) -> np.ndarray:
    """Flag exactly the highest-risk floor(budget*n) rows with stable ties."""

    probability = np.asarray(probability, dtype=np.float64)
    if not 0.0 < audit_budget < 1.0:
        raise ValueError("audit_budget must lie in (0, 1)")
    count = max(1, int(math.floor(audit_budget * len(probability))))
    order = np.argsort(-probability, kind="stable")
    flags = np.zeros(len(probability), dtype=bool)
    flags[order[:count]] = True
    return flags


def operational_metrics(
    target: np.ndarray,
    magnitude: np.ndarray,
    probability: np.ndarray,
    audit_budget: float,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.float64) > 0.5
    magnitude = np.asarray(magnitude, dtype=np.float64)
    flags = exact_budget_flags(probability, audit_budget)
    accepted = ~flags
    tail_count = max(int(target.sum()), 1)
    flagged_count = max(int(flags.sum()), 1)
    missed = accepted & target
    return {
        "audit_budget": float(audit_budget),
        "flagged_rows": int(flags.sum()),
        "flag_rate": float(flags.mean()),
        "tail_recall": float((flags & target).sum() / tail_count),
        "tail_precision": float((flags & target).sum() / flagged_count),
        "missed_tail_rows": int(missed.sum()),
        "missed_tail_fraction": float(missed.sum() / tail_count),
        "accepted_tail_rate": float(target[accepted].mean()) if accepted.any() else float("nan"),
        "missed_tail_magnitude_sum_eur_mwh": float(magnitude[missed].sum()),
        "missed_tail_magnitude_mean_eur_mwh": float(magnitude[missed].mean()) if missed.any() else 0.0,
        "accepted_p95_magnitude_eur_mwh": float(np.quantile(magnitude[accepted], 0.95)) if accepted.any() else float("nan"),
    }


def decision_cost(
    target: np.ndarray,
    probability: np.ndarray,
    audit_budget: float,
    *,
    missed_tail_cost: float,
    audit_cost: float,
) -> float:
    target = np.asarray(target, dtype=np.float64) > 0.5
    flags = exact_budget_flags(probability, audit_budget)
    costs = missed_tail_cost * ((~flags) & target).astype(np.float64) + audit_cost * flags.astype(np.float64)
    return float(costs.mean())


def _calendar_day_rows(timestamps: pd.DatetimeIndex) -> list[np.ndarray]:
    """Return one row-index array per observed UTC calendar day."""

    if timestamps.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    normalized = timestamps.tz_convert("UTC").normalize()
    return [np.flatnonzero(normalized == day) for day in pd.Index(normalized).unique()]


def _moving_calendar_blocks(
    timestamps: pd.DatetimeIndex, block_days: int
) -> list[np.ndarray]:
    if block_days < 1:
        raise ValueError("block_days must be positive")
    days = _calendar_day_rows(timestamps)
    if not days:
        raise ValueError("timestamps cannot be empty")
    width = min(block_days, len(days))
    return [
        np.concatenate([days[(start + offset) % len(days)] for offset in range(width)])
        for start in range(len(days))
    ]


def paired_calendar_block_bootstrap(
    timestamps: pd.DatetimeIndex,
    row_difference: np.ndarray,
    *,
    block_days: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    row_difference = np.asarray(row_difference, dtype=np.float64)
    if len(timestamps) != len(row_difference):
        raise ValueError("timestamps and row_difference lengths differ")
    blocks = _moving_calendar_blocks(timestamps, block_days)
    day_count = len(_calendar_day_rows(timestamps))
    rng = np.random.default_rng(seed + block_days)
    draws = np.empty(replicates, dtype=np.float64)
    sampled_blocks = int(math.ceil(day_count / min(block_days, day_count)))
    for draw in range(replicates):
        sampled = rng.integers(0, len(blocks), size=sampled_blocks)
        rows = np.concatenate([blocks[index] for index in sampled])
        draws[draw] = float(row_difference[rows].mean())
    return {
        "estimate": float(row_difference.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_below_zero": float(np.mean(draws < 0.0)),
        "block_days": int(block_days),
        "possible_moving_blocks": int(len(blocks)),
        "sampled_blocks_per_replicate": sampled_blocks,
        "replicates": int(replicates),
    }


def threshold_operational_metrics(
    target: np.ndarray,
    magnitude: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate a threshold fixed on pre-holdout calibration data."""

    target = np.asarray(target, dtype=np.float64) > 0.5
    magnitude = np.asarray(magnitude, dtype=np.float64)
    flags = np.asarray(probability, dtype=np.float64) > float(threshold)
    accepted = ~flags
    tail_count = max(int(target.sum()), 1)
    flagged_count = max(int(flags.sum()), 1)
    missed = accepted & target
    return {
        "probability_threshold": float(threshold),
        "flagged_rows": int(flags.sum()),
        "realized_flag_rate": float(flags.mean()),
        "tail_recall": float((flags & target).sum() / tail_count),
        "tail_precision": float((flags & target).sum() / flagged_count),
        "missed_tail_rows": int(missed.sum()),
        "missed_tail_fraction": float(missed.sum() / tail_count),
        "accepted_tail_rate": float(target[accepted].mean()) if accepted.any() else float("nan"),
        "missed_tail_magnitude_sum_eur_mwh": float(magnitude[missed].sum()),
    }


def compare_brier(
    timestamps: pd.DatetimeIndex,
    target: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    *,
    block_days: Iterable[int],
    replicates: int,
    seed: int,
    absolute_margin: float,
    relative_margin: float,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    row_difference = np.square(candidate - target) - np.square(comparator - target)
    comparator_brier = float(np.square(comparator - target).mean())
    margin = min(float(absolute_margin), float(relative_margin) * comparator_brier)
    intervals = {
        str(days): paired_calendar_block_bootstrap(
            timestamps,
            row_difference,
            block_days=int(days),
            replicates=replicates,
            seed=seed,
        )
        for days in block_days
    }
    worst_upper = max(float(value["ci95"][1]) for value in intervals.values())
    return {
        "candidate_minus_comparator_brier": float(row_difference.mean()),
        "comparator_brier": comparator_brier,
        "absolute_margin": float(absolute_margin),
        "relative_margin": float(relative_margin),
        "effective_margin": float(margin),
        "calendar_block_bootstrap": intervals,
        "worst_upper_95ci": float(worst_upper),
        "noninferior_all_block_lengths": bool(worst_upper < margin),
        "superior_all_block_lengths": bool(worst_upper < 0.0),
    }


def compare_operational_cost(
    timestamps: pd.DatetimeIndex,
    target: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    *,
    audit_budget: float,
    missed_tail_cost: float,
    audit_cost: float,
    block_days: Iterable[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    target_bool = np.asarray(target, dtype=np.float64) > 0.5
    candidate_flags = exact_budget_flags(candidate, audit_budget)
    comparator_flags = exact_budget_flags(comparator, audit_budget)
    candidate_cost = missed_tail_cost * ((~candidate_flags) & target_bool).astype(float) + audit_cost * candidate_flags.astype(float)
    comparator_cost = missed_tail_cost * ((~comparator_flags) & target_bool).astype(float) + audit_cost * comparator_flags.astype(float)
    difference = candidate_cost - comparator_cost
    return {
        "candidate_mean_cost": float(candidate_cost.mean()),
        "comparator_mean_cost": float(comparator_cost.mean()),
        "candidate_minus_comparator_cost": float(difference.mean()),
        "calendar_block_bootstrap": {
            str(days): paired_calendar_block_bootstrap(
                timestamps,
                difference,
                block_days=int(days),
                replicates=replicates,
                seed=seed + 1000,
            )
            for days in block_days
        },
    }
