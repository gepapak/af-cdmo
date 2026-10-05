"""Portable cache-only metrics; cloned frozen numerical function bodies.

No provider data, private path, model reconstruction or neural dependency.
All bootstrap grids and pairs retain the original exploratory estimands.
"""
from __future__ import annotations
import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

METHODS = ("field12", "analytic_current33", "first_moments48", "quotient84")
BLOCKS = (7, 14, 28)
REPLICATES = 2000
SEED = 20261002
BOOTSTRAP_SEED = 20261002
EPSILON = 1.0e-7
METRICS = ("escalation_fraction", "accepted_tail_rate", "missed_tail_fraction", "tail_recall", "escalation_precision", "retained_mean_residual_eur_mwh")
PAIR_GRID = tuple((METHODS[j], METHODS[i]) for i, j in itertools.combinations(range(4), 2))
HGB_PARAMETERS = dict(loss="log_loss", learning_rate=0.05, max_iter=220,
    max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=3.0, random_state=SEED)

def ratios(counts: np.ndarray) -> dict[str, np.ndarray]:
    n, flagged, tails, caught, residual = np.moveaxis(counts, -1, 0)
    accepted = n - flagged
    missed = tails - caught
    def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)
    return {
        "escalation_fraction": divide(flagged, n),
        "accepted_tail_rate": divide(missed, accepted),
        "missed_tail_fraction": divide(missed, tails),
        "tail_recall": divide(caught, tails),
        "escalation_precision": divide(caught, flagged),
        "retained_mean_residual_eur_mwh": divide(residual, accepted),
    }

def daily_budget(probability: np.ndarray, day_index: np.ndarray, budget: float) -> np.ndarray:
    """Decisions use scores and dates only; outcomes never enter this function."""
    flagged = np.zeros(len(probability), dtype=float)
    for day in np.unique(day_index):
        rows = np.flatnonzero(day_index == day)
        count = int(math.ceil(float(budget) * len(rows)))
        order = np.argsort(-probability[rows], kind="stable")
        flagged[rows[order[:count]]] = 1.0
    return flagged

def calendar_weights(days: int, replicates: int, seed: int, block_days: int = 7) -> np.ndarray:
    """Circular blocks of seven contiguous UTC calendar days, including gaps."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(replicates, math.ceil(days / block_days)))
    indices = (starts[:, :, None] + np.arange(block_days)) % days
    indices = indices.reshape(replicates, -1)[:, :days]
    weights = np.zeros((replicates, days), dtype=np.int32)
    for replicate, row in enumerate(indices):
        weights[replicate] = np.bincount(row, minlength=days)
    assert np.all(weights.sum(axis=1) == days)
    return weights

def aggregate_days(flagged: np.ndarray, target: np.ndarray, magnitude: np.ndarray, day_index: np.ndarray, days: int) -> np.ndarray:
    n = np.bincount(day_index, minlength=days).astype(float)
    return np.stack((
        n,
        np.bincount(day_index, weights=flagged, minlength=days),
        np.bincount(day_index, weights=target, minlength=days),
        np.bincount(day_index, weights=flagged * target, minlength=days),
        np.bincount(day_index, weights=(1 - flagged) * magnitude, minlength=days),
    ), axis=-1)

def interval(value: np.ndarray) -> list[float | None]:
    finite = value[np.isfinite(value)]
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])] if len(finite) else [None, None]

def control_features(features: np.ndarray, method: str) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 84:
        raise ValueError("Expected exact frozen 84-column feature matrix")
    if method == "field12":
        return features[:, :12]
    if method == "analytic_current33":
        current = -np.expm1(features[:, 0])[:, None] * features[:, 12:43]
        return np.column_stack((features[:, :2], current))
    if method == "first_moments48":
        return features[:, :48]
    if method == "quotient84":
        return features
    raise KeyError(method)

def verify_field_identity(features: np.ndarray, indices: np.ndarray) -> float:
    current = control_features(features, "analytic_current33")[:, 2:]
    physical = current[:, indices]
    physical = physical - physical.mean(axis=1, keepdims=True)
    error = float(np.max(np.abs(physical - features[:, 2:12]), initial=0))
    if error > 1e-9:
        raise RuntimeError(f"Analytic current does not reproduce the ten-zone field: {error}")
    return error

def read_matrix(path: Path) -> tuple:
    with np.load(path, allow_pickle=False) as data:
        return (data["features"], data["target"], data["residual_magnitude_eur_mwh"],
                pd.DatetimeIndex(pd.to_datetime(data["timestamp_utc"], utc=True)))

def auc_grouping(probability: np.ndarray) -> np.ndarray:
    return np.unique(probability, return_inverse=True)[1]

def weighted_auc(target: np.ndarray, groups: np.ndarray, row_weights: np.ndarray) -> float:
    positive = np.bincount(groups, weights=target * row_weights)
    negative = np.bincount(groups, weights=(1 - target) * row_weights)
    denominator = positive.sum() * negative.sum()
    if denominator == 0:
        return float("nan")
    return float(np.dot(positive, np.cumsum(negative) - 0.5 * negative) / denominator)

def summarize_draws(estimate: float, draws: np.ndarray) -> dict:
    finite = draws[np.isfinite(draws)]
    return dict(estimate=float(estimate), ci95=operational.interval(draws),
                finite_replicates=int(len(finite)), invalid_replicates=int(len(draws) - len(finite)))

def analyze_window(matrix: tuple, probabilities: dict, thresholds: dict) -> dict:
    _, target, magnitude, timestamps = matrix
    dates = timestamps.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    day_index = calendar.get_indexer(dates)
    days = len(calendar)
    day_n = np.bincount(day_index, minlength=days).astype(float)
    classification = {method: temporal._safe_classification(target, value) for method, value in probabilities.items()}
    clipped = {method: np.clip(value, 1e-8, 1 - 1e-8) for method, value in probabilities.items()}
    loss_day, groups = {}, {}
    for method, value in clipped.items():
        losses = dict(brier=(value - target) ** 2,
                      log_loss=-target * np.log(value) - (1 - target) * np.log1p(-value))
        loss_day[method] = {name: np.bincount(day_index, weights=loss, minlength=days) for name, loss in losses.items()}
        groups[method] = auc_grouping(value)
        assert abs(weighted_auc(target, groups[method], np.ones(len(target))) - classification[method]["roc_auc"]) < 1e-12
    decisions = {}
    for method, value in probabilities.items():
        decisions[f"fixed_december/{method}"] = (value > thresholds[method]).astype(float)
        for budget in (0.1, 0.2, 0.3):
            decisions[f"daily_{budget:.2f}/{method}"] = operational.daily_budget(value, day_index, budget)
        selected = np.argsort(-value, kind="stable")[:int(np.floor(0.20 * len(target)))]
        global_flagged = np.zeros(len(target), dtype=float)
        global_flagged[selected] = 1.0
        decisions[f"global_0.20/{method}"] = global_flagged
    policy_counts = {name: operational.aggregate_days(value, target, magnitude, day_index, days) for name, value in decisions.items()}
    policy_point = {}
    for name, counts in policy_counts.items():
        n, flagged, tails, caught, _ = counts.sum(axis=0)
        policy_point[name] = dict(rows=int(n), reviewed=int(flagged), accepted=int(n - flagged),
            tail_events=int(tails), caught=int(caught), missed=int(tails - caught),
            metrics={key: float(value) for key, value in operational.ratios(counts.sum(axis=0)).items()})
    classification_uncertainty, pair_uncertainty, policy_uncertainty = {}, {}, {}
    policy_pairs = {}
    for block in BLOCKS:
        print(f"  Computing paired {block}-calendar-day uncertainty on {len(target)} rows.", flush=True)
        day_weights = operational.calendar_weights(days, REPLICATES, SEED, block_days=block)
        denominator = day_weights @ day_n
        draws = {method: {name: (day_weights @ day_loss) / denominator for name, day_loss in loss_day[method].items()} for method in METHODS}
        for method in METHODS:
            auc = np.empty(REPLICATES, dtype=float)
            for replicate, weights in enumerate(day_weights):
                auc[replicate] = weighted_auc(target, groups[method], weights[day_index])
            draws[method]["roc_auc"] = auc
        classification_uncertainty[str(block)] = {method: {name: summarize_draws(classification[method][name], value)
            for name, value in method_draws.items()} for method, method_draws in draws.items()}
        pair_uncertainty[str(block)] = {f"{candidate}_minus_{baseline}": {metric: summarize_draws(
            classification[candidate][metric] - classification[baseline][metric], draws[candidate][metric] - draws[baseline][metric])
            for metric in ("brier", "log_loss", "roc_auc")} for candidate, baseline in PAIR_GRID}
        policy_draws = {name: operational.ratios(day_weights @ counts) for name, counts in policy_counts.items()}
        policy_uncertainty[str(block)] = {name: {metric: summarize_draws(policy_point[name]["metrics"][metric], value)
            for metric, value in values.items()} for name, values in policy_draws.items()}
        policy_pairs[str(block)] = {}
        for policy in ("fixed_december", "daily_0.10", "daily_0.20", "daily_0.30", "global_0.20"):
            policy_pairs[str(block)][policy] = {f"{candidate}_minus_{baseline}": {metric: summarize_draws(
                policy_point[f"{policy}/{candidate}"]["metrics"][metric] - policy_point[f"{policy}/{baseline}"]["metrics"][metric],
                policy_draws[f"{policy}/{candidate}"][metric] - policy_draws[f"{policy}/{baseline}"][metric])
                for metric in operational.METRICS} for candidate, baseline in PAIR_GRID}
    return dict(rows=len(target), first_utc=str(timestamps[0]), last_utc=str(timestamps[-1]),
        tail_events=int(target.sum()), observed_days=int(len(np.unique(day_index))), calendar_days=days,
        classification=classification, classification_uncertainty=classification_uncertainty,
        paired_classification=pair_uncertainty, policies=policy_point,
        policy_uncertainty=policy_uncertainty, paired_policy_metrics=policy_pairs)

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

@dataclass(frozen=True)
class PlattMap:
    intercept: float
    coefficient: float

    def probability(self, logits: np.ndarray) -> np.ndarray:
        value = self.intercept + self.coefficient * np.asarray(logits, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))

def _fit_platt(logits: np.ndarray, target: np.ndarray) -> PlattMap:
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=BOOTSTRAP_SEED)
    model.fit(np.asarray(logits)[:, None], target)
    return PlattMap(float(model.intercept_[0]), float(model.coef_[0, 0]))

def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))

def center_price_field(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("Price field must have at least one dimension")
    return values - values.mean(axis=-1, keepdims=True)

def quotient_feature(
    canonical: np.ndarray,
    weights: np.ndarray,
    total: float,
    real_zone_indices: np.ndarray,
) -> np.ndarray:
    """Fixed invariant moments for the multi-zone classical control."""

    canonical = np.asarray(canonical, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(weights) or total <= 0.0:
        dimension = canonical.shape[1]
        return np.zeros(2 + len(real_zone_indices) + 2 * dimension, dtype=np.float64)
    normalized = weights / weights.sum()
    mean = normalized @ canonical
    second = normalized @ np.square(canonical)
    dual_potential = total * mean[real_zone_indices]
    analytic = center_price_field(-dual_potential)
    return np.concatenate(
        [np.asarray([np.log1p(total), 1.0]), analytic, mean, second]
    )

operational = SimpleNamespace(ratios=ratios, daily_budget=daily_budget,
    calendar_weights=calendar_weights, aggregate_days=aggregate_days,
    interval=interval, METRICS=METRICS)
temporal = SimpleNamespace(_safe_classification=_safe_classification)
