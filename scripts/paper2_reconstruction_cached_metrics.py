"""Portable exact numerical bodies for the frozen exploratory reconstruction audit.

No provider access, model loading, fitting, or neural dependencies.
"""
from __future__ import annotations
import itertools
from types import SimpleNamespace
import numpy as np
import pandas as pd
try:
    from . import paper2_nested_cached_metrics as cached
except ImportError:
    import paper2_nested_cached_metrics as cached
METHODS = ("analytic_reference", "field12", "analytic_current33", "first_moments48", "quotient84")
PAIRS = tuple((METHODS[j], METHODS[i]) for i, j in itertools.combinations(range(5), 2))
nested = SimpleNamespace(BLOCKS=cached.BLOCKS, REPLICATES=cached.REPLICATES,
                         SEED=cached.SEED, summarize_draws=cached.summarize_draws)
operational = SimpleNamespace(calendar_weights=cached.calendar_weights)

def center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)

def pairs(field: np.ndarray) -> np.ndarray:
    return np.column_stack([field[:, i] - field[:, j] for i in range(field.shape[1]) for j in range(i + 1, field.shape[1])])

def weighted_quantile(sorted_values: np.ndarray, sorted_weights: np.ndarray, q: float) -> float:
    cumulative = np.cumsum(sorted_weights)
    size = int(cumulative[-1])
    position = (size - 1) * q
    (lower, upper) = (int(np.floor(position)), int(np.ceil(position)))
    low = sorted_values[np.searchsorted(cumulative, lower + 1)]
    high = sorted_values[np.searchsorted(cumulative, upper + 1)]
    return float(low + (position - lower) * (high - low))

def analyze(matrix: tuple, target: np.ndarray, predictions: dict) -> tuple[dict, dict]:
    (_, _, _, timestamps) = matrix
    dates = timestamps.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq='D')
    day = calendar.get_indexer(dates)
    count_day = np.bincount(day, minlength=len(calendar)).astype(float)
    actual_pair = pairs(target)
    (point, losses, loss_day, squared_day, ordering) = ({}, {}, {}, {}, {})
    for (method, prediction) in predictions.items():
        error = pairs(prediction) - actual_pair
        absolute = np.abs(error).mean(axis=1)
        squared = np.square(error).mean(axis=1)
        losses[method] = absolute
        point[method] = dict(pairwise_mae=float(absolute.mean()), p90_timestamp_mae=float(np.quantile(absolute, 0.9)), pairwise_rmse=float(np.sqrt(squared.mean())), maximum_field_sum=float(np.max(np.abs(prediction.sum(axis=1)), initial=0)))
        loss_day[method] = np.bincount(day, weights=absolute, minlength=len(calendar))
        squared_day[method] = np.bincount(day, weights=squared, minlength=len(calendar))
        ordering[method] = np.argsort(absolute, kind='stable')
    (uncertainty, paired) = ({}, {})
    for block in nested.BLOCKS:
        print(f'  Reconstruction uncertainty {block}calendar days, {len(timestamps)}rows.', flush=True)
        weights = operational.calendar_weights(len(calendar), nested.REPLICATES, nested.SEED, block_days=block)
        denominator = weights @ count_day
        draws = {}
        for method in METHODS:
            sorted_rows = ordering[method]
            sorted_values = losses[method][sorted_rows]
            sorted_day = day[sorted_rows]
            p90 = np.asarray([weighted_quantile(sorted_values, draw[sorted_day], 0.9) for draw in weights])
            draws[method] = dict(pairwise_mae=weights @ loss_day[method] / denominator, pairwise_rmse=np.sqrt(weights @ squared_day[method] / denominator), p90_timestamp_mae=p90)
        uncertainty[str(block)] = {method: {metric: nested.summarize_draws(point[method][metric], values) for (metric, values) in method_draws.items()} for (method, method_draws) in draws.items()}
        paired[str(block)] = {f'{candidate}_minus_{baseline}': {metric: nested.summarize_draws(point[candidate][metric] - point[baseline][metric], draws[candidate][metric] - draws[baseline][metric]) for metric in ('pairwise_mae', 'p90_timestamp_mae', 'pairwise_rmse')} for (candidate, baseline) in PAIRS}
    return (dict(rows=len(timestamps), first_utc=str(timestamps[0]), last_utc=str(timestamps[-1]), point=point, uncertainty=uncertainty, paired=paired), losses)
