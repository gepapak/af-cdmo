"""Portable attribution numerical kernel from the frozen two-path audit.

All authorized input paths are explicit; no prices, models or network access.
The archive loop and primary scalar endpoints preserve the original AST.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PRIMARY_ZONES = ("DK1", "DK2", "FI", "NO1", "NO2", "NO3", "NO5", "SE1", "SE3", "SE4")
HVDC = (("DK1_SB", "DK2_SB"), ("FI_FS", "SE3_FS"), ("DK1_KS", "SE3_KS"), ("SE3_SWL", "SE4_SWL"))

def compute(archive: Path, geometry: Path, predictions: Path) -> tuple[dict, dict]:
    geo = json.loads(geometry.read_text(encoding='utf-8-sig'))
    zone_order = geo['geometry']['zone_order']
    unit_columns = ['unit_ptdf_' + zone for zone in zone_order]
    equality = np.ones((1 + len(HVDC), len(zone_order)))
    equality[1:] = 0
    for (row, endpoints) in enumerate(HVDC, 1):
        for zone in endpoints:
            equality[row, zone_order.index(zone)] = 1
    projector = np.eye(len(zone_order)) - equality.T @ np.linalg.pinv(equality @ equality.T) @ equality
    assert np.linalg.matrix_rank(equality) == 5 and np.max(np.abs(projector @ projector - projector)) < 1e-12
    indices = np.array([zone_order.index(zone) for zone in PRIMARY_ZONES])
    frame = pd.read_csv(predictions, usecols=['timestamp_utc'])
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc, utc=True))
    assert len(times) == 14482 and times.is_unique and times.is_monotonic_increasing
    (pair_i, pair_j) = np.triu_indices(len(PRIMARY_ZONES), 1)
    assert len(pair_i) == 45
    (fields_a, fields_b) = ({}, {})
    total_a = np.zeros((len(times), len(indices)))
    total_b = np.zeros_like(total_a)
    max_mass_difference = 0.0
    max_row_difference = 0.0
    selected_rows = 0
    present = np.zeros(len(times), dtype=bool)
    usecols = ['delivery_utc', 'publication_utc', 'tso', 'shadowPrice', 'ptdf_l2_norm', 'dual_mass_eur_mwh', *unit_columns]
    with threadpool_limits(limits=1):
        for chunk in pd.read_csv(archive, usecols=usecols, chunksize=20000):
            delivery = pd.DatetimeIndex(pd.to_datetime(chunk.delivery_utc, utc=True, format='mixed'))
            publication = pd.DatetimeIndex(pd.to_datetime(chunk.publication_utc, utc=True, format='mixed'))
            cohort_indices = times.get_indexer(delivery)
            keep = (cohort_indices >= 0) & (publication <= delivery - pd.Timedelta(minutes=60))
            selected = chunk.loc[keep].copy()
            row_index = cohort_indices[keep]
            if not len(selected):
                continue
            numeric = selected[['shadowPrice', 'ptdf_l2_norm', 'dual_mass_eur_mwh', *unit_columns]].to_numpy(dtype=float)
            assert np.all(np.isfinite(numeric)) and np.all(numeric[:, :3] > 0)
            (shadow, norm, mass) = (numeric[:, 0], numeric[:, 1], numeric[:, 2])
            unit = numeric[:, 3:]
            raw = unit * norm[:, None]
            field_a = -shadow[:, None] * (raw @ projector)[:, indices]
            field_b = -mass[:, None] * (unit @ projector)[:, indices]
            field_a -= field_a.mean(axis=1, keepdims=True)
            field_b -= field_b.mean(axis=1, keepdims=True)
            max_mass_difference = max(max_mass_difference, float(np.max(np.abs(mass - shadow * norm))))
            max_row_difference = max(max_row_difference, float(np.max(np.abs(field_a - field_b))))
            present[row_index] = True
            selected_rows += len(selected)
            np.add.at(total_a, row_index, field_a)
            np.add.at(total_b, row_index, field_b)
            tags = selected.tso.fillna('UNKNOWN').astype(str).to_numpy()
            for tag in np.unique(tags):
                if tag not in fields_a:
                    fields_a[tag] = np.zeros_like(total_a)
                    fields_b[tag] = np.zeros_like(total_a)
                mask = tags == tag
                np.add.at(fields_a[tag], row_index[mask], field_a[mask])
                np.add.at(fields_b[tag], row_index[mask], field_b[mask])
    tags = sorted(fields_a)
    group_a = np.stack([fields_a[tag] for tag in tags], axis=2)
    group_b = np.stack([fields_b[tag] for tag in tags], axis=2)
    sums_a = float(np.max(np.abs(group_a.sum(axis=2) - total_a)))
    sums_b = float(np.max(np.abs(group_b.sum(axis=2) - total_b)))
    pair_a = group_a[:, pair_i] - group_a[:, pair_j]
    pair_b = group_b[:, pair_i] - group_b[:, pair_j]
    (score_a, score_b) = (np.abs(pair_a), np.abs(pair_b))
    (selected_a, selected_b) = (np.argmax(score_a, axis=2), np.argmax(score_b, axis=2))
    nonzero = np.max(score_a, axis=2) > 0
    routing_flips = int(np.sum((selected_a != selected_b) & nonzero))
    sorted_scores = np.sort(score_a, axis=2)
    margins = sorted_scores[:, :, -1] - sorted_scores[:, :, -2]
    differences = np.max(np.abs(score_b - score_a), axis=2)
    certified = margins > 2 * differences
    assert not np.any((selected_a != selected_b) & certified)
    difference = float(np.max(np.abs(pair_b - pair_a)))
    primary = {'cohort_timestamps': len(times), 'selected_positive_mass_rows': selected_rows, 'timestamp_pair_opportunities': len(times) * 45, 'nonzero_group_attribution_opportunities': int(nonzero.sum()), 'timestamps_with_retained_certificate_rows': int(present.sum()), 'timestamps_without_retained_rows': int((~present).sum()), 'tso_provenance_tags': tags, 'maximum_mass_identity_difference_eur_mwh': max_mass_difference, 'maximum_centered_row_field_difference_eur_mwh': max_row_difference, 'maximum_total_spread_difference_eur_mwh': float(np.max(np.abs(total_b[:, pair_i] - total_b[:, pair_j] - (total_a[:, pair_i] - total_a[:, pair_j])))), 'maximum_group_spread_component_difference_eur_mwh': difference, 'group_sum_current_conservation_max_difference_route_a': sums_a, 'group_sum_current_conservation_max_difference_route_b': sums_b, 'top_absolute_contributor_tag_routing_flips': routing_flips, 'routing_flip_fraction_among_nonzero': routing_flips / int(nonzero.sum()), 'exact_tied_top_margins_among_nonzero': int(np.sum((margins == 0) & nonzero)), 'routing_margin_certified_opportunities': int(np.sum(certified & nonzero)), 'no_price_targets_or_model_scores_used': True}
    context = dict(times=times, tags=tags, score_a=score_a, score_b=score_b, selected_a=selected_a, selected_b=selected_b, nonzero=nonzero, margins=margins, differences=differences, routing_flips=routing_flips)
    return primary, context

def quantiles(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {'count': 0}
    return {'count': int(values.size), **{str(p): float(np.quantile(values, p)) for p in (0, 0.25, 0.5, 0.75, 1)}}

def row_route_fields(unit: np.ndarray, norm: np.ndarray, shadow: np.ndarray,
                     mass: np.ndarray, projector: np.ndarray, indices: np.ndarray) -> tuple:
    raw = unit * norm[:, None]
    field_a = -shadow[:, None] * (raw @ projector)[:, indices]
    field_b = -mass[:, None] * (unit @ projector)[:, indices]
    field_a -= field_a.mean(axis=1, keepdims=True)
    field_b -= field_b.mean(axis=1, keepdims=True)
    return field_a, field_b


def routing_arrays(score_a: np.ndarray, score_b: np.ndarray) -> dict:
    if score_a.shape != score_b.shape or score_a.ndim != 3 or score_a.shape[2] < 2:
        raise ValueError("Two aligned timestamp/pair/tag score arrays with at least two tags required")
    if np.any(score_a < 0) or np.any(score_b < 0) or not np.isfinite(score_a).all() or not np.isfinite(score_b).all():
        raise ValueError("Finite nonnegative absolute-component scores required")
    selected_a, selected_b = np.argmax(score_a, axis=2), np.argmax(score_b, axis=2)
    nonzero = np.max(score_a, axis=2) > 0
    sorted_scores = np.sort(score_a, axis=2)
    margins = sorted_scores[:, :, -1] - sorted_scores[:, :, -2]
    differences = np.max(np.abs(score_b - score_a), axis=2)
    return dict(score_a=score_a, score_b=score_b, selected_a=selected_a, selected_b=selected_b,
                nonzero=nonzero, margins=margins, differences=differences,
                routing_flips=int(np.sum((selected_a != selected_b) & nonzero)))


def descriptive_detail(context: dict) -> dict:
    nonzero = context["nonzero"]
    margin = context["margins"]
    drift = context["differences"]
    score_a, score_b = context["score_a"], context["score_b"]
    selected_a, selected_b = context["selected_a"], context["selected_b"]
    flip = (selected_a != selected_b) & nonzero
    individual_certificate = margin > 2 * drift
    global_drift = float(np.max(drift))
    global_certificate = margin > 2 * global_drift
    allzero_a = np.max(score_a, axis=2) == 0
    allzero_b = np.max(score_b, axis=2) == 0
    assert not np.any(flip & individual_certificate)
    assert not np.any(flip & global_certificate)
    assert np.all(nonzero == ~allzero_a)
    assert context["routing_flips"] == int(flip.sum())
    tags = context["tags"]
    transitions = {}
    for a, b in zip(selected_a[flip], selected_b[flip]):
        key = tags[a] + " -> " + tags[b]
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "cohort_timestamps": int(len(context["times"])),
        "pair_opportunities": int(nonzero.size),
        "nonzero_opportunities": int(nonzero.sum()),
        "exact_all_zero_route_a": int(allzero_a.sum()),
        "exact_all_zero_route_b": int(allzero_b.sum()),
        "exact_all_zero_screen_membership_changes": int(np.sum(allzero_a != allzero_b)),
        "routing_flips": int(flip.sum()),
        "routing_flip_distinct_timestamps": int(np.any(flip, axis=1).sum()),
        "routing_flip_distinct_zone_pairs": int(np.any(flip, axis=0).sum()),
        "individual_margin_certified_nonzero": int(np.sum(individual_certificate & nonzero)),
        "individual_margin_certified_flips": int(np.sum(individual_certificate & flip)),
        "global_score_drift_eur_mwh": global_drift,
        "global_margin_certified_nonzero": int(np.sum(global_certificate & nonzero)),
        "global_margin_certified_flips": int(np.sum(global_certificate & flip)),
        "flip_route_a_top_absolute_component_eur_mwh": quantiles(np.max(score_a, axis=2)[flip]),
        "flip_route_b_top_absolute_component_eur_mwh": quantiles(np.max(score_b, axis=2)[flip]),
        "flip_route_a_rank_margin_eur_mwh": quantiles(margin[flip]),
        "flip_max_score_drift_eur_mwh": quantiles(drift[flip]),
        "routing_transition_counts": transitions,
        "checks": {"all_flips_outside_certified_margin": True,
                   "exact_zero_screen_identical": bool(np.all(allzero_a == allzero_b)),
                   "no_new_tolerance_or_policy": True},
    }
