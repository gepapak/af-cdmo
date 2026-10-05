"""Portable, exploratory triage audit from explicitly supplied authorized inputs.

No model fitting, calibration, data acquisition, or historical artifact mutation.
Daily workload comparisons describe batch review after a day's certificates are
available; prices are withheld from decisions and revealed only for evaluation.
They do not simulate observed outages or establish deployed monetary value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import paper2_portable_audit_utils as portable

import numpy as np
import pandas as pd

METHODS = ("qct_full", "qct_no_tangent", "hgb_quotient", "uniform_set_transformer", "hgb_calendar")
METRICS = ("escalation_fraction", "accepted_tail_rate", "missed_tail_fraction", "tail_recall", "escalation_precision", "retained_mean_residual_eur_mwh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def metrics_record(counts: np.ndarray, bootstrap_counts: np.ndarray) -> dict:
    values = ratios(counts)
    bootstrap = ratios(bootstrap_counts)
    n, flagged, tails, caught, _ = counts
    return {
        "rows": int(n), "escalated": float(flagged), "accepted": float(n - flagged),
        "tail_events": int(tails), "caught_tail_events": float(caught), "missed_tail_events": float(tails - caught),
        "metrics": {name: {"estimate": float(value) if np.isfinite(value) else None, "calendar_7day_95ci": interval(bootstrap[name])} for name, value in values.items()},
    }


def self_test() -> None:
    dates = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.8, 0.8, 0.1, 0.4, 0.9])
    target = np.array([1, 0, 1, 0, 1], dtype=float)
    decision = daily_budget(scores, dates, 0.5)
    assert np.array_equal(decision, [1, 1, 0, 0, 1])
    counts = aggregate_days(decision, target, np.ones(5), dates, 3).sum(axis=0)
    assert np.array_equal(counts, [5, 3, 3, 2, 2])
    assert ratios(counts)["accepted_tail_rate"] == 0.5
    weights = calendar_weights(17, 13, 7)
    assert weights.shape == (13, 17) and np.all(weights.sum(axis=1) == 17)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5102026)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--temporal-audit", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    portable.add_source_arguments(parser)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Synthetic count, label-free ranking, tie rule and calendar-block checks passed.")
        return
    if args.replicates < 100:
        raise ValueError("At least 100 uncertainty replicates are required")
    root = Path(__file__).resolve().parents[1]
    required = ("report", "temporal_audit", "predictions", "manifest", "output")
    if any(getattr(args, name) is None for name in required):
        parser.error("Supply --report, --temporal-audit, --predictions, --manifest and --output")
    report_path, audit_path = args.report.resolve(), args.temporal_audit.resolve()
    prediction_path, manifest_path = args.predictions.resolve(), args.manifest.resolve()
    manifest = portable.read_json(manifest_path)
    source_checks = [portable.manifest_check(path, manifest, logical) for path, logical in (
        (report_path, "af_cdmo_probabilistic_headroom/quotient_tangent_risk_confirmation_v5.json"),
        (audit_path, "af_cdmo_probabilistic_headroom/qct_risk_temporal_robustness_v5/temporal_robustness_audit.json"),
        (prediction_path, "af_cdmo_probabilistic_headroom/qct_risk_temporal_robustness_v5/confirmation_predictions_internal.csv.gz"),
    )]
    report = portable.map_report_sources(portable.read_json(report_path), args, root,
        names=("dual_archive", "geometry_audit", "price_root"))
    original_audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    # Verify the original provider-source bytes and geometry through the report,
    # without reading values into the new analysis or emitting provider records.
    raw_source_checks = []
    entries = [("dual_archive", report["sources"]["dual_archive"], report["sources"]["dual_archive_sha256"]), ("geometry_audit", report["sources"]["geometry_audit"], report["sources"]["geometry_audit_sha256"])]
    entries.extend((zone, entry["path"], entry["sha256"]) for zone, entry in report["sources"]["price_provenance"].items())
    for label, path, expected_digest in entries:
        digest = sha256(Path(path))
        assert digest == expected_digest.lower(), f"Source hash mismatch: {label}"
        raw_source_checks.append({"source": label, "sha256": digest, "verified": True})
    frame = pd.read_csv(prediction_path)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    assert len(frame) == 14482 and timestamps.is_unique and timestamps.is_monotonic_increasing
    assert timestamps.min() >= pd.Timestamp(report["protocol"]["confirmation_start"])
    assert timestamps.max() < pd.Timestamp(report["protocol"]["confirmation_end_exclusive"])
    target = frame["tail_target"].to_numpy(dtype=float)
    magnitude = frame["residual_magnitude_eur_mwh"].to_numpy(dtype=float)
    assert set(np.unique(target)) == {0.0, 1.0}
    assert np.all(np.isfinite(magnitude)) and np.all(magnitude >= 0)
    assert np.array_equal(target, magnitude > report["protocol"]["tail_thresholds_eur_mwh"][1])
    dates = timestamps.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    day_index = calendar.get_indexer(dates)
    weights = calendar_weights(len(calendar), args.replicates, args.seed)
    policies = {"all_accept": np.zeros(len(frame)), "all_escalate": np.ones(len(frame))}
    thresholds = {}
    probabilities = {}
    for method in METHODS:
        probability = frame[f"probability_{method}"].to_numpy(dtype=float)
        assert np.all(np.isfinite(probability)) and np.all((probability >= 0) & (probability <= 1))
        probabilities[method] = probability
        thresholds[method] = report["methods"][method]["decision_at_calibration_80pct_coverage"]["probability_threshold_from_december"]
        policies[f"fixed/{method}"] = (probability > thresholds[method]).astype(float)
        # Model-free expected random review, with the same workload as each
        # fixed method in each day. This is conditional expectation, not one
        # favorable random draw or a prospective realized workload guarantee.
        random = np.zeros(len(frame))
        for day in np.unique(day_index):
            rows = np.flatnonzero(day_index == day)
            random[rows] = policies[f"fixed/{method}"][rows].mean()
        policies[f"fixed_random_expectation/{method}"] = random
        policies[f"fixed_random_global_expectation/{method}"] = np.full(len(frame), policies[f"fixed/{method}"].mean())
    for budget in (0.10, 0.20, 0.30):
        for method in METHODS:
            policies[f"daily_{budget:.2f}/{method}"] = daily_budget(probabilities[method], day_index, budget)
        random = np.zeros(len(frame))
        for day in np.unique(day_index):
            rows = np.flatnonzero(day_index == day)
            random[rows] = math.ceil(budget * len(rows)) / len(rows)
        policies[f"daily_{budget:.2f}/random_review_expectation"] = random
    policy_names = list(policies)
    day_counts = np.stack([aggregate_days(flagged, target, magnitude, day_index, len(calendar)) for flagged in policies.values()], axis=1)
    counts = day_counts.sum(axis=0)
    bootstrap_counts = np.einsum("bd,dpc->bpc", weights, day_counts, optimize=True)
    records = {name: metrics_record(counts[index], bootstrap_counts[:, index]) for index, name in enumerate(policy_names)}
    for method in METHODS:
        record = records[f"fixed/{method}"]
        frozen = report["methods"][method]["decision_at_calibration_80pct_coverage"]
        assert record["escalated"] == frozen["flagged_rows"]
        for measured, original in (("escalation_fraction", "flag_rate"), ("tail_recall", "tail_recall"), ("escalation_precision", "tail_precision"), ("accepted_tail_rate", "accepted_tail_rate")):
            assert np.isclose(record["metrics"][measured]["estimate"], frozen[original], rtol=0, atol=1e-12)
        expected_brier = original_audit["registered_ablation_secondary_classifier"]["overall"].get(method, original_audit["overall"].get(method))["brier"]
        assert np.isclose(np.mean((probabilities[method] - target) ** 2), expected_brier, rtol=0, atol=1e-12)
    # Daily budget methods have exactly equal observed workloads, before labels.
    for budget in (0.10, 0.20, 0.30):
        assert len({records[f"daily_{budget:.2f}/{method}"]["escalated"] for method in METHODS}) == 1
    assert np.all(counts[:, 3] <= counts[:, 2]) and np.all(counts[:, 1] <= counts[:, 0])
    assert np.all(counts[:, 3] <= counts[:, 1] + 1e-9)
    budget_refinement = {
        "scope": "Post-confirmation exploratory audit of daily score ranking under the retained held-out nonuniform mass refinement; original physical timestamps provide stable deterministic tie-breaks. No targets are used for selection.",
        "full_qct_refined_probability_available": "probability_refined_qct_full" in frame.columns,
        "full_qct_fixed_rule_registered_contract": report["invariance_and_sensitivity"],
        "methods": {},
    }
    for method in ("qct_full", "qct_no_tangent", "uniform_set_transformer"):
        refined_column = f"probability_refined_{method}"
        if refined_column not in frame.columns:
            budget_refinement["methods"][method] = {"status": "not_evaluated_cached_refined_probabilities_absent", "new_inference": False}
            continue
        refined_probability = frame[refined_column].to_numpy(dtype=float)
        original_probability = probabilities[method]
        assert np.all(np.isfinite(refined_probability)) and np.all((refined_probability >= 0) & (refined_probability <= 1))
        fixed_before = policies[f"fixed/{method}"]
        fixed_after = (refined_probability > thresholds[method]).astype(float)
        method_refinement = {
            "status": "cached_probabilities_evaluated",
            "scope": "registered secondary ablation" if method == "qct_no_tangent" else "ordinary attention control",
            "max_probability_drift": float(np.max(np.abs(refined_probability - original_probability))),
            "fixed_rule_membership_flips": int(np.sum(fixed_before != fixed_after)),
            "daily_budget": {},
        }
        for budget in (0.10, 0.20, 0.30):
            before = policies[f"daily_{budget:.2f}/{method}"]
            after = daily_budget(refined_probability, day_index, budget)
            before_daily = aggregate_days(before, target, magnitude, day_index, len(calendar))
            after_daily = aggregate_days(after, target, magnitude, day_index, len(calendar))
            assert np.array_equal(before_daily[:, 1], after_daily[:, 1])
            counts_before = before_daily.sum(axis=0)
            counts_after = after_daily.sum(axis=0)
            bootstrap_before = weights @ before_daily
            bootstrap_after = weights @ after_daily
            metrics_before = ratios(counts_before)
            metrics_after = ratios(counts_after)
            boot_before = ratios(bootstrap_before)
            boot_after = ratios(bootstrap_after)
            margins = []
            per_day_certified = []
            changed_days = 0
            for day in np.unique(day_index):
                rows = np.flatnonzero(day_index == day)
                k = math.ceil(budget * len(rows))
                ordered = np.sort(original_probability[rows])[::-1]
                margin = float(ordered[k - 1] - ordered[k])
                drift = float(np.max(np.abs(refined_probability[rows] - original_probability[rows])))
                margins.append(margin)
                per_day_certified.append(margin > 2 * drift)
                changed_days += int(np.any(before[rows] != after[rows]))
                if margin > 2 * drift:
                    assert np.array_equal(before[rows], after[rows])
            method_refinement["daily_budget"][str(budget)] = {
                "budget_fraction": budget,
                "original_escalated": int(before.sum()), "refined_escalated": int(after.sum()),
                "equal_workload_each_day": True,
                "selected_set_membership_flips": int(np.sum(before != after)),
                "membership_flip_fraction": float(np.mean(before != after)),
                "newly_escalated": int(np.sum((before == 0) & (after == 1))),
                "newly_accepted": int(np.sum((before == 1) & (after == 0))),
                "days_with_membership_change": changed_days,
                "tail_event_membership_flips": int(np.sum((before != after) & (target == 1))),
                "original_caught_tail_events": int(np.dot(before, target)),
                "refined_caught_tail_events": int(np.dot(after, target)),
                "original_missed_tail_events": int(np.dot(1 - before, target)),
                "refined_missed_tail_events": int(np.dot(1 - after, target)),
                "minimum_original_daily_selection_margin": float(min(margins)),
                "days_with_margin_exceeding_twice_drift": int(sum(per_day_certified)),
                "calendar_days": len(calendar),
                "refined_minus_original": {metric: {"estimate": float(metrics_after[metric] - metrics_before[metric]), "calendar_7day_95ci": interval(boot_after[metric] - boot_before[metric])} for metric in METRICS},
            }
        budget_refinement["methods"][method] = method_refinement
    bootstrap_metrics = ratios(bootstrap_counts)
    point_metrics = ratios(counts)
    comparisons = {}
    for prefix in ("fixed", "daily_0.10", "daily_0.20", "daily_0.30"):
        for candidate in ("qct_full", "qct_no_tangent"):
            for baseline in ("hgb_quotient", "uniform_set_transformer", "hgb_calendar"):
                ci = policy_names.index(f"{prefix}/{candidate}")
                bi = policy_names.index(f"{prefix}/{baseline}")
                comparisons[f"{prefix}/{candidate}_minus_{baseline}"] = {
                    metric: {"estimate": float(point_metrics[metric][ci] - point_metrics[metric][bi]), "calendar_7day_95ci": interval(bootstrap_metrics[metric][:, ci] - bootstrap_metrics[metric][:, bi])} for metric in METRICS
                }
        for candidate in ("qct_full", "qct_no_tangent"):
            baseline_policies = (f"fixed_random_expectation/{candidate}", f"fixed_random_global_expectation/{candidate}", "all_accept") if prefix == "fixed" else (f"{prefix}/random_review_expectation",)
            ci = policy_names.index(f"{prefix}/{candidate}")
            for baseline_policy in baseline_policies:
                bi = policy_names.index(baseline_policy)
                comparisons[f"{prefix}/{candidate}_minus_{baseline_policy}"] = {
                    metric: {"estimate": float(point_metrics[metric][ci] - point_metrics[metric][bi]) if np.isfinite(point_metrics[metric][ci] - point_metrics[metric][bi]) else None, "calendar_7day_95ci": interval(bootstrap_metrics[metric][:, ci] - bootstrap_metrics[metric][:, bi])} for metric in METRICS
                }
    cost_ratios = (1, 2, 5, 10, 20, 50, 100)
    costs = {}
    fixed_cost_comparisons = {}
    for name in ("all_accept", "all_escalate", *(f"fixed/{method}" for method in METHODS)):
        index = policy_names.index(name)
        costs[name] = {}
        for penalty in cost_ratios:
            point = (counts[index, 1] + penalty * (counts[index, 2] - counts[index, 3])) / counts[index, 0]
            boot = (bootstrap_counts[:, index, 1] + penalty * (bootstrap_counts[:, index, 2] - bootstrap_counts[:, index, 3])) / bootstrap_counts[:, index, 0]
            costs[name][str(penalty)] = {"normalized_cost": float(point), "calendar_7day_95ci": interval(boot)}
    for baseline in ("hgb_quotient", "uniform_set_transformer", "hgb_calendar"):
        ci = policy_names.index("fixed/qct_full")
        bi = policy_names.index(f"fixed/{baseline}")
        fixed_cost_comparisons[baseline] = {}
        for penalty in cost_ratios:
            difference = (counts[ci, 1] - counts[bi, 1] + penalty * (counts[bi, 3] - counts[ci, 3])) / len(frame)
            boot = (bootstrap_counts[:, ci, 1] - bootstrap_counts[:, bi, 1] + penalty * (bootstrap_counts[:, bi, 3] - bootstrap_counts[:, ci, 3])) / bootstrap_counts[:, ci, 0]
            fixed_cost_comparisons[baseline][str(penalty)] = {"qct_full_minus_baseline_normalized_cost": float(difference), "calendar_7day_95ci": interval(boot)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete_post_confirmation_exploratory_audit",
        "analysis_date": "2026-10-05",
        "workflow": "Hypothetical post-clearing review when verified prices are missing, delayed or quarantined; certificate-only scores decide escalation; verified prices reveal residual labels later. Complete verified prices would allow direct residual calculation.",
        "scientific_boundary": {
            "new_estimands_are_post_confirmation_exploratory": True,
            "new_confirmation_set": False, "retraining": False, "recalibration": False,
            "threshold_selection_on_confirmation_targets": False,
            "qct_full_remains_registered_primary": True,
            "qct_no_tangent_is_registered_secondary_not_replacement": True,
            "outages_are_hypothetical_not_observed": True,
            "no_observed_monetary_or_welfare_value": True,
            "outputs_private_not_in_public_release": True,
            "multiple_exploratory_intervals_are_not_multiplicity_adjusted": True,
        },
        "provenance": {"sources": source_checks, "provider_sources_and_geometry": raw_source_checks, "frozen_manifest": str(manifest_path), "frozen_manifest_sha256": sha256(manifest_path), "audit_engine": str(Path(__file__).resolve()), "audit_engine_sha256": sha256(Path(__file__).resolve())},
        "sample": {"rows": len(frame), "tail_events": int(target.sum()), "prevalence": float(target.mean()), "calendar_days": len(calendar), "observed_days": int(len(np.unique(day_index))), "minimum_observations_per_observed_day": int(pd.Series(day_index).value_counts().min()), "maximum_observations_per_day": int(pd.Series(day_index).value_counts().max()), "first_timestamp": timestamps.min().isoformat(), "last_timestamp": timestamps.max().isoformat(), "tail_threshold_eur_mwh": report["protocol"]["tail_thresholds_eur_mwh"][1]},
        "calibration": {"fit_end_exclusive": report["protocol"]["fit_end_exclusive"], "tuning_end_exclusive": report["protocol"]["tuning_end_exclusive"], "calibration_end_exclusive": report["protocol"]["calibration_end_exclusive"], "fixed_probability_thresholds": thresholds, "fixed_rule": "escalate iff probability > the method-specific frozen December 80th percentile; realized March-August workload need not equal20%"},
        "uncertainty": {"replicates": args.replicates, "seed": args.seed, "method": "paired circular moving blocks of seven contiguous UTC calendar days; all methods share calendar-resampling weights; ratios recomputed in every replicate; missing dates remain calendar gaps", "random_review": "conditional expectations, not favorable realized draws; fixed_random_expectation matches each method's realized within-day workload and therefore inherits that daily allocation; fixed_random_global_expectation uses a constant post-hoc workload fraction matched over the full sample; daily budget random review has independently specified ceil(budget*n_day) workload. Intervals reflect calendar sampling only, not additional random-policy realization variance."},
        "daily_budget_rule": "For each UTC calendar day, escalate ceil(budget * available certificate count) highest-score timestamps; descending scores and chronological stable tie-break; batches use all scores in that day and no outcomes. Retrospective batch-review policy, not causal real-time deployment or an observed outage.",
        "policies": records,
        "paired_candidate_minus_baseline": comparisons,
        "daily_budget_refinement": budget_refinement,
        "cost_definition": "illustrative unit review cost1 plus missed-tail cost ratio rho; no cost for correct acceptance; escalation is assumed to permit follow-up review, not guarantee correction; ratios have no empirical monetary calibration",
        "fixed_policy_cost_sensitivity": costs,
        "fixed_qct_full_minus_baseline_cost_sensitivity": fixed_cost_comparisons,
        "simple_structure_control": {"features": ["any positive-mass atom present", "log1p(projected total dual mass)", "mean absolute analytic pairwise spread", "maximum absolute analytic pairwise spread"], "frozen_brier": report["methods"]["hgb_simple_structure"]["classification"]["brier"], "frozen_fixed_decision": report["methods"]["hgb_simple_structure"]["decision_at_calibration_80pct_coverage"], "per_timestamp_scores_in_retained_archive": False, "new_paired_intervals_not_computed": True, "analytic_field_only_risk_control_exists": False, "interpretation": "Includes analytic field scale plus dual-mass/presence; this is not a strictly analytic-field-only calibrated control. Analytic reconstruction residual cannot itself be a deployable escalation score while verified prices are withheld."},
        "checks": {"frozen_report_audit_predictions_hashes_verified": True, "provider_source_geometry_hashes_verified": len(raw_source_checks), "chronology_unique_labels_probabilities_valid": True, "all_frozen_count_rate_brier_replications_passed": True, "daily_workload_matched_exactly": True, "refinement_daily_workloads_and_rank_margin_guarantees_verified": True, "count_identities_passed": True, "synthetic_checks_passed": True},
    }
    report_out = output / "operational_triage_audit.json"
    report_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_rows = []
    for name, record in records.items():
        row = {"policy": name, **{key: value for key, value in record.items() if key != "metrics"}}
        for metric, values in record["metrics"].items():
            row[metric] = values["estimate"]
            row[metric + "_ci95_low"], row[metric + "_ci95_high"] = values["calendar_7day_95ci"]
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(output / "policy_metrics.csv", index=False)
    summary = ["# Exploratory operational triage audit", "", result["workflow"], "", "These are post-confirmation exploratory analyses of the existing frozen 14,482 timestamps, not an independent confirmation, an observed outage or deployed economic value. Full QCT remains primary; no-tangent is secondary. No fitting or recalibration occurred.", "", "| Policy | Reviewed | Tail events caught | Tail events missed | Accepted tail rate |", "|---|---:|---:|---:|---:|"]
    for name in ("all_accept", *(f"fixed/{m}" for m in METHODS), "fixed_random_expectation/qct_full", "fixed_random_global_expectation/qct_full", *(f"daily_0.20/{m}" for m in METHODS), "daily_0.20/random_review_expectation"):
        record = records[name]
        rate = record["metrics"]["accepted_tail_rate"]["estimate"]
        summary.append(f"| {name} | {record['escalated']:.2f} ({100*record['metrics']['escalation_fraction']['estimate']:.2f}%) | {record['caught_tail_events']:.2f} | {record['missed_tail_events']:.2f} | {100*rate:.3f}% |")
    summary.extend(["", "## Full QCT minus invariant HGB", "", "Negative retained-risk/missed-event differences favor QCT; fixed-rule workload differs. Daily-budget workload is matched exactly."])
    for prefix in ("fixed", "daily_0.10", "daily_0.20", "daily_0.30"):
        comparison = comparisons[f"{prefix}/qct_full_minus_hgb_quotient"]
        for metric in ("accepted_tail_rate", "missed_tail_fraction", "escalation_fraction"):
            estimate = comparison[metric]["estimate"]
            low, high = comparison[metric]["calendar_7day_95ci"]
            summary.append(f"- {prefix}, {metric}: {100*estimate:+.3f} percentage points, paired seven-calendar-day95% interval [{100*low:+.3f}, {100*high:+.3f}].")
    summary.extend(["", "## Held-out refinement of the daily budget policy", "", "The original physical timestamp supplies the stable tie-break. Cached refined full-QCT probabilities are absent, so its daily-budget membership was not inferred or evaluated. The registered full-QCT fixed-rule result remains authoritative.", "", "| Method | Daily budget | Membership flips | Tail membership flips | Caught tails, original/refined | Daily workload equal |", "|---|---:|---:|---:|---:|---|"])
    for method in ("qct_no_tangent", "uniform_set_transformer"):
        for budget, values in budget_refinement["methods"][method]["daily_budget"].items():
            summary.append(f"| {method} | {100*float(budget):.0f}% | {values['selected_set_membership_flips']} | {values['tail_event_membership_flips']} | {values['original_caught_tail_events']}/{values['refined_caught_tail_events']} | Yes |")
    summary.extend(["", "## Limits", "", "- Frozen December thresholds are reproduced, but thresholds targeting 20% review in December produce different confirmation workloads.", "- New matched-budget and cost estimands are exploratory; no confirmatory model-selection or superiority claim follows.", "- Random-review counts are conditional expectations, not realized integer counts; intervals omit random-policy realization variance. Fixed daily-matched random review inherits QCT's daily workload allocation; fixed global random uses a post-hoc matched constant fraction. The daily budget random baseline has an independently specified workload.", "- Verified same-delivery prices are withheld hypothetically. No natural outage, missingness mechanism, human review benefit, welfare impact or monetary cost is measured.", "- Analytic residual is an evaluation outcome, not an admissible score when the price feed is unavailable.", "- Simple-structure HGB already uses analytic-field mean/max spread plus mass/presence; per-timestamp scores were not retained, so only historical aggregate results are reported.", "- All generated outputs are private; no timestamp predictions or provider records are copied into the public package."])
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "sample": result["sample"], "fixed_qct_full": records["fixed/qct_full"], "daily20_qct_full_minus_hgb": comparisons["daily_0.20/qct_full_minus_hgb_quotient"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
