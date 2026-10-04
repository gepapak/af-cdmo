"""Evidence-integrity checks use synthetic aggregate records, never market data."""

from __future__ import annotations

import copy

import pytest

from scripts.audit_core_external_readiness import METHODS, SEEDS, audit_report
from scripts.benchmark_af_cdmo_core_external import _summarize


@pytest.fixture
def report() -> dict:
    presentations = ["original", *[f"synthetic_rewrite_{index}" for index in range(55)]]
    records = []
    training = []
    for method in METHODS:
        for seed_index, seed in enumerate(SEEDS):
            if method == "analytic_gauge" and seed != 7:
                continue
            training.append({
                "method": method,
                "seed": seed,
                "parameter_count": 0 if method == "analytic_gauge" else 5180 if "raw_deepset" in method else 18060,
            })
            for presentation in (presentations if seed == 7 else ["original"]):
                records.append({
                    "method": method,
                    "seed": seed,
                    "presentation": presentation,
                    "metrics": {
                        "pairwise_mae_eur_mwh": 3.0 + seed_index / 10,
                        "pairwise_rmse_eur_mwh": 5.0 + seed_index / 10,
                        "alert_flip_rate": seed_index / 100,
                        "cycle_inconsistency_max_eur_mwh": 0.0,
                    },
                    "drift": {
                        "presentation_drift_max_eur_mwh": 0.0,
                        "presentation_alert_flip_rate": 0.0,
                    },
                })
    return {
        "protocol": "af_cdmo_core_external_replication_v1",
        "frozen_configuration": {
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "batch_size": 128,
            "epochs": 60,
            "invariance_seed": 7,
            "alert_threshold_eur_mwh": 5.0,
            "smoke_presentations": False,
            "presentations": presentations,
            "presentation_count": 56,
        },
        "data": {
            "physical_zones": [f"zone_{index}" for index in range(12)],
            "all_hub_coordinates": [f"hub_{index}" for index in range(23)],
            "split": {
                "fit_end_utc": "2025-04-01T00:00:00+00:00",
                "validation_end_utc": "2025-07-01T00:00:00+00:00",
                "evaluation_end_utc": "2025-10-01T00:00:00+00:00",
                "rows": {"fit": 4036, "validation": 2118, "evaluation": 2071},
            },
        },
        "records": records,
        "training": training,
        "summary": _summarize(records),
        "structural_contract": {
            "passed": True,
            "invariance_failures": {},
            "cycle_failures": {},
            "performance_used_as_pass_fail_criterion": False,
        },
    }


def failed_names(report: dict) -> set[str]:
    return {item["name"] for item in audit_report(report) if not item["passed"]}


def test_runner_aggregate_records_pass_independent_arithmetic(report: dict) -> None:
    assert not failed_names(report)


def test_corrupted_seed_interval_is_rejected(report: dict) -> None:
    report["summary"]["clean"]["core_rank1_af_qdm_residual"]["pairwise_mae_bootstrap_95_ci_eur_mwh"][0] += 0.01
    assert "clean_summary__core_rank1_af_qdm_residual" in failed_names(report)


def test_duplicate_record_is_rejected_even_with_unchanged_maxima(report: dict) -> None:
    report["records"].append(copy.deepcopy(report["records"][0]))
    assert "record_allocation" in failed_names(report)


def test_seed_only_stress_allocation_cannot_be_relabelled(report: dict) -> None:
    record = copy.deepcopy(next(item for item in report["records"] if item["method"] == "core_rank1_af_qdm_residual" and item["presentation"] != "original"))
    record["seed"] = 42
    report["records"].append(record)
    assert "record_allocation" in failed_names(report)


def test_structural_pass_cannot_hide_recomputed_drift_failure(report: dict) -> None:
    next(item for item in report["records"] if item["method"] == "core_rank1_af_qdm_residual" and item["presentation"] != "original")["drift"]["presentation_drift_max_eur_mwh"] = 1.1e-3
    report["summary"] = _summarize(report["records"])
    assert "structural_contract" in failed_names(report)


def test_alert_disagreement_summary_is_verified(report: dict) -> None:
    report["summary"]["clean"]["ambient_raw_deepset"]["alert_flip_rate_mean"] = 0.0
    assert "clean_summary__ambient_raw_deepset" in failed_names(report)
