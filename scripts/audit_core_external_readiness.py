"""Verify retained Core v1 evidence without training or reading market rows.

Aggregate arithmetic and metadata links are checked by default. With
--verify_local_data, derived-file hashes are also checked by streaming bytes.
Historical engine hashes and timestamp losses were not retained by v1; this
audit reports those limits instead of retrospectively inventing provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202)
METHODS = (
    "analytic_gauge",
    "ambient_raw_deepset",
    "projection_only_raw_deepset",
    "ambient_cqdm_residual",
    "core_rank1_af_qdm_residual",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: Any, expected: Any) -> bool:
    return bool(np.allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11))


def seed_ci(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [values[0], values[0]]
    rng = np.random.default_rng(20261003)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def audit_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the full completed v1 confirmation, not a smoke report."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("benchmark_protocol", report.get("protocol") == "af_cdmo_core_external_replication_v1", "Core v1")
    config = report["frozen_configuration"]
    check("full_confirmation_configuration", (
        tuple(config["methods"]) == METHODS
        and tuple(config["seeds"]) == SEEDS
        and config["batch_size"] == 128
        and config["epochs"] == 60
        and config["invariance_seed"] == 7
        and config["alert_threshold_eur_mwh"] == 5.0
        and config["smoke_presentations"] is False
    ), "ten seeds, batch 128, at most 60 epochs, seed-7 stress, 5 EUR/MWh alerts")
    presentations = config["presentations"]
    check("presentation_registry", len(presentations) == len(set(presentations)) == config["presentation_count"] == 56 and "original" in presentations, "56 distinct registered presentations")
    data = report["data"]
    check("dimensions", len(data["physical_zones"]) == 12 and len(data["all_hub_coordinates"]) == 23, "12 physical zones, 66 pairs, 23 hubs")
    check("chronology", data["split"] == {
        "fit_end_utc": "2025-04-01T00:00:00+00:00",
        "validation_end_utc": "2025-07-01T00:00:00+00:00",
        "evaluation_end_utc": "2025-10-01T00:00:00+00:00",
        "rows": {"fit": 4036, "validation": 2118, "evaluation": 2071},
    }, "4036/2118/2071 chronologically split timestamps")

    expected_records = {
        (method, seed, presentation)
        for method in METHODS
        for seed in SEEDS
        if method != "analytic_gauge" or seed == 7
        for presentation in (presentations if seed == 7 else ["original"])
    }
    record_keys = [(item["method"], item["seed"], item["presentation"]) for item in report["records"]]
    check("record_allocation", len(record_keys) == len(set(record_keys)) and set(record_keys) == expected_records, f"{len(record_keys)} records; full registry only on seed 7")
    training_keys = [(item["method"], item["seed"]) for item in report["training"]]
    expected_training = {(method, seed) for method in METHODS for seed in SEEDS if method != "analytic_gauge" or seed == 7}
    check("training_allocation", len(training_keys) == len(set(training_keys)) and set(training_keys) == expected_training, "40 stochastic fits and one deterministic analytic control")
    expected_parameters = {method: (0 if method == "analytic_gauge" else 5180 if "raw_deepset" in method else 18060) for method in METHODS}
    check("model_parameter_counts", all(item["parameter_count"] == expected_parameters[item["method"]] for item in report["training"]), "raw 5180; attention 18060; unequal heads and analytic assistance")

    for method in METHODS:
        records = [item for item in report["records"] if item["method"] == method]
        clean = sorted((item for item in records if item["presentation"] == "original"), key=lambda item: SEEDS.index(item["seed"]))
        values = [item["metrics"]["pairwise_mae_eur_mwh"] for item in clean]
        summary = report["summary"]["clean"][method]
        reconstructed = {
            "seeds": len(values),
            "pairwise_mae_mean_eur_mwh": float(np.mean(values)),
            "pairwise_mae_std_eur_mwh": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "pairwise_mae_seedwise_eur_mwh": values,
            "pairwise_mae_bootstrap_95_ci_eur_mwh": seed_ci(values),
            "pairwise_rmse_mean_eur_mwh": float(np.mean([item["metrics"]["pairwise_rmse_eur_mwh"] for item in clean])),
            "alert_flip_rate_mean": float(np.mean([item["metrics"]["alert_flip_rate"] for item in clean])),
        }
        check(f"clean_summary__{method}", all(close(summary[name], value) for name, value in reconstructed.items()), "recomputed means, sample SD, seed CI and classification disagreement")
        invariance = report["summary"]["presentation_invariance"][method]
        drift = max(item["drift"]["presentation_drift_max_eur_mwh"] for item in records)
        flips = max(item["drift"]["presentation_alert_flip_rate"] for item in records)
        check(f"invariance_summary__{method}", close(invariance["maximum_presentation_drift_eur_mwh"], drift) and close(invariance["maximum_presentation_alert_flip_rate"], flips), "recomputed maximal pairwise drift and presentation-alert flip rate")
    contract = report["structural_contract"]
    invariant_pass = all(report["summary"]["presentation_invariance"][method]["maximum_presentation_drift_eur_mwh"] <= 1.0e-3 for method in ("analytic_gauge", "core_rank1_af_qdm_residual"))
    cycle_pass = all(item["metrics"]["cycle_inconsistency_max_eur_mwh"] <= 2.0e-5 for item in report["records"])
    check("structural_contract", invariant_pass and cycle_pass and contract["passed"] is True and not contract["invariance_failures"] and not contract["cycle_failures"] and contract["performance_used_as_pass_fail_criterion"] is False, "existing declared drift/cycle tolerances; performance does not determine pass")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=Path("external_core_replication/core_external_confirmation_v1_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("results/core_external_readiness_audit.json"))
    parser.add_argument("--verify_local_data", action="store_true", help="Also hash derived market data; bytes only, no row extraction.")
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = args.report if args.report.is_absolute() else root / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = audit_report(report)

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # Resolve adjacent metadata first so local absolute paths in historical
    # reports do not make copies of the research folder unusable.
    manifest_path = report_path.parent / Path(report["source_manifest"].replace("\\", "/")).name
    geometry_path = report_path.parent / Path(report["geometry_audit"].replace("\\", "/")).name
    check("manifest_hash", manifest_path.is_file() and sha256_file(manifest_path) == report["source_manifest_sha256"], "retained manifest hash matches benchmark")
    check("geometry_hash", geometry_path.is_file() and sha256_file(geometry_path) == report["geometry_audit_sha256"], "retained geometry audit hash matches benchmark")
    if manifest_path.is_file() and geometry_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        check("metadata_links", manifest["protocol"] == "core_ccr_external_replication_v1" and geometry["passed"] is True and geometry["registered_geometry"]["rank"] == 1 and manifest["artifacts"]["dual"]["sha256"] == report["data"]["dual_sha256"] == geometry["dual_artifact_sha256"] and manifest["artifacts"]["centered_price_field"]["sha256"] == report["data"]["price_sha256"], "rank-1 passed geometry and consistent recorded derived-data hashes")
        check("causal_metadata", manifest["dual_audit"]["causal_publication_share"] == 1.0 and manifest["causal_revision_audit"]["excluded_delivery_interval_count"] == 541, "retained metadata reports causal rows and 541 excluded intervals; not a new raw-response audit")
        if args.verify_local_data:
            for name, entry in manifest["artifacts"].items():
                path = Path(entry["path"])
                if not path.is_file():
                    basename = Path(entry["path"].replace("\\", "/")).name
                    path = manifest_path.parent / ("data" if name in {"dual", "centered_price_field"} else "") / basename
                check(f"local_artifact_hash__{name}", path.is_file() and sha256_file(path) == entry["sha256"], "streamed file hash matches retained manifest")

    output = {
        "schema_version": 1,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "review_type": "aggregate_arithmetic_and_retained_artifact_links",
        "benchmark_sha256": sha256_file(report_path),
        "local_data_hashes_verified": args.verify_local_data,
        "present_source_inventory": {
            name: sha256_file(root / "scripts" / name)
            for name in (
                "benchmark_af_cdmo_core_external.py",
                "core_external_replication.py",
                "download_jao_core_external_replication.py",
                "audit_af_cdmo_core_geometry.py",
                "af_cdmo_real_extension_v13.py",
                "multizone_quotient_gauge_v6.py",
                "summarize_af_cdmo_core_external.py",
            )
            if (root / "scripts" / name).is_file()
        },
        "checks": checks,
        "limitations": [
            "No historical Core source hashes retained; current source is not proof of historical engine identity.",
            "No Core checkpoint or timestamp predictions/losses retained; no temporal-block uncertainty can be reconstructed from aggregate scores.",
            "Clean intervals resample seed-level scores conditional on the dataset; full presentation registry is seed 7 only.",
            "Raw models directly predict prices while quotient models add analytic residuals and have different heads; raw-vs-AF accuracy is not a matched quotient ablation.",
        ],
        "metric_definitions": {
            "clean_alert_flip_rate": "predicted-vs-observed absolute pairwise spread >=5 EUR/MWh disagreement, averaged over timestamp-pair cells",
            "presentation_alert_flip_rate": "original-vs-rewritten absolute predicted pairwise spread >=5 EUR/MWh disagreement, averaged over timestamp-pair cells",
            "uncertainty": "20,000 bootstrap resamples of seed-level MAE with RNG seed 20261003",
        },
    }
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['detail']}")
    print(f"[REPORT] {output_path}")
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
