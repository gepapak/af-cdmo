"""Fail-closed artifact audit for the AF-CDMO v14 confirmation controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.af_cdmo_real_extension_v13 import (
        RealAffineGeometry,
        presentation_names,
    )
    from scripts.benchmark_af_cdmo_real_v14 import (
        ALL_METHODS,
        ENGINE_PATHS,
        GAUGE_AMPLITUDES,
        NEURAL_METHODS,
        PRIMARY_NORDIC_ZONES,
        RAM_PERTURBATIONS,
    )
    from scripts.cqdm_utils import sha256_file
except ModuleNotFoundError:
    from af_cdmo_real_extension_v13 import RealAffineGeometry, presentation_names
    from benchmark_af_cdmo_real_v14 import (
        ALL_METHODS,
        ENGINE_PATHS,
        GAUGE_AMPLITUDES,
        NEURAL_METHODS,
        PRIMARY_NORDIC_ZONES,
        RAM_PERTURBATIONS,
    )
    from cqdm_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "official_jao_dual_measure" / "af_cdmo_real_development_v14.json"
)
DEFAULT_MANIFEST = ROOT / "af_cdmo_real_development_v14_manifest.json"
EXPECTED_SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--required_role",
        choices=("development", "frozen_transition", "confirmation"),
        default="development",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; use --force")
    _require(report_path.is_file(), f"Missing AF-CDMO v14 report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    _require(
        protocol.get("name") == "af_cdmo_real_loss_safe_v14",
        "Unexpected protocol name",
    )
    _require(protocol.get("study_role") == args.required_role, "Study-role mismatch")
    _require(protocol.get("not_a_forecast") is True, "Task boundary is missing")
    _require(tuple(protocol.get("zones", ())) == PRIMARY_NORDIC_ZONES, "Zone-panel drift")
    _require(protocol.get("seeds") == EXPECTED_SEEDS, "Seed schedule drift")
    _require(protocol.get("skagerrak_excluded") is True, "Skagerrak must be excluded")
    _require(
        float(protocol.get("all_zone_price_panel_coverage", 0.0))
        >= float(protocol.get("minimum_required_price_panel_coverage", 1.0)),
        "Price-panel coverage contract failed",
    )

    source = report["source"]
    dual_path = Path(source["dual_archive"]).expanduser().resolve()
    geometry_path = Path(source["geometry_audit"]).expanduser().resolve()
    _require(dual_path.is_file() and geometry_path.is_file(), "Source artifact missing")
    _require(
        sha256_file(dual_path) == source["dual_archive_sha256"],
        "Dual-archive hash drift",
    )
    _require(
        sha256_file(geometry_path) == source["geometry_audit_sha256"],
        "Geometry-audit hash drift",
    )
    price_sources = source.get("day_ahead_prices", {})
    _require(
        set(price_sources) == set(PRIMARY_NORDIC_ZONES),
        "Day-ahead price-source registry drift",
    )
    verified_price_sources: list[dict[str, Any]] = []
    for zone in PRIMARY_NORDIC_ZONES:
        record = price_sources[zone]
        price_path = Path(record["path"]).expanduser().resolve()
        _require(price_path.is_file(), f"Missing {zone} day-ahead price source")
        price_hash = sha256_file(price_path)
        _require(
            price_hash == record["sha256"],
            f"Day-ahead price-source hash drift for {zone}",
        )
        verified_price_sources.append(
            {"zone": zone, "path": str(price_path), "sha256": price_hash}
        )
    geometry_report = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry = RealAffineGeometry.loss_safe(geometry_report["geometry"]["zone_order"])
    expected_presentations = list(presentation_names(geometry))
    _require(
        protocol.get("presentations") == expected_presentations,
        "Presentation registry drift",
    )

    _require(
        report.get("frozen_v11_reproduction_contract", {}).get("status") == "passed",
        "Frozen v11 reproduction contract failed",
    )
    invariance = report["invariance_contract"]
    _require(invariance.get("status") == "passed", "Prediction invariance failed")
    tolerance = float(invariance["tolerance_eur_mwh"])
    for method in ("af_qdm_residual", "analytic_gauge"):
        _require(
            float(invariance["method_max_abs_prediction_drift_eur_mwh"][method])
            <= tolerance,
            f"Exact prediction drift exceeded tolerance for {method}",
        )
    _require(
        float(
            invariance["method_affine_only_max_abs_prediction_drift_eur_mwh"][
                "af_projected_raw_residual"
            ]
        )
        <= tolerance,
        "Projection-only affine subgroup contract failed",
    )
    _require(
        float(
            invariance["method_affine_only_max_abs_prediction_drift_eur_mwh"][
                "af_uniform_attention_residual"
            ]
        )
        <= tolerance,
        "Uniform-mass affine subgroup contract failed",
    )
    analytic_drift = invariance["analytic_pair_drift_eur_mwh"]
    _require(
        float(analytic_drift["maximum"]) <= float(analytic_drift["tolerance"]),
        "Analytic tangent-query conservation failed",
    )

    results = report["results"]
    _require(tuple(results) == ALL_METHODS, "Method registry drift")
    for method in ALL_METHODS:
        _require(
            list(results[method]["stresses"]) == expected_presentations,
            f"Stress registry drift for {method}",
        )
        if method in NEURAL_METHODS:
            for stress in expected_presentations:
                members = results[method]["stresses"][stress]["seed_members"]
                _require(
                    [member["seed"] for member in members] == EXPECTED_SEEDS,
                    f"Seed-member registry drift for {method}/{stress}",
                )

    comparisons = report["paired_comparisons"]
    for baseline in (*NEURAL_METHODS.keys(), "analytic_gauge"):
        if baseline == "af_qdm_residual":
            continue
        for estimand in ("original", "orbit_mean", "adversarial_envelope", "worst_fixed"):
            _require(
                f"{estimand}::{baseline}_to_af_qdm_residual" in comparisons,
                f"Missing {estimand} comparison for {baseline}",
            )
        if baseline in NEURAL_METHODS:
            _require(
                f"seed_level::{baseline}_to_af_qdm_residual" in comparisons,
                f"Missing seed-level comparison for {baseline}",
            )

    fairness = report.get("fairness_contract", {})
    for baseline, control in (
        ("af_qdm_residual", "af_uniform_attention_residual"),
        ("ambient_cqdm_residual", "ambient_gauge_augmented_residual"),
    ):
        record = fairness.get(f"{baseline}_to_{control}")
        _require(record is not None, f"Missing fairness contract for {control}")
        _require(
            len(set(record["parameter_count_by_seed"])) == 1,
            f"Parameter count changed across seeds for {control}",
        )
        _require(
            len(record["actual_best_epochs_baseline"]) == len(EXPECTED_SEEDS)
            and len(record["actual_best_epochs_control"]) == len(EXPECTED_SEEDS),
            f"Best-epoch registry drift for {control}",
        )
        _require(
            record.get("realized_update_counts_forced_equal") is False,
            f"Realized-update disclosure missing for {control}",
        )

    refinement = report.get("mass_refinement_contract", {})
    _require(
        refinement.get("training_exposure") is False,
        "Nonuniform refinement must remain held out from training",
    )
    _require(
        float(refinement.get("maximum_conserved_moment_drift", np.inf))
        <= tolerance,
        "Nonuniform refinement did not conserve the AF measure moment",
    )
    _require(
        float(refinement.get("maximum_total_mass_drift", np.inf)) <= tolerance,
        "Nonuniform refinement did not conserve total dual mass",
    )
    _require(
        float(
            refinement["methods"]["af_qdm_residual"][
                "max_abs_pairwise_prediction_drift_eur_mwh"
            ]
        )
        <= tolerance,
        "AF-QDM failed held-out nonuniform-refinement invariance",
    )

    gauge_sensitivity = report.get("gauge_amplitude_sensitivity", {})
    _require(
        gauge_sensitivity.get("amplitudes") == list(GAUGE_AMPLITUDES),
        "Gauge-amplitude registry drift",
    )
    _require(
        float(gauge_sensitivity.get("af_qdm_maximum_drift_eur_mwh", np.inf))
        <= tolerance,
        "AF-QDM failed the registered gauge-amplitude sweep",
    )
    negative = report.get("non_equivalent_ram_sensitivity", {})
    _require(
        negative.get("perturbations") == list(RAM_PERTURBATIONS),
        "Non-equivalent RAM perturbation registry drift",
    )
    for method in ("af_qdm_residual", "af_uniform_attention_residual"):
        records = negative.get("methods", {}).get(method, [])
        _require(
            len(records) == len(RAM_PERTURBATIONS),
            f"Missing RAM sensitivity records for {method}",
        )
        _require(
            all(
                np.isfinite(record["max_abs_pairwise_prediction_drift_eur_mwh"])
                and record["max_abs_pairwise_prediction_drift_eur_mwh"] >= 0.0
                for record in records
            ),
            f"Invalid RAM sensitivity values for {method}",
        )
    conditioning = report.get("conditioning_sensitivity", {})
    bins = conditioning.get("bins", [])
    _require(
        conditioning.get("partition") == "stable_rank_balanced_quintiles",
        "Conditioning sensitivity must use stable rank-balanced quintiles",
    )
    _require(len(bins) == 5, "Conditioning sensitivity must contain five bins")
    _require(
        all(int(item["rows"]) > 0 for item in bins),
        "Conditioning sensitivity contains an empty bin",
    )
    _require(
        int(conditioning.get("evaluation_rows", -1))
        == int(report["rows"]["evaluation"]),
        "Conditioning diagnostic evaluation-row count drift",
    )
    included_rows = int(conditioning.get("included_rows", -1))
    excluded_rows = int(conditioning.get("excluded_rows", -1))
    _require(
        included_rows >= 5 and excluded_rows >= 0,
        "Invalid conditioning inclusion/exclusion counts",
    )
    _require(
        included_rows + excluded_rows == int(report["rows"]["evaluation"]),
        "Conditioning inclusion/exclusion counts do not cover evaluation rows",
    )
    _require(
        sum(int(item["rows"]) for item in bins) == included_rows,
        "Conditioning bins do not partition the defined conditioning rows",
    )
    _require(
        len(conditioning.get("excluded_timestamps_utc", [])) == excluded_rows,
        "Conditioning exclusion timestamp count drift",
    )

    loss_record = report["timestamp_loss_archive"]
    loss_path = Path(loss_record["path"])
    _require(loss_path.is_file(), f"Missing timestamp loss archive: {loss_path}")
    _require(sha256_file(loss_path) == loss_record["sha256"], "Loss-archive hash drift")
    with np.load(loss_path) as archive:
        expected_keys = {"timestamps_utc_ns"}
        for method in ALL_METHODS:
            expected_keys.update(
                f"loss__{method}__{stress}" for stress in expected_presentations
            )
        for method in NEURAL_METHODS:
            expected_keys.update(
                f"member_loss__{method}__{stress}"
                for stress in expected_presentations
            )
        _require(set(archive.files) == expected_keys, "Loss-array registry drift")
        timestamps = np.asarray(archive["timestamps_utc_ns"])
        _require(timestamps.ndim == 1, "Timestamp array must be one-dimensional")
        _require(
            np.issubdtype(timestamps.dtype, np.integer),
            "Timestamp array must use integer UTC nanoseconds",
        )
        _require(
            len(timestamps) < 2 or np.all(np.diff(timestamps) > 0),
            "Evaluation timestamps must be strictly increasing and unique",
        )
        rows = len(timestamps)
        _require(rows == int(report["rows"]["evaluation"]), "Loss-row count drift")
        for key in archive.files:
            values = archive[key]
            _require(np.isfinite(values).all(), f"Non-finite loss array: {key}")
            if key.startswith("loss__"):
                _require(values.shape == (rows,), f"Wrong loss shape: {key}")
                _require(np.all(values >= 0.0), f"Negative loss value: {key}")
            elif key.startswith("member_loss__"):
                _require(
                    values.shape == (len(EXPECTED_SEEDS), rows),
                    f"Wrong member-loss shape: {key}",
                )
                _require(np.all(values >= 0.0), f"Negative member loss: {key}")

    recorded_engines = {
        Path(item["path"]).resolve(): item["sha256"]
        for item in report["reproducibility"]["engines"]
    }
    _require(set(recorded_engines) == {path.resolve() for path in ENGINE_PATHS}, "Engine registry drift")
    for path, recorded_hash in recorded_engines.items():
        _require(path.is_file(), f"Missing engine: {path}")
        _require(sha256_file(path) == recorded_hash, f"Engine hash drift: {path}")

    checkpoint_record = report["reproducibility"].get("model_checkpoints", {})
    checkpoint_files = checkpoint_record.get("files", [])
    _require(
        len(checkpoint_files) == len(NEURAL_METHODS) * len(EXPECTED_SEEDS),
        "Model-checkpoint registry is incomplete",
    )
    _require(
        len(str(checkpoint_record.get("training_signature", ""))) == 64,
        "Invalid model-checkpoint training signature",
    )
    verified_checkpoints = []
    for item in checkpoint_files:
        path = Path(item["path"]).resolve()
        _require(path.is_file(), f"Missing model checkpoint: {path}")
        current_hash = sha256_file(path)
        _require(current_hash == item["sha256"], f"Checkpoint hash drift: {path}")
        verified_checkpoints.append({"path": str(path), "sha256": current_hash})

    manifest = {
        "status": "complete",
        "protocol": protocol["name"],
        "study_role": protocol["study_role"],
        "scope": (
            "real Nordic solved-certificate representation robustness with "
            "matched mass and augmentation controls; same-delivery "
            "reconstruction, not forecasting or trading evidence"
        ),
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "source_artifacts": {
            "dual_archive": {
                "path": str(dual_path),
                "sha256": sha256_file(dual_path),
            },
            "geometry_audit": {
                "path": str(geometry_path),
                "sha256": sha256_file(geometry_path),
            },
            "day_ahead_prices": verified_price_sources,
        },
        "timestamp_loss_archive": {
            "path": str(loss_path),
            "sha256": sha256_file(loss_path),
            "arrays": len(expected_keys),
            "evaluation_rows": rows,
        },
        "seeds": EXPECTED_SEEDS,
        "methods": list(ALL_METHODS),
        "presentations": expected_presentations,
        "invariance_contract": invariance,
        "frozen_v11_reproduction_contract": report[
            "frozen_v11_reproduction_contract"
        ],
        "engines": [
            {"path": str(path), "sha256": recorded_hash}
            for path, recorded_hash in recorded_engines.items()
        ],
        "model_checkpoints": {
            "training_protocol_version": checkpoint_record.get(
                "training_protocol_version"
            ),
            "training_signature": checkpoint_record["training_signature"],
            "reused_count": int(checkpoint_record.get("reused_count", 0)),
            "files": verified_checkpoints,
        },
        "artifact_auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    _atomic_json(manifest_path, manifest)
    print(f"[OK] AF-CDMO real v14 {args.required_role} status=complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
