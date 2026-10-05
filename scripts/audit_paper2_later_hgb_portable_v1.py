"""Score a historically reproduced Paper2 HGB on consumed later QCT data.

No training is done here. Inputs are the later period already opened by the
QCT study and the original model recreated separately using historical data.
All inputs and output locations are supplied explicitly. Provider archives,
fitted models and predictions remain private and are excluded from releases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = None
PRICE_ROOT = None
CHECKPOINT = None
GEOMETRY = None
OUTPUT = ROOT / "results/later_hgb_portable_v1"
SOURCE = ROOT / "af_cdmo_probabilistic_headroom"
OLD_REPORT = SOURCE / "quotient_tangent_risk_confirmation_v5.json"
LATE_REPORT = None
LATE_ARCHIVE = None
FULL_MODEL = ROOT / "results/learning_value_v1/full_quotient_hgb_reproduction.joblib"
sys.path.insert(0, str(ROOT))
from scripts import paper2_secondary_temporal_metrics as metrics
from scripts import paper2_portable_audit_utils as portable


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(name: str, value: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def imports():
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    from af_cdmo_probabilistic_headroom import confirm_quotient_tangent_risk_v5 as frozen
    from scripts.af_cdmo_real_extension_v13 import RealAffineGaugeDataset, RealAffineGeometry
    torch.set_num_threads(1)
    return np, pd, torch, threadpool_limits, frozen, RealAffineGaugeDataset, RealAffineGeometry


def prepare_features() -> None:
    np, pd, torch, threadpool_limits, frozen, RealAffineGaugeDataset, RealAffineGeometry = imports()
    report = read_json(OLD_REPORT)
    later = read_json(LATE_REPORT)
    certificate, price_root, checkpoint = CERTIFICATE, PRICE_ROOT, CHECKPOINT
    expected_checkpoint = next(record["sha256"] for record in report["checkpoints"] if record["method"] == "qct_full" and int(record["seed"]) == 7)
    if sha256(checkpoint) != expected_checkpoint:
        raise RuntimeError("Frozen scaling checkpoint changed")
    if sha256(certificate) != later["inputs"]["hashes"]["holdout_dual"]:
        raise RuntimeError("Later certificate differs from the consumed source")
    if sha256(LATE_ARCHIVE) != later["prediction_archive"]["sha256"]:
        raise RuntimeError("Consumed prediction archive changed")
    source_checks = {
        Path(frozen.__file__).resolve(): report["sources"]["confirmation_engine_sha256"],
        Path(frozen.qct.__file__).resolve(): report["sources"]["development_engine_sha256"],
    }
    for path, expected in source_checks.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen engine changed: {path}")
    save("hgb_feature_plan.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_role": "post-confirmation secondary temporal comparator audit",
        "outcome_exposure": "Existing QCT and reconstructed-HGB point outcomes were already known; not untouched independent confirmation.",
        "feature_source": "Original fixed quotient_feature, with original checkpoint scales and exact Paper2 RiskDataset threshold",
        "no_training": True,
        "no_neural_inference": True,
        "no_fs_vano_access": True,
        "certificate_sha256": sha256(certificate),
        "scaling_checkpoint_sha256": expected_checkpoint,
        "existing_prediction_archive_sha256": sha256(LATE_ARCHIVE),
        "script_sha256": sha256(Path(__file__)),
        "original_report_sha256": sha256(OLD_REPORT),
        "later_report_sha256": sha256(LATE_REPORT),
        "statistics_engine_sha256": sha256(Path(metrics.__file__)),
    })
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target_scale = float(payload["target_scale"])
    ram_scale = float(payload["canonical_ram_scale"])
    threshold = float(report["protocol"]["tail_thresholds_eur_mwh"][1])
    with threadpool_limits(limits=1):
        frame, certificate_audit = frozen.qct._read_certificate(certificate, 60.0)
        unit_columns = certificate_audit["unit_columns"]
        input_zones = tuple(column.removeprefix("unit_ptdf_") for column in unit_columns)
        geometry = RealAffineGeometry.loss_safe(input_zones)
        geometry_path = GEOMETRY
        if sha256(geometry_path) != report["sources"]["geometry_audit_sha256"]:
            raise RuntimeError("Geometry source changed")
        if list(input_zones) != read_json(geometry_path)["geometry"]["zone_order"]:
            raise RuntimeError("Later geometry coordinate order changed")
        zones = frozen.qct.PRIMARY_ZONES
        real_indices = np.asarray([unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64)
        prices, price_provenance = frozen.qct._load_price_panel(price_root, zones)
        for zone, record in price_provenance.items():
            if record["sha256"] != later["inputs"]["hashes"][f"holdout_price_{zone}"]:
                raise RuntimeError("Later price source changed")
        samples = frozen.qct._build_vector_samples(frame, prices, unit_columns, zones, frozen.qct.FIT_END, pd.Timestamp("2026-10-01", tz="UTC"))
        samples = [sample for sample in samples if pd.Timestamp("2026-08-20", tz="UTC") <= sample["timestamp"] < pd.Timestamp("2026-10-01", tz="UTC")]
        # raw_scale only affects the unused raw_scaled output. Original
        # canonical/weights/mass/analytic fields are unaffected by this value.
        base = RealAffineGaugeDataset(samples, raw_scale=np.ones(1 + len(input_zones) + 5), target_scale=target_scale, presentation="original", representation="af", real_zone_indices=real_indices, geometry=geometry, canonical_ram_scale=ram_scale)
        dataset = frozen.RiskDataset(base, target_scale=target_scale, threshold=threshold)
        feature_rows, targets, magnitude, times = [], [], [], []
        for index in range(len(dataset)):
            item = dataset[index]
            total = float(np.expm1(float(item["total_scaled"])))
            feature_rows.append(frozen.quotient_feature(item["canonical"], item["weights"], total, real_indices))
            targets.append(item["tail_target"])
            magnitude.append(item["residual_magnitude"])
            times.append(pd.Timestamp(item["timestamp"]).isoformat())
            dataset._cache[index] = None
        feature_matrix = np.asarray(feature_rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        magnitude = np.asarray(magnitude, dtype=float)
        times = np.asarray(times)
    with np.load(LATE_ARCHIVE, allow_pickle=False) as cached:
        if not np.array_equal(times, cached["timestamp_utc"]):
            raise RuntimeError("Reconstructed later timestamps differ from cached run")
        if not np.array_equal(target, cached["target"]):
            raise RuntimeError("Reconstructed later targets differ from cached run")
        if not np.allclose(magnitude, cached["residual_magnitude_eur_mwh"], atol=1e-7, rtol=0):
            raise RuntimeError("Reconstructed later residual magnitude differs from cached run")
        magnitude_error = float(np.max(np.abs(magnitude - cached["residual_magnitude_eur_mwh"])))
    path = OUTPUT / "later_quotient_features_private.npz"
    np.savez_compressed(path, features=feature_matrix, target=target, residual_magnitude_eur_mwh=magnitude, timestamp_utc=times)
    save("hgb_feature_audit.json", {
        "status": "passed",
        "rows": len(target),
        "feature_dimensions": int(feature_matrix.shape[1]),
        "all_features_finite": bool(np.all(np.isfinite(feature_matrix))),
        "cached_timestamps_match_exactly": True,
        "cached_targets_match_exactly": True,
        "cached_residual_magnitude_max_abs_error": magnitude_error,
        "target_scale": target_scale,
        "canonical_ram_scale": ram_scale,
        "real_indices": real_indices.tolist(),
        "tail_threshold_eur_mwh": threshold,
        "features_sha256": sha256(path),
        "original_checkpoint_sha256": expected_checkpoint,
        "geometry_sha256": sha256(geometry_path),
        "certificate_sha256": sha256(certificate),
        "price_provenance": price_provenance,
        "feature_plan_sha256": sha256(OUTPUT / "hgb_feature_plan.json"),
    })
    print(json.dumps({"status": "features_ready", "rows": len(target), "feature_dimensions": int(feature_matrix.shape[1]), "cached_magnitude_max_error": magnitude_error}))


def score() -> None:
    np, pd, _, threadpool_limits, frozen, _, _ = imports()
    import joblib

    if not FULL_MODEL.is_file():
        raise RuntimeError("Original full HGB must be recreated and verified first")
    original = read_json(OLD_REPORT)
    feature_audit = read_json(OUTPUT / "hgb_feature_audit.json")
    feature_plan = read_json(OUTPUT / "hgb_feature_plan.json")
    for path, key in ((Path(__file__), "script_sha256"), (OLD_REPORT, "original_report_sha256"),
                      (LATE_REPORT, "later_report_sha256"), (Path(metrics.__file__), "statistics_engine_sha256")):
        if sha256(path) != feature_plan[key]:
            raise RuntimeError(f"Input or adapter changed after feature plan: {key}")
    feature_path = OUTPUT / "later_quotient_features_private.npz"
    if sha256(feature_path) != feature_audit["features_sha256"]:
        raise RuntimeError("Prepared feature archive changed")
    model = joblib.load(FULL_MODEL)
    if model["platt"].__dict__ != original["control_details"]["hgb_quotient"]["platt"]:
        raise RuntimeError("Reconstructed full HGB Platt map differs from actual Paper2 confirmation")
    threshold = original["methods"]["hgb_quotient"]["decision_at_calibration_80pct_coverage"]["probability_threshold_from_december"]
    save("hgb_scoring_plan.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_role": "post-confirmation secondary temporal reuse",
        "outcome_exposure": "Prior QCT neural outcomes were known before this comparator audit; no new blinded confirmation claim.",
        "model_path": str(FULL_MODEL),
        "model_sha256": sha256(FULL_MODEL),
        "feature_sha256": feature_audit["features_sha256"],
        "frozen_december_threshold": threshold,
        "all_model_fitting_and_calibration_before_2026_01_01": True,
        "bootstrap": {"block_days": [7, 14, 28], "replicates": 2000, "seed": 20261003},
        "no_later_training_calibration_selection": True,
    })
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_matrix = archive["features"]
        target = archive["target"]
        magnitude = archive["residual_magnitude_eur_mwh"]
        timestamps = pd.to_datetime(archive["timestamp_utc"], utc=True)
    with threadpool_limits(limits=1):
        raw_probability = model["classifier"].predict_proba(feature_matrix)[:, 1]
        probability = model["platt"].probability(frozen._logit(raw_probability))
    with np.load(LATE_ARCHIVE, allow_pickle=False) as cached:
        full = cached["probability__qct_full"]
        no_tangent = cached["probability__qct_no_tangent"]
        uniform = cached["probability__uniform_set_transformer"]
    metrics_path = Path(metrics.__file__)
    timestamps = portable.validate_predictions(timestamps, target, magnitude,
        {"qct_full": full, "qct_no_tangent": no_tangent, "uniform_set_transformer": uniform, "hgb_quotient": probability},
        float(original["protocol"]["tail_thresholds_eur_mwh"][1]))
    with np.load(LATE_ARCHIVE, allow_pickle=False) as cached:
        if not np.array_equal(np.asarray([item.isoformat() for item in timestamps]), cached["timestamp_utc"]) or not np.array_equal(target, cached["target"]) or not np.array_equal(magnitude, cached["residual_magnitude_eur_mwh"]):
            raise RuntimeError("Prepared HGB features and neural cached outcomes do not align exactly")
    result = {
        "status": "passed_original_hgb_secondary_later_scoring",
        "study_role": "post-confirmation secondary temporal reuse of consumed QCT chronology",
        "rows": len(target),
        "tail_events": int(target.sum()),
        "hgb_classification": metrics.classification_metrics(target, probability),
        "hgb_fixed_december_threshold": metrics.threshold_operational_metrics(target, magnitude, probability, threshold),
        "hgb_common_budget_grid": {str(budget): metrics.operational_metrics(target, magnitude, probability, budget) for budget in (0.05, 0.10, 0.15, 0.20)},
        "paired_neural_minus_original_hgb": {
            method: {
                "brier": metrics.compare_brier(timestamps, target, candidate, probability, block_days=[7, 14, 28], replicates=2000, seed=20261003, absolute_margin=0.0025, relative_margin=0.05),
                "20pct_cost_audit_cost_0_05": metrics.compare_operational_cost(timestamps, target, candidate, probability, audit_budget=0.20, missed_tail_cost=1.0, audit_cost=0.05, block_days=[7, 14, 28], replicates=2000, seed=20261003),
            } for method, candidate in (("qct_full", full), ("qct_no_tangent", no_tangent), ("uniform_set_transformer", uniform))
        },
        "model_sha256": sha256(FULL_MODEL),
        "feature_sha256": sha256(feature_path),
        "statistics_engine_sha256": sha256(metrics_path),
        "scoring_plan_sha256": sha256(OUTPUT / "hgb_scoring_plan.json"),
        "source_neural_archive_sha256": sha256(LATE_ARCHIVE),
        "original_paper2_hgb_platt_reproduced_exactly": True,
        "later_fit_calibration_selection_performed": False,
        "fs_vano_confirmation_accessed": False,
    }
    prediction_path = OUTPUT / "later_original_hgb_predictions_private.npz"
    np.savez_compressed(prediction_path, probability=probability, target=target, residual_magnitude_eur_mwh=magnitude, timestamp_utc=np.asarray([item.isoformat() for item in timestamps]))
    result["prediction_archive_sha256"] = sha256(prediction_path)
    save("original_hgb_audit.json", result)
    print(json.dumps({"output": str(OUTPUT / "original_hgb_audit.json"), "hgb_brier": result["hgb_classification"]["brier"], "full_minus_hgb": result["paired_neural_minus_original_hgb"]["qct_full"]["brier"]["candidate_minus_comparator_brier"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    global OLD_REPORT, LATE_REPORT, LATE_ARCHIVE, FULL_MODEL, OUTPUT, CERTIFICATE, PRICE_ROOT, CHECKPOINT, GEOMETRY
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--later-report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Trusted historical full HGB plus Platt joblib; no fitting here")
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--price-root", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original qct_full seed7 scaling checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    OLD_REPORT, LATE_REPORT, LATE_ARCHIVE, FULL_MODEL = args.original_report.resolve(), args.later_report.resolve(), args.predictions.resolve(), args.model.resolve()
    OUTPUT, CERTIFICATE, PRICE_ROOT, CHECKPOINT, GEOMETRY = args.output.resolve(), args.dual_archive.resolve(), args.price_root.resolve(), args.checkpoint.resolve(), args.geometry_audit.resolve()
    if args.prepare == args.score:
        parser.error("Choose exactly one of --prepare and --score")
    prepare_features() if args.prepare else score()


if __name__ == "__main__":
    main()
