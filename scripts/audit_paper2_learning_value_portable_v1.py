"""Post-confirmation feature-information ablation for Paper2.

Fits a fixed analytic-field-only HGB control and reproduces the existing full
quotient HGB. No neural architecture or confirmation-based model selection.
Private provider records and fitted models remain outside the public release.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Set before importing numerical runtimes; preserve one CPU thread throughout.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from af_cdmo_probabilistic_headroom import confirm_quotient_tangent_risk_v5 as frozen
from af_cdmo_probabilistic_headroom import audit_qct_risk_temporal_robustness_v5 as temporal
from scripts import audit_paper2_operational_triage_portable_v1 as operational
from scripts import paper2_portable_audit_utils as portable

SOURCE = ROOT / "af_cdmo_probabilistic_headroom"
REPORT = SOURCE / "quotient_tangent_risk_confirmation_v5.json"
PREDICTIONS = SOURCE / "qct_risk_temporal_robustness_v5/confirmation_predictions_internal.csv.gz"
DEFAULT_OUTPUT = ROOT / "results/learning_value_portable_v1"
MANIFEST = SOURCE / "qct_risk_post_confirmation_v5_manifest.json"
MAPPED_REPORT = None
SEED = 20261002
REPLICATES = 2000
HGB_PARAMETERS = dict(loss="log_loss", learning_rate=0.05, max_iter=220,
                      max_leaf_nodes=15, min_samples_leaf=50,
                      l2_regularization=3.0, random_state=SEED)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inventory(report: dict) -> list[dict]:
    sources = report["sources"]
    checks = {}
    manifest_path = MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for path, logical in (
        (REPORT, "af_cdmo_probabilistic_headroom/quotient_tangent_risk_confirmation_v5.json"),
        (PREDICTIONS, "af_cdmo_probabilistic_headroom/qct_risk_temporal_robustness_v5/confirmation_predictions_internal.csv.gz"),
    ):
        record = portable.manifest_check(path, manifest, logical)
        checks[path.resolve()] = record["sha256"]
    for label in ("dual_archive", "geometry_audit", "development_artifact",
                  "confirmation_reference", "confirmation_engine", "protocol_note"):
        checks[Path(sources[label]).resolve()] = sources[label + "_sha256"].lower()
    for entry in sources["price_provenance"].values():
        checks[Path(entry["path"]).resolve()] = entry["sha256"].lower()
    checks[Path(frozen.qct.__file__).resolve()] = sources["development_engine_sha256"].lower()
    files = {**checks, Path(__file__).resolve(): None,
             Path(temporal.__file__).resolve(): None,
             Path(operational.__file__).resolve(): None,
             manifest_path.resolve(): None}
    # Capture every currently imported repository Python dependency. Import
    # inventory is execution provenance, not proof of historical source bytes.
    for module in list(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py" and path.is_file():
                files.setdefault(path, None)
    output = []
    for path, reference in sorted(files.items(), key=lambda item: str(item[0])):
        digest = sha256(path)
        if reference and digest != reference:
            raise RuntimeError(f"Frozen source mismatch: {path}")
        output.append(dict(path=str(path), sha256=digest,
                           verified_against_frozen_reference=reference is not None))
    return output


def plan(sources: list[dict], report: dict) -> dict:
    return dict(
        study="Paper2 analytic-field-only information ablation v1",
        status="post_confirmation_diagnostic_protocol",
        interpretation="Parameters frozen now; original confirmation outcomes were already known. This is not a newly preregistered confirmation or a neural-superiority test.",
        candidate="HGB on first 12 fixed quotient_feature columns: log1p projected mass, certificate presence, all ten centered analytic-field coordinates",
        baseline="Frozen full-quotient HGB probabilities; refitted full control verifies reproduction only",
        primary_contrast="full quotient HGB minus analytic-field-only HGB Brier",
        hgb_parameters=HGB_PARAMETERS,
        fit_end_exclusive="2025-11-01T00:00:00+00:00",
        model_selection="None; inherited fixed HGB configuration; November tuning and January-February development are not used",
        calibration_start="2025-12-01T00:00:00+00:00",
        calibration_end_exclusive="2026-01-01T00:00:00+00:00",
        probability_calibration="Existing December Platt fitting algorithm, fit without confirmation labels",
        frozen_event_threshold_eur_mwh=report["protocol"]["tail_thresholds_eur_mwh"][1],
        confirmation_start=report["protocol"]["confirmation_start"],
        confirmation_end_exclusive=report["protocol"]["confirmation_end_exclusive"],
        fixed_decision="Score > December 0.80 quantile; confirmation workload may differ from calibration workload",
        fixed_workload="Supplemental retrospective daily batch: ceil(0.20 * available timestamps) highest scores, stable chronological tie rule; no labels in ranking",
        uncertainty="Paired circular seven UTC-calendar-day blocks preserving missing days; ratio denominators recomputed in every draw",
        bootstrap_replicates=REPLICATES, bootstrap_seed=SEED,
        resource_policy="CPU only; one numerical thread; 600 MiB guard when optional psutil is available",
        sources=sources,
    )


def features(dataset, real_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values, labels, magnitude, times = [], [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        total = float(np.expm1(float(item["total_scaled"])))
        values.append(frozen.quotient_feature(item["canonical"], item["weights"], total, real_indices))
        labels.append(item["tail_target"])
        magnitude.append(item["residual_magnitude"])
        times.append(item["timestamp"])
        # The exact helpers cache arrays; release each completed item to avoid
        # retaining duplicate canonical matrices throughout the analysis.
        dataset._cache[index] = None
    array = np.asarray(values, dtype=np.float64)
    if array.shape[1] < 12 or not np.all(np.isfinite(array)):
        raise RuntimeError("Invalid fixed quotient feature matrix")
    return array, np.asarray(labels, dtype=float), np.asarray(magnitude, dtype=float), pd.DatetimeIndex(times)


def fitted_probabilities(fit_x, fit_y, cal_x, cal_y, eval_x, model_path: Path):
    classifier = HistGradientBoostingClassifier(**HGB_PARAMETERS).fit(fit_x, fit_y)
    calibration_raw = classifier.predict_proba(cal_x)[:, 1]
    calibrator = frozen._fit_platt(frozen._logit(calibration_raw), cal_y)
    # Store the fitted estimator and December calibration before scoring the
    # confirmation block. No scoring outcome affects either object.
    joblib.dump(dict(classifier=classifier, platt=calibrator), model_path)
    evaluation_raw = classifier.predict_proba(eval_x)[:, 1]
    cal = calibrator.probability(frozen._logit(calibration_raw))
    score = calibrator.probability(frozen._logit(evaluation_raw))
    return cal, score, dict(platt=calibrator.__dict__, actual_hgb_parameters=classifier.get_params(),
                           iterations=int(classifier.n_iter_), model_sha256=sha256(model_path))


def contrast_metrics(target, magnitude, probabilities, thresholds, timestamps):
    dates = timestamps.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    day_index = calendar.get_indexer(dates)
    weights = operational.calendar_weights(len(calendar), REPLICATES, SEED)
    records, bootstrap = {}, {}
    for method, probability in probabilities.items():
        for label, flagged in (
            ("fixed_december", (probability > thresholds[method]).astype(float)),
            ("daily_20pct", operational.daily_budget(probability, day_index, 0.20)),
        ):
            key = f"{label}/{method}"
            counts = operational.aggregate_days(flagged, target, magnitude, day_index, len(calendar))
            drawn = weights @ counts
            records[key] = operational.metrics_record(counts.sum(axis=0), drawn)
            bootstrap[key] = operational.ratios(drawn)
    paired = {}
    for label in ("fixed_december", "daily_20pct"):
        candidate = f"{label}/hgb_quotient"
        baseline = f"{label}/hgb_analytic_field_only"
        paired[label] = {}
        for metric in operational.METRICS:
            estimate = records[candidate]["metrics"][metric]["estimate"] - records[baseline]["metrics"][metric]["estimate"]
            difference = bootstrap[candidate][metric] - bootstrap[baseline][metric]
            paired[label][metric] = dict(full_quotient_minus_field_only=estimate,
                                         calendar_7day_95ci=operational.interval(difference))
    workloads = [records[f"daily_20pct/{method}"]["escalated"] for method in probabilities]
    if len(set(workloads)) != 1:
        raise RuntimeError("Daily fixed-workload methods have different budgets")
    return records, paired


def self_test() -> None:
    operational.self_test()
    x = np.arange(18 * 4, dtype=float).reshape(4, 18)
    field_only = x[:, :12]
    assert field_only.shape == (4, 12) and np.array_equal(field_only, x[:, np.arange(12)])
    assert HGB_PARAMETERS == dict(loss="log_loss", learning_rate=0.05, max_iter=220,
                                 max_leaf_nodes=15, min_samples_leaf=50,
                                 l2_regularization=3.0, random_state=20261002)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    global REPORT, PREDICTIONS, MANIFEST, MAPPED_REPORT
    parser.add_argument("--output_dir", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    portable.add_source_arguments(parser)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Feature subset, fixed configuration, daily budget and calendar bootstrap checks passed.")
        return
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Private diagnostic output must remain within Paper2/results")
    output.mkdir(parents=True, exist_ok=True)
    if args.report is None or args.predictions is None or args.manifest is None:
        parser.error("Supply --report, --predictions and --manifest")
    REPORT, PREDICTIONS, MANIFEST = args.report.resolve(), args.predictions.resolve(), args.manifest.resolve()
    report = portable.map_report_sources(portable.read_json(REPORT), args, ROOT)
    MAPPED_REPORT = report
    sources = inventory(report)
    protocol = plan(sources, report)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if {key: value for key, value in previous.items() if key != "frozen_at_utc"} != protocol:
            raise RuntimeError("Protocol or frozen source inventory changed; preserve and review any amendment before scoring")
    else:
        if not args.protocol_only:
            raise RuntimeError("Run --protocol-only in a separate command before fitting or scoring")
        protocol["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(protocol_path, protocol)
    if args.protocol_only:
        print(f"Protocol frozen before scoring: {protocol_path}")
        return
    available = psutil.virtual_memory().available if psutil is not None else None
    if available is not None and available < 600 * 1024 ** 2:
        raise RuntimeError(f"Insufficient available memory for source reconstruction: {available} bytes")
    torch.set_num_threads(1)
    print("Reconstructing exact fit, December calibration and existing confirmation inputs.", flush=True)
    with threadpool_limits(limits=1):
        context = temporal._reconstruct(report)
        matrices = {}
        for split in ("fit", "calibration", "confirmation"):
            matrices[split] = features(context[split], context["real_indices"])
            print(f"Extracted {split}: {matrices[split][0].shape}", flush=True)
        fit_x, fit_y, _, fit_times = matrices["fit"]
        cal_x, cal_y, _, cal_times = matrices["calibration"]
        evaluation_x, target, magnitude, timestamps = matrices["confirmation"]
        if fit_times.max() >= pd.Timestamp(protocol["fit_end_exclusive"]):
            raise RuntimeError("Fit features crossed the frozen fit boundary")
        if cal_times.min() < pd.Timestamp(protocol["calibration_start"]) or cal_times.max() >= pd.Timestamp(protocol["calibration_end_exclusive"]):
            raise RuntimeError("Probability calibration crossed its December boundary")
        saved = pd.read_csv(PREDICTIONS)
        saved_times = pd.DatetimeIndex(pd.to_datetime(saved["timestamp_utc"], utc=True))
        if not timestamps.equals(saved_times) or not np.array_equal(target, saved["tail_target"].to_numpy(dtype=float)):
            raise RuntimeError("Reconstructed confirmation labels/timestamps do not reproduce frozen per-time predictions")
        magnitude_error = float(np.max(np.abs(magnitude - saved["residual_magnitude_eur_mwh"].to_numpy()), initial=0))
        if magnitude_error > 1e-8:
            raise RuntimeError(f"Reconstructed residual magnitudes differ: {magnitude_error}")
        frozen_probability = saved["probability_hgb_quotient"].to_numpy(dtype=float)
        del context
        gc.collect()
        print("Fitting fixed analytic-field-only HGB and reproducing full-quotient HGB; one CPU thread.", flush=True)
        field_cal, field_probability, field_details = fitted_probabilities(
            fit_x[:, :12], fit_y, cal_x[:, :12], cal_y, evaluation_x[:, :12], output / "analytic_field_only_hgb.joblib")
        full_cal, full_probability, full_details = fitted_probabilities(
            fit_x, fit_y, cal_x, cal_y, evaluation_x, output / "full_quotient_hgb_reproduction.joblib")
    maximum_reproduction_error = float(np.max(np.abs(full_probability - frozen_probability), initial=0))
    if maximum_reproduction_error > 1e-9:
        raise RuntimeError(f"Full HGB fails exact frozen probability reproduction: {maximum_reproduction_error}")
    thresholds = dict(hgb_analytic_field_only=float(np.quantile(field_cal, 0.80)),
                      hgb_quotient=report["methods"]["hgb_quotient"]["decision_at_calibration_80pct_coverage"]["probability_threshold_from_december"])
    full_threshold_error = abs(float(np.quantile(full_cal, 0.80)) - thresholds["hgb_quotient"])
    if full_threshold_error > 1e-9:
        raise RuntimeError("Full HGB December threshold did not reproduce")
    probabilities = dict(hgb_analytic_field_only=field_probability, hgb_quotient=frozen_probability)
    records, metric_contrasts = contrast_metrics(target, magnitude, probabilities, thresholds, timestamps)
    brier_difference = (frozen_probability - target) ** 2 - (field_probability - target) ** 2
    brier = temporal._calendar_grid_block_bootstrap(brier_difference, timestamps,
        block_days=7, replicates=REPLICATES, seed=SEED)
    result = dict(
        status="post_confirmation_diagnostic_complete", protocol_sha256=sha256(protocol_path),
        interpretation=protocol["interpretation"], resource_policy=protocol["resource_policy"],
        source_inventory=sources, source_hashes_unchanged=inventory(report) == sources,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     sklearn=sklearn.__version__, torch=torch.__version__),
        rows={name: int(len(value[1])) for name, value in matrices.items()},
        feature_dimensions=dict(analytic_field_only=12, full_quotient=int(fit_x.shape[1])),
        reproduction=dict(maximum_full_hgb_probability_error=maximum_reproduction_error,
                          full_december_threshold_error=full_threshold_error,
                          residual_magnitude_error=magnitude_error),
        fitted_models=dict(analytic_field_only=field_details, full_quotient_reproduction=full_details),
        december_thresholds=thresholds,
        classification={method: temporal._safe_classification(target, probability) for method, probability in probabilities.items()},
        paired_brier_full_quotient_minus_field_only=brier,
        workload_and_risk=records, paired_workload_metrics=metric_contrasts,
        admissible_claims=["A fixed within-model feature-information diagnostic; no architecture or hyperparameter selection.",
            "Additional atom information is empirically supported only if the full-versus-field-only interval resolves a gain.",
            "Does not establish neural superiority, data-feed-outage frequency, deployed monetary value, or missing-price robustness under a different missingness process."],
    )
    if not result["source_hashes_unchanged"]:
        raise RuntimeError("Source bytes changed during diagnostic")
    exported = pd.DataFrame(dict(timestamp_utc=timestamps.astype(str), tail_target=target,
        residual_magnitude_eur_mwh=magnitude, probability_hgb_analytic_field_only=field_probability,
        probability_hgb_quotient=frozen_probability))
    exported.to_csv(output / "confirmation_predictions_internal.csv.gz", index=False, compression="gzip")
    result["prediction_file_sha256"] = sha256(output / "confirmation_predictions_internal.csv.gz")
    atomic_json(output / "learning_value_audit.json", result)
    print(json.dumps(dict(status=result["status"], classification=result["classification"], paired_brier=brier,
                          reproduction=result["reproduction"]), indent=2))


if __name__ == "__main__":
    main()
