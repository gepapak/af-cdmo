"""Reproduce the exploratory nested HGB audit from authorized private caches.

Every input and output path is explicit. Cache-only refitting reproduces the
four fixed historical HGB controls; saved-model mode does no fitting. Neither
mode acquires data, reconstructs raw certificates, or runs a neural model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import paper2_nested_cached_metrics as metrics


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_matrix(matrix: tuple, name: str) -> None:
    features, target, magnitude, times = matrix
    if features.shape != (len(times), 84) or target.shape != (len(times),) or magnitude.shape != (len(times),):
        raise ValueError(f"Invalid cache alignment/dimensions: {name}")
    if not len(times) or not times.is_unique or not times.is_monotonic_increasing or times.hasnans:
        raise ValueError(f"Unique chronological UTC timestamps required: {name}")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(magnitude)) or np.any(magnitude < 0):
        raise ValueError(f"Nonfinite/negative cached values: {name}")
    if not np.all(np.isin(target, (0, 1))) or np.unique(target).size != 2:
        raise ValueError(f"Both original binary target classes required: {name}")


def platt_values(model: dict) -> dict[str, float]:
    stored = model["platt"]
    return {name: float(stored[name] if isinstance(stored, dict) else getattr(stored, name))
            for name in ("intercept", "coefficient")}


def score_model(model: dict, features: np.ndarray) -> np.ndarray:
    value = platt_values(model)
    raw = model["classifier"].predict_proba(features)[:, 1]
    return metrics.PlattMap(**value).probability(metrics._logit(raw))


def verify_predictions(path: Path, matrix: tuple, probabilities: dict) -> dict:
    frame = pd.read_csv(path)
    _, target, magnitude, times = matrix
    reference_times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    if not reference_times.equals(times) or not np.array_equal(target, frame["tail_target"].to_numpy(dtype=float)):
        raise ValueError("Cached original labels/timestamps differ from supplied reference")
    magnitude_error = float(np.max(np.abs(magnitude - frame["residual_magnitude_eur_mwh"].to_numpy(dtype=float))))
    if magnitude_error > 1e-8:
        raise ValueError("Cached magnitudes differ from supplied reference")
    errors = {method: float(np.max(np.abs(value - frame[f"probability_{method}"].to_numpy(dtype=float))))
              for method, value in probabilities.items()}
    if max(errors.values()) > 1e-9:
        raise RuntimeError(f"Historical predictions fail exact reproduction: {errors}")
    return {"maximum_probability_error": errors, "maximum_magnitude_error": magnitude_error,
            "original_labels_and_times_identical": True}


def verify_audit(candidate: dict, reference: dict) -> float:
    """Verify all finite numerical values in the original metric/policy grid."""
    maximum = 0.0

    def visit(value, previous, key):
        nonlocal maximum
        if isinstance(value, dict):
            if set(value) != set(previous):
                raise ValueError(f"Metric grid key mismatch: {key}")
            for name in value:
                visit(value[name], previous[name], key + "/" + name)
        elif isinstance(value, list):
            if len(value) != len(previous):
                raise ValueError(f"Metric list mismatch: {key}")
            for index, item in enumerate(value):
                visit(item, previous[index], key + "/" + str(index))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            difference = abs(float(value) - float(previous))
            maximum = max(maximum, difference)
            if difference > 1e-8:
                raise ValueError(f"Metric reproduction mismatch: {key}: {difference}")
        elif value != previous:
            raise ValueError(f"Metric reproduction mismatch: {key}")

    visit(candidate, reference, "window")
    return maximum


def self_test() -> None:
    target = np.asarray([0, 1, 0, 1, 1], dtype=float)
    score = np.asarray([.3, .3, .1, .8, .6])
    weights = np.asarray([2, 1, 3, 2, 1], dtype=float)
    actual = metrics.weighted_auc(target, metrics.auc_grouping(score), weights)
    assert abs(actual - roc_auc_score(target, score, sample_weight=weights)) < 1e-12
    assert np.isnan(metrics.weighted_auc(target, metrics.auc_grouping(score), (1 - target)))
    canonical = np.arange(3 * 36, dtype=float).reshape(3, 36) / 100
    row_weights = np.asarray([.2, .3, .5])
    indices = np.asarray([0, 6, 9, 12, 13, 17, 19, 20, 22, 26])
    x = metrics.quotient_feature(canonical, row_weights, 4.2, indices)[None, :]
    assert [metrics.control_features(x, name).shape[1] for name in metrics.METHODS] == [12, 33, 48, 84]
    assert metrics.verify_field_identity(x, indices) < 1e-12
    current = metrics.control_features(x, "analytic_current33")[:, 2:]
    assert np.max(np.abs(current + 4.2 * (row_weights @ canonical[:,:31])[None, :])) < 1e-12
    empty = metrics.quotient_feature(canonical[:0], row_weights[:0], 0, indices)[None, :]
    assert np.all(metrics.control_features(empty, "analytic_current33") == 0)
    times = pd.DatetimeIndex(pd.to_datetime(["2026-01-01T00:00Z", "2026-01-03T00:00Z"], utc=True))
    calendar = pd.date_range(times.min().normalize(), times.max().normalize(), freq="D")
    day_index = calendar.get_indexer(times.normalize())
    assert np.array_equal(day_index, [0, 2])
    day_counts = metrics.aggregate_days(np.asarray([1., 0.]), np.asarray([1., 0.]), np.asarray([2., 1.]), day_index, 3)
    assert np.array_equal(day_counts[1], np.zeros(5)), "Missing calendar day must remain empty"
    day_weights = metrics.calendar_weights(3, 11, 77, block_days=2)
    assert day_weights.shape == (11, 3) and np.all(day_weights.sum(axis=1) == 3)
    selected = metrics.daily_budget(np.asarray([.8, .8, .1]), np.zeros(3, dtype=int), .2)
    assert np.array_equal(selected, [1, 0, 0]), "Chronological tie break must remain stable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--confirmation-cache", type=Path)
    parser.add_argument("--later-cache", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--cache-origin-manifest", type=Path,
                        help="Original executed manifest, required for serialization-normalized private caches")
    parser.add_argument("--later-feature-audit", type=Path)
    parser.add_argument("--later-origin-feature-audit", type=Path,
                        help="Original later audit, required when supplying a normalization-annotated later audit")
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--confirmation-predictions", type=Path)
    parser.add_argument("--later-predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("refit", "saved-models"), default="refit")
    parser.add_argument("--model", action="append", default=[], metavar="METHOD=PATH",
                        help="Trusted historical joblib, one for each of four methods in saved-models mode")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Complete-current identity, exact feature mapping, tied/degenerate weighted AUC, missing-calendar days and stable ranking checks passed.")
        return
    path_names = ("fit_cache", "calibration_cache", "confirmation_cache", "later_cache", "cache_manifest",
                  "later_feature_audit", "reference_protocol", "reference_report", "confirmation_predictions", "later_predictions", "output")
    if any(getattr(args, name) is None for name in path_names):
        parser.error("Supply explicit caches, manifests/audits, reference protocol/report/predictions and output")
    paths = {name: getattr(args, name).resolve() for name in path_names}
    models = {}
    for specification in args.model:
        name, separator, filename = specification.partition("=")
        if not separator or name not in metrics.METHODS or name in models:
            parser.error("--model requires one unique known METHOD=PATH")
        models[name] = Path(filename).resolve()
    if args.mode == "saved-models" and set(models) != set(metrics.METHODS):
        parser.error("saved-models mode requires all four explicit --model paths")
    if args.mode == "refit" and models:
        parser.error("--model is used only with saved-models")
    output = paths.pop("output")
    if output.exists() and any(output == path.parent for path in paths.values()):
        raise ValueError("Output directory must differ from every source directory")
    output.mkdir(parents=True, exist_ok=True)
    reference = read_json(paths["reference_report"])
    reference_protocol = read_json(paths["reference_protocol"])
    manifest = read_json(paths["cache_manifest"])
    later_feature_audit = read_json(paths["later_feature_audit"])
    if "normalization" in manifest:
        if args.cache_origin_manifest is None:
            parser.error("Normalized caches require --cache-origin-manifest")
        origin_path = args.cache_origin_manifest.resolve()
        origin = read_json(origin_path)
        if sha256(origin_path) != manifest["normalization"]["source_manifest_sha256"] or sha256(origin_path) != reference["cache_manifest_sha256"]:
            raise RuntimeError("Normalized caches do not identify the original executed manifest")
        if {name: entry["sha256"] for name, entry in origin["files"].items()} != manifest["normalization"]["source_files"]:
            raise RuntimeError("Normalized cache source hashes differ from original executed sources")
        paths["cache_origin_manifest"] = origin_path
    elif sha256(paths["cache_manifest"]) != reference["cache_manifest_sha256"]:
        raise RuntimeError("Historical cache manifest changed")
    if "normalization" in later_feature_audit:
        if args.later_origin_feature_audit is None:
            parser.error("Normalized later metadata require --later-origin-feature-audit")
        origin_path = args.later_origin_feature_audit.resolve()
        origin = read_json(origin_path)
        if sha256(origin_path) != later_feature_audit["normalization"]["source_feature_audit_sha256"] or origin["features_sha256"] != later_feature_audit["normalization"]["source_features_sha256"]:
            raise RuntimeError("Normalized later source provenance differs from original audit")
        paths["later_origin_feature_audit"] = origin_path
    if sha256(paths["reference_protocol"]) != reference["protocol_sha256"] or manifest["protocol_sha256"] != reference["protocol_sha256"]:
        raise RuntimeError("Supplied historical protocol/report/cache provenance differs")
    for split in ("fit", "calibration", "confirmation"):
        if sha256(paths[split + "_cache"]) != manifest["files"][split]["sha256"]:
            raise RuntimeError("Authorized exact historical cache hash mismatch: " + split)
    if sha256(paths["later_cache"]) != later_feature_audit["features_sha256"]:
        raise RuntimeError("Authorized consumed-later cache hash mismatch")
    for name, path in models.items():
        if sha256(path) != reference["models"][name]["model_sha256"]:
            raise RuntimeError("Supplied historical model hash differs: " + name)
    sources = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sorted({**paths, **{f"model_{name}": path for name, path in models.items()}, "engine": Path(__file__).resolve(), "metrics": Path(metrics.__file__).resolve()}.items())}
    plan = {"study": "Portable cache-only reproduction of Paper2 nested learning-value audit", "mode": args.mode,
            "interpretation": "Both evaluation periods were previously examined; reproducing exploratory fixed controls and the complete original grid, not new blinded confirmation.",
            "no_neural_inference_or_raw_source_reconstruction": True, "sources": sources,
            "fixed_hgb_parameters": metrics.HGB_PARAMETERS, "bootstrap_seed": metrics.SEED,
            "bootstrap_replicates": metrics.REPLICATES, "block_days": list(metrics.BLOCKS),
            "methods": list(metrics.METHODS), "complete_pair_grid": [list(pair) for pair in metrics.PAIR_GRID],
            "chronology": {key: reference_protocol[key] for key in ("fit_end_exclusive", "calibration_start", "calibration_end_exclusive")},
            "original_label_rule": "Preserve exact cached float32-derived RiskDataset labels; never derive fit labels from float64 magnitudes.",
            "policies": reference_protocol["policies"]}
    plan_path = output / "protocol.json"
    if args.protocol_only:
        if plan_path.exists():
            raise RuntimeError("Portable execution plan already exists; refuse overwrite")
        atomic_json(plan_path, {**plan, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
        print("Portable cached execution plan frozen before scoring: " + str(plan_path))
        return
    if not plan_path.exists() or {key: value for key, value in read_json(plan_path).items() if key != "frozen_at_utc"} != plan:
        raise RuntimeError("Run a separate --protocol-only invocation; then preserve exact source bytes")
    matrices = {name: metrics.read_matrix(paths[name + "_cache"]) for name in ("fit", "calibration", "confirmation", "later")}
    indices = np.asarray(manifest["real_indices"], dtype=int)
    errors = {}
    for name, matrix in matrices.items():
        check_matrix(matrix, name)
        errors[name] = metrics.verify_field_identity(matrix[0], indices)
    fit_x, fit_y, _, fit_times = matrices["fit"]
    cal_x, cal_y, _, cal_times = matrices["calibration"]
    if fit_times.max() >= pd.Timestamp(reference_protocol["fit_end_exclusive"]) or cal_times.min() < pd.Timestamp(reference_protocol["calibration_start"]) or cal_times.max() >= pd.Timestamp(reference_protocol["calibration_end_exclusive"]):
        raise RuntimeError("Historical fit/calibration boundaries violated")
    if matrices["confirmation"][3].max() >= matrices["later"][3].min():
        raise RuntimeError("Confirmation and consumed-later windows overlap")
    probabilities = {"confirmation": {}, "later": {}}
    thresholds, model_details = {}, {}
    with threadpool_limits(limits=1):
        for method in metrics.METHODS:
            if args.mode == "refit":
                print("Fitting exact cached historical control: " + method, flush=True)
                classifier = HistGradientBoostingClassifier(**metrics.HGB_PARAMETERS).fit(metrics.control_features(fit_x, method), fit_y)
                raw = classifier.predict_proba(metrics.control_features(cal_x, method))[:, 1]
                platt = metrics._fit_platt(metrics._logit(raw), cal_y)
                model = {"classifier": classifier, "platt": {"intercept": platt.intercept, "coefficient": platt.coefficient}}
                model_path = output / (method + ".joblib")
                joblib.dump(model, model_path)
            else:
                model_path = models[method]
                # Only caller-authorized model files are read. Historical model
                # bundles may require the original PlattMap module installed.
                model = joblib.load(model_path)
            if model["classifier"].get_params() != reference["models"][method]["actual_hgb_parameters"]:
                raise RuntimeError("Historical HGB parameters changed: " + method)
            if platt_values(model) != reference["models"][method]["platt"]:
                raise RuntimeError("December Platt map failed exact reproduction: " + method)
            cal = score_model(model, metrics.control_features(cal_x, method))
            thresholds[method] = float(np.quantile(cal, .8))
            if abs(thresholds[method] - reference["december_thresholds"][method]) > 1e-9:
                raise RuntimeError("December threshold failed reproduction: " + method)
            for window in probabilities:
                probabilities[window][method] = score_model(model, metrics.control_features(matrices[window][0], method))
            model_details[method] = {"model_sha256": sha256(model_path), "platt": platt_values(model), "iterations": int(model["classifier"].n_iter_)}
        reproduction = {window: verify_predictions(paths[window + "_predictions"], matrices[window], values) for window, values in probabilities.items()}
        windows = {window: metrics.analyze_window(matrices[window], values, thresholds) for window, values in probabilities.items()}
    reproduction["metric_grid_maximum_error"] = {"confirmation": verify_audit(windows["confirmation"], reference["confirmation"]), "later": verify_audit(windows["later"], reference["later_consumed"])}
    if any(sha256(Path(entry["path"])) != entry["sha256"] for entry in sources.values()):
        raise RuntimeError("A source changed during the portable audit")
    result = {"status": "portable_cached_nested_audit_reproduced", "protocol_sha256": sha256(plan_path),
              "interpretation": plan["interpretation"], "source_inventory": sources, "mode": args.mode,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "models": model_details, "december_thresholds": thresholds, "field_identity_errors": errors,
              "reproduction": reproduction, "confirmation": windows["confirmation"], "later_consumed": windows["later"],
              "admissible_claims": reference["admissible_claims"]}
    atomic_json(output / "nested_learning_value_audit.json", result)
    print(json.dumps({"status": result["status"], "mode": args.mode, "reproduction": reproduction}, indent=2))


if __name__ == "__main__":
    main()
