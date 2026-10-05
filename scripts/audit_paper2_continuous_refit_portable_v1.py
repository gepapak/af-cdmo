"""Refit fixed linear and nonlinear continuous controls from authorized caches.

All data, manifest and reference paths are explicit. Numeric NPZ readers keep
pickle disabled. No provider download, neural inference or neural fitting.
Run --protocol-only in a separate invocation before any new fitting/scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from . import paper2_nested_cached_metrics as cached
    from . import paper2_reconstruction_cached_metrics as metrics
except ImportError:
    import paper2_nested_cached_metrics as cached
    import paper2_reconstruction_cached_metrics as metrics

CONTROLS = ("field12", "analytic_current33", "first_moments48", "quotient84")
HGB_PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=220,
    max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=3.0, random_state=20261002)
RIDGE_PARAMETERS = dict(alpha=1.0, fit_intercept=True)
BLOCKS, REPLICATES, SEED = (7, 14, 28), 2000, 20261002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_times(raw: np.ndarray, name: str) -> pd.DatetimeIndex:
    if raw.ndim != 1 or raw.dtype.kind not in "SU":
        raise ValueError(f"Fixed-width string timestamps required without pickle: {name}")
    times = pd.DatetimeIndex(pd.to_datetime(raw.astype(str), utc=True))
    if not len(times) or times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError(f"Unique chronological UTC timestamps required: {name}")
    return times


def load_window(feature_path: Path, target_path: Path) -> tuple:
    matrix = cached.read_matrix(feature_path)
    x, labels, magnitude, times = matrix
    if x.shape != (len(times), 84) or not np.isfinite(x).all():
        raise ValueError("Exact finite 84-column feature matrix required")
    with np.load(target_path, allow_pickle=False) as archive:
        if not times.equals(check_times(archive["timestamp_utc"], str(target_path))):
            raise ValueError("Feature/target timestamp order differs")
        target = archive["target_field"]
        analytic = archive["analytic_field"]
        residual = archive["residual_field"]
    for name, field in (("target", target), ("analytic", analytic), ("residual", residual)):
        if field.shape != (len(times), 10) or not np.issubdtype(field.dtype, np.number) or not np.isfinite(field).all():
            raise ValueError(f"Finite ten-zone field required: {name}")
    if not np.array_equal(analytic, x[:, 2:12]):
        raise ValueError("Analytic target-cache field differs from the frozen features")
    if np.max(np.abs(target - metrics.center(target))) > 1e-8:
        raise ValueError("Trusted target is not centered")
    if not np.array_equal(residual, metrics.center(target - analytic)):
        raise ValueError("Exact centered residual identity differs")
    error = float(np.max(np.abs(np.abs(metrics.pairs(residual)).mean(axis=1) - magnitude)))
    if error > 1e-3:
        raise ValueError("Original analytic-residual magnitude guard failed")
    return matrix, target, residual, dict(rows=len(times), target_alignment=True,
        maximum_analytic_magnitude_difference=error, target_centering=True)


def read_hgb_prediction(path: Path, matrix: tuple, target: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if not matrix[3].equals(check_times(archive["timestamp_utc"], str(path))):
            raise ValueError("Original HGB predictions have different timestamps")
        values = {name: archive["field_" + name] for name in CONTROLS}
        for name, field in values.items():
            if field.shape != target.shape or not np.isfinite(field).all():
                raise ValueError("Invalid original HGB field")
            loss = np.abs(metrics.pairs(field) - metrics.pairs(target)).mean(axis=1)
            if not np.array_equal(loss, archive["timestamp_mae_" + name]):
                raise ValueError("Original HGB per-timestamp loss differs against supplied targets")
    return values


def analyze(times: pd.DatetimeIndex, target: np.ndarray, predictions: dict) -> tuple:
    methods = tuple(predictions)
    dates = times.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    day = calendar.get_indexer(dates)
    counts = np.bincount(day, minlength=len(calendar)).astype(float)
    actual = metrics.pairs(target)
    point, losses, absolute_day, squared_day, order = {}, {}, {}, {}, {}
    for method, field in predictions.items():
        errors = metrics.pairs(field) - actual
        absolute = np.abs(errors).mean(axis=1)
        squared = np.square(errors).mean(axis=1)
        losses[method] = absolute
        point[method] = dict(pairwise_mae=float(absolute.mean()),
            p90_timestamp_mae=float(np.quantile(absolute, .90)), pairwise_rmse=float(np.sqrt(squared.mean())),
            maximum_field_sum=float(np.max(np.abs(field.sum(axis=1)))))
        absolute_day[method] = np.bincount(day, weights=absolute, minlength=len(calendar))
        squared_day[method] = np.bincount(day, weights=squared, minlength=len(calendar))
        order[method] = np.argsort(absolute, kind="stable")
    pairs = tuple((methods[j], methods[i]) for i, j in itertools.combinations(range(len(methods)), 2))
    uncertainty, paired, matched = {}, {}, {}
    for block in BLOCKS:
        print(f"  Complete {len(methods)}-method uncertainty: {block} calendar days, {len(times)} rows.", flush=True)
        weights = cached.calendar_weights(len(calendar), REPLICATES, SEED, block_days=block)
        denominator = weights @ counts
        if np.any(denominator == 0):
            raise ValueError("A bootstrap draw contains no observations")
        draws = {}
        for method in methods:
            ordered = order[method]
            p90 = np.asarray([metrics.weighted_quantile(losses[method][ordered], draw[day[ordered]], .90) for draw in weights])
            draws[method] = dict(pairwise_mae=(weights @ absolute_day[method]) / denominator,
                pairwise_rmse=np.sqrt((weights @ squared_day[method]) / denominator), p90_timestamp_mae=p90)
        uncertainty[str(block)] = {method: {metric: cached.summarize_draws(point[method][metric], values)
            for metric, values in draws[method].items()} for method in methods}
        def contrast(candidate, baseline):
            return {metric: cached.summarize_draws(point[candidate][metric] - point[baseline][metric],
                draws[candidate][metric] - draws[baseline][metric])
                for metric in ("pairwise_mae", "p90_timestamp_mae", "pairwise_rmse")}
        paired[str(block)] = {f"{candidate}_minus_{baseline}": contrast(candidate, baseline) for candidate, baseline in pairs}
        matched[str(block)] = {name: contrast("ridge_" + name, "hgb_" + name) for name in CONTROLS
            if "ridge_" + name in predictions and "hgb_" + name in predictions}
    return dict(rows=len(times), first_utc=str(times[0]), last_utc=str(times[-1]),
        point=point, uncertainty=uncertainty, paired=paired, matched_linear_minus_hgb=matched), losses


def self_test() -> None:
    rng = np.random.default_rng(119)
    for size in (3, 19):
        values = np.sort(rng.normal(size=size))
        weights = rng.integers(0, 5, size=size)
        weights[0] = 1
        assert abs(metrics.weighted_quantile(values, weights, .90) - np.quantile(np.repeat(values, weights), .90)) < 1e-12
    x = rng.normal(size=(30, 7))
    target = metrics.center(rng.normal(size=(30, 10)))
    pipeline = make_pipeline(StandardScaler(), Ridge(**RIDGE_PARAMETERS)).fit(x, target)
    assert np.allclose(pipeline[0].mean_, x.mean(axis=0))
    assert pipeline.predict(x).shape == (30, 10)
    assert np.allclose(metrics.pairs(target), metrics.pairs(metrics.center(target)))
    dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-01T00:00Z", "2026-01-03T00:00Z"], utc=True))
    assert np.array_equal(pd.date_range(dates.min(), dates.max()).get_indexer(dates), [0, 2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ridge", "hgb", "both"), default="both",
        help="ridge fits four linear controls and compares with cached HGB; hgb/both genuinely refit HGB")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    for name in ("fit-features", "fit-targets", "confirmation-features", "confirmation-targets",
        "later-features", "later-targets", "feature-provenance", "continuous-manifest",
        "confirmation-hgb-predictions", "later-hgb-predictions", "hgb-reference-protocol",
        "hgb-reference-report", "output", "preserve-manifest"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Weighted p90, fit-only standardization, multioutput field, centering and gap-calendar checks passed.")
        return
    input_names = ("fit_features", "fit_targets", "confirmation_features", "confirmation_targets",
        "later_features", "later_targets", "feature_provenance", "continuous_manifest",
        "confirmation_hgb_predictions", "later_hgb_predictions", "hgb_reference_protocol", "hgb_reference_report")
    if args.output is None or any(getattr(args, name) is None for name in input_names):
        parser.error("Every explicit input/reference path and --output is required")
    paths = {name: getattr(args, name).expanduser().resolve() for name in input_names}
    paths.update(driver=Path(__file__).resolve(), feature_calendar_helper=Path(cached.__file__).resolve(),
        reconstruction_helper=Path(metrics.__file__).resolve())
    if args.preserve_manifest:
        paths["preserve_manifest"] = args.preserve_manifest.resolve()
    sources = {name: dict(path=str(path), sha256=sha256(path)) for name, path in paths.items()}
    original = read_json(args.hgb_reference_report)
    if sha256(args.hgb_reference_protocol) != original["protocol_sha256"]:
        raise ValueError("Original HGB reference report/protocol mismatch")
    original_plan = read_json(args.hgb_reference_protocol)
    if original_plan["hgb_parameters"] != HGB_PARAMETERS or original_plan["expected_fit_rows"] != 20073:
        raise ValueError("Original HGB fixed fit procedure differs")
    provenance = read_json(args.feature_provenance)
    old_hashes = {entry["sha256"] for entry in original["source_inventory"]}
    for logical, name in (("fit", "fit_features"), ("confirmation", "confirmation_features"), ("later", "later_features")):
        entry = provenance["files"][logical]
        if sha256(paths[name]) != entry["sha256"] or entry["source_sha256"] not in old_hashes:
            raise ValueError(f"Feature serialization lineage differs: {logical}")
        if not entry["all_numeric_arrays_identical"] or not entry["UTC_timestamp_values_identical"]:
            raise ValueError("Converted cache changes numerical or timestamp content")
    authority = read_json(args.continuous_manifest)
    if authority["reference_report_sha256"] != sha256(args.hgb_reference_report):
        raise ValueError("Continuous input authority names a different HGB report")
    for name in ("fit_targets", "confirmation_targets", "later_targets", "confirmation_hgb_predictions", "later_hgb_predictions"):
        if sha256(paths[name]) != authority["files"][name]["sha256"]:
            raise ValueError(f"Original reviewed continuous cache differs: {name}")
    preserved = read_json(args.preserve_manifest)["files"] if args.preserve_manifest else []
    def verify_preserved():
        for entry in preserved:
            if sha256(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError("An earlier scientific artifact changed")
    verify_preserved()
    names = ["analytic_reference"] + ["hgb_" + name for name in CONTROLS]
    if args.mode in ("ridge", "both"):
        names += ["ridge_" + name for name in CONTROLS]
    protocol = dict(study="Fixed portable continuous linear/nonlinear controls v1", mode=args.mode,
        status="post_confirmation_exploratory_falsification",
        interpretation="Both evaluation periods and HGB outcomes were examined before this plan. New ridge outcomes were unknown at its initial freeze; no search, later fitting, calibration, neural fitting or fresh holdout.",
        methods=names, ridge_parameters=RIDGE_PARAMETERS, standardization="StandardScaler fitted only on the exact20073preNovfitrows, separate perfeaturecontrol; multioutputRidge",
        hgb_parameters=HGB_PARAMETERS, hgb_mode="cached original reference inridge mode; genuinely fit40regressors in hgb/both modes",
        fit_end_exclusive="2025-11-01T00:00:00+00:00", expected_fit_rows=20073,
        output_zones=original["output_zones"], output="center predicted residual;add identical analytic field;center;form all45pairs",
        primary_contrast="ridge_quotient84 minus ridge_analytic_current33 mainconfirmation meanabsolute pairwise error,7calendar-day blocks" if args.mode != "hgb" else "reproductionoforiginalfixedHGBcompletegrid",
        complete_pair_grid=[[names[j], names[i]] for i, j in itertools.combinations(range(len(names)), 2)],
        matched_contrasts="ridge minus HGB at each of12,33,48,84features",
        metrics=["pairwise_mae", "p90_timestamp_mae", "pairwise_rmse"], blocks=list(BLOCKS), replicates=REPLICATES, seed=SEED,
        uncertainty="shared circular UTC-calendar blocks including feed gaps;weightedlinear p90 equivalent to repeated sample; exploratory unadjusted intervals",
        no_continuous_calibration=True, no_matched_original_neural_claim=True, numeric_pickle_disabled=True,
        original_hgb_protocol_sha256=original["protocol_sha256"], sources=sources,
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "protocol.json"
    if frozen_path.exists():
        old = read_json(frozen_path)
        if {key: value for key, value in old.items() if key != "frozen_at_utc"} != protocol:
            raise RuntimeError("Executed protocol/source changed; preserve prior attempt before amending")
    elif args.protocol_only:
        atomic_json(frozen_path, {**protocol, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    else:
        raise RuntimeError("Run --protocol-only separately before fitting/scoring")
    if args.protocol_only:
        print(f"Fixed procedure and all input hashes frozen: {frozen_path}")
        return
    with threadpool_limits(limits=1):
        matrices, trusted, residual, identity = {}, {}, {}, {}
        for split, feature_path, target_path in (("fit", args.fit_features, args.fit_targets),
            ("confirmation", args.confirmation_features, args.confirmation_targets),
            ("later_consumed", args.later_features, args.later_targets)):
            matrices[split], trusted[split], residual[split], identity[split] = load_window(feature_path, target_path)
        if len(matrices["fit"][0]) != 20073 or matrices["fit"][3].max() >= pd.Timestamp(protocol["fit_end_exclusive"]):
            raise ValueError("Original pre-November risk-fit timestamp intersection differs")
        if len(matrices["confirmation"][0]) != original["confirmation"]["rows"] or len(matrices["later_consumed"][0]) != original["later_consumed"]["rows"]:
            raise ValueError("Original evaluation cohort sizes differ")
        recorded = {"confirmation": read_hgb_prediction(args.confirmation_hgb_predictions, matrices["confirmation"], trusted["confirmation"]),
            "later_consumed": read_hgb_prediction(args.later_hgb_predictions, matrices["later_consumed"], trusted["later_consumed"])}
        predictions = {split: {"analytic_reference": metrics.center(matrices[split][0][:, 2:12])}
            for split in ("confirmation", "later_consumed")}
        models, verification = {}, {}
        for name in CONTROLS:
            fit_x = cached.control_features(matrices["fit"][0], name)
            if args.mode in ("hgb", "both"):
                print(f"Refitting ten original fixed HGB residual regressors: {name}", flush=True)
                hgb = [HistGradientBoostingRegressor(**HGB_PARAMETERS).fit(fit_x, residual["fit"][:, zone]) for zone in range(10)]
                path = args.output / f"hgb_{name}_residual_regressors.joblib"
                joblib.dump(dict(regressors=hgb, parameters=HGB_PARAMETERS, output_zones=original["output_zones"]), path)
                models["hgb_" + name] = dict(sha256=sha256(path), iterations=[int(model.n_iter_) for model in hgb])
                for split in predictions:
                    x = cached.control_features(matrices[split][0], name)
                    field = metrics.center(matrices[split][0][:, 2:12] + metrics.center(np.column_stack([model.predict(x) for model in hgb])))
                    maximum = float(np.max(np.abs(field - recorded[split][name])))
                    if maximum > 1e-8:
                        raise ValueError(f"Historical HGB field reconstruction differs beyond tolerance: {split}/{name}: {maximum}")
                    verification[f"{split}/{name}"] = maximum
                    predictions[split]["hgb_" + name] = field
            else:
                for split in predictions:
                    predictions[split]["hgb_" + name] = recorded[split][name]
            if args.mode in ("ridge", "both"):
                print(f"Fitting one fixed multioutput scaled Ridge: {name}", flush=True)
                ridge = make_pipeline(StandardScaler(), Ridge(**RIDGE_PARAMETERS)).fit(fit_x, residual["fit"])
                path = args.output / f"ridge_{name}_residual_regressor.joblib"
                joblib.dump(dict(regressor=ridge, parameters=RIDGE_PARAMETERS, output_zones=original["output_zones"]), path)
                models["ridge_" + name] = dict(sha256=sha256(path), scaler_fit_rows=int(ridge[0].n_samples_seen_),
                    scaler_mean= ridge[0].mean_.tolist())
                for split in predictions:
                    x = cached.control_features(matrices[split][0], name)
                    predictions[split]["ridge_" + name] = metrics.center(matrices[split][0][:, 2:12] + metrics.center(ridge.predict(x)))
        audits = {}
        for split in predictions:
            predictions[split] = {name: predictions[split][name] for name in names}
            audits[split], losses = analyze(matrices[split][3], trusted[split], predictions[split])
            np.savez_compressed(args.output / f"{split}_predictions_private.npz",
                **{f"field_{name}": field for name, field in predictions[split].items()},
                **{f"timestamp_mae_{name}": loss for name, loss in losses.items()},
                timestamp_utc=matrices[split][3].astype(str).to_numpy(dtype=str))
    if any(sha256(path) != sources[name]["sha256"] for name, path in paths.items()):
        raise RuntimeError("An input or executed driver/helper changed during fitting")
    verify_preserved()
    result = dict(status="fixed_continuous_linear_nonlinear_audit_complete", mode=args.mode,
        protocol_sha256=sha256(frozen_path), sources=sources, source_hashes_unchanged=True,
        original_artifact_hashes_unchanged=True, original_hgb_field_verification=verification,
        interpretation=protocol["interpretation"], models=models, target_identity=identity,
        output_zones=protocol["output_zones"], **audits)
    atomic_json(args.output / "continuous_refit_audit.json", result)
    print(json.dumps(dict(status=result["status"], mode=args.mode,
        points={split: audit["point"] for split, audit in audits.items()},
        original_hgb_field_verification=verification), indent=2))


if __name__ == "__main__":
    main()
