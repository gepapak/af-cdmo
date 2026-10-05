"""Replay the complete exploratory reconstruction grid from authorized caches.

All paths are explicit. This program never reads provider files, loads models,
fits a regressor, or runs a neural network. Numeric NPZ inputs require pickle
disabled. The original protocol/report and new replay protocol remain distinct.
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
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
try:
    from . import paper2_reconstruction_cached_metrics as metrics
    from . import paper2_nested_cached_metrics as cached
except ImportError:
    import paper2_reconstruction_cached_metrics as metrics
    import paper2_nested_cached_metrics as cached


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_times(raw: np.ndarray, name: str) -> pd.DatetimeIndex:
    if raw.ndim != 1 or raw.dtype.kind not in "SU":
        raise ValueError(f"Fixed-width string timestamps required, without pickle: {name}")
    times = pd.DatetimeIndex(pd.to_datetime(raw.astype(str), utc=True))
    if not len(times) or times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError(f"Unique chronological UTC timestamps required: {name}")
    return times


def read_window(target_path: Path, prediction_path: Path, zones: list[str]) -> tuple:
    with np.load(target_path, allow_pickle=False) as target_cache:
        times = check_times(target_cache["timestamp_utc"], str(target_path))
        target = target_cache["target_field"]
        analytic = target_cache["analytic_field"]
        residual = target_cache["residual_field"]
    with np.load(prediction_path, allow_pickle=False) as prediction_cache:
        if not times.equals(check_times(prediction_cache["timestamp_utc"], str(prediction_path))):
            raise ValueError("Prediction and target timestamps differ")
        predictions = {method: prediction_cache["field_" + method] for method in metrics.METHODS}
        stored_loss = {method: prediction_cache["timestamp_mae_" + method] for method in metrics.METHODS}
    if len(zones) != 10 or len(set(zones)) != 10:
        raise ValueError("Original ten-zone order required")
    for name, field in {"target": target, "analytic": analytic, "residual": residual, **predictions}.items():
        if field.shape != (len(times), 10) or not np.issubdtype(field.dtype, np.number) or not np.isfinite(field).all():
            raise ValueError(f"Invalid finite ten-zone field: {name}")
    if np.max(np.abs(target - metrics.center(target))) > 1e-8:
        raise ValueError("Cached trusted target is not centered")
    if np.max(np.abs(metrics.center(target - analytic) - residual)) > 1e-8:
        raise ValueError("Cached residual identity differs")
    if np.max(np.abs(metrics.center(analytic) - predictions["analytic_reference"])) > 1e-8:
        raise ValueError("Cached analytic-reference identity differs")
    for method, values in stored_loss.items():
        recomputed = np.abs(metrics.pairs(predictions[method]) - metrics.pairs(target)).mean(axis=1)
        if values.shape != (len(times),) or not np.isfinite(values).all() or np.max(np.abs(recomputed - values)) > 1e-8:
            raise ValueError(f"Original per-timestamp pairwise loss differs: {method}")
    return (None, None, None, times), target, predictions


def verify_grid(candidate: dict, reference: dict) -> float:
    maximum = 0.0
    def visit(value, original, path):
        nonlocal maximum
        if isinstance(value, dict):
            if set(value) != set(original):
                raise ValueError(f"Grid keys differ: {path}")
            for key in value:
                visit(value[key], original[key], path + "/" + key)
        elif isinstance(value, list):
            if len(value) != len(original):
                raise ValueError(f"Grid lengths differ: {path}")
            for index, item in enumerate(value):
                visit(item, original[index], path + "/" + str(index))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if not np.isfinite(value) or not np.isfinite(original):
                raise ValueError(f"Nonfinite reference/replay statistic: {path}")
            difference = abs(float(value) - float(original))
            maximum = max(maximum, difference)
            if difference > 1e-8:
                raise ValueError(f"Reference reconstruction statistic differs: {path}: {difference}")
        elif value != original:
            raise ValueError(f"Reference reconstruction value differs: {path}")
    visit(candidate, reference, "window")
    return maximum


def self_test() -> None:
    rng = np.random.default_rng(119)
    for size in (2, 8, 27):
        values = np.sort(rng.normal(size=size))
        weights = rng.integers(0, 6, size=size)
        weights[0] = 1
        for q in (0, .5, .9, 1):
            expected = np.quantile(np.repeat(values, weights), q)
            assert abs(metrics.weighted_quantile(values, weights, q) - expected) < 1e-12
    field = rng.normal(size=(5, 10))
    assert np.allclose(metrics.pairs(field), metrics.pairs(metrics.center(field)))
    assert np.allclose(metrics.center(field).sum(axis=1), 0)
    target = np.zeros((2, 10))
    times = pd.DatetimeIndex(pd.to_datetime(["2026-01-01T00:00Z", "2026-01-03T00:00Z"], utc=True))
    # Jan 2 has no observations and remains a calendar day in every bootstrap.
    calendar = pd.date_range(times.min().normalize(), times.max().normalize(), freq="D")
    assert np.array_equal(calendar.get_indexer(times.normalize()), [0, 2])
    weights = cached.calendar_weights(3, 17, 77, block_days=7)
    assert weights.shape == (17, 3) and np.all(weights.sum(axis=1) == 3)
    predictions = {name: target.copy() for name in metrics.METHODS}
    candidate, losses = metrics.analyze((None, None, None, times), target, predictions)
    assert all(value["pairwise_mae"] == 0 for value in candidate["point"].values())
    assert all(value["ci95"] == [0., 0.] for pair in candidate["paired"]["7"].values() for value in pair.values())
    assert all(np.array_equal(loss, np.zeros(2)) for loss in losses.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    for name in ("confirmation-targets", "confirmation-predictions", "later-targets", "later-predictions", "reference-protocol", "reference-report", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Expanded weighted quantiles, pair/centering identities, complete zero-loss grid, and missing-calendar-day checks passed.")
        return
    input_names = ("confirmation_targets", "confirmation_predictions", "later_targets", "later_predictions", "reference_protocol", "reference_report")
    if args.output is None or any(getattr(args, name) is None for name in input_names):
        parser.error("Every cache/reference path and --output is required")
    reference = read_json(args.reference_report)
    if sha256(args.reference_protocol) != reference["protocol_sha256"]:
        raise ValueError("Original report does not match the supplied executed protocol")
    paths = {name: getattr(args, name).resolve() for name in input_names}
    paths.update(replay_driver=Path(__file__).resolve(), numerical_helper=Path(metrics.__file__).resolve(), calendar_helper=Path(cached.__file__).resolve())
    sources = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    protocol = dict(study="Portable cache-only reconstruction replay v1",
        status="reproduction_of_already_examined_exploratory_outputs",
        procedure="No fit/model load/provider acquisition/neural inference; complete original five-method, ten-pair, three-metric, 7/14/28-day grid",
        original_protocol_sha256=reference["protocol_sha256"], sources=sources,
        methods=list(metrics.METHODS), output_zones=reference["output_zones"],
        tolerance=1e-8, replicates=cached.REPLICATES, bootstrap_seed=cached.SEED,
        p90="Original linear quantile over per-timestamp mean absolute pair errors; integer calendar weights equivalent to explicit repeat",
        software={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        interpretation="Original outcomes already examined; this replay creates no fresh confirmation. NN fit chronology differs. No neural necessity or deployed market-benefit claim.")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        frozen = read_json(protocol_path)
        if {key: value for key, value in frozen.items() if key != "frozen_at_utc"} != protocol:
            raise RuntimeError("Replay protocol/source changed; preserve the old attempt before amending")
    else:
        if not args.protocol_only:
            raise RuntimeError("Run --protocol-only separately before replay")
        atomic_json(protocol_path, {**protocol, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    if args.protocol_only:
        print(f"Replay inputs and numerical procedure frozen: {protocol_path}")
        return
    windows, verification = {}, {}
    with threadpool_limits(limits=1):
        for window, target_path, prediction_path in (("confirmation", args.confirmation_targets, args.confirmation_predictions),
                                                     ("later_consumed", args.later_targets, args.later_predictions)):
            matrix, target, predictions = read_window(target_path, prediction_path, reference["output_zones"])
            windows[window], _ = metrics.analyze(matrix, target, predictions)
            verification[window] = {"maximum_complete_grid_error": verify_grid(windows[window], reference[window]),
                                    "target_prediction_timestamp_alignment": True, "original_per_timestamp_losses_match": True}
    if any(sha256(path) != sources[name]["sha256"] for name, path in paths.items()):
        raise RuntimeError("An input or replay source changed during execution")
    result = dict(status="complete_original_exploratory_reconstruction_grid_reproduced", protocol_sha256=sha256(protocol_path),
                  source_hashes_unchanged=True, source_inventory=sources, verification=verification,
                  output_zones=reference["output_zones"], interpretation=protocol["interpretation"], **windows)
    atomic_json(args.output / "fixed_reconstruction_audit.json", result)
    print(json.dumps({"status": result["status"], "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
