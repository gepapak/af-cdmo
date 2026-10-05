"""Input relocation and contracts for code-only Paper2 secondary auditors.

No provider data are downloaded or embedded. Original input hashes remain the
authority after relocation; adapting a path never changes an expected digest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, help="Authorized relocated Paper2 input tree; expected hashes stay fixed")
    for name in ("dual-archive", "geometry-audit", "development-artifact", "confirmation-reference", "price-root"):
        parser.add_argument("--" + name, type=Path)


def relocated(path: str, input_root: Path | None) -> Path:
    """Resolve an original report path under an explicit authorized input root."""
    if input_root is None:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Supply an explicit input path or --input-root for {candidate.name}")
        return candidate.resolve()
    parts = PureWindowsPath(path).parts if "\\" in path or (len(path) > 1 and path[1] == ":") else Path(path).parts
    indices = [index for index, value in enumerate(parts) if value.lower() == "paper2"]
    if not indices:
        if Path(path).is_absolute() or PureWindowsPath(path).is_absolute():
            raise ValueError("Absolute source path has no Paper2 anchor; supply its explicit override")
        suffix = parts
    else:
        suffix = parts[indices[-1] + 1:]
    root = input_root.expanduser().resolve()
    candidate = root.joinpath(*suffix).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Relocated source escapes the explicit input root")
    if not candidate.is_file() and not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def map_report_sources(report: dict, args: argparse.Namespace, root: Path, names: tuple[str, ...] | None = None) -> dict:
    """Return an in-memory path map; never edit the frozen report bytes."""
    mapped = copy.deepcopy(report)
    sources = mapped["sources"]
    for name in names or ("dual_archive", "geometry_audit", "development_artifact", "confirmation_reference", "price_root"):
        if name not in sources:
            continue
        explicit = getattr(args, name, None)
        sources[name] = str(explicit.expanduser().resolve() if explicit else relocated(sources[name], args.input_root))
    for name in ("confirmation_engine", "protocol_note"):
        if name in sources:
            filename = PureWindowsPath(sources[name]).name
            sources[name] = str(root / "af_cdmo_probabilistic_headroom" / filename)
    price_root = Path(sources["price_root"]) if "price_root" in sources else None
    for entry in sources.get("price_provenance", {}).values():
        filename = PureWindowsPath(entry["path"]).name
        entry["path"] = str(price_root / filename) if price_root else str(relocated(entry["path"], args.input_root))
    return mapped


def manifest_check(path: Path, manifest: dict, logical_path: str) -> dict:
    records = [item for item in manifest["files"] if item["path"].replace("\\", "/") == logical_path]
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one frozen manifest record for {logical_path}")
    digest = sha256(path)
    if digest != records[0]["sha256"].lower():
        raise RuntimeError(f"Frozen manifest hash mismatch: {logical_path}")
    return dict(path=str(path.resolve()), sha256=digest, verified_against_frozen_manifest=True)


def validate_predictions(timestamps, target, magnitude, probabilities: dict, tail_threshold: float) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    target = np.asarray(target, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    if not len(times) or not times.is_unique or not times.is_monotonic_increasing or times.hasnans:
        raise ValueError("Prediction timestamps must be nonempty, unique, ordered and valid")
    if target.shape != (len(times),) or magnitude.shape != (len(times),):
        raise ValueError("Prediction targets/magnitudes do not align with timestamps")
    if not np.isfinite(tail_threshold) or not np.all(np.isfinite(magnitude)) or np.any(magnitude < 0):
        raise ValueError("Finite nonnegative residual magnitudes and a finite threshold are required")
    if not np.all(np.isin(target, [0.0, 1.0])) or not np.array_equal(target, (magnitude > tail_threshold).astype(float)):
        raise ValueError("Cached labels differ from the exact original threshold definition")
    for method, probability in probabilities.items():
        values = np.asarray(probability, dtype=float)
        if values.shape != (len(times),) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid or unaligned probabilities: {method}")
    return times


def aligned_hgb_csv(path: Path, timestamps: pd.DatetimeIndex, target: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Accept only explicitly aligned, externally supplied CSV or NPZ scores."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            frame = pd.DataFrame({name: archive[name] for name in archive.files})
    else:
        frame = pd.read_csv(path)
    column = "probability_hgb_quotient" if "probability_hgb_quotient" in frame else "probability"
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    if not times.equals(timestamps):
        raise ValueError("Supplied HGB timestamps must exactly match the cached cohort order")
    target_column = "tail_target" if "tail_target" in frame else "target"
    if not np.array_equal(frame[target_column].to_numpy(dtype=float), target):
        raise ValueError("Supplied HGB targets differ from the cached cohort")
    if not np.array_equal(frame["residual_magnitude_eur_mwh"].to_numpy(dtype=float), magnitude):
        raise ValueError("Supplied HGB magnitudes differ from the cached cohort")
    probability = frame[column].to_numpy(dtype=float)
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Supplied HGB probabilities must be finite values in [0,1]")
    return probability
