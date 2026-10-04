"""Evaluate physical normalized-RAM percentage sensitivity with frozen v14 models.

This companion analysis does not retrain any model and does not modify the
frozen v14 confirmation artifacts.  It perturbs the normalized physical RAM
coordinate ``beta`` before the nonlinear feature map:

    beta' = (1 + delta) * beta
    s'    = asinh(beta' / S)

The experiment is a local input-sensitivity diagnostic with frozen shadow
prices.  It is not a counterfactual market clearing and does not establish the
physical accuracy of a response under a newly solved market state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from scripts.af_cdmo_real_controls_v14 import UniformMassDataset
    from scripts.af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
    )
    from scripts.benchmark_certificate_governance_v5 import _read_certificate
    from scripts.benchmark_multizone_quotient_gauge_v6 import (
        _build_vector_samples,
        _load_price_panel,
        _parse_utc,
    )
    from scripts.benchmark_af_cdmo_real_v14 import _ensemble
    from scripts.cqdm_utils import sha256_file
    from scripts.multizone_quotient_gauge_v6 import (
        MultiZoneRegressor,
        predict_multizone,
    )
except ModuleNotFoundError:
    from af_cdmo_real_controls_v14 import UniformMassDataset
    from af_cdmo_real_extension_v13 import RealAffineGaugeDataset, RealAffineGeometry
    from benchmark_certificate_governance_v5 import _read_certificate
    from benchmark_multizone_quotient_gauge_v6 import (
        _build_vector_samples,
        _load_price_panel,
        _parse_utc,
    )
    from benchmark_af_cdmo_real_v14 import _ensemble
    from cqdm_utils import sha256_file
    from multizone_quotient_gauge_v6 import MultiZoneRegressor, predict_multizone


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "official_jao_dual_measure"
DEFAULT_REPORT = ARTIFACT_ROOT / "af_cdmo_real_confirmation_v14.json"
DEFAULT_PROGRESS = (
    ARTIFACT_ROOT / "af_cdmo_real_confirmation_v14_evaluation_progress.pt"
)
DEFAULT_CHECKPOINT_DIR = ARTIFACT_ROOT / "af_cdmo_real_confirmation_v14_checkpoints"
DEFAULT_DUAL = ARTIFACT_ROOT / "NordicBindingDualMeasure_2025_2026_full_history.csv.gz"
DEFAULT_GEOMETRY = ARTIFACT_ROOT / "af_cdmo_real_geometry_audit_v13.json"
DEFAULT_PRICE_ROOT = ROOT / "official_entsoe_nordic_day_ahead"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "af_cdmo_physical_ram_sensitivity_v15.json"
PERCENTAGE_CHANGES = (-0.20, -0.05, 0.05, 0.20)
METHODS = ("af_qdm_residual", "af_uniform_attention_residual")


class PhysicalPercentageRAMDataset(Dataset):
    """Scale normalized physical RAM before its frozen asinh feature map."""

    def __init__(self, base: RealAffineGaugeDataset, *, delta: float) -> None:
        value = float(delta)
        if not np.isfinite(value) or value <= -1.0:
            raise ValueError("RAM percentage delta must be finite and greater than -1")
        if base.representation != "af":
            raise ValueError("Physical RAM sensitivity requires the AF representation")
        self.base = base
        self.delta = value

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        canonical = np.asarray(item["canonical"], dtype=np.float32).copy()
        if len(canonical):
            ram_index = self.base.ptdf_count
            transformed = canonical[:, ram_index].astype(np.float64)
            normalized_ram = np.sinh(transformed)
            transformed_new = np.arcsinh((1.0 + self.delta) * normalized_ram)
            if not np.isfinite(transformed_new).all():
                raise RuntimeError("Physical RAM transformation produced non-finite values")
            canonical[:, ram_index] = transformed_new.astype(np.float32)
        item["canonical"] = canonical
        return item


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _summary(perturbed: np.ndarray, original: np.ndarray) -> dict[str, Any]:
    drift = np.abs(np.asarray(perturbed, dtype=np.float64) - original)
    timestamp_max = drift.max(axis=1)
    return {
        "mean_abs_pairwise_drift_eur_mwh": float(drift.mean()),
        "median_abs_pairwise_drift_eur_mwh": float(np.median(drift)),
        "p90_abs_pairwise_drift_eur_mwh": float(np.quantile(drift, 0.90)),
        "p95_abs_pairwise_drift_eur_mwh": float(np.quantile(drift, 0.95)),
        "p99_abs_pairwise_drift_eur_mwh": float(np.quantile(drift, 0.99)),
        "max_abs_pairwise_drift_eur_mwh": float(drift.max(initial=0.0)),
        "timestamp_max_drift_eur_mwh": {
            "median": float(np.median(timestamp_max)),
            "p90": float(np.quantile(timestamp_max, 0.90)),
            "p95": float(np.quantile(timestamp_max, 0.95)),
            "p99": float(np.quantile(timestamp_max, 0.99)),
            "maximum": float(timestamp_max.max(initial=0.0)),
        },
        "timestamp_share_above": {
            "1e-3_eur_mwh": float(np.mean(timestamp_max > 1.0e-3)),
            "1e-2_eur_mwh": float(np.mean(timestamp_max > 1.0e-2)),
            "1e-1_eur_mwh": float(np.mean(timestamp_max > 1.0e-1)),
            "1_eur_mwh": float(np.mean(timestamp_max > 1.0)),
        },
    }


def _expected_checkpoint_hashes(report: dict[str, Any]) -> dict[str, str]:
    records = report["reproducibility"]["model_checkpoints"]["files"]
    return {Path(record["path"]).name: record["sha256"] for record in records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--geometry_audit", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--price_root", type=Path, default=DEFAULT_PRICE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reproduction_tolerance", type=float, default=2.0e-5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "report": args.report.expanduser().resolve(),
        "progress": args.progress.expanduser().resolve(),
        "checkpoint_dir": args.checkpoint_dir.expanduser().resolve(),
        "dual": args.dual.expanduser().resolve(),
        "geometry": args.geometry_audit.expanduser().resolve(),
        "price_root": args.price_root.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    if paths["output"].exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {paths['output']}; pass --force explicitly"
        )
    if args.batch_size < 1 or args.reproduction_tolerance <= 0.0:
        raise ValueError("Batch size and reproduction tolerance must be positive")
    for name in ("report", "progress", "dual", "geometry"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    if not paths["checkpoint_dir"].is_dir() or not paths["price_root"].is_dir():
        raise FileNotFoundError("Checkpoint or price directory is missing")

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    protocol = report["protocol"]
    if protocol.get("study_role") != "confirmation":
        raise RuntimeError("RAM v15 must use the frozen confirmation study")
    seeds = [int(value) for value in protocol["seeds"]]
    zones = tuple(protocol["zones"])
    if len(seeds) != 10:
        raise RuntimeError("Frozen confirmation must contain ten seeds")

    if sha256_file(paths["dual"]) != report["source"]["dual_archive_sha256"]:
        raise RuntimeError("Dual archive hash differs from the frozen confirmation")
    if sha256_file(paths["geometry"]) != report["source"]["geometry_audit_sha256"]:
        raise RuntimeError("Geometry audit hash differs from the frozen confirmation")
    expected_progress_hash = report["reproducibility"]["evaluation_progress"]["sha256"]
    if sha256_file(paths["progress"]) != expected_progress_hash:
        raise RuntimeError("Evaluation-progress hash differs from the frozen confirmation")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    checkpoint_hashes = _expected_checkpoint_hashes(report)
    first_checkpoint_path = paths["checkpoint_dir"] / f"{METHODS[0]}__seed{seeds[0]}.pt"
    first_checkpoint = torch.load(
        first_checkpoint_path, map_location="cpu", weights_only=False
    )
    frozen_payload = first_checkpoint["training_signature_payload"]
    target_scale = float(frozen_payload["target_scale"])
    canonical_ram_scale = float(frozen_payload["canonical_ram_scale"])
    if not np.isclose(target_scale, report["target_scale_eur_mwh"], atol=1.0e-12):
        raise RuntimeError("Frozen target scale does not match the confirmation report")

    geometry_audit = json.loads(paths["geometry"].read_text(encoding="utf-8"))
    frame, certificate_audit = _read_certificate(paths["dual"], 60.0)
    unit_columns = certificate_audit["unit_columns"]
    input_zones = tuple(column.removeprefix("unit_ptdf_") for column in unit_columns)
    geometry = RealAffineGeometry.loss_safe(input_zones)
    if list(input_zones) != geometry_audit["geometry"]["zone_order"]:
        raise RuntimeError("Current PTDF order differs from the frozen geometry audit")

    price_panel, price_provenance = _load_price_panel(paths["price_root"], zones)
    frozen_price_provenance = report["source"]["day_ahead_prices"]
    for zone in zones:
        if price_provenance[zone]["sha256"] != frozen_price_provenance[zone]["sha256"]:
            raise RuntimeError(f"Price source hash differs for {zone}")

    fit_end = _parse_utc(protocol["fit_end_exclusive"])
    validation_end = _parse_utc(protocol["internal_validation_end_exclusive"])
    evaluation_end = _parse_utc(protocol["evaluation_end_exclusive"])
    samples = _build_vector_samples(
        frame, price_panel, unit_columns, zones, fit_end, evaluation_end
    )
    evaluation_samples = [
        sample
        for sample in samples
        if validation_end <= sample["timestamp"] < evaluation_end
    ]
    if len(evaluation_samples) != int(report["rows"]["evaluation"]):
        raise RuntimeError("Reconstructed evaluation row count differs from v14")

    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    raw_scale = np.asarray(frozen_payload["raw_scales"]["af"], dtype=np.float64)
    base_dataset = RealAffineGaugeDataset(
        evaluation_samples,
        raw_scale=raw_scale,
        target_scale=target_scale,
        presentation="original",
        representation="af",
        real_zone_indices=real_indices,
        geometry=geometry,
        canonical_ram_scale=canonical_ram_scale,
        analytic_tolerance=float(protocol["analytic_invariance_tolerance_eur_mwh"]),
    )

    progress = torch.load(paths["progress"], map_location="cpu", weights_only=False)
    original_predictions = {
        method: np.asarray(progress["original_predictions"][method], dtype=np.float64)
        for method in METHODS
    }
    expected_shape = (len(evaluation_samples), len(zones) * (len(zones) - 1) // 2)
    if any(value.shape != expected_shape for value in original_predictions.values()):
        raise RuntimeError("Frozen original prediction array has an unexpected shape")

    raw_dim = 1 + len(unit_columns) + 5
    canonical_dim = len(unit_columns) + 5
    models: dict[str, list[MultiZoneRegressor]] = {method: [] for method in METHODS}
    checkpoint_records = []
    for method in METHODS:
        for seed in seeds:
            checkpoint_path = paths["checkpoint_dir"] / f"{method}__seed{seed}.pt"
            expected_hash = checkpoint_hashes.get(checkpoint_path.name)
            if expected_hash is None or sha256_file(checkpoint_path) != expected_hash:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if checkpoint.get("method") != method or int(checkpoint.get("seed")) != seed:
                raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
            if checkpoint.get("training_signature") != first_checkpoint.get(
                "training_signature"
            ):
                raise RuntimeError("Frozen checkpoints do not share one training contract")
            model = MultiZoneRegressor(
                "cqdm_gauge_residual",
                raw_dim,
                canonical_dim,
                len(zones),
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            models[method].append(model)
            checkpoint_records.append(
                {
                    "method": method,
                    "seed": seed,
                    "path": str(checkpoint_path),
                    "sha256": expected_hash,
                }
            )

    def predict(method: str, dataset: Dataset) -> tuple[np.ndarray, list[np.ndarray]]:
        view: Dataset = (
            UniformMassDataset(dataset)
            if method == "af_uniform_attention_residual"
            else dataset
        )
        members = [
            predict_multizone(
                model,
                view,
                batch_size=args.batch_size,
                target_scale=target_scale,
                device=device,
            )
            for model in models[method]
        ]
        ensemble = _ensemble(members)
        return np.asarray(ensemble["pairwise"], dtype=np.float64), [
            np.asarray(member["pairwise"], dtype=np.float64) for member in members
        ]

    reproduction = {}
    baseline_dataset = PhysicalPercentageRAMDataset(base_dataset, delta=0.0)
    for method in METHODS:
        rebuilt, _ = predict(method, baseline_dataset)
        drift = np.abs(rebuilt - original_predictions[method])
        maximum = float(drift.max(initial=0.0))
        reproduction[method] = {
            "maximum_abs_pairwise_difference_eur_mwh": maximum,
            "tolerance_eur_mwh": args.reproduction_tolerance,
            "passed": maximum <= args.reproduction_tolerance,
        }
        if maximum > args.reproduction_tolerance:
            raise RuntimeError(
                f"Frozen original prediction reproduction failed for {method}: {maximum}"
            )

    methods: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for delta in PERCENTAGE_CHANGES:
        perturbed_dataset = PhysicalPercentageRAMDataset(base_dataset, delta=delta)
        for method in METHODS:
            ensemble, members = predict(method, perturbed_dataset)
            member_maxima = [
                float(np.abs(member - original_predictions[method]).max(initial=0.0))
                for member in members
            ]
            methods[method].append(
                {
                    "normalized_physical_ram_fractional_change": delta,
                    **_summary(ensemble, original_predictions[method]),
                    "seed_member_max_abs_drift_eur_mwh": {
                        "median": float(np.median(member_maxima)),
                        "minimum": float(np.min(member_maxima)),
                        "maximum": float(np.max(member_maxima)),
                    },
                }
            )
        print(f"[RAM] completed delta={delta:+.0%}", flush=True)

    output = {
        "protocol": {
            "name": "af_cdmo_physical_normalized_ram_sensitivity_v15",
            "study_role": "post_confirmation_diagnostic",
            "models_retrained": False,
            "frozen_confirmation_rows": len(evaluation_samples),
            "seeds": seeds,
            "zones": list(zones),
            "percentage_changes": list(PERCENTAGE_CHANGES),
            "transformation": (
                "beta_prime=(1+delta)*beta; "
                "s_prime=asinh(beta_prime/canonical_ram_scale)"
            ),
            "interpretation": (
                "Local model-input sensitivity with frozen shadow prices; not a "
                "counterfactual market clearing or physical-response validation."
            ),
        },
        "sources": {
            "frozen_report": {
                "path": str(paths["report"]),
                "sha256": sha256_file(paths["report"]),
            },
            "frozen_evaluation_progress": {
                "path": str(paths["progress"]),
                "sha256": expected_progress_hash,
            },
            "dual_archive": {
                "path": str(paths["dual"]),
                "sha256": sha256_file(paths["dual"]),
            },
            "geometry_audit": {
                "path": str(paths["geometry"]),
                "sha256": sha256_file(paths["geometry"]),
            },
            "engine": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "checkpoints": checkpoint_records,
        },
        "frozen_original_reproduction": reproduction,
        "methods": methods,
    }
    _atomic_json(paths["output"], output)
    print(f"[OK] wrote {paths['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
