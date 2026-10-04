"""Frozen-method external replication of AF-CDMO on Core-CCR certificates.

Weights are refit chronologically because Core and Nordic have different input
and output dimensions.  Architecture, optimization, representations, stress
orbit, seeds, and metrics remain frozen.  This is method-level external
replication, not zero-shot neural-weight transfer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from scripts.af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
        presentation_names,
        raw_rows_for_scale,
    )
    from scripts.core_external_replication import (
        CORE_CANONICAL_SCALAR_COLUMNS,
        CORE_PHYSICAL_ZONES,
        CORE_RAW_TAIL_COLUMNS,
        atomic_json,
        global_balance_geometry,
        parse_utc,
        sha256_file,
    )
    from scripts.multizone_quotient_gauge_v6 import (
        center_price_field,
        cycle_inconsistency,
        field_to_pairwise,
        fit_multizone,
        predict_multizone,
    )
except ModuleNotFoundError:
    from af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
        presentation_names,
        raw_rows_for_scale,
    )
    from core_external_replication import (
        CORE_CANONICAL_SCALAR_COLUMNS,
        CORE_PHYSICAL_ZONES,
        CORE_RAW_TAIL_COLUMNS,
        atomic_json,
        global_balance_geometry,
        parse_utc,
        sha256_file,
    )
    from multizone_quotient_gauge_v6 import (
        center_price_field,
        cycle_inconsistency,
        field_to_pairwise,
        fit_multizone,
        predict_multizone,
    )


METHODS: dict[str, tuple[str, str] | None] = {
    "analytic_gauge": None,
    "ambient_raw_deepset": ("raw_deepset_gauge", "ambient"),
    "projection_only_raw_deepset": ("raw_deepset_gauge", "af"),
    "ambient_cqdm_residual": ("cqdm_gauge_residual", "ambient"),
    "core_rank1_af_qdm_residual": ("cqdm_gauge_residual", "af"),
}


class MaterializedDataset(Dataset):
    """Immutable cache of a deterministic certificate presentation.

    Training and validation always use the clean ``original`` presentation,
    which has no epoch-dependent augmentation. Materializing it once removes
    repeated projection/canonicalization without changing any model input.
    """

    def __init__(self, source: Dataset) -> None:
        self.items = [source[index] for index in range(len(source))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    def set_epoch(self, epoch: int) -> None:
        del epoch


def _positive_scales(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or not len(values):
        raise ValueError("Scale estimation requires a non-empty matrix")
    scales = np.ones(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        finite = np.abs(values[:, index])
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        if len(finite):
            scales[index] = max(float(np.median(finite)), 1.0e-6)
    return scales


def _resolve_artifact(
    manifest_path: Path, manifest: dict[str, Any], name: str
) -> Path:
    artifact = manifest.get("artifacts", {}).get(name, {})
    path = Path(str(artifact.get("path", "")))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Core artifact '{name}' is missing: {path}")
    expected = artifact.get("sha256")
    if expected and sha256_file(path) != expected:
        raise RuntimeError(f"Core artifact '{name}' hash mismatch")
    return path


def _read_inputs(
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "core_ccr_external_replication_v1":
        raise RuntimeError("Unexpected Core external-data manifest protocol")
    dual_path = _resolve_artifact(manifest_path, manifest, "dual")
    price_path = _resolve_artifact(manifest_path, manifest, "centered_price_field")
    dual = pd.read_csv(dual_path)
    price = pd.read_csv(price_path)
    dual["delivery_utc"] = pd.to_datetime(
        dual["delivery_utc"], utc=True, errors="raise", format="mixed"
    )
    dual["publication_utc"] = pd.to_datetime(
        dual["publication_utc"], utc=True, errors="raise", format="mixed"
    )
    price["delivery_utc"] = pd.to_datetime(
        price["delivery_utc"], utc=True, errors="raise", format="mixed"
    )
    if price["delivery_utc"].duplicated().any():
        raise RuntimeError("Core centered price field contains duplicate timestamps")
    if (dual["publication_utc"] > dual["delivery_utc"]).any():
        raise RuntimeError("Core certificate includes post-delivery publication rows")
    return dual, price, manifest, dual_path, price_path


def _build_samples(
    dual: pd.DataFrame,
    price: pd.DataFrame,
    *,
    unit_columns: list[str],
    fit_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    fit = dual.loc[dual["delivery_utc"] < fit_end]
    if fit.empty:
        raise RuntimeError("No Core dual rows precede the fit boundary")
    scalar_scales: dict[str, float] = {}
    for column in CORE_CANONICAL_SCALAR_COLUMNS:
        finite = np.abs(fit[column].fillna(0.0).to_numpy(dtype=np.float64))
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        scalar_scales[column] = max(float(np.median(finite)), 1.0e-6) if len(finite) else 1.0

    physical_indices = np.asarray(
        [unit_columns.index(f"unit_hub_{zone}") for zone in CORE_PHYSICAL_ZONES],
        dtype=np.int64,
    )
    groups: dict[pd.Timestamp, dict[str, Any]] = {}
    for delivery, group in dual.groupby("delivery_utc", sort=True):
        unit = group[unit_columns].to_numpy(dtype=np.float64)
        norm = group["ptdf_l2_norm"].to_numpy(dtype=np.float64)
        shadow = group["shadowPrice"].to_numpy(dtype=np.float64)
        rhs = group["ram"].to_numpy(dtype=np.float64)
        raw_tail = group[list(CORE_RAW_TAIL_COLUMNS)].fillna(0.0).to_numpy(
            dtype=np.float64
        )
        raw = np.column_stack(
            [shadow, unit * norm[:, None], rhs, raw_tail]
        ).astype(np.float32)
        canonical_scalars = np.column_stack(
            [
                np.arcsinh(
                    group[column].fillna(0.0).to_numpy(dtype=np.float64)
                    / scalar_scales[column]
                )
                for column in CORE_CANONICAL_SCALAR_COLUMNS
            ]
        )
        canonical = np.column_stack([unit, canonical_scalars]).astype(np.float32)
        mass = group["dual_mass_eur_mwh"].to_numpy(dtype=np.float64)
        total = float(mass.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError(f"Non-positive Core dual mass at {delivery}")
        weights = (mass / total).astype(np.float32)
        analytic = center_price_field(-(mass @ unit[:, physical_indices]))
        groups[pd.Timestamp(delivery)] = {
            "raw": raw,
            "canonical": canonical,
            "weights": weights,
            "total": total,
            "analytic_vector": analytic.astype(np.float32),
        }

    target_columns = [
        f"price_{zone}_centered_eur_mwh" for zone in CORE_PHYSICAL_ZONES
    ]
    missing = sorted(set(target_columns).difference(price.columns))
    if missing:
        raise RuntimeError(f"Core centered-price artifact omitted columns: {missing}")
    raw_dim = 1 + len(unit_columns) + 5
    canonical_dim = len(unit_columns) + 5
    start = min(dual["delivery_utc"].min(), price["delivery_utc"].min())
    samples: list[dict[str, Any]] = []
    for _, row in price.sort_values("delivery_utc").iterrows():
        timestamp = pd.Timestamp(row["delivery_utc"])
        if timestamp < start or timestamp >= evaluation_end:
            continue
        target = center_price_field(row[target_columns].to_numpy(dtype=np.float64))
        certificate = groups.get(
            timestamp,
            {
                "raw": np.empty((0, raw_dim), dtype=np.float32),
                "canonical": np.empty((0, canonical_dim), dtype=np.float32),
                "weights": np.empty(0, dtype=np.float32),
                "total": 0.0,
                "analytic_vector": np.zeros(len(CORE_PHYSICAL_ZONES), dtype=np.float32),
            },
        )
        samples.append(
            {
                "timestamp": timestamp,
                "target_vector": target.astype(np.float32),
                **certificate,
            }
        )
    if not samples:
        raise RuntimeError("No Core price/certificate samples were built")
    return samples, scalar_scales


def _split_samples(
    samples: list[dict[str, Any]],
    fit_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = [sample for sample in samples if sample["timestamp"] < fit_end]
    validation = [
        sample
        for sample in samples
        if fit_end <= sample["timestamp"] < validation_end
    ]
    evaluation = [
        sample
        for sample in samples
        if validation_end <= sample["timestamp"] < evaluation_end
    ]
    return train, validation, evaluation


def _analytic_prediction(dataset: RealAffineGaugeDataset) -> dict[str, Any]:
    fields = []
    targets = []
    timestamps = []
    for index in range(len(dataset)):
        item = dataset[index]
        fields.append(item["analytic_vector"])
        targets.append(item["target_vector"])
        timestamps.append(item["timestamp"])
    field = np.stack(fields).astype(np.float64)
    return {
        "field": field,
        "pairwise": field_to_pairwise(field),
        "target": np.stack(targets).astype(np.float64),
        "timestamps": timestamps,
    }


def _prediction_metrics(
    prediction: dict[str, Any],
    *,
    alert_threshold_eur_mwh: float,
) -> dict[str, float]:
    target_field = np.asarray(prediction["target"], dtype=np.float64)
    target_pair = field_to_pairwise(target_field)
    predicted_pair = np.asarray(prediction["pairwise"], dtype=np.float64)
    error = predicted_pair - target_pair
    timestamp_mae = np.mean(np.abs(error), axis=1)
    active = np.abs(target_pair) >= 1.0
    if np.any(active):
        sign_accuracy = float(
            np.mean(np.sign(predicted_pair[active]) == np.sign(target_pair[active]))
        )
    else:
        sign_accuracy = float("nan")
    target_alert = np.abs(target_pair) >= alert_threshold_eur_mwh
    predicted_alert = np.abs(predicted_pair) >= alert_threshold_eur_mwh
    cycles = cycle_inconsistency(predicted_pair, len(CORE_PHYSICAL_ZONES))
    metrics = {
        "pairwise_mae_eur_mwh": float(np.mean(np.abs(error))),
        "pairwise_rmse_eur_mwh": float(np.sqrt(np.mean(np.square(error)))),
        "timestamp_mae_p95_eur_mwh": float(np.quantile(timestamp_mae, 0.95)),
        "timestamp_mae_max_eur_mwh": float(np.max(timestamp_mae)),
        "nontrivial_spread_sign_accuracy": sign_accuracy,
        "alert_flip_rate": float(np.mean(target_alert != predicted_alert)),
        "cycle_inconsistency_max_eur_mwh": float(np.max(cycles, initial=0.0)),
    }
    if prediction.get("field") is not None:
        predicted_field = np.asarray(prediction["field"], dtype=np.float64)
        metrics["centered_field_mae_eur_mwh"] = float(
            np.mean(np.abs(predicted_field - target_field))
        )
    return metrics


def _drift_metrics(
    clean: dict[str, Any], stressed: dict[str, Any], *, alert_threshold_eur_mwh: float
) -> dict[str, float]:
    clean_pair = np.asarray(clean["pairwise"], dtype=np.float64)
    stressed_pair = np.asarray(stressed["pairwise"], dtype=np.float64)
    if clean_pair.shape != stressed_pair.shape:
        raise RuntimeError("Presentation stress changed prediction shape")
    absolute = np.abs(stressed_pair - clean_pair)
    clean_alert = np.abs(clean_pair) >= alert_threshold_eur_mwh
    stress_alert = np.abs(stressed_pair) >= alert_threshold_eur_mwh
    return {
        "presentation_drift_mean_eur_mwh": float(np.mean(absolute)),
        "presentation_drift_p95_eur_mwh": float(np.quantile(absolute, 0.95)),
        "presentation_drift_max_eur_mwh": float(np.max(absolute, initial=0.0)),
        "presentation_alert_flip_rate": float(np.mean(clean_alert != stress_alert)),
    }


def _bootstrap_mean_ci(values: list[float], *, seed: int = 20261003) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [record for record in records if record["presentation"] == "original"]
    clean_summary: dict[str, Any] = {}
    for method in METHODS:
        selected = [record for record in clean if record["method"] == method]
        if not selected:
            continue
        values = [record["metrics"]["pairwise_mae_eur_mwh"] for record in selected]
        clean_summary[method] = {
            "seeds": len(selected),
            "pairwise_mae_mean_eur_mwh": float(np.mean(values)),
            "pairwise_mae_std_eur_mwh": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "pairwise_mae_bootstrap_95_ci_eur_mwh": _bootstrap_mean_ci(values),
            "pairwise_mae_seedwise_eur_mwh": values,
            "pairwise_rmse_mean_eur_mwh": float(
                np.mean([record["metrics"]["pairwise_rmse_eur_mwh"] for record in selected])
            ),
            "alert_flip_rate_mean": float(
                np.mean([record["metrics"]["alert_flip_rate"] for record in selected])
            ),
        }
    invariance = {}
    for method in METHODS:
        selected = [record for record in records if record["method"] == method]
        if not selected:
            continue
        invariance[method] = {
            "maximum_presentation_drift_eur_mwh": float(
                max(record["drift"]["presentation_drift_max_eur_mwh"] for record in selected)
            ),
            "maximum_presentation_alert_flip_rate": float(
                max(record["drift"]["presentation_alert_flip_rate"] for record in selected)
            ),
        }
    return {"clean": clean_summary, "presentation_invariance": invariance}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _configuration_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry_audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit_end", required=True)
    parser.add_argument("--validation_end", required=True)
    parser.add_argument("--evaluation_end", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum_split_rows", type=int, default=48)
    parser.add_argument("--alert_threshold_eur_mwh", type=float, default=5.0)
    parser.add_argument("--invariance_seed", type=int, default=7)
    parser.add_argument("--smoke_presentations", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.minimum_split_rows < 2:
        raise ValueError("Invalid training or split setting")
    if args.alert_threshold_eur_mwh <= 0.0 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Alert threshold must be positive and seeds unique")
    if args.invariance_seed not in args.seeds:
        raise ValueError("--invariance_seed must be included in --seeds")
    output = args.output.resolve()

    manifest_path = args.manifest.resolve()
    geometry_audit_path = args.geometry_audit.resolve()
    audit = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("Core geometry audit has not passed")
    if audit.get("source_manifest") != str(manifest_path):
        raise RuntimeError("Geometry audit was not produced from the supplied manifest")

    fit_end = parse_utc(args.fit_end)
    validation_end = parse_utc(args.validation_end)
    evaluation_end = parse_utc(args.evaluation_end)
    if not fit_end < validation_end < evaluation_end:
        raise ValueError("Chronology must satisfy fit < validation < evaluation end")
    dual, price, manifest, dual_path, price_path = _read_inputs(manifest_path)
    source_end = parse_utc(manifest["period"]["end_utc_exclusive"])
    if evaluation_end > source_end:
        raise RuntimeError("Evaluation end exceeds the registered Core source period")

    unit_columns = [column for column in dual if column.startswith("unit_hub_")]
    hub_names = tuple(column.removeprefix("unit_hub_") for column in unit_columns)
    if tuple(hub_names[: len(CORE_PHYSICAL_ZONES)]) != CORE_PHYSICAL_ZONES:
        raise RuntimeError("Core physical-zone coordinate order changed")
    equality, tangent = global_balance_geometry(hub_names)
    geometry = RealAffineGeometry(
        zone_names=hub_names,
        equality_names=("global_net_position_balance",),
        equality_matrix=equality,
        tangent_projector=tangent,
        rank=1,
    )
    physical_indices = np.asarray(
        [hub_names.index(zone) for zone in CORE_PHYSICAL_ZONES], dtype=np.int64
    )
    samples, scalar_scales = _build_samples(
        dual,
        price,
        unit_columns=unit_columns,
        fit_end=fit_end,
        evaluation_end=evaluation_end,
    )
    train_samples, validation_samples, evaluation_samples = _split_samples(
        samples, fit_end, validation_end, evaluation_end
    )
    split_sizes = {
        "fit": len(train_samples),
        "validation": len(validation_samples),
        "evaluation": len(evaluation_samples),
    }
    if min(split_sizes.values()) < args.minimum_split_rows:
        raise RuntimeError(f"Core chronological split is too small: {split_sizes}")

    target_values = np.concatenate(
        [np.abs(sample["target_vector"]) for sample in train_samples]
    )
    nonzero_target = target_values[target_values > 0.0]
    target_scale = max(float(np.median(nonzero_target)), 1.0) if len(nonzero_target) else 1.0
    canonical_ram_scale = scalar_scales["canonical_ram_mw"]
    raw_scales: dict[str, np.ndarray] = {}
    for representation in ("ambient", "af"):
        raw_rows = raw_rows_for_scale(
            train_samples,
            representation=representation,
            real_zone_indices=physical_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
        )
        raw_scales[representation] = _positive_scales(raw_rows)

    all_presentations = presentation_names(geometry)
    if args.smoke_presentations:
        requested = (
            "original",
            "combined",
            f"slack_{hub_names[0]}",
            f"slack_{hub_names[-1]}",
            "af_projected_representative",
            "equality_reflection",
            "random_equality_gauge",
            "combined_random_equality_gauge",
        )
        presentations = tuple(name for name in requested if name in all_presentations)
    else:
        presentations = all_presentations
    if "original" not in presentations:
        raise RuntimeError("Original presentation is mandatory")

    device = _device(args.device)
    raw_dim = 1 + len(unit_columns) + 5
    canonical_dim = len(unit_columns) + 5
    run_configuration = {
        "source_manifest_sha256": sha256_file(manifest_path),
        "fit_end": fit_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device_type": str(device),
        "minimum_split_rows": args.minimum_split_rows,
        "alert_threshold_eur_mwh": args.alert_threshold_eur_mwh,
        "invariance_seed": args.invariance_seed,
        "presentations": list(presentations),
        "methods": list(METHODS),
    }
    run_hash = _configuration_hash(run_configuration)
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    records: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    completed: set[str] = set()
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        migrate_progress = False
        if progress.get("configuration_hash") != run_hash:
            previous_configuration = dict(progress.get("configuration", {}))
            # v1 progress initially included the whole audit-file hash.  That
            # hash changes when only ``created_utc`` changes, despite identical
            # checked geometry.  Permit this one stable migration and rewrite
            # the progress ledger under the content configuration below.
            previous_configuration.pop("geometry_audit_sha256", None)
            if previous_configuration != run_configuration:
                raise RuntimeError(
                    "Existing Core benchmark progress has a different "
                    f"configuration: {progress_path}"
                )
            migrate_progress = True
        records = list(progress.get("records", []))
        training = list(progress.get("training", []))
        completed = set(progress.get("completed", []))
        if migrate_progress:
            atomic_json(
                progress_path,
                {
                    **progress,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration": run_configuration,
                    "configuration_hash": run_hash,
                },
            )
        print(
            f"[RESUME] completed_method_runs={len(completed)} from {progress_path}",
            flush=True,
        )
    for seed in args.seeds:
        for method, specification in METHODS.items():
            if specification is None and seed != args.invariance_seed:
                # The analytic control has no stochastic fit; duplicating it
                # across random seeds would manufacture a sample size.
                continue
            completion_key = f"{seed}:{method}"
            if completion_key in completed:
                print(f"[SKIP] completed {completion_key}", flush=True)
                continue
            if specification is None:
                mode = None
                representation = "af"
                model = None
                fit_metadata = {
                    "seed": seed,
                    "parameter_count": 0,
                    "best_epoch": 0,
                    "validation_objective": None,
                }
            else:
                mode, representation = specification
                train_dataset = MaterializedDataset(RealAffineGaugeDataset(
                    train_samples,
                    raw_scales[representation],
                    target_scale,
                    "original",
                    representation,
                    physical_indices,
                    geometry,
                    canonical_ram_scale,
                ))
                validation_dataset = MaterializedDataset(RealAffineGaugeDataset(
                    validation_samples,
                    raw_scales[representation],
                    target_scale,
                    "original",
                    representation,
                    physical_indices,
                    geometry,
                    canonical_ram_scale,
                ))
                model, fit_metadata = fit_multizone(
                    mode,
                    seed,
                    train_dataset,
                    validation_dataset,
                    raw_dim=raw_dim,
                    canonical_dim=canonical_dim,
                    zone_count=len(CORE_PHYSICAL_ZONES),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    device=device,
                )
            training.append({"method": method, "representation": representation, **fit_metadata})
            print(
                f"[FIT] seed={seed} method={method} best_epoch={fit_metadata['best_epoch']}",
                flush=True,
            )

            evaluated_presentations = (
                presentations if seed == args.invariance_seed else ("original",)
            )
            predictions: dict[str, dict[str, Any]] = {}
            for presentation in evaluated_presentations:
                dataset = RealAffineGaugeDataset(
                    evaluation_samples,
                    raw_scales[representation],
                    target_scale,
                    presentation,
                    representation,
                    physical_indices,
                    geometry,
                    canonical_ram_scale,
                )
                if model is None:
                    prediction = _analytic_prediction(dataset)
                else:
                    prediction = predict_multizone(
                        model,
                        dataset,
                        batch_size=args.batch_size,
                        target_scale=target_scale,
                        device=device,
                    )
                predictions[presentation] = prediction
            clean_prediction = predictions["original"]
            for presentation, prediction in predictions.items():
                records.append(
                    {
                        "seed": seed,
                        "method": method,
                        "representation": representation,
                        "presentation": presentation,
                        "metrics": _prediction_metrics(
                            prediction,
                            alert_threshold_eur_mwh=args.alert_threshold_eur_mwh,
                        ),
                        "drift": _drift_metrics(
                            clean_prediction,
                            prediction,
                            alert_threshold_eur_mwh=args.alert_threshold_eur_mwh,
                        ),
                    }
                )
            completed.add(completion_key)
            atomic_json(
                progress_path,
                {
                    "protocol": "af_cdmo_core_external_progress_v1",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration": run_configuration,
                    "configuration_hash": run_hash,
                    "completed": sorted(completed),
                    "training": training,
                    "records": records,
                },
            )
            print(
                f"[EVAL] seed={seed} method={method} "
                f"presentations={len(evaluated_presentations)}",
                flush=True,
            )

    summary = _summarize(records)
    invariance = summary["presentation_invariance"]
    invariant_failures = {}
    for method in ("analytic_gauge", "core_rank1_af_qdm_residual"):
        maximum = invariance[method]["maximum_presentation_drift_eur_mwh"]
        # The algebraic geometry audit above is checked near machine precision.
        # This end-to-end threshold permits only float32 model/data roundoff.
        tolerance = 1.0e-3
        if maximum > tolerance:
            invariant_failures[method] = {"maximum": maximum, "tolerance": tolerance}
    cycle_failures = {
        f"{record['method']}:{record['seed']}:{record['presentation']}": record["metrics"][
            "cycle_inconsistency_max_eur_mwh"
        ]
        for record in records
        if record["metrics"]["cycle_inconsistency_max_eur_mwh"] > 2.0e-5
    }
    structural_pass = not invariant_failures and not cycle_failures

    report = {
        "protocol": "af_cdmo_core_external_replication_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_type": (
            "Frozen-method external replication with chronological Core refitting; "
            "not zero-shot neural-weight transfer"
        ),
        "claim_boundary": (
            "Core quotient uses only global net-position balance (rank 1); it does "
            "not assert recovery of Core technical or ALEGrO equalities."
        ),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "geometry_audit": str(geometry_audit_path),
        "geometry_audit_sha256": sha256_file(geometry_audit_path),
        "data": {
            "dual_path": str(dual_path),
            "dual_sha256": sha256_file(dual_path),
            "price_path": str(price_path),
            "price_sha256": sha256_file(price_path),
            "physical_zones": list(CORE_PHYSICAL_ZONES),
            "all_hub_coordinates": list(hub_names),
            "split": {
                "fit_end_utc": fit_end.isoformat(),
                "validation_end_utc": validation_end.isoformat(),
                "evaluation_end_utc": evaluation_end.isoformat(),
                "rows": split_sizes,
            },
            "certificate_coverage": {
                name: float(
                    np.mean([len(sample["weights"]) > 0 for sample in values])
                )
                for name, values in (
                    ("fit", train_samples),
                    ("validation", validation_samples),
                    ("evaluation", evaluation_samples),
                )
            },
        },
        "frozen_configuration": {
            "methods": list(METHODS),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "target_scale_eur_mwh": target_scale,
            "canonical_scalar_scales": scalar_scales,
            "alert_threshold_eur_mwh": args.alert_threshold_eur_mwh,
            "presentations": list(presentations),
            "presentation_count": len(presentations),
            "invariance_seed": args.invariance_seed,
            "stress_allocation": (
                "All seeds use the clean presentation; the complete registered "
                "presentation orbit is evaluated for every method on invariance_seed only"
            ),
            "deterministic_train_validation_materialized": True,
            "smoke_presentations": args.smoke_presentations,
        },
        "training": training,
        "records": records,
        "summary": summary,
        "structural_contract": {
            "passed": structural_pass,
            "invariance_failures": invariant_failures,
            "cycle_failures": cycle_failures,
            "performance_used_as_pass_fail_criterion": False,
        },
    }
    atomic_json(output, report)
    if not structural_pass:
        raise RuntimeError(
            "Core external benchmark violated structural invariance/cycle contracts: "
            f"invariance={invariant_failures}, cycles={len(cycle_failures)}"
        )
    print(f"[OK] Core external benchmark: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
