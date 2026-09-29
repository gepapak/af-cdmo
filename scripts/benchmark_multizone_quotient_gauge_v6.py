"""Benchmark a quotient-gauge network on the relative Nordic zonal price field.

The task reconstructs simultaneously observed day-ahead zonal prices from a
causally available solved flow-based certificate. It is not a future-price
forecast. The gauge model predicts one centered field and derives every bilateral
spread, enforcing antisymmetry and cycle consistency by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import ConcatDataset, Dataset

try:
    from scripts.benchmark_certificate_governance_v5 import (
        AUGMENTATION_STRESSES,
        CANONICAL_SCALAR_COLUMNS,
        DEFAULT_DUAL,
        DEFAULT_EVALUATION_END,
        DEFAULT_FIT_END,
        DEFAULT_VALIDATION_END,
        RAW_SCALAR_COLUMNS,
        _positive_scales,
        _read_certificate,
        _stress_certificate,
    )
    from scripts.cqdm_utils import (
        DAY_AHEAD_QUARTER_HOUR_START,
        paired_block_bootstrap,
        sha256_file,
    )
    from scripts.multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        MultiZoneRegressor,
        center_price_field,
        cycle_inconsistency,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
        quotient_feature,
    )
except ModuleNotFoundError:
    from benchmark_certificate_governance_v5 import (
        AUGMENTATION_STRESSES,
        CANONICAL_SCALAR_COLUMNS,
        DEFAULT_DUAL,
        DEFAULT_EVALUATION_END,
        DEFAULT_FIT_END,
        DEFAULT_VALIDATION_END,
        RAW_SCALAR_COLUMNS,
        _positive_scales,
        _read_certificate,
        _stress_certificate,
    )
    from cqdm_utils import (
        DAY_AHEAD_QUARTER_HOUR_START,
        paired_block_bootstrap,
        sha256_file,
    )
    from multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        MultiZoneRegressor,
        center_price_field,
        cycle_inconsistency,
        field_to_pairwise,
        fit_multizone,
        pair_indices,
        predict_multizone,
        quotient_feature,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_ROOT = ROOT / "official_entsoe_nordic_day_ahead"
DEFAULT_OUTPUT = (
    ROOT / "official_jao_dual_measure" / "multizone_quotient_gauge_development_v6.json"
)
STRESSES = ("original", "combined")
NEURAL_MODES = (
    "raw_deepset_gauge",
    "raw_deepset_augmented_gauge",
    "cqdm_gauge_residual",
    "cqdm_pairwise_residual",
)


def _load_zone_price(price_root: Path, zone: str) -> tuple[pd.Series, dict[str, Any]]:
    path = price_root / f"DayAheadPrices_{zone}_ENTSOE.csv.gz"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing official ENTSO-E price artifact for {zone}: {path}. "
            "Run scripts/download_entsoe_nordic_day_ahead.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sha256") != sha256_file(path):
        raise RuntimeError(f"Price metadata hash mismatch for {zone}")
    frame = pd.read_csv(path, usecols=["delivery_utc", "zone", "price_eur_mwh"])
    if set(frame["zone"].dropna().astype(str)) != {zone}:
        raise RuntimeError(f"Unexpected zone labels in {path}")
    frame["delivery_utc"] = pd.to_datetime(
        frame["delivery_utc"], utc=True, errors="raise", format="mixed"
    )
    frame["price_eur_mwh"] = pd.to_numeric(frame["price_eur_mwh"], errors="coerce")
    if frame["price_eur_mwh"].isna().any():
        raise RuntimeError(f"Non-numeric prices in {path}")
    if frame["delivery_utc"].duplicated().any():
        raise RuntimeError(f"Duplicate delivery timestamps in {path}")
    series = pd.Series(
        frame["price_eur_mwh"].to_numpy(dtype=np.float64),
        index=frame["delivery_utc"],
        name=zone,
    ).sort_index()
    pre = series.loc[series.index < DAY_AHEAD_QUARTER_HOUR_START]
    post = series.loc[series.index >= DAY_AHEAD_QUARTER_HOUR_START]
    if len(pre) and np.all(pre.index.minute == 0):
        expanded = []
        for offset in (0, 15, 30, 45):
            piece = pre.copy()
            piece.index = piece.index + pd.Timedelta(minutes=offset)
            expanded.append(piece)
        pre = pd.concat(expanded).sort_index()
    combined = pd.concat([pre, post]).sort_index()
    if combined.index.duplicated().any():
        raise RuntimeError(f"Resolution expansion created duplicate prices for {zone}")
    return combined, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows_native": int(len(series)),
        "rows_market_grid": int(len(combined)),
        "first_utc": combined.index.min().isoformat(),
        "last_utc": combined.index.max().isoformat(),
    }


def _load_price_panel(
    price_root: Path, zones: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    series = []
    provenance = {}
    for zone in zones:
        value, metadata = _load_zone_price(price_root, zone)
        series.append(value)
        provenance[zone] = metadata
    panel = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if panel.empty:
        raise RuntimeError("Nordic price series have no common timestamps")
    return panel, provenance


def _build_vector_samples(
    frame: pd.DataFrame,
    price_panel: pd.DataFrame,
    unit_columns: list[str],
    zones: tuple[str, ...],
    fit_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    fit = frame.loc[frame["delivery_utc"] < fit_end]
    canonical_scales: dict[str, float] = {}
    for column in CANONICAL_SCALAR_COLUMNS:
        finite = np.abs(fit[column].to_numpy(dtype=np.float64))
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        canonical_scales[column] = max(float(np.median(finite)), 1.0e-6) if len(finite) else 1.0
    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    groups: dict[pd.Timestamp, dict[str, Any]] = {}
    for delivery, group in frame.groupby("delivery_utc", sort=False):
        unit = group[unit_columns].fillna(0.0).to_numpy(dtype=np.float64)
        norm = group["ptdf_l2_norm"].to_numpy(dtype=np.float64)
        shadow = group["shadowPrice"].to_numpy(dtype=np.float64)
        raw_scalars = group[list(RAW_SCALAR_COLUMNS)].fillna(0.0).to_numpy(dtype=np.float64)
        raw = np.column_stack([shadow, unit * norm[:, None], raw_scalars]).astype(np.float32)
        canonical_scalars = np.column_stack(
            [
                np.arcsinh(
                    group[column].fillna(0.0).to_numpy(dtype=np.float64)
                    / canonical_scales[column]
                )
                for column in CANONICAL_SCALAR_COLUMNS
            ]
        )
        canonical = np.column_stack([unit, canonical_scalars]).astype(np.float32)
        mass = group["dual_mass_eur_mwh"].to_numpy(dtype=np.float64)
        total = float(mass.sum())
        weights = (mass / total).astype(np.float32)
        dual_potential = mass @ unit[:, real_indices]
        analytic = center_price_field(-dual_potential)
        groups[pd.Timestamp(delivery)] = {
            "raw": raw,
            "canonical": canonical,
            "weights": weights,
            "total": total,
            "analytic_vector": analytic.astype(np.float32),
        }

    start = frame["delivery_utc"].min()
    raw_dim = 1 + len(unit_columns) + len(RAW_SCALAR_COLUMNS)
    canonical_dim = len(unit_columns) + len(CANONICAL_SCALAR_COLUMNS)
    samples = []
    for timestamp, row in price_panel.iterrows():
        timestamp = pd.Timestamp(timestamp)
        if timestamp < start or timestamp >= evaluation_end:
            continue
        target = center_price_field(row.to_numpy(dtype=np.float64)).astype(np.float32)
        certificate = groups.get(
            timestamp,
            {
                "raw": np.empty((0, raw_dim), dtype=np.float32),
                "canonical": np.empty((0, canonical_dim), dtype=np.float32),
                "weights": np.empty(0, dtype=np.float32),
                "total": 0.0,
                "analytic_vector": np.zeros(len(zones), dtype=np.float32),
            },
        )
        samples.append({"timestamp": timestamp, "target_vector": target, **certificate})
    return samples


class VectorCertificateDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        raw_scale: np.ndarray,
        target_scale: float,
        stress: str,
        real_zone_indices: np.ndarray,
        ptdf_count: int,
    ) -> None:
        self.samples = samples
        self.raw_scale = raw_scale
        self.target_scale = target_scale
        self.stress = stress
        self.real_zone_indices = np.asarray(real_zone_indices, dtype=np.int64)
        self.row_scaled_columns = np.asarray(
            [*range(1, 1 + ptdf_count), 1 + ptdf_count], dtype=np.int64
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        stress = self.stress
        seed = 20260904 + index
        if stress == "augmentation":
            stress = AUGMENTATION_STRESSES[(index + self.epoch) % len(AUGMENTATION_STRESSES)]
            seed += self.epoch * max(len(self.samples), 1)
        raw, canonical, weights = _stress_certificate(
            sample["raw"],
            sample["canonical"],
            sample["weights"],
            stress,
            seed,
            self.row_scaled_columns,
        )
        if len(weights):
            dual_potential = sample["total"] * (
                weights.astype(np.float64) @ canonical[:, self.real_zone_indices].astype(np.float64)
            )
            analytic = center_price_field(-dual_potential)
        else:
            analytic = np.zeros(len(self.real_zone_indices), dtype=np.float64)
        target = sample["target_vector"].astype(np.float64)
        return {
            **sample,
            "raw_scaled": np.arcsinh(raw / self.raw_scale).astype(np.float32),
            "canonical": canonical,
            "weights": weights,
            "analytic_vector": analytic.astype(np.float32),
            "analytic_scaled": (analytic / self.target_scale).astype(np.float32),
            "target_scaled": (target / self.target_scale).astype(np.float32),
            "residual_scaled": ((target - analytic) / self.target_scale).astype(np.float32),
            "pair_target_scaled": (field_to_pairwise(target) / self.target_scale).astype(np.float32),
            "pair_residual_scaled": (
                (field_to_pairwise(target) - field_to_pairwise(analytic)) / self.target_scale
            ).astype(np.float32),
            "total_scaled": np.float32(np.log1p(sample["total"])),
        }


def _parse_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Chronology boundaries must include a timezone")
    return timestamp.tz_convert("UTC")


def _features(dataset: VectorCertificateDataset) -> np.ndarray:
    return np.stack(
        [
            quotient_feature(
                dataset[index]["canonical"],
                dataset[index]["weights"],
                dataset.samples[index]["total"],
                dataset.real_zone_indices,
            )
            for index in range(len(dataset))
        ]
    )


def _classical_predictions(
    train_dataset: VectorCertificateDataset,
    evaluation_datasets: dict[str, VectorCertificateDataset],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray | None]]], dict[str, Any]]:
    train_features = _features(train_dataset)
    train_target = np.stack([sample["target_vector"] for sample in train_dataset.samples])
    train_analytic = np.stack([sample["analytic_vector"] for sample in train_dataset.samples])
    train_pairs = field_to_pairwise(train_target)

    affine = LinearRegression().fit(train_analytic, train_target)
    zone_models = [
        HistGradientBoostingRegressor(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=20260904 + zone_index,
        ).fit(train_features, train_target[:, zone_index])
        for zone_index in range(train_target.shape[1])
    ]
    pair_models = [
        HistGradientBoostingRegressor(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=20261000 + pair_index,
        ).fit(train_features, train_pairs[:, pair_index])
        for pair_index in range(train_pairs.shape[1])
    ]
    output: dict[str, dict[str, dict[str, np.ndarray | None]]] = {
        "analytic_gauge": {},
        "affine_analytic_gauge": {},
        "invariant_hgb_gauge": {},
        "independent_pairwise_hgb": {},
    }
    for stress, dataset in evaluation_datasets.items():
        features = _features(dataset)
        analytic = np.stack([dataset[index]["analytic_vector"] for index in range(len(dataset))])
        affine_field = center_price_field(affine.predict(analytic))
        hgb_field = center_price_field(
            np.column_stack([model.predict(features) for model in zone_models])
        )
        output["analytic_gauge"][stress] = {
            "field": analytic,
            "pairwise": field_to_pairwise(analytic),
        }
        output["affine_analytic_gauge"][stress] = {
            "field": affine_field,
            "pairwise": field_to_pairwise(affine_field),
        }
        output["invariant_hgb_gauge"][stress] = {
            "field": hgb_field,
            "pairwise": field_to_pairwise(hgb_field),
        }
        output["independent_pairwise_hgb"][stress] = {
            "field": None,
            "pairwise": np.column_stack([model.predict(features) for model in pair_models]),
        }
    return output, {
        "affine_parameters": {
            "coefficient_shape": list(affine.coef_.shape),
            "intercept": affine.intercept_.tolist(),
        },
        "invariant_hgb_zone_models": len(zone_models),
        "independent_pairwise_hgb_models": len(pair_models),
    }


def _summarize_method(
    target_field: np.ndarray,
    field: np.ndarray | None,
    pairwise: np.ndarray,
    original_pairwise: np.ndarray,
    threshold: float,
    zone_count: int,
) -> tuple[dict[str, float | None], np.ndarray]:
    target_pairs = field_to_pairwise(target_field)
    pair_error = np.abs(target_pairs - pairwise)
    per_timestamp = pair_error.mean(axis=1)
    drift = np.abs(pairwise - original_pairwise)
    original_alert = np.max(np.abs(target_pairs - original_pairwise), axis=1) > threshold
    candidate_alert = np.max(pair_error, axis=1) > threshold
    cycles = cycle_inconsistency(pairwise, zone_count)
    result: dict[str, float | None] = {
        "mean_pairwise_mae_eur_mwh": float(per_timestamp.mean()),
        "p95_timestamp_pairwise_mae_eur_mwh": float(np.quantile(per_timestamp, 0.95)),
        "mean_abs_pairwise_prediction_drift_eur_mwh": float(drift.mean()),
        "max_abs_pairwise_prediction_drift_eur_mwh": float(drift.max(initial=0.0)),
        "alert_flip_rate": float(np.mean(original_alert != candidate_alert)),
        "mean_abs_cycle_inconsistency_eur_mwh": float(cycles.mean()),
        "p95_abs_cycle_inconsistency_eur_mwh": float(np.quantile(cycles, 0.95)),
        "max_abs_cycle_inconsistency_eur_mwh": float(cycles.max(initial=0.0)),
        "mean_centered_zone_mae_eur_mwh": (
            float(np.mean(np.abs(target_field - field))) if field is not None else None
        ),
        "max_abs_gauge_sum_eur_mwh": (
            float(np.max(np.abs(field.sum(axis=1)), initial=0.0)) if field is not None else None
        ),
    }
    return result, per_timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--price_root", type=Path, default=DEFAULT_PRICE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zones", nargs="+", choices=REAL_NORDIC_ZONES, default=REAL_NORDIC_ZONES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--alert_quantile", type=float, default=0.95)
    parser.add_argument("--min_price_panel_coverage", type=float, default=0.90)
    parser.add_argument(
        "--study_role",
        choices=("development", "frozen_transition", "confirmation"),
        default="development",
    )
    parser.add_argument("--fit_end", default=DEFAULT_FIT_END.isoformat())
    parser.add_argument("--validation_end", default=DEFAULT_VALIDATION_END.isoformat())
    parser.add_argument("--evaluation_end", default=DEFAULT_EVALUATION_END.isoformat())
    args = parser.parse_args()

    zones = tuple(dict.fromkeys(args.zones))
    if len(zones) < 3:
        raise ValueError("The multi-zone experiment requires at least three distinct zones")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or not 0.0 < args.alert_quantile < 1.0
        or not 0.0 < args.min_price_panel_coverage <= 1.0
    ):
        raise ValueError("Invalid fit or alert configuration")
    fit_end = _parse_utc(args.fit_end)
    validation_end = _parse_utc(args.validation_end)
    evaluation_end = _parse_utc(args.evaluation_end)
    if not fit_end < validation_end < evaluation_end:
        raise ValueError("Require fit_end < validation_end < evaluation_end")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    frame, certificate_audit = _read_certificate(args.dual.resolve(), 60.0)
    unit_columns = certificate_audit["unit_columns"]
    missing_ptdf = [zone for zone in zones if f"unit_ptdf_{zone}" not in unit_columns]
    if missing_ptdf:
        raise RuntimeError(f"Certificate omitted real-zone PTDF channels: {missing_ptdf}")
    price_panel, price_provenance = _load_price_panel(args.price_root.resolve(), zones)
    panel_start = max(frame["delivery_utc"].min(), price_panel.index.min())
    expected_grid = pd.date_range(
        start=panel_start,
        end=evaluation_end,
        freq="15min",
        inclusive="left",
    )
    covered_grid = price_panel.index.intersection(expected_grid)
    price_panel_coverage = len(covered_grid) / max(len(expected_grid), 1)
    if price_panel_coverage < args.min_price_panel_coverage:
        raise RuntimeError(
            "All-zone ENTSO-E price-panel coverage is too low: "
            f"{price_panel_coverage:.2%} < {args.min_price_panel_coverage:.2%}. "
            "Do not impute missing post-transition market prices."
        )
    samples = _build_vector_samples(
        frame,
        price_panel,
        unit_columns,
        zones,
        fit_end,
        evaluation_end,
    )
    train_samples = [sample for sample in samples if sample["timestamp"] < fit_end]
    validation_samples = [
        sample for sample in samples if fit_end <= sample["timestamp"] < validation_end
    ]
    evaluation_samples = [
        sample for sample in samples if validation_end <= sample["timestamp"] < evaluation_end
    ]
    if not train_samples or not validation_samples or not evaluation_samples:
        raise RuntimeError(
            "Chronology produced an empty split after all-zone inner alignment: "
            f"fit={len(train_samples)}, validation={len(validation_samples)}, "
            f"evaluation={len(evaluation_samples)}"
        )
    raw_values = np.concatenate(
        [sample["raw"] for sample in train_samples if len(sample["raw"])], axis=0
    ).astype(np.float64)
    raw_scale = _positive_scales(raw_values)
    target_values = np.stack([sample["target_vector"] for sample in train_samples])
    target_scale = max(float(np.median(np.abs(target_values))), 1.0)
    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    dataset_kwargs = {
        "raw_scale": raw_scale,
        "target_scale": target_scale,
        "real_zone_indices": real_indices,
        "ptdf_count": len(unit_columns),
    }
    train_dataset = VectorCertificateDataset(train_samples, stress="original", **dataset_kwargs)
    validation_dataset = VectorCertificateDataset(validation_samples, stress="original", **dataset_kwargs)
    augmented_train_dataset = VectorCertificateDataset(train_samples, stress="augmentation", **dataset_kwargs)
    augmented_validation_dataset = ConcatDataset(
        [
            VectorCertificateDataset(validation_samples, stress=stress, **dataset_kwargs)
            for stress in AUGMENTATION_STRESSES
        ]
    )
    evaluation_datasets = {
        stress: VectorCertificateDataset(evaluation_samples, stress=stress, **dataset_kwargs)
        for stress in STRESSES
    }

    raw_dim = train_samples[0]["raw"].shape[1]
    canonical_dim = train_samples[0]["canonical"].shape[1]
    models: dict[str, list[MultiZoneRegressor]] = {mode: [] for mode in NEURAL_MODES}
    fits: dict[str, list[dict[str, Any]]] = {mode: [] for mode in NEURAL_MODES}
    for mode in NEURAL_MODES:
        for seed in args.seeds:
            augmented = mode == "raw_deepset_augmented_gauge"
            model, fit = fit_multizone(
                mode,
                seed,
                augmented_train_dataset if augmented else train_dataset,
                augmented_validation_dataset if augmented else validation_dataset,
                raw_dim=raw_dim,
                canonical_dim=canonical_dim,
                zone_count=len(zones),
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=device,
            )
            models[mode].append(model)
            fits[mode].append(fit)
            print(
                f"[FIT] mode={mode} seed={seed} epoch={fit['best_epoch']} "
                f"validation={fit['validation_objective']:.6f}",
                flush=True,
            )

    predictions: dict[str, dict[str, dict[str, np.ndarray | None]]] = {
        mode: {} for mode in NEURAL_MODES
    }
    neural_members: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {} for mode in NEURAL_MODES
    }
    for stress, dataset in evaluation_datasets.items():
        for mode in NEURAL_MODES:
            members = [
                predict_multizone(
                    model,
                    dataset,
                    batch_size=args.batch_size,
                    target_scale=target_scale,
                    device=device,
                )
                for model in models[mode]
            ]
            neural_members[mode][stress] = members
            fields = [member["field"] for member in members if member["field"] is not None]
            predictions[mode][stress] = {
                "field": np.mean(np.stack(fields), axis=0) if fields else None,
                "pairwise": np.mean(
                    np.stack([member["pairwise"] for member in members]), axis=0
                ),
            }
    classical_datasets = {**evaluation_datasets, "validation": validation_dataset}
    classical_predictions, classical_fit = _classical_predictions(
        train_dataset, classical_datasets
    )
    validation_classical = {
        method: {"original": by_stress.pop("validation")}
        for method, by_stress in classical_predictions.items()
    }
    predictions = {**classical_predictions, **predictions}

    validation_predictions: dict[str, np.ndarray] = {}
    validation_predictions.update(
        {
            method: value["original"]["pairwise"]
            for method, value in validation_classical.items()
        }
    )
    for mode in NEURAL_MODES:
        members = [
            predict_multizone(
                model,
                validation_dataset,
                batch_size=args.batch_size,
                target_scale=target_scale,
                device=device,
            )
            for model in models[mode]
        ]
        validation_predictions[mode] = np.mean(
            np.stack([member["pairwise"] for member in members]), axis=0
        )
    validation_target = np.stack([sample["target_vector"] for sample in validation_samples])
    validation_pairs = field_to_pairwise(validation_target)
    thresholds = {
        method: float(
            np.quantile(
                np.max(np.abs(validation_pairs - prediction), axis=1),
                args.alert_quantile,
            )
        )
        for method, prediction in validation_predictions.items()
    }

    target_field = np.stack([sample["target_vector"] for sample in evaluation_samples])
    results: dict[str, Any] = {}
    per_timestamp_losses: dict[str, np.ndarray] = {}
    for method, by_stress in predictions.items():
        original_pairs = np.asarray(by_stress["original"]["pairwise"])
        results[method] = {
            "validation_max_pair_error_alert_threshold_eur_mwh": thresholds[method],
            "stresses": {},
        }
        for stress in STRESSES:
            summary, losses = _summarize_method(
                target_field,
                by_stress[stress]["field"],
                np.asarray(by_stress[stress]["pairwise"]),
                original_pairs,
                thresholds[method],
                len(zones),
            )
            if method in neural_members:
                summary["seed_members"] = []
                for seed, member in zip(args.seeds, neural_members[method][stress]):
                    member_summary, _ = _summarize_method(
                        target_field,
                        member["field"],
                        member["pairwise"],
                        neural_members[method]["original"][
                            args.seeds.index(seed)
                        ]["pairwise"],
                        thresholds[method],
                        len(zones),
                    )
                    summary["seed_members"].append(
                        {
                            "seed": seed,
                            "mean_pairwise_mae_eur_mwh": member_summary[
                                "mean_pairwise_mae_eur_mwh"
                            ],
                            "mean_centered_zone_mae_eur_mwh": member_summary[
                                "mean_centered_zone_mae_eur_mwh"
                            ],
                            "mean_abs_cycle_inconsistency_eur_mwh": member_summary[
                                "mean_abs_cycle_inconsistency_eur_mwh"
                            ],
                            "mean_abs_pairwise_prediction_drift_eur_mwh": member_summary[
                                "mean_abs_pairwise_prediction_drift_eur_mwh"
                            ],
                        }
                    )
            results[method]["stresses"][stress] = summary
            if stress == "original":
                per_timestamp_losses[method] = losses

    comparisons = {}
    for baseline in (
        "raw_deepset_gauge",
        "raw_deepset_augmented_gauge",
        "cqdm_pairwise_residual",
        "independent_pairwise_hgb",
        "invariant_hgb_gauge",
    ):
        comparisons[f"cqdm_gauge_residual_minus_{baseline}"] = paired_block_bootstrap(
            {"timestamp_pairwise_mae": per_timestamp_losses[baseline]},
            {"timestamp_pairwise_mae": per_timestamp_losses["cqdm_gauge_residual"]},
        )["timestamp_pairwise_mae"]

    report = {
        "protocol": {
            "name": "multi_zone_quotient_gauge_v6",
            "study_role": args.study_role,
            "task": (
                "simultaneous reconstruction of the observed centered Nordic "
                "day-ahead zonal price field from a timely solved certificate"
            ),
            "not_a_forecast": True,
            "zones": list(zones),
            "pairs": [f"{zones[left]}-{zones[right]}" for left, right in pair_indices(len(zones))],
            "fit_end_exclusive": fit_end.isoformat(),
            "internal_validation_end_exclusive": validation_end.isoformat(),
            "evaluation_end_exclusive": evaluation_end.isoformat(),
            "seeds": args.seeds,
            "stresses": list(STRESSES),
            "all_zone_price_panel_coverage": price_panel_coverage,
            "minimum_required_price_panel_coverage": args.min_price_panel_coverage,
            "gauge": "sum of predicted zonal prices is exactly zero",
            "structural_guarantees": [
                "certificate row-rewrite invariance for CQDM",
                "common-price-reference invariance",
                "bilateral antisymmetry",
                "cycle consistency for field-derived spreads",
            ],
        },
        "source": {
            "certificate": certificate_audit,
            "day_ahead_prices": price_provenance,
        },
        "rows": {
            "fit": len(train_samples),
            "internal_validation": len(validation_samples),
            "evaluation": len(evaluation_samples),
        },
        "target_scale_eur_mwh": target_scale,
        "fits": {**fits, "classical": classical_fit},
        "results": results,
        "paired_original_formulation_comparisons": comparisons,
        "reproducibility": {
            "engine": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "model_engine": {
                "path": str((ROOT / "scripts/multizone_quotient_gauge_v6.py").resolve()),
                "sha256": sha256_file(ROOT / "scripts/multizone_quotient_gauge_v6.py"),
            },
            "dual_source_sha256": sha256_file(args.dual.resolve()),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
