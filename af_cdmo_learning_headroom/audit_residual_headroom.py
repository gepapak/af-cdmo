"""Audit whether AF-CDMO's post-KKT residual contains learnable signal.

This is an exploratory development-only audit. It deliberately stops before the
registered confirmation period and never reads AF-CDMO confirmation artifacts.
It asks a narrower question than the AF-CDMO benchmark: after the analytic KKT
price field is removed, can invariant certificate moments, causal calendar
features, or strictly lagged market history improve the residual prediction?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

try:
    from scripts.benchmark_certificate_governance_v5 import (
        DEFAULT_DUAL,
        _read_certificate,
    )
    from scripts.benchmark_multizone_quotient_gauge_v6 import (
        DEFAULT_PRICE_ROOT,
        _build_vector_samples,
        _load_price_panel,
    )
    from scripts.cqdm_utils import paired_block_bootstrap, sha256_file
    from scripts.multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        center_price_field,
        field_to_pairwise,
        quotient_feature,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.benchmark_certificate_governance_v5 import (
        DEFAULT_DUAL,
        _read_certificate,
    )
    from scripts.benchmark_multizone_quotient_gauge_v6 import (
        DEFAULT_PRICE_ROOT,
        _build_vector_samples,
        _load_price_panel,
    )
    from scripts.cqdm_utils import paired_block_bootstrap, sha256_file
    from scripts.multizone_quotient_gauge_v6 import (
        REAL_NORDIC_ZONES,
        center_price_field,
        field_to_pairwise,
        quotient_feature,
    )


PRIMARY_ZONES = (
    "DK1",
    "DK2",
    "FI",
    "NO1",
    "NO2",
    "NO3",
    "NO5",
    "SE1",
    "SE3",
    "SE4",
)
FIT_END = pd.Timestamp("2025-11-01", tz="UTC")
SELECTION_END = pd.Timestamp("2026-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2026-03-01", tz="UTC")
CONFIRMATION_START = DEVELOPMENT_END
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--price_root", type=Path, default=DEFAULT_PRICE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "af_cdmo_learning_headroom"
        / "residual_headroom_development_v1.json",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calendar(timestamp: pd.Timestamp) -> np.ndarray:
    quarter = timestamp.hour * 4 + timestamp.minute // 15
    weekday = timestamp.dayofweek
    day_of_year = timestamp.dayofyear
    return np.asarray(
        [
            np.sin(2.0 * np.pi * quarter / 96.0),
            np.cos(2.0 * np.pi * quarter / 96.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            np.sin(2.0 * np.pi * day_of_year / 365.25),
            np.cos(2.0 * np.pi * day_of_year / 365.25),
        ],
        dtype=np.float64,
    )


def _pairwise_timestamp_loss(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target_pairs = field_to_pairwise(target)
    prediction_pairs = field_to_pairwise(prediction)
    return np.mean(np.abs(target_pairs - prediction_pairs), axis=1)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = center_price_field(target)
    prediction = center_price_field(prediction)
    pair_error = field_to_pairwise(prediction) - field_to_pairwise(target)
    timestamp_loss = np.mean(np.abs(pair_error), axis=1)
    return {
        "mean_pairwise_mae_eur_mwh": float(timestamp_loss.mean()),
        "p95_timestamp_pairwise_mae_eur_mwh": float(
            np.quantile(timestamp_loss, 0.95)
        ),
        "mean_centered_zone_mae_eur_mwh": float(
            np.mean(np.abs(prediction - target))
        ),
        "pairwise_rmse_eur_mwh": float(np.sqrt(np.mean(np.square(pair_error)))),
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _build_arrays(
    samples: list[dict[str, Any]],
    real_indices: np.ndarray,
) -> dict[str, Any]:
    by_timestamp = {pd.Timestamp(sample["timestamp"]): sample for sample in samples}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        timestamp = pd.Timestamp(sample["timestamp"])
        target = np.asarray(sample["target_vector"], dtype=np.float64)
        analytic = np.asarray(sample["analytic_vector"], dtype=np.float64)
        residual = center_price_field(target - analytic)
        quotient = quotient_feature(
            np.asarray(sample["canonical"], dtype=np.float64),
            np.asarray(sample["weights"], dtype=np.float64),
            float(sample["total"]),
            real_indices,
        )

        lagged = []
        lag_flags = []
        for lag in (pd.Timedelta(days=1), pd.Timedelta(days=7)):
            previous = by_timestamp.get(timestamp - lag)
            if previous is None:
                lagged.append(np.zeros_like(residual))
                lag_flags.append(0.0)
            else:
                previous_target = np.asarray(previous["target_vector"], dtype=np.float64)
                previous_analytic = np.asarray(
                    previous["analytic_vector"], dtype=np.float64
                )
                lagged.append(center_price_field(previous_target - previous_analytic))
                lag_flags.append(1.0)

        rows.append(
            {
                "timestamp": timestamp,
                "target": target,
                "analytic": analytic,
                "residual": residual,
                "quotient": quotient,
                "calendar": _calendar(timestamp),
                "lagged": np.concatenate([*lagged, np.asarray(lag_flags)]),
                "lag_day": lagged[0],
                "lag_week": lagged[1],
                "lag_day_present": bool(lag_flags[0]),
                "lag_week_present": bool(lag_flags[1]),
            }
        )

    return {
        "timestamps": pd.DatetimeIndex([row["timestamp"] for row in rows]),
        "target": np.stack([row["target"] for row in rows]),
        "analytic": np.stack([row["analytic"] for row in rows]),
        "residual": np.stack([row["residual"] for row in rows]),
        "quotient": np.stack([row["quotient"] for row in rows]),
        "calendar": np.stack([row["calendar"] for row in rows]),
        "lagged": np.stack([row["lagged"] for row in rows]),
        "lag_day": np.stack([row["lag_day"] for row in rows]),
        "lag_week": np.stack([row["lag_week"] for row in rows]),
        "lag_day_present": np.asarray([row["lag_day_present"] for row in rows]),
        "lag_week_present": np.asarray([row["lag_week_present"] for row in rows]),
    }


def _split_mask(timestamps: pd.DatetimeIndex, start: pd.Timestamp | None, end: pd.Timestamp) -> np.ndarray:
    mask = timestamps < end
    if start is not None:
        mask &= timestamps >= start
    return np.asarray(mask)


def _select_ridge(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_selection: np.ndarray,
    y_selection: np.ndarray,
    analytic_selection: np.ndarray,
    target_selection: np.ndarray,
) -> tuple[float, float, list[dict[str, float]]]:
    records = []
    best: tuple[float, float, float] | None = None
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_fit, y_fit)
        raw_residual = center_price_field(model.predict(x_selection))
        for shrinkage in SHRINKAGES:
            prediction = center_price_field(
                analytic_selection + shrinkage * raw_residual
            )
            loss = float(
                _pairwise_timestamp_loss(target_selection, prediction).mean()
            )
            records.append(
                {"alpha": float(alpha), "shrinkage": shrinkage, "selection_mae": loss}
            )
            key = (loss, alpha, shrinkage)
            if best is None or key < best:
                best = key
    assert best is not None
    return float(best[1]), float(best[2]), records


def _fit_ridge(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_selection: np.ndarray,
    y_selection: np.ndarray,
    x_evaluation: np.ndarray,
    analytic_selection: np.ndarray,
    target_selection: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha, shrinkage, candidates = _select_ridge(
        x_fit,
        y_fit,
        x_selection,
        y_selection,
        analytic_selection,
        target_selection,
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(
        np.concatenate([x_fit, x_selection]),
        np.concatenate([y_fit, y_selection]),
    )
    residual = center_price_field(model.predict(x_evaluation))
    return shrinkage * residual, {
        "selected_alpha": alpha,
        "selected_shrinkage": shrinkage,
        "selection_candidates": candidates,
    }


def _select_hgb_shrinkage(
    models: list[HistGradientBoostingRegressor],
    x_selection: np.ndarray,
    analytic_selection: np.ndarray,
    target_selection: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    raw = center_price_field(
        np.column_stack([model.predict(x_selection) for model in models])
    )
    records = []
    best: tuple[float, float] | None = None
    for shrinkage in SHRINKAGES:
        prediction = center_price_field(analytic_selection + shrinkage * raw)
        loss = float(_pairwise_timestamp_loss(target_selection, prediction).mean())
        records.append({"shrinkage": shrinkage, "selection_mae": loss})
        key = (loss, shrinkage)
        if best is None or key < best:
            best = key
    assert best is not None
    return float(best[1]), records


def _new_hgb(zone_index: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=20260930 + zone_index,
    )


def _fit_hgb(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_selection: np.ndarray,
    y_selection: np.ndarray,
    x_evaluation: np.ndarray,
    analytic_selection: np.ndarray,
    target_selection: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection_models = [
        _new_hgb(zone).fit(x_fit, y_fit[:, zone]) for zone in range(y_fit.shape[1])
    ]
    shrinkage, candidates = _select_hgb_shrinkage(
        selection_models,
        x_selection,
        analytic_selection,
        target_selection,
    )
    x_train = np.concatenate([x_fit, x_selection])
    y_train = np.concatenate([y_fit, y_selection])
    models = [
        _new_hgb(zone).fit(x_train, y_train[:, zone])
        for zone in range(y_train.shape[1])
    ]
    residual = center_price_field(
        np.column_stack([model.predict(x_evaluation) for model in models])
    )
    return shrinkage * residual, {
        "selected_shrinkage": shrinkage,
        "selection_candidates": candidates,
        "model": {
            "type": "per-zone HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "max_iter": 180,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 40,
            "l2_regularization": 2.0,
        },
    }


def main() -> int:
    args = _parse_args()
    if DEVELOPMENT_END > CONFIRMATION_START:
        raise RuntimeError("Development audit is not allowed to enter confirmation")

    dual_path = args.dual.expanduser().resolve()
    price_root = args.price_root.expanduser().resolve()
    frame, certificate_audit = _read_certificate(dual_path, 60.0)
    unit_columns = certificate_audit["unit_columns"]
    zones = PRIMARY_ZONES
    if not set(zones).issubset(REAL_NORDIC_ZONES):
        raise RuntimeError("Primary zones are inconsistent with the registered zone set")
    missing = [zone for zone in zones if f"unit_ptdf_{zone}" not in unit_columns]
    if missing:
        raise RuntimeError(f"Certificate is missing PTDFs for: {missing}")

    price_panel, price_provenance = _load_price_panel(price_root, zones)
    samples = _build_vector_samples(
        frame,
        price_panel,
        unit_columns,
        zones,
        FIT_END,
        DEVELOPMENT_END,
    )
    if args.smoke:
        smoke_parts = (
            [sample for sample in samples if sample["timestamp"] < FIT_END][-2000:],
            [
                sample
                for sample in samples
                if FIT_END <= sample["timestamp"] < SELECTION_END
            ][-2000:],
            [
                sample
                for sample in samples
                if SELECTION_END <= sample["timestamp"] < DEVELOPMENT_END
            ][:2000],
        )
        samples = [sample for part in smoke_parts for sample in part]

    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    arrays = _build_arrays(samples, real_indices)
    timestamps = arrays["timestamps"]
    if len(timestamps) == 0 or timestamps.max() >= CONFIRMATION_START:
        raise RuntimeError("Audit chronology is empty or crosses confirmation start")

    fit = _split_mask(timestamps, None, FIT_END)
    selection = _split_mask(timestamps, FIT_END, SELECTION_END)
    evaluation = _split_mask(timestamps, SELECTION_END, DEVELOPMENT_END)
    if min(fit.sum(), selection.sum(), evaluation.sum()) == 0:
        raise RuntimeError(
            "Empty chronology split: "
            f"fit={fit.sum()}, selection={selection.sum()}, evaluation={evaluation.sum()}"
        )

    features = {
        "context_only": np.concatenate([arrays["calendar"], arrays["lagged"]], axis=1),
        "quotient_only": arrays["quotient"],
        "quotient_calendar": np.concatenate(
            [arrays["quotient"], arrays["calendar"]], axis=1
        ),
        "quotient_calendar_lagged": np.concatenate(
            [arrays["quotient"], arrays["calendar"], arrays["lagged"]], axis=1
        ),
    }

    target_eval = arrays["target"][evaluation]
    analytic_eval = arrays["analytic"][evaluation]
    target_selection = arrays["target"][selection]
    analytic_selection = arrays["analytic"][selection]
    residual_fit = arrays["residual"][fit]
    residual_selection = arrays["residual"][selection]

    predictions: dict[str, np.ndarray] = {"analytic_zero_residual": analytic_eval}
    tuning: dict[str, Any] = {}

    day_present = arrays["lag_day_present"][evaluation]
    week_present = arrays["lag_week_present"][evaluation]
    day_residual = np.where(
        day_present[:, None], arrays["lag_day"][evaluation], 0.0
    )
    week_residual = np.where(
        week_present[:, None], arrays["lag_week"][evaluation], 0.0
    )
    predictions["previous_day_residual"] = center_price_field(
        analytic_eval + day_residual
    )
    predictions["previous_week_residual"] = center_price_field(
        analytic_eval + week_residual
    )

    for name, matrix in features.items():
        ridge_residual, ridge_tuning = _fit_ridge(
            matrix[fit],
            residual_fit,
            matrix[selection],
            residual_selection,
            matrix[evaluation],
            analytic_selection,
            target_selection,
        )
        method = f"ridge_{name}"
        predictions[method] = center_price_field(analytic_eval + ridge_residual)
        tuning[method] = ridge_tuning

    hgb_name = "hgb_quotient_calendar_lagged"
    hgb_matrix = features["quotient_calendar_lagged"]
    hgb_residual, hgb_tuning = _fit_hgb(
        hgb_matrix[fit],
        residual_fit,
        hgb_matrix[selection],
        residual_selection,
        hgb_matrix[evaluation],
        analytic_selection,
        target_selection,
    )
    predictions[hgb_name] = center_price_field(analytic_eval + hgb_residual)
    tuning[hgb_name] = hgb_tuning

    metrics = {name: _metrics(target_eval, value) for name, value in predictions.items()}
    baseline_losses = _pairwise_timestamp_loss(
        target_eval, predictions["analytic_zero_residual"]
    )
    comparisons = {}
    for name, value in predictions.items():
        if name == "analytic_zero_residual":
            continue
        comparisons[name] = paired_block_bootstrap(
            {"timestamp_pairwise_mae": baseline_losses},
            {"timestamp_pairwise_mae": _pairwise_timestamp_loss(target_eval, value)},
        )["timestamp_pairwise_mae"]

    residual_eval = arrays["residual"][evaluation]
    residual_stats = {
        "mean_abs_residual_eur_mwh": float(np.mean(np.abs(residual_eval))),
        "pairwise_residual_mae_eur_mwh": float(
            np.mean(np.abs(field_to_pairwise(residual_eval)))
        ),
        "day_lag_flat_correlation": _safe_corr(
            residual_eval[day_present].ravel(), day_residual[day_present].ravel()
        ),
        "week_lag_flat_correlation": _safe_corr(
            residual_eval[week_present].ravel(), week_residual[week_present].ravel()
        ),
        "day_lag_coverage": float(day_present.mean()),
        "week_lag_coverage": float(week_present.mean()),
    }

    report = {
        "protocol": {
            "name": "af_cdmo_residual_headroom_development_v1",
            "purpose": "diagnose AF-CDMO learning limitation before architecture changes",
            "study_role": "exploratory_development_only",
            "not_a_confirmation_result": True,
            "task": "same-delivery centered zonal price reconstruction",
            "information_sets": {
                "quotient": "registered invariant certificate moments",
                "calendar": "delivery timestamp only",
                "lagged": "strictly previous-day and previous-week residual fields",
            },
            "fit_end_exclusive": FIT_END.isoformat(),
            "selection_end_exclusive": SELECTION_END.isoformat(),
            "development_end_exclusive": DEVELOPMENT_END.isoformat(),
            "confirmation_start_exclusive": CONFIRMATION_START.isoformat(),
            "confirmation_artifacts_read": False,
            "zones": list(zones),
            "smoke": bool(args.smoke),
        },
        "rows": {
            "fit": int(fit.sum()),
            "selection": int(selection.sum()),
            "development_evaluation": int(evaluation.sum()),
        },
        "feature_dimensions": {name: int(value.shape[1]) for name, value in features.items()},
        "residual_diagnostics": residual_stats,
        "development_evaluation_metrics": metrics,
        "paired_vs_analytic": comparisons,
        "selection_tuning": tuning,
        "sources": {
            "dual_path": str(dual_path),
            "dual_sha256": sha256_file(dual_path),
            "price_provenance": price_provenance,
            "engine_path": str(Path(__file__).resolve()),
            "engine_sha256": _sha256(Path(__file__).resolve()),
        },
        "interpretation_contract": {
            "positive_headroom_requires": (
                "a candidate with positive block-bootstrap skill over the analytic "
                "baseline and a 95% interval excluding zero"
            ),
            "no_dl_claim": (
                "This audit tests information headroom with simple controls; it does "
                "not establish a neural-architecture contribution."
            ),
            "confirmation_rule": (
                "No model or hyperparameter may be selected using timestamps on or "
                "after 2026-03-01 UTC."
            ),
        },
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"[OK] wrote {output}")
    print(json.dumps({"rows": report["rows"], "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
