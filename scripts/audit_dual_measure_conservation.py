"""Audit the economic conservation law carried by the canonical dual measure.

For a flow-based inequality row ``a_i^T x <= b_i`` with multiplier
``lambda_i``, canonicalization stores ``u_i = a_i / ||a_i||`` and dual mass
``m_i = lambda_i ||a_i||``. Therefore, for a zonal transfer ``delta``,
``sum_i m_i <u_i, delta> = sum_i lambda_i <a_i, delta>``.

The comparison with the observed day-ahead spread is a diagnostic, not a claim
that CNEC duals explain bid bounds or every other KKT term in the auction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.cqdm_utils import expand_native_hourly_certificates
except ModuleNotFoundError:
    from cqdm_utils import expand_native_hourly_certificates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUAL = (
    ROOT
    / "official_jao_dual_measure"
    / "NordicBindingDualMeasure_2025_2026_full_history.csv.gz"
)
DEFAULT_MARKET_ROOT = ROOT / "official_post_golive_market"
DEFAULT_OUTPUT = (
    ROOT
    / "official_jao_dual_measure"
    / "dual_measure_conservation_full_history_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_transfer_potential(
    mass: np.ndarray, unit_from: np.ndarray, unit_to: np.ndarray
) -> float:
    """Return the invariant dual potential for a zonal transfer."""

    return float(np.dot(mass, unit_to - unit_from))


def raw_transfer_potential(
    multiplier: np.ndarray, row_from: np.ndarray, row_to: np.ndarray
) -> float:
    """Return the same potential in a non-canonical row representation."""

    return float(np.dot(multiplier, row_to - row_from))


def _load_day_ahead(root: Path, area: str) -> pd.Series:
    path = root / f"DayAheadPrices_{area}_post_golive.csv.gz"
    frame = pd.read_csv(path, usecols=["TimeUTC", "DayAheadPriceEUR"])
    frame["delivery_utc"] = pd.to_datetime(
        frame["TimeUTC"], utc=True, errors="raise", format="mixed"
    )
    values = pd.to_numeric(frame["DayAheadPriceEUR"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame["delivery_utc"], name=area)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[finite]
    predicted = predicted[finite]
    if not len(actual):
        raise RuntimeError("No finite conservation-audit observations")
    residual = predicted - actual
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0.0 and np.std(predicted) > 0.0
        else float("nan")
    )
    return {
        "rows": int(len(actual)),
        "correlation": correlation,
        "mae_eur_mwh": float(np.mean(np.abs(residual))),
        "rmse_eur_mwh": float(np.sqrt(np.mean(np.square(residual)))),
        "within_0_01_eur_fraction": float(np.mean(np.abs(residual) <= 0.01)),
        "within_0_10_eur_fraction": float(np.mean(np.abs(residual) <= 0.10)),
        "within_1_00_eur_fraction": float(np.mean(np.abs(residual) <= 1.00)),
        "actual_p01_p50_p99_eur_mwh": [
            float(value) for value in np.quantile(actual, [0.01, 0.50, 0.99])
        ],
        "predicted_p01_p50_p99_eur_mwh": [
            float(value) for value in np.quantile(predicted, [0.01, 0.50, 0.99])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--market_root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lead_minutes", type=float, default=60.0)
    args = parser.parse_args()

    dual_path = args.dual.resolve()
    metadata_path = dual_path.with_suffix(dual_path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = _sha256(dual_path)
    if metadata.get("sha256") != digest:
        raise RuntimeError("Dual-measure artifact hash mismatch")

    columns = [
        "delivery_utc",
        "publication_utc",
        "dual_mass_eur_mwh",
        "unit_ptdf_DK1",
        "unit_ptdf_DK2",
    ]
    frame = pd.read_csv(dual_path, usecols=columns)
    frame["delivery_utc"] = pd.to_datetime(
        frame["delivery_utc"], utc=True, errors="raise", format="mixed"
    )
    frame["publication_utc"] = pd.to_datetime(
        frame["publication_utc"], utc=True, errors="raise", format="mixed"
    )
    frame = expand_native_hourly_certificates(frame)
    decision = frame["delivery_utc"] - pd.to_timedelta(args.lead_minutes, unit="m")
    frame = frame.loc[frame["publication_utc"] <= decision].copy()
    numeric = ["dual_mass_eur_mwh", "unit_ptdf_DK1", "unit_ptdf_DK2"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric)
    frame = frame.loc[frame["dual_mass_eur_mwh"] > 0.0]

    frame["dk1_to_dk2_potential_eur_mwh"] = frame["dual_mass_eur_mwh"] * (
        frame["unit_ptdf_DK2"] - frame["unit_ptdf_DK1"]
    )
    potential = frame.groupby("delivery_utc")[
        "dk1_to_dk2_potential_eur_mwh"
    ].sum()
    actual = (
        _load_day_ahead(args.market_root.resolve(), "DK1")
        - _load_day_ahead(args.market_root.resolve(), "DK2")
    ).rename("observed_dk1_minus_dk2_eur_mwh")
    joined = pd.concat(
        [actual, potential.rename("certificate_dk1_minus_dk2_eur_mwh")],
        axis=1,
        join="inner",
    ).dropna()

    sample = frame.head(min(len(frame), 4096))
    mass = sample["dual_mass_eur_mwh"].to_numpy(dtype=np.float64)
    unit_from = sample["unit_ptdf_DK1"].to_numpy(dtype=np.float64)
    unit_to = sample["unit_ptdf_DK2"].to_numpy(dtype=np.float64)
    norm = np.geomspace(0.1, 10.0, num=len(sample))
    raw = raw_transfer_potential(mass / norm, norm * unit_from, norm * unit_to)
    canonical = canonical_transfer_potential(mass, unit_from, unit_to)

    report = {
        "purpose": (
            "Verify CQDM's exact dual-potential conservation law and diagnose how "
            "much DK1-DK2 day-ahead spread is represented by CNEC duals."
        ),
        "not_a_claim": (
            "CNEC duals need not reconstruct price-bound, bid-constraint, or other "
            "non-network KKT components of the auction price spread."
        ),
        "source": {
            "dual_path": str(dual_path),
            "dual_sha256": digest,
            "lead_minutes": args.lead_minutes,
            "timely_binding_rows": int(len(frame)),
        },
        "algebraic_identity": {
            "canonical_potential": canonical,
            "reconstructed_raw_potential": raw,
            "absolute_error": float(abs(canonical - raw)),
        },
        "dk1_minus_dk2": _metrics(
            joined["observed_dk1_minus_dk2_eur_mwh"].to_numpy(dtype=np.float64),
            joined["certificate_dk1_minus_dk2_eur_mwh"].to_numpy(dtype=np.float64),
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
