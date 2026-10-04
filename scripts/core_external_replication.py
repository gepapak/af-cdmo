"""Shared data algebra for the Core-CCR external AF-CDMO replication.

This module deliberately contains no model fitting.  It turns JAO Core
day-ahead active-constraint rows into positive dual-measure atoms and directed
Core price spreads into a centered physical-zone price field.  The only Core
equality used by the external replication is global net-position balance.  We
do not infer undocumented ALEGrO or virtual-hub equalities from the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CORE_API_BASE = "https://publicationtool.jao.eu/core/api/data"
CORE_SHADOW_ENDPOINT = f"{CORE_API_BASE}/shadowPrices"
CORE_SPREAD_ENDPOINT = f"{CORE_API_BASE}/priceSpread"

CORE_PHYSICAL_ZONES = (
    "AT",
    "BE",
    "CZ",
    "DE",
    "FR",
    "HR",
    "HU",
    "NL",
    "PL",
    "RO",
    "SI",
    "SK",
)

CORE_IDENTITY_COLUMNS = (
    "id",
    "dateTimeUtc",
    "tso",
    "cnecName",
    "cnecEic",
    "direction",
    "contName",
    "branchEic",
    "hubFrom",
    "hubTo",
)
CORE_NUMERIC_COLUMNS = (
    "shadowPrice",
    "ram",
    "ramMcp",
    "imax",
    "fmax",
    "frm",
    "fref",
    "f0core",
    "f0all",
    "fuaf",
    "amr",
    "ltaMargin",
    "cva",
    "iva",
    "ftotalLtn",
    "minRamFactor",
    "maxZ2ZPtdf",
)
CORE_RAW_TAIL_COLUMNS = ("ramMcp", "fmax", "fref", "f0all")
CORE_CANONICAL_SCALAR_COLUMNS = (
    "canonical_ram_mw",
    *CORE_RAW_TAIL_COLUMNS,
)
SHADOW_BINDING_EPS = 1.0e-6
_BORDER_RE = re.compile(r"^border_(.+)$")


class CoreSchemaError(RuntimeError):
    """Raised when the public Core payload cannot support the preregistration."""


@dataclass(frozen=True)
class PriceFieldDiagnostics:
    rows: int
    connected_rows: int
    missing_rows: int
    reverse_pairs_checked: int
    reverse_antisymmetry_max_eur_mwh: float
    cycle_residual_rms_eur_mwh: float
    cycle_residual_p95_eur_mwh: float
    cycle_residual_max_eur_mwh: float
    minimum_graph_rank: int
    maximum_graph_rank: int
    spread_convention: str


def parse_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def core_local_delivery_windows(
    start: Any,
    end: Any,
    *,
    timezone_name: str = "Europe/Brussels",
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Partition a UTC interval without crossing a Core local delivery day.

    Core day-ahead publication batches follow CET/CEST delivery dates.  A UTC
    calendar-day request can straddle two such batches (for example 23:00Z is
    already the next local day in winter), making the response-level
    ``lastModifiedOn`` unusable for row-level causality.  These windows are
    therefore bounded by actual local midnights and naturally contain 23 or 25
    hours across daylight-saving transitions.
    """

    cursor = parse_utc(start)
    stop = parse_utc(end)
    if cursor >= stop:
        raise ValueError("Delivery-window start must precede end")
    zone = ZoneInfo(timezone_name)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while cursor < stop:
        local = cursor.tz_convert(zone)
        next_date: date = local.date() + timedelta(days=1)
        boundary_local = pd.Timestamp(datetime.combine(next_date, time.min), tz=zone)
        boundary = boundary_local.tz_convert("UTC")
        next_cursor = min(boundary, stop)
        if next_cursor <= cursor:
            raise RuntimeError("Failed to advance Core local delivery window")
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return tuple(windows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=indent, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a byte-stable gzip CSV (fixed gzip timestamp and no index)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    temporary.replace(path)


def core_hub_columns(columns: Iterable[str]) -> tuple[str, ...]:
    discovered = {str(column) for column in columns if str(column).startswith("hub_")}
    required = {f"hub_{zone}" for zone in CORE_PHYSICAL_ZONES}
    missing = sorted(required.difference(discovered))
    if missing:
        raise CoreSchemaError(f"Core active-constraint payload omitted hubs: {missing}")
    physical = tuple(f"hub_{zone}" for zone in CORE_PHYSICAL_ZONES)
    virtual = tuple(sorted(discovered.difference(physical)))
    return (*physical, *virtual)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def canonicalize_shadow_rows(
    rows: list[dict[str, Any]],
    *,
    shadow_epsilon: float = SHADOW_BINDING_EPS,
    norm_tolerance: float = 1.0e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Canonicalize Core active CNECs while preserving every observed hub axis.

    Each input row must carry ``_publication_utc`` injected from the top-level
    JAO response.  This is required because the Core endpoint does not expose a
    row-level publication timestamp.
    """

    if shadow_epsilon < 0.0 or norm_tolerance <= 0.0:
        raise ValueError("Invalid shadow or norm tolerance")
    if not rows:
        raise CoreSchemaError("No Core active-constraint rows were supplied")
    frame = pd.DataFrame.from_records(rows)
    required = {"dateTimeUtc", "shadowPrice", "ram", "_publication_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CoreSchemaError(f"Core active-constraint response omitted fields: {missing}")

    hubs = core_hub_columns(frame.columns)
    for column in CORE_NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    _numeric(frame, (*CORE_NUMERIC_COLUMNS, *hubs))
    if frame["shadowPrice"].isna().any():
        raise CoreSchemaError("Core shadowPrice contains non-numeric values")
    if (frame["shadowPrice"] < -shadow_epsilon).any():
        minimum = float(frame["shadowPrice"].min())
        raise CoreSchemaError(f"Core active multipliers must be non-negative; min={minimum}")

    raw_rows = int(len(frame))
    frame = frame.loc[frame["shadowPrice"] > shadow_epsilon].copy()
    if frame.empty:
        raise CoreSchemaError("No positive-dual Core active constraints remain")
    if frame[list(hubs)].isna().all(axis=1).any():
        raise CoreSchemaError("At least one positive-dual row has no PTDF coordinates")
    ptdf = frame[list(hubs)].fillna(0.0).to_numpy(dtype=np.float64)
    norms = np.linalg.norm(ptdf, axis=1)
    zero_norm = norms <= norm_tolerance
    if np.any(zero_norm):
        bad = int(np.sum(zero_norm))
        raise CoreSchemaError(
            f"{bad} positive-dual Core rows have zero PTDF norm; refusing to drop them"
        )

    frame["delivery_utc"] = pd.to_datetime(
        frame["dateTimeUtc"], utc=True, errors="raise", format="mixed"
    )
    frame["publication_utc"] = pd.to_datetime(
        frame["_publication_utc"], utc=True, errors="raise", format="mixed"
    )
    if (frame["publication_utc"] > frame["delivery_utc"]).any():
        offending = frame.loc[
            frame["publication_utc"] > frame["delivery_utc"],
            ["delivery_utc", "publication_utc"],
        ].head(3)
        raise CoreSchemaError(
            "Post-delivery active certificates violate causal availability: "
            f"{offending.to_dict(orient='records')}"
        )

    frame["ptdf_l2_norm"] = norms
    frame["dual_mass_eur_mwh"] = frame["shadowPrice"].to_numpy(dtype=np.float64) * norms
    frame["canonical_ram_mw"] = frame["ram"].fillna(0.0).to_numpy(dtype=np.float64) / norms
    for index, column in enumerate(hubs):
        frame[f"unit_{column}"] = ptdf[:, index] / norms

    identity = [column for column in CORE_IDENTITY_COLUMNS if column in frame]
    output_columns = [
        "delivery_utc",
        "publication_utc",
        *identity,
        *CORE_NUMERIC_COLUMNS,
        "ptdf_l2_norm",
        "dual_mass_eur_mwh",
        "canonical_ram_mw",
        *(f"unit_{column}" for column in hubs),
    ]
    output = frame[output_columns].copy()
    output = output.replace([np.inf, -np.inf], np.nan)
    essential = [
        "delivery_utc",
        "publication_utc",
        "shadowPrice",
        "ram",
        "ptdf_l2_norm",
        "dual_mass_eur_mwh",
        "canonical_ram_mw",
        *(f"unit_{column}" for column in hubs),
    ]
    if output[essential].isna().any().any():
        bad = output[essential].isna().sum()
        raise CoreSchemaError(
            f"Canonical Core certificate contains missing essential values: "
            f"{bad[bad > 0].to_dict()}"
        )
    output = output.sort_values(
        ["delivery_utc", "cnecName", "contName", "id"],
        na_position="last",
    ).reset_index(drop=True)
    audit = {
        "raw_rows": raw_rows,
        "binding_rows": int(len(output)),
        "delivery_timestamps": int(output["delivery_utc"].nunique()),
        "hub_columns": list(hubs),
        "physical_hub_count": len(CORE_PHYSICAL_ZONES),
        "virtual_hub_count": len(hubs) - len(CORE_PHYSICAL_ZONES),
        "minimum_shadow_price": float(output["shadowPrice"].min()),
        "maximum_shadow_price": float(output["shadowPrice"].max()),
        "minimum_ptdf_norm": float(output["ptdf_l2_norm"].min()),
        "causal_publication_share": float(
            np.mean(output["publication_utc"] <= output["delivery_utc"])
        ),
    }
    return output, audit


def spread_columns(columns: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted(str(column) for column in columns if _BORDER_RE.match(str(column))))
    if not values:
        raise CoreSchemaError("Core price-spread response contains no border columns")
    return values


def _edge_values(
    row: pd.Series,
    left: str,
    right: str,
    *,
    convention: str,
) -> tuple[float | None, float | None]:
    if convention not in {"source_minus_destination", "destination_minus_source"}:
        raise ValueError(f"Unknown price-spread convention: {convention}")
    sign = 1.0 if convention == "source_minus_destination" else -1.0
    forward = row.get(f"border_{left}_{right}", np.nan)
    reverse = row.get(f"border_{right}_{left}", np.nan)
    forward_value = None if pd.isna(forward) else sign * float(forward)
    reverse_value = None if pd.isna(reverse) else -sign * float(reverse)
    return forward_value, reverse_value


def reconstruct_centered_price_field(
    spread_rows: pd.DataFrame,
    *,
    convention: str,
    zones: tuple[str, ...] = CORE_PHYSICAL_ZONES,
    reverse_tolerance: float = 1.0e-6,
) -> tuple[pd.DataFrame, PriceFieldDiagnostics]:
    """Recover the centered physical-zone field from the directed spread graph."""

    if "dateTimeUtc" not in spread_rows:
        raise CoreSchemaError("Core price-spread response omitted dateTimeUtc")
    frame = spread_rows.copy()
    columns = spread_columns(frame.columns)
    _numeric(frame, columns)
    frame = frame.copy()
    frame["delivery_utc"] = pd.to_datetime(
        frame["dateTimeUtc"], utc=True, errors="raise", format="mixed"
    )
    if frame["delivery_utc"].duplicated().any():
        grouped = []
        for delivery, group in frame.groupby("delivery_utc", sort=True):
            combined: dict[str, Any] = {"delivery_utc": delivery, "dateTimeUtc": delivery}
            for column in columns:
                finite = group[column].dropna().to_numpy(dtype=np.float64)
                if len(finite) and np.ptp(finite) > reverse_tolerance:
                    raise CoreSchemaError(
                        f"Conflicting duplicate spread values at {delivery} for {column}"
                    )
                combined[column] = float(finite[0]) if len(finite) else np.nan
            grouped.append(combined)
        frame = pd.DataFrame(grouped)

    records: list[dict[str, Any]] = []
    reverse_errors: list[float] = []
    cycle_residuals: list[float] = []
    ranks: list[int] = []
    missing_rows = 0
    zone_index = {zone: index for index, zone in enumerate(zones)}
    for _, row in frame.sort_values("delivery_utc").iterrows():
        incidence: list[np.ndarray] = []
        observations: list[float] = []
        reverse_checked = 0
        for left_index, left in enumerate(zones):
            for right in zones[left_index + 1 :]:
                forward, reverse = _edge_values(
                    row, left, right, convention=convention
                )
                if forward is None and reverse is None:
                    continue
                if forward is not None and reverse is not None:
                    error = abs(forward - reverse)
                    reverse_errors.append(error)
                    reverse_checked += 1
                    if error > reverse_tolerance:
                        raise CoreSchemaError(
                            "Directed Core spreads violate reverse anti-symmetry at "
                            f"{row['delivery_utc']}: {left}/{right} error={error:.6g}"
                        )
                    value = 0.5 * (forward + reverse)
                else:
                    value = forward if forward is not None else reverse
                edge = np.zeros(len(zones), dtype=np.float64)
                edge[zone_index[left]] = 1.0
                edge[zone_index[right]] = -1.0
                incidence.append(edge)
                observations.append(float(value))

        if not incidence:
            missing_rows += 1
            continue
        matrix = np.stack(incidence)
        values = np.asarray(observations, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(matrix))
        ranks.append(rank)
        if rank != len(zones) - 1:
            raise CoreSchemaError(
                f"Disconnected Core physical-zone spread graph at {row['delivery_utc']}: "
                f"rank={rank}, expected={len(zones) - 1}"
            )
        augmented = np.vstack([matrix, np.ones((1, len(zones)), dtype=np.float64)])
        right_hand_side = np.concatenate([values, np.zeros(1, dtype=np.float64)])
        field, *_ = np.linalg.lstsq(augmented, right_hand_side, rcond=None)
        field -= field.mean()
        residual = matrix @ field - values
        cycle_residuals.extend(np.abs(residual).tolist())
        record = {
            "delivery_utc": row["delivery_utc"],
            **{
                f"price_{zone}_centered_eur_mwh": float(field[index])
                for index, zone in enumerate(zones)
            },
            "spread_graph_rank": rank,
            "spread_edges": len(matrix),
            "spread_reverse_pairs_checked": reverse_checked,
            "spread_cycle_residual_rms_eur_mwh": float(
                np.sqrt(np.mean(np.square(residual)))
            ),
            "spread_cycle_residual_max_eur_mwh": float(
                np.max(np.abs(residual), initial=0.0)
            ),
        }
        records.append(record)

    if not records:
        raise CoreSchemaError("No connected Core price-spread fields were reconstructed")
    residual_array = np.asarray(cycle_residuals, dtype=np.float64)
    reverse_array = np.asarray(reverse_errors, dtype=np.float64)
    diagnostics = PriceFieldDiagnostics(
        rows=int(len(frame)),
        connected_rows=len(records),
        missing_rows=missing_rows,
        reverse_pairs_checked=int(len(reverse_array)),
        reverse_antisymmetry_max_eur_mwh=float(
            np.max(reverse_array, initial=0.0)
        ),
        cycle_residual_rms_eur_mwh=float(
            np.sqrt(np.mean(np.square(residual_array))) if len(residual_array) else 0.0
        ),
        cycle_residual_p95_eur_mwh=float(
            np.quantile(residual_array, 0.95) if len(residual_array) else 0.0
        ),
        cycle_residual_max_eur_mwh=float(
            np.max(residual_array, initial=0.0)
        ),
        minimum_graph_rank=int(min(ranks)),
        maximum_graph_rank=int(max(ranks)),
        spread_convention=convention,
    )
    return pd.DataFrame.from_records(records), diagnostics


def analytic_core_price_field(
    dual_frame: pd.DataFrame,
    *,
    zones: tuple[str, ...] = CORE_PHYSICAL_ZONES,
) -> pd.DataFrame:
    unit_columns = [f"unit_hub_{zone}" for zone in zones]
    missing = sorted(set(unit_columns).difference(dual_frame.columns))
    if missing:
        raise CoreSchemaError(f"Canonical dual table omitted physical hubs: {missing}")
    records = []
    for delivery, group in dual_frame.groupby("delivery_utc", sort=True):
        mass = group["dual_mass_eur_mwh"].to_numpy(dtype=np.float64)
        unit = group[unit_columns].to_numpy(dtype=np.float64)
        field = -(mass @ unit)
        field -= field.mean()
        records.append(
            {
                "delivery_utc": pd.Timestamp(delivery),
                **{
                    f"analytic_{zone}_eur_mwh": float(field[index])
                    for index, zone in enumerate(zones)
                },
            }
        )
    return pd.DataFrame.from_records(records)


def infer_spread_convention(
    spread_rows: pd.DataFrame,
    dual_frame: pd.DataFrame,
    *,
    audit_end: pd.Timestamp,
    minimum_overlap: int = 24,
    decisiveness_ratio: float = 0.90,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    """Resolve the API sign convention using only the preregistered audit prefix."""

    audit_end = parse_utc(audit_end)
    analytic = analytic_core_price_field(dual_frame)
    score: dict[str, float] = {}
    panels: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}
    analytic_columns = [f"analytic_{zone}_eur_mwh" for zone in CORE_PHYSICAL_ZONES]
    for convention in ("source_minus_destination", "destination_minus_source"):
        panel, diag = reconstruct_centered_price_field(
            spread_rows, convention=convention
        )
        merged = panel.merge(analytic, on="delivery_utc", how="inner")
        merged = merged.loc[merged["delivery_utc"] < audit_end]
        if len(merged) < minimum_overlap:
            raise CoreSchemaError(
                f"Only {len(merged)} sign-audit timestamps precede {audit_end}; "
                f"need at least {minimum_overlap}"
            )
        target = merged[
            [f"price_{zone}_centered_eur_mwh" for zone in CORE_PHYSICAL_ZONES]
        ].to_numpy(dtype=np.float64)
        predicted = merged[analytic_columns].to_numpy(dtype=np.float64)
        score[convention] = float(np.mean(np.abs(target - predicted)))
        panels[convention] = panel
        diagnostics[convention] = diag.__dict__

    selected = min(score, key=score.get)
    rejected = next(value for value in score if value != selected)
    ratio = score[selected] / max(score[rejected], 1.0e-12)
    if ratio > decisiveness_ratio:
        raise CoreSchemaError(
            "Core price-spread sign convention is empirically ambiguous on the "
            f"registered audit prefix: scores={score}, ratio={ratio:.4f}"
        )
    audit = {
        "audit_end_utc": audit_end.isoformat(),
        "minimum_overlap": minimum_overlap,
        "decisiveness_ratio_threshold": decisiveness_ratio,
        "scores_mae_eur_mwh": score,
        "selected": selected,
        "selected_to_rejected_ratio": ratio,
        "field_diagnostics": diagnostics,
        "purpose": "one-bit external API schema orientation; not model selection",
    }
    return selected, audit, panels[selected]


def global_balance_geometry(hub_names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(hub_names)
    if len(names) < 2 or len(names) != len(set(names)):
        raise CoreSchemaError("Core hub names must be unique and non-trivial")
    equality = np.ones((1, len(names)), dtype=np.float64)
    tangent = np.eye(len(names), dtype=np.float64) - np.ones(
        (len(names), len(names)), dtype=np.float64
    ) / len(names)
    return equality, tangent


def dataframe_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
