"""Audit a loss-safe affine geometry for a real Nordic AF-CDMO study.

This is a descriptive, fail-closed audit.  It does not train a model and does
not modify any frozen v10-v12 artifact.  The declared geometry contains only:

* the global net-position balance already used by Slack-QDM; and
* four internal Nordic HVDC endpoint equalities with zero implicit loss factor.

Skagerrak is deliberately excluded because its equality is direction-dependent
through a non-zero implicit loss factor.  The audit verifies that every selected
physical-zone transfer query is tangent to the declared geometry and quantifies
how much of the published PTDF row norm is an exactly removable equality gauge.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.cqdm_utils import sha256_file
    from scripts.multizone_quotient_gauge_v6 import REAL_NORDIC_ZONES
except ModuleNotFoundError:
    from cqdm_utils import sha256_file
    from multizone_quotient_gauge_v6 import REAL_NORDIC_ZONES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUAL = (
    ROOT
    / "official_jao_dual_measure"
    / "NordicBindingDualMeasure_2025_2026_full_history.csv.gz"
)
DEFAULT_AFFINE_AUDIT = (
    ROOT / "official_jao_dual_measure" / "affine_feasible_gauge_audit_v11.json"
)
DEFAULT_OUTPUT = (
    ROOT / "official_jao_dual_measure" / "af_cdmo_real_geometry_audit_v13.json"
)

HANDBOOK_URL = (
    "https://publicationtool.jao.eu/PublicationHandbook/"
    "Nordic_PublicationTool_Handbook_v1.5.pdf"
)
NORDIC_RCC_QA_URL = "https://nordic-rcc.net/flow-based/qa-epr/"

# These are the internal Nordic HVDC endpoint pairs documented as equality
# constraints.  The Nordic RCC states that Skagerrak has a non-zero,
# direction-dependent implicit loss factor and that the other internal Nordic
# HVDC links have zero ILF.  Skagerrak is therefore absent by construction.
ZERO_LOSS_INTERNAL_HVDC: dict[str, tuple[str, str]] = {
    "storebaelt": ("DK1_SB", "DK2_SB"),
    "fennoskan": ("SE3_FS", "FI_FS"),
    "kontiskan": ("DK1_KS", "SE3_KS"),
    "southwestlink": ("SE3_SWL", "SE4_SWL"),
}


def build_loss_safe_geometry(
    zone_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return global balance plus the four zero-loss HVDC equalities."""

    if len(zone_names) != len(set(zone_names)) or len(zone_names) < 2:
        raise ValueError("zone_names must be unique and contain at least two zones")
    zone_to_index = {zone: index for index, zone in enumerate(zone_names)}
    required = {
        zone for pair in ZERO_LOSS_INTERNAL_HVDC.values() for zone in pair
    }
    missing = sorted(required.difference(zone_to_index))
    if missing:
        raise RuntimeError(f"Archive is missing required HVDC endpoint zones: {missing}")

    rows = [np.ones(len(zone_names), dtype=np.float64)]
    names = ["global_net_position_balance"]
    for name, pair in ZERO_LOSS_INTERNAL_HVDC.items():
        row = np.zeros(len(zone_names), dtype=np.float64)
        for zone in pair:
            row[zone_to_index[zone]] = 1.0
        rows.append(row)
        names.append(f"internal_hvdc_{name}")
    return np.vstack(rows), names


def tangent_projector(equality: np.ndarray) -> np.ndarray:
    matrix = np.asarray(equality, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("equality must be a finite matrix")
    projector = np.eye(matrix.shape[1]) - matrix.T @ np.linalg.pinv(
        matrix @ matrix.T
    ) @ matrix
    return 0.5 * (projector + projector.T)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0)
    labels = ("min", "p01", "p05", "p10", "median", "p90", "p95", "p99", "max")
    return {
        label: float(value)
        for label, value in zip(labels, np.quantile(values, levels), strict=True)
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--affine_audit", type=Path, default=DEFAULT_AFFINE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--zones", nargs="+", choices=REAL_NORDIC_ZONES, default=REAL_NORDIC_ZONES
    )
    parser.add_argument("--chunk_size", type=int, default=50_000)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dual = args.dual.expanduser().resolve()
    affine_audit = args.affine_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; use --force")
    if args.chunk_size < 1 or args.tolerance <= 0.0:
        raise ValueError("chunk_size and tolerance must be positive")
    if not dual.is_file() or not affine_audit.is_file():
        raise FileNotFoundError("Required frozen dual archive or affine audit is missing")

    metadata_path = dual.with_suffix(dual.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dual metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dual_hash = sha256_file(dual)
    if metadata.get("sha256") != dual_hash:
        raise RuntimeError("Frozen dual archive hash does not match its metadata")

    prior_audit = json.loads(affine_audit.read_text(encoding="utf-8"))
    zone_names = list(prior_audit["scope"]["zone_order"])
    equality, equality_names = build_loss_safe_geometry(zone_names)
    projector = tangent_projector(equality)
    rank = int(np.linalg.matrix_rank(equality))
    idempotence = float(np.max(np.abs(projector @ projector - projector)))
    orthogonality = float(np.max(np.abs(equality @ projector)))
    if idempotence > args.tolerance or orthogonality > args.tolerance:
        raise RuntimeError("Loss-safe tangent projector failed its numerical contract")

    observed_zones = tuple(dict.fromkeys(args.zones))
    if len(observed_zones) < 3:
        raise ValueError("At least three distinct observed physical zones are required")
    missing_observed = sorted(set(observed_zones).difference(zone_names))
    if missing_observed:
        raise RuntimeError(f"Observed zones are absent from the PTDF archive: {missing_observed}")
    zone_to_index = {zone: index for index, zone in enumerate(zone_names)}
    eye = np.eye(len(zone_names), dtype=np.float64)
    queries = []
    for left, right in combinations(observed_zones, 2):
        direction = eye[zone_to_index[left]] - eye[zone_to_index[right]]
        residual = equality @ direction
        max_residual = float(np.max(np.abs(residual), initial=0.0))
        queries.append(
            {
                "pair": f"{left}-{right}",
                "max_abs_equality_residual": max_residual,
                "tangent": max_residual <= args.tolerance,
            }
        )
    invalid_queries = [item for item in queries if not item["tangent"]]
    if invalid_queries:
        raise RuntimeError(
            "At least one requested price-pair query is not tangent: "
            + ", ".join(item["pair"] for item in invalid_queries)
        )

    unit_columns = [f"unit_ptdf_{zone}" for zone in zone_names]
    columns = ["ptdf_l2_norm", *unit_columns]
    norm_ratios: list[np.ndarray] = []
    equality_residual_max = 0.0
    sample_normals: list[np.ndarray] = []
    rows = 0
    for chunk in pd.read_csv(dual, usecols=columns, chunksize=args.chunk_size):
        numeric = chunk[columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise RuntimeError("Non-numeric PTDF values encountered in frozen archive")
        published_norm = numeric["ptdf_l2_norm"].to_numpy(dtype=np.float64)
        normals = numeric[unit_columns].to_numpy(dtype=np.float64) * published_norm[:, None]
        ambient_norm = np.linalg.norm(normals, axis=1)
        if np.any(ambient_norm <= 0.0):
            raise RuntimeError("Frozen archive contains a zero ambient PTDF norm")
        projected = normals @ projector
        projected_norm = np.linalg.norm(projected, axis=1)
        if np.any(projected_norm <= args.tolerance):
            raise RuntimeError(
                "A positive-dual row is constant on the loss-safe feasible manifold"
            )
        norm_ratios.append(projected_norm / ambient_norm)
        equality_residual_max = max(
            equality_residual_max,
            float(np.max(np.abs(projected @ equality.T), initial=0.0)),
        )
        if sum(len(value) for value in sample_normals) < 4096:
            remaining = 4096 - sum(len(value) for value in sample_normals)
            sample_normals.append(normals[:remaining].copy())
        rows += len(normals)

    ratios = np.concatenate(norm_ratios)
    sample = np.concatenate(sample_normals)
    rng = np.random.default_rng(20260920)
    gauge = rng.normal(0.0, 2.0, size=(len(sample), equality.shape[0]))
    rewritten = sample + gauge @ equality
    rewrite_drift = float(
        np.max(np.abs((rewritten @ projector) - (sample @ projector)), initial=0.0)
    )
    if rewrite_drift > 100.0 * args.tolerance:
        raise RuntimeError("Equality-gauge rewrite changed the projected PTDF rows")

    basis_rng = np.random.default_rng(20260921)
    basis_transform, _ = np.linalg.qr(
        basis_rng.normal(size=(equality.shape[0], equality.shape[0]))
    )
    basis_transform = np.diag(np.linspace(0.5, 2.0, equality.shape[0])) @ basis_transform
    rewritten_equality = basis_transform @ equality
    rewritten_projector = tangent_projector(rewritten_equality)
    basis_projector_drift = float(
        np.max(np.abs(rewritten_projector - projector), initial=0.0)
    )
    basis_projection_drift = float(
        np.max(
            np.abs((sample @ rewritten_projector) - (sample @ projector)),
            initial=0.0,
        )
    )
    if max(basis_projector_drift, basis_projection_drift) > 100.0 * args.tolerance:
        raise RuntimeError("Full-rank equality-basis rewrite changed the quotient")

    report = {
        "protocol": {
            "name": "af_cdmo_real_loss_safe_geometry_v13",
            "purpose": (
                "Pre-register a conservative real-market affine geometry and verify "
                "that the observed Nordic zonal-price queries are well-defined on it."
            ),
            "descriptive_only": True,
            "not_a_predictive_result": True,
            "skagerrak_excluded": True,
            "reason_skagerrak_excluded": (
                "Its documented implicit-loss equality is direction-dependent."
            ),
        },
        "sources": {
            "dual_archive": str(dual),
            "dual_archive_sha256": dual_hash,
            "dual_metadata": str(metadata_path),
            "dual_metadata_sha256": sha256_file(metadata_path),
            "prior_affine_audit": str(affine_audit),
            "prior_affine_audit_sha256": sha256_file(affine_audit),
            "jao_handbook": HANDBOOK_URL,
            "nordic_rcc_equality_qa": NORDIC_RCC_QA_URL,
        },
        "geometry": {
            "ambient_dimension": len(zone_names),
            "equality_rows": equality_names,
            "equality_rank": rank,
            "tangent_dimension": len(zone_names) - rank,
            "zone_order": zone_names,
            "projector_idempotence_max_abs": idempotence,
            "projector_equality_residual_max_abs": orthogonality,
        },
        "observed_transfer_queries": {
            "zones": list(observed_zones),
            "pair_count": len(queries),
            "all_tangent": True,
            "max_abs_equality_residual": max(
                item["max_abs_equality_residual"] for item in queries
            ),
            "pairs": queries,
        },
        "archive_projection": {
            "rows": rows,
            "projected_to_ambient_norm_ratio": _quantiles(ratios),
            "mean_fraction_ambient_norm_removed": float(np.mean(1.0 - ratios)),
            "rows_at_or_below_projected_norm_tolerance": 0,
            "projected_equality_residual_max_abs": equality_residual_max,
        },
        "registered_gauge_rewrite": {
            "sample_rows": len(sample),
            "seed": 20260920,
            "projected_normal_max_abs_drift": rewrite_drift,
            "status": "passed",
        },
        "registered_equality_basis_rewrite": {
            "seed": 20260921,
            "transform_rank": int(np.linalg.matrix_rank(basis_transform)),
            "projector_max_abs_drift": basis_projector_drift,
            "projected_normal_max_abs_drift": basis_projection_drift,
            "status": "passed",
        },
        "claim_boundary": {
            "supported": (
                "The frozen real certificate contains a material, exactly removable "
                "rank-five representation gauge under the declared loss-safe geometry, "
                "and all registered physical-zone spread queries are tangent to it."
            ),
            "not_supported": (
                "This audit does not establish predictive superiority, operational "
                "benefit, or worldwide priority for AF-CDMO. Those require a frozen "
                "chronological benchmark against matched controls."
            ),
        },
    }
    _atomic_json(output, report)
    print(f"[OK] wrote {output}")
    print(
        f"[OK] rank={rank} tangent_dim={len(zone_names) - rank} rows={rows} "
        f"pairs={len(queries)} mean_norm_removed={np.mean(1.0 - ratios):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
