"""Audit the conservative Core rank-1 AF-CDMO geometry on real JAO rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from af_cdmo.core import (
    AffineFeasibleGeometry,
    canonicalize_certificate,
    equality_gauge_scale_rewrite,
    measure_restricted_lagrangian,
    restricted_lagrangian,
    split_certificate,
)

try:
    from scripts.core_external_replication import (
        CORE_PHYSICAL_ZONES,
        atomic_json,
        global_balance_geometry,
        sha256_file,
    )
except ModuleNotFoundError:
    from core_external_replication import (
        CORE_PHYSICAL_ZONES,
        atomic_json,
        global_balance_geometry,
        sha256_file,
    )


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _resolve_from_manifest(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest.get("artifacts", {}).get("dual", {})
    path = Path(str(artifact.get("path", "")))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Core dual artifact from manifest is missing: {path}")
    expected = artifact.get("sha256")
    if expected and sha256_file(path) != expected:
        raise RuntimeError("Core dual artifact hash does not match its manifest")
    return path, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_groups", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=2.0e-9)
    args = parser.parse_args()
    if args.max_groups < 1 or args.tolerance <= 0.0:
        raise ValueError("Invalid group count or tolerance")

    manifest_path = args.manifest.resolve()
    dual_path, manifest = _resolve_from_manifest(manifest_path)
    frame = pd.read_csv(dual_path)
    frame["delivery_utc"] = pd.to_datetime(
        frame["delivery_utc"], utc=True, errors="raise", format="mixed"
    )
    frame["publication_utc"] = pd.to_datetime(
        frame["publication_utc"], utc=True, errors="raise", format="mixed"
    )
    unit_columns = [column for column in frame if column.startswith("unit_hub_")]
    if len(unit_columns) < len(CORE_PHYSICAL_ZONES):
        raise RuntimeError("Canonical Core table has too few hub coordinates")
    hub_names = tuple(column.removeprefix("unit_hub_") for column in unit_columns)
    if tuple(hub_names[: len(CORE_PHYSICAL_ZONES)]) != CORE_PHYSICAL_ZONES:
        raise RuntimeError("Physical Core hubs are not in the registered coordinate order")

    equality, expected_tangent = global_balance_geometry(hub_names)
    geometry = AffineFeasibleGeometry.from_equalities(equality, np.zeros(1))
    checks: dict[str, float | int | bool] = {
        "ambient_dimension": geometry.ambient_dimension,
        "geometry_rank": geometry.rank,
        "tangent_dimension": geometry.tangent_dimension,
        "projector_idempotence_error": _max_abs(
            geometry.tangent_projector @ geometry.tangent_projector
            - geometry.tangent_projector
        ),
        "projector_symmetry_error": _max_abs(
            geometry.tangent_projector - geometry.tangent_projector.T
        ),
        "projector_closed_form_error": _max_abs(
            geometry.tangent_projector - expected_tangent
        ),
        "equality_projector_error": _max_abs(
            equality @ geometry.tangent_projector
        ),
    }
    physical_indices = [hub_names.index(zone) for zone in CORE_PHYSICAL_ZONES]
    query_errors = []
    for left_position, left in enumerate(physical_indices):
        for right in physical_indices[left_position + 1 :]:
            query = np.zeros(len(hub_names), dtype=np.float64)
            query[left] = 1.0
            query[right] = -1.0
            query_errors.append(_max_abs(equality @ query))
    checks["physical_pair_query_count"] = len(query_errors)
    checks["physical_pair_tangency_error"] = max(query_errors, default=0.0)

    rng = np.random.default_rng(20261003)
    row_direction_errors = []
    row_rhs_errors = []
    row_mass_errors = []
    dual_current_errors = []
    total_mass_errors = []
    split_signature_errors = []
    restricted_errors = []
    slack_errors = []
    groups_checked = 0
    for _, group in frame.groupby("delivery_utc", sort=True):
        if groups_checked >= args.max_groups:
            break
        normals = (
            group[unit_columns].to_numpy(dtype=np.float64)
            * group["ptdf_l2_norm"].to_numpy(dtype=np.float64)[:, None]
        )
        rhs = group["ram"].to_numpy(dtype=np.float64)
        multipliers = group["shadowPrice"].to_numpy(dtype=np.float64)
        tail = group[["ramMcp", "fmax", "fref", "f0all"]].fillna(0.0).to_numpy(
            dtype=np.float64
        )
        original = canonicalize_certificate(
            geometry, normals, rhs, multipliers, tail
        )

        scales = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=len(normals)))
        gauge = rng.normal(0.0, 2.0, size=(len(normals), 1))
        rewritten_normals, rewritten_rhs, rewritten_dual = equality_gauge_scale_rewrite(
            geometry,
            normals,
            rhs,
            multipliers,
            scales=scales,
            gauge_coefficients=gauge,
        )
        rewritten = canonicalize_certificate(
            geometry, rewritten_normals, rewritten_rhs, rewritten_dual, tail
        )
        row_direction_errors.append(_max_abs(original.directions - rewritten.directions))
        row_rhs_errors.append(_max_abs(original.normalized_rhs - rewritten.normalized_rhs))
        row_mass_errors.append(_max_abs(original.masses - rewritten.masses))
        dual_current_errors.append(_max_abs(original.dual_current - rewritten.dual_current))
        total_mass_errors.append(
            abs(original.total_mass - rewritten.total_mass)
            / max(1.0, abs(original.total_mass), abs(rewritten.total_mass))
        )

        fractions = rng.uniform(0.1, 0.9, size=len(normals))
        split_normals, split_rhs, split_dual, split_tail = split_certificate(
            normals, rhs, multipliers, tail, fractions=fractions
        )
        split = canonicalize_certificate(
            geometry, split_normals, split_rhs, split_dual, split_tail
        )
        split_signature_errors.append(
            _max_abs(original.moment_signature() - split.moment_signature())
        )

        displacement = geometry.project(rng.normal(size=len(hub_names)))
        restricted_raw = restricted_lagrangian(
            geometry, normals, rhs, multipliers, displacement
        )
        restricted_rewritten = restricted_lagrangian(
            geometry,
            rewritten_normals,
            rewritten_rhs,
            rewritten_dual,
            displacement,
        )
        restricted_measure = measure_restricted_lagrangian(
            geometry, original, displacement
        )
        restricted_scale = max(
            1.0,
            abs(restricted_raw),
            abs(restricted_rewritten),
            abs(restricted_measure),
        )
        restricted_errors.append(
            max(
                abs(restricted_raw - restricted_rewritten),
                abs(restricted_raw - restricted_measure),
            )
            / restricted_scale
        )

        slack = int(rng.integers(0, len(hub_names)))
        slack_normals = normals - normals[:, [slack]]
        slack_measure = canonicalize_certificate(
            geometry, slack_normals, rhs, multipliers, tail
        )
        slack_errors.append(
            max(
                _max_abs(original.dual_current - slack_measure.dual_current),
                abs(original.total_mass - slack_measure.total_mass)
                / max(1.0, abs(original.total_mass), abs(slack_measure.total_mass)),
            )
        )
        groups_checked += 1

    if groups_checked == 0:
        raise RuntimeError("No Core delivery groups were available for geometry audit")
    checks.update(
        {
            "delivery_groups_checked": groups_checked,
            "scale_gauge_direction_error": max(row_direction_errors),
            "scale_gauge_rhs_error": max(row_rhs_errors),
            "scale_gauge_mass_error": max(row_mass_errors),
            "scale_gauge_dual_current_error": max(dual_current_errors),
            "scale_gauge_total_mass_error": max(total_mass_errors),
            "split_moment_signature_error": max(split_signature_errors),
            "restricted_lagrangian_error": max(restricted_errors),
            "global_slack_dual_measure_error": max(slack_errors),
            "causal_publication_share": float(
                np.mean(frame["publication_utc"] <= frame["delivery_utc"])
            ),
        }
    )
    exact_keys = (
        "projector_idempotence_error",
        "projector_symmetry_error",
        "projector_closed_form_error",
        "equality_projector_error",
        "physical_pair_tangency_error",
        "scale_gauge_direction_error",
        "scale_gauge_rhs_error",
        "scale_gauge_mass_error",
        "scale_gauge_dual_current_error",
        "scale_gauge_total_mass_error",
        "split_moment_signature_error",
        "restricted_lagrangian_error",
        "global_slack_dual_measure_error",
    )
    failures = {key: checks[key] for key in exact_keys if float(checks[key]) > args.tolerance}
    if geometry.rank != 1:
        failures["geometry_rank"] = geometry.rank
    if checks["causal_publication_share"] != 1.0:
        failures["causal_publication_share"] = checks["causal_publication_share"]

    report = {
        "protocol": "core_ccr_rank1_geometry_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_protocol": manifest.get("protocol"),
        "dual_artifact": str(dual_path),
        "dual_artifact_sha256": sha256_file(dual_path),
        "registered_geometry": {
            "equalities": ["global_net_position_balance"],
            "rank": 1,
            "claim_boundary": (
                "This audit does not encode or claim recovery of Core's additional "
                "technical or virtual-hub equalities."
            ),
        },
        "tolerance": args.tolerance,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    output = args.output.resolve()
    atomic_json(output, report)
    if failures:
        raise RuntimeError(f"Core geometry audit failed: {failures}")
    print(f"[OK] Core rank-1 geometry audit: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
