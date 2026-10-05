"""Audit a real regional slack-reference row without changing frozen models.

Explicit authorized certificate inputs only. This tests geometric row
equivalence under separately declared synchronous balances and fails closed
for price queries that do not belong to that tangent space. It does not claim
whole-certificate equivalence, predictive performance or operational deployment.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NEWS_URL = "https://nordic-rcc.net/wp-content/uploads/2025/06/News-update-5-June-2025.pdf"
BALANCE_URL = "https://nordic-rcc.net/flow-based/qa-epr/"
PHYSICAL_ZONES = ("DK1", "DK2", "FI", "NO1", "NO2", "NO3", "NO5", "SE1", "SE3", "SE4")
ZERO_LOSS_PAIRS = (("DK1_SB", "DK2_SB"), ("FI_FS", "SE3_FS"),
                   ("DK1_KS", "SE3_KS"), ("SE3_SWL", "SE4_SWL"))
AUXILIARY_FIELDS = ("flowFb", "fmax", "fref", "fall")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def declared_geometries(zones: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(zones) != len(set(zones)):
        raise ValueError("Zone coordinates must be unique")
    required = set(PHYSICAL_ZONES) | {zone for pair in ZERO_LOSS_PAIRS for zone in pair}
    if not required.issubset(zones):
        raise ValueError("Required physical and zero-loss endpoint coordinates are missing")
    equality = np.zeros((5, len(zones)))
    equality[0] = 1.0
    for row, pair in enumerate(ZERO_LOSS_PAIRS, 1):
        for zone in pair:
            equality[row, zones.index(zone)] = 1.0
    # Primary Q&A defines Jutland as every DK1* coordinate, all others Nordic.
    regional = np.asarray([float(not zone.startswith("DK1")) for zone in zones])
    if regional.sum() == 0 or regional.sum() == len(zones):
        raise ValueError("Both synchronous blocks must be present")
    augmented = np.vstack((equality, regional))
    if np.linalg.matrix_rank(equality) != 5 or np.linalg.matrix_rank(augmented) != 6:
        raise ValueError("Declared baseline/augmented equality ranks must be 5/6")
    return equality, augmented, regional


def projector(equality: np.ndarray) -> np.ndarray:
    matrix = np.asarray(equality, dtype=float)
    result = np.eye(matrix.shape[1]) - matrix.T @ np.linalg.pinv(matrix @ matrix.T) @ matrix
    return 0.5 * (result + result.T)


def require_tangent_query(query: np.ndarray, equality: np.ndarray, tolerance: float = 1e-10) -> None:
    if query.shape != (equality.shape[1],) or not np.all(np.isfinite(query)):
        raise ValueError("Query must have finite declared coordinates")
    residual = float(np.max(np.abs(equality @ query)))
    if residual > tolerance:
        raise ValueError("Price query is outside the declared tangent space; relative area-price/equality-dual information is required")


def geometric_row(normal: np.ndarray, ram: float, equality: np.ndarray) -> dict:
    represented = np.asarray(normal, dtype=float) @ projector(equality)
    length = float(np.linalg.norm(represented))
    if length <= 1e-12 or not np.all(np.isfinite(represented)) or not np.isfinite(ram):
        raise ValueError("Nonzero finite projected normal and finite RHS are required")
    # Both regional balances are declared homogeneous; x0=0 is feasible.
    return {"projected_normal": represented, "norm": length, "direction": represented / length,
            "normalized_rhs": float(ram) / length}


def row_comparison(before: dict, after: dict, zones: list[str], equality: np.ndarray) -> dict:
    normal = [np.asarray([float(row["unit_ptdf_" + zone]) for zone in zones]) * float(row["ptdf_l2_norm"])
              for row in (before, after)]
    geometric = [geometric_row(a, float(row["ram"]), equality) for a, row in zip(normal, (before, after))]
    direction_error = float(np.max(np.abs(geometric[0]["direction"] - geometric[1]["direction"])))
    rhs_error = abs(geometric[0]["normalized_rhs"] - geometric[1]["normalized_rhs"])
    projected_error = float(np.max(np.abs(geometric[0]["projected_normal"] - geometric[1]["projected_normal"])))
    auxiliary = {field: {"before": float(before[field]), "after": float(after[field]),
                         "delta": float(after[field]) - float(before[field])} for field in AUXILIARY_FIELDS}
    lambdas = [float(row["shadowPrice"]) for row in (before, after)]
    if not np.all(np.isfinite(lambdas)) or np.any(np.asarray(lambdas) < 0) or any(not np.isfinite(value["before"]) or not np.isfinite(value["after"]) for value in auxiliary.values()):
        raise ValueError("Finite nonnegative dual multipliers and finite auxiliary fields are required")
    masses = [weight * value["norm"] for weight, value in zip(lambdas, geometric)]
    q_equal = all(value["delta"] == 0 for value in auxiliary.values())
    geometric_equal = direction_error < 1e-10 and rhs_error < 1e-8
    return dict(projected_normal_max_abs_difference=projected_error, direction_max_abs_difference=direction_error,
        normalized_rhs_abs_difference=rhs_error, geometric_row_support_equivalent=geometric_equal,
        raw_normal_difference_l2=float(np.linalg.norm(normal[1] - normal[0])),
        projected_norm_before=geometric[0]["norm"], projected_norm_after=geometric[1]["norm"],
        normalized_rhs_before=geometric[0]["normalized_rhs"], normalized_rhs_after=geometric[1]["normalized_rhs"],
        shadow_price_before=lambdas[0], shadow_price_after=lambdas[1], shadow_price_equal=lambdas[0] == lambdas[1],
        projected_mass_before=masses[0], projected_mass_after=masses[1],
        projected_mass_abs_difference=abs(masses[1] - masses[0]), auxiliary_fields=auxiliary,
        retained_auxiliary_mark_equal=q_equal,
        full_single_atom_measure_equal=geometric_equal and q_equal and abs(masses[1] - masses[0]) < 1e-8,
        interpretation="Geometric support equivalence alone does not establish full (u,beta,q,mass) atom equality or whole solved-certificate equivalence.")


def read_endpoint_rows(path: Path, mrid: str, before: str, after: str) -> tuple[dict, dict]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    matches = {before: [], after: []}
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mrId"] == mrid and row["delivery_utc"] in matches:
                matches[row["delivery_utc"]].append(row)
    if any(len(rows) != 1 for rows in matches.values()):
        raise ValueError("Each specified identity/delivery endpoint must have exactly one archived row")
    return matches[before][0], matches[after][0]


def query_audit(zones: list[str], equality: np.ndarray) -> dict:
    accepted, rejected = [], []
    for index, a in enumerate(PHYSICAL_ZONES):
        for b in PHYSICAL_ZONES[index + 1:]:
            query = np.zeros(len(zones))
            query[zones.index(a)], query[zones.index(b)] = 1., -1.
            try:
                require_tangent_query(query, equality)
                accepted.append(a + "-" + b)
            except ValueError:
                rejected.append(a + "-" + b)
    return dict(accepted_count=len(accepted), rejected_count=len(rejected), accepted_pairs=accepted,
                rejected_pairs=rejected, rejection_semantics="Fail closed; no price or monitoring prediction is generated for a rejected query")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--mrid", required=True)
    parser.add_argument("--before", required=True, help="Exact archived delivery_utc string")
    parser.add_argument("--after", required=True, help="Exact archived delivery_utc string")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan == args.run or args.before == args.after:
        parser.error("Choose exactly one of --plan/--run and two different endpoints")
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "protocol.json"
    specification = dict(study_role="Post-confirmation descriptive operational geometry case; no price/performance evaluation",
        prior_exposure="Official event and local same-row PTDF/RAM/auxiliary signature were inspected before this bounded audit; no price/model outcomes; not blinded event discovery",
        inputs={"dual_archive": {"path": str(args.dual_archive.resolve()), "sha256": sha256(args.dual_archive)},
                "geometry_audit": {"path": str(args.geometry_audit.resolve()), "sha256": sha256(args.geometry_audit)}},
        endpoints=dict(mrId=args.mrid, before=args.before, after=args.after),
        geometric_test="Compare projected direction and normalized RHS under original rank5 and declared rank6; classify q/lambda/mass separately",
        geometry_declaration="Homogeneous global balance plus four zero-loss HVDC equalities, augmented by separate Nordic synchronous balance on every non-DK1* coordinate",
        query_policy="Fail closed for price differences outside the declared tangent space",
        sources={"observed_reference_event": NEWS_URL, "separate_balances_and_coordinate_blocks": BALANCE_URL},
        source_code_sha256=sha256(Path(__file__)), no_model_training_or_inference=True,
        no_market_price_or_target_inputs=True, original_frozen_geometry_or_models_modified=False)
    if args.plan:
        if plan_path.exists():
            raise FileExistsError("Preserve the existing protocol; use a new output directory for an amendment")
        specification["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(plan_path, specification)
        print("Bounded operational geometry protocol frozen before endpoint extraction.")
        return
    if not plan_path.is_file():
        raise RuntimeError("Run --plan separately before endpoint extraction")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if {key: value for key, value in plan.items() if key != "frozen_at_utc"} != specification:
        raise RuntimeError("Frozen protocol, inputs or source changed")
    geometry = json.loads(args.geometry_audit.read_text(encoding="utf-8-sig"))["geometry"]
    zones = geometry["zone_order"]
    e5, e6, regional = declared_geometries(zones)
    before, after = read_endpoint_rows(args.dual_archive, args.mrid, args.before, args.after)
    result = dict(study_role=specification["study_role"], protocol_sha256=sha256(plan_path),
        input_checks=specification["inputs"], endpoints=specification["endpoints"], constraint_name=before["cnecName"],
        original_rank5_comparison=row_comparison(before, after, zones, e5),
        declared_rank6_comparison=row_comparison(before, after, zones, e6),
        regional_balance_support=[zone for zone, value in zip(zones, regional) if value],
        regional_reference_rank5_projected_norm=float(np.linalg.norm(regional @ projector(e5))),
        original_rank5_queries=query_audit(zones, e5), declared_rank6_queries=query_audit(zones, e6),
        actual_same_delivery_version_pair=False, whole_solved_certificate_equivalence_claimed=False,
        original_models_operational_protection_claimed=False, price_or_predictive_performance_evaluated=False,
        interpretation="Real archived same-identity geometric row representatives can be equivalent on the declared rank6 feasible space across different deliveries. Differing dual mass or auxiliary marks and changes in other rows prevent a whole-certificate null claim. Original rank5 experiments retain their narrower registered scope.")
    save(args.output / "regional_reference_audit.json", result)
    print(json.dumps(dict(output=str(args.output / "regional_reference_audit.json"),
        rank5_geometry_equivalent=result["original_rank5_comparison"]["geometric_row_support_equivalent"],
        rank6_geometry_equivalent=result["declared_rank6_comparison"]["geometric_row_support_equivalent"],
        full_atom_equal=result["declared_rank6_comparison"]["full_single_atom_measure_equal"],
        queries_accepted=result["declared_rank6_queries"]["accepted_count"],
        queries_rejected=result["declared_rank6_queries"]["rejected_count"])))


if __name__ == "__main__":
    main()
