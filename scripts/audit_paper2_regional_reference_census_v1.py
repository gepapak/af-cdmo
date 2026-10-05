"""Exhaustive common-row census at explicitly selected regional-reference times.

All stored common positive-dual identities are retained, including rows whose
projected normal cannot be normalized. No prices, fitting or neural inference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_paper2_regional_reference_v1 as reference


def load_groups(path: Path, endpoints: tuple[str, str]) -> dict[str, dict[str, dict]]:
    groups = {endpoint: {} for endpoint in endpoints}
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            endpoint = row["delivery_utc"]
            if endpoint not in groups:
                continue
            if row["mrId"] in groups[endpoint]:
                raise ValueError("Repeated delivery/mRID in archive; do not silently deduplicate")
            groups[endpoint][row["mrId"]] = row
    return groups


def classify(before: dict, after: dict, zones: list[str], equality: np.ndarray) -> dict:
    normals = [np.asarray([float(row["unit_ptdf_" + zone]) for zone in zones]) * float(row["ptdf_l2_norm"])
               for row in (before, after)]
    lengths = [float(np.linalg.norm(normal @ reference.projector(equality))) for normal in normals]
    if min(lengths) <= 1e-12:
        return dict(status="fail_closed_projected_zero_normal", geometric_row_support_equivalent=None,
            projected_norm_before=lengths[0], projected_norm_after=lengths[1],
            normalized_direction_or_rhs_evaluated=False,
            reason="Positive-dual constant constraints cannot enter the normalized retained-row quotient")
    return dict(status="normalized_rows_evaluated", **reference.row_comparison(before, after, zones, equality))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan == args.run or args.before == args.after:
        parser.error("Choose one of --plan/--run and two different endpoints")
    protocol = dict(role="Post-confirmation exhaustive descriptive geometry census",
        prior_exposure="May28regional PTDFsignature and oneDK2_SV_IMP geometric pair already examined; no price/model outcomes",
        criterion="Every mRID stored exactly once in both explicit delivery endpoints; no outcome or normal-size selection",
        target_population="Common positive-dual rows in the archived subset; zero-shadow/unrecorded constraints are not available",
        endpoints=dict(before=args.before, after=args.after),
        inputs={"dual_archive": {"path": str(args.dual_archive.resolve()), "sha256": reference.sha256(args.dual_archive)},
                "geometry_audit": {"path": str(args.geometry_audit.resolve()), "sha256": reference.sha256(args.geometry_audit)}},
        scripts={"census": reference.sha256(Path(__file__)), "geometry_helper": reference.sha256(Path(reference.__file__))},
        tolerances=dict(projected_zero_norm=1e-12, direction_max_abs=1e-10, normalized_rhs_abs=1e-8),
        zero_normal_policy="Report and fail closed; never normalize or label undefined row supports equal",
        no_model_or_price_outcome_inputs=True, original_geometry_or_models_modified=False)
    plan_path = args.output / "protocol.json"
    if args.plan:
        if plan_path.exists():
            raise FileExistsError("Preserve existing protocol; use new output for an amendment")
        protocol["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        reference.save(plan_path, protocol)
        print("Exhaustive common-row census protocol frozen before endpoint extraction.")
        return
    if not plan_path.is_file():
        raise RuntimeError("Run --plan separately before endpoint extraction")
    stored = json.loads(plan_path.read_text(encoding="utf-8"))
    if {key: value for key, value in stored.items() if key != "frozen_at_utc"} != protocol:
        raise RuntimeError("Inputs, scripts or frozen census specification changed")
    zones = json.loads(args.geometry_audit.read_text(encoding="utf-8-sig"))["geometry"]["zone_order"]
    e5, e6, regional = reference.declared_geometries(zones)
    groups = load_groups(args.dual_archive, (args.before, args.after))
    a, b = groups[args.before], groups[args.after]
    common = sorted(set(a) & set(b))
    records = []
    for mrid in common:
        normals = [np.asarray([float(row["unit_ptdf_" + zone]) for zone in zones]) * float(row["ptdf_l2_norm"]) for row in (a[mrid], b[mrid])]
        delta = normals[1] - normals[0]
        coefficient = float(delta @ regional / (regional @ regional))
        records.append(dict(mrId=mrid, cnec_name_before=a[mrid]["cnecName"], cnec_name_after=b[mrid]["cnecName"],
            raw_delta_l2=float(np.linalg.norm(delta)), regional_shift_coefficient=coefficient,
            residual_beyond_regional_shift_l2=float(np.linalg.norm(delta - coefficient * regional)),
            original_rank5=classify(a[mrid], b[mrid], zones, e5),
            declared_rank6=classify(a[mrid], b[mrid], zones, e6)))
    summary = dict(before_rows=len(a), after_rows=len(b), common_rows=len(common),
        before_only=len(set(a) - set(b)), after_only=len(set(b) - set(a)),
        raw_normal_changed_rows=sum(record["raw_delta_l2"] > 1e-10 for record in records),
        rank5_geometric_equal_rows=sum(record["original_rank5"]["geometric_row_support_equivalent"] is True for record in records),
        rank6_geometric_equal_rows=sum(record["declared_rank6"]["geometric_row_support_equivalent"] is True for record in records),
        rank6_geometric_equal_with_changed_raw_normal_rows=sum(record["declared_rank6"]["geometric_row_support_equivalent"] is True and record["raw_delta_l2"] > 1e-10 for record in records),
        rank5_constant_fail_closed_rows=sum(record["original_rank5"]["status"] == "fail_closed_projected_zero_normal" for record in records),
        rank6_constant_fail_closed_rows=sum(record["declared_rank6"]["status"] == "fail_closed_projected_zero_normal" for record in records),
        rank6_full_single_atom_equal_rows=sum(record["declared_rank6"].get("full_single_atom_measure_equal") is True for record in records))
    report = dict(protocol_sha256=reference.sha256(plan_path), inputs=protocol["inputs"], summary=summary,
        common_row_records=records, declared_rank6_query_contract=reference.query_audit(zones, e6),
        same_delivery_revision_pair=False, whole_certificate_null_equivalence_claimed=False,
        interpretation="All stored commonrows are retained. Geometry-onlymatches across differentdeliveries do not imply unchangeddualmeasure, marks,pricesorotherconstraints. Originalrank5modelassurance remains unchanged.")
    reference.save(args.output / "regional_reference_census.json", report)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
