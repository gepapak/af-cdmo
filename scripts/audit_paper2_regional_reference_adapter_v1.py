"""Source-inspired regional-reference rewrite of one authorized certificate.

The rewritten certificate is controlled, not an observed second publication.
No prices, fitting or neural inference. Original Paper2 engines are unchanged.
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


def certificate(path: Path, delivery: str, zones: list[str]) -> dict:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["delivery_utc"] == delivery:
                rows.append(row)
    if not rows or len({row["mrId"] for row in rows}) != len(rows):
        raise ValueError("One nonempty certificate with unique retained row identities is required")
    normals = np.asarray([[float(row["unit_ptdf_" + zone]) * float(row["ptdf_l2_norm"])
                           for zone in zones] for row in rows])
    values = dict(normals=normals, ram=np.asarray([float(row["ram"]) for row in rows]),
        dual=np.asarray([float(row["shadowPrice"]) for row in rows]),
        auxiliary=np.asarray([[float(row[field]) for field in reference.AUXILIARY_FIELDS] for row in rows]),
        mrids=[row["mrId"] for row in rows])
    if any(not np.all(np.isfinite(values[key])) for key in ("normals", "ram", "dual", "auxiliary")):
        raise ValueError("Certificate numerical fields must be finite")
    if np.any(values["dual"] <= 0):
        raise ValueError("This archived positive-dual subset requires strictly positive multipliers")
    return values


def canonical(values: dict, equality: np.ndarray) -> dict:
    represented = values["normals"] @ reference.projector(equality)
    length = np.linalg.norm(represented, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("Positive-dual projected constant row: fail closed before normalization")
    return dict(directions=represented / length[:, None], beta=values["ram"] / length,
        mass=values["dual"] * length, auxiliary=values["auxiliary"],
        analytic_field=-(values["dual"][:, None] * represented).sum(axis=0),
        projected_norm=length)


def admissible_spreads(field: np.ndarray, zones: list[str], equality: np.ndarray) -> dict:
    result = {}
    for a_index, a in enumerate(reference.PHYSICAL_ZONES):
        for b in reference.PHYSICAL_ZONES[a_index + 1:]:
            query = np.zeros(len(zones))
            query[zones.index(a)], query[zones.index(b)] = 1., -1.
            try:
                reference.require_tangent_query(query, equality)
            except ValueError:
                continue
            result[a + "-" + b] = float(field @ query)
    return result


def compare(values: dict, rewritten: dict, zones: list[str], equality: np.ndarray,
            common_query_equality: np.ndarray) -> dict:
    before, after = canonical(values, equality), canonical(rewritten, equality)
    spreads = [admissible_spreads(item["analytic_field"], zones, common_query_equality)
               for item in (before, after)]
    changes = {key: abs(spreads[1][key] - spreads[0][key]) for key in spreads[0]}
    direction = float(np.max(np.abs(before["directions"] - after["directions"])))
    beta = float(np.max(np.abs(before["beta"] - after["beta"])))
    mass = float(np.max(np.abs(before["mass"] - after["mass"])))
    q_equal = bool(np.array_equal(before["auxiliary"], after["auxiliary"]))
    return dict(projected_direction_max_abs_difference=direction,
        normalized_rhs_max_abs_difference=beta, projected_mass_max_abs_difference=mass,
        total_mass_before=float(before["mass"].sum()), total_mass_after=float(after["mass"].sum()),
        retained_auxiliary_marks_exactly_equal=q_equal,
        corresponding_fine_measure_atoms_equal=direction <= 1e-10 and beta <= 1e-8 and mass <= 1e-8 and q_equal,
        analytic_field_max_abs_difference=float(np.max(np.abs(before["analytic_field"] - after["analytic_field"]))),
        admissible_pair_count=len(changes), analytic_spread_max_abs_difference_eur_mwh=max(changes.values()),
        admissible_spread_abs_differences_eur_mwh=changes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan == args.run:
        parser.error("Choose exactly one of --plan/--run")
    protocol = dict(role="Post-confirmation controlled adapter contract on an observed event-adjacent certificate",
        prior_exposure="Official May28reference event, archived adjacent-row signature, and exhaustive9-row census were examined; no prices/model outcomes",
        inputs={"dual_archive": {"path": str(args.dual_archive.resolve()), "sha256": reference.sha256(args.dual_archive)},
                "geometry_audit": {"path": str(args.geometry_audit.resolve()), "sha256": reference.sha256(args.geometry_audit)}},
        delivery=args.delivery,
        rewrite="For every stored positive-dual row, a_new = a - a(SE2)*v_N; v_N=1 on every non-DK1* coordinate. RAM, lambda and every auxiliary field held fixed.",
        geometry="Compare original rank5 quotient with separately declared homogeneous rank6 geometry adding Nordic synchronous balance; x0=0",
        query_policy="Only36 within-Nordic physical differences are evaluated; all9 DK1-cross-region queries fail closed and have no output",
        claims="Controlled re-expression, not observed whole-certificate version equivalence, market accuracy or original rank5 operational coverage",
        source_urls=[reference.NEWS_URL, reference.BALANCE_URL],
        source_code_sha256=reference.sha256(Path(__file__)), helper_sha256=reference.sha256(Path(reference.__file__)),
        no_market_prices_targets_training_or_neural_inference=True)
    protocol_path = args.output / "protocol.json"
    if args.plan:
        if protocol_path.exists():
            raise FileExistsError("Use a new directory for protocol amendments")
        reference.save(protocol_path, dict(protocol, frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        print("Adapter protocol frozen before certificate extraction.")
        return
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if {key: value for key, value in frozen.items() if key != "frozen_at_utc"} != protocol:
        raise RuntimeError("Source, input or protocol changed after freeze")
    zones = json.loads(args.geometry_audit.read_text(encoding="utf-8-sig"))["geometry"]["zone_order"]
    e5, e6, regional = reference.declared_geometries(zones)
    values = certificate(args.dual_archive, args.delivery, zones)
    shift = values["normals"][:, zones.index("SE2")]
    rewritten = dict(values, normals=values["normals"] - shift[:, None] * regional[None, :])
    result = dict(protocol_sha256=reference.sha256(protocol_path), row_count=len(values["dual"]),
        original_rank5=compare(values, rewritten, zones, e5, e6),
        declared_rank6=compare(values, rewritten, zones, e6, e6),
        declared_rank6_query_policy=reference.query_audit(zones, e6),
        actual_second_published_certificate=False, price_accuracy_or_calibration_evaluated=False,
        original_frozen_models_or_geometry_modified=False)
    reference.save(args.output / "regional_reference_adapter.json", result)
    print(json.dumps(dict(rows=result["row_count"], rank5=result["original_rank5"],
        rank6=result["declared_rank6"], rejected_pairs=result["declared_rank6_query_policy"]["rejected_pairs"])))


if __name__ == "__main__":
    main()
