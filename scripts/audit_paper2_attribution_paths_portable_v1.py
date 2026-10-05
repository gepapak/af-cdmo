"""Portable two-path analytic-attribution integrity replay.

Explicit authorized input paths, frozen --plan then --run, and all original
confirmation times/pairs/tags. No prices, fitted models, scores or network.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import pandas as pd
try:
    from . import paper2_attribution_paths_metrics as metrics
except ImportError:
    import paper2_attribution_paths_metrics as metrics


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_reference(candidate: dict, reference: dict) -> None:
    """Exact comparison of all reproduced scalars, counts, margins and quantiles."""
    for key, value in candidate.items():
        if isinstance(value, dict):
            verify_reference(value, reference[key])
        elif value != reference[key]:
            raise RuntimeError(f"Frozen attribution endpoint differs: {key}: {value!r} versus {reference[key]!r}")


def self_test() -> None:
    rng = np.random.default_rng(73)
    unit = rng.normal(size=(7, 6))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    norm = np.asarray([.5, 1, 2, 5, 10, 15, 20.])
    shadow = np.asarray([1, 4, 2, .5, 7, 8, 3.])
    mass = norm * shadow
    projector = np.eye(6) - np.ones((6, 6)) / 6
    indices = np.asarray([0, 2, 4])
    route_a, route_b = metrics.row_route_fields(unit, norm, shadow, mass, projector, indices)
    assert np.allclose(route_a, route_b, rtol=0, atol=1e-13)
    assert np.max(np.abs(route_a.sum(axis=1))) < 1e-13
    scale = np.asarray([2, .5, 4, .25, 8, .125, 16.])
    changed_a, changed_b = metrics.row_route_fields(unit, norm * scale, shadow / scale, mass, projector, indices)
    assert np.allclose(changed_a, route_a, rtol=0, atol=1e-13)
    assert np.array_equal(changed_b, route_b)
    score_a = np.asarray([[[3., 1.], [1., 1.], [0., 0.], [1e-15, 0.]]])
    score_b = np.asarray([[[2.9, 1.1], [1., 1. + 1e-12], [0., 1e-30], [0., 1e-15]]])
    context = metrics.routing_arrays(score_a, score_b)
    context.update(tags=["A", "B"], times=["synthetic"])
    detail = metrics.descriptive_detail(context)
    assert detail["routing_flips"] == 2
    assert detail["individual_margin_certified_nonzero"] == 1
    assert detail["individual_margin_certified_flips"] == 0
    assert detail["exact_all_zero_route_a"] == 1 and detail["exact_all_zero_route_b"] == 0
    assert detail["exact_all_zero_screen_membership_changes"] == 1
    assert not detail["checks"]["exact_zero_screen_identical"]
    assert context["selected_a"][0, 1] == 0, "Exact tie uses first lexicographically ordered tag"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    for name in ("dual-archive", "geometry-audit", "timestamp-csv", "frozen-risk-report", "output", "reference-primary", "reference-detail"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Normalization/unit-scale, centering, exact tie, near-tie margin, and all-zero-screen tests passed.")
        return
    required = ("dual_archive", "geometry_audit", "timestamp_csv", "frozen_risk_report", "output")
    if any(getattr(args, name) is None for name in required) or not (args.plan or args.run):
        parser.error("All authorized input paths, --output and either --plan or --run are required")
    if (args.reference_primary is None) != (args.reference_detail is None):
        parser.error("Provide both reference reports or neither")
    paths = {name: getattr(args, name).resolve() for name in required[:-1]}
    paths.update(driver=Path(__file__).resolve(), numerical_helper=Path(metrics.__file__).resolve())
    if args.reference_primary is not None:
        paths.update(reference_primary=args.reference_primary.resolve(), reference_detail=args.reference_detail.resolve())
    sources = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    frozen_report = read_json(args.frozen_risk_report)
    if sources["dual_archive"]["sha256"] != frozen_report["sources"]["dual_archive_sha256"]:
        raise RuntimeError("Archive does not match frozen risk-source bytes")
    if sources["geometry_audit"]["sha256"] != frozen_report["sources"]["geometry_audit_sha256"]:
        raise RuntimeError("Geometry does not match frozen risk-source bytes")
    if frozen_report["rows"]["confirmation"] != 14482:
        raise RuntimeError("This exact replay requires the original full confirmation cohort")
    plan = dict(study="Portable two-path analytic-attribution replay v1",
        status="reproduction_of_already_examined_post_confirmation_integrity_endpoints",
        prior_exposure="Original primary and descriptive outcomes are known. This portable replay creates no new confirmation or preregistration.",
        sources=sources, cohort="all14482originalconfirmationtimestamps, all45physicalpairs, allretainedpositivearchive rows, allobserved provenance tags",
        chronology={key: frozen_report["protocol"][key] for key in ("confirmation_start", "confirmation_end_exclusive")},
        geometry="Original declared rank5 global balance and four loss-free HVDC endpoint equalities; not a regional rank6 adapter",
        publication_proxy="publication_utc <= delivery_utc - 60minutes; proxy is not definitive first-publication evidence",
        archive_chunk_rows=20000, routes=["shadowPrice*(unit_ptdf*stored norm), projected and centered", "stored dual_mass*unit_ptdf, projected and centered"],
        policy="All45pairs/allrecordedtags; top absolute signed-component tag; lexicographic ties; exact route-A nonzero screen; no chosen tolerance",
        branches="Original18primaryendpoint values preserved; full previously executed zero-screen/margin/flip-detail branch reported separately",
        limits=["Both routes share one acquired archive; no independent source-version pair or acquisition validation",
                "TSO tags are provenance labels, not economic causation or inferred ownership",
                "No prices, outcome labels, NN scores, model fitting, inference or API acquisition",
                "No deployed analyst benefit, welfare, outage frequency or neural-added-value conclusion",
                "Allprovider records, timestamp-level fields and privately generated reports remain private"],
        software={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__})
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "analysis_plan.json"
    if args.plan:
        if plan_path.exists():
            raise RuntimeError("Replay plan exists; preserve it before any amendment")
        atomic_json(plan_path, {**plan, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Portable attribution replay frozen: {plan_path}")
        return
    if not plan_path.exists() or {key: value for key, value in read_json(plan_path).items() if key != "frozen_at_utc"} != plan:
        raise RuntimeError("Missing plan, source change or silently amended replay procedure")
    frame = pd.read_csv(args.timestamp_csv, usecols=["timestamp_utc"])
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc, utc=True))
    if (times.hasnans or len(times) != 14482 or not times.is_unique or not times.is_monotonic_increasing or
        times.min() < pd.Timestamp(plan["chronology"]["confirmation_start"]) or
        times.max() >= pd.Timestamp(plan["chronology"]["confirmation_end_exclusive"])):
        raise ValueError("Original chronological native-quarter-hour confirmation timestamps required")
    if np.any(times.minute % 15) or np.any(times.second) or np.any(times.microsecond):
        raise ValueError("Native confirmation quarter-hour grid required; no hourly expansion")
    if args.reference_primary is not None:
        source_reference = read_json(args.reference_primary)["source_inventory"]
        for name, original in (("dual_archive", "dual_archive"), ("geometry_audit", "geometry"), ("timestamp_csv", "confirmation_timestamps")):
            if sources[name]["sha256"] != source_reference[original]["sha256"]:
                raise RuntimeError(f"Exact reproduction input differs from original attribution source: {name}")
    primary, context = metrics.compute(args.dual_archive, args.geometry_audit, args.timestamp_csv)
    detail = metrics.descriptive_detail(context)
    verification = {"original_reference_reports_supplied": args.reference_primary is not None}
    if args.reference_primary is not None:
        verify_reference(primary, read_json(args.reference_primary))
        verify_reference(detail, read_json(args.reference_detail))
        verification.update(all18_primary_endpoints_exact=True, entire_descriptive_detail_branch_exact=True)
    if any(sha256(path) != sources[name]["sha256"] for name, path in paths.items()):
        raise RuntimeError("An input or source changed during the portable replay")
    shared = dict(analysis_plan_sha256=sha256(plan_path), source_inventory=sources, source_hashes_unchanged=True,
                  limits=plan["limits"], verification=verification)
    atomic_json(args.output / "attribution_path_audit.json", dict(status="complete_portable_original_primary_attribution_replay", **shared, **primary))
    atomic_json(args.output / "routing_detail_audit.json", dict(status="complete_portable_separate_descriptive_detail_replay", **shared, **detail))
    print(json.dumps(dict(status="complete_portable_two_path_attribution_replay", verification=verification,
        cohort_timestamps=primary["cohort_timestamps"], selected_rows=primary["selected_positive_mass_rows"],
        primary_routing_flips=primary["top_absolute_contributor_tag_routing_flips"],
        exact_zero_screen_changes=detail["exact_all_zero_screen_membership_changes"]), indent=2))


if __name__ == "__main__":
    main()
