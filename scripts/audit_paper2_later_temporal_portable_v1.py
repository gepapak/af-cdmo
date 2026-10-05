"""Secondary Paper2 temporal audit from previously opened, frozen QCT outputs.

This uses the consumed Paper3 QCT chronology, never the FS-VANO confirmation.
Neural inference and training are not performed. The cached reconstructed HGB
uses a slightly different fit-tail threshold; that difference is disclosed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

PAPER2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER2))
from scripts import paper2_secondary_temporal_metrics as metrics
from scripts import paper2_portable_audit_utils as portable
CHECKPOINT_ROOT = None
COMPARISON_CHECKPOINT_ROOT = None
HGB_PREDICTIONS = None
OUTPUT = PAPER2 / "results/later_temporal_portable_v1"
PAPER2_REPORT = PAPER2 / "af_cdmo_probabilistic_headroom/quotient_tangent_risk_confirmation_v5.json"
LATER_REPORT = None
ARCHIVE = None
METHODS = ("qct_full", "qct_no_tangent", "uniform_set_transformer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_plan() -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_role": "post-confirmation_secondary_later_temporal_reuse",
        "outcome_exposure": "Point results in the source report were known before this audit plan; this is not a new blinded confirmation.",
        "cohort": "2026-08-20 through 2026-09-30; disjoint later period previously consumed in the QCT study",
        "methods": list(METHODS),
        "primary_reuse_method": "qct_full",
        "tail_threshold": "Use exact Paper2 confirmation threshold; verify cached later targets agree",
        "operating_rule": "Original December-calibrated probability thresholds, plus complete fixed budget grid 5/10/15/20 percent",
        "paired_comparisons": "Full-minus-uniform and full-minus-no-tangent with 7/14/28-day calendar block intervals; reconstructed-HGB point contrast only without cached individual HGB outputs",
        "bootstrap": {"block_days": [7, 14, 28], "replicates": 2000, "seed": 20261003},
        "no_training": True,
        "no_neural_inference": True,
        "no_later_fit_or_calibration_or_selection": True,
        "no_fs_vano_confirmation_access": True,
        "inputs": {
            "paper2_confirmation_report": {"path": str(PAPER2_REPORT), "sha256": sha256(PAPER2_REPORT)},
            "existing_later_report": {"path": str(LATER_REPORT), "sha256": sha256(LATER_REPORT)},
            "existing_later_predictions": {"path": str(ARCHIVE), "sha256": sha256(ARCHIVE)},
        },
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "comparison_checkpoint_root": str(COMPARISON_CHECKPOINT_ROOT) if COMPARISON_CHECKPOINT_ROOT else None,
        "supplied_hgb": {"path": str(HGB_PREDICTIONS), "sha256": sha256(HGB_PREDICTIONS)} if HGB_PREDICTIONS else None,
        "statistics_engine_sha256": sha256(Path(metrics.__file__)),
    }


def audit() -> dict:
    import numpy as np
    import pandas as pd

    plan_path = OUTPUT / "analysis_plan.json"
    if not plan_path.is_file():
        raise RuntimeError("Write the secondary plan with --plan before --run")
    plan = read_json(plan_path)
    current_inputs = {"paper2_confirmation_report": PAPER2_REPORT, "existing_later_report": LATER_REPORT,
                      "existing_later_predictions": ARCHIVE}
    if any(str(path) != plan["inputs"][name]["path"] for name, path in current_inputs.items()):
        raise RuntimeError("CLI input paths differ from the frozen plan")
    if plan["checkpoint_root"] != str(CHECKPOINT_ROOT) or plan["comparison_checkpoint_root"] != (str(COMPARISON_CHECKPOINT_ROOT) if COMPARISON_CHECKPOINT_ROOT else None):
        raise RuntimeError("Checkpoint locations differ from the frozen plan")
    for record in plan["inputs"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Input changed after plan was written")
    if plan["script_sha256"] != sha256(Path(__file__)) or plan["statistics_engine_sha256"] != sha256(Path(metrics.__file__)):
        raise RuntimeError("Adapter/statistics code changed after plan")
    if plan["supplied_hgb"] is not None:
        if HGB_PREDICTIONS is None or sha256(HGB_PREDICTIONS) != plan["supplied_hgb"]["sha256"]:
            raise RuntimeError("Supplied HGB probabilities changed after plan")
    elif HGB_PREDICTIONS is not None:
        raise RuntimeError("Supplied HGB probabilities were not included in the plan")
    original = read_json(PAPER2_REPORT)
    later = read_json(LATER_REPORT)
    if sha256(ARCHIVE) != later["prediction_archive"]["sha256"]:
        raise RuntimeError("Cached archive differs from original later report")
    originals = {(record["method"], int(record["seed"])): record for record in original["checkpoints"]}
    checkpoint_records = []
    for method in METHODS:
        for record in later["checkpoint_records"][method]:
            seed = int(record["seed"])
            source = CHECKPOINT_ROOT / f"{method}__seed{seed}.pt"
            copy = COMPARISON_CHECKPOINT_ROOT / source.name if COMPARISON_CHECKPOINT_ROOT is not None else None
            expected = originals[(method, seed)]["sha256"]
            matched = sha256(source) == record["sha256"] == expected
            if copy is not None:
                matched = matched and sha256(copy) == expected
            if not matched:
                raise RuntimeError(f"Checkpoint identity failed: {method}/{seed}")
            checkpoint_records.append({"method": method, "seed": seed, "sha256": expected, "original_bytes_and_report_identity_verified": matched, "copied_bytes_verified": copy is not None})
    if original["sources"]["confirmation_engine_sha256"] != later["engine"]["frozen_confirmation_engine_sha256"]:
        raise RuntimeError("Confirmation-engine identity failed")
    metrics_digest = sha256(Path(metrics.__file__))
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        timestamps = pd.to_datetime(archive["timestamp_utc"], utc=True)
        magnitude = np.asarray(archive["residual_magnitude_eur_mwh"], dtype=float)
        threshold = float(original["protocol"]["tail_thresholds_eur_mwh"][1])
        target = (magnitude > threshold).astype(float)
        if not np.array_equal(target, archive["target"]):
            raise RuntimeError("Exact Paper2 threshold changes cached later targets")
        probabilities = {method: np.asarray(archive[f"probability__{method}"], dtype=float) for method in METHODS}
    timestamps = portable.validate_predictions(timestamps, target, magnitude, probabilities, threshold)
    if len(timestamps) != later["holdout"]["rows"] or int(target.sum()) != later["holdout"]["tail_events"]:
        raise RuntimeError("Cached cohort summary mismatch")
    methods = {}
    for method, probability in probabilities.items():
        classification = metrics.classification_metrics(target, probability)
        if abs(classification["brier"] - later["method_results"][method]["classification"]["brier"]) > 1e-12:
            raise RuntimeError("Cached classification metric did not reproduce")
        fixed_threshold = original["methods"][method]["decision_at_calibration_80pct_coverage"]["probability_threshold_from_december"]
        if fixed_threshold != later["method_results"][method]["frozen_calibration_threshold_operations"]["0.200"]["probability_threshold"]:
            raise RuntimeError("Original December probability threshold differs")
        methods[method] = {
            "classification": classification,
            "original_december_threshold_operations": metrics.threshold_operational_metrics(target, magnitude, probability, fixed_threshold),
            "fixed_budget_grid": {str(budget): metrics.operational_metrics(target, magnitude, probability, budget) for budget in (0.05, 0.10, 0.15, 0.20)},
        }
    comparisons = {}
    for comparator in ("uniform_set_transformer", "qct_no_tangent"):
        comparisons[comparator] = {
            "brier": metrics.compare_brier(timestamps, target, probabilities["qct_full"], probabilities[comparator], block_days=[7, 14, 28], replicates=2000, seed=20261003, absolute_margin=0.0025, relative_margin=0.05),
            "cost_at_20pct_audit_cost_0_05": metrics.compare_operational_cost(timestamps, target, probabilities["qct_full"], probabilities[comparator], audit_budget=0.20, missed_tail_cost=1.0, audit_cost=0.05, block_days=[7, 14, 28], replicates=2000, seed=20261003),
        }
    hgb_brier = later["primary_comparisons"]["hgb_quotient"]["brier"]["comparator_brier"]
    original_cost = later["primary_comparisons"]["hgb_quotient"]["operational_cost_grid"]["budget=0.200|audit_cost=0.050"]
    flags = methods["qct_full"]["fixed_budget_grid"]["0.2"]["flagged_rows"]
    inferred_missed = original_cost["comparator_mean_cost"] * len(target) - 0.05 * flags
    if abs(inferred_missed - round(inferred_missed)) > 1e-8:
        raise RuntimeError("Reported HGB decision cost does not yield an integer missed-tail count")
    hgb = {
        "brier": hgb_brier,
        "full_minus_reconstructed_hgb_brier_point": methods["qct_full"]["classification"]["brier"] - hgb_brier,
        "full_minus_hgb_paired_interval": None,
        "interval_limitation": "HGB individual probabilities were omitted from the original cached archive; no interval fabricated.",
        "at_20pct_budget": {"flagged_rows": flags, "missed_tail_rows": int(round(inferred_missed)), "tail_recall": 1.0 - round(inferred_missed) / int(target.sum()), "count_recovered_from_reported_cost": True},
        "historical_fit_reconstructed_not_original_serialized_model": True,
        "paper2_fit_tail_threshold_eur_mwh": threshold,
        "later_study_fit_tail_threshold_eur_mwh": later["protocol"]["tail_threshold_eur_mwh"],
        "paper2_hgb_platt": original["control_details"]["hgb_quotient"]["platt"],
        "later_hgb_platt": later["control_details"]["hgb_quotient"]["platt"],
        "caveat": "The later reconstruction used the preregistered double-precision development tail threshold rather than the actual confirmation float32-derived threshold; this changes some historical fit labels and Platt coefficients. Same original neural checkpoints and December neural decision thresholds are verified exactly.",
    }
    report = {
        "status": "complete_secondary_temporal_reuse",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_role": plan["study_role"],
        "outcome_exposure": plan["outcome_exposure"],
        "plan_sha256": sha256(plan_path),
        "inputs": plan["inputs"],
        "script_sha256": sha256(Path(__file__)),
        "statistics_engine_sha256": metrics_digest,
        "checkpoint_identity": checkpoint_records,
        "holdout": later["holdout"],
        "exact_paper2_tail_threshold_eur_mwh": threshold,
        "cached_later_targets_match_exact_paper2_definition": True,
        "original_neural_december_thresholds_match": True,
        "methods": methods,
        "paired_full_comparisons": comparisons,
        "existing_reconstructed_hgb_comparator": hgb,
        "existing_no_tangent_primary_comparisons": {method: value["brier"] for method, value in later["primary_comparisons"].items()},
        "existing_no_tangent_refinement_contract": later["rewrite_audit"]["equivalent_nonuniform_refinement"],
        "interpretation": "Identical frozen Paper2 neural checkpoints retain no demonstrated predictive necessity over the reconstructed invariant HGB in this later regime. Structural invariance and temporal predictive transfer are separate claims.",
        "training_or_neural_inference_performed": False,
        "later_fit_calibration_selection_performed": False,
        "fs_vano_confirmation_accessed": False,
    }
    if HGB_PREDICTIONS is not None:
        hgb_probability = portable.aligned_hgb_csv(HGB_PREDICTIONS, timestamps, target, magnitude)
        report["supplied_original_hgb"] = {
            "input_path": str(HGB_PREDICTIONS), "input_sha256": sha256(HGB_PREDICTIONS),
            "identity_limit": "Alignment is verified; original model identity must be established by the separate historical reproduction audit.",
            "classification": metrics.classification_metrics(target, hgb_probability),
            "fixed_december_operations": metrics.threshold_operational_metrics(target, magnitude, hgb_probability,
                original["methods"]["hgb_quotient"]["decision_at_calibration_80pct_coverage"]["probability_threshold_from_december"]),
            "common_20pct_operations": metrics.operational_metrics(target, magnitude, hgb_probability, 0.20),
            "paired_full_brier": metrics.compare_brier(timestamps, target, probabilities["qct_full"], hgb_probability,
                block_days=[7, 14, 28], replicates=2000, seed=20261003, absolute_margin=0.0025, relative_margin=0.05),
            "paired_full_20pct_cost": metrics.compare_operational_cost(timestamps, target, probabilities["qct_full"], hgb_probability,
                audit_budget=0.20, missed_tail_cost=1.0, audit_cost=0.05, block_days=[7, 14, 28], replicates=2000, seed=20261003),
        }
    save(OUTPUT / "audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    global PAPER2_REPORT, LATER_REPORT, ARCHIVE, OUTPUT, CHECKPOINT_ROOT, COMPARISON_CHECKPOINT_ROOT, HGB_PREDICTIONS
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--later-report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--comparison-checkpoint-root", type=Path)
    parser.add_argument("--hgb-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    PAPER2_REPORT, LATER_REPORT, ARCHIVE, OUTPUT = args.original_report.resolve(), args.later_report.resolve(), args.predictions.resolve(), args.output.resolve()
    CHECKPOINT_ROOT = args.checkpoint_root.resolve()
    COMPARISON_CHECKPOINT_ROOT = args.comparison_checkpoint_root.resolve() if args.comparison_checkpoint_root else None
    HGB_PREDICTIONS = args.hgb_predictions.resolve() if args.hgb_predictions else None
    if args.plan == args.run:
        parser.error("Choose exactly one of --plan and --run")
    if args.plan:
        save(OUTPUT / "analysis_plan.json", make_plan())
        print("Secondary plan written; point-result exposure explicitly disclosed.")
    else:
        report = audit()
        print(json.dumps({"output": str(OUTPUT / "audit.json"), "rows": report["holdout"]["rows"], "methods": {name: value["classification"]["brier"] for name, value in report["methods"].items()}, "older_reconstructed_hgb_brier": report["existing_reconstructed_hgb_comparator"]["brier"], "supplied_original_hgb_brier": report.get("supplied_original_hgb", {}).get("classification", {}).get("brier"), "no_inference": True}))


if __name__ == "__main__":
    main()
