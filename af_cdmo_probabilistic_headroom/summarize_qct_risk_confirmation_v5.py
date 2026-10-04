"""Create complete, mechanically derived QCT-Risk confirmation tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "af_cdmo_probabilistic_headroom"
    / "quotient_tangent_risk_confirmation_v5.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _value(mapping: dict[str, Any] | None, key: str) -> Any:
    if not mapping:
        return None
    return mapping.get(key)


def _method_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, record in report["methods"].items():
        classification = record["classification"]
        coverage = record.get("risk_coverage", {}).get("operating_points", {}).get("0.80", {})
        decision = record.get("decision_at_calibration_80pct_coverage", {})
        tangent = record.get("tangent_residual", {})
        rows.append(
            {
                "method": method,
                "prevalence": classification.get("prevalence"),
                "brier": classification.get("brier"),
                "log_loss": classification.get("log_loss"),
                "roc_auc": classification.get("roc_auc"),
                "average_precision": classification.get("average_precision"),
                "ece_10bin": classification.get("expected_calibration_error_10bin"),
                "calibration_intercept": _value(classification.get("calibration"), "intercept"),
                "calibration_slope": _value(classification.get("calibration"), "slope"),
                "risk_coverage_80_tail_rate": coverage.get("tail_rate"),
                "risk_coverage_80_tail_false_negative_fraction": coverage.get(
                    "tail_false_negative_fraction"
                ),
                "risk_coverage_80_mean_residual_eur_mwh": coverage.get(
                    "mean_residual_magnitude_eur_mwh"
                ),
                "fixed_decision_flag_rate": decision.get("flag_rate"),
                "fixed_decision_tail_recall": decision.get("tail_recall"),
                "fixed_decision_tail_precision": decision.get("tail_precision"),
                "fixed_decision_accepted_tail_rate": decision.get("accepted_tail_rate"),
                "tangent_pairwise_mae_eur_mwh": tangent.get("mean_pairwise_mae_eur_mwh"),
                "tangent_pairwise_rmse_eur_mwh": tangent.get("pairwise_rmse_eur_mwh"),
            }
        )
    return rows


def _paired_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison, record in report.get("paired_brier", {}).items():
        interval = record.get("block_bootstrap_95ci", [None, None])
        rows.append(
            {
                "comparison": comparison,
                "candidate_minus_baseline_brier": record.get(
                    "candidate_minus_baseline_brier"
                ),
                "ci95_low": interval[0],
                "ci95_high": interval[1],
                "bootstrap_probability_of_improvement": record.get(
                    "probability_of_improvement"
                ),
                "block_rows": record.get("block_rows"),
                "bootstrap_replicates": record.get("replicates"),
            }
        )
    return rows


def _seed_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, seeds in report.get("seed_results", {}).items():
        for seed, record in seeds.items():
            classification = record.get("classification", {})
            tangent = record.get("tangent_residual", {})
            training = record.get("training", {})
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "brier": classification.get("brier"),
                    "log_loss": classification.get("log_loss"),
                    "roc_auc": classification.get("roc_auc"),
                    "average_precision": classification.get("average_precision"),
                    "tangent_pairwise_mae_eur_mwh": tangent.get(
                        "mean_pairwise_mae_eur_mwh"
                    ),
                    "best_epoch": training.get("best_epoch"),
                    "best_tuning_brier": training.get("best_tuning_brier"),
                    "best_tuning_average_precision": training.get(
                        "best_tuning_average_precision"
                    ),
                    "parameter_count": training.get("parameter_count"),
                }
            )
    return rows


def _supplemental_contract_review(report: dict[str, Any]) -> dict[str, Any]:
    invariance = report.get("invariance_and_sensitivity", {})
    refinement = invariance.get("held_out_nonuniform_mass_refinement", {}).get(
        "methods", {}
    )
    qct_refinement = refinement.get("qct_full", {})
    tolerance = 1.0e-4
    refinement_pass = bool(qct_refinement) and (
        float(qct_refinement.get("max_abs_probability_drift", float("inf")))
        <= tolerance
        and float(qct_refinement.get("decision_flip_rate", float("inf"))) == 0.0
    )
    registered_pass = bool(
        report.get("decision", {}).get("registered_rewrite_contract_pass", False)
    )
    ram_records = (
        invariance.get("non_equivalent_ram_sensitivity", {})
        .get("methods", {})
        .get("qct_full", {})
    )
    maximum_ram_probability_response = max(
        (
            float(record.get("max_abs_probability_change", 0.0))
            for record in ram_records.values()
        ),
        default=0.0,
    )
    return {
        "registered_rewrite_contract_pass": registered_pass,
        "held_out_nonuniform_refinement_pass": refinement_pass,
        "refinement_probability_tolerance": tolerance,
        "full_registered_structural_contract_pass": bool(
            registered_pass and refinement_pass
        ),
        "maximum_non_equivalent_ram_probability_response": maximum_ram_probability_response,
        "non_equivalent_ram_response_is_diagnostic_not_a_preregistered_threshold": True,
    }


def _certificate_only_reconstruction_review(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize the registered tangent head as certificate-only field correction.

    The analytic-only predictor has a zero learned residual, so its pairwise MAE
    is exactly the mean realized residual magnitude already stored in the frozen
    report. These are point-estimate diagnostics, not an additional selection
    rule or a substitute for the registered Brier hypothesis.
    """
    analytic_mae = float(
        report.get("confirmation_target", {}).get(
            "mean_residual_magnitude_eur_mwh", float("nan")
        )
    )
    methods = report.get("methods", {})

    def tangent_mae(method: str) -> float:
        return float(
            methods.get(method, {})
            .get("tangent_residual", {})
            .get("mean_pairwise_mae_eur_mwh", float("nan"))
        )

    qct_mae = tangent_mae("qct_full")
    hgb_mae = tangent_mae("hgb_quotient")

    def relative_reduction(reference: float, candidate: float) -> float:
        if not np.isfinite(reference) or reference <= 0.0 or not np.isfinite(candidate):
            return float("nan")
        return float((reference - candidate) / reference)

    return {
        "interpretation": "certificate-only centered price-spread reconstruction",
        "analytic_kkt_only_pairwise_mae_eur_mwh": analytic_mae,
        "qct_full_pairwise_mae_eur_mwh": qct_mae,
        "hgb_quotient_pairwise_mae_eur_mwh": hgb_mae,
        "qct_relative_mae_reduction_vs_analytic_only": relative_reduction(
            analytic_mae, qct_mae
        ),
        "qct_relative_mae_reduction_vs_hgb_quotient": relative_reduction(
            hgb_mae, qct_mae
        ),
        "qct_beats_analytic_only_point_estimate": bool(
            np.isfinite(qct_mae) and np.isfinite(analytic_mae) and qct_mae < analytic_mae
        ),
        "qct_beats_hgb_quotient_point_estimate": bool(
            np.isfinite(qct_mae) and np.isfinite(hgb_mae) and qct_mae < hgb_mae
        ),
        "registered_primary_gate_unchanged": True,
        "point_estimate_only_no_paired_mae_interval": True,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(
    path: Path,
    report_path: Path,
    report: dict[str, Any],
    methods: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    decision = report.get("decision", {})
    invariant = report.get("invariance_and_sensitivity", {})
    rewrites = invariant.get("registered_rewrites", {})
    refinement = invariant.get("held_out_nonuniform_mass_refinement", {}).get("methods", {})
    supplemental = _supplemental_contract_review(report)
    reconstruction = _certificate_only_reconstruction_review(report)
    is_smoke = bool(report.get("protocol", {}).get("smoke", False))
    study_status = "smoke_only_not_inferential" if is_smoke else report.get("status")
    lines = [
        "# QCT-Risk v5 confirmation summary",
        "",
        f"- Source: `{report_path}`",
        f"- SHA-256: `{_sha256(report_path)}`",
        f"- Status: `{study_status}`",
        f"- Confirmation rows: `{report.get('rows', {}).get('confirmation')}`",
        "",
            "## Registered decisions",
            "",
    ]
    for key, value in decision.items():
        lines.append(f"- `{key}`: **{value}**")
    lines.extend(
        [
            "",
            "The automated inclusion flag covers the preregistered primary Brier and exact-rewrite gates. It is necessary but not sufficient: seed stability, risk-coverage usefulness, non-equivalent sensitivity, and the invariant HGB comparison still require explicit review.",
            "",
            "## Method metrics",
            "",
            *_markdown_table(
                methods,
                [
                    "method",
                    "brier",
                    "roc_auc",
                    "average_precision",
                    "ece_10bin",
                    "risk_coverage_80_tail_rate",
                    "fixed_decision_accepted_tail_rate",
                    "tangent_pairwise_mae_eur_mwh",
                ],
            ),
            "",
            "## Paired Brier comparisons",
            "",
            *_markdown_table(
                paired,
                [
                    "comparison",
                    "candidate_minus_baseline_brier",
                    "ci95_low",
                    "ci95_high",
                    "bootstrap_probability_of_improvement",
                ],
            ),
            "",
            "## Structural contracts",
            "",
            f"- Registered rewrite maximum probability drift: `{rewrites.get('maximum_probability_drift')}`",
            f"- Registered rewrite maximum decision flip rate: `{rewrites.get('maximum_decision_flip_rate')}`",
            f"- QCT held-out refinement: `{json.dumps(refinement.get('qct_full', {}), sort_keys=True)}`",
            f"- Uniform Set Transformer held-out refinement: `{json.dumps(refinement.get('uniform_set_transformer', {}), sort_keys=True)}`",
            f"- Full registered structural contract (including held-out refinement): **{supplemental['full_registered_structural_contract_pass']}**",
            f"- Maximum QCT response to non-equivalent RAM perturbation: `{supplemental['maximum_non_equivalent_ram_probability_response']}`",
            "",
            "## Certificate-only field reconstruction",
            "",
            f"- Analytic KKT-only pairwise MAE: `{reconstruction['analytic_kkt_only_pairwise_mae_eur_mwh']}` EUR/MWh",
            f"- QCT pairwise MAE: `{reconstruction['qct_full_pairwise_mae_eur_mwh']}` EUR/MWh",
            f"- Invariant HGB pairwise MAE: `{reconstruction['hgb_quotient_pairwise_mae_eur_mwh']}` EUR/MWh",
            f"- QCT relative MAE reduction versus analytic-only: `{reconstruction['qct_relative_mae_reduction_vs_analytic_only']}`",
            f"- QCT relative MAE reduction versus HGB: `{reconstruction['qct_relative_mae_reduction_vs_hgb_quotient']}`",
            "",
            "This is a registered secondary point-estimate diagnostic for reconstructing a missing or quarantined centered price-spread feed from the certificate alone. It is not a future-price forecast and has no paired MAE significance claim.",
            "",
            "The tables are a direct transformation of the frozen report. They do not apply post-hoc model selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    report_path = args.report.expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {"status", "protocol", "methods", "decision"}
    missing = required - report.keys()
    if missing:
        raise RuntimeError(f"Report is missing required fields: {sorted(missing)}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else report_path.with_suffix("").with_name(report_path.stem + "_tables")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    is_smoke = bool(report.get("protocol", {}).get("smoke", False))
    methods = _method_rows(report)
    paired = _paired_rows(report)
    seeds = _seed_rows(report)
    _write_csv(output_dir / "method_metrics.csv", methods)
    _write_csv(output_dir / "paired_brier.csv", paired)
    _write_csv(output_dir / "seed_metrics.csv", seeds)
    (output_dir / "decision_and_contracts.json").write_text(
        json.dumps(
            {
                "source": str(report_path),
                "source_sha256": _sha256(report_path),
                "status": (
                    "smoke_only_not_inferential" if is_smoke else report["status"]
                ),
                "decision": report["decision"],
                "supplemental_contract_review": _supplemental_contract_review(report),
                "certificate_only_reconstruction_review": _certificate_only_reconstruction_review(
                    report
                ),
                "invariance_and_sensitivity": report.get(
                    "invariance_and_sensitivity", {}
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "SUMMARY.md", report_path, report, methods, paired)
    print(f"[OK] wrote QCT-Risk tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
