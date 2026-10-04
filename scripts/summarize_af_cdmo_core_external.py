"""Create compact CSV/Markdown tables from a completed Core replication."""

from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path

import pandas as pd


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, record in frame.iterrows():
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.6g}" if isinstance(value, numbers.Real) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol") != "af_cdmo_core_external_replication_v1":
        raise RuntimeError("Unexpected Core benchmark protocol")
    if not report.get("structural_contract", {}).get("passed"):
        raise RuntimeError("Cannot summarize a structurally failed Core benchmark")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clean_rows = []
    for method, values in report["summary"]["clean"].items():
        low, high = values["pairwise_mae_bootstrap_95_ci_eur_mwh"]
        clean_rows.append(
            {
                "method": method,
                "seeds": values["seeds"],
                "pairwise_mae_mean_eur_mwh": values["pairwise_mae_mean_eur_mwh"],
                "pairwise_mae_std_eur_mwh": values["pairwise_mae_std_eur_mwh"],
                "pairwise_mae_ci95_low_eur_mwh": low,
                "pairwise_mae_ci95_high_eur_mwh": high,
                "pairwise_rmse_mean_eur_mwh": values["pairwise_rmse_mean_eur_mwh"],
                "spread_threshold_disagreement_rate_mean": values["alert_flip_rate_mean"],
            }
        )
    clean = pd.DataFrame(clean_rows).sort_values("pairwise_mae_mean_eur_mwh")
    clean_path = output / "core_external_clean_summary.csv"
    clean.to_csv(clean_path, index=False)

    invariance_rows = [
        {"method": method, **values}
        for method, values in report["summary"]["presentation_invariance"].items()
    ]
    invariance = pd.DataFrame(invariance_rows).sort_values(
        "maximum_presentation_drift_eur_mwh"
    )
    invariance_path = output / "core_external_invariance_summary.csv"
    invariance.to_csv(invariance_path, index=False)

    split = report["data"]["split"]
    parameter_counts = {
        entry["method"]: entry["parameter_count"] for entry in report["training"]
    }
    lines = [
        "# Core-CCR external AF-CDMO replication",
        "",
        f"Structural contract: **PASS**",
        "",
        "This is an external method replication with chronological Core refitting "
        "and disclosed geometry, dimension, batch-size, raw-baseline, and alert-metric adaptations.",
        "",
        f"- Fit end: `{split['fit_end_utc']}`",
        f"- Validation end: `{split['validation_end_utc']}`",
        f"- Evaluation end: `{split['evaluation_end_utc']}`",
        f"- Split rows: `{split['rows']}`",
        f"- Presentations: `{report['frozen_configuration']['presentation_count']}`",
        f"- Seeds: `{report['frozen_configuration']['seeds']}`",
        f"- Batch size: `{report['frozen_configuration']['batch_size']}` (Nordic v14: `256`)",
        f"- Full presentation registry evaluated on seed: `{report['frozen_configuration']['invariance_seed']}`",
        "- Clean scores are means of seedwise scores; they are not scores of an ensemble prediction.",
        "",
        "## Clean-presentation performance",
        "",
        _markdown(clean),
        "",
        "The 95% intervals bootstrap seed-level MAEs 20,000 times and describe "
        "training variability conditional on this market period. They do not quantify "
        "temporal sampling uncertainty. The analytic control is deterministic and is evaluated once.",
        "",
        "The spread-threshold disagreement rate compares predicted and observed "
        "absolute pairwise spreads against 5 EUR/MWh, averaged over timestamp-pair cells. "
        "It measures classification disagreement, not presentation instability.",
        "",
        "## Registered-presentation stability",
        "",
        _markdown(invariance),
        "",
        "Presentation alert flips compare the original and rewritten predictions at the "
        "same 5 EUR/MWh absolute-spread threshold, averaged over timestamp-pair cells. "
        f"All listed presentations were evaluated only on seed {report['frozen_configuration']['invariance_seed']}; "
        f"all {len(report['frozen_configuration']['seeds'])} learned seeds have clean scores. "
        "These Core metrics differ from Nordic timestamp-level maximum-reconstruction-error alerts.",
        "",
        "## Claim boundary",
        "",
        report["claim_boundary"],
        "",
        f"Core raw controls directly predict a centered price field ({parameter_counts['ambient_raw_deepset']:,} parameters). "
        "The quotient attention controls predict a residual added to an analytic field "
        f"and also consume that field as input ({parameter_counts['core_rank1_af_qdm_residual']:,} parameters). Their accuracy comparison "
        "describes complete configurations and does not isolate an effect of quotienting. "
        "The Nordic projected-raw control uses an analytic residual channel; the formulations differ.",
        "",
        "The retained v1 benchmark contains aggregate records and hashes of its data manifest, "
        "geometry audit, and derived data, but no historical source hashes, checkpoints, "
        "or timestamp predictions/losses. Temporal-block intervals cannot be recovered from "
        "these aggregates, and current source fingerprints are not historical engine provenance.",
        "",
        "Performance was not used as a pass/fail condition. Any null or adverse external "
        "result must be reported without retuning the frozen protocol.",
        "",
    ]
    summary_path = output / "CORE_EXTERNAL_RESULT_SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Core external summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
