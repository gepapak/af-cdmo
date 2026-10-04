"""Summarize frozen v14 presentation stress by registered rewrite family."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.cqdm_utils import sha256_file
except ModuleNotFoundError:
    from cqdm_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "official_jao_dual_measure" / "af_cdmo_real_confirmation_v14.json"
DEFAULT_OUTPUT = (
    ROOT / "official_jao_dual_measure" / "af_cdmo_presentation_stress_summary_v15.json"
)


def _family(name: str) -> tuple[str, bool]:
    combined = name == "combined" or name.startswith("combined_")
    base = name.removeprefix("combined_") if combined else name
    if base == "original":
        return "identity", combined
    if base == "combined":
        return "row_scale_split_permutation", True
    if base == "projected_representative":
        return "global_projected_representative", combined
    if base.startswith("slack_"):
        return "global_slack", combined
    if base == "af_projected_representative":
        return "affine_projected_representative", combined
    if base == "equality_reflection":
        return "equality_basis_reflection", combined
    if base == "random_equality_gauge":
        return "random_equality_gauge", combined
    if base.startswith("gauge_"):
        return "single_equality_row_gauge", combined
    raise ValueError(f"Unknown registered presentation: {name}")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    mean_drifts = np.asarray(
        [record["mean_abs_pairwise_prediction_drift_eur_mwh"] for record in records],
        dtype=np.float64,
    )
    max_drifts = np.asarray(
        [record["max_abs_pairwise_prediction_drift_eur_mwh"] for record in records],
        dtype=np.float64,
    )
    alert_rates = np.asarray(
        [record["alert_flip_rate"] for record in records], dtype=np.float64
    )
    return {
        "presentation_count": len(records),
        "presentation_mean_drift_eur_mwh": {
            "median": float(np.median(mean_drifts)),
            "p95": float(np.quantile(mean_drifts, 0.95)),
            "maximum": float(mean_drifts.max(initial=0.0)),
        },
        "presentation_max_drift_eur_mwh": {
            "median": float(np.median(max_drifts)),
            "p95": float(np.quantile(max_drifts, 0.95)),
            "maximum": float(max_drifts.max(initial=0.0)),
        },
        "timestamp_alert_flip_share": {
            "median_across_presentations": float(np.median(alert_rates)),
            "p95_across_presentations": float(np.quantile(alert_rates, 0.95)),
            "maximum_across_presentations": float(alert_rates.max(initial=0.0)),
        },
        "share_of_presentations_with_any_alert_flip": float(np.mean(alert_rates > 0.0)),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv_output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    csv_output = (
        args.csv_output.expanduser().resolve()
        if args.csv_output is not None
        else output.with_suffix(".csv")
    )
    existing = [path for path in (output, csv_output) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )
    report = json.loads(source.read_text(encoding="utf-8"))
    if report["protocol"].get("study_role") != "confirmation":
        raise RuntimeError("Stress summary must use the frozen confirmation report")

    methods: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for method, method_result in report["results"].items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        all_nonidentity: list[dict[str, Any]] = []
        for name, record in method_result["stresses"].items():
            family, combined = _family(name)
            if name != "original":
                all_nonidentity.append(record)
            label = f"combined+{family}" if combined and family != "identity" else family
            grouped.setdefault(label, []).append(record)
        grouped_summary = {
            family: _aggregate(records) for family, records in sorted(grouped.items())
        }
        methods[method] = {
            "all_nonidentity_presentations": _aggregate(all_nonidentity),
            "families": grouped_summary,
        }
        for family, values in grouped_summary.items():
            flat_rows.append(
                {
                    "method": method,
                    "family": family,
                    "presentation_count": values["presentation_count"],
                    "median_mean_drift_eur_mwh": values[
                        "presentation_mean_drift_eur_mwh"
                    ]["median"],
                    "p95_mean_drift_eur_mwh": values[
                        "presentation_mean_drift_eur_mwh"
                    ]["p95"],
                    "maximum_drift_eur_mwh": values[
                        "presentation_max_drift_eur_mwh"
                    ]["maximum"],
                    "median_alert_flip_share": values[
                        "timestamp_alert_flip_share"
                    ]["median_across_presentations"],
                    "maximum_alert_flip_share": values[
                        "timestamp_alert_flip_share"
                    ]["maximum_across_presentations"],
                }
            )

    result = {
        "protocol": {
            "name": "af_cdmo_registered_presentation_stress_summary_v15",
            "study_role": "post_confirmation_descriptive_summary",
            "models_retrained": False,
            "unit_of_family_quantiles": "registered presentations",
            "timestamp_effect_metric": "within-presentation retrospective alert flip share",
            "natural_rewrite_frequency_estimated": False,
        },
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "engine": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "methods": methods,
    }
    _atomic_json(output, result)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_output.with_suffix(csv_output.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    os.replace(temporary_csv, csv_output)
    print(f"[OK] wrote {output}")
    print(f"[OK] wrote {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
