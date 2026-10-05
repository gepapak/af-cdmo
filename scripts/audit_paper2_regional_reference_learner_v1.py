"""One frozen learner contract under a source-inspired regional-reference rule.

Explicit authorized local inputs only. Rank6 inputs are outside the original
training geometry. This evaluates re-expression stability, never accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_paper2_regional_reference_adapter_v1 as adapter
from scripts import audit_paper2_regional_reference_v1 as reference


def learner_input(values: dict, equality: np.ndarray, zones: list[str],
                  target_scale: float, ram_scale: float, q_scales: np.ndarray) -> dict:
    """Match original AF data precision, with an explicitly supplied geometry."""
    represented = values["normals"] @ reference.projector(equality)
    length = np.linalg.norm(represented, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("Projected constant positive-dual row: fail closed")
    directions = represented / length[:, None]
    mass = values["dual"] * length
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Finite positive total mass required")
    # Original ambient canonical q is cast to float32 before AF reconstruction.
    q = np.arcsinh(values["auxiliary"] / q_scales).astype(np.float32).astype(np.float64)
    canonical = np.column_stack((directions, np.arcsinh(values["ram"] / length / ram_scale), q)).astype(np.float32)
    indices = [zones.index(zone) for zone in reference.PHYSICAL_ZONES]
    analytic = -(mass @ directions[:, indices])
    analytic -= analytic.mean()
    return dict(canonical=canonical, weights=(mass / total).astype(np.float32),
        total_scaled=np.float32(np.log1p(total)), analytic_scaled=(analytic / target_scale).astype(np.float32),
        analytic_raw=analytic.astype(np.float32))


def predict(model, values: dict, target_scale: float) -> np.ndarray:
    import torch
    batch = dict(canonical=torch.from_numpy(values["canonical"][None]),
        weights=torch.from_numpy(values["weights"][None]),
        mask=torch.ones((1, len(values["weights"])), dtype=torch.bool),
        total=torch.tensor([[values["total_scaled"]]], dtype=torch.float32),
        present=torch.ones((1, 1), dtype=torch.float32),
        analytic=torch.from_numpy(values["analytic_scaled"][None]))
    with torch.no_grad():
        residual = model(batch, torch.device("cpu")).cpu().numpy()[0] * target_scale
    field = np.asarray(values["analytic_raw"] + residual, dtype=np.float64)
    return field - field.mean()


def emitted_spreads(field: np.ndarray, zones: list[str], equality: np.ndarray) -> dict:
    physical_field = np.zeros(len(zones))
    for zone, value in zip(reference.PHYSICAL_ZONES, field):
        physical_field[zones.index(zone)] = value
    return adapter.admissible_spreads(physical_field, zones, equality)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--historical-scale-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan == args.run:
        parser.error("Choose exactly one of --plan/--run")
    dependencies = [Path(__file__), Path(adapter.__file__), Path(reference.__file__),
        ROOT / "scripts/multizone_quotient_gauge_v6.py", ROOT / "scripts/benchmark_certificate_governance_v5.py",
        ROOT / "scripts/cqdm_utils.py"]
    protocol = dict(role="Post-confirmation frozen seed7 learner input/output contract; no accuracy evaluation",
        prior_exposure="Observed adjacent-row geometry/census and model-free controlled adapter already examined; no learner inference for this operational rule",
        inputs={key: {"path": str(path.resolve()), "sha256": reference.sha256(path)} for key, path in
                (("dual_archive", args.dual_archive), ("geometry_audit", args.geometry_audit),
                 ("historical_scale_archive", args.historical_scale_archive), ("checkpoint", args.checkpoint))},
        source_code_hashes={path.name: reference.sha256(path) for path in dependencies},
        delivery=args.delivery, checkpoint_identity=dict(method="af_qdm_residual", kind="cqdm", seed=7),
        rewrite="All stored rows: a_new=a-a(SE2)*v_N; RAM,lambda,q unchanged; original raw a/lambda/RAM precision float32 preserved before representation",
        scale_policy="Exact checkpoint target/RAM scales; q positive medians from original timely resolution-expanded training archive and checkpoint fit cutoff; no later fitting",
        comparison="Same unchanged original learner on original rank5 inputs and separately adapted rank6 inputs; each before/after comparison emits only36 admissible within-Nordic pairs",
        query_policy="All9 DK1-crossregional pairs absent; no crossregional predictions or targets exported",
        claims="Rank6 input out-of-training-geometry; controlled re-expression stability only, not accuracy/calibration/prospective benefit; original confirmation unchanged",
        no_labels_market_prices_fitting_or_model_changes=True, cpu_threads=1)
    plan_path = args.output / "protocol.json"
    if args.plan:
        if plan_path.exists():
            raise FileExistsError("Preserve frozen protocol; amendments require a new directory")
        reference.save(plan_path, dict(protocol, frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        print("Learner protocol frozen before checkpoint loading or inference.")
        return
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if {key: value for key, value in frozen.items() if key != "frozen_at_utc"} != protocol:
        raise RuntimeError("Frozen protocol, source or inputs changed")
    import pandas as pd
    import torch
    from scripts.benchmark_certificate_governance_v5 import _read_certificate
    from scripts.multizone_quotient_gauge_v6 import MultiZoneRegressor
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if any(checkpoint[key] != value for key, value in protocol["checkpoint_identity"].items()):
        raise RuntimeError("Wrong method/kind/seed checkpoint")
    training = checkpoint["training_signature_payload"]
    if reference.sha256(args.historical_scale_archive) != training["dual_archive_sha256"]:
        raise RuntimeError("Auxiliary scales require the exact original training archive")
    if reference.sha256(args.geometry_audit) != training["geometry_audit_sha256"]:
        raise RuntimeError("Geometry audit differs from original checkpoint provenance")
    frame, audit = _read_certificate(args.historical_scale_archive, 60.0)
    fit = frame.loc[frame["delivery_utc"] < pd.Timestamp(training["fit_end_exclusive"])]
    q_scales = []
    for field in reference.AUXILIARY_FIELDS:
        finite = np.abs(fit[field].to_numpy(dtype=np.float64))
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        q_scales.append(max(float(np.median(finite)), 1e-6) if len(finite) else 1.)
    fit_rows = len(fit)
    del frame, fit
    zones = json.loads(args.geometry_audit.read_text(encoding="utf-8-sig"))["geometry"]["zone_order"]
    if training["zones"] != list(reference.PHYSICAL_ZONES):
        raise RuntimeError("Original output-zone order differs from the declared physical queries")
    e5, e6, regional = reference.declared_geometries(zones)
    values = adapter.certificate(args.dual_archive, args.delivery, zones)
    # Original _build_vector_samples stores raw rows as float32 before AF views.
    for key in ("normals", "ram", "dual"):
        values[key] = values[key].astype(np.float32).astype(np.float64)
    rewritten = dict(values, normals=values["normals"] - values["normals"][:, zones.index("SE2"), None] * regional[None, :])
    model = MultiZoneRegressor("cqdm_gauge_residual", 1 + len(zones) + 5, len(zones) + 5, len(reference.PHYSICAL_ZONES))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    target_scale, ram_scale = training["target_scale"], training["canonical_ram_scale"]
    comparisons = {}
    for name, equality in (("original_rank5", e5), ("adapted_rank6_out_of_training_geometry", e6)):
        representations = [learner_input(item, equality, zones, target_scale, ram_scale, np.asarray(q_scales))
                           for item in (values, rewritten)]
        spreads = [emitted_spreads(predict(model, item, target_scale), zones, e6) for item in representations]
        changes = {key: abs(spreads[1][key] - spreads[0][key]) for key in spreads[0]}
        comparisons[name] = dict(admissible_pair_count=len(changes), learner_spread_max_abs_difference_eur_mwh=max(changes.values()),
            learner_spread_abs_differences_eur_mwh=changes,
            canonical_float32_max_abs_difference=float(np.max(np.abs(representations[1]["canonical"] - representations[0]["canonical"]))),
            weights_float32_max_abs_difference=float(np.max(np.abs(representations[1]["weights"] - representations[0]["weights"]))),
            analytic_input_float32_max_abs_difference=float(np.max(np.abs(representations[1]["analytic_scaled"] - representations[0]["analytic_scaled"]))))
    result = dict(protocol_sha256=reference.sha256(plan_path), comparisons=comparisons,
        scales=dict(target=target_scale, canonical_ram=ram_scale, auxiliary=dict(zip(reference.AUXILIARY_FIELDS, q_scales)),
                    original_training_archive_timely_rows=audit["timely_rows_after_resolution_expansion"], scale_fit_rows=fit_rows),
        checkpoint_identity=protocol["checkpoint_identity"], checkpoint_sha256=reference.sha256(args.checkpoint),
        query_policy=reference.query_audit(zones, e6), rank6_inputs_out_of_training_geometry=True,
        original_checkpoint_modified=False, original_confirmation_modified=False,
        accuracy_calibration_labels_or_prospective_benefit_evaluated=False)
    reference.save(args.output / "regional_reference_learner.json", result)
    print(json.dumps(dict(comparisons={key: {"max_spread_drift": value["learner_spread_max_abs_difference_eur_mwh"],
        "canonical_drift": value["canonical_float32_max_abs_difference"]} for key, value in comparisons.items()},
        scales=result["scales"], rejected_count=result["query_policy"]["rejected_count"])))


if __name__ == "__main__":
    main()
