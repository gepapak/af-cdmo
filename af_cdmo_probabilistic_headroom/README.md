# AF-CDMO Probabilistic Assurance Extension

This directory contains the isolated development, preregistration, frozen
confirmation, and post-confirmation robustness audit for QCT-Risk. It does not
modify the frozen AF-CDMO benchmark.

QCT-Risk is a **post-clearing market-certificate assurance layer**. It consumes
the affine-feasible quotient of a solved positive-dual certificate and estimates
whether the analytic certificate-to-price attribution residual is in its upper
tail. It is not a future-price forecast, trading signal, or market-clearing
surrogate.

## Frozen chronology

- fit: timestamps before 2025-11-01 UTC;
- selection: 2025-11-01 through 2025-12-31 UTC;
- development evaluation: 2026-01-01 through 2026-02-28 UTC; and
- untouched confirmation: timestamps on or after 2026-03-01 UTC.

## Confirmed result

The frozen five-seed confirmation contains 14,482 timestamps. `qct_full`
achieves Brier loss 0.040265, ROC-AUC 0.9380, and average precision 0.8215. It
beats unconditional and calendar controls under the preregistered seven-day
paired block bootstrap and satisfies the complete registered rewrite contract:
maximum probability drift is 2.54e-7 and no accept/escalate decision flips.

The evidence does **not** establish neural predictive superiority over the
invariant HGB quotient control: the seven-day interval for the Brier difference
crosses zero. That limitation is binding.

The registered `qct_no_tangent` ablation has the best neural point Brier
(0.037829). Its held-out nonuniform refinement changes probabilities by at most
4.20e-7 with zero decision flips, while the ordinary Set Transformer reaches
0.6421 maximum drift and changes 428 of 14,482 decisions (2.96%), including 39
tail-event decisions. Their predictive Brier difference is not statistically
resolved. This supports a structural assurance claim, not a generic DL-accuracy
claim.

Certificate-only tangent reconstruction is secondary: QCT improves the analytic
KKT point MAE by 19.6%, but is worse than invariant HGB and does not remain
significant under weekly block dependence.

## Reproduction and historical verification

The public package contains the implementations, protocol, theorem note, and
synthetic tests. It excludes market inputs, checkpoints, timestamp-level
outputs, and the frozen result JSON files named below.

Run the data-free tests from the repository root:

```powershell
python -m pytest tests/test_qct_risk_confirmation_v5.py -q
```

The development implementation exposes its local-data options with:

```powershell
python af_cdmo_probabilistic_headroom/train_quotient_tangent_operator_v3.py --help
```

Exact historical confirmation additionally requires the matching authorized
Nordic geometry audit, AF-CDMO confirmation reference report, and
`quotient_tangent_operator_development_v3.json`. The confirmation engine checks
the development JSON against the preregistered SHA-256 in
`QCT_RISK_CONFIRMATION_PROTOCOL_V5.md`. A newly generated JSON on another
machine can have different path metadata and will not pass that byte-level
guard. Do not remove the guard or describe such a run as recertification of
the original study. The public package does not supply every private
prerequisite for an exact historical rerun.

With the matching artifacts and authorized data supplied locally, the frozen
confirmation entry point is:

```powershell
python af_cdmo_probabilistic_headroom/confirm_quotient_tangent_risk_v5.py
```

## Reproduce the post-confirmation audit

With the complete frozen confirmation report available locally, run from the
repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qct_risk_post_confirmation_v5.ps1
```

Primary artifacts:

- `quotient_tangent_risk_confirmation_v5.json` -- immutable frozen report;
- `qct_risk_confirmation_v5_summary/` -- tables copied from the frozen report;
- `qct_risk_temporal_robustness_v5/` -- temporal, ablation, tangent, and
  refinement audits; and
- `QCT_RISK_FINAL_VERDICT_2026-10-02.md` -- local research verdict, excluded
  from the code-only package; its bounded claims are summarized above.

The main paper contribution remains AF-CDMO's market-specific economic quotient
and exact representation-assurance contract. QCT-Risk is a bounded operational
extension for missing, quarantined, asynchronous, or independently checked
price feeds.
