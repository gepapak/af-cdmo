# AF-CDMO v14 Submission Readiness

Status date: 2026-09-29

## Scientific status

The final confirmation study is computationally certified under the current
engine source. It evaluates 14,482 confirmation timestamps, ten preregistered
training seeds, seven matched methods, and 80 registered equivalent
presentations. AF-CDMO satisfies its exact presentation-invariance and
zero-alert-flip contracts while remaining responsive to non-equivalent RAM
perturbations.

The submission is defensible as a **market-certificate representation-assurance
and monitoring paper**. It is not a future-price forecasting paper, a trading
paper, or evidence that neural learning universally improves clean-form
prediction.

## Defensible primary claim

> Factoring solved Nordic flow-based certificates through the declared
> positive-dual affine-feasible quotient eliminates representation-induced
> prediction and monitoring instability for registered economically equivalent
> certificate rewrites, while preserving sensitivity to non-equivalent market
> changes.

The narrower learned contribution is a market-specific neural factorization of
that quotient. Generic attention, Deep Sets, KKT duality, positive row scaling,
and finite-measure aggregation are prior ingredients and must not be claimed as
new individually.

## Required interpretation

- Lead with representation risk, market monitoring, congestion certificates,
  and governance of flow-based market analytics.
- Report clean accuracy and robustness together.
- State that projection-only and uniform-mass controls can have lower clean MAE.
- State that the analytic gauge is approximately tied with AF-CDMO on the main
  confirmation task; the learned residual is not established as universally
  superior.
- Use the held-out nonuniform refinement result to distinguish conserved dual
  mass from uniform token attention.
- Describe `lastModifiedOn` as a conservative availability proxy.
- Disclose that headline predictions are ten-seed ensembles and provide the
  seed-level tables already stored in the reports.

## Computational actions

- [x] Final confirmation report passes the fail-closed auditor under the current
  benchmark hash.
- [x] The focused theorem, geometry, benchmark, control, and calendar-bootstrap
  tests pass.
- [x] Recompute the development report under the final benchmark source.
- [ ] Recompute the frozen-transition report under the final benchmark source.
- [ ] Run the strict submission-readiness audit after both reports are
  recertified.

The development role has been recertified under the final benchmark source.
The remaining stale frozen-transition report is a valid historical output, but
its embedded benchmark hash predates the held-out refinement implementation.
Hashes must not be edited manually. Re-evaluation can reuse compatible trained
checkpoints and does not require model retraining:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\recertify_af_cdmo_v14_final.ps1"
```

This is a long evaluation-only run. It creates backups and restores them if a
role fails.

## External actions

- [ ] Revoke or rotate every credential previously pasted into a chat or
  transcript.
- [ ] Obtain written JAO authorization for the intended academic ML use and
  release boundary.
- [ ] Confirm the applicable ENTSO-E attribution and redistribution terms.
- [ ] Ask an independent mathematical reviewer to check the exact quotient and
  factorization theorem before submission.
- [x] Select a software license for the code-only repository (Apache-2.0).

Until the authorization item is resolved, publish only the code-only package
created by `scripts/build_public_release.py` and do not upload private source
archives, trained weights, or timestamp-level derivatives.
