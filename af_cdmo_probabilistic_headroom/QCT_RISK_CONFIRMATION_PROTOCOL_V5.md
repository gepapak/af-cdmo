# QCT-Risk v5 confirmation protocol

## Scientific boundary

This protocol evaluates an isolated extension to the frozen AF-CDMO v14 study.
It does not alter the AF-CDMO implementation, its confirmation report, the
manuscript, or the code-only release.

The extension is a same-delivery market-certificate assurance mechanism. It is
not a future-price forecast and does not claim trading profitability. Its target
is whether the centered analytic KKT price field has a mean absolute pairwise
residual spread above the fit-period q90 threshold.

## Frozen development decision

- Selected architecture: `quotient_tangent_measure_operator_v3`, variant `full`.
- Development artifact SHA-256:
  `18cc7313a78afeddd6a90fc60394a6bd269128ebcee25a93dd5fbc9b97c6f821`.
- Development engine SHA-256:
  `d88d6cb50f476f7646bd0397f20c419ae2332477604d4470f956edf79d81f558`.
- Seeds: `7, 42, 123, 2025, 3007`.
- Fit: timestamps before 2025-11-01 UTC.
- Model selection: November 2025 only.
- Probability calibration: December 2025 only.
- Development evaluation used to choose the architecture: January-February 2026.
- Untouched confirmation: 2026-03-01 through the frozen AF-CDMO confirmation end.
- Main threshold: fit-period q90 residual magnitude, expected to reproduce
  `5.592173604525909 EUR/MWh` before confirmation is evaluated.
- Reproduction from serialized float32 targets is accepted only within an
  absolute numerical tolerance of `1e-6 EUR/MWh`; this tolerance does not
  alter the fixed threshold used to label or evaluate any sample.

The v4 calendar extension is rejected from the confirmation campaign because it
was inferior on the development block. No model or threshold may be selected
using confirmation outcomes.

## Methods

The registered methods are:

1. unconditional fit-period prevalence;
2. calendar-only gradient boosting;
3. simple certificate-structure gradient boosting using presence, total dual
   mass, and analytic-field scale;
4. gradient boosting on fixed exact-quotient moments;
5. mass-weighted quotient Deep Sets;
6. ordinary uniform-token Set Transformer;
7. QCT-Risk v3 full;
8. QCT-Risk without atom interactions;
9. QCT-Risk without the tangent auxiliary loss.

The last two methods are mechanism ablations, not alternative candidates.

## Primary and secondary hypotheses

Primary confirmation hypothesis:

> QCT-Risk has lower Brier loss than the unconditional and calendar controls,
> with a paired circular moving-block bootstrap 95% interval below zero.

Structural contract:

> QCT-Risk predictions and accept/escalate decisions remain unchanged within
> numerical tolerance under every registered equivalent certificate rewrite and
> under held-out nonuniform mass-conserving refinement.

Secondary comparisons against quotient HGB, quotient Deep Sets, and ordinary
Set Transformer are reported without requiring neural superiority. Failure to
beat quotient HGB must be reported explicitly.

## Metrics

- Brier loss, log loss, ROC-AUC, and average precision;
- expected calibration error, calibration intercept, and calibration slope;
- reliability bins;
- risk-coverage curves and area under the selective tail-risk curve;
- retained residual mean, p95, tail rate, and tail false-negative fraction at
  50%, 70%, 80%, 90%, 95%, and 100% coverage;
- fixed 80%-coverage accept/escalate operating point derived from December only;
- pairwise tangent-residual MAE and RMSE;
- exact rewrite probability drift, tangent drift, and decision flips;
- response to non-equivalent RAM perturbations;
- paired seven-day circular moving-block bootstrap intervals.

## Interpretation rules

- A significant win over unconditional/calendar supports certificate-conditioned
  selective assurance.
- A tie or loss against quotient HGB precludes a DL-superiority claim.
- Exact invariance plus competitive prediction supports a structured neural
  assurance claim, not a new generic Transformer claim.
- If confirmation fails, the extension stays out of the AF-CDMO paper.
- JAO data remain non-redistributable without written authorization.
