# AF-CDMO v13: Loss-Safe Real-Market Extension

## Status

This is a preregistered design, not a completed result. It is isolated from the
frozen v10-v12 evidence. The descriptive geometry audit is:

`official_jao_dual_measure/af_cdmo_real_geometry_audit_v13.json`

The audit passes, but no predictive claim follows from that fact alone.

## Motivation

The synthetic v12 benchmark verifies the full affine-feasible quotient and
feasible-cotangent output under a known equality system. Its main empirical
limitation is that the complete operational Nordic equality system and its
output map were not operator-validated.

The real Nordic archive supports a narrower loss-safe extension. The declared
rank-five geometry contains:

1. global net-position balance, already used by Slack-QDM v11;
2. Storebaelt endpoint balance;
3. Fennoskan endpoint balance;
4. Kontiskan endpoint balance; and
5. SouthWestLink endpoint balance.

The four internal HVDC rows are restricted to links documented with zero
implicit loss factor. Skagerrak is excluded because its equality is
direction-dependent under a non-zero implicit loss factor. No zero-loss
approximation is permitted in the primary protocol.

## Geometry Gate

On the frozen 31-coordinate JAO archive, the rank-five audit reports:

- equality rank: `5`;
- tangent dimension: `26`;
- positive-dual rows: `249,795`;
- mean fraction of ambient PTDF norm removed: `0.166701`;
- rows degenerate after projection: `0`; and
- maximum equality residual over all 66 observed physical-zone transfer
  queries: `0`.

Thus every registered physical price-spread query is well-defined on the
declared quotient. This is an algebraic admissibility result, not evidence that
the learned model is more accurate.

## Proposed Task

Reconstruct the same-delivery centered Nordic day-ahead zonal price field from
an availability-filtered solved flow-based certificate, using `lastModifiedOn`
as a conservative proxy rather than definitive first-publication time. This remains a certificate
interpretation task, not a future-price forecast.

Use the same ten-zone, high-coverage panel and chronological boundaries as the
frozen v11 study:

`DK1 DK2 FI NO1 NO2 NO3 NO5 SE1 SE3 SE4`

The wider twelve-zone panel is eligible only as a declared coverage
sensitivity because its joint coverage is approximately `84.49%`, below the
primary `90%` threshold.

## Methods

The primary matched comparison must hold target, chronology, seeds, optimizer,
network width, residual target, and output geometry fixed.

1. `ambient_cqdm_residual`: row-scale/refinement quotient only.
2. `slack_qdm_residual`: adds the rank-one global-balance quotient.
3. `af_projected_raw_residual`: rank-five projection without the CQDM
   scale/refinement measure quotient.
4. `af_qdm_residual`: rank-five projection followed by the CQDM positive-dual
   measure quotient.
5. `analytic_gauge`: conserved congestion-potential control without a learned
   residual.

The new mechanism claim belongs to method 4. Method 3 isolates whether any
gain comes merely from projection rather than the complete quotient.

## Registered Presentation Orbit

Evaluation must include, at minimum:

- original certificate syntax;
- positive row scaling, dual compensation, split/merge, and permutation;
- all v11 global-slack representatives;
- the rank-five orthogonal representative;
- deterministic reflection of the equality-row-space component;
- deterministic random equality-gauge additions;
- one endpoint-slack representative for every retained zero-loss HVDC row;
- full-rank rewrites of the equality-row basis; and
- combinations of the preceding row and equality rewrites.

AF-QDM must fail closed if its continuous predictions exceed the preregistered
numerical invariance tolerance. Tree models, if retained, require a separate
feature-invariance contract because split decisions are discontinuous.

The archived certificate tensors are stored in single precision. The
implementation therefore separates algebraic exactness from serialization
roundoff: physical-query conservation has a registered `1e-3 EUR/MWh`
numerical guard and its observed maximum is reported for fit, validation, and
every evaluation presentation. The engineering smoke test observed a maximum
of `3.9586e-5 EUR/MWh`. The guard does not alter features, labels, predictions,
or losses.

## Chronology and Statistics

Reuse the frozen v11 chronology and ten preregistered seeds. Development may
fix implementation and numerical tolerances. The market-transition and
confirmation windows are then frozen. Confirmation must report:

- original-presentation pairwise MAE and RMSE;
- orbit-mean MAE;
- worst fixed-presentation MAE;
- timestamp-wise adversarial-envelope MAE;
- paired moving-block bootstrap confidence intervals;
- per-seed losses and paired sign tests; and
- exact feature and prediction drift under every registered null rewrite.

No hyperparameter may be selected on the confirmation window.

## Hypotheses

The ordered hypotheses are:

1. **Structural:** AF-QDM is invariant to the full registered rank-five
   presentation orbit.
2. **Mechanism:** Slack-QDM and ambient CQDM exhibit non-zero drift under at
   least one retained internal-HVDC equality-gauge rewrite.
3. **Robustness:** AF-QDM reduces worst-presentation and adversarial-envelope
   loss relative to the matched controls.
4. **Clean accuracy:** AF-QDM is not assumed to win on the original
   presentation. Any clean-form gain is secondary and must be reported with a
   paired interval.

## Honest Failure Modes

The extension is negative or inconclusive if any of the following holds:

- the equality-gauge stresses do not materially move the control models;
- AF-QDM is invariant but has no worst-presentation advantage;
- the projection-only control matches AF-QDM, eliminating the measure-quotient
  mechanism claim;
- the result depends on one seed or one chronology;
- the observed price labels cannot be aligned without imputation; or
- the declared equalities cannot be defended from operator documentation.

Even a positive result supports formulation robustness for solved certificate
interpretation. It does not establish a universal optimization learner, a
future-price forecasting advantage, or worldwide priority.

## Source Boundary

- JAO Nordic Publication Tool Handbook:
  <https://publicationtool.jao.eu/PublicationHandbook/Nordic_PublicationTool_Handbook_v1.5.pdf>
- Nordic RCC flow-based Q&A, including internal-HVDC equality and implicit-loss
  discussion: <https://nordic-rcc.net/flow-based/qa-epr/>
- Frozen v11 evidence and losses:
  `official_jao_dual_measure/slack_qdm_extension_*_v11.json` and matching
  `*_timestamp_losses.npz` files.
