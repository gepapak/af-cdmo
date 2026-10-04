# Core-CCR External Generalization Protocol

## Purpose

This protocol tests whether the representation-assurance result developed on
Nordic flow-based day-ahead certificates transfers to a different European
capacity-calculation region. It is an external **method replication** on Core
CCR data with disclosed adaptations. Core has different hub coordinates and a
different physical-zone output dimension, so weights are refit chronologically.
The quotient construction, mass-attention encoder, optimizer settings, stopping
rule, rewrite families, and seed list are reused. Input/output dimensions,
equality geometry, batch size, raw-baseline prediction formulation, and alert
definition differ from the Nordic confirmation, as detailed below. The result
supports external representation assurance; it is not a zero-shot transfer of
Nordic weights or a matched ablation isolating the predictive effect of the
quotient. This document now records the completed v1 configuration accurately;
its clarification does not change the frozen results or imply a new run.

Official sources:

- JAO Core Publication Tool: <https://www.jao.eu/publication-tool>
- JAO Core API: <https://publicationtool.jao.eu/core/api>
- Core Publication Handbook: <https://publicationtool.jao.eu/core/CORE_PublicationHandbook>

## Registered scope

The public active-constraint table exposes Core PTDF coordinates, RAM, RAM at
MCP, and positive shadow prices. It does not provide enough authoritative
structure to reconstruct all four Core technical equalities and ALEGrO
virtual-hub equations without assumptions. Therefore the external geometry is
restricted to the universally valid equality

\[
  \mathbf{1}^{\top}x=0.
\]

The Core external study is consequently a conservative rank-1 quotient. It
does not claim to validate the richer rank-5 Nordic equality model.

## Data and chronology

The full registered period is `[2024-10-01, 2025-10-01)` UTC.

- Fit: timestamps before `2025-04-01T00:00:00Z`.
- Validation: `[2025-04-01, 2025-07-01)`.
- Untouched confirmation: `[2025-07-01, 2025-10-01)`.
- Fixed seeds: `7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202`.

Active constraints are requested in windows bounded by Europe/Brussels local
midnights (including 23/25-hour DST days). The top-level JAO `lastModifiedOn`
timestamp is attached only to rows from that local delivery day; any row whose
publication time is later than delivery fails the build. UTC-calendar windows
are explicitly forbidden because they can straddle two Core publication
batches. If a local-day response has a later revision timestamp, it is
automatically re-queried at the published MTU resolution (hourly or quarter
hourly). Intervals that remain post-delivery revised are excluded from both
certificate inputs and price targets and are enumerated in the manifest; they
are never treated as uncongested empty certificates.
Core directed price
spreads are converted to a centered 12-zone price field by graph least squares.
The API sign convention is resolved once on the first 30 days by comparison to
the analytic KKT field; both candidate signs, scores, and the decision ratio are
recorded. This is a one-bit schema audit, not model selection.

## Completed v1 methods and disclosed adaptations

1. `analytic_gauge`: analytic dual field; no learned residual.
2. `ambient_raw_deepset`: raw active-constraint set in ambient coordinates;
   directly predicts the centered price field.
3. `projection_only_raw_deepset`: global-balance projection without the
   positive-dual measure quotient; directly predicts the centered price field.
4. `ambient_cqdm_residual`: scale/refinement quotient without equality-gauge
   removal.
5. `core_rank1_af_qdm_residual`: global-balance projection followed by the
   positive-dual measure quotient and the frozen residual network.

The learned quotient arms add a centered neural residual to the negative
analytic dual field and also receive that field as an input. The two Core raw
controls have no analytic residual channel. They use sum pooling and have 5,180
parameters, while the quotient attention arms have 18,060. The Nordic projected
raw control learns an analytic-field residual. Core raw-versus-AF performance
therefore compares complete model configurations and cannot establish an
accuracy benefit caused solely by quotienting. Ambient CQDM and Core AF-CDMO
share the analytic residual formulation and attention architecture, so their
comparison more closely isolates equality-gauge removal.

The inherited encoder and optimizer settings are used without Core-specific tuning:
AdamW, learning rate `8e-4`, weight decay `2e-3`, Smooth-L1 objective, gradient
norm `2`, validation early stopping patience `8`, and the existing 48/32 atom
encoder plus four-query mass attention residual head in the quotient arms. The
completed full run uses at most 60 epochs and batch size 128; the Nordic v14
confirmation used batch size 256. Core uses 23 ambient coordinates, a rank-1
equality, and a centered 12-zone output (66 pairs), versus the rank-5 Nordic
geometry and ten-zone output. Each region's hub-dependent presentation count
follows the shared rewrite families; the complete Core registry has 56
presentations. The smoke command uses a smaller dataset, epoch/batch settings,
and presentation subset and does not supply the reported confirmation result.

## Registered presentation orbit

Evaluation includes the original certificate and:

- row permutation, positive row scaling, and mass-conserving duplicate split;
- their combined rewrite;
- every published hub used as a global slack;
- combined rewrite under every global slack;
- the orthogonally projected representative;
- equality reflection;
- random global-balance gauge addition;
- combined row rewrite plus each AF presentation.

All 66 physical-zone pair queries are verified tangent to global balance.
The complete presentation orbit is evaluated for every method on the
preregistered invariance seed `7`. All ten seeds are evaluated on the clean
confirmation presentation. The deterministic analytic control is evaluated
once rather than duplicated ten times as pseudo-replication.

## Outcomes

Primary numerical outcomes are clean pairwise MAE and RMSE, averaged over 66
pairs and the retained confirmation timestamps. Clean values are means of
seedwise scores, not scores of an ensemble-averaged Core prediction.
Representation assurance outcomes are maximum prediction drift, alert flips,
and cycle inconsistency over the registered presentation orbit.

Core alerts are `abs(predicted_pairwise_spread) >= 5 EUR/MWh`. Presentation flip
rates compare a rewritten prediction with the original prediction and average
over timestamp-pair cells (2,071 times 66 in the completed confirmation).
The clean-performance `alert_flip_rate` field instead compares those predicted
alerts with `abs(observed_pairwise_spread) >= 5 EUR/MWh`; it is classification
disagreement, not a presentation-instability rate. Neither Core quantity is
directly comparable with the Nordic timestamp-level reconstruction-error
alerts, which threshold the largest pairwise reconstruction error using
method-specific validation thresholds.

The reported 95% intervals resample ten seed-level MAEs 20,000 times using
bootstrap seed `20261003`. They quantify initialization/training variability
conditional on the retained market period. They do not quantify uncertainty
over dependent timestamps, and the deterministic analytic method is not given
an artificial seed sample. The complete rewrite registry was evaluated only
on seed 7; the other nine learned seeds were evaluated on the original
presentation.

Performance is never a pass/fail criterion. The run fails only for a broken
schema, causal violation, disconnected spread graph, ambiguous spread sign,
failed quotient algebra, prediction-cycle violation, or non-invariant AF
output beyond declared floating-point tolerance. A null or adverse external
performance result is retained and reported without retuning.

## Failure modes and claim discipline

- If Core AF-QDM is invariant but not more accurate, the external result
  supports representation assurance, not predictive superiority.
- If the rank-1 Core quotient is insufficient, the result cannot be used to
  reject the richer Nordic geometry.
- No Core model is selected using the confirmation interval.
- No Nordic result is overwritten by this workflow.
- The raw-versus-AF MAE reduction is descriptive of complete configurations;
  a claim that quotienting itself improves accuracy requires raw controls with
  the same analytic residual assistance and an appropriate paired uncertainty
  analysis. Neither that ablation nor temporal sampling uncertainty is supplied
  by v1.
- The v1 report retains aggregate scores, training metadata, and hashes of its
  manifest, geometry audit, and derived data files. It does not retain model
  checkpoints, timestamp-level predictions/losses, or historical source hashes.
  Thus temporal-block intervals cannot be reconstructed from its aggregates,
  and a current source fingerprint must not be described as proof of the source
  used by the historical run. The separate readiness audit checks the retained
  artifact links and recomputes aggregate summaries without refitting.
- Raw JAO responses and derived market rows remain local unless redistribution
  and ML/AI-use authorization is documented.

## Commands

Smoke verification:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_core_external_replication.ps1 -Smoke
```

Full configured replication:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_core_external_replication.ps1
```

Read-only verification of the completed aggregate evidence and, optionally,
hashes of locally held data (no training or network access):

```powershell
python -m scripts.audit_core_external_readiness --verify_local_data
```
