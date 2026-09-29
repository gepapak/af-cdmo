# AF-CDMO v14: Final T-EMPR Positioning

## Recommended title

**Representation-Safe Learning from Flow-Based Market Certificates: An
Affine-Feasible Dual-Measure Operator for Nordic Market Monitoring**

## Journal fit

The work should be submitted as a market-monitoring and flow-based
market-coupling contribution. IEEE Transactions on Energy Markets, Policy and
Regulation explicitly includes market monitoring, market coupling, congestion
management, pricing, and settlement. It also states that forecasting-method or
generic optimization papers with only incidental market relevance are out of
scope. The manuscript must therefore lead with the operational representation
risk and its market-governance consequences, not with neural architecture.

Official scope:

https://ieee-pes.org/publications/transactions-on-energy-markets-policy-and-regulation/

## Central problem

Operator-published solved flow-based certificates can admit multiple numerical
presentations that encode the same feasible market object and restricted dual
functional. A downstream model that reacts to row ordering, units, slack choice,
declared equality gauges, or duplicated dual mass can change a market-monitoring
decision even though the represented economics did not change.

## Contribution hierarchy

1. **Market object.** Define the positive-dual affine-feasible certificate
   quotient appropriate to the audited Nordic flow-based geometry.
2. **Conservation.** Prove that the finite dual-measure representation preserves
   the restricted solved-certificate Lagrangian functional on feasible market
   displacements.
3. **Output geometry.** Return the canonical feasible-cotangent representative,
   ensuring that reported physical price spreads respect the declared market
   manifold.
4. **Exact architecture.** Construct AF-CDMO so registered presentation rewrites
   cannot change predictions or monitoring alerts by design.
5. **Adversarial protocol.** Evaluate 80 equivalent presentations, ten training
   seeds, three chronological evaluation windows, matched projection,
   uniform-mass, augmentation, slack-only, ambient, and analytic controls, plus
   a non-equivalent RAM negative control.

Generic KKT duality, set networks, attention, finite measures, and projection
are not individually novel. The defensible novelty is their market-derived
factorization through this specific solved-certificate quotient, together with
the exact representation-risk statement and registered market-monitoring test.

## Confirmation evidence

The confirmation window contains 14,482 timestamps and 80 registered
presentations.

- AF-CDMO maximum registered prediction drift:
  `2.3651e-5 EUR/MWh` at a `1e-3 EUR/MWh` tolerance.
- AF-CDMO registered monitoring-alert flip rate: `0`.
- Ambient model maximum drift: `198.726 EUR/MWh`.
- Slack-only model maximum drift: `158.444 EUR/MWh`.
- Ambient and slack-only maximum alert-flip rates: approximately `30.1%` and
  `36.4%`.
- AF-CDMO adversarial-envelope improvement over ambient:
  `61.49%`, block-bootstrap 95% interval `[50.21%, 71.53%]`.
- AF-CDMO adversarial-envelope improvement over slack-only:
  `58.74%`, interval `[48.14%, 68.12%]`.
- Under held-out nonuniform mass refinement, AF-CDMO maximum ensemble drift is
  `2.265e-5 EUR/MWh`; uniform token attention drifts by up to
  `45.99 EUR/MWh`.
- Under non-equivalent +/-20% RAM perturbations, AF-CDMO mean prediction movement
  is approximately `0.30-0.31 EUR/MWh`, with maxima above `6.5 EUR/MWh`.

These results support exact representation assurance and nontrivial physical
sensitivity. They do not support universal predictive superiority.

## Results that must be reported prominently

- Projection-only has lower clean confirmation MAE than AF-CDMO:
  `1.674` versus `2.283 EUR/MWh`; AF-CDMO's clean loss is approximately
  `36.37%` higher.
- Uniform attention also has lower clean confirmation MAE on the standard
  presentations, although it fails the held-out nonuniform refinement test.
- The analytic gauge and AF-CDMO are practically tied on the main confirmation
  task; the paired interval is wide and crosses zero.
- AF-CDMO's adversarial improvement over projection-only is approximately
  `10.77%`, but its temporal bootstrap interval crosses zero.

These controls prevent a predictive-superiority claim. They strengthen the
representation-assurance claim by showing exactly what the learned component
does and does not add.

## Reviewer-facing answers

**Why use learning when the analytic gauge approximately ties it?**

The theorem and analytic control establish the invariant market coordinate.
AF-CDMO demonstrates how a learned nonlinear residual can be restricted to that
coordinate without reopening presentation risk. The paper should not claim that
the nonlinear residual is always necessary; it should present the analytic
control as an important deployment option and scientific boundary.

**Why is lower clean MAE not the primary endpoint?**

The registered endpoint is decision instability over economically equivalent
presentations. A clean-form model can be more accurate on one syntax while
producing different prices or alerts under an equivalent syntax. The paper must
report the accuracy--assurance tradeoff rather than conceal it.

**Are the transformations economically equivalent?**

Only the explicitly proved and audited relation is claimed: permutation,
positive row scaling with reciprocal dual scaling, declared affine equality
gauges/bases, and conserved-mass split/merge. Active-set, topology, RAM, loss,
and arbitrary dual-selection changes are excluded.

**Is the task causal forecasting?**

No. It is same-delivery reconstruction from a solved certificate. The archive
uses `lastModifiedOn` as a conservative availability proxy, not definitive
first-publication evidence.

## Submission blockers

1. Recertify development and frozen-transition reports under the final engine
   hash.
2. Obtain written JAO authorization and rotate exposed credentials.
3. Complete independent theorem review.
4. Release only the verified code-only package unless data-provider permission
   explicitly allows more.

