# AF-CDMO Novelty and Evidence Audit

## Executive verdict

AF-CDMO is a credible and unusually well-motivated **architecture
contribution**, with a completed ten-seed synthetic structural result. The
evidence establishes exact representation assurance and a modest matched-task
accuracy gain; it does not establish Nordic-market predictive or operational
superiority.
The implementation demonstrates that one can map an affine-feasible solved
inequality-dual certificate to an exactly presentation-invariant positive
measure and then return a canonical feasible-cotangent output. The registered
invariances are not merely encouraged by augmentation: they are enforced by the
representation and output projection.

The strongest defensible novelty is the following composition:

1. derive the nuisance relation from the restricted KKT/Lagrangian semantics of
   a solved market certificate;
2. quotient row order, positive row scaling with reciprocal dual scaling,
   dual-mass-conserving split/merge, equality-row-space gauge, and equality-basis
   rewrites;
3. represent the resulting certificate as a positive dual measure; and
4. map it to a cotangent class of feasible market transfers, represented
   canonically in `ker(E)`.

A targeted search found substantial prior art for every broad ingredient:
permutation-invariant set networks, functions on measures, mass-aware attention,
canonicalization, gauge/equivariant networks, KKT-aware learning, optimization
formulation augmentation, and projected outputs. It did **not** identify a prior
paper with AF-CDMO's exact market-certificate quotient and feasible-cotangent
codomain. This supports a qualified statement such as "to the best of our
documented search," not a worldwide-first or groundbreaking claim.

The result status is asymmetric:

| Evidence dimension | Current verdict |
|---|---|
| Algebraic construction | Strong |
| Exact invariance implementation | Strong on unit tests and a ten-seed structural benchmark |
| Materiality of the nuisance in real JAO data | Strong descriptive evidence |
| Clean matched-task predictive advantage | Modest; 1.28% over Slack-QDM and 2.80% over ambient CQDM |
| Ten-seed AF-CDMO evidence | Complete; per-seed companion in progress |
| Real Nordic AF-CDMO prediction | Not available |
| Operational usefulness | Not established |
| Standalone paper readiness | Structural/theoretical paper possible; market paper not yet |

AF-CDMO is therefore a defensible theoretical/architectural component of
Paper2. Its headline must be exact invariance under economically null
certificate rewrites, not a claim of real-market forecasting superiority.

## 1. The mathematical object

Let the affine feasible market set be

```text
M = {x in R^d : E x = e} = x0 + ker(E),
```

where `x0 = E^+ e` is the minimum-Euclidean-norm base point and `P` is the
orthogonal projector onto `ker(E)`. For a positive-dual inequality row
`(a,b,lambda)`, AF-CDMO registers the equivalence

```text
a'      = c a + E^T eta,
b'      = c b + eta^T e,
lambda' = lambda / c,                 c > 0.
```

For every `x in M`, this gives

```text
a'^T x - b' = c(a^T x - b).
```

The two rows therefore define the same restricted half-space, and reciprocal
dual scaling preserves their restricted Lagrangian contribution. AF-CDMO maps
the row to

```text
u    = P a / ||P a||,
beta = (b - a^T x0) / ||P a||,
m    = lambda ||P a||.
```

Under the registered rewrite, `Pa' = cPa`, the effective right-hand side scales
by `c`, and `m` is unchanged. The row atom `(u,beta)` and its positive mass are
therefore invariant. Splitting a coincident row while conserving multiplier
mass leaves the measure unchanged; row permutation is irrelevant by
construction.

For a certificate `C`, the quotient is the finite positive measure

```text
Q_AF(C) = sum_i m_i delta_(u_i,beta_i,q_i).
```

For any feasible displacement `delta in ker(E)`, it preserves the restricted
dual current:

```text
integral <u,delta> dQ_AF
  = sum_i lambda_i <P a_i,delta>
  = sum_i lambda_i <a_i,delta>.
```

The output covector is defined only modulo `im(E^T)`, because adding an equality
normal has zero action on every feasible displacement. The code uses the
Euclidean inner product to select the unique representative in `ker(E)`. This
is a valid cotangent construction, but the manuscript must say that Euclidean
geometry selects the representative; it is not coordinate-free under arbitrary
changes of the decision-variable basis.

## 2. What the implementation gets right

The implementation in `af_cdmo/core.py` is faithful to the derivation:

- SVD constructs a symmetric idempotent projector onto `ker(E)`.
- A pseudoinverse constructs the minimum-norm affine base point.
- Equality-system consistency is checked before use.
- Positive-dual rows are projected and translated before normalization.
- Dual mass is `lambda ||Pa||`, so positive scaling and split/merge preserve it.
- Positive-dual rows that are constant on the feasible manifold fail closed.
- Negative inequality multipliers fail closed.
- Output vectors are projected back into the feasible cotangent representative.

The neural implementation in `af_cdmo/models.py` also respects the quotient.
Its attention numerator is linear in normalized dual mass. Consequently,
splitting one atom into coincident copies with conserved mass does not alter the
attention integral. It adds a learned projected residual to the analytic
projected KKT current and projects the result again.

The test suite covers 11 important properties: projector geometry, combined
orbit invariance, equality-basis invariance, restricted-Lagrangian conservation,
cotangent-gauge invariance, neural invariance, the insufficiency of a rank-one
slack quotient for higher-rank gauge, and fail-closed handling of degenerate,
inconsistent, and negative-dual cases.

## 3. Current numerical evidence

The full frozen report is
`results/af_cdmo_v12/synthetic_comparison.json`, SHA-256
`8ff6900b4367dd833efee752794ff4d926bcc9b84ce68fab588493ed28e4617d`.
It is distinct from the retained smoke artifact. Its protocol is:

| Property | Value |
|---|---:|
| Seeds | 10 (`7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202`) |
| Training certificates | 2,048 |
| Validation certificates | 512 |
| Evaluation certificates | 1,024 |
| Epochs | 60 |
| Ambient dimension | 8 |
| Equality rank | 3 |
| Tangent dimension | 5 |

The exactness result is good:

```text
AF-CDMO maximum intrinsic drift over the complete synthetic orbit
  = 1.662442002e-6
required tolerance
  = 2.0e-5
analytic-control drift
  = 0
```

On the combined scale/split/permutation/affine-gauge/equality-basis rewrite:

| Method | Original intrinsic MAE | Combined intrinsic MAE | Maximum drift |
|---|---:|---:|---:|
| Ambient CQDM | 0.068527 | 0.139567 | 1.179163 |
| Rank-one Slack-QDM | 0.067470 | 0.127719 | 1.145936 |
| Projection-only Deep Set | 0.091275 | 0.215810 | 2.320685 |
| AF-CDMO | 0.066610 | 0.066610 | 0.000001662 |
| Analytic projected current | 0.290863 | 0.290863 | 0 |

This establishes **structural robustness**: rank-one Slack-QDM does not remove a
higher-rank equality gauge, and projection alone does not remove scaling and
refinement dependence. On the original presentation, AF-CDMO improves ensemble
MAE by 2.80% over ambient CQDM, 1.28% over Slack-QDM, 27.02% over the
projection-only control, and 77.10% over the analytic current. Under the
combined rewrite, the respective gains are 52.27%, 47.85%, 69.14%, and 77.10%.
The clean gain over Slack-QDM is real within this matched benchmark but small;
the principal result remains removal of formulation drift.

For the validation objective, AF-CDMO improves over each learned control in all
10 paired initialization seeds. The mean paired improvements are
`5.497e-5` versus ambient CQDM, `2.900e-5` versus Slack-QDM, and `9.032e-4`
versus projection-only; each exact two-sided sign test gives `p=0.001953`.
These are initialization-seed comparisons on one fixed synthetic data-generating
process, not independent market replications. A separately hashed seedwise
companion is being run to retain per-seed evaluation predictions and paired
evaluation uncertainty without modifying the frozen v12 implementation.

## 4. Why the predictive result is not independent

The synthetic target in `benchmark_af_cdmo_extension_v12.py::_target_map` is
constructed directly from AF-CDMO quotient statistics: the mass-weighted atom
mean, second moment, nonlinear mass summaries, and analytic cotangent. The
benchmark is therefore deliberately matched to AF-CDMO's inductive bias.

That is legitimate for falsifying implementation errors and demonstrating that
the registered quotient retains target-relevant information. It is not valid
evidence that AF-CDMO predicts an independently generated physical or market
quantity better than alternatives. The benchmark also averages predictions
across seeds before computing final metrics, so the future full run should
retain per-seed metrics and confidence intervals rather than report only the
ensemble mean.

Every learned method is given the true full-geometry analytic cotangent as an
input. This is useful for isolating residual representation quality, but it
also explains why the three measure-based models are almost tied and why the
analytic control remains so strong. A reviewer could reasonably conclude that
the learning head adds little beyond the known KKT current.

The projection-only control is also weaker in two ways: it has 5,336 parameters
versus 16,792 for the measure models, and ordinary sum pooling necessarily
double-counts split rows. A stronger paper needs a parameter-matched projected
raw model, a presentation-augmentation baseline, and a projected mass-aware
control without AF canonicalization.

## 5. Real-data structural evidence

The archive audit is materially useful. It covers 249,795 official JAO rows in
31 PTDF coordinates. Under a handbook-derived rank-seven candidate equality
system, full affine projection removes a mean 20.93% and median 14.97% of the
ambient PTDF-row norm, with no row becoming degenerate. Rank-one global-slack
projection removes only 8.13% on average. Thus, the proposed extra gauge is not
merely a synthetic corner case.

However, this remains descriptive evidence. The rank-seven equality system has
not been validated against all operational net-position equations, and
Skagerrak uses direction-dependent implicit losses. The map from the
31-coordinate cotangent to the observed real-zone price field also remains to
be derived. The repository correctly refuses to label this audit as a Nordic
AF-CDMO predictive result.

The official JAO handbook supports the existence of real and virtual bidding
zones, PTDF coordinates, allocation constraints, and balance conditions.^11 It
does not by itself validate the exact fixed rank-seven affine manifold used in
the descriptive audit. That distinction must remain explicit.

## 6. Novelty collision analysis

| Nearby literature | What already exists | AF-CDMO distinction that remains |
|---|---|---|
| Deep Sets and Set Transformer^1,2 | Permutation-invariant set aggregation and attention | KKT-derived continuous quotient beyond row order |
| Neural functions on measures and continuum attention^3,4 | Learning on measures, varying cardinality, function-space attention | Dual mass is forced by certificate refinement economics |
| Mass-aware neural operators^5 | Mass-weighted attention for resolution robustness | Mass is `lambda ||Pa||`, not geometric quadrature weight |
| Canonicalization and invariant learning^6,7 | Canonical forms and exact architectural invariance | Canonical object is a solved market certificate modulo its economic rewrite law |
| Category-equivariant neural networks^8 | General theory for non-invertible/compositional symmetry | AF-CDMO supplies a concrete KKT quotient and market cotangent, not generic categorical theory |
| QP augmentation^9 | Constraint scaling and reciprocal dual transformations as optimality-preserving augmentation | Exact factorization through a quotient of an already solved certificate |
| KKT Nets and optimization GNNs^10 | KKT-informed learning and prediction of primal/dual solutions | AF-CDMO consumes published solved duals and removes presentation nuisance post-solve |
| Neural Certificate Pricing^14 and DualCert^15 | Learned dual-certificate prices, structured primal recovery, and constraint-coupled KKT-manifold transitions | AF-CDMO neither predicts certificates nor learns an optimizer transition; it quotients an already solved certificate before downstream inference |
| Output projection / constrained neural nets | Hard-constrained outputs | Output is specifically a price/transfer cotangent quotient |
| Electricity dual-pricing literature^12 | Multipliers, PTDFs, price differences, dual non-uniqueness | Learned representation assurance under equivalent certificate presentations |

The search therefore rejects broad claims such as:

- first permutation-invariant market network;
- first neural network on measures;
- first mass-aware attention;
- first canonical or gauge-invariant network;
- first KKT-informed neural network;
- first projected-output model; or
- first use of PTDFs and shadow prices for price analysis.

The defensible novelty sentence is narrower:

> AF-CDMO factors post-clearing inference through a KKT-derived positive-measure
> quotient of affine-feasible solved inequality-dual certificates and returns a
> feasible-transfer cotangent representative, giving exact invariance to a
> registered family of economically null formulation rewrites.

## 7. Boundaries that must be explicit

1. **Not arbitrary affine-coordinate invariance.** The method is invariant to
   equality-row-basis rewrites and row-space gauge, not to every invertible
   reparameterization of market decision coordinates.
2. **Euclidean metric choice.** Orthogonal projection and the canonical
   cotangent representative depend on the selected coordinate metric.
3. **Positive duals only.** Negative duals are rejected; zero-dual rows are
   discarded.
4. **No constant-on-manifold atom branch.** Positive-dual rows with `Pa=0` fail
   closed. A complete operational model must decide their semantics.
5. **Registered refinements only.** Split/merge invariance assumes coincident
   atoms and duplicated auxiliary fields with conserved nonnegative dual mass.
6. **No resolution of general dual degeneracy.** Distinct active-row supports
   can generate the same aggregate stationarity current without being the same
   fine measure.
7. **No current operational map.** The 31-coordinate Nordic output map and
   time-varying/loss-aware equality model are unresolved.

## 8. Evidence needed for a strong claim

### Minimum mathematical package

1. State and prove well-definedness under the complete registered orbit.
2. Prove completeness of the finite positive measure for that orbit, with all
   exclusions stated.
3. Prove restricted-Lagrangian and feasible-transfer conservation.
4. Formalize the cotangent quotient and the metric-dependent canonical
   representative.
5. Give a continuity/stability result away from `||Pa||=0` and quantify the
   conditioning near degeneracy.
6. State a representation theorem or universal-approximation result for
   continuous functions on the bounded AF quotient, rather than relying only on
   the rank-one theorem already in `CQDM_THEOREM_NOTE.md`.

### Minimum empirical package

1. Complete and audit the isolated seedwise companion; preserve per-seed
   predictions and paired intervals rather than only an ensemble score.
2. Add parameter-matched and augmentation-trained controls.
3. Add out-of-family invariant targets not generated from the exact moments fed
   to AF-CDMO.
4. Add non-equivalent perturbations to verify that the quotient is not simply
   insensitive to everything.
5. Stress near-singular equality systems, near-degenerate projected rows,
   variable row counts, zero mass, and ill-conditioned equality-basis rewrites.
6. Validate the operational equality residuals using published net positions,
   including direction-dependent HVDC loss treatment.
7. Derive and freeze the feasible-cotangent-to-observed-price map before the
   confirmation period.
8. Compare ambient CQDM, Slack-QDM, projection-only, AF-CDMO, analytic current,
   and classical AF quotient moments on real chronology and market-design shift.
9. Report original-presentation accuracy, worst-presentation accuracy, orbit
   diameter, alert instability, and economically scaled transfer-value error.

## 9. Publication assessment

AF-CDMO is strong enough to include as a **formal architecture contribution**
and carefully labeled synthetic structural study. It is not strong enough to
support a headline claim that a new DL method improves Nordic market prediction
or operations.

The full synthetic comparison now confirms exact robustness. Completing the
formal theorem package and stronger controls would make AF-CDMO a meaningful
theoretical component of Paper2 even before a real predictive win. For a strong
standalone T-EMPR market claim, the operator-valid geometry and real-data
comparison remain essential.

The correct current conclusion is:

> The algebra, implementation, and ten-seed exact structural result are strong.
> The accuracy evidence is matched and synthetic. No real-market superiority
> claim is supported yet.

## Sources

1. Zaheer et al., “[Deep Sets](https://arxiv.org/abs/1703.06114),” NeurIPS, 2017.
2. Lee et al., “[Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://proceedings.mlr.press/v97/lee19d.html),” ICML, 2019.
3. Zweig and Bruna, “[A Functional Perspective on Learning Symmetric Functions with Neural Networks](https://proceedings.mlr.press/v139/zweig21a.html),” ICML, 2021.
4. Calvello et al., “[Continuum Attention for Neural Operators](https://www.jmlr.org/papers/v26/24-0879.html),” JMLR, 2025.
5. Yang, Du, and Liu, “[Learning Laplacian Eigenspace with Mass-Aware Neural Operators on Point Clouds](https://arxiv.org/abs/2605.24390),” 2026.
6. Ma et al., “[A Canonicalization Perspective on Invariant and Equivariant Learning](https://arxiv.org/abs/2405.18378),” NeurIPS, 2024.
7. Moskalev et al., “[On Genuine Invariance Learning without Weight-Tying](https://proceedings.mlr.press/v221/moskalev23a.html),” TAG-ML, 2023.
8. Maruyama, “[Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems](https://arxiv.org/abs/2511.18417),” 2025.
9. Qian and Morris, “[Principled Data Augmentation for Learning to Solve Quadratic Programming Problems](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c07d71ff0bc042e4b9acd626a79597fa-Abstract-Conference.html),” NeurIPS, 2025.
10. Arvind, Pomaje, and Bhat, “[Karush-Kuhn-Tucker Condition-Trained Neural Networks](https://arxiv.org/abs/2410.15973),” 2024.
11. JAO, “[Nordic Publication Tool Handbook](https://publicationtool.jao.eu/nordic/Nordic_PublicationHandbook),” version 1.5/1.7 publication page, 2026.
12. Feng et al., “[Spot Pricing When Lagrange Multipliers Are Not Unique](https://doi.org/10.1109/TPWRS.2011.2159629),” IEEE Transactions on Power Systems, 2012.
13. Nordic CCR TSOs, “[Methodology and Concepts for the Nordic Flow-Based Market Coupling](https://consultations.entsoe.eu/markets/capacity-calculation-methodology-proposal-for-the/supporting_documents/Supporting%20Document%20and%20Impact%20Assessment%20%20for%20the%20Nordic%20CCM.pdf),” 2017.
14. Chen, Zhang, and Qian, “[Neural Certificate Pricing for Combinatorial Optimization Problems](https://arxiv.org/abs/2607.01185),” 2026.
15. Song et al., “[DualCert: A Solver for the Traveling Salesman Problem with Constraint-Coupled Learning](https://arxiv.org/abs/2608.09042),” 2026.
