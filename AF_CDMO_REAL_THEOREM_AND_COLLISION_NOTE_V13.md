# AF-CDMO v13: Real-Market Specialization and Collision Boundary

## Status

This note specializes the AF-CDMO v12 mathematics to the loss-safe Nordic
geometry used by the v13 development experiment. It is a proof and claim
boundary, not a record of a completed predictive result. The v13 report and
fail-closed manifest remain the only admissible sources for empirical claims.

## 1. Real loss-safe geometry

Let `x in R^31` denote the published Nordic PTDF-coordinate net-position
vector. The primary v13 study declares the homogeneous equality system

```text
E x = 0,
```

whose five rows encode global net-position balance and endpoint balance for
Storebaelt, Fennoskan, Kontiskan, and SouthWestLink. These four internal HVDC
links are retained because the operator documentation assigns them zero
implicit loss factor. Skagerrak is excluded because its implicit-loss relation
is direction dependent.

The Euclidean tangent projector is

```text
P_E = I - E^T (E E^T)^+ E.
```

The audited geometry has rank five and tangent dimension 26. On the frozen
archive, no positive-dual PTDF row becomes degenerate after projection.

## 2. Certificate quotient

For each retained solved inequality row `(a_i,b_i,lambda_i,q_i)`, with
`lambda_i>0` and `P_E a_i != 0`, define

```text
r_i    = ||P_E a_i||_2,
u_i    = P_E a_i / r_i,
beta_i = b_i / r_i,
m_i    = lambda_i r_i,
z_i    = (u_i,beta_i,q_i),
Q_E(C) = sum_i m_i delta_(z_i).
```

The zero right-hand side of the five equality rows is why no affine base-point
term appears in `beta_i` here. This is the homogeneous specialization of the
v12 construction.

### Proposition 1: nested quotient

Let `P_1` project only against global balance. Because the global-balance row
space is a subspace of `row(E)`,

```text
P_E P_1 = P_1 P_E = P_E.
```

Therefore AF-QDM strictly extends the Slack-QDM canonicalization whenever a
published PTDF row has a nonzero component along at least one retained
internal-HVDC equality that is not collinear with global balance.

The archive audit measures this condition directly: the mean fraction of
ambient PTDF norm removed by `P_E` is approximately `0.166701`, and the
registered internal-HVDC gauge rewrites alter both the ambient and slack-only
representations while leaving AF-QDM unchanged.

### Proposition 2: homogeneous affine-gauge invariance

For any `c>0` and equality coefficient vector `eta`, rewrite one row as

```text
a'      = c a + E^T eta,
b'      = c b,
lambda' = lambda/c.
```

Then `(u,beta,m,q)` is unchanged. This follows from `P_E E^T=0` and positive
homogeneity of the norm. Consequently `Q_E` is invariant to row order,
positive row scaling with reciprocal dual scaling, equality-row-space gauge,
and mass-conserving split/merge of coincident atoms.

### Proposition 3: restricted dual-functional conservation

For every feasible displacement `delta in ker(E)`,

```text
sum_i lambda_i (a_i^T delta - b_i)
  = integral (u^T delta - beta) dQ_E(u,beta,q).
```

Thus the quotient does not merely remove syntax: it preserves the solved
certificate's dual functional on the feasible market manifold.

## 3. Physical price-spread observability

Ambient price covectors `p` and `p+E^T eta` agree on every feasible transfer.
A physical spread query `q_ij=e_i-e_j` is therefore well-defined on the
cotangent quotient exactly when

```text
E q_ij = 0.
```

The geometry audit checks all 66 observed physical-zone transfer queries and
finds zero equality residual. The v13 target is consequently the centered
ten-zone physical price field, an equivalent coordinate representation of
those pairwise spreads. This does not identify virtual-zone endpoint prices
or claim that every ambient 31-coordinate price is observable.

## 4. Exact presentation-risk statement

Let `f` be any deterministic network whose input depends only on `Q_E(C)` and
whose reported output is the centered physical price field. For every
certificate presentation in the declared equivalence class,

```text
f(C') = f(C).
```

Hence every pointwise loss against a presentation-invariant physical-price
target is constant over the complete declared orbit. This is an architectural
guarantee. Finite formulation augmentation can estimate robustness on sampled
rewrites but cannot, by itself, certify the unsampled continuous equality
gauge.

The statement is deliberately narrow. It excludes non-equivalent changes to
RAM, active-set membership, dual selection, market topology, implicit-loss
rules, and arbitrary changes of PTDF coordinates.

### Corollary 4.1: task sufficiency without Bayes-information loss

Let `Y` be the physical-spread target and suppose its conditional law is
constant on the registered equivalence classes:

```text
Law(Y | C) = Law(Y | C') whenever Q_E(C)=Q_E(C').
```

Then every Bayes act for any loss admitting a measurable Bayes decision can
be chosen to factor through `Q_E`. This follows because the conditional risk
is itself a function of the quotient class. Restricting the predictor to the
AF quotient therefore removes nuisance presentation information without
excluding a Bayes-optimal predictor for the declared invariant task.

This result does not assert that the real target is fully determined by the
certificate, nor that AF-QDM will have lower finite-sample clean loss. It says
only that, under the stated task-invariance assumption, the discarded syntax
cannot be decision-relevant in the population problem.

### Corollary 4.2: zero registered presentation premium

For a predictor `f`, target `y`, and presentation set `G`, define

```text
Pi_G(f;C,y) = sup_(g in G) L(f(gC),y) - L(f(C),y).
```

AF-QDM has `Pi_G=0` for every registered null orbit whenever the numerical
invariance contract holds exactly (or is bounded by the registered tolerance
for a Lipschitz loss). A non-quotiented model may have positive or negative
pointwise differences, but its worst-presentation premium is not certified to
vanish. This is the precise robustness estimand behind the orbit and
adversarial-envelope tables.

## 5. What is and is not a literature collision

| Closest area | Established contribution | Remaining AF-CDMO distinction |
|---|---|---|
| Deep Sets / Set Transformer | Permutation-invariant learning on sets | Does not derive the KKT/equality-gauge quotient, reciprocal dual-mass law, or split/merge conservation |
| Neural functions on measures | Generic learning on finite or probability measures | Does not identify the market certificate measure or prove its maximality for the declared rewrite relation |
| Optimization-preserving augmentation | Samples equivalent LP/QP formulations, including scaling transforms, to improve learning | Augmentation rather than an exact inference-time quotient of an already solved positive-dual certificate |
| EquivaMap | Discovers and verifies mappings between equivalent optimization formulations | Equivalence detection, not a neural factor map through a market-specific solved-certificate quotient |
| Feasibility projection / constrained neural outputs | Enforces output feasibility | Does not remove economically null input-certificate presentations or preserve their restricted dual functional |
| LP/QP graph neural networks | Learns objective values, solutions, or solver guidance with permutation-aware encodings | Supplied formulations remain sensitive to continuous row scaling, equality gauge, and refinement unless separately handled |
| Flow-based market and price models | Models PTDF-based coupling, dispatch, or zonal prices | Does not provide exact solved-certificate presentation assurance under the registered quotient |

Primary collision anchors:

- Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114).
- Lee et al., [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).
- Zweig and Bruna, [A Functional Perspective on Learning Symmetric Functions with Neural Networks](https://proceedings.mlr.press/v139/zweig21a.html).
- Qian and Morris, [Principled Data Augmentation for Learning to Solve Quadratic Programming Problems](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c07d71ff0bc042e4b9acd626a79597fa-Abstract-Conference.html).
- Zhai et al., [EquivaMap](https://proceedings.mlr.press/v267/zhai25a.html).
- Liang et al., [Low Complexity Homeomorphic Projection](https://proceedings.mlr.press/v202/liang23a.html).
- Chen et al., [Expressive Power of Graph Neural Networks for (Mixed-Integer) Quadratic Programs](https://proceedings.mlr.press/v267/chen25j.html).

No source in the documented search supplied the same end-to-end construction:
an already solved positive-dual market certificate, exact quotienting of row
order, positive scale, dual-mass refinement, and equality-row-space gauge,
followed by physical feasible-transfer price reconstruction. This supports
only the qualified phrase **"to the best of our documented search"**. It is
not proof of worldwide priority.

## 6. Empirical decision rule

The five methods form the following mechanism grid:

| Method | Row scale/order/refinement quotient | Global slack quotient | Four retained HVDC equality gauges | Learned residual |
|---|---:|---:|---:|---:|
| Ambient CQDM | yes | no | no | yes |
| Slack-QDM | yes | yes | no | yes |
| AF projected raw | no | yes | yes | yes |
| AF-QDM | yes | yes | yes | yes |
| Analytic gauge control | yes by linear conservation | yes | yes | no |

This grid tests whether a favorable AF-QDM result can be attributed merely to
projection, merely to CQDM row normalization, or merely to the analytic KKT
channel. Slack-QDM versus AF-QDM is the parameter-matched test of the added
rank-four internal-HVDC geometry. Projection-only versus AF-QDM is a diagnostic
test of whether the row-measure quotient remains necessary once full
projection is present; because those two arms use different pooling
architectures, it is not by itself a parameter-matched necessity proof.

The real v13 experiment supports the proposed contribution only if all of the
following hold:

1. AF-QDM satisfies its preregistered exact prediction-invariance tolerance.
2. Ambient and slack-only controls exhibit material drift under at least one
   retained internal-HVDC rewrite.
3. AF-QDM improves worst-presentation or timestamp-adversarial loss with a
   paired interval excluding zero.
4. The effect is not carried by one initialization seed.
5. Projection-only does not fully match AF-QDM on combined row/equality
   rewrites.
6. Frozen v11 clean controls reproduce numerically.

Failure of clean-form superiority is not fatal because it was not assumed.
Failure of items 1-3 is fatal to the real-market robustness claim and must be
reported as a null result.

## 7. Defensible novelty sentence

> We derive and implement a market-specific neural factorization through a
> positive-dual affine-feasible certificate quotient, preserving the solved
> certificate's restricted dual functional while making physical spread
> reconstruction exactly invariant to documented row and equality-gauge
> presentations.

Do not shorten this to "a new invariant neural network" or "the first
formulation-invariant optimizer"; both would exceed the evidence and collide
with established literature.
