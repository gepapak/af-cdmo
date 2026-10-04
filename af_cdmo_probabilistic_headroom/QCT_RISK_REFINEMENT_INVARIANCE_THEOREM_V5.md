# QCT-Risk market refinement-invariance specialization

Date: 2 October 2026

Status: mathematical implementation note for the isolated QCT-Risk extension.
It does not change the frozen AF-CDMO study or the registered QCT-Risk
confirmation protocol.

Generic refinement symmetry of positive-kernel attention is prior art. In
particular, Du and Chen (2026), *Refinement Symmetry in Multimodal
Transformers*, prove that linear local mass weighting is forced by split
invariance under their stated assumptions:
https://arxiv.org/abs/2609.32669. The result below is therefore not claimed as
a new generic attention theorem. It is the market-specific specialization that
links conserved shadow-price mass, the AF-CDMO economic quotient, the
gauge-tangent output, and the final assurance decision.

## 1. Certificate measure and declared rewrites

After AF-CDMO canonicalization, a positive-dual solved certificate is represented
by the finite measure

\[
  \mu = \sum_{i=1}^{n} m_i\,\delta_{x_i}, \qquad m_i > 0,
\]

where `x_i` is a canonical affine-feasible constraint atom and `m_i` is its
dual mass. A mass-conserving refinement replaces one atom
`m delta_x` by finitely many identical atoms

\[
  m\,\delta_x \longmapsto \sum_{k=1}^{K} m_k\,\delta_x,
  \qquad m_k > 0,\quad \sum_k m_k=m.
\]

This rewrite changes the row-level presentation but not the finite measure.
Permutation and the inverse merge operation are included automatically.

## 2. Dual-mass attention

For a query state `h_i`, QCT-Risk uses the normalized integral

\[
 A_i(\mu)
 =
 \frac{\int \exp\!\left(\langle q(h_i),k(h(x))\rangle/\sqrt d\right)
                 v(h(x))\,d\mu(x)}
        {\int \exp\!\left(\langle q(h_i),k(h(x))\rangle/\sqrt d\right)
                 \,d\mu(x)}.
\]

The implemented finite-sum form is exactly the numerator
`sum_j exp(score_ij) m_j v_j` divided by
`sum_j exp(score_ij) m_j`. Learned-query pooling has the same form. The raw
atom moments are also measure integrals normalized by total dual mass.

## 3. Theorem

**Proposition 1 (market-certificate specialization of exact mass-refinement
invariance).** Assume that:

1. canonicalization maps every registered economically equivalent row rewrite
   to the same atom location and conserved dual mass;
2. atom encoders and pointwise feed-forward maps are shared across atoms;
3. every atom-to-atom and learned-query attention normalization is taken with
   respect to dual mass as above;
4. global summaries consist only of finite-measure integrals, total mass,
   presence, and other quotient-invariant fields; and
5. inference is deterministic with dropout disabled (`model.eval()`), and
   trained parameters, normalization scales, calibration thresholds, and
   external context are fixed across the compared presentations.

Then every layer of QCT-Risk, its pooled latent state, its centered tangent
residual, its tail probability, and every deterministic threshold decision are
invariant under arbitrary finite mass-conserving split/merge refinements and row
permutations.

**Proof.** A refinement does not change `mu`. Therefore, for any unchanged query,
the contributions of the refined copies to the attention numerator sum to

\[
  \sum_k m_k e^{s(x)}v(x)=m e^{s(x)}v(x),
\]

and their contributions to the denominator sum to

\[
  \sum_k m_k e^{s(x)}=m e^{s(x)}.
\]

Thus the attention output is unchanged. All copies have the same initial state,
receive the same measure-integrated message, and pass through the same pointwise
map, so they remain identical. Induction proves the statement for every
interaction block. The same finite-measure argument proves invariance of
learned-query pooling and first and second moments. Concatenating invariant
summaries and applying deterministic shared maps preserves invariance. Finally,
centering and deterministic probability thresholding preserve equality of the
outputs. Row permutation changes only summation order. QED.

## 4. Corollaries for the market certificate

**Corollary 1 (registered presentation invariance).** If AF-CDMO
canonicalization is invariant to positive row scaling with reciprocal dual
scaling and to additions from the declared market-balance equality row space,
QCT-Risk factors through the same market-certificate quotient. Its outputs are
therefore constant over every registered certificate presentation orbit.

**Corollary 2 (price-gauge feasibility).** For an unconstrained network output
`y` over `Z` zones, QCT-Risk returns

\[
  r = y - \frac{\mathbf 1^T y}{Z}\mathbf 1.
\]

Hence `1^T r = 0` exactly in real arithmetic. All pairwise residual spreads are
antisymmetric and satisfy cycle consistency because they are differences of one
centered zonal field.

**Corollary 3 (decision invariance).** If `rho(mu)` is the invariant tail-risk
probability and `c` is a threshold fixed from the calibration block, then
`1[rho(mu)>c]` is invariant to every registered rewrite. The selective action
cannot be changed by a presentation-only modification of the certificate.

## 5. Why uniform-token attention is not sufficient

Ordinary token attention integrates against the empirical counting measure
`sum_i delta_{x_i}`. Duplicating one row changes that measure even when the
economic dual mass is divided among the copies. Its softmax denominator and
generally its output therefore change. Permutation invariance alone does not
imply mass-refinement invariance.

This distinction is tested directly in
`tests/test_qct_risk_confirmation_v5.py`: the mass-weighted QCT and Deep Sets
controls remain invariant under nonuniform refinement, while the ordinary Set
Transformer is an intentionally non-invariant negative control.

## 6. Scope and non-claims

- The result is an architectural contract, not evidence of predictive skill.
- It applies to identical-atom refinements with conserved positive dual mass;
  it does not claim invariance to economically different mass relocation.
- Finite-precision implementations are evaluated with explicit numerical
  tolerances; the mathematical statement is exact in real arithmetic.
- Generic set attention, finite-measure networks, refinement-symmetric
  attention, selective prediction, and tangent projection have prior art. The
  defensible contribution is their market-specific derivation from the solved-
  certificate quotient and the exact coupling of economic representation
  invariance to the assurance decision.

## 7. Empirical falsification contract

The confirmation report must show all of the following independently of this
proof:

1. zero decision flips across registered equivalent rewrites;
2. negligible numerical drift under held-out nonuniform refinement;
3. nonzero response to non-equivalent RAM perturbations or mass relocation; and
4. useful untouched-period discrimination and risk-coverage behavior against
   invariant classical controls.

If item 4 fails, the theorem remains true but the learned assurance layer is not
empirically useful and must stay out of the paper's main contribution set.
