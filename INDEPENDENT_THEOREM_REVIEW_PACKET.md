# Independent Review Request: AF-CDMO Quotient Claims

This document is a checklist for an external reviewer. It is not a substitute
for independent review and should not be completed by the paper's authors.

## Material to review

1. `AF_CDMO_THEOREM_NOTE_V12.md`, especially Theorems 1-7 and Proposition 8.
2. `AF_CDMO_REAL_THEOREM_AND_COLLISION_NOTE_V13.md`, especially Propositions
   1-3 and Corollaries 4.1-4.2.
3. `af_cdmo/core.py` and `af_cdmo/models.py`.
4. `scripts/af_cdmo_real_extension_v13.py` and
   `scripts/af_cdmo_real_controls_v14.py`.
5. The focused tests under `tests/test_af_cdmo_*`.

## Questions requiring an explicit verdict

- Does full-column-rank equality re-presentation preserve the affine set,
  minimum-norm base point, tangent projector, and cotangent quotient as stated?
- Is `(u, beta, m, q)` a complete invariant for one retained positive-dual row
  under the declared positive-scale/equality-gauge relation?
- Does equality of the resulting finite positive measures characterize exactly
  the declared finite-certificate equivalence after coincident-atom merging?
- Does the quotient preserve the restricted Lagrangian functional on every
  feasible displacement with the stated sign convention?
- Are all physical price-spread queries used empirically tangent to the retained
  real-market equality manifold?
- Is the Bayes-factorization corollary stated with sufficient measurability and
  existence assumptions?
- Does the refinement-compatible attention theorem require any assumption not
  currently stated, particularly non-degenerate comparison atoms or arbitrary
  value-map separation?
- Are the conditioning bounds correct near the projected-row degeneracy
  boundary?
- Do the software tests cover the full declared relation, including redundant
  equality bases, nonuniform mass refinements, and non-equivalent negative
  controls?

## Claim discipline

The reviewer should reject any interpretation broader than the declared
equivalence relation. The current theory does not establish invariance to
arbitrary variable-coordinate transformations, arbitrary dual degeneracy,
topology changes, active-set changes, non-equivalent RAM changes, or every
possible optimization reformulation.

Please return a signed or attributable review note listing accepted statements,
required corrections, and any missing assumptions. Keep that note outside the
public repository until the reviewer consents to disclosure.

