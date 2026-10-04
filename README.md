# AF-CDMO: Representation-Safe Learning from Flow-Based Market Certificates

AF-CDMO is a research implementation of the **Affine-Feasible Certificate
Dual-Measure Operator** for representation-safe analysis of solved flow-based
market certificates, with Nordic and Core-CCR studies.

The empirical task reconstructs the same-delivery centered Nordic day-ahead
zonal price field from a solved flow-based certificate. It is **not** a
future-price forecast, a bidding strategy, or a trading-profit experiment.

## Why this repository exists

Equivalent encodings of the same market certificate can differ in row order,
constraint scaling, equality gauge, or refinement. A model that reacts to
those presentation choices can produce different prices or monitoring alerts
without any economic change in the underlying certificate.

AF-CDMO factors a certificate through a positive-dual affine-feasible quotient
before learning a residual map. The registered equivalence family covers:

- row permutation;
- positive row scaling with reciprocal dual scaling;
- declared affine equality-gauge and basis rewrites; and
- mass-conserving split/merge refinements.

The reusable structural model includes a full ambient-output projection. The
frozen Nordic experiment reported in the paper instead combines a negative
projected-dual analytic field with a centered ten-zone neural residual. Its
pairwise spreads therefore come from one cycle-consistent physical-zone field;
the empirical residual is not claimed to be a full 31-coordinate cotangent.
Exact invariance is treated as an executable contract, not merely as a training
objective.

## Scientific claim boundary

The repository supports a market-certificate representation-assurance and
monitoring contribution. The evidence does **not** establish:

- future-price forecasting superiority;
- operational welfare improvement or trading profitability;
- universal superiority over invariant analytic controls; or
- worldwide priority for generic invariant neural networks.

Projection-only and analytic controls are retained because clean-form accuracy
and representation assurance are separate questions. Generic linear
mass-weighting and refinement symmetry are prior work; the scoped contribution
is the market-derived affine quotient, conserved shadow-price mass, physical
output semantics, and end-to-end registered-rewrite assurance contract. The
public tests encode the exact quotient, structural projection, centered-output,
and registered-rewrite contracts.

## Repository layout

```text
af_cdmo/                         Core geometry, quotient, and neural models
scripts/                         Benchmarks, controls, auditors, and runners
tests/                           Synthetic theorem and invariance tests
af_cdmo_probabilistic_headroom/  Bounded selective-monitor extension
af_cdmo_learning_headroom/       Shared chronological residual-data utilities
.github/workflows/tests.yml      Public continuous-integration workflow
DATA_ACCESS_AND_RELEASE.md       Data authorization and release boundary
CORE_EXTERNAL_GENERALIZATION_PROTOCOL.md
                                Preregistered Core-CCR replication boundary
CONTRIBUTING.md                  Contribution and contract-preservation rules
SECURITY.md                      Credential and sensitive-data reporting policy
release_manifest.json            Generated SHA-256 allowlist for this release
```

The public package is deliberately code-only. Raw market records, trained
weights, timestamp-level derivatives, credentials, and local execution output
are excluded.

## Quick start

Python 3.10 is the reference environment.

```powershell
python scripts\verify_release_manifest.py --strict
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest tests -q
```

On Linux or macOS, create the environment with `python3.10 -m venv .venv`,
activate it with `source .venv/bin/activate`, and use forward slashes in script
paths. The GitHub Actions workflow is configured to test Windows and Linux;
it has not been executed remotely for this prepared release. The supplied
campaign wrappers require PowerShell; the Python entry points also expose
their arguments through `--help`.

The focused public suite contains deterministic geometry, theorem,
presentation-invariance, matched-control, and calendar-bootstrap tests. It
does not require restricted market data.

## Authorized real-data reproduction

Market data are not redistributed. The provider conditions checked for the
study are recorded in `DATA_ACCESS_AND_RELEASE.md`; obtain authorization for
the intended use before acquiring, using, or releasing source or derivative
artifacts.

The Nordic runner verifies the historical study and requires its authorized
prerequisite artifacts as well as the source data:

- the canonical dual archive and its `.metadata.json` SHA-256 sidecar;
- the ten-zone day-ahead price panel;
- `official_jao_dual_measure/af_cdmo_real_geometry_audit_v13.json`; and
- the development, market-design-shift, and confirmation
  `slack_qdm_extension_*_v11.json` reference reports.

These files are excluded from this package. Market CSVs alone do not satisfy
the historical verification contract. Once matching prerequisite artifacts
are supplied locally, run the three registered chronological roles:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_af_cdmo_real_v14.ps1" -StudyRole development
powershell -ExecutionPolicy Bypass -File ".\scripts\run_af_cdmo_real_v14.ps1" -StudyRole frozen_transition
powershell -ExecutionPolicy Bypass -File ".\scripts\run_af_cdmo_real_v14.ps1" -StudyRole confirmation
```

The runners enforce chronology, source and engine hashes, the ten-seed
registry, matched controls, exact invariance contracts, checkpoint integrity,
and fail-closed result manifests. `lastModifiedOn` is used only as a
conservative availability proxy, not as proof of first-publication time.

### Core-CCR external replication

The release includes an external replication for the public Core publication
schema. It reuses the neural architecture and main optimization settings while
adapting the chronology, zone set, ambient coordinates, and equality rank.
Core uses training batches of 128; the Nordic study uses 256. The Core raw and
projected-raw controls predict the price field directly, whereas AF-CDMO adds
a learned residual to the analytic field. Their accuracy difference compares
complete configurations and does not isolate a quotient-only advantage.
The Core construction deliberately uses only the globally balanced rank-one
geometry identifiable from public tables; it does not claim recovery of the
four technical equalities or ALEGrO equations.

The runner downloads public Core records, excludes delivery intervals still
revised after delivery, audits the geometry, fits ten chronological seeds, and
evaluates a 56-presentation orbit for the registered invariance seed:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_core_external_replication.ps1"
```

Core alert flips concern absolute predicted pairwise spreads above a fixed
5 EUR/MWh threshold and are averaged over timestamp-pair combinations. Nordic
alerts concern the largest reconstruction error per timestamp with
validation-calibrated thresholds; the percentages are not directly comparable.
Core seed-bootstrap accuracy intervals describe initialization variability
conditional on the observed chronology, not temporal sampling uncertainty.

Raw responses, derived market tables, checkpoints, and progress ledgers remain
local and are ignored by the public repository. The exact claim boundary and
failure conditions are registered in `CORE_EXTERNAL_GENERALIZATION_PROTOCOL.md`.

The code-only release also contains the registered five-seed selective-monitor
extension and its focused tests. Its result is bounded: it demonstrates stable
accept/escalate decisions under registered rewrites and held-out refinement,
but does not establish predictive superiority over invariant gradient
boosting. Reproduction of its frozen confirmation requires the authorized
market inputs and the matching frozen development artifact described in
`af_cdmo_probabilistic_headroom/README.md`.

## Release integrity

`release_manifest.json` records the byte length and SHA-256 digest of every
allowlisted file except the manifest itself. Verify an untouched checkout
before installation or tests create local metadata and caches:

```powershell
python scripts\verify_release_manifest.py --strict
```

After tests or editable installation, omit `--strict` to allow conventional
Python environment and cache metadata. Unexpected `build/` or `dist/` files
and restricted outputs hidden in `__pycache__/` still fail verification.

Maintainers rebuild the code-only package from the research workspace or an
unchanged public checkout with:

```powershell
python scripts\build_public_release.py --force
```

The builder audits a temporary staging tree before replacing the previous
generated release. It fails if it encounters disallowed data/model extensions,
absolute local user paths, or likely embedded credentials.

From the research workspace, prepare the local GitHub folder and its upload
archive after rebuilding:

```powershell
python scripts\package_public_release.py
```

This copies only manifest-managed files to `GitHub/AF-CDMO/`, preserves any
existing Git metadata, and writes `GitHub/AF-CDMO-code-only.zip` without runtime
caches. It performs no commit, push, or publication.

## Citation

Use the metadata in `CITATION.cff`. Add the article DOI and archival software
identifier when those identifiers are fixed.

## License and contributions

The code is licensed under Apache License 2.0; see `LICENSE`. Contributions
must preserve the exact-invariance contracts, causal data boundary, and public
release restrictions described in `CONTRIBUTING.md` and `SECURITY.md`.
