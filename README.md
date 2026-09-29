# AF-CDMO: Representation-Safe Learning from Flow-Based Market Certificates

AF-CDMO is a research implementation of the **Affine-Feasible Certificate
Dual-Measure Operator** for representation-safe analysis of solved Nordic
flow-based market certificates.

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

The output is a feasible cotangent representative of the conserved dual
functional. Exact invariance is treated as an executable contract, not merely
as a training objective.

## Scientific claim boundary

The repository supports a market-certificate representation-assurance and
monitoring contribution. The evidence does **not** establish:

- future-price forecasting superiority;
- operational welfare improvement or trading profitability;
- universal superiority over invariant analytic controls; or
- worldwide priority for generic invariant neural networks.

Projection-only and analytic controls are retained because clean-form accuracy
and representation assurance are separate questions. See
`SUBMISSION_READINESS.md` and the theorem/collision notes for the precise claim.

## Repository layout

```text
af_cdmo/                         Core geometry, quotient, and neural models
scripts/                         Benchmarks, controls, auditors, and runners
tests/                           Synthetic theorem and invariance tests
.github/workflows/tests.yml      Public continuous-integration workflow
DATA_ACCESS_AND_RELEASE.md       Data authorization and release boundary
SUBMISSION_READINESS.md          Scientific and external readiness checklist
release_manifest.json            Generated SHA-256 allowlist for this release
```

The public package is deliberately code-only. Raw market records, trained
weights, timestamp-level derivatives, credentials, and local execution output
are excluded.

## Quick start

Python 3.10 is the reference environment.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts\verify_release_manifest.py
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest tests -q
```

The focused public suite contains deterministic geometry, theorem,
presentation-invariance, matched-control, and calendar-bootstrap tests. It
does not require restricted market data.

## Authorized real-data reproduction

Market data are not redistributed. JAO currently requires prior authorization
for ML/AI use of its content. Read `DATA_ACCESS_AND_RELEASE.md` before acquiring,
using, or releasing any source or derivative artifact.

An authorized user supplies the expected local inputs under the ignored data
directories and runs the three preregistered chronological roles:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_af_cdmo_real_v14.ps1" -StudyRole development
powershell -ExecutionPolicy Bypass -File ".\scripts\run_af_cdmo_real_v14.ps1" -StudyRole frozen_transition
powershell -ExecutionPolicy Bypass -File ".\scripts\run_af_cdmo_real_v14.ps1" -StudyRole confirmation
```

The runners enforce chronology, source and engine hashes, the ten-seed
registry, matched controls, exact invariance contracts, checkpoint integrity,
and fail-closed result manifests. `lastModifiedOn` is used only as a
conservative availability proxy, not as proof of first-publication time.

## Release integrity

`release_manifest.json` records the byte length and SHA-256 digest of every
allowlisted file except the manifest itself. Verify an untouched checkout
before installation or tests create local metadata and caches:

```powershell
python scripts\verify_release_manifest.py
```

Maintainers rebuild the code-only package from the private research workspace
with:

```powershell
python scripts\build_public_release.py --force
```

The builder fails if it encounters disallowed data/model extensions, absolute
local user paths, or likely embedded credentials.

## Citation

Use the metadata in `CITATION.cff`. Add the article DOI and archival software
identifier when those identifiers are fixed.

## License and contributions

The code is licensed under Apache License 2.0; see `LICENSE`. Contributions
must preserve the exact-invariance contracts, causal data boundary, and public
release restrictions described in `CONTRIBUTING.md` and `SECURITY.md`.
