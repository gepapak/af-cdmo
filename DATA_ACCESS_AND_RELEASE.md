# Data Access and Public-Release Policy

## Default release boundary

This project must be released as **code-only** unless the data providers grant
written permission covering the intended use and redistribution. The default
public package therefore excludes:

- JAO source archives and metadata derived from row-level JAO records;
- ENTSO-E source archives;
- model checkpoints trained from those archives;
- resumable evaluation states;
- timestamp-level loss archives; and
- credentials, tokens, local authorization records, logs, and process files.

The code-only release can include the model implementation, theorem notes,
tests, synthetic checks, schema documentation, and scripts that operate on
data supplied by an authorized user.

## JAO research and publication scope

Checked on 2026-10-05: JAO defines Content to include derivations. Section B.2
prohibits ML/AI/advanced-statistical use without prior authorization. This clause
appears under restricted-area provisions and does not itself specify *written*
authorization. Section A.4 separately requires advance written authorization
for dissemination and extraction, except for Legitimate Use or other permission
under the terms. Source credit is required, and tokens must remain confidential.
[JAO General Terms and Conditions, Definitions, A.3–A.4 and B.2](https://www.jao.eu/terms-conditions)

The applicability of the restricted-area clause to the public Nordic/Core
records and completed academic analyses needs clarification. Public access
alone does not establish the necessary use or publication rights. Resolve the
scope of academic ML use, aggregate article statistics/figures, attribution,
and any derivative release with JAO's licensing contact, `contact@jao.eu`.
The current request excludes raw records, trained weights, and timestamp losses.

Retaining written clarification is this project's evidence policy; it is not
a verbatim requirement of the ML clause. Retain an applicable permission or
existing-rights record privately under `jao_authorization_private/`. No such
record has been verified here. The code-only boundary controls redistribution
and does not establish permission for the underlying research use.

## ENTSO-E price-data rights and attribution

Checked on 2026-10-05: the current legal page links the 2023 platform terms and
18 October 2023 free-reuse list. The list does not enumerate Article 12.1.d
day-ahead prices; it therefore does not establish CC BY 4.0 coverage for the
Nordic price labels. Their primary owners are power exchanges or TSOs.
[Legal documents](https://transparencyplatform.zendesk.com/hc/en-us/articles/40921911218961-Legal-Terms-and-Conditions),
[free-reuse list](https://transparencyplatform.zendesk.com/hc/en-us/article_attachments/40921869379729),
[Energy Prices 12.1.d](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647234190100-Energy-Prices-12-1-D)

Clause 3.1 requires platform attribution, no implied endorsement, and prior
agreement from the primary rights holder where reuse risks prejudicing copyright
or related rights. The reviewed terms contain no explicit ML/AI prohibition.
Clarify the applicable rights for academic analysis and aggregate publication;
do not assume an unrestricted price-data licence or a blanket written-ML-
authorization requirement. ENTSO-E publishes `transparency@entsoe.eu` for
platform enquiries.
[Platform terms, clauses 2.5 and 3.1](https://transparencyplatform.zendesk.com/hc/en-us/article_attachments/40921869376401),
[contact](https://www.entsoe.eu/data/transparency-platform/data-providers/)

Credit JAO's Publication Tool for Nordic/Core certificates and Core spreads,
and ENTSO-E's Transparency Platform for Nordic price labels. State that quotient
transforms and aggregate analyses are the authors' work and imply no provider
endorsement. Do not assert permission or invent data-acquisition dates.

## Token security

Any token pasted into chat, a terminal transcript, email, or another third-party
system must be treated as exposed. Revoke or rotate it at the issuing service.
Use an environment variable for future access and never place a token on a
command line or in a committed file.

For ENTSO-E acquisition:

```powershell
$env:ENTSOE_API_TOKEN = "<new token>"
```

Clear it after use:

```powershell
Remove-Item Env:ENTSOE_API_TOKEN
```

## Reproducibility with authorized data

An authorized reproducer supplies the expected source files locally under:

- `official_jao_dual_measure/`
- `official_entsoe_nordic_day_ahead/`
- `external_core_replication/` for locally acquired Core source and derived
  tables, download ledgers, and evaluation artifacts.

The fail-closed auditors verify source SHA-256 hashes, chronology, method and
seed registries, trained-checkpoint hashes, exact invariance contracts, and
the aggregate report. The Nordic historical runner also requires the geometry
audit and frozen v11 reference reports documented in `README.md`; QCT exact
confirmation requires its matching frozen development JSON. These authorized
prerequisite artifacts are excluded from the package. Raw data are not needed
in the public Git repository.

## Availability-time limitation

The local canonical JAO archive maps the API field `lastModifiedOn` to
`publication_utc`. Public API documentation does not establish that this field
is always the first time a record became available to every market participant.
The manuscript must therefore describe it as a **conservative availability
proxy** and include the corresponding limitation. It must not claim that the
field provides definitive first-publication or information-set provenance.

## Release command

Build the allowlisted code-only package with:

```powershell
python scripts/build_public_release.py --force
python public_release/af_cdmo_code_only/scripts/verify_release_manifest.py --strict
python scripts/package_public_release.py
```

The builder rejects forbidden binary/data extensions, absolute local paths,
and likely embedded credentials before writing its SHA-256 manifest.
