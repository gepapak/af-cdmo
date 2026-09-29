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

## JAO authorization is an external submission blocker

As checked on 2026-09-28, the JAO General Terms and Conditions define content
broadly and prohibit using the content or information systems to train machine
learning or AI systems without prior authorization. They also require access
tokens to remain confidential. See:

https://www.jao.eu/terms-conditions

Before public release or journal submission, the author must obtain and retain
written clarification from JAO covering at least:

1. academic ML/AI use of the Nordic flow-based publication data;
2. publication of aggregate statistics and figures;
3. redistribution, or the absence of redistribution, of raw and transformed
   records;
4. publication of trained weights or timestamp-level derivatives; and
5. the citation and attribution text JAO expects.

An authorization record is intentionally kept outside version control under
`jao_authorization_private/`. The repository cannot create or infer this legal
permission.

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

The fail-closed auditors verify source SHA-256 hashes, chronology, method and
seed registries, trained-checkpoint hashes, exact invariance contracts, and
the aggregate report. Raw data are not needed in the public Git repository.

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
```

The builder rejects forbidden binary/data extensions, absolute local paths,
and likely embedded credentials before writing its SHA-256 manifest.

