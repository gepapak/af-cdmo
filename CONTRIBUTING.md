# Contributing

Contributions are welcome when they preserve the scientific and release
contracts of this repository.

## Before opening a change

1. Do not commit raw or transformed provider data, credentials, authorization
   records, trained weights, timestamp-level outputs, logs, or process files.
2. Keep chronology and information-availability assumptions explicit.
3. Do not weaken or bypass exact presentation-invariance checks.
4. Add a focused regression test for every behavioral change.
5. Separate new scientific claims from engineering refactors.

## Local checks

```powershell
python -m pip install -e ".[test]"
python -m pytest tests -q
python scripts\build_public_release.py --force
python public_release\af_cdmo_code_only\scripts\verify_release_manifest.py --strict
python scripts\package_public_release.py
```

The generated folder and ZIP contain only the release allowlist. Rebuild and
sync the manifest after changing a public source file; CI verifies the manifest
before installing dependencies. In a public checkout, the builder uses the
existing `README.md` as its README source. Historical Nordic and QCT guards
require the authorized prerequisite artifacts documented in the READMEs;
source market records alone do not recreate every frozen audit reference.

Changes to the quotient, projection, rewrite registry, or theorem assumptions
also require updating the relevant theorem note and an independent
mathematical review before those changes are used in a manuscript claim.

## Reporting results

Report all registered seeds and controls. Do not select presentations, dates,
or seeds after seeing outcomes. Report unresolved comparisons and null results
faithfully, and do
not describe `lastModifiedOn` as definitive first-publication provenance.
