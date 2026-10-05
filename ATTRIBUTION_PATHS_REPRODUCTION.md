# Two-path attribution integrity reproduction

This optional audit replays an already examined numerical integrity endpoint
with explicit authorized inputs. It compares original shadow price times
reconstructed PTDF against stored normalized dual mass times unit direction.
Both routes project through the original declared rank-five geometry and form
the same ten-zone centered analytic field. The complete original 14,482 native
quarter-hour confirmation timestamps, all 45 pairs, all eligible rows and all
recorded TSO provenance tags are retained. No prices, outcome labels, learned
scores, fitted models, neural inference or network access are used.

Keep `scripts/audit_paper2_attribution_paths_portable_v1.py` and
`scripts/paper2_attribution_paths_metrics.py` together. The original private
primary and descriptive engines remain unchanged. The portable kernel preserves
the executed archive-loop and primary-endpoint numerical bodies; descriptive
zero-screen and margin results are a separate output rather than an amendment
to the primary count. No introspection or profiling hook is required.

```powershell
python scripts/audit_paper2_attribution_paths_portable_v1.py --self-test
python -m pytest tests/test_attribution_paths_portable.py
$auditArgs = @(
  '--dual-archive', 'AUTHORIZED/certificate.csv.gz',
  '--geometry-audit', 'AUTHORIZED/geometry.json',
  '--timestamp-csv', 'AUTHORIZED/original_confirmation_timestamps.csv.gz',
  '--frozen-risk-report', 'AUTHORIZED/quotient_tangent_risk_confirmation_v5.json',
  '--output', 'PRIVATE_NEW_OUTPUT'
)
python scripts/audit_paper2_attribution_paths_portable_v1.py @auditArgs --plan
python scripts/audit_paper2_attribution_paths_portable_v1.py @auditArgs --run
```

The timestamp CSV needs only `timestamp_utc`. The archive needs
`delivery_utc`, `publication_utc`, `tso`, `shadowPrice`, `ptdf_l2_norm`,
`dual_mass_eur_mwh` and every named `unit_ptdf_<zone>` coordinate from
`geometry.zone_order`. Retained rows must have finite positive shadow price,
norm and mass. The availability proxy is publication metadata at least 60
minutes before delivery; it is not independently logged first publication.
The archive and geometry hashes must match the supplied frozen risk report.
The full chronological cohort and native quarter-hour grid are checked.
Source hashes and the replay procedure are frozen before the archive pass;
any amendment requires a new output directory.

If the original attribution reports are authorized and available, add both
`--reference-primary AUTHORIZED/attribution_path_audit.json` and
`--reference-detail AUTHORIZED/routing_detail_audit.json` to **both** commands.
This mode checks exact original archive/geometry/timestamp input bytes and
requires exact agreement of every original primary scalar, every descriptive
count, margin quantile and tag transition. A separately exported timestamp-only
CSV can reproduce the numerical procedure but will not match the original
full prediction CSV's byte hash for that strict reference mode.

The primary decision is the tag with the largest absolute signed component,
with a lexicographic tie break, on route-A nonzero opportunities. Its frozen
count excludes opportunities that are exactly zero in route A. The descriptive
branch separately reports both exact-zero screens and membership differences,
all routing flips, their component/margin/drift quantiles, and certified margins
greater than twice either local or global observed score drift. No tolerance
is selected to remove inconvenient near ties or zero-screen changes.

The completed original audit retains 87,600 rows. It reports 280 primary
near-zero routing flips and 3,633 exact-zero-screen changes; no observed routing
flip satisfies the measured two-path margin certificate. These negative
outcomes remain visible in both replay outputs. Recorded TSO tags identify
archive provenance, not causal ownership or welfare responsibility. Shared
archive agreement is not independent acquisition validation, an observed
same-delivery publication revision, a human-benefit study, or neural added
value. Provider records and generated timestamp-level fields remain private
and are not packaged with this code.
