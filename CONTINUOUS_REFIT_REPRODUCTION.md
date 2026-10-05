# Portable continuous-control refitting

`scripts/audit_paper2_continuous_refit_portable_v1.py` genuinely fits the fixed
continuous controls from authorized numeric caches. It supports:

- `--mode ridge`: fit four StandardScaler-plus-Ridge multioutput residual maps,
  comparing them with the supplied original four HGB prediction caches.
- `--mode hgb`: refit all 40 original HGB residual regressors and check their
  fields against the original prediction caches.
- `--mode both`: refit both estimator families and compute the complete
  nine-method, 36-pair comparison grid.

The program requires no neural inference, neural weights, or provider download.
Provider data, targets, feature caches, fitted models, and historical reports
are excluded from the code-only release. Obtain authorized private inputs
independently. All paths below may be relocated. Each invocation reads numeric
NPZ files with pickle disabled and validates input hashes and timestamp order.

The required arguments are:

```text
--fit-features FIT_FEATURES.npz
--fit-targets FIT_TARGETS.npz
--confirmation-features CONFIRMATION_FEATURES.npz
--confirmation-targets CONFIRMATION_TARGETS.npz
--later-features LATER_FEATURES.npz
--later-targets LATER_TARGETS.npz
--feature-provenance CONVERSION_PROVENANCE.json
--continuous-manifest CONTINUOUS_INPUTS_MANIFEST.json
--confirmation-hgb-predictions ORIGINAL_CONFIRMATION_PREDICTIONS.npz
--later-hgb-predictions ORIGINAL_LATER_PREDICTIONS.npz
--hgb-reference-protocol ORIGINAL_HGB_PROTOCOL.json
--hgb-reference-report ORIGINAL_HGB_REPORT.json
--output NEW_OUTPUT_DIRECTORY
```

First run the desired command with `--protocol-only`, then repeat the identical
command without it. A new output directory is required for changed inputs,
software, or mode. The optional `--preserve-manifest` checks additional previous
artifact hashes before and after fitting. No existing input file is modified.

Feature caches contain numeric `features` (84 columns), `target`,
`residual_magnitude_eur_mwh`, and fixed-width Unicode `timestamp_utc`. Target
caches contain ten-zone `target_field`, `analytic_field`, `residual_field`, and
the same timestamps. Target and analytic fields must align exactly; residuals
must equal the centered target-minus-analytic field. The trusted target's
analytic-error magnitude must match the original risk cache within the
previously declared 0.001 EUR/MWh tolerance. Feature conversion provenance
links unchanged numeric arrays and timestamps to the original source hashes
recorded by the HGB report. The continuous input manifest records the reviewed
target/prediction cache hashes and original HGB report hash.

The four feature controls use 12 selected physical-field features, the complete
31-coordinate current plus mass/presence (33), all first moments (48), and first
plus squared moments (84). All estimators fit the same 20,073 timestamps before
November 2025. Ridge has fixed alpha1 and an intercept; its StandardScaler fits
only on these training rows. HGB settings are fixed at 220 iterations, learning
rate.05, 15 leaves, minimum leaf size50, L2 penalty3 and seed20261002. No
continuous calibration or later fitting is performed. Every predicted field
is centered before all 45 unordered physical-zone differences are formed.

Outputs include private fitted models and fields, mean pairwise MAE, p90 of
per-timestamp mean absolute pair error, pairwise RMSE, and all paired 7/14/28-day
calendar-block intervals with 2,000 shared draws. Calendar gaps remain in the
resampling grid; the p90 is the exact linear quantile under integer repeated
sample weights. Matched ridge-minus-HGB contrasts are also retained. A refit on
a different software platform need not reproduce model bytes; the HGB field
agreement guard permits at most1e-8 EUR/MWh against the supplied original cache
and reports the actual maximum discrepancy.

The initial ridge-v1 protocol was frozen before its new ridge outcomes. Both
evaluation periods and HGB outcomes had already been examined. Subsequent
portable validation is a reproduction, not fresh confirmation. All intervals
are exploratory and unadjusted; the original neural reconstruction chronology
differs and is not a matched comparator. A fixed ridge failure does not prove
universal nonlinear or neural necessity. Full current33 is a deterministic
nonlinear transform of quotient84; the linear model spans are not nested
because logmass and normalized-direction coordinates do not directly supply
their mass-times-direction product.

Run `python scripts/audit_paper2_continuous_refit_portable_v1.py --self-test`
for the deterministic arithmetic checks, and
`python -m pytest tests/test_continuous_portable_refit.py -q` for safe-cache
rejection, exact identity, UTC-gap, weighted-quantile, and fit-only scaler tests.
