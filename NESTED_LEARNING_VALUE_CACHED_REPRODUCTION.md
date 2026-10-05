# Reproduce the exploratory nested risk-control audit

`scripts/audit_paper2_nested_learning_value_portable_cached_v1.py` reproduces the complete four-control result grid from explicitly supplied, authorized feature caches. It acquires no provider data, reconstructs no raw certificates, and runs no neural model. Saved-model mode performs no fitting. Refit mode fits the same fixed historical HGB configuration and December Platt maps and requires exact reproduction of the reference results.

The controls are:

| Control | Exact frozen input |
| --- | --- |
| `field12` | log1p projected mass, presence, ten centered physical-zone analytic-current coordinates |
| `analytic_current33` | mass, presence, all 31 uncentered projected negative-current coordinates reconstructed as `-expm1(features[:,0])*features[:,12:43]` |
| `first_moments48` | first 48 original columns: the 12 summaries plus 36 normalized mass-weighted canonical first moments |
| `quotient84` | all 84 original columns, adding 36 normalized mass-weighted squared canonical moments |

Every pair, Brier/log-loss/AUC comparison, 7/14/28 calendar-day interval, fixed December cutoff, daily 10/20/30% budget and global 20% budget is retained. Daily budgets use `ceil(budget * timestamps_in_day)`; global 20% uses `floor(0.20 * total_timestamps)`. These rules differ. Rankings are label-free completed-batch policies with chronological ties; they are not online arrival-by-arrival policies.

## Required private inputs

Feature NPZ files for historical fit, December calibration, original confirmation, and the already-consumed later period must contain `features` (finite float64, 84 columns), `target` (the exact original binary labels), `residual_magnitude_eur_mwh`, and ordered unique UTC `timestamp_utc`. Provide the historical cache manifest, the later feature-audit report, original nested protocol/report, and its two private prediction CSVs. Saved-model mode additionally requires an explicitly named trusted joblib for each control. These inputs and fitted models are excluded from the public repository.

NPZ timestamps must be fixed Unicode, so every public load retains `allow_pickle=False`. Three original owned caches stored timestamps as object arrays. Those originals were preserved; separate private copies changed timestamp serialization only after their original executed SHA hashes were checked. Every numerical array and UTC timestamp was verified identical. For such annotated normalized caches, additionally supply `--cache-origin-manifest` and `--later-origin-feature-audit`, identifying the preserved original manifests/audits. The public adapter validates their linkage and never enables generic NPZ object loading.

The historical feature cache was generated from the original frozen `RiskDataset` items: the exact float32-derived `tail_target` is retained, while the fixed `quotient_feature` maps canonical atoms, normalized weights, projected total mass and ordered physical-zone indices into float64 features. Fit ends before 1 November 2025; probability calibration uses December 2025. November and January–February outcomes are not used to select these controls. Raw acquisition sources, geometry, checkpoint scales and original script hashes are recorded in the historical protocol. The later cache applies those original scales to the previously examined 20 August–30 September 2026 cohort, with no later fit or calibration.

Do not reconstruct fit labels from exported float64 magnitudes: the historical computation uses float32 comparisons, and a boundary value can differ. Do not replace missing metadata or current coordinates with a new normalization. Authorized cache regeneration should use the original frozen dataset/feature construction and preserve its exact labels, timestamps, geometry/schema, scales and source provenance. This cache-only program verifies those prepared hashes; it does not reconstruct or certify a fresh provider export.

## Execution

Install the project's numerical dependencies. The cache-only refit path uses NumPy, pandas, scikit-learn, joblib and threadpoolctl; it has no torch dependency. Historical saved-model bundles contain the original `PlattMap` class and require the original package module to deserialize. New portable refit bundles store plain calibration coefficients.

Run the meaningful arithmetic tests first:

```text
python scripts/audit_paper2_nested_learning_value_portable_cached_v1.py --self-test
```

Supply paths using the following command, first with `--protocol-only` appended and then without it. `private` and `work/nested_reproduction` are example directories chosen by the caller.

```text
python scripts/audit_paper2_nested_learning_value_portable_cached_v1.py --mode saved-models --fit-cache private/fit_features.npz --calibration-cache private/calibration_features.npz --confirmation-cache private/confirmation_features.npz --later-cache private/later_features.npz --cache-manifest private/feature_cache_manifest.json --later-feature-audit private/hgb_feature_audit.json --reference-protocol private/nested_protocol.json --reference-report private/nested_audit.json --confirmation-predictions private/confirmation_predictions.csv.gz --later-predictions private/later_predictions.csv.gz --model field12=private/field12.joblib --model analytic_current33=private/analytic_current33.joblib --model first_moments48=private/first_moments48.joblib --model quotient84=private/quotient84.joblib --output work/nested_reproduction
```

For cache-only fitting, use `--mode refit` and omit all four `--model` options. Use a different output directory and freeze its execution plan separately. Fixed fitting parameters and bootstrap grids are inherited; no outcome-driven choice is exposed. Library changes can prevent exact reproduction; the program stops if historical probabilities, Platt maps, cutoffs or the complete reference metric grid differ beyond the stated numerical checks.

The program requires the output directory to differ from every source directory, freezes an execution plan before scoring, verifies every supplied cache/model against the historical manifest/report, and rechecks all source hashes after analysis. It writes a new private aggregate report; raw feature records and predictions are not copied into the public project.

## Interpretation

Both evaluation periods were already examined before the additional controls. This is post-confirmation exploratory evidence with unadjusted comparisons. A result under the fixed HGB estimator can reflect approximation and calibration as well as available information. It does not establish Bayes sufficiency, neural necessity, neural superiority, an observed price outage, monetary savings, or prospective monitoring benefit.

`scripts/paper2_nested_cached_metrics.py` preserves all 18 original numerical function/class ASTs. The independent tests verify complete-current/physical-field identity, exact feature mapping, empty support, weighted AUC with ties and one-class draws, missing calendar days, and chronological ranking ties. The historical private AST provenance and complete run verification remain in the submission-preparation workspace.
