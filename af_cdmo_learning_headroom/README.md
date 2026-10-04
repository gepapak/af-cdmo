# AF-CDMO Learning-Headroom Audit

This folder contains isolated exploratory work and the chronological data
utilities used by the QCT extension. The public package includes its source
code; it does not include its local inputs or output artifacts. Running the
audit does not modify the frozen AF-CDMO models or confirmation report.

The audit tests whether the centered price-field residual left after the analytic
KKT cotangent contains chronologically learnable signal in:

- invariant certificate moments;
- delivery-calendar context; and
- strictly lagged previous-day and previous-week market residuals.

The chronology is fixed to fit before 2025-11-01, select during November and
December 2025, and evaluate during January and February 2026. The registered
confirmation period beginning 2026-03-01 is not read.

With authorized Nordic market data and the geometry audit described in the
main README supplied locally, run from the repository root:

```powershell
python af_cdmo_learning_headroom\audit_residual_headroom.py
```

The purpose is falsification. A new neural residual is justified only if a simple
control shows positive paired block-bootstrap skill over the analytic KKT field.
