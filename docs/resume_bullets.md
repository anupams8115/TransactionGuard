# Resume bullets

Each bullet describes implemented, tested work, and the numbers are measured. Pick 3–4. Adjust
the wording, but keep the numbers and qualifiers (for example "on a laptop", "with CI").

- Built **TransactionGuard**, an explainable transaction-risk system (FastAPI, PostgreSQL/SQLAlchemy,
  Celery/Redis, Next.js/TypeScript), that scores transactions, explains each score with SHAP, and
  routes alerts to an analyst review queue with role-based access and workspace isolation.
- Trained and compared Logistic Regression, XGBoost and Isolation Forest on the public ULB
  credit-card dataset with a **chronological split**. Reached test average precision of
  **0.79 (95% CI 0.69–0.87)** against a 0.0013 baseline, and reported via **paired bootstrap** that
  the gap to Logistic Regression was not statistically settled.
- Prevented data leakage with past-only account-history features (vectorised `searchsorted`,
  143k rows in 0.3 s), train-only preprocessing and validation-only tuning, calibration and
  thresholds, **verified by a test that shuffles test labels and asserts that no training decision
  changes**.
- Calibrated probabilities with Platt scaling (test ECE 0.0002) and derived **exact SHAP
  contributions on the calibrated log-odds scale**, with a per-prediction additivity check that
  refuses to show invalid explanations.
- Made scoring **idempotent under retries and races** using database unique constraints, chunked
  commits and bounded Celery retries. Covered crash-and-retry and duplicate-submission cases in
  95 automated tests plus a Playwright browser test.
- Added MLflow experiment tracking (dataset fingerprint, split, seed, threshold, code revision) and
  **Evidently** drift and data-quality monitoring. Found and fixed an inverted drift-direction bug
  by cross-checking against Evidently's aggregate counts.
- Measured end-to-end API latency at **~41 ms p50 / 50 ms p95** on a laptop, and diagnosed a SQLite
  single-writer bottleneck under concurrency. Cut demo-data loading time about 10× by fixing
  quadratic ORM flush overhead.
