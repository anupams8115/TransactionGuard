# 5-minute demo script

Setup: API on :8000, dashboard on :3000, demo data seeded. Sign in as `admin@demo.example`.

**0:00 · The problem (Overview page).** "Fraud is rare, about 0.13% of transactions here, so
analysts can't look at everything. TransactionGuard ranks transactions so they review the riskiest
ones first, and it explains each score. These are risk signals, not verdicts." Point at the purple
banner: synthetic data is always labelled.

**0:45 · Score one transaction (Score transaction).** Click *Fill with an example*, then *Score
transaction*. "The form is generated from the data contract, so it asks for exactly what the model
uses. The score is a **calibrated** probability, compared with a threshold chosen on validation
data." Walk through the explanation bars: "These are SHAP contributions on the log-odds scale.
They add up exactly to the score, and the system checks that. They describe the model, not causes."

**1:45 · Trace (Open full trace and review).** "Every prediction records its model version,
preprocessing version, threshold policy, calibration parameters, training run and the bundle's
SHA-256, so I can reproduce any decision." Set the status to *Resolved* without an assessment and
show the error, then pick *Inconclusive*, add a note and save. "Analyst opinions are stored apart
from confirmed labels and never become training data automatically."

**2:45 · Batch (Batch upload).** Upload a CSV that contains one bad row. "Rows are validated
against the contract and the bad one is reported, not guessed. The job runs on Celery, and
re-uploading the same file returns the same job."

**3:30 · Models.** "Chronological split; the test set was touched once. XGBoost won on
validation, but look at the bootstrap interval: the gap to Logistic Regression includes zero.
I report that rather than claim XGBoost is better. Isolation Forest is an unsupervised baseline,
so its anomaly score is never treated as a probability."

**4:20 · Monitoring.** Generate a report. "Evidently compares recent traffic with the validation
period. Some features drifted, which is a reason to investigate. Performance only appears when
confirmed labels exist; here they are replayed dataset labels and marked as such."

**4:50 · Close.** "95 automated tests including a leakage test that shuffles the test labels,
a browser end-to-end test, CI with real PostgreSQL and Redis, and measured latency of about 41 ms
p50 per API call on a laptop."
