# Interview guide

Questions you are likely to get about TransactionGuard. Each one has a short answer to say out
loud, the reasoning behind it, and where it lives in the code so you can show it.

---

## A. Problem framing and data

**1. Why not just use accuracy?**
Because 99.87% of transactions are genuine. A model that says "never fraud" gets 99.87% accuracy
and catches nothing. I use **average precision (AP)**, the area under the precision-recall curve.
Its no-skill value equals the fraud rate (0.0013 here), so 0.79 means a lot. ROC-AUC is reported
too, but it looks great on rare events because the false-positive rate is divided by a huge number
of negatives. → `training/evaluation.py`

**2. Why did you choose this dataset?**
It is public, downloadable without a login, has a clear licence, and is real (not simulated),
which makes the numbers meaningful. The trade-off is that the features are PCA components with
secret meanings and there are no account IDs, so I also built a synthetic dataset to show
behavioural history features. Results from the two are never mixed. → `docs/dataset.md`

**3. What is the data contract?**
One Python object that lists every column, its type, allowed range or categories, and its role
(id, timestamp, feature, target). Validation, the API's request schema and the frontend form are
all generated from it, so they can't disagree. Missing fields are rejected, never filled with
made-up defaults. → `data/contract.py`

## B. Leakage (the most important topic)

**4. What is data leakage? Give examples you prevented.**
Leakage is when training uses information that wouldn't exist at prediction time, so offline
scores look better than reality. I prevented four kinds:
- *Future leakage:* history features use only **strictly earlier** transactions of the same account.
- *Split leakage:* I split **by time** (train on the past, test on the future), not randomly.
- *Preprocessing leakage:* imputation medians, scaling and one-hot vocabularies are **fitted on
  training rows only**.
- *Selection leakage:* tuning, calibration, threshold and model choice use **validation only**;
  the test set is scored once at the end.
→ `features/engineering.py`, `training/splits.py`, `training/train.py`

**5. How do you *prove* there's no leakage?**
Tests. One changes all *future* rows and checks that past features don't move. Another shuffles
the *test-set labels*, retrains, and asserts that every decision (parameters, threshold,
calibration, selected model) is identical. If the test set influenced anything, that test fails.
→ `tests/unit/test_features_leakage.py`, `test_training_and_bundles.py::test_test_labels_cannot_change_any_decision`

**6. You computed features before splitting. Isn't that leakage?**
Not here, because those features are *stateless and past-only*: each row looks only at earlier
rows of its own account, never at dataset-wide statistics. Anything that learns from data (medians,
scalers, encoders) is fitted after the split, on train only.

**7. How are history features computed efficiently?**
I sort by (account, time), encode each row as `account_code × BIG + time`, and use binary search
(`np.searchsorted`). "How many transactions in the last hour" becomes the difference between two
search positions. `side="left"` excludes rows with the same timestamp, which enforces *strictly*
earlier. It handles 143k rows in 0.3 s with no Python loops.

## C. Modelling

**8. How did you handle class imbalance? Why not SMOTE?**
With class weights: logistic regression uses `class_weight="balanced"`, and XGBoost uses
`scale_pos_weight` (whose strength Optuna tunes). SMOTE invents synthetic fraud rows by
interpolation, which makes little sense for mixed or categorical transaction data, and applying it
before splitting leaks. Weights change the loss, not the data.

**9. Why include a dummy model and Isolation Forest?**
The dummy is the "no skill" reference: if a model can't beat it, it's useless. Isolation Forest
shows what an *unsupervised* detector does without labels. It scored AP 0.07 against 0.79 for
XGBoost, which demonstrates that labels matter a lot here. Its output is an anomaly score, so I
compare it only through ranking metrics, never by score values.

**10. How did you tune hyperparameters?**
Optuna with a TPE sampler, a fixed seed, a fixed number of trials and a timeout (the "budget").
The objective is validation AP. Default parameters are always evaluated first. → `training/models.py`

**11. Did XGBoost win?**
It had the best *validation* AP, so the rule selected it. On the test set, a paired bootstrap
gives XGBoost − LogReg = +0.034 with a 95% CI of [−0.028, +0.113], which includes zero. So I can't
claim it's better, and I say so. With only 75 test frauds, small gaps are noise.

## D. Calibration, thresholds, explanations

**12. What is calibration and why did you need it?**
A calibrated model saying "5%" should be right about 5% of the time. Class weighting pushes
scores up, so I fit **Platt scaling** on validation: `p = sigmoid(a · logit + b)`. Test ECE is
0.0002. Isolation Forest is not calibrated and is never called a probability.

**13. How did you pick the threshold?**
On validation, with an explicit business objective: maximise recall while keeping precision ≥ 0.5
(at least half of the alerts are real). Two other policies are implemented: analyst capacity
(alerts per day) and cost (FP cost vs FN cost). If a constraint can't be met, the code falls back
and **records** that it did. → `choose_threshold`

**14. How do SHAP explanations work here?**
SHAP splits a prediction into per-feature contributions that add up exactly: `output = base +
Σ contributions`. I explain the model's log-odds, then multiply by the Platt slope `a`. Because
calibration is linear on the log-odds scale, the contributions are *exact on the calibrated scale*.
The code checks that the sum matches for every prediction; if it doesn't, it shows "failed"
instead of a made-up explanation. → `inference/explain.py`

**15. How do you handle one-hot features in explanations?**
"channel" becomes four columns after encoding. SHAP values are additive, so I sum the four into
one "channel" contribution and keep the parts for detail. The mapping from encoded columns back to
raw features is built from the fitted pipeline's own feature names.

**16. Does SHAP tell you *why* fraud happened?**
No. It tells you how the *model* used the inputs. It is not causal and it isn't proof. The UI says
this next to every explanation.

**17. Why explain Isolation Forest on a log2 scale?**
Its score is `2^(−path length / c)`, which is exponential in what SHAP decomposes (path length).
On the log2 scale the relationship is exactly linear, so contributions add up. I found this
because the additivity check *refused* the first version, which is what that check is for.

## E. Backend engineering

**18. How do you make scoring idempotent?**
The database enforces it. There is a unique constraint on (workspace, transaction id) for
transactions and on (transaction, model version) for predictions. Resubmitting the same payload
returns the stored prediction (HTTP 200 with `idempotent_replay`); the same id with a *different*
payload returns 409. If two requests race, the loser's insert fails and it returns the winner's
row. Batch uploads also accept an `Idempotency-Key` header. → `inference/service.py`

**19. What happens if a batch job crashes halfway?**
Each 1,000-row chunk is committed separately. Celery uses `acks_late`, so an unfinished job is
redelivered. The rerun reuses predictions that already exist, so nothing is duplicated. Transient
errors (database hiccups) retry with backoff up to a limit; others fail fast with a clear message
that the UI shows. Job state lives in the database, not only in Celery. → `worker/tasks.py`

**20. How does authentication work?**
Passwords are hashed with Argon2id. Login returns a short-lived JWT, also set as an `HttpOnly`,
`SameSite=Strict` cookie so JavaScript can't read it. Every request reloads the user, so a
deactivated user loses access immediately. Cookie-authenticated writes also need an
`X-Requested-With` header as CSRF protection. Roles are viewer < analyst < admin, and every query
filters by workspace, so other tenants' data returns 404 (not 403, to avoid revealing that it
exists). → `api/security.py`, `api/deps.py`

**21. Why is loading a model file a security risk?**
`joblib`/`pickle` can run code while loading. So bundles are loaded only if they are listed in a
registry written by training, sit inside the bundles folder, and match their recorded SHA-256.
There is no endpoint for uploading models. MLflow stores the XGBoost model in its native JSON
format and sklearn models with `skops`, both of which avoid pickle.

**22. Why store the feature vector with each prediction?**
So the prediction is reproducible. If more history arrives later (including late, out-of-order
events), an explanation recomputed from the stored vector still describes exactly what was scored.
It also feeds drift monitoring.

**23. Tell me about a bug you found.**
Three good ones:
1. Request logging deadlocked SQLite. The metrics write waited for the request's own open
   transaction. I moved metrics to an in-memory queue flushed by a background thread.
2. Demo seeding took 22 minutes for 20k rows because one ORM session grew across all chunks. A
   transaction per chunk plus a bulk insert made it about 10× faster.
3. Drift flags were inverted. For large samples Evidently switches from p-values (small = drift)
   to distances (large = drift), and I had read everything as a p-value. A cross-check against
   Evidently's own count exposed it, and a test now pins the logic.

## F. Monitoring and operations

**24. What's the difference between drift and model degradation?**
Drift means the *inputs or scores* changed relative to the reference period. Degradation means
*accuracy* dropped, which you can only measure with confirmed outcomes such as chargebacks, and
those arrive late. So the report shows drift always, and shows performance only when enough
confirmed labels exist. Analyst opinions are never used as labels.

**25. What did monitoring find?**
On the demo windows: 8 of 31 columns drifted for creditcard and 3 for synthetic. One finding is a
design flaw: `acct_prior_txn_count` is cumulative, so it grows over time and *always* drifts. A
better feature would use a fixed window. That's a good example of monitoring catching a feature
design problem.

**26. How fast is it? Can it scale?**
Measured on my laptop: about 41 ms p50 per API call (creditcard, no SHAP) and about 25 req/s. With
4 concurrent clients throughput didn't improve and p95 jumped, because SQLite allows one writer at
a time and there was one process. PostgreSQL with several workers should help, but I haven't
measured it, so I don't claim it. → `docs/benchmarks.md`

## G. "What would you do next?"

- Replace cumulative features with windowed ones; add merchant-level history.
- Run the load test on PostgreSQL with multiple workers; add rate limiting.
- Add a champion/challenger shadow mode before switching models.
- Model delayed labels explicitly (a gap between train and validation equal to chargeback delay;
  `SplitConfig.gap_seconds` already supports it).
- Add the optional Kafka ingestion path with a dead-letter queue.
