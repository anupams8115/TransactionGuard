# Model card: TransactionGuard credit-card risk model

| | |
|---|---|
| Model version | `creditcard-xgboost-20261014T115618Z` (active), preprocessing `pp-1.0` |
| Type | XGBoost gradient-boosted trees (300 trees, depth 3) + Platt calibration on log-odds |
| Output | Calibrated probability that the source would label the transaction as fraud |
| Decision | Flag for analyst review when score ≥ 0.0860 |
| Training data | ULB/OpenML credit-card data, first 60% of time (170,888 rows, 360 frauds) |
| Seed / budget | 42 / Optuna `small` (15 trials per model) |
| Code | `src/transactionguard/training/train.py`. MLflow parent run is logged per session. |

## Intended use

- **For:** ranking transactions for **human review**, with an explanation of each score, in a
  portfolio or teaching setting.
- **Not for:** automatically blocking payments, declining customers, or any decision about a
  person without human review. The score is a risk signal, **not proof of fraud**.
- **Not validated for:** any data other than the 2013 European card data it was trained on.
  Real deployment would need retraining and validation on the operator's own data.

## How it was built (and why)

- **Chronological split** 60/20/20 by time. It mirrors deployment (train on the past, score the
  future). Random splits leak future fraud campaigns into training.
- **Validation is the only place decisions are made:** Optuna's objective (average precision),
  Platt calibration, threshold choice and model selection. The **test set was scored once**,
  afterwards. A unit test shuffles the test labels and checks that no decision changes.
- **Imbalance:** class weighting (`scale_pos_weight`, with its strength tuned as none / sqrt /
  balanced). No SMOTE, because synthetic minority rows would be invented data, and resampling
  before splitting leaks.
- **Calibration:** weighting inflates probabilities, so we fit `p = sigmoid(a·logit + b)` on
  validation (a = 1.03, b = −0.45 for the selected model). Test ECE is 0.00019 and Brier is
  0.00041.
- **Threshold policy:** maximise recall subject to validation precision ≥ 0.5. On validation the
  constraint was met.

## Results on the untouched test set

Test: 56,962 transactions, 75 frauds (0.132%). The primary metric is **average precision (AP)**,
the step-wise area under the precision-recall curve; its no-skill value equals the prevalence.

| Model | Val AP | Test AP [95% CI] | Precision | Recall | F1 | FPR | TP / FP / FN | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Dummy | 0.001 | 0.001 | 0.001 | 1.000 | 0.003 | 1.000 | 75 / 56,887 / 0 | 0.500 |
| Logistic Regression | 0.762 | 0.748 [0.643, 0.861] | 0.740 | 0.720 | 0.730 | 0.00033 | 54 / 19 / 21 | 0.983 |
| **XGBoost** | 0.797 | **0.789** [0.692, 0.872] | 0.740 | 0.760 | 0.750 | 0.00035 | 57 / 20 / 18 | 0.975 |
| Isolation Forest* | 0.034 | 0.073 [0.055, 0.102] | 0.116 | 0.387 | 0.178 | 0.00388 | 29 / 221 / 46 | 0.956 |

\*Isolation Forest is unsupervised (trained on mixed, unlabelled data). Its output is an anomaly
score, not a probability, so only its ranking metrics (AP, ROC-AUC) are comparable with the other
models. Its threshold policy (precision ≥ 0.5) could not be met on validation, so it fell back to
the max-F1 threshold, and the report records that.

**Read the intervals.** The paired bootstrap difference (XGBoost − Logistic Regression) in test AP
is +0.034 with a 95% CI of [−0.028, +0.113]. **XGBoost was selected because it had the higher
validation AP, but the test set cannot distinguish it from Logistic Regression.** With 57
validation frauds and 75 test frauds, small differences are noise. Logistic Regression would be a
defensible choice when simplicity or explanation stability matters more.

ROC-AUC is near 0.98 for both supervised models. On rare events it looks impressive because the
huge negative class dilutes false positives, which is why AP is the headline metric.

## Explanations

SHAP TreeExplainer (tree-path-dependent) on the model's log-odds, multiplied by the Platt slope,
so contributions are **exact on the calibrated log-odds scale**:
`calibrated_logit = base + Σ contributions` (additivity is checked per prediction, tolerance
1e-3). If a check fails, the explanation is marked `failed` rather than shown.
Global importance (mean |SHAP| on a validation sample): V4, V14, V12, V10, V11, V1, V7, Amount.
These are PCA components with confidential meaning, so no business interpretation is offered.

## Limitations and risks

- 2 days of 2013 data from one European issuer, so generalisation to other periods and markets is
  unknown.
- No account identifiers, so no behavioural history features and no unseen-customer evaluation.
- Labels come from the data owner; label noise and delayed chargebacks are not modelled.
- Small positive counts make the threshold and calibration estimates noisy.
- Monitoring on the demo window flagged drift in 8 of 31 monitored columns (including the score).
  That shows the inputs changed; it is not proof that accuracy changed.
- The model may perform differently across customer groups; no fairness analysis is possible
  without demographic data, and none is claimed.
- Analyst review outcomes are **not** fed back into training automatically. Retraining is a
  manual, reviewed step.

## Synthetic model (demo only)

`synthetic-xgboost-20261014T114016Z`: test AP 0.974 [0.960, 0.987]. **This score is high by
construction**, because the generator planted the patterns. It only demonstrates the history
features and the workflow.
