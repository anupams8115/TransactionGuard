"""Metrics, threshold policies and probability calibration.

Key ideas (interview-friendly):
  * With ~0.2-0.6% positives, accuracy is useless (predicting "never fraud" scores 99.5%).
    Our headline metric is **average precision (AP)**: the area under the precision-recall curve
    computed as a step-wise sum. Its "no-skill" value equals the prevalence.
  * ROC-AUC is reported as a supplementary metric: it can look excellent on rare-event data
    because the huge negative class dilutes false positives.
  * The operating **threshold** is a business decision, chosen on validation data with an
    explicit objective, never on the test set.
  * Class weighting distorts probabilities, so supervised scores are **calibrated** (Platt
    scaling on the model's log-odds, fitted on validation data) before being shown as
    probabilities. Isolation Forest outputs an anomaly score, which is never called a probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

EPS = 1e-12


# ------------------------------------------------------------------------------ calibration
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class PlattCalibrator:
    """calibrated_logodds = a * raw_logodds + b, fitted on validation data.

    Because this is an affine map of the log-odds, SHAP contributions computed on the raw
    log-odds scale can be multiplied by ``a`` to become exact contributions on the calibrated
    log-odds scale (base value -> a*base + b). See ``inference/explain.py``.
    """

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, raw_logodds: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularised
        lr.fit(raw_logodds.reshape(-1, 1), y)
        self.a, self.b, self.fitted = float(lr.coef_[0, 0]), float(lr.intercept_[0]), True
        return self

    def calibrated_logodds(self, raw_logodds: np.ndarray) -> np.ndarray:
        return self.a * raw_logodds + self.b

    def transform(self, raw_logodds: np.ndarray) -> np.ndarray:
        return sigmoid(self.calibrated_logodds(raw_logodds))


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability table with quantile bins (equal-count bins work better for rare events)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = bins == b
        if not m.any():
            continue
        mean_p, frac = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": b, "count": int(m.sum()), "mean_predicted": mean_p, "observed_rate": frac})
        ece += m.mean() * abs(mean_p - frac)
    return {"brier": float(brier_score_loss(y, p)), "ece": float(ece), "bins": rows}


# ------------------------------------------------------------------------------ thresholds
PolicyName = Literal["min_precision", "review_capacity", "cost"]


@dataclass
class ThresholdPolicy:
    """Explicit objective for choosing the operating point on VALIDATION data.

    * min_precision  : maximise recall subject to precision >= ``min_precision``.
    * review_capacity: alert on at most ``alerts_per_day`` transactions/day (analyst capacity).
    * cost           : minimise ``fp_cost * FP + fn_cost * FN``.
    """

    name: PolicyName = "min_precision"
    min_precision: float = 0.5
    alerts_per_day: float = 50.0
    fp_cost: float = 1.0
    fn_cost: float = 25.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdChoice:
    threshold: float
    policy: dict
    constraint_met: bool
    note: str
    validation_metrics: dict = field(default_factory=dict)


def choose_threshold(
    y: np.ndarray, score: np.ndarray, policy: ThresholdPolicy, *, period_days: float | None = None
) -> ThresholdChoice:
    """Pick a threshold; a transaction is flagged when ``score >= threshold``."""
    candidates = np.unique(score)[::-1]  # high -> low; each is a possible cut
    # Vectorised confusion counts for "flag if score >= c".
    order = np.argsort(-score, kind="mergesort")
    s_sorted, y_sorted = score[order], y[order]
    tp_cum, fp_cum = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    last = np.searchsorted(-s_sorted, -candidates, side="right") - 1  # last index with s >= c
    tp, fp = tp_cum[last].astype(float), fp_cum[last].astype(float)
    pos = y.sum()
    fn = pos - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(pos, 1)

    note, met = "", True
    if policy.name == "min_precision":
        ok = precision >= policy.min_precision
        if ok.any():
            best = int(np.flatnonzero(ok)[np.argmax(recall[ok])])
        else:
            met = False
            f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)
            best = int(np.argmax(f1))
            note = (
                f"No threshold reached precision >= {policy.min_precision} on validation; "
                "fell back to the max-F1 threshold."
            )
    elif policy.name == "review_capacity":
        if not period_days:
            raise ValueError("review_capacity needs the validation period length in days")
        alerts_per_day = (tp + fp) / period_days
        ok = alerts_per_day <= policy.alerts_per_day
        if ok.any():
            best = int(np.flatnonzero(ok)[np.argmax(recall[ok])])
        else:
            met, best = False, 0
            note = "Even the highest score exceeds capacity; using the strictest threshold."
    elif policy.name == "cost":
        cost = policy.fp_cost * fp + policy.fn_cost * fn
        best = int(np.argmin(cost))
    else:  # pragma: no cover
        raise ValueError(policy.name)

    thr = float(candidates[best])
    return ThresholdChoice(
        threshold=thr,
        policy=policy.to_dict(),
        constraint_met=met,
        note=note,
        validation_metrics=classification_metrics(y, score, thr),
    )


# ------------------------------------------------------------------------------ metrics
def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    has_both = 0 < y.sum() < len(y)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)) if has_both else None,
        "roc_auc": float(roc_auc_score(y, score)) if has_both else None,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "alerts": int(tp + fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ap(y: np.ndarray, scores: dict[str, np.ndarray], n_boot: int = 500, seed: int = 0) -> dict:
    """Stratified bootstrap of test-set average precision.

    With only tens of positives, AP moves a lot from sample to sample. Resampling positives and
    negatives separately (same indices for every model = *paired*) gives a 95% interval per model
    and for the difference between models, so we do not over-claim a "winner".
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            draws[k].append(average_precision_score(y[idx], s[idx]))
    out: dict = {"n_boot": n_boot, "method": "stratified paired bootstrap", "models": {}, "differences": {}}
    for k, v in draws.items():
        out["models"][k] = {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    names = list(scores)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = np.array(draws[a]) - np.array(draws[b])
            out["differences"][f"{a} - {b}"] = {
                "mean": float(d.mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "share_positive": float((d > 0).mean()),
            }
    return out


def pr_curve_points(y: np.ndarray, score: np.ndarray, max_points: int = 200) -> list[dict]:
    p, r, _ = precision_recall_curve(y, score)
    idx = np.unique(np.linspace(0, len(p) - 1, min(max_points, len(p))).astype(int))
    return [{"precision": float(p[i]), "recall": float(r[i])} for i in idx]
