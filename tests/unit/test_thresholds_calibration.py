"""Threshold policies, metrics and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from transactionguard.training.evaluation import (
    PlattCalibrator,
    ThresholdPolicy,
    calibration_report,
    choose_threshold,
    classification_metrics,
    logit,
)

# 10 scored transactions, sorted by score; 1 = fraud
SCORES = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])
LABELS = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])


def test_flag_rule_is_score_greater_or_equal():
    m = classification_metrics(LABELS, SCORES, threshold=0.70)
    assert m["alerts"] == 4 and m["confusion_matrix"] == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["false_positive_rate"] == pytest.approx(1 / 6)


def test_min_precision_maximises_recall_under_constraint():
    c = choose_threshold(LABELS, SCORES, ThresholdPolicy("min_precision", min_precision=0.7))
    # threshold 0.70 -> P=0.75, R=0.75; threshold 0.40 -> P=4/7<0.7. Best feasible is 0.70.
    assert c.threshold == pytest.approx(0.70) and c.constraint_met


def test_min_precision_infeasible_falls_back_and_says_so():
    y = np.array([0, 0, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    c = choose_threshold(y, s, ThresholdPolicy("min_precision", min_precision=0.99))
    assert not c.constraint_met and "fell back" in c.note


def test_review_capacity_limits_daily_alerts():
    c = choose_threshold(LABELS, SCORES, ThresholdPolicy("review_capacity", alerts_per_day=2), period_days=2)
    assert c.validation_metrics["alerts"] <= 4 and c.constraint_met


def test_cost_policy_prefers_catching_fraud_when_fn_expensive():
    cheap_fn = choose_threshold(LABELS, SCORES, ThresholdPolicy("cost", fp_cost=1, fn_cost=0.1))
    pricey_fn = choose_threshold(LABELS, SCORES, ThresholdPolicy("cost", fp_cost=1, fn_cost=100))
    assert pricey_fn.threshold < cheap_fn.threshold
    assert pricey_fn.validation_metrics["recall"] == 1.0


def test_average_precision_of_random_scores_is_near_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.01).astype(int)
    m = classification_metrics(y, rng.random(50_000), 0.5)
    assert m["average_precision"] == pytest.approx(y.mean(), abs=0.005)


def test_platt_calibration_fixes_overconfident_scores():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(-4, 2, 40_000)
    y = (rng.random(40_000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    overconfident = 3 * true_logit + 2  # monotone distortion, like class weighting
    cal = PlattCalibrator().fit(overconfident, y)
    assert cal.a == pytest.approx(1 / 3, rel=0.1)
    raw_p = 1 / (1 + np.exp(-overconfident))
    assert calibration_report(y, cal.transform(overconfident))["ece"] < calibration_report(y, raw_p)["ece"]


def test_calibration_preserves_ranking():
    cal = PlattCalibrator().fit(logit(SCORES), LABELS)
    assert (np.argsort(cal.transform(logit(SCORES))) == np.argsort(SCORES)).all()


def test_drift_statistics_are_read_in_the_right_direction():
    from transactionguard.monitoring.drift import drift_decision

    assert drift_decision("K-S p_value", 0.001, 0.05)["drift_detected"] is True
    assert drift_decision("K-S p_value", 0.40, 0.05)["drift_detected"] is False
    assert drift_decision("Wasserstein distance (normed)", 4.6, 0.1)["drift_detected"] is True
    assert drift_decision("Wasserstein distance (normed)", 0.05, 0.1)["drift_detected"] is False
    assert drift_decision("Jensen-Shannon distance", 0.2, 0.1)["statistic_kind"] == "distance"
