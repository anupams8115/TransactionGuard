"""Leakage prevention and training/inference feature consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transactionguard.data.contract import CREDITCARD, SYNTHETIC
from transactionguard.features.engineering import (
    CREDITCARD_FEATURES,
    SYNTHETIC_FEATURES,
    build_features,
    feature_frame,
)

HISTORY_COLS = [
    c for c in SYNTHETIC_FEATURES.numeric if c.startswith(("acct_", "amount_to", "secs_", "is_new"))
]


def test_features_ignore_future_rows(small_synthetic):
    """Changing or deleting FUTURE transactions must not change any earlier row's features."""
    df = small_synthetic.sort_values("timestamp").reset_index(drop=True)
    cut = df["timestamp"].iloc[len(df) // 2]
    base = build_features(df, SYNTHETIC)

    future = df["timestamp"] > cut
    tampered = df.copy()
    tampered.loc[future, "amount"] *= 100
    tampered.loc[future, "country"] = "NG"
    truncated = df[~future]
    for other in (build_features(tampered, SYNTHETIC), build_features(truncated, SYNTHETIC)):
        pd.testing.assert_frame_equal(
            base.loc[~future, HISTORY_COLS].reset_index(drop=True),
            other.loc[~future.reindex(other.index, fill_value=False), HISTORY_COLS].reset_index(drop=True),
        )


def test_labels_do_not_influence_features(small_synthetic):
    flipped = small_synthetic.assign(is_fraud=1 - small_synthetic["is_fraud"])
    pd.testing.assert_frame_equal(
        feature_frame(small_synthetic, SYNTHETIC), feature_frame(flipped, SYNTHETIC)
    )


def test_history_is_strictly_past_and_matches_brute_force(small_synthetic):
    df = small_synthetic
    feats = build_features(df, SYNTHETIC)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(df), size=40, replace=False):
        row = df.iloc[i]
        acct = df[df["account_id"] == row["account_id"]]
        prior = acct[acct["timestamp"] < row["timestamp"]]  # strictly earlier
        f = feats.iloc[i]
        assert f["acct_prior_txn_count"] == len(prior)
        assert f["acct_txn_count_24h"] == (prior["timestamp"] >= row["timestamp"] - pd.Timedelta("24h")).sum()
        last30 = prior[prior["timestamp"] >= row["timestamp"] - pd.Timedelta("30D")]
        if len(last30):
            assert f["acct_amount_mean_30d"] == pytest.approx(last30["amount"].mean())
        else:
            assert np.isnan(f["acct_amount_mean_30d"])
        assert f["is_new_merchant_for_account"] == float(
            not (prior["merchant_id"] == row["merchant_id"]).any()
        )


def test_same_timestamp_rows_do_not_see_each_other(small_synthetic):
    row = small_synthetic.iloc[[100]].copy()
    twin = row.copy()
    twin["transaction_id"] = "twin"
    feats = build_features(pd.concat([row, twin], ignore_index=True), SYNTHETIC)
    assert (feats["acct_prior_txn_count"] == 0).all()


def test_inference_on_account_history_equals_training_features(small_synthetic):
    """The API computes features from the account's stored past + the new row; it must equal
    what training computed on the full dataset."""
    full = build_features(small_synthetic, SYNTHETIC)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(small_synthetic), size=25, replace=False):
        row = small_synthetic.iloc[i]
        history = small_synthetic[
            (small_synthetic["account_id"] == row["account_id"])
            & (small_synthetic["timestamp"] < row["timestamp"])
        ]
        online = build_features(pd.concat([history, small_synthetic.iloc[[i]]]), SYNTHETIC).iloc[-1]
        cols = SYNTHETIC_FEATURES.all
        pd.testing.assert_series_equal(online[cols], full.iloc[i][cols], check_names=False)


def test_excluded_columns_never_reach_the_model():
    for spec in (SYNTHETIC_FEATURES, CREDITCARD_FEATURES):
        assert not set(spec.all) & set(spec.excluded)
    assert "Class" not in CREDITCARD_FEATURES.all and "Time" not in CREDITCARD_FEATURES.all
    assert "is_fraud" not in SYNTHETIC_FEATURES.all


def test_creditcard_hour_of_cycle():
    row = {"transaction_id": "a", "Time": 90_000.0, "Amount": 1.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    out = build_features(pd.DataFrame([row]), CREDITCARD)
    assert out["hour_of_cycle"].iloc[0] == pytest.approx((90_000 % 86_400) / 3600)
