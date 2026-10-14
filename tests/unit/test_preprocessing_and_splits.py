"""Preprocessing is fitted on training data only, handles unseen categories, and splits are sound."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transactionguard.data.contract import SYNTHETIC
from transactionguard.features.engineering import SYNTHETIC_FEATURES, feature_frame
from transactionguard.features.pipeline import build_preprocessor, transformed_to_original
from transactionguard.training.splits import SplitConfig, make_split


@pytest.fixture
def frames(small_synthetic):
    df = small_synthetic.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    return df, feature_frame(df, SYNTHETIC)


def test_imputer_statistics_come_from_training_rows_only(frames):
    df, X = frames
    split = make_split(df, SYNTHETIC, SplitConfig())
    pre = build_preprocessor(SYNTHETIC_FEATURES).fit(X.iloc[split.train])
    imputer = pre.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    cols = list(SYNTHETIC_FEATURES.numeric)
    i = cols.index("secs_since_prev_txn")
    assert imputer.statistics_[i] == pytest.approx(X.iloc[split.train]["secs_since_prev_txn"].median())
    assert imputer.statistics_[i] != pytest.approx(X["secs_since_prev_txn"].median())


def test_unseen_category_is_handled_and_detectable(frames):
    _, X = frames
    train = X[X["country"] != "BR"]  # BR never seen in training
    pre = build_preprocessor(SYNTHETIC_FEATURES).fit(train)
    row = X[X["country"] == "BR"].head(1)
    assert len(row) == 1
    out = pre.transform(row)
    country_cols = [c for c in out.columns if c.startswith("cat__country_")]
    assert out[country_cols].to_numpy().sum() == 0  # all-zeros encoding, no crash
    encoder = pre.named_steps["columns"].named_transformers_["cat"]
    assert "BR" not in encoder.categories_[list(SYNTHETIC_FEATURES.categorical).index("country")]


def test_preprocessing_is_deterministic_and_consistent(frames):
    _, X = frames
    pre = build_preprocessor(SYNTHETIC_FEATURES).fit(X.head(5000))
    a = pre.transform(X.tail(300))
    b = pre.transform(X.tail(300).copy())
    pd.testing.assert_frame_equal(a, b)
    one_by_one = pd.concat([pre.transform(X.tail(300).iloc[[i]]) for i in range(5)])
    pd.testing.assert_frame_equal(one_by_one, a.head(5))  # single-row == batch transform


def test_every_transformed_column_maps_to_a_raw_feature(frames):
    _, X = frames
    pre = build_preprocessor(SYNTHETIC_FEATURES).fit(X)
    mapping = transformed_to_original(pre, SYNTHETIC_FEATURES)
    assert set(mapping) == set(pre.get_feature_names_out())
    assert mapping["cat__channel_online"] == "channel"
    assert mapping["amount__amount"] == "amount"
    assert mapping["num__amount_to_acct_mean_30d"] == "amount_to_acct_mean_30d"
    assert mapping["num__missingindicator_secs_since_prev_txn"] == "secs_since_prev_txn"


def test_chronological_split_respects_time(frames):
    df, _ = frames
    s = make_split(df, SYNTHETIC, SplitConfig())
    t = df["timestamp"]
    assert t.iloc[s.train].max() < t.iloc[s.valid].min()
    assert t.iloc[s.valid].max() < t.iloc[s.test].min()
    assert len(s.train) + len(s.valid) + len(s.test) == len(df)


def test_gap_drops_rows_between_periods(frames):
    df, _ = frames
    s = make_split(df, SYNTHETIC, SplitConfig(gap_seconds=86_400))
    t = df["timestamp"]
    assert (t.iloc[s.valid].min() - t.iloc[s.train].max()) > pd.Timedelta("1D")


def test_group_split_has_no_shared_accounts(frames):
    df, _ = frames
    s = make_split(df, SYNTHETIC, SplitConfig(strategy="group"))
    accts = [set(df["account_id"].iloc[ix]) for ix in (s.train, s.valid, s.test)]
    assert not (accts[0] & accts[1] or accts[0] & accts[2] or accts[1] & accts[2])
    assert np.isclose(len(s.train) / len(df), 0.6, atol=0.1)
