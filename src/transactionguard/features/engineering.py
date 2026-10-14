"""Stateless, leakage-safe feature engineering.

"Stateless" = nothing is *fitted* here; each output row depends only on that row and on
**strictly earlier** transactions of the same account. That is why it is safe to run on the full
time-ordered dataset before splitting: no row can see the future or a dataset-wide statistic.
Anything that *is* fitted (imputation medians, scaling, one-hot vocabularies) lives in
``features/pipeline.py`` and is fitted on the training split only.

The same function is used at inference: the API passes the account's stored earlier transactions
plus the new one, then keeps the new row. Training/inference consistency is unit-tested.

Implementation trick for strictly-past windows (vectorised, no Python loops):
  sort rows by (account, time), encode each row as key*BIG + t, and use ``np.searchsorted``.
  #rows of account k in [t - w, t) = searchsorted(k*BIG + t) - searchsorted(k*BIG + t - w)
  ``side="left"`` excludes rows with exactly the same timestamp, so "past" means strictly earlier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transactionguard.data.contract import DataContract

WINDOWS = {"1h": 3600, "24h": 86_400, "7d": 7 * 86_400}
MEAN_WINDOW_S = 30 * 86_400
_BIG = np.int64(10**11)  # > any time range in seconds we will see (~3,000 years)


@dataclass(frozen=True)
class FeatureSpec:
    """Which engineered columns the model consumes, by preprocessing branch."""

    amount: tuple[str, ...]  # log1p -> impute -> scale
    numeric: tuple[str, ...]  # impute (+ missing indicator) -> scale
    categorical: tuple[str, ...]  # one-hot (unknown categories handled explicitly)
    # Raw contract columns that must never reach the model (ids, labels, absolute time).
    excluded: tuple[str, ...]
    descriptions: dict[str, str]

    @property
    def all(self) -> list[str]:
        return [*self.amount, *self.numeric, *self.categorical]


CREDITCARD_FEATURES = FeatureSpec(
    amount=("Amount",),
    numeric=("hour_of_cycle", *(f"V{i}" for i in range(1, 29))),
    categorical=(),
    excluded=("transaction_id", "Time", "Class"),
    descriptions={
        "Amount": "Transaction amount (log1p-transformed inside the model pipeline).",
        "hour_of_cycle": "(Time mod 86400)/3600: position within a 24h cycle measured from the "
        "first transaction. Not the local clock hour (start time unknown).",
        **{
            f"V{i}": f"PCA component V{i} as released by the data owner (meaning confidential)."
            for i in range(1, 29)
        },
    },
)

SYNTHETIC_FEATURES = FeatureSpec(
    amount=("amount",),
    numeric=(
        "hour_of_day",
        "is_night",
        "is_weekend",
        "acct_prior_txn_count",
        "acct_txn_count_1h",
        "acct_txn_count_24h",
        "acct_txn_count_7d",
        "acct_amount_mean_30d",
        "amount_to_acct_mean_30d",
        "secs_since_prev_txn",
        "is_new_merchant_for_account",
        "is_new_country_for_account",
    ),
    categorical=("channel", "merchant_category", "country"),
    excluded=("transaction_id", "timestamp", "account_id", "merchant_id", "is_fraud"),
    descriptions={
        "amount": "Transaction amount (log1p-transformed inside the model pipeline).",
        "hour_of_day": "Hour of the transaction timestamp (UTC).",
        "is_night": "1 if the UTC hour is between 00:00 and 05:59.",
        "is_weekend": "1 if the transaction is on Saturday or Sunday (UTC).",
        "acct_prior_txn_count": "Number of earlier transactions by this account in the data.",
        "acct_txn_count_1h": "Earlier transactions by this account in the previous 1 hour.",
        "acct_txn_count_24h": "Earlier transactions by this account in the previous 24 hours.",
        "acct_txn_count_7d": "Earlier transactions by this account in the previous 7 days.",
        "acct_amount_mean_30d": "Mean amount of this account's earlier transactions in the "
        "previous 30 days (missing if none).",
        "amount_to_acct_mean_30d": "amount / acct_amount_mean_30d (missing if no 30-day history).",
        "secs_since_prev_txn": "Seconds since this account's previous transaction (missing if none).",
        "is_new_merchant_for_account": "1 if the account has not used this merchant before.",
        "is_new_country_for_account": "1 if the account has not transacted in this country before.",
        "channel": "Channel (online, pos, atm, transfer).",
        "merchant_category": "Merchant category.",
        "country": "Transaction country.",
    },
)

FEATURE_SPECS = {"creditcard": CREDITCARD_FEATURES, "synthetic": SYNTHETIC_FEATURES}


# ---------------------------------------------------------------------------------- helpers
def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(ts):
        return ts.to_numpy(np.float64).astype(np.int64)
    # Unit-agnostic: pandas 3 may store datetimes as s/ms/us/ns, so never rely on astype(int64).
    delta = pd.to_datetime(ts, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")
    return (delta // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def _past_window_stats(
    keys: np.ndarray, t: np.ndarray, amount: np.ndarray, windows: dict[str, int], mean_window: int
) -> dict[str, np.ndarray]:
    """Counts/means over strictly-earlier rows with the same key. Inputs are row-aligned."""
    t_rel = t - t.min() if len(t) else t
    order = np.lexsort((t_rel, keys))
    combined = keys[order].astype(np.int64) * _BIG + t_rel[order]
    query = keys.astype(np.int64) * _BIG + t_rel

    # Start index of each key's block = all of that key's rows are "after" this point.
    block_start = np.searchsorted(combined, keys.astype(np.int64) * _BIG + np.int64(-(10**10)), "left")
    before_t = np.searchsorted(combined, query, "left")

    out: dict[str, np.ndarray] = {"prior_count": (before_t - block_start).astype(np.float64)}
    for name, w in windows.items():
        lo = np.searchsorted(combined, query - w, "left")
        out[f"count_{name}"] = (before_t - lo).astype(np.float64)

    csum = np.concatenate([[0.0], np.cumsum(amount[order])])
    lo = np.searchsorted(combined, query - mean_window, "left")
    n = before_t - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mean_amount"] = np.where(n > 0, (csum[before_t] - csum[lo]) / np.maximum(n, 1), np.nan)
    sorted_t = t_rel[order]
    has_prev = before_t > block_start
    prev_t = sorted_t[np.maximum(before_t - 1, 0)]
    out["secs_since_prev"] = np.where(has_prev, (t_rel - prev_t).astype(np.float64), np.nan)
    return out


def _codes(*cols: pd.Series) -> np.ndarray:
    frame = pd.concat([c.astype("string").reset_index(drop=True) for c in cols], axis=1)
    return (
        pd.MultiIndex.from_frame(frame).factorize()[0].astype(np.int64)
        if len(cols) > 1
        else pd.factorize(frame.iloc[:, 0])[0].astype(np.int64)
    )


# ---------------------------------------------------------------------------------- public API
def build_features(df: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    """Return ``df`` plus engineered columns. Row order and index are preserved."""
    out = df.copy()
    if contract.name == "creditcard":
        out["hour_of_cycle"] = (out["Time"].astype("float64") % 86_400) / 3600.0
        return out
    if contract.name != "synthetic":
        raise ValueError(f"No feature recipe for contract {contract.name!r}")

    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["hour_of_day"] = ts.dt.hour.astype("float64")
    out["is_night"] = (ts.dt.hour < 6).astype("float64")
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("float64")

    t = _epoch_seconds(out["timestamp"])
    amount = out["amount"].to_numpy(np.float64)
    acct = _codes(out["account_id"])
    stats = _past_window_stats(acct, t, amount, WINDOWS, MEAN_WINDOW_S)
    out["acct_prior_txn_count"] = stats["prior_count"]
    for name in WINDOWS:
        out[f"acct_txn_count_{name}"] = stats[f"count_{name}"]
    out["acct_amount_mean_30d"] = stats["mean_amount"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["amount_to_acct_mean_30d"] = amount / stats["mean_amount"]
    out["secs_since_prev_txn"] = stats["secs_since_prev"]

    # "New merchant/country" = no strictly-earlier row with the same (account, value) pair.
    for col, feat in (
        ("merchant_id", "is_new_merchant_for_account"),
        ("country", "is_new_country_for_account"),
    ):
        pair = _codes(out["account_id"], out[col])
        prior = _past_window_stats(pair, t, amount, {}, 1)["prior_count"]
        out[feat] = (prior == 0).astype("float64")
    return out


def feature_frame(df: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    """Engineered features restricted to exactly what the model consumes (no ids/labels/time)."""
    spec = FEATURE_SPECS[contract.name]
    built = build_features(df, contract)
    leaked = set(spec.all) & set(spec.excluded)
    assert not leaked, f"Excluded columns used as features: {leaked}"
    return built[spec.all]
