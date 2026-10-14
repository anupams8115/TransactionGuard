"""Deterministic SYNTHETIC transaction generator.

Purpose: installation checks, tests and UI demos without downloading anything.
Everything here is invented. Results on this data must be labelled "synthetic" and never
presented as benchmark evidence.

Design:
  * Each account has a home country, a spending level, favourite categories and favourite
    merchants, so "normal behaviour" exists and history features are meaningful.
  * Fraud is injected through three scenarios (account-takeover bursts, card testing, and
    single skimmed-card purchases), plus *legitimate* look-alikes (genuine travel abroad,
    occasional big purchases), so the task is not trivially separable.
  * A fixed seed makes the output byte-for-byte reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transactionguard.data.contract import CHANNELS, COUNTRIES, MERCHANT_CATEGORIES

END = pd.Timestamp("2026-06-30T23:59:59", tz="UTC")  # fixed so output never depends on "now"
MERCHANTS_PER_CATEGORY = 40
_DAY_HOUR_P = np.array(
    [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.5, 3, 4, 5, 5.5, 6, 6.5, 6, 5.5, 5.5, 6, 6.5, 7, 7, 6, 4.5, 3, 2]
)
_DAY_HOUR_P = _DAY_HOUR_P / _DAY_HOUR_P.sum()
_NIGHT_HOUR_P = np.roll(_DAY_HOUR_P[::-1], 3)  # fraud skews towards night-time
_NIGHT_HOUR_P = _NIGHT_HOUR_P / _NIGHT_HOUR_P.sum()

_CAT_BASE_P = np.array([0.24, 0.16, 0.10, 0.10, 0.06, 0.05, 0.10, 0.12, 0.05, 0.02])
_RISKY_CATS = ["electronics", "gaming", "crypto_exchange", "travel"]
_CAT_INDEX = {c: i for i, c in enumerate(MERCHANT_CATEGORIES)}


@dataclass(frozen=True)
class SyntheticConfig:
    n_accounts: int = 1500
    days: int = 60
    mean_txn_per_day: float = 1.6
    fraud_account_rate: float = 0.10  # share of accounts that experience at least one fraud event
    seed: int = 42


def _merchant(cat: str, k: int) -> str:
    return f"m_{cat[:4]}_{k:03d}"


def generate(config: SyntheticConfig | None = None) -> pd.DataFrame:
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    start = END - pd.Timedelta(days=cfg.days)
    rows: list[tuple] = []

    country_p = np.array([0.55, 0.15, 0.10, 0.06, 0.05, 0.04, 0.03, 0.02])
    for a in range(cfg.n_accounts):
        acct = f"acct_{a:05d}"
        home = COUNTRIES[rng.choice(len(COUNTRIES), p=country_p)]
        spend_scale = float(rng.lognormal(mean=3.4, sigma=0.6))  # typical amount ~ e^3.4 ≈ 30
        cat_p = rng.dirichlet(_CAT_BASE_P * 40)
        favourites = {
            c: [int(k) for k in rng.choice(MERCHANTS_PER_CATEGORY, size=4, replace=False)]
            for c in MERCHANT_CATEGORIES
        }
        rate = cfg.mean_txn_per_day * float(rng.gamma(4.0, 0.25))
        n = max(3, int(rng.poisson(rate * cfg.days)))
        days = rng.uniform(0, cfg.days, size=n).astype(int)
        hours = rng.choice(24, size=n, p=_DAY_HOUR_P)
        secs = rng.integers(0, 3600, size=n)
        cats = rng.choice(len(MERCHANT_CATEGORIES), size=n, p=cat_p)
        # A few accounts genuinely travel abroad for a few days (legitimate look-alike of fraud).
        travel_day = int(rng.integers(0, cfg.days)) if rng.random() < 0.15 else -100
        travel_country = COUNTRIES[int(rng.integers(0, len(COUNTRIES)))]
        for d, h, s, ci in zip(days, hours, secs, cats, strict=True):
            cat = MERCHANT_CATEGORIES[ci]
            ts = float(d) * 86400 + float(h) * 3600 + float(s)  # seconds after `start`
            abroad = travel_day <= d < travel_day + 4
            country = travel_country if abroad else home
            if cat in ("travel", "electronics", "fashion"):
                amount = spend_scale * rng.lognormal(1.0, 0.7)
            else:
                amount = spend_scale * rng.lognormal(0.0, 0.5)
            if rng.random() < 0.01:  # legitimate occasional large purchase
                amount *= rng.uniform(5, 15)
            channel = rng.choice(CHANNELS, p=[0.38, 0.45, 0.07, 0.10])
            if cat == "utilities":
                channel = "online"
            merchant_k = (
                favourites[cat][int(rng.integers(0, 4))]
                if rng.random() < 0.8
                else int(rng.integers(0, MERCHANTS_PER_CATEGORY))
            )
            rows.append(
                (acct, ts, _merchant(cat, merchant_k), round(float(amount), 2), str(channel), cat, country, 0)
            )

        # ------------------------------------------------------------ fraud scenarios
        if rng.random() < cfg.fraud_account_rate:
            scenario = rng.choice(["takeover", "card_testing", "skimmed"], p=[0.5, 0.3, 0.2])
            t0 = float(int(rng.uniform(5, cfg.days - 0.5))) * 86400
            t0 += float(int(rng.choice(24, p=_NIGHT_HOUR_P))) * 3600 + 60 * float(rng.integers(0, 60))
            foreign = [c for c in COUNTRIES if c != home]
            if scenario == "takeover":
                k = int(rng.integers(3, 9))
                country = foreign[int(rng.integers(0, len(foreign)))] if rng.random() < 0.7 else home
                for j in range(k):
                    cat = _RISKY_CATS[int(rng.integers(0, len(_RISKY_CATS)))]
                    rows.append(
                        (
                            acct,
                            t0 + 60 * (float(rng.uniform(0, 120)) + j),
                            _merchant(cat, int(rng.integers(0, MERCHANTS_PER_CATEGORY))),
                            round(float(spend_scale * rng.uniform(2, 12)), 2),
                            str(rng.choice(["online", "transfer"])),
                            cat,
                            country,
                            1,
                        )
                    )
            elif scenario == "card_testing":
                k = int(rng.integers(4, 12))
                for j in range(k):
                    cat = str(rng.choice(["gaming", "entertainment", "fashion"]))
                    rows.append(
                        (
                            acct,
                            t0 + 60 * (float(rng.uniform(0, 30)) + j * 0.5),
                            _merchant(cat, int(rng.integers(0, MERCHANTS_PER_CATEGORY))),
                            round(float(rng.uniform(0.5, 5.0)), 2),
                            "online",
                            cat,
                            home,
                            1,
                        )
                    )
            else:  # skimmed card: one plausible-looking in-person purchase
                cat = str(rng.choice(["electronics", "fashion", "fuel"]))
                country = foreign[int(rng.integers(0, len(foreign)))] if rng.random() < 0.5 else home
                rows.append(
                    (
                        acct,
                        t0,
                        _merchant(cat, int(rng.integers(0, MERCHANTS_PER_CATEGORY))),
                        round(float(spend_scale * rng.uniform(1.5, 6)), 2),
                        "pos",
                        cat,
                        country,
                        1,
                    )
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "account_id",
            "timestamp",
            "merchant_id",
            "amount",
            "channel",
            "merchant_category",
            "country",
            "is_fraud",
        ],
    )
    df["timestamp"] = start + pd.to_timedelta(df["timestamp"].round(0), unit="s")
    df = df[df["timestamp"] <= END].copy()
    df["amount"] = df["amount"].clip(lower=0.01, upper=50_000.0)
    df = df.sort_values(["timestamp", "account_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"syn-{i:07d}" for i in range(len(df))])
    return df[
        [
            "transaction_id",
            "timestamp",
            "account_id",
            "merchant_id",
            "amount",
            "channel",
            "merchant_category",
            "country",
            "is_fraud",
        ]
    ]
