"""Train / validation / test splitting.

Default: **chronological** (train on the past, validate on the next period, test on the most
recent period). That mirrors deployment, where a model trained today scores tomorrow's traffic.
A random split would let the model "see the future" (e.g. the same fraud campaign on both sides).

Optional: **group** split by account (synthetic only) to estimate generalisation to accounts the
model has never seen. History features still use only each account's own past.

Roles (enforced by the training code, see ``train.py``):
  train      -> fit preprocessing + models
  validation -> Optuna tuning, model selection, calibration, threshold selection
  test       -> touched ONCE at the end for reporting; never used for any decision
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from transactionguard.data.contract import DataContract

Strategy = Literal["chronological", "group"]


@dataclass(frozen=True)
class SplitConfig:
    strategy: Strategy = "chronological"
    train_frac: float = 0.6
    valid_frac: float = 0.2
    # Optional gap (seconds) dropped between periods, e.g. to mimic label delay. Rows inside the
    # gap are excluded from both neighbouring sets.
    gap_seconds: float = 0.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    description: dict

    def check_disjoint(self) -> None:
        a, b, c = set(self.train), set(self.valid), set(self.test)
        assert not (a & b or a & c or b & c), "split index sets overlap"


def _time_seconds(df: pd.DataFrame, contract: DataContract) -> np.ndarray:
    ts = df[contract.timestamp_column]
    if pd.api.types.is_numeric_dtype(ts):
        return ts.to_numpy(np.float64)
    return (
        (pd.to_datetime(ts, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    ).to_numpy(np.float64)


def make_split(df: pd.DataFrame, contract: DataContract, cfg: SplitConfig) -> Split:
    """Return positional indices into ``df``."""
    t = _time_seconds(df, contract)
    y = df[contract.target_column].to_numpy()
    if cfg.strategy == "chronological":
        # Cut on *time* quantiles of the row order so equal timestamps never straddle a boundary.
        t1 = np.quantile(t, cfg.train_frac, method="lower")
        t2 = np.quantile(t, cfg.train_frac + cfg.valid_frac, method="lower")
        g = cfg.gap_seconds
        train = np.flatnonzero(t <= t1)
        valid = np.flatnonzero((t > t1 + g) & (t <= t2))
        test = np.flatnonzero(t > t2 + g)
        bounds = {"train_end": float(t1), "valid_end": float(t2)}
    elif cfg.strategy == "group":
        if contract.entity_column is None:
            raise ValueError(f"{contract.name} has no entity column; group split impossible")
        groups = df[contract.entity_column].to_numpy()
        idx = np.arange(len(df))
        outer = GroupShuffleSplit(n_splits=1, train_size=cfg.train_frac, random_state=cfg.seed)
        train, rest = next(outer.split(idx, groups=groups))
        inner_frac = cfg.valid_frac / (1 - cfg.train_frac)
        inner = GroupShuffleSplit(n_splits=1, train_size=inner_frac, random_state=cfg.seed + 1)
        v, te = next(inner.split(rest, groups=groups[rest]))
        valid, test = rest[v], rest[te]
        bounds = {}
    else:  # pragma: no cover
        raise ValueError(cfg.strategy)

    def _stats(ix: np.ndarray) -> dict:
        return {
            "rows": len(ix),
            "positives": int(y[ix].sum()),
            "prevalence": float(y[ix].mean()) if len(ix) else 0.0,
            "time_start": float(t[ix].min()) if len(ix) else None,
            "time_end": float(t[ix].max()) if len(ix) else None,
        }

    split = Split(
        train=train,
        valid=valid,
        test=test,
        description={
            "config": cfg.to_dict(),
            **bounds,
            "train": _stats(train),
            "valid": _stats(valid),
            "test": _stats(test),
        },
    )
    split.check_disjoint()
    for name in ("train", "valid", "test"):
        if split.description[name]["positives"] < 5:
            raise ValueError(f"{name} split has <5 positives; metrics would be meaningless")
    return split
