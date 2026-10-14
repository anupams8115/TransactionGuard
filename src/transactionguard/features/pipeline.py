"""Fitted preprocessing (scikit-learn ``ColumnTransformer``).

Everything that learns from data lives here and is fitted ONLY on the training split, then saved
inside the model bundle so inference applies byte-identical transformations:

  amount columns : log1p -> median imputation (+ missing flag) -> standard scaling
  numeric columns: median imputation (+ missing flag) -> standard scaling
  categoricals   : one-hot; categories unseen in training map to an all-zeros vector
                   (``handle_unknown="ignore"``) and are *counted* at inference so monitoring
                   can report them rather than hiding them.
  selection      : drop zero-variance columns (fitted on training data).

Missing values only arise from genuinely-absent history (e.g. no previous transaction); the
added missing-indicator column lets the model learn "no history" explicitly instead of us
pretending a value exists.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from transactionguard.features.engineering import FeatureSpec


def _log1p(x: np.ndarray) -> np.ndarray:
    # Module-level function (not a lambda) so the fitted pipeline can be pickled.
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(spec: FeatureSpec, *, scale: bool = True) -> Pipeline:
    scaler = StandardScaler() if scale else "passthrough"
    branches = []
    if spec.amount:
        branches.append(
            (
                "amount",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", scaler),
                    ]
                ),
                list(spec.amount),
            )
        )
    if spec.numeric:
        branches.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", scaler),
                    ]
                ),
                list(spec.numeric),
            )
        )
    if spec.categorical:
        branches.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                list(spec.categorical),
            )
        )
    ct = ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("columns", ct), ("select", VarianceThreshold(threshold=0.0))]).set_output(
        transform="pandas"
    )


def transformed_to_original(preprocessor: Pipeline, spec: FeatureSpec) -> dict[str, str]:
    """Map every transformed column name (e.g. ``cat__channel_online``,
    ``num__missingindicator_secs_since_prev_txn``) back to the raw feature it came from.

    SHAP values are additive, so contributions of all columns from one raw feature (one-hot
    group, value + missing flag) can be summed into one grouped contribution.
    """
    names = list(preprocessor.get_feature_names_out())
    mapping: dict[str, str] = {}
    # Longest-first so a feature named "amount" does not swallow "amount_to_acct_mean_30d".
    raw = sorted(spec.all, key=len, reverse=True)
    for name in names:
        _, _, rest = name.partition("__")
        rest = rest.removeprefix("missingindicator_")
        for feat in raw:
            if rest == feat or (feat in spec.categorical and rest.startswith(feat + "_")):
                mapping[name] = feat
                break
        else:  # pragma: no cover - would indicate a naming bug; fail loudly
            raise ValueError(f"Cannot map transformed feature {name!r} to a raw feature")
    return mapping
