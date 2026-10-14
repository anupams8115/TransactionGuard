"""Candidate models and their (bounded) Optuna search spaces.

* dummy            - predicts the training fraud rate for everyone. Any useful model must beat it;
                     its AP equals the prevalence ("no skill" line).
* logreg           - linear baseline; class_weight="balanced" up-weights the rare class in the loss
                     (no synthetic rows are invented, unlike SMOTE).
* xgboost          - gradient-boosted trees; imbalance handled with ``scale_pos_weight`` whose
                     strength (none / sqrt / full ratio) is itself tuned.
* isolation_forest - UNSUPERVISED anomaly detector trained on mixed (unlabelled) training data by
                     default, as it would be when labels are unavailable. Its output is an anomaly
                     score, not a probability. Labels are used only on validation, to pick
                     hyperparameters and a threshold (making the overall procedure semi-supervised).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ModelName = Literal["dummy", "logreg", "xgboost", "isolation_forest"]
SUPERVISED: tuple[ModelName, ...] = ("dummy", "logreg", "xgboost")
# Order = simplicity; used to break near-ties in model selection in favour of simpler models.
SIMPLICITY = {"dummy": 0, "logreg": 1, "isolation_forest": 2, "xgboost": 3}
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 2))


@dataclass(frozen=True)
class Budget:
    n_trials: int
    timeout_s: float  # per model


BUDGETS = {
    "smoke": Budget(n_trials=2, timeout_s=60),
    "small": Budget(n_trials=15, timeout_s=300),
    "medium": Budget(n_trials=40, timeout_s=900),
}


def suggest_params(name: ModelName, trial: optuna.Trial) -> dict[str, Any]:
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "pos_weighting": trial.suggest_categorical("pos_weighting", ["none", "sqrt", "balanced"]),
        }
    if name == "isolation_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_samples": trial.suggest_categorical("max_samples", [256, 1024, 4096]),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    return {}


def default_params(name: ModelName) -> dict[str, Any]:
    return {
        "dummy": {},
        "logreg": {"C": 1.0},
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "pos_weighting": "sqrt",
        },
        "isolation_forest": {"n_estimators": 200, "max_samples": 1024, "max_features": 1.0},
    }[name]


def make_estimator(name: ModelName, params: dict[str, Any], *, y_train: np.ndarray, seed: int) -> Any:
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=3000, random_state=seed)
    if name == "xgboost":
        p = dict(params)
        ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        weighting = p.pop("pos_weighting")
        spw = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": ratio}[weighting]
        return XGBClassifier(
            **p,
            scale_pos_weight=spw,
            tree_method="hist",
            n_jobs=N_JOBS,
            random_state=seed,
            eval_metric="aucpr",
            verbosity=0,
        )
    if name == "isolation_forest":
        return IsolationForest(**params, contamination="auto", random_state=seed, n_jobs=N_JOBS)
    raise ValueError(name)
