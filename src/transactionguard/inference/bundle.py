"""Model bundles: preprocessing + model + calibrator + threshold, versioned together.

Saving all of these as ONE artifact guarantees that inference applies exactly the transformations
the model was trained with, and that every prediction can be traced to a model version, its
preprocessing, the threshold and the training run that produced them.

Security: joblib/pickle can execute code when loading, so we ONLY load bundles that
  1. live under the configured ``bundles_dir`` (no arbitrary paths), and
  2. are listed in ``registry.json`` (written only by the training CLI), and
  3. match the SHA-256 recorded at training time.
There is deliberately no API endpoint that accepts model files from users.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from transactionguard.data.contract import DataContract, get_contract
from transactionguard.features.engineering import FEATURE_SPECS, FeatureSpec, build_features
from transactionguard.training.evaluation import PlattCalibrator, logit, sigmoid

ScoreKind = Literal["calibrated_probability", "anomaly_score", "constant_prior"]

SCORE_MEANINGS: dict[str, str] = {
    "calibrated_probability": (
        "Estimated probability that the source would label this transaction as fraud, after "
        "Platt calibration on validation data. A risk signal for review, not proof of fraud."
    ),
    "anomaly_score": (
        "Isolation Forest anomaly score (higher = more unusual relative to training data). "
        "NOT a probability and not comparable in scale with probability models."
    ),
    "constant_prior": "Constant training-set fraud rate (baseline model; ignores inputs).",
}


class UntrustedArtifactError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    model_version: str
    model_name: str
    contract_name: str
    score_kind: ScoreKind
    pipeline: Pipeline  # steps: ("pre", fitted preprocessor), ("model", fitted estimator)
    threshold: float
    threshold_policy: dict
    calibrator: PlattCalibrator | None = None
    params: dict = field(default_factory=dict)
    # Small sample of *transformed* training rows (background data for SHAP LinearExplainer).
    shap_background: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- derived properties
    @property
    def contract(self) -> DataContract:
        return get_contract(self.contract_name)

    @property
    def feature_spec(self) -> FeatureSpec:
        return FEATURE_SPECS[self.contract_name]

    @property
    def preprocessor(self) -> Pipeline:
        return self.pipeline.named_steps["pre"]

    @property
    def estimator(self) -> Any:
        return self.pipeline.named_steps["model"]

    @property
    def score_meaning(self) -> str:
        return SCORE_MEANINGS[self.score_kind]

    @property
    def preprocessing_version(self) -> str:
        return self.metadata.get("preprocessing_version", "unknown")

    # ---------------------------------------------------------------- scoring
    def features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw contract columns -> engineered model inputs (stateless; see engineering.py)."""
        return build_features(raw, self.contract)[self.feature_spec.all]

    def raw_output(self, X: pd.DataFrame) -> np.ndarray:
        """The model's native output: log-odds for classifiers, anomaly score for IForest."""
        Xt = self.preprocessor.transform(X)
        return self.raw_output_transformed(Xt)

    def raw_output_transformed(self, Xt: pd.DataFrame) -> np.ndarray:
        est = self.estimator
        if self.score_kind == "anomaly_score":
            return -est.score_samples(Xt)  # flip so that higher = more anomalous
        if hasattr(est, "decision_function"):  # logistic regression: exact log-odds
            return np.asarray(est.decision_function(Xt), dtype=float)
        if est.__class__.__name__ == "XGBClassifier":
            return np.asarray(est.predict(Xt, output_margin=True), dtype=float)
        return logit(est.predict_proba(Xt)[:, 1])

    def score_from_raw(self, raw: np.ndarray) -> np.ndarray:
        if self.score_kind == "calibrated_probability":
            assert self.calibrator is not None
            return self.calibrator.transform(raw)
        if self.score_kind == "constant_prior":
            return sigmoid(raw)
        return raw

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.score_from_raw(self.raw_output(X))

    def flag(self, scores: np.ndarray) -> np.ndarray:
        return scores >= self.threshold

    def summary(self) -> dict:
        return {
            "model_version": self.model_version,
            "model_name": self.model_name,
            "contract": self.contract_name,
            "score_kind": self.score_kind,
            "score_meaning": self.score_meaning,
            "threshold": self.threshold,
            "threshold_policy": self.threshold_policy,
            "preprocessing_version": self.preprocessing_version,
            "calibration": None
            if self.calibrator is None
            else {"method": "platt_on_logodds", "a": self.calibrator.a, "b": self.calibrator.b},
            **{k: v for k, v in self.metadata.items() if k not in {"global_importance"}},
        }


# ==================================================================================== registry
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BundleRegistry:
    """Tiny file-based registry: ``<bundles_dir>/registry.json`` + one folder per version."""

    def __init__(self, bundles_dir: Path):
        self.dir = Path(bundles_dir).resolve()
        self.path = self.dir / "registry.json"

    def read(self) -> dict:
        if not self.path.exists():
            return {"bundles": {}, "active": {}}
        return json.loads(self.path.read_text())

    def _write(self, reg: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a crash never leaves a half-written registry.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(reg, fh, indent=2, default=str)
        os.replace(tmp, self.path)

    def save(self, bundle: ModelBundle, *, activate: bool = False) -> Path:
        folder = self.dir / bundle.model_version
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / "bundle.joblib"
        if file.exists():
            raise FileExistsError(f"{bundle.model_version} already saved; versions are immutable")
        joblib.dump(bundle, file, compress=3)
        reg = self.read()
        reg["bundles"][bundle.model_version] = {
            "file": f"{bundle.model_version}/bundle.joblib",
            "sha256": _sha256(file),
            "saved_at": datetime.now(UTC).isoformat(),
            **{k: v for k, v in bundle.summary().items() if k != "model_version"},
        }
        if activate:
            reg["active"][bundle.contract_name] = bundle.model_version
        self._write(reg)
        return file

    def annotate(self, model_version: str, **fields: object) -> None:
        """Attach metadata known only after saving (e.g. the MLflow run id)."""
        reg = self.read()
        reg["bundles"][model_version].update(fields)
        self._write(reg)

    def activate(self, model_version: str) -> None:
        reg = self.read()
        if model_version not in reg["bundles"]:
            raise KeyError(model_version)
        reg["active"][reg["bundles"][model_version]["contract"]] = model_version
        self._write(reg)

    def load(self, model_version: str) -> ModelBundle:
        reg = self.read()
        entry = reg["bundles"].get(model_version)
        if entry is None:
            raise UntrustedArtifactError(f"{model_version!r} is not in the trusted registry")
        file = (self.dir / entry["file"]).resolve()
        if not file.is_relative_to(self.dir):
            raise UntrustedArtifactError("bundle path escapes the bundles directory")
        if _sha256(file) != entry["sha256"]:
            raise UntrustedArtifactError(f"checksum mismatch for {model_version}; refusing to load")
        bundle = joblib.load(file)  # safe only because of the checks above
        if not isinstance(bundle, ModelBundle) or bundle.model_version != model_version:
            raise UntrustedArtifactError("artifact content does not match its registry entry")
        return bundle

    def active_version(self, contract_name: str) -> str | None:
        return self.read()["active"].get(contract_name)
