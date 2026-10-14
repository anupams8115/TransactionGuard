"""SHAP explanations (local + global), grouped back to raw features.

What SHAP gives us: for one prediction, numbers phi_j such that
    model_output(x) = base_value + sum_j phi_j
i.e. how much each feature pushed the output above/below the average (base) output.

Output scale (always reported alongside the numbers):
  * classifiers -> **calibrated log-odds**. SHAP explains the model's raw log-odds; our Platt
    calibrator is affine (z_cal = a*z + b), so phi_j * a are exact contributions on the calibrated
    log-odds scale with base a*base + b. Probability = sigmoid(base + sum(phi)).
    We do NOT split contributions on the probability scale: sigmoid is non-linear, so per-feature
    "probability points" are not additive.
  * isolation forest -> log2(anomaly score). The IF score is 2^(-E[path length]/c(n)), which is
    exponential in the path length SHAP explains; on the log2 scale the relation is exactly
    affine, so contributions are additive and verifiable.

Grouping: one raw feature may become several model columns (one-hot categories, value + missing
flag). SHAP values are additive, so we sum them per raw feature and keep the parts for detail.

Honesty rules: SHAP describes the *model's* behaviour, not causes in the world, and it never
proves fraud. If the explainer fails or its values do not add up to the model output, we return
status="failed" with the reason - we never fabricate an explanation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from transactionguard.core.logging import get_logger
from transactionguard.features.engineering import FEATURE_SPECS
from transactionguard.features.pipeline import transformed_to_original
from transactionguard.inference.bundle import ModelBundle

log = get_logger(__name__)
Status = Literal["ok", "unavailable", "failed"]
ADDITIVITY_TOL = 1e-3
DISCLAIMER = (
    "Contributions describe how this model arrived at its score. They are not causes and do not prove fraud."
)


@dataclass
class FeatureContribution:
    feature: str
    value: Any  # raw (pre-scaling) feature value shown to the analyst
    contribution: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]
    description: str
    components: list[dict] = field(default_factory=list)  # per transformed column


@dataclass
class Explanation:
    status: Status
    method: str | None = None
    output_scale: str | None = None
    base_value: float | None = None
    output_value: float | None = None
    contributions: list[FeatureContribution] = field(default_factory=list)
    summary: str | None = None
    reason: str | None = None
    additivity_error: float | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def _json_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


class BundleExplainer:
    """Builds the right SHAP explainer once per bundle and reuses it."""

    def __init__(self, bundle: ModelBundle):
        import shap

        self.bundle = bundle
        self.spec = FEATURE_SPECS[bundle.contract_name]
        self.mapping = transformed_to_original(bundle.preprocessor, self.spec)
        est = bundle.estimator
        self.explainer: Any = None
        self.unavailable_reason: str | None = None
        name = bundle.model_name
        if name == "xgboost":
            self.explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
            self.method = "shap.TreeExplainer (tree_path_dependent, raw log-odds)"
        elif name == "logreg":
            if bundle.shap_background is None:
                self.unavailable_reason = "bundle has no background sample for LinearExplainer"
            else:
                masker = shap.maskers.Independent(bundle.shap_background, max_samples=200)
                self.explainer = shap.LinearExplainer(est, masker)
                self.method = "shap.LinearExplainer (interventional, raw log-odds)"
        elif name == "isolation_forest":
            self.explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
            self.method = "shap.TreeExplainer on IsolationForest (anomaly score)"
        else:
            self.unavailable_reason = f"no explainer for model type {name!r} (it ignores its inputs)"

    # -------------------------------------------------------------------------------- core
    def _shap_on_output_scale(self, Xt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
        """Return (phi[n, k], base[n], scale) such that base + phi.sum(1) == bundle output."""
        b = self.bundle
        sv = self.explainer(Xt)
        phi = np.asarray(sv.values, dtype=float)
        base = np.broadcast_to(np.asarray(sv.base_values, dtype=float).reshape(-1), (len(Xt),)).copy()
        if b.model_name == "isolation_forest":
            # SHAP explains IsolationForest's internal path-length output. Training stores the
            # affine map from that output to our anomaly score (see fit_iforest_shap_affine);
            # the additivity check in explain() verifies it is exact for every row.
            affine = b.metadata.get("iforest_shap_affine")
            if affine is None:
                raise RuntimeError("bundle lacks the SHAP->anomaly-score mapping")
            slope, intercept = affine
            return (
                phi * slope,
                base * slope + intercept,
                "log2(anomaly score) = -(average path length)/c(n); higher = more unusual",
            )
        assert b.calibrator is not None
        a, c = b.calibrator.a, b.calibrator.b
        return phi * a, base * a + c, "calibrated log-odds (probability = sigmoid(base + sum))"

    def _expected_output(self, Xt: pd.DataFrame) -> np.ndarray:
        return explained_output(self.bundle, Xt)

    def explain(self, X: pd.DataFrame, top_k: int = 8) -> list[Explanation]:
        """Explain each row of engineered features ``X`` (output of ``bundle.features``)."""
        if self.explainer is None:
            return [Explanation(status="unavailable", reason=self.unavailable_reason)] * len(X)
        try:
            Xt = self.bundle.preprocessor.transform(X)
            phi, base, scale = self._shap_on_output_scale(Xt)
            expected = self._expected_output(Xt)
        except Exception as exc:  # explainer failure must be visible, never invented
            log.warning("explainer_failed", model_version=self.bundle.model_version, error=str(exc))
            return [
                Explanation(status="failed", method=self.method, reason=f"{type(exc).__name__}: {exc}")
            ] * len(X)

        cols = list(Xt.columns)
        results = []
        for i in range(len(X)):
            total = base[i] + phi[i].sum()
            err = float(abs(total - expected[i]))
            if err > ADDITIVITY_TOL * max(1.0, abs(expected[i])):
                results.append(
                    Explanation(
                        status="failed",
                        method=self.method,
                        additivity_error=err,
                        reason=f"SHAP values do not add up to the model output (error {err:.4g})",
                    )
                )
                continue
            grouped: dict[str, list[tuple[str, float]]] = {}
            for j, col in enumerate(cols):
                grouped.setdefault(self.mapping[col], []).append((col, float(phi[i, j])))
            contribs = []
            for feat, parts in grouped.items():
                total_c = sum(p for _, p in parts)
                contribs.append(
                    FeatureContribution(
                        feature=feat,
                        value=_json_value(X.iloc[i][feat]),
                        contribution=total_c,
                        direction="increases_risk"
                        if total_c > 1e-9
                        else "decreases_risk"
                        if total_c < -1e-9
                        else "neutral",
                        description=self.spec.descriptions.get(feat, feat),
                        components=[{"column": c, "contribution": p} for c, p in parts]
                        if len(parts) > 1
                        else [],
                    )
                )
            contribs.sort(key=lambda c: abs(c.contribution), reverse=True)
            results.append(
                Explanation(
                    status="ok",
                    method=self.method,
                    output_scale=scale,
                    base_value=float(base[i]),
                    output_value=float(total),
                    contributions=contribs[:top_k],
                    additivity_error=err,
                    summary=plain_language(contribs[:top_k], scale),
                )
            )
        return results

    def global_importance(self, X: pd.DataFrame) -> list[dict]:
        """Mean |grouped contribution| over a sample (a global summary, not a causal ranking)."""
        exps = self.explain(X, top_k=10_000)
        ok = [e for e in exps if e.status == "ok"]
        if not ok:
            return []
        acc: dict[str, list[float]] = {}
        for e in ok:
            for c in e.contributions:
                acc.setdefault(c.feature, []).append(c.contribution)
        rows = [
            {
                "feature": f,
                "mean_abs_contribution": float(np.mean(np.abs(v))),
                "mean_contribution": float(np.mean(v)),
                "n": len(v),
            }
            for f, v in acc.items()
        ]
        return sorted(rows, key=lambda r: r["mean_abs_contribution"], reverse=True)


def explained_output(bundle: ModelBundle, Xt: pd.DataFrame) -> np.ndarray:
    """The quantity whose SHAP decomposition we report (see module docstring)."""
    raw = bundle.raw_output_transformed(Xt)
    if bundle.score_kind == "calibrated_probability":
        assert bundle.calibrator is not None
        return bundle.calibrator.calibrated_logodds(raw)
    if bundle.score_kind == "anomaly_score":
        return np.log2(raw)
    return raw


def fit_iforest_shap_affine(bundle: ModelBundle, X: pd.DataFrame) -> list[float] | None:
    """Fit log2(anomaly_score) = slope * shap_output + intercept (training-time only)."""
    import shap

    Xt = bundle.preprocessor.transform(X)
    sv = shap.TreeExplainer(bundle.estimator, feature_perturbation="tree_path_dependent")(Xt)
    shap_out = np.asarray(sv.base_values, dtype=float).reshape(-1) + np.asarray(sv.values).sum(1)
    target = explained_output(bundle, Xt)
    if np.ptp(shap_out) == 0:
        return None
    slope, intercept = np.polyfit(shap_out, target, 1)
    resid = np.abs(slope * shap_out + intercept - target).max()
    return [float(slope), float(intercept)] if resid < ADDITIVITY_TOL else None


def _fmt(v: Any) -> str:
    if v is None:
        return "missing (no history)"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def plain_language(contribs: list[FeatureContribution], scale: str) -> str:
    """Template text built strictly from the computed contributions (no invented reasons)."""
    up = [c for c in contribs if c.direction == "increases_risk"][:3]
    down = [c for c in contribs if c.direction == "decreases_risk"][:2]
    parts = []
    if up:
        parts.append(
            "Pushed the score up: "
            + "; ".join(f"{c.feature} = {_fmt(c.value)} ({c.contribution:+.2f})" for c in up)
            + "."
        )
    if down:
        parts.append(
            "Pushed the score down: "
            + "; ".join(f"{c.feature} = {_fmt(c.value)} ({c.contribution:+.2f})" for c in down)
            + "."
        )
    if not parts:
        return "No feature moved the score noticeably from the base value."
    return " ".join(parts) + f" Units: {scale}."
