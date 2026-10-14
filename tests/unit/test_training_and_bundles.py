"""Training session, test-set isolation, trusted model loading and explanations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from transactionguard.core.config import get_settings
from transactionguard.data import sources
from transactionguard.inference.bundle import BundleRegistry, UntrustedArtifactError
from transactionguard.inference.explain import BundleExplainer


def _registry() -> BundleRegistry:
    return BundleRegistry(get_settings().bundles_dir)


def test_training_report_is_complete(trained_small):
    r = trained_small
    assert r["synthetic"] is True
    names = {x["model"] for x in r["results"]}
    assert names == {"dummy", "logreg", "xgboost", "isolation_forest"}
    for res in r["results"]:
        for key in (
            "average_precision",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "roc_auc",
            "confusion_matrix",
            "prevalence",
            "n",
            "positives",
        ):
            assert key in res["test"]
    dummy = next(x for x in r["results"] if x["model"] == "dummy")
    assert dummy["test"]["average_precision"] == pytest.approx(dummy["test"]["prevalence"], rel=0.05)
    iforest = next(x for x in r["results"] if x["model"] == "isolation_forest")
    assert iforest["score_kind"] == "anomaly_score" and iforest["test_calibration"] is None


def test_bootstrap_intervals_reported(trained_small):
    u = trained_small["test_uncertainty"]
    assert set(u["models"]) == {"logreg", "xgboost", "isolation_forest"}
    for res in trained_small["results"]:
        if res["model"] != "dummy":
            lo, hi = res["test"]["average_precision_ci95"]
            assert lo <= hi


def test_selection_uses_validation_not_test(trained_small):
    r = trained_small
    best_val = max(v for k, v in r["selection"]["validation_ap"].items() if k in ("logreg", "xgboost"))
    sel = r["selection"]["selected"]
    assert r["selection"]["validation_ap"][sel] >= best_val - 0.005


def test_test_labels_cannot_change_any_decision(prepared_small, monkeypatch):
    """Shuffle the TEST-period labels: params, thresholds, calibration and the selected model
    must be identical, proving the test set influenced nothing but the reported metrics."""
    from transactionguard.training.splits import SplitConfig, make_split
    from transactionguard.training.train import TrainConfig, train

    cfg = TrainConfig(
        dataset="synthetic",
        budget="smoke",
        log_mlflow=False,
        activate=False,
        n_bootstrap=20,
        models=("dummy", "logreg", "xgboost"),
        split=SplitConfig(seed=7),
        seed=7,
    )
    first = train(cfg)

    df = sources.load("synthetic").sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    te = make_split(df, sources.SYNTHETIC, cfg.split).test
    rng = np.random.default_rng(123)
    df.loc[te, "is_fraud"] = rng.permutation(df.loc[te, "is_fraud"].to_numpy())
    original_load = sources.load
    monkeypatch.setattr(sources, "load", lambda name: df if name == "synthetic" else original_load(name))
    second = train(cfg)

    assert first["selection"] == second["selection"]
    for a, b in zip(first["results"], second["results"], strict=True):
        assert a["params"] == b["params"]
        assert a["threshold"] == b["threshold"]
        assert a["validation"] == b["validation"]
    assert first["results"][1]["test"] != second["results"][1]["test"]  # sanity: test did change


def test_bundle_round_trip_reproduces_scores(trained_small):
    reg = _registry()
    version = reg.active_version("synthetic")
    b = reg.load(version)
    raw = sources.load("synthetic").tail(500)
    s1 = b.score(b.features(raw))
    s2 = reg.load(version).score(b.features(raw))
    np.testing.assert_allclose(s1, s2)
    assert ((s1 >= 0) & (s1 <= 1)).all()


def test_tampered_bundle_is_refused(trained_small):
    reg = _registry()
    version = reg.active_version("synthetic")
    path = get_settings().bundles_dir / version / "bundle.joblib"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(UntrustedArtifactError, match="checksum"):
        reg.load(version)


def test_unregistered_or_escaping_bundles_are_refused(trained_small):
    reg = _registry()
    with pytest.raises(UntrustedArtifactError):
        reg.load("not-a-registered-model")
    data = reg.read()
    version = next(iter(data["bundles"]))
    data["bundles"][version]["file"] = "../../evil.joblib"
    reg.path.write_text(json.dumps(data))
    with pytest.raises(UntrustedArtifactError, match="escapes"):
        reg.load(version)


def test_threshold_applied_with_greater_or_equal(trained_small):
    b = _registry().load(_registry().active_version("synthetic"))
    s = np.array([b.threshold - 1e-9, b.threshold, b.threshold + 1e-9])
    assert b.flag(s).tolist() == [False, True, True]


# ------------------------------------------------------------------------------- explanations
@pytest.mark.parametrize("model", ["xgboost", "logreg", "isolation_forest"])
def test_explanations_are_additive_and_grouped(trained_small, model):
    reg = _registry()
    version = next(v for v, e in reg.read()["bundles"].items() if e["model_name"] == model)
    b = reg.load(version)
    raw = sources.load("synthetic").tail(50)
    X = b.features(raw)
    exps = BundleExplainer(b).explain(X, top_k=1000)
    assert all(e.status == "ok" for e in exps), [e.reason for e in exps]
    from transactionguard.inference.explain import explained_output

    expected = explained_output(b, b.preprocessor.transform(X))
    for e, exp_out in zip(exps, expected, strict=True):
        total = e.base_value + sum(c.contribution for c in e.contributions)
        assert total == pytest.approx(exp_out, abs=1e-3)
        assert e.output_scale
        feats = {c.feature for c in e.contributions}
        assert not any(f.startswith("cat__") for f in feats)  # grouped back to raw names
        channel = next(c for c in e.contributions if c.feature == "channel")
        if channel.components:
            assert channel.contribution == pytest.approx(sum(p["contribution"] for p in channel.components))
    if b.score_kind == "calibrated_probability":
        probs = b.score(X)
        np.testing.assert_allclose(1 / (1 + np.exp(-expected)), probs, rtol=1e-6)


def test_explainer_failure_is_reported_not_invented(trained_small, monkeypatch):
    b = _registry().load(_registry().active_version("synthetic"))
    ex = BundleExplainer(b)

    def boom(*_a, **_k):
        raise RuntimeError("explainer crashed")

    monkeypatch.setattr(ex, "explainer", boom)
    out = ex.explain(b.features(sources.load("synthetic").tail(2)))
    assert all(e.status == "failed" and "crashed" in e.reason and not e.contributions for e in out)


def test_global_importance_saved_with_bundle(trained_small):
    b = _registry().load(_registry().active_version("synthetic"))
    gi = b.metadata["global_importance"]
    assert gi and gi == sorted(gi, key=lambda r: -r["mean_abs_contribution"])
    assert isinstance(pd.DataFrame(gi), pd.DataFrame)
