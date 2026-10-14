"""End-to-end training session: data -> features -> split -> tune -> calibrate -> threshold ->
select -> (only then) test -> save bundles -> MLflow.

The order of operations IS the leakage protection:
  1. features are stateless/past-only, so computing them before splitting is safe;
  2. preprocessing is fitted on train only;
  3. tuning, calibration, threshold and model selection use validation only;
  4. the test set is scored once, after every decision is frozen, and only for reporting.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from transactionguard.core.config import PROJECT_ROOT, get_settings
from transactionguard.core.logging import get_logger
from transactionguard.data import sources
from transactionguard.data.contract import get_contract
from transactionguard.features.engineering import FEATURE_SPECS, build_features
from transactionguard.features.pipeline import build_preprocessor
from transactionguard.inference.bundle import BundleRegistry, ModelBundle
from transactionguard.training import models as M
from transactionguard.training.evaluation import (
    PlattCalibrator,
    ThresholdPolicy,
    bootstrap_ap,
    calibration_report,
    choose_threshold,
    classification_metrics,
    pr_curve_points,
)
from transactionguard.training.splits import SplitConfig, make_split

log = get_logger(__name__)
PREPROCESSING_VERSION = "pp-1.0"  # bump when features/engineering.py or pipeline.py changes


@dataclass
class TrainConfig:
    dataset: str = "synthetic"
    models: tuple[str, ...] = ("dummy", "logreg", "xgboost", "isolation_forest")
    budget: str = "small"
    split: SplitConfig = field(default_factory=SplitConfig)
    policy: ThresholdPolicy = field(default_factory=ThresholdPolicy)
    iforest_training_data: str = "mixed"  # "mixed" (unlabelled) | "normal_only"
    selection_tolerance: float = 0.005  # prefer a simpler model within this validation-AP margin
    n_bootstrap: int = 500
    seed: int = 42
    activate: bool = True
    log_mlflow: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        return d


def code_revision() -> dict:
    # `git` is resolved from PATH on purpose (developer machines and CI differ); args are constant.
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_sha": sha, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "git_dirty": None, "note": "no git commit available"}


def _fit_candidate(
    name: str, params: dict, Xt: pd.DataFrame, y: np.ndarray, seed: int, iforest_data: str
) -> Any:
    est = M.make_estimator(name, params, y_train=y, seed=seed)  # type: ignore[arg-type]
    if name == "isolation_forest":
        Xfit = Xt[y == 0] if iforest_data == "normal_only" else Xt
        return est.fit(Xfit)
    return est.fit(Xt, y)


def _raw(est: Any, name: str, Xt: pd.DataFrame) -> np.ndarray:
    tmp = ModelBundle(
        "tmp",
        name,
        "synthetic",
        "anomaly_score" if name == "isolation_forest" else "calibrated_probability",
        Pipeline([("pre", "passthrough"), ("model", est)]),
        threshold=0.0,
        threshold_policy={},
    )
    return tmp.raw_output_transformed(Xt)


def _latency(bundle: ModelBundle, raw_rows: pd.DataFrame, n_single: int = 200) -> dict:
    """Wall-clock latency of feature build + preprocessing + model + calibration (no DB, no SHAP)."""
    sample = raw_rows.head(n_single)
    times = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        bundle.score(bundle.features(sample.iloc[[i]]))
        times.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    bundle.score(bundle.features(raw_rows))
    batch_s = time.perf_counter() - t0
    return {
        "single_row_ms_p50": float(np.percentile(times, 50)),
        "single_row_ms_p95": float(np.percentile(times, 95)),
        "batch_rows": len(raw_rows),
        "batch_rows_per_s": float(len(raw_rows) / batch_s),
        "note": "In-process scoring only: excludes HTTP, database history lookup and SHAP. "
        "Account-history features in batch mode are computed within the batch.",
    }


def train(cfg: TrainConfig) -> dict:
    settings = get_settings()
    contract = get_contract(cfg.dataset)
    spec = FEATURE_SPECS[cfg.dataset]
    budget = M.BUDGETS[cfg.budget]
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    t_start = time.perf_counter()

    # ---------------------------------------------------------------- 1. data + features
    df = sources.load(cfg.dataset)
    fingerprint = sources.fingerprint(df)
    sort_cols = [contract.timestamp_column, contract.id_column]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    feats = build_features(df, contract)
    X, y = feats[spec.all], df[contract.target_column].astype(int).to_numpy()

    # ---------------------------------------------------------------- 2. split
    split = make_split(df, contract, cfg.split)
    tr, va, te = split.train, split.valid, split.test
    t_col = df[contract.timestamp_column]
    span = t_col.iloc[va].max() - t_col.iloc[va].min()
    valid_days = (span.total_seconds() if hasattr(span, "total_seconds") else float(span)) / 86_400
    log.info(
        "split_ready",
        dataset=cfg.dataset,
        strategy=cfg.split.strategy,
        train=len(tr),
        valid=len(va),
        test=len(te),
    )

    # ---------------------------------------------------------------- 3. preprocessing (train only)
    pre = build_preprocessor(spec, scale=True).fit(X.iloc[tr])
    Xt_tr, Xt_va = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])

    # ---------------------------------------------------------------- 4. tune + calibrate + threshold
    candidates: dict[str, dict] = {}
    for name in cfg.models:
        t0 = time.perf_counter()
        if name == "dummy":
            best_params, n_trials = {}, 0
        else:

            def objective(trial: optuna.Trial, name: str = name) -> float:
                params = M.suggest_params(name, trial)  # type: ignore[arg-type]
                est = _fit_candidate(name, params, Xt_tr, y[tr], cfg.seed, cfg.iforest_training_data)
                return float(average_precision_score(y[va], _raw(est, name, Xt_va)))

            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
            )
            study.enqueue_trial(M.default_params(name))  # always evaluate sensible defaults
            study.optimize(objective, n_trials=budget.n_trials, timeout=budget.timeout_s)
            best_params, n_trials = study.best_params, len(study.trials)

        est = _fit_candidate(name, best_params, Xt_tr, y[tr], cfg.seed, cfg.iforest_training_data)
        raw_va = _raw(est, name, Xt_va)
        if name == "isolation_forest":
            score_kind, calibrator, score_va = "anomaly_score", None, raw_va
        elif name == "dummy":
            score_kind, calibrator = "constant_prior", None
            score_va = 1 / (1 + np.exp(-raw_va))
        else:
            score_kind = "calibrated_probability"
            calibrator = PlattCalibrator().fit(raw_va, y[va])
            score_va = calibrator.transform(raw_va)
        choice = choose_threshold(y[va], score_va, cfg.policy, period_days=valid_days)
        candidates[name] = {
            "name": name,
            "estimator": est,
            "params": best_params,
            "n_trials": n_trials,
            "score_kind": score_kind,
            "calibrator": calibrator,
            "threshold": choice,
            "valid_metrics": choice.validation_metrics,
            "valid_calibration": calibration_report(y[va], score_va)
            if score_kind == "calibrated_probability"
            else None,
            "fit_seconds": time.perf_counter() - t0,
        }
        log.info(
            "candidate_done",
            model=name,
            valid_ap=round(choice.validation_metrics["average_precision"], 4),
            threshold=round(choice.threshold, 6),
            trials=n_trials,
        )

    # ---------------------------------------------------------------- 5. select (validation only)
    eligible = [c for c in candidates.values() if c["score_kind"] == "calibrated_probability"]
    if not eligible:
        raise RuntimeError("No supervised probability model was trained; cannot select a model.")
    best_ap = max(c["valid_metrics"]["average_precision"] for c in eligible)
    near = [
        c for c in eligible if c["valid_metrics"]["average_precision"] >= best_ap - cfg.selection_tolerance
    ]
    selected = min(near, key=lambda c: M.SIMPLICITY[c["name"]])
    selection = {
        "rule": f"highest validation average precision; among models within "
        f"{cfg.selection_tolerance} AP of the best, prefer the simpler one",
        "selected": selected["name"],
        "validation_ap": {c["name"]: c["valid_metrics"]["average_precision"] for c in candidates.values()},
    }
    log.info("model_selected", **{"model": selected["name"]})

    # ---------------------------------------------------------------- 6. build bundles
    revision = code_revision()
    bg_idx = np.random.default_rng(cfg.seed).choice(len(tr), size=min(200, len(tr)), replace=False)
    bundles: dict[str, ModelBundle] = {}
    for name, c in candidates.items():
        version = f"{cfg.dataset}-{name}-{session_id}"
        bundle = ModelBundle(
            model_version=version,
            model_name=name,
            contract_name=cfg.dataset,
            score_kind=c["score_kind"],
            pipeline=Pipeline([("pre", pre), ("model", c["estimator"])]),
            threshold=c["threshold"].threshold,
            threshold_policy={
                **c["threshold"].policy,
                "constraint_met": c["threshold"].constraint_met,
                "note": c["threshold"].note,
                "selected_on": "validation",
            },
            calibrator=c["calibrator"],
            params=c["params"],
            shap_background=Xt_tr.iloc[bg_idx].reset_index(drop=True),
            metadata={
                "training_session": session_id,
                "dataset_fingerprint": fingerprint,
                "dataset_provenance": contract.provenance,
                "preprocessing_version": PREPROCESSING_VERSION,
                "features": spec.all,
                "split": split.description,
                "seed": cfg.seed,
                "code_revision": revision,
                "selected": name == selected["name"],
            },
        )
        bundles[name] = bundle

    # ---------------------------------------------------------------- 7. explanations (validation sample)
    from transactionguard.inference.explain import BundleExplainer, fit_iforest_shap_affine

    rng = np.random.default_rng(cfg.seed)
    ex_idx = va[rng.choice(len(va), size=min(500, len(va)), replace=False)]
    for name, bundle in bundles.items():
        if name == "isolation_forest":
            bundle.metadata["iforest_shap_affine"] = fit_iforest_shap_affine(bundle, X.iloc[ex_idx])
        if name == "dummy":
            continue
        try:
            bundle.metadata["global_importance"] = BundleExplainer(bundle).global_importance(X.iloc[ex_idx])
        except Exception as exc:  # recorded, not hidden
            bundle.metadata["global_importance_error"] = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------- 8. TEST: first and only touch
    results = []
    for name, c in candidates.items():
        b = bundles[name]
        score_te = b.score(X.iloc[te])
        test_metrics = classification_metrics(y[te], score_te, b.threshold)
        results.append(
            {
                "model": name,
                "model_version": b.model_version,
                "score_kind": b.score_kind,
                "score_meaning": b.score_meaning,
                "params": c["params"],
                "optuna_trials": c["n_trials"],
                "threshold": b.threshold,
                "threshold_policy": b.threshold_policy,
                "validation": c["valid_metrics"],
                "validation_calibration": c["valid_calibration"],
                "test": test_metrics,
                "test_calibration": calibration_report(y[te], score_te)
                if b.score_kind == "calibrated_probability"
                else None,
                "test_pr_curve": pr_curve_points(y[te], score_te),
                "latency": _latency(b, df.iloc[te]) if name != "dummy" else None,
                "global_importance": b.metadata.get("global_importance", []),
                "fit_seconds": c["fit_seconds"],
                "selected": name == selected["name"],
            }
        )
        b.metadata["test_metrics"] = test_metrics

    test_scores = {
        r["model"]: bundles[r["model"]].score(X.iloc[te]) for r in results if r["model"] != "dummy"
    }
    uncertainty = bootstrap_ap(y[te], test_scores, n_boot=cfg.n_bootstrap, seed=cfg.seed)
    for r in results:
        if r["model"] in uncertainty["models"]:
            r["test"]["average_precision_ci95"] = uncertainty["models"][r["model"]]["ci95"]

    report = {
        "session_id": session_id,
        "test_uncertainty": uncertainty,
        "dataset": cfg.dataset,
        "provenance": contract.provenance,
        "synthetic": contract.provenance == "synthetic",
        "dataset_fingerprint": fingerprint,
        "config": cfg.to_dict(),
        "split": split.description,
        "selection": selection,
        "primary_metric": "average_precision (area under the precision-recall curve, step-wise)",
        "code_revision": revision,
        "results": results,
        "duration_s": time.perf_counter() - t_start,
        "created_at": datetime.now(UTC).isoformat(),
    }

    # ---------------------------------------------------------------- 9. persist
    from transactionguard.monitoring.drift import save_reference

    registry = BundleRegistry(settings.bundles_dir)
    period = {
        "time_start": split.description["valid"]["time_start"],
        "time_end": split.description["valid"]["time_end"],
        "time_unit": "seconds since epoch"
        if contract.field(contract.timestamp_column).dtype == "datetime"
        else "seconds since first transaction",
    }
    for name, b in bundles.items():
        if name != "dummy":
            save_reference(
                b, X.iloc[va], period, seed=cfg.seed
            )  # the dummy is a reference line, not something to deploy
            registry.save(b, activate=cfg.activate and name == selected["name"])
    eval_dir = settings.artifacts_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / f"{cfg.dataset}-{session_id}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(report_path)

    if cfg.log_mlflow:
        from transactionguard.training.tracking import log_session

        report["mlflow"] = log_session(report, bundles, selected["name"], report_path)
        for name, run_id in report["mlflow"]["child_run_ids"].items():
            if name != "dummy":
                registry.annotate(
                    bundles[name].model_version,
                    mlflow_run_id=run_id,
                    mlflow_parent_run_id=report["mlflow"]["parent_run_id"],
                )
        report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info(
        "training_complete",
        session=session_id,
        selected=selected["name"],
        seconds=round(report["duration_s"], 1),
    )
    return report
