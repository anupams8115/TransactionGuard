"""MLflow experiment tracking and model registration.

One *parent* run per training session (dataset fingerprint, split, seed, policy, code revision)
and one *child* run per candidate model (params, validation + test metrics, threshold, bundle).
The selected model's estimator is registered in the MLflow Model Registry as
``transactionguard-<dataset>`` so versions can be compared in the MLflow UI.

View with:  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   ->  http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger
from transactionguard.inference.bundle import ModelBundle

log = get_logger(__name__)


def _flat_metrics(prefix: str, m: dict | None) -> dict[str, float]:
    if not m:
        return {}
    out = {
        f"{prefix}_{k}": float(v)
        for k, v in m.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None
    }
    for k, v in (m.get("confusion_matrix") or {}).items():
        out[f"{prefix}_cm_{k}"] = float(v)
    return out


def log_session(report: dict, bundles: dict[str, ModelBundle], selected: str, report_path: Path) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment)
    cfg = report["config"]
    run_ids: dict[str, str] = {}
    with mlflow.start_run(run_name=f"{report['dataset']}-{report['session_id']}") as parent:
        mlflow.set_tags(
            {
                "dataset": report["dataset"],
                "provenance": report["provenance"],
                "synthetic": str(report["synthetic"]),
                "selected_model": selected,
                "git_sha": str(report["code_revision"].get("git_sha")),
                "git_dirty": str(report["code_revision"].get("git_dirty")),
            }
        )
        mlflow.log_params(
            {
                "dataset_fingerprint": report["dataset_fingerprint"],
                "split_strategy": cfg["split"]["strategy"],
                "train_frac": cfg["split"]["train_frac"],
                "valid_frac": cfg["split"]["valid_frac"],
                "gap_seconds": cfg["split"]["gap_seconds"],
                "seed": cfg["seed"],
                "budget": cfg["budget"],
                "threshold_policy": cfg["policy"]["name"],
                "min_precision": cfg["policy"]["min_precision"],
                "iforest_training_data": cfg["iforest_training_data"],
                "preprocessing_version": next(iter(bundles.values())).preprocessing_version,
            }
        )
        mlflow.log_dict(report["split"], "split.json")
        mlflow.log_artifact(str(report_path))

        for res in report["results"]:
            name = res["model"]
            b = bundles[name]
            with mlflow.start_run(run_name=name, nested=True) as child:
                run_ids[name] = child.info.run_id
                mlflow.set_tags(
                    {
                        "model": name,
                        "score_kind": b.score_kind,
                        "model_version": b.model_version,
                        "selected": str(res["selected"]),
                    }
                )
                mlflow.log_params({f"hp_{k}": v for k, v in res["params"].items()})
                mlflow.log_params({"threshold": b.threshold, "optuna_trials": res["optuna_trials"]})
                mlflow.log_metrics(
                    {**_flat_metrics("valid", res["validation"]), **_flat_metrics("test", res["test"])}
                )
                if res.get("latency"):
                    mlflow.log_metrics({k: v for k, v in res["latency"].items() if isinstance(v, float)})
                mlflow.log_dict(
                    {"preprocessing": repr(b.preprocessor), "features": b.metadata["features"]},
                    "preprocessing.json",
                )
                if name != "dummy":
                    bundle_file = s.bundles_dir / b.model_version / "bundle.joblib"
                    if bundle_file.exists():
                        mlflow.log_artifact(str(bundle_file), artifact_path="bundle")
                if name == selected:
                    sample = b.shap_background  # transformed rows; signature of the estimator input
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        common = {
                            "name": "model",
                            "registered_model_name": f"transactionguard-{report['dataset']}",
                            "signature": infer_signature(sample, b.raw_output_transformed(sample)),
                        }
                        if name == "xgboost":
                            # Native XGBoost JSON format: no pickle involved.
                            import mlflow.xgboost

                            mlflow.xgboost.log_model(b.estimator, model_format="json", **common)
                        else:
                            # MLflow serialises sklearn estimators with skops (safe loading).
                            mlflow.sklearn.log_model(sk_model=b.estimator, **common)
        parent_id = parent.info.run_id
    log.info("mlflow_logged", parent_run=parent_id)
    return {
        "tracking_uri": s.mlflow_tracking_uri,
        "experiment": s.mlflow_experiment,
        "parent_run_id": parent_id,
        "child_run_ids": run_ids,
    }
