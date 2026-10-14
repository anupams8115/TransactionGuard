"""Monitoring reports with Evidently.

Reference  = a sample of the model's VALIDATION period (engineered features + model scores),
             saved next to the bundle at training time. It is the data the threshold was chosen on.
Current    = predictions scored in this workspace during the monitoring window, using the exact
             feature vectors stored with each prediction (feature logging).

What we report, and how to read it:
  * data quality  - missing values per feature, schema violations (invalid batch rows, 422s)
  * data drift    - per-feature distribution tests (Evidently: K-S for numeric, chi-square for
                    categorical at 5%); a drifted feature means the INPUTS changed
  * score drift   - distribution of model scores + alert rate vs. the reference alert rate
  * performance   - ONLY when enough independently confirmed labels exist in the window
Drift is a reason to investigate, not proof that the model got worse.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger
from transactionguard.db.models import (
    BatchJob,
    ConfirmedLabel,
    JobStatus,
    MonitoringReport,
    Prediction,
    RequestLog,
    Transaction,
)
from transactionguard.db.session import as_utc, session_scope
from transactionguard.inference.bundle import ModelBundle
from transactionguard.inference.service import MODEL_STORE, now_utc

log = get_logger(__name__)
MIN_CURRENT_ROWS = 100
MIN_LABELS, MIN_LABEL_POSITIVES = 50, 5
REFERENCE_FILE = "reference.parquet"


# ============================================================================ reference data
def save_reference(
    bundle: ModelBundle, X_valid: pd.DataFrame, period: dict, max_rows: int = 20_000, seed: int = 42
) -> Path:
    """Called by training: persist validation-period features + scores as the drift reference."""
    sample = (
        X_valid.sample(n=min(max_rows, len(X_valid)), random_state=seed)
        if len(X_valid) > max_rows
        else X_valid
    )
    ref = sample.reset_index(drop=True).copy()
    ref["score"] = bundle.score(sample)
    ref["flagged"] = bundle.flag(ref["score"].to_numpy())
    folder = get_settings().bundles_dir / bundle.model_version
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / REFERENCE_FILE
    ref.to_parquet(path, index=False)
    bundle.metadata["reference"] = {"rows": len(ref), "source": "validation split", **period}
    return path


def load_reference(model_version: str) -> pd.DataFrame:
    path = get_settings().bundles_dir / model_version / REFERENCE_FILE
    if not path.exists():
        raise FileNotFoundError(f"No monitoring reference saved for {model_version}; retrain the model.")
    return pd.read_parquet(path)  # parquet is data, not code: safe to load


# ============================================================================ current window
def current_window(db: Session, workspace_id: uuid.UUID, bundle: ModelBundle, start, end) -> pd.DataFrame:
    rows = db.execute(
        select(
            Prediction.features,
            Prediction.score,
            Prediction.flagged,
            Prediction.created_at,
            Prediction.transaction_pk,
        )
        .join(Transaction, Prediction.transaction_pk == Transaction.id)
        .where(
            Prediction.workspace_id == workspace_id,
            Transaction.contract == bundle.contract_name,
            Prediction.preprocessing_version == bundle.preprocessing_version,
            Prediction.score_kind == bundle.score_kind,
            Prediction.created_at >= start,
            Prediction.created_at <= end,
        )
    ).all()
    if not rows:
        return pd.DataFrame(columns=[*bundle.feature_spec.all, "score", "flagged", "created_at", "txn"])
    df = pd.DataFrame([r.features for r in rows], columns=bundle.feature_spec.all)
    for c in (*bundle.feature_spec.amount, *bundle.feature_spec.numeric):
        df[c] = pd.to_numeric(df[c]).astype("float64")
    df["score"] = [r.score for r in rows]
    df["flagged"] = [r.flagged for r in rows]
    df["created_at"] = [as_utc(r.created_at) for r in rows]
    df["txn"] = [r.transaction_pk for r in rows]
    return df


# ============================================================================ the report
def drift_decision(method: str, value: float, threshold: float) -> dict[str, Any]:
    """Interpret an Evidently drift statistic correctly.

    Evidently picks the test by sample size: statistical tests (K-S, chi-square) return a p-value,
    where SMALL means drift; for large samples it uses distances (Wasserstein, Jensen-Shannon),
    where LARGE means drift. Reading every value as a p-value inverts the result.
    """
    if "p_value" in method:
        return {
            "statistic": value,
            "statistic_kind": "p_value",
            "threshold": threshold,
            "drift_detected": value < threshold,
        }
    return {
        "statistic": value,
        "statistic_kind": "distance",
        "threshold": threshold,
        "drift_detected": value >= threshold,
    }


def _evidently(ref: pd.DataFrame, cur: pd.DataFrame, bundle: ModelBundle, html_path: Path) -> dict[str, Any]:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    spec = bundle.feature_spec
    numeric = [*spec.amount, *spec.numeric, "score"]
    cols = [*numeric, *spec.categorical]
    dd = DataDefinition(numerical_columns=numeric, categorical_columns=list(spec.categorical))
    snap = Report([DataDriftPreset(columns=cols), DataSummaryPreset(columns=cols)]).run(
        Dataset.from_pandas(cur[cols], data_definition=dd), Dataset.from_pandas(ref[cols], data_definition=dd)
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(html_path))

    drift: dict[str, Any] = {"columns": {}}
    missing: dict[str, float] = {}
    for m in snap.dict()["metrics"]:
        name, value = m["metric_name"], m["value"]
        cfg = m.get("config", {})
        if name.startswith("DriftedColumnsCount"):
            drift["drifted_columns"] = int(value["count"])
            drift["drifted_share"] = float(value["share"])
        elif name.startswith("ValueDrift"):
            method = str(cfg.get("method"))
            thr = float(cfg.get("threshold", 0.05))
            drift["columns"][cfg["column"]] = {"method": method, **drift_decision(method, float(value), thr)}
        elif name.startswith("MissingValueCount(column="):
            missing[cfg.get("column") or name.split("=")[1].rstrip(")")] = float(value["share"])
    return {"drift": drift, "missing_share_current": missing}


def build_report(db: Session, report: MonitoringReport, window_days: int) -> dict[str, Any]:
    s = get_settings()
    bundle = MODEL_STORE.get(report.model_version)
    end = now_utc()
    start = end - timedelta(days=window_days)
    report.window_start, report.window_end = start, end
    ref = load_reference(bundle.model_version)
    cur = current_window(db, report.workspace_id, bundle, start, end)

    summary: dict[str, Any] = {
        "model_version": bundle.model_version,
        "synthetic": bundle.contract.provenance == "synthetic",
        "reference": {**bundle.metadata.get("reference", {}), "rows": len(ref)},
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": window_days, "rows": len(cur)},
        "interpretation": (
            "Drift means the inputs or scores changed relative to the validation period. "
            "It is a reason to investigate, not proof that model accuracy dropped. "
            "Accuracy can only be measured with independently confirmed labels."
        ),
    }
    # ---- schema violations in the window (batch validation + rejected API requests)
    from transactionguard.monitoring.metrics import METRICS

    METRICS.flush()
    jobs = db.scalars(
        select(BatchJob).where(BatchJob.workspace_id == report.workspace_id, BatchJob.created_at >= start)
    ).all()
    codes: dict[str, int] = {}
    for j in jobs:
        for issue in (j.validation_report or {}).get("issues", []):
            codes[issue["code"]] = codes.get(issue["code"], 0) + int(issue["count"])
    rejected = (
        db.scalar(
            select(func.count()).where(
                RequestLog.route == "/api/v1/predictions",
                RequestLog.status_code == 422,
                RequestLog.created_at >= start,
            )
        )
        or 0
    )
    summary["schema_violations"] = {
        "batch_rows_total": sum(j.total_rows for j in jobs),
        "batch_rows_invalid": sum(j.invalid_rows for j in jobs),
        "issues_by_code": codes,
        "rejected_single_requests_422": int(rejected),
    }
    # ---- alert volume + score distribution
    q = [0.5, 0.9, 0.99]
    summary["scores"] = {
        "score_kind": bundle.score_kind,
        "threshold": bundle.threshold,
        "reference_alert_rate": float(ref["flagged"].mean()),
        "current_alert_rate": float(cur["flagged"].mean()) if len(cur) else None,
        "current_alerts": int(cur["flagged"].sum()) if len(cur) else 0,
        "reference_quantiles": dict(zip(map(str, q), np.quantile(ref["score"], q).tolist(), strict=True)),
        "current_quantiles": dict(zip(map(str, q), np.quantile(cur["score"], q).tolist(), strict=True))
        if len(cur)
        else None,
    }
    if len(cur):
        daily = (
            cur.assign(day=pd.to_datetime(cur["created_at"]).dt.strftime("%Y-%m-%d"))
            .groupby("day")["flagged"]
            .agg(["count", "sum"])
            .reset_index()
        )
        summary["alerts_by_day"] = [
            {"day": r.day, "scored": int(r["count"]), "alerts": int(r["sum"])} for _, r in daily.iterrows()
        ]

    # ---- data quality + drift (Evidently)
    if len(cur) < MIN_CURRENT_ROWS:
        summary["status"] = "insufficient_data"
        summary["note"] = f"Only {len(cur)} scored rows in the window; at least {MIN_CURRENT_ROWS} needed."
    else:
        html = s.reports_dir / "monitoring" / f"{report.id}.html"
        summary.update(_evidently(ref, cur, bundle, html))
        report.html_path = str(html)
        summary["missing_share_reference"] = ref[bundle.feature_spec.all].isna().mean().round(4).to_dict()
        summary["status"] = (
            "drift_detected" if summary["drift"].get("drifted_columns") else "no_drift_detected"
        )

    # ---- performance: only with enough confirmed labels
    labels = db.execute(
        select(ConfirmedLabel.transaction_pk, ConfirmedLabel.is_fraud).where(
            ConfirmedLabel.workspace_id == report.workspace_id
        )
    ).all()
    lab = {t: bool(f) for t, f in labels}
    labelled = cur[cur["txn"].isin(lab)] if len(cur) else cur
    if len(labelled) >= MIN_LABELS and sum(lab[t] for t in labelled["txn"]) >= MIN_LABEL_POSITIVES:
        from transactionguard.training.evaluation import classification_metrics

        y = np.array([lab[t] for t in labelled["txn"]], dtype=int)
        summary["performance"] = {
            "available": True,
            "labelled_rows": len(labelled),
            "label_coverage": len(labelled) / len(cur),
            "metrics": classification_metrics(y, labelled["score"].to_numpy(), bundle.threshold),
            "caveat": "Labels may be incomplete or delayed; labelled rows may not be representative.",
        }
    else:
        summary["performance"] = {
            "available": False,
            "labelled_rows": len(labelled),
            "reason": f"Needs >= {MIN_LABELS} confirmed labels with >= {MIN_LABEL_POSITIVES} frauds "
            "in the window. Analyst opinions are not used as labels.",
        }
    return summary


def run_monitoring_report(report_id: uuid.UUID) -> str:
    with session_scope() as db:
        report = db.get(MonitoringReport, report_id)
        if report is None:
            return "missing"
        report.status = JobStatus.running
        window_days = int((report.summary or {}).get("requested_window_days", 7))
    try:
        with session_scope() as db:
            report = db.get(MonitoringReport, report_id)
            assert report is not None
            summary = build_report(db, report, window_days)
            report.summary, report.status, report.finished_at = summary, JobStatus.succeeded, now_utc()
        return "succeeded"
    except Exception as exc:
        log.error("monitoring_report_failed", report_id=str(report_id), error=f"{type(exc).__name__}: {exc}")
        with session_scope() as db:
            report = db.get(MonitoringReport, report_id)
            if report is not None:
                report.status, report.error, report.finished_at = (
                    JobStatus.failed,
                    f"{type(exc).__name__}: {exc}",
                    now_utc(),
                )
        return "failed"


# ============================================================================ operational metrics
def operational_metrics(db: Session, since) -> dict[str, Any]:
    from transactionguard.monitoring.metrics import METRICS

    METRICS.flush()
    rows = db.execute(
        select(RequestLog.route, RequestLog.method, RequestLog.status_code, RequestLog.latency_ms).where(
            RequestLog.created_at >= since
        )
    ).all()
    df = pd.DataFrame(rows, columns=["route", "method", "status", "latency_ms"])
    routes = []
    if len(df):
        for (route, method), g in df.groupby(["route", "method"]):
            routes.append(
                {
                    "route": route,
                    "method": method,
                    "requests": len(g),
                    "p50_ms": float(g["latency_ms"].quantile(0.5)),
                    "p95_ms": float(g["latency_ms"].quantile(0.95)),
                    "client_error_rate": float(((g["status"] >= 400) & (g["status"] < 500)).mean()),
                    "server_error_rate": float((g["status"] >= 500).mean()),
                }
            )
        routes.sort(key=lambda r: -r["requests"])
    jobs = db.execute(
        select(BatchJob.status, func.count()).where(BatchJob.created_at >= since).group_by(BatchJob.status)
    ).all()
    return {
        "since": since.isoformat(),
        "requests": len(df),
        "server_error_rate": float((df["status"] >= 500).mean()) if len(df) else 0.0,
        "routes": routes,
        "batch_jobs": {str(s.value if hasattr(s, "value") else s): int(n) for s, n in jobs},
        "note": "Measured inside the API process (excludes network and proxy time).",
    }
