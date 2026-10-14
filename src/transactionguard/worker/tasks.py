"""Background jobs: batch scoring, batch explanations, monitoring reports.

Each task is a thin Celery wrapper around a plain function (``run_*``) so the logic is testable
without a broker. Job state lives in PostgreSQL, not in Celery's result backend, so the API can
always tell the user exactly what happened (queued / running / succeeded / failed + error).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger
from transactionguard.db.models import BatchJob, ExplanationStatus, JobStatus, Prediction
from transactionguard.db.session import session_scope
from transactionguard.inference.service import (
    MODEL_STORE,
    features_from_record,
    now_utc,
    score_transactions,
)
from transactionguard.worker.celery_app import celery_app

log = get_logger(__name__)
CHUNK_ROWS = 1_000
# Errors worth retrying (DB/broker hiccups). Bad input or a broken model is NOT retried.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (OperationalError, ConnectionError, TimeoutError)


class TransientJobError(RuntimeError):
    pass


# ============================================================================== batch scoring
def run_batch_job(job_id: uuid.UUID) -> str:
    """Score a batch job. Safe to call again after a crash: rows already scored are reused."""
    with session_scope() as db:
        job = db.get(BatchJob, job_id)
        if job is None:
            log.warning("batch_job_missing", job_id=str(job_id))
            return "missing"
        if job.status == JobStatus.succeeded:
            return "already_succeeded"  # duplicate delivery of a finished job
        job.status, job.attempts, job.started_at, job.error = (
            JobStatus.running,
            job.attempts + 1,
            now_utc(),
            None,
        )
        workspace_id, owner_id, contract = job.workspace_id, job.owner_id, job.contract
        model_version, input_path = job.model_version, Path(job.input_path)

    bundle = MODEL_STORE.get(model_version)
    df = pd.read_parquet(input_path)
    payloads = df.to_dict(orient="records")
    processed = flagged = replayed = 0
    conflicts: list[dict[str, str]] = []
    for start in range(0, len(payloads), CHUNK_ROWS):
        chunk = payloads[start : start + CHUNK_ROWS]
        with session_scope() as db:  # one transaction per chunk: progress survives a crash
            res = score_transactions(
                db,
                workspace_id=workspace_id,
                contract_name=contract,
                payloads=chunk,
                bundle=bundle,
                user_id=owner_id,
                batch_job_id=job_id,
            )
            processed += len(chunk)
            flagged += sum(o.prediction.flagged for o in res.outcomes)
            replayed += sum(o.replayed for o in res.outcomes)
            conflicts += res.conflicts
            job = db.get(BatchJob, job_id)
            assert job is not None
            job.processed_rows, job.flagged_rows = processed, flagged

    with session_scope() as db:
        job = db.get(BatchJob, job_id)
        assert job is not None
        report = dict(job.validation_report or {})
        report["scoring"] = {
            "replayed_existing_predictions": replayed,
            "conflicts": conflicts[:100],
            "n_conflicts": len(conflicts),
        }
        job.validation_report = report
        job.status, job.finished_at = JobStatus.succeeded, now_utc()
    log.info("batch_job_succeeded", job_id=str(job_id), rows=processed, flagged=flagged, replayed=replayed)
    return "succeeded"


def mark_job_failed(job_id: uuid.UUID, message: str, *, final: bool) -> None:
    with session_scope() as db:
        job = db.get(BatchJob, job_id)
        if job is None:
            return
        job.error = message[:2000]
        job.status = JobStatus.failed if final else JobStatus.queued
        if final:
            job.finished_at = now_utc()


@celery_app.task(bind=True, name="tg.score_batch", acks_late=True)
def score_batch_task(self, job_id: str) -> str:  # type: ignore[no-untyped-def]
    s = get_settings()
    jid = uuid.UUID(job_id)
    try:
        result = run_batch_job(jid)
    except (*TRANSIENT_ERRORS, TransientJobError) as exc:
        final = self.request.retries >= s.celery_max_retries
        mark_job_failed(jid, f"{type(exc).__name__}: {exc}" + ("" if final else " (retrying)"), final=final)
        if final:
            log.error("batch_job_failed", job_id=job_id, retries=self.request.retries)
            return "failed"
        raise self.retry(exc=exc, countdown=min(60, 2 ** (self.request.retries + 1))) from exc
    except Exception as exc:  # non-transient: fail immediately with a clear message
        mark_job_failed(jid, f"{type(exc).__name__}: {exc}", final=True)
        log.error("batch_job_failed", job_id=job_id, error=type(exc).__name__)
        return "failed"
    if result == "succeeded":
        dispatch_explain_batch(job_id)
    return result


# ============================================================================== explanations
def run_explain_batch(job_id: uuid.UUID) -> int:
    """Explain the flagged predictions of a batch (pending ones only), vectorised per model."""
    with session_scope() as db:
        preds = db.scalars(
            select(Prediction).where(
                Prediction.batch_job_id == job_id,
                Prediction.explanation_status == ExplanationStatus.pending,
            )
        ).all()
        if not preds:
            return 0
        by_model: dict[str, list[Prediction]] = {}
        for p in preds:
            by_model.setdefault(p.model_version, []).append(p)
        for version, group in by_model.items():
            bundle = MODEL_STORE.get(version)
            X = pd.concat([features_from_record(bundle, p.features) for p in group], ignore_index=True)
            exps = MODEL_STORE.explainer(bundle).explain(X)
            for p, e in zip(group, exps, strict=True):
                p.explanation, p.explanation_status = e.to_dict(), ExplanationStatus(e.status)
        return len(preds)


@celery_app.task(name="tg.explain_batch", acks_late=True)
def explain_batch_task(job_id: str) -> int:
    try:
        return run_explain_batch(uuid.UUID(job_id))
    except Exception as exc:  # leave rows 'pending' visible + log; the UI can request one-by-one
        log.error("explain_batch_failed", job_id=job_id, error=f"{type(exc).__name__}: {exc}")
        with session_scope() as db:
            for p in db.scalars(
                select(Prediction).where(
                    Prediction.batch_job_id == uuid.UUID(job_id),
                    Prediction.explanation_status == ExplanationStatus.pending,
                )
            ):
                p.explanation_status = ExplanationStatus.failed
                p.explanation = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        return 0


def dispatch_explain_batch(job_id: str) -> None:
    explain_batch_task.apply_async(args=[job_id]) if not get_settings().celery_eager else explain_batch_task(
        job_id
    )


# ============================================================================== monitoring
@celery_app.task(name="tg.monitoring_report", acks_late=True)
def monitoring_report_task(report_id: str) -> str:
    from transactionguard.monitoring.drift import run_monitoring_report

    return run_monitoring_report(uuid.UUID(report_id))
