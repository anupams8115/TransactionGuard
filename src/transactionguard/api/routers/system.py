"""Health, readiness and the overview dashboard summary."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Response, status
from sqlalchemy import func, select, text

from transactionguard.api.deps import DB, CurrentUser
from transactionguard.core.config import get_settings
from transactionguard.data.contract import CONTRACTS
from transactionguard.db.models import BatchJob, JobStatus, Prediction, Review, ReviewStatus, Transaction
from transactionguard.db.session import get_engine
from transactionguard.inference.service import MODEL_STORE, now_utc

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Does not touch dependencies."""
    return {"status": "ok"}


@router.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    """Readiness: can we actually serve? Checks the DB, model bundles and (if used) Redis."""
    s = get_settings()
    checks: dict[str, Any] = {}
    try:
        with get_engine(s.database_url).connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    models: dict[str, str] = {}
    for name in CONTRACTS:
        try:
            models[name] = MODEL_STORE.active(name).model_version
        except Exception as exc:
            models[name] = f"unavailable: {type(exc).__name__}"
    checks["models"] = models
    if s.celery_eager:
        checks["queue"] = "eager (in-process; Redis not used)"
    else:
        try:
            import redis

            redis.Redis.from_url(s.redis_url, socket_connect_timeout=2).ping()
            checks["queue"] = "ok"
        except Exception as exc:
            checks["queue"] = f"error: {type(exc).__name__}"
    any_model = any(not v.startswith("unavailable") for v in models.values())
    ok = checks["database"] == "ok" and any_model and not str(checks["queue"]).startswith("error")
    response.status_code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "not_ready", "checks": checks}


@router.get("/overview")
def overview(user: CurrentUser, db: DB) -> dict[str, Any]:
    ws = user.workspace_id
    since = now_utc() - timedelta(days=7)
    preds = select(Prediction).where(Prediction.workspace_id == ws)
    n_total = db.scalar(select(func.count()).select_from(preds.subquery())) or 0
    n_flagged = db.scalar(select(func.count()).select_from(preds.where(Prediction.flagged).subquery())) or 0
    n_week = (
        db.scalar(select(func.count()).select_from(preds.where(Prediction.created_at >= since).subquery()))
        or 0
    )
    backlog = dict(
        db.execute(
            select(Review.status, func.count()).where(Review.workspace_id == ws).group_by(Review.status)
        ).all()
    )
    jobs = dict(
        db.execute(
            select(BatchJob.status, func.count()).where(BatchJob.workspace_id == ws).group_by(BatchJob.status)
        ).all()
    )
    synthetic = (
        db.scalar(select(func.count()).where(Transaction.workspace_id == ws, Transaction.synthetic.is_(True)))
        or 0
    )
    return {
        "predictions_total": n_total,
        "predictions_last_7d": n_week,
        "flagged_total": n_flagged,
        "review_backlog": {s.value: int(backlog.get(s, 0)) for s in ReviewStatus},
        "batch_jobs": {s.value: int(jobs.get(s, 0)) for s in JobStatus},
        "synthetic_transactions": synthetic,
        "active_models": MODEL_STORE.registry.read().get("active", {}),
    }
