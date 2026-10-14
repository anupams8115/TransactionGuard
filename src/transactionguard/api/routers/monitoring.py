"""Monitoring reports (Evidently) and operational metrics."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from transactionguard.api.deps import DB, Admin, Analyst, CurrentUser
from transactionguard.api.schemas import MonitoringReportOut, MonitoringRequest
from transactionguard.core.config import get_settings
from transactionguard.db.models import MonitoringReport
from transactionguard.inference.service import (
    MODEL_STORE,
    ModelNotAvailableError,
    now_utc,
    sync_model_versions,
)
from transactionguard.monitoring.drift import operational_metrics
from transactionguard.worker.dispatch import dispatch
from transactionguard.worker.tasks import monitoring_report_task

router = APIRouter(tags=["monitoring"])


def _owned(db: DB, user: CurrentUser, report_id: uuid.UUID) -> MonitoringReport:
    r = db.get(MonitoringReport, report_id)
    if r is None or r.workspace_id != user.workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return r


@router.post("/monitoring/reports", response_model=MonitoringReportOut, status_code=202)
def request_report(
    body: MonitoringRequest, user: Analyst, db: DB, background: BackgroundTasks
) -> MonitoringReportOut:
    try:
        bundle = MODEL_STORE.active(body.contract)
    except ModelNotAvailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    sync_model_versions(db)
    report = MonitoringReport(
        workspace_id=user.workspace_id,
        requested_by=user.id,
        contract=body.contract,
        model_version=bundle.model_version,
        summary={"requested_window_days": body.window_days},
    )
    db.add(report)
    db.commit()
    dispatch(monitoring_report_task, str(report.id), background=background)
    return MonitoringReportOut.model_validate(report)


@router.get("/monitoring/reports", response_model=list[MonitoringReportOut])
def list_reports(
    user: CurrentUser, db: DB, limit: Annotated[int, Query(le=100)] = 20
) -> list[MonitoringReportOut]:
    rows = db.scalars(
        select(MonitoringReport)
        .where(MonitoringReport.workspace_id == user.workspace_id)
        .order_by(MonitoringReport.created_at.desc())
        .limit(limit)
    )
    return [MonitoringReportOut.model_validate(r) for r in rows]


@router.get("/monitoring/reports/{report_id}", response_model=MonitoringReportOut)
def get_report(report_id: uuid.UUID, user: CurrentUser, db: DB) -> MonitoringReportOut:
    return MonitoringReportOut.model_validate(_owned(db, user, report_id))


@router.get("/monitoring/reports/{report_id}/html", response_class=FileResponse)
def report_html(report_id: uuid.UUID, user: CurrentUser, db: DB) -> FileResponse:
    r = _owned(db, user, report_id)
    if not r.html_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This report has no HTML (not enough data or failed)")
    path = Path(r.html_path).resolve()
    if not path.is_relative_to(get_settings().reports_dir.resolve()) or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report file missing")
    return FileResponse(path, media_type="text/html")


@router.get("/monitoring/operational")
def operational(_: Admin, db: DB, hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24) -> dict[str, Any]:
    """System-wide request latency, error rates and batch-job outcomes (admin only)."""
    return operational_metrics(db, now_utc() - timedelta(hours=hours))
