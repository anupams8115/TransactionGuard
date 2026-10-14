"""Batch CSV scoring: upload -> validate -> queue -> worker scores -> poll status/results."""

from __future__ import annotations

import hashlib
import io
import uuid
from typing import Annotated

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from transactionguard.api.deps import DB, Analyst, CurrentUser
from transactionguard.api.schemas import BatchJobOut
from transactionguard.core.config import get_settings
from transactionguard.data.contract import CONTRACTS, get_contract
from transactionguard.data.validation import validate_frame
from transactionguard.db.models import BatchJob, Role, User
from transactionguard.inference.service import MODEL_STORE, ModelNotAvailableError, sync_model_versions
from transactionguard.worker.dispatch import dispatch
from transactionguard.worker.tasks import score_batch_task

router = APIRouter(tags=["batches"])


def _out(job: BatchJob, replay: bool = False) -> BatchJobOut:
    out = BatchJobOut.model_validate(job)
    out.idempotent_replay = replay
    return out


def _visible_job(db: Session, user: User, job_id: uuid.UUID) -> BatchJob:
    """Ownership check: the job's owner, or an admin of the same workspace. Others get 404."""
    job = db.get(BatchJob, job_id)
    if (
        job is None
        or job.workspace_id != user.workspace_id
        or (job.owner_id != user.id and user.role != Role.admin)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch job not found")
    return job


@router.post("/batches", response_model=BatchJobOut, status_code=202)
async def submit_batch(
    user: Analyst,
    db: DB,
    background: BackgroundTasks,
    response: Response,
    file: Annotated[UploadFile, File(description="CSV with the contract's input columns")],
    contract: Annotated[str, Form()],
    idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
) -> BatchJobOut:
    s = get_settings()
    if contract not in CONTRACTS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown contract {contract!r}")
    limit = s.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {s.max_upload_mb} MB")
    digest = hashlib.sha256(content).hexdigest()

    # Idempotency: the same key returns the same job instead of creating a duplicate.
    if idempotency_key:
        existing = db.scalars(
            select(BatchJob).where(
                BatchJob.workspace_id == user.workspace_id, BatchJob.idempotency_key == idempotency_key
            )
        ).first()
        if existing is not None:
            if existing.file_sha256 != digest:
                raise HTTPException(status.HTTP_409_CONFLICT, "Idempotency-Key reused with a different file")
            response.status_code = status.HTTP_200_OK
            return _out(existing, replay=True)

    try:
        bundle = MODEL_STORE.active(contract)
    except ModelNotAvailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    try:
        # Read everything as text; the contract validator does explicit, reported type coercion.
        raw = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Could not parse CSV: {exc}") from exc
    if len(raw) == 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "CSV has no rows")
    if len(raw) > s.max_batch_rows:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"At most {s.max_batch_rows} rows per batch")

    c = get_contract(contract)
    coerced, report = validate_frame(raw, c, require_target=False)
    if any(i.code == "missing_columns" for i in report.issues):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, report.to_dict())
    valid = coerced[~coerced[c.id_column].astype(str).isin(report.invalid_row_ids)]
    if valid[c.id_column].isna().any():
        valid = valid[valid[c.id_column].notna()]

    job_id = uuid.uuid4()
    upload_dir = s.data_dir / "uploads" / str(user.workspace_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"{job_id}.parquet"
    valid.to_parquet(input_path, index=False)

    sync_model_versions(db)
    job = BatchJob(
        id=job_id,
        workspace_id=user.workspace_id,
        owner_id=user.id,
        contract=contract,
        model_version=bundle.model_version,
        idempotency_key=idempotency_key,
        file_name=(file.filename or "upload.csv")[:255],
        file_sha256=digest,
        input_path=str(input_path),
        total_rows=len(raw),
        valid_rows=len(valid),
        invalid_rows=len(raw) - len(valid),
        validation_report=report.to_dict(),
    )
    db.add(job)
    db.commit()  # the worker must be able to see the job before it starts

    if len(valid) == 0:
        job.status, job.error = job.status.failed, "No valid rows to score; see the validation report"
        db.commit()
        return _out(job)
    job.task_id = dispatch(score_batch_task, str(job_id), background=background)
    db.commit()
    return _out(job)


@router.get("/batches", response_model=list[BatchJobOut])
def list_batches(user: CurrentUser, db: DB, limit: int = 50) -> list[BatchJobOut]:
    stmt = select(BatchJob).where(BatchJob.workspace_id == user.workspace_id)
    if user.role != Role.admin:
        stmt = stmt.where(BatchJob.owner_id == user.id)
    jobs = db.scalars(stmt.order_by(BatchJob.created_at.desc()).limit(min(limit, 200)))
    return [_out(j) for j in jobs]


@router.get("/batches/{job_id}", response_model=BatchJobOut)
def get_batch(job_id: uuid.UUID, user: CurrentUser, db: DB) -> BatchJobOut:
    return _out(_visible_job(db, user, job_id))


@router.post("/batches/{job_id}/retry", response_model=BatchJobOut)
def retry_batch(job_id: uuid.UUID, user: Analyst, db: DB, background: BackgroundTasks) -> BatchJobOut:
    """Re-run a failed job. Already-scored rows are reused, so a retry never duplicates."""
    job = _visible_job(db, user, job_id)
    if job.status != job.status.failed:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Only failed jobs can be retried (status={job.status})"
        )
    job.status, job.error = job.status.queued, None
    db.commit()
    job.task_id = dispatch(score_batch_task, str(job.id), background=background)
    db.commit()
    return _out(job)
