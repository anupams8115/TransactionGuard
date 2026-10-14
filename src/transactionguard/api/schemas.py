"""Pydantic request/response models (the API's public, typed interface)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from transactionguard.db.models import (
    AnalystAssessment,
    ExplanationStatus,
    JobStatus,
    LabelSource,
    Recommendation,
    ReviewStatus,
    Role,
)


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    items: list[T]
    total: int
    page: int
    page_size: int


# ------------------------------------------------------------------------------ auth / users
class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_in: int


class UserOut(ORM):
    id: uuid.UUID
    email: str
    role: Role
    workspace_id: uuid.UUID
    workspace_name: str | None = None
    is_active: bool


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    role: Role = Role.analyst


# ------------------------------------------------------------------------------ predictions
class PredictRequest(BaseModel):
    """`transaction` is validated against the contract's strict request model in the router
    (unknown fields, NaN/inf, invalid categories and missing fields are rejected with 422)."""

    contract: Literal["creditcard", "synthetic"]
    transaction: dict[str, Any]
    explain: bool = True


class ReviewOut(ORM):
    status: ReviewStatus
    assessment: AnalystAssessment | None
    note: str | None
    reviewer_id: uuid.UUID | None
    updated_at: datetime


class PredictionOut(BaseModel):
    id: uuid.UUID
    transaction_id: str
    contract: str
    synthetic: bool
    account_id: str | None
    event_time: datetime | None
    model_version: str
    preprocessing_version: str
    score: float
    score_kind: str
    score_meaning: str
    threshold: float
    flagged: bool
    recommendation: Recommendation
    explanation_status: ExplanationStatus
    latency_ms: float
    warnings: list[str]
    batch_job_id: uuid.UUID | None
    created_at: datetime
    review: ReviewOut | None
    confirmed_label: bool | None = None
    confirmed_label_source: str | None = None
    idempotent_replay: bool = False


class PredictionDetail(PredictionOut):
    transaction: dict[str, Any]
    features: dict[str, Any]
    explanation: dict[str, Any] | None
    trace: dict[str, Any]


# ------------------------------------------------------------------------------ review / labels
class ReviewUpdate(BaseModel):
    status: ReviewStatus
    assessment: AnalystAssessment | None = None
    note: str | None = Field(default=None, max_length=2000)


class ReviewEventOut(ORM):
    actor_id: uuid.UUID
    from_status: str | None
    to_status: str
    assessment: str | None
    note: str | None
    created_at: datetime


class LabelIn(BaseModel):
    is_fraud: bool
    source: LabelSource


# ------------------------------------------------------------------------------ batches
class BatchJobOut(ORM):
    id: uuid.UUID
    contract: str
    model_version: str
    file_name: str
    status: JobStatus
    total_rows: int
    valid_rows: int
    invalid_rows: int
    processed_rows: int
    flagged_rows: int
    attempts: int
    error: str | None
    validation_report: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    idempotent_replay: bool = False


# ------------------------------------------------------------------------------ monitoring
class MonitoringRequest(BaseModel):
    contract: Literal["creditcard", "synthetic"]
    window_days: int = Field(default=7, ge=1, le=365)


class MonitoringReportOut(ORM):
    id: uuid.UUID
    contract: str
    model_version: str
    status: JobStatus
    window_start: datetime | None
    window_end: datetime | None
    summary: dict[str, Any] | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None
