"""SQLAlchemy 2.0 ORM models (typed ``Mapped[...]`` style).

Design notes:
  * Every business table carries ``workspace_id``; every query filters on it (tenant isolation).
  * Idempotency is enforced by the database, not by hope:
      - one stored transaction per (workspace, contract, transaction_id)
      - one prediction per (transaction, model_version)
    so client resubmissions and Celery retries can never create duplicates.
  * Analyst *opinions* (``Review``) are stored separately from independently *confirmed labels*
    (``ConfirmedLabel``). Neither is ever used to retrain automatically.
  * Timestamps are timezone-aware UTC.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}


def _enum(cls: type[enum.Enum]) -> Enum:
    # Store readable values; native_enum=False keeps migrations portable (VARCHAR + CHECK).
    return Enum(cls, native_enum=False, values_callable=lambda e: [m.value for m in e], length=32)


class Role(enum.StrEnum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class Recommendation(enum.StrEnum):
    review = "review"
    no_action = "no_action"


class ExplanationStatus(enum.StrEnum):
    not_requested = "not_requested"
    pending = "pending"
    ok = "ok"
    failed = "failed"
    unavailable = "unavailable"


class ReviewStatus(enum.StrEnum):
    unreviewed = "unreviewed"
    under_review = "under_review"
    resolved = "resolved"


class AnalystAssessment(enum.StrEnum):
    """The analyst's documented opinion. NOT a confirmed fraud label."""

    suspicious = "suspicious"
    not_suspicious = "not_suspicious"
    inconclusive = "inconclusive"


class LabelSource(enum.StrEnum):
    chargeback = "chargeback"
    customer_confirmed = "customer_confirmed"
    investigation_closed = "investigation_closed"
    dataset_label = "dataset_label"  # replayed benchmark labels (demo only)


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


# ============================================================================== tenancy & users
class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(_enum(Role), default=Role.analyst)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    workspace: Mapped[Workspace] = relationship()


# ============================================================================== models
class ModelVersion(Base):
    """Mirror of the trusted bundle registry so predictions can reference a model by FK."""

    __tablename__ = "model_versions"
    model_version: Mapped[str] = mapped_column(String(120), primary_key=True)
    contract: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(32))
    score_kind: Mapped[str] = mapped_column(String(32))
    threshold: Mapped[float] = mapped_column(Float)
    preprocessing_version: Mapped[str] = mapped_column(String(32))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ============================================================================== transactions
class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("workspace_id", "contract", "transaction_id", name="uq_txn_workspace_contract_id"),
        Index("ix_txn_history", "workspace_id", "contract", "account_id", "event_time"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    contract: Mapped[str] = mapped_column(String(32))
    transaction_id: Mapped[str] = mapped_column(String(64))
    account_id: Mapped[str | None] = mapped_column(String(64))
    # Event time from the payload (UTC). For relative-time datasets we store elapsed seconds.
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    event_seconds: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("transaction_pk", "model_version", name="uq_prediction_txn_model"),
        Index("ix_pred_ws_created", "workspace_id", "created_at"),
        Index("ix_pred_ws_flagged_score", "workspace_id", "flagged", "score"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    transaction_pk: Mapped[uuid.UUID] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"))
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.model_version"), index=True)
    preprocessing_version: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)
    score_kind: Mapped[str] = mapped_column(String(32))
    threshold: Mapped[float] = mapped_column(Float)
    flagged: Mapped[bool] = mapped_column(Boolean)
    recommendation: Mapped[Recommendation] = mapped_column(_enum(Recommendation))
    explanation_status: Mapped[ExplanationStatus] = mapped_column(
        _enum(ExplanationStatus), default=ExplanationStatus.not_requested
    )
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    warnings: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    # Exact model inputs at scoring time (feature logging): makes later explanations reproduce the
    # scored inputs even if more history arrives, and feeds drift monitoring.
    features: Mapped[dict[str, Any]] = mapped_column(JSONType)
    latency_ms: Mapped[float] = mapped_column(Float)
    batch_job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("batch_jobs.id"), index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    transaction: Mapped[Transaction] = relationship(lazy="joined")
    review: Mapped[Review | None] = relationship(back_populates="prediction", uselist=False, lazy="joined")


# ============================================================================== analyst review
class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), unique=True
    )
    status: Mapped[ReviewStatus] = mapped_column(_enum(ReviewStatus), default=ReviewStatus.unreviewed)
    assessment: Mapped[AnalystAssessment | None] = mapped_column(_enum(AnalystAssessment))
    note: Mapped[str | None] = mapped_column(Text)
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    prediction: Mapped[Prediction] = relationship(back_populates="review")


class ReviewEvent(Base):
    """Append-only audit trail of review changes."""

    __tablename__ = "review_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    review_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str] = mapped_column(String(32))
    assessment: Mapped[str | None] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConfirmedLabel(Base):
    """An outcome confirmed independently of the model and the analyst's opinion (e.g. chargeback).
    Used only to measure model performance in monitoring, never for automatic retraining."""

    __tablename__ = "confirmed_labels"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    transaction_pk: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), unique=True
    )
    is_fraud: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[LabelSource] = mapped_column(_enum(LabelSource))
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    recorded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))


# ============================================================================== async jobs
class BatchJob(Base):
    __tablename__ = "batch_jobs"
    __table_args__ = (UniqueConstraint("workspace_id", "idempotency_key", name="uq_job_idempotency"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    contract: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.model_version"))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    file_name: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64))
    input_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[JobStatus] = mapped_column(_enum(JobStatus), default=JobStatus.queued, index=True)
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    valid_rows: Mapped[int] = mapped_column(Integer, default=0)
    invalid_rows: Mapped[int] = mapped_column(Integer, default=0)
    processed_rows: Mapped[int] = mapped_column(Integer, default=0)
    flagged_rows: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    validation_report: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    task_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MonitoringReport(Base):
    __tablename__ = "monitoring_reports"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    requested_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    contract: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.model_version"))
    status: Mapped[JobStatus] = mapped_column(_enum(JobStatus), default=JobStatus.queued)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    html_path: Mapped[str | None] = mapped_column(String(500))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RequestLog(Base):
    """Operational metrics: one row per API request (route template, status, latency). No payloads."""

    __tablename__ = "request_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route: Mapped[str] = mapped_column(String(200), index=True)
    method: Mapped[str] = mapped_column(String(8))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
