"""Single-transaction scoring, prediction history, explanations, analyst reviews, confirmed labels."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import ValidationError
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from transactionguard.api.deps import DB, Admin, Analyst, CurrentUser
from transactionguard.api.schemas import (
    LabelIn,
    Page,
    PredictionDetail,
    PredictionOut,
    PredictRequest,
    ReviewEventOut,
    ReviewOut,
    ReviewUpdate,
)
from transactionguard.core.config import get_settings
from transactionguard.data.contract import get_contract
from transactionguard.db.models import (
    ConfirmedLabel,
    ExplanationStatus,
    Prediction,
    Review,
    ReviewEvent,
    ReviewStatus,
    Transaction,
    User,
)
from transactionguard.db.session import as_utc
from transactionguard.inference.bundle import SCORE_MEANINGS
from transactionguard.inference.service import (
    MODEL_STORE,
    ModelNotAvailableError,
    explain_prediction,
    score_transactions,
)

router = APIRouter(tags=["predictions"])


# ------------------------------------------------------------------------------ serialisation
def to_out(p: Prediction, *, replay: bool = False, label: ConfirmedLabel | None = None) -> PredictionOut:
    t = p.transaction
    return PredictionOut(
        id=p.id,
        transaction_id=t.transaction_id,
        contract=t.contract,
        synthetic=t.synthetic,
        account_id=t.account_id,
        event_time=as_utc(t.event_time),
        model_version=p.model_version,
        preprocessing_version=p.preprocessing_version,
        score=p.score,
        score_kind=p.score_kind,
        score_meaning=SCORE_MEANINGS.get(p.score_kind, ""),
        threshold=p.threshold,
        flagged=p.flagged,
        recommendation=p.recommendation,
        explanation_status=p.explanation_status,
        latency_ms=p.latency_ms,
        warnings=p.warnings or [],
        batch_job_id=p.batch_job_id,
        created_at=as_utc(p.created_at),  # type: ignore[arg-type]
        review=ReviewOut.model_validate(p.review) if p.review else None,
        confirmed_label=None if label is None else label.is_fraud,
        confirmed_label_source=None if label is None else label.source.value,
        idempotent_replay=replay,
    )


def _label_for(db: Session, txn_pk: uuid.UUID) -> ConfirmedLabel | None:
    return db.scalars(select(ConfirmedLabel).where(ConfirmedLabel.transaction_pk == txn_pk)).first()


def _get_owned(db: Session, user: User, prediction_id: uuid.UUID) -> Prediction:
    """Workspace isolation: other workspaces' predictions are indistinguishable from missing ones."""
    p = db.get(Prediction, prediction_id)
    if p is None or p.workspace_id != user.workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found")
    return p


# ------------------------------------------------------------------------------ scoring
@router.post("/predictions", response_model=PredictionOut)
def predict(body: PredictRequest, user: Analyst, db: DB, response: Response) -> PredictionOut:
    """Score one transaction. Resubmitting the same transaction_id + payload returns the stored
    prediction (200, ``idempotent_replay=true``); a different payload for that id returns 409."""
    contract = get_contract(body.contract)
    try:
        txn = contract.request_model()(**body.transaction)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            [{"loc": ["transaction", *e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()],
        ) from exc
    payload = txn.model_dump()
    try:
        bundle = MODEL_STORE.active(body.contract)
    except ModelNotAvailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    result = score_transactions(
        db,
        workspace_id=user.workspace_id,
        contract_name=body.contract,
        payloads=[payload],
        bundle=bundle,
        user_id=user.id,
        explain=body.explain and get_settings().explain_inline,
    )
    if result.conflicts:
        raise HTTPException(status.HTTP_409_CONFLICT, result.conflicts[0]["reason"])
    outcome = result.outcomes[0]
    response.status_code = status.HTTP_200_OK if outcome.replayed else status.HTTP_201_CREATED
    return to_out(outcome.prediction, replay=outcome.replayed)


# ------------------------------------------------------------------------------ history
SortKey = Literal["created_desc", "created_asc", "score_desc", "score_asc"]


def _filtered(
    user: User,
    *,
    contract: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    score_min: float | None,
    score_max: float | None,
    model_version: str | None,
    review_status: ReviewStatus | Literal["none"] | None,
    flagged: bool | None,
    batch_job_id: uuid.UUID | None,
    q: str | None,
) -> Select[Any]:
    stmt = (
        select(Prediction)
        .join(Transaction, Prediction.transaction_pk == Transaction.id)
        .outerjoin(Review, Review.prediction_id == Prediction.id)
        .where(Prediction.workspace_id == user.workspace_id)
    )
    if contract:
        stmt = stmt.where(Transaction.contract == contract)
    if date_from:
        stmt = stmt.where(Prediction.created_at >= date_from)
    if date_to:
        stmt = stmt.where(Prediction.created_at <= date_to)
    if score_min is not None:
        stmt = stmt.where(Prediction.score >= score_min)
    if score_max is not None:
        stmt = stmt.where(Prediction.score <= score_max)
    if model_version:
        stmt = stmt.where(Prediction.model_version == model_version)
    if review_status == "none":
        stmt = stmt.where(Review.id.is_(None))
    elif review_status is not None:
        stmt = stmt.where(Review.status == review_status)
    if flagged is not None:
        stmt = stmt.where(Prediction.flagged == flagged)
    if batch_job_id:
        stmt = stmt.where(Prediction.batch_job_id == batch_job_id)
    if q:
        like = f"{q.strip()[:64]}%"
        stmt = stmt.where(or_(Transaction.transaction_id.like(like), Transaction.account_id.like(like)))
    return stmt


@router.get("/predictions", response_model=Page[PredictionOut])
def list_predictions(
    user: CurrentUser,
    db: DB,
    contract: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    score_min: Annotated[float | None, Query(ge=0)] = None,
    score_max: float | None = None,
    model_version: str | None = None,
    review_status: ReviewStatus | Literal["none"] | None = None,
    flagged: bool | None = None,
    batch_job_id: uuid.UUID | None = None,
    q: Annotated[str | None, Query(max_length=64)] = None,
    sort: SortKey = "created_desc",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> Page[PredictionOut]:
    stmt = _filtered(
        user,
        contract=contract,
        date_from=date_from,
        date_to=date_to,
        score_min=score_min,
        score_max=score_max,
        model_version=model_version,
        review_status=review_status,
        flagged=flagged,
        batch_job_id=batch_job_id,
        q=q,
    )
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    order = {
        "created_desc": Prediction.created_at.desc(),
        "created_asc": Prediction.created_at.asc(),
        "score_desc": Prediction.score.desc(),
        "score_asc": Prediction.score.asc(),
    }[sort]
    rows = db.scalars(stmt.order_by(order, Prediction.id).offset((page - 1) * page_size).limit(page_size))
    items = [to_out(p) for p in rows.unique()]
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.get("/predictions/{prediction_id}", response_model=PredictionDetail)
def prediction_detail(prediction_id: uuid.UUID, user: CurrentUser, db: DB) -> PredictionDetail:
    p = _get_owned(db, user, prediction_id)
    base = to_out(p, label=_label_for(db, p.transaction_pk))
    entry = MODEL_STORE.registry.read()["bundles"].get(p.model_version, {})
    trace = {
        "model_version": p.model_version,
        "model_name": entry.get("model_name"),
        "preprocessing_version": p.preprocessing_version,
        "threshold_applied": p.threshold,
        "threshold_policy": entry.get("threshold_policy"),
        "score_kind": p.score_kind,
        "calibration": entry.get("calibration"),
        "training_session": entry.get("training_session"),
        "dataset_fingerprint": entry.get("dataset_fingerprint"),
        "mlflow_run_id": entry.get("mlflow_run_id"),
        "bundle_sha256": entry.get("sha256"),
        "code_revision": entry.get("code_revision"),
    }
    return PredictionDetail(
        **base.model_dump(),
        transaction=p.transaction.payload,
        features=p.features,
        explanation=p.explanation,
        trace=trace,
    )


@router.get("/predictions/{prediction_id}/explanation")
def get_explanation(prediction_id: uuid.UUID, user: CurrentUser, db: DB) -> dict[str, Any]:
    p = _get_owned(db, user, prediction_id)
    return {"status": p.explanation_status, "explanation": p.explanation}


@router.post("/predictions/{prediction_id}/explanation")
def compute_explanation(prediction_id: uuid.UUID, user: Analyst, db: DB) -> dict[str, Any]:
    """Compute (or recompute after a failure) the explanation from the stored feature vector."""
    p = _get_owned(db, user, prediction_id)
    if p.explanation_status != ExplanationStatus.ok:
        explain_prediction(p)
        db.flush()
    return {"status": p.explanation_status, "explanation": p.explanation}


# ------------------------------------------------------------------------------ reviews
@router.get("/reviews/queue", response_model=Page[PredictionOut])
def review_queue(
    user: CurrentUser,
    db: DB,
    contract: str | None = None,
    include_resolved: bool = False,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> Page[PredictionOut]:
    """Flagged predictions awaiting review, highest score first."""
    stmt = (
        select(Prediction)
        .join(Review, Review.prediction_id == Prediction.id)
        .join(Transaction, Prediction.transaction_pk == Transaction.id)
        .where(Prediction.workspace_id == user.workspace_id, Prediction.flagged.is_(True))
    )
    if not include_resolved:
        stmt = stmt.where(Review.status != ReviewStatus.resolved)
    if contract:
        stmt = stmt.where(Transaction.contract == contract)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(Prediction.score.desc(), Prediction.id).offset((page - 1) * page_size).limit(page_size)
    )
    return Page(items=[to_out(p) for p in rows.unique()], total=total, page=page, page_size=page_size)


@router.put("/predictions/{prediction_id}/review", response_model=ReviewOut)
def update_review(prediction_id: uuid.UUID, body: ReviewUpdate, user: Analyst, db: DB) -> ReviewOut:
    """Record the analyst's workflow status and documented opinion.

    This is an OPINION, kept separate from confirmed labels; it never becomes a training label.
    Resolving requires an assessment so outcomes are always documented.
    """
    p = _get_owned(db, user, prediction_id)
    if body.status == ReviewStatus.resolved and body.assessment is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Resolving a review requires an assessment"
        )
    review = p.review or Review(workspace_id=user.workspace_id, prediction_id=p.id)
    previous = review.status.value if p.review else None
    review.status = body.status
    review.assessment = body.assessment
    review.note = body.note
    review.reviewer_id = user.id
    db.add(review)
    db.flush()
    db.add(
        ReviewEvent(
            review_id=review.id,
            actor_id=user.id,
            from_status=previous,
            to_status=body.status.value,
            assessment=body.assessment.value if body.assessment else None,
            note=body.note,
        )
    )
    db.flush()
    db.refresh(review)
    return ReviewOut.model_validate(review)


@router.get("/predictions/{prediction_id}/review/history", response_model=list[ReviewEventOut])
def review_history(prediction_id: uuid.UUID, user: CurrentUser, db: DB) -> list[ReviewEventOut]:
    p = _get_owned(db, user, prediction_id)
    if not p.review:
        return []
    events = db.scalars(
        select(ReviewEvent).where(ReviewEvent.review_id == p.review.id).order_by(ReviewEvent.created_at)
    )
    return [ReviewEventOut.model_validate(e) for e in events]


@router.put("/predictions/{prediction_id}/label")
def record_confirmed_label(prediction_id: uuid.UUID, body: LabelIn, user: Admin, db: DB) -> dict[str, Any]:
    """Record an independently confirmed outcome (admin only). Used for monitoring metrics,
    never for automatic retraining."""
    p = _get_owned(db, user, prediction_id)
    lab = db.scalars(select(ConfirmedLabel).where(ConfirmedLabel.transaction_pk == p.transaction_pk)).first()
    lab = lab or ConfirmedLabel(workspace_id=user.workspace_id, transaction_pk=p.transaction_pk)
    lab.is_fraud, lab.source, lab.recorded_by = body.is_fraud, body.source, user.id
    db.add(lab)
    db.flush()
    return {"transaction_id": p.transaction.transaction_id, "is_fraud": lab.is_fraud, "source": lab.source}
