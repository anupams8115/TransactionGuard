"""Online scoring: shared by the API (single transactions) and the Celery worker (batches).

Flow for a list of validated transactions:
  1. Store each transaction once per workspace (idempotent; same id + different payload = conflict).
  2. Reuse any prediction that already exists for (transaction, model_version)  -> no duplicates.
  3. For the rest, rebuild features with the SAME stateless function used in training, feeding it
     the account's earlier transactions loaded from the database (history features).
  4. Score, apply the bundle's threshold, store, and optionally explain.
Database unique constraints make steps 1-2 safe under concurrent requests and job retries.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger
from transactionguard.data.contract import DataContract, get_contract
from transactionguard.db.models import (
    ExplanationStatus,
    ModelVersion,
    Prediction,
    Recommendation,
    Review,
    ReviewStatus,
    Transaction,
)
from transactionguard.db.session import as_utc
from transactionguard.inference.bundle import BundleRegistry, ModelBundle

log = get_logger(__name__)


class ScoringConflictError(ValueError):
    """Same transaction_id submitted again with a different payload."""


class ModelNotAvailableError(RuntimeError):
    pass


# ============================================================================ model store
class ModelStore:
    """Thread-safe cache of trusted bundles; notices when the registry's active model changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundles: dict[str, ModelBundle] = {}
        self._explainers: dict[str, Any] = {}

    @property
    def registry(self) -> BundleRegistry:
        return BundleRegistry(get_settings().bundles_dir)

    def get(self, model_version: str) -> ModelBundle:
        with self._lock:
            if model_version not in self._bundles:
                self._bundles[model_version] = self.registry.load(model_version)
            return self._bundles[model_version]

    def active(self, contract: str) -> ModelBundle:
        version = self.registry.active_version(contract)
        if version is None:
            raise ModelNotAvailableError(
                f"No active model for '{contract}'. Train one with `tg train {contract}`."
            )
        return self.get(version)

    def explainer(self, bundle: ModelBundle) -> Any:
        from transactionguard.inference.explain import BundleExplainer

        with self._lock:
            if bundle.model_version not in self._explainers:
                self._explainers[bundle.model_version] = BundleExplainer(bundle)
            return self._explainers[bundle.model_version]

    def clear(self) -> None:
        with self._lock:
            self._bundles.clear()
            self._explainers.clear()


MODEL_STORE = ModelStore()


def sync_model_versions(session: Session) -> int:
    """Mirror the trusted registry into the model_versions table (FK target for predictions)."""
    reg = MODEL_STORE.registry.read()
    active = set(reg.get("active", {}).values())
    n = 0
    for version, e in reg.get("bundles", {}).items():
        row = session.get(ModelVersion, version) or ModelVersion(model_version=version)
        row.contract, row.model_name = e["contract"], e["model_name"]
        row.score_kind, row.threshold = e["score_kind"], float(e["threshold"])
        row.preprocessing_version = e.get("preprocessing_version", "unknown")
        row.mlflow_run_id, row.sha256 = e.get("mlflow_run_id"), e["sha256"]
        row.is_active = version in active
        session.add(row)
        n += 1
    session.flush()
    return n


# ============================================================================ helpers
def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, (pd.Timestamp, datetime)):
            out[k] = (
                pd.Timestamp(v).tz_convert("UTC").isoformat()
                if pd.Timestamp(v).tzinfo
                else pd.Timestamp(v).tz_localize("UTC").isoformat()
            )
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _event_fields(contract: DataContract, row: dict[str, Any]) -> tuple[datetime | None, float | None]:
    ts = row[contract.timestamp_column]
    if contract.field(contract.timestamp_column).dtype == "datetime":
        return pd.Timestamp(ts).tz_convert("UTC").to_pydatetime() if pd.Timestamp(
            ts
        ).tzinfo else pd.Timestamp(ts).tz_localize("UTC").to_pydatetime(), None
    return None, float(ts)


def _raw_frame(contract: DataContract, payloads: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(payloads, columns=contract.input_column_names)
    for f in contract.input_fields:
        if f.dtype == "datetime":
            df[f.name] = pd.to_datetime(df[f.name], utc=True)
        elif f.dtype in ("float", "int"):
            df[f.name] = pd.to_numeric(df[f.name])
    return df


def _unseen_categories(bundle: ModelBundle, X: pd.DataFrame) -> list[list[str]]:
    """Per row: categorical values valid under the contract but never seen in training."""
    spec = bundle.feature_spec
    if not spec.categorical:
        return [[] for _ in range(len(X))]
    enc = bundle.preprocessor.named_steps["columns"].named_transformers_["cat"]
    known = {c: set(map(str, cats)) for c, cats in zip(spec.categorical, enc.categories_, strict=True)}
    return [
        [f"{c}={row[c]}" for c in spec.categorical if str(row[c]) not in known[c]] for _, row in X.iterrows()
    ]


def _feature_record(row: pd.Series) -> dict[str, Any]:
    return {
        k: (None if (isinstance(v, float) and np.isnan(v)) else v.item() if isinstance(v, np.generic) else v)
        for k, v in row.items()
    }


def features_from_record(bundle: ModelBundle, record: dict[str, Any]) -> pd.DataFrame:
    X = pd.DataFrame([record], columns=bundle.feature_spec.all)
    for c in (*bundle.feature_spec.amount, *bundle.feature_spec.numeric):
        X[c] = pd.to_numeric(X[c]).astype("float64")  # None -> NaN (genuinely missing history)
    return X


@dataclass
class ScoreOutcome:
    prediction: Prediction
    replayed: bool  # True if this prediction already existed (idempotent replay)


@dataclass
class BatchResult:
    outcomes: list[ScoreOutcome] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)


# ============================================================================ scoring
def score_transactions(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    contract_name: str,
    payloads: list[dict[str, Any]],
    bundle: ModelBundle | None = None,
    user_id: uuid.UUID | None = None,
    batch_job_id: uuid.UUID | None = None,
    explain: bool = False,
) -> BatchResult:
    """Score already-validated payloads. Commits are left to the caller."""
    contract = get_contract(contract_name)
    bundle = bundle or MODEL_STORE.active(contract_name)
    if bundle.contract_name != contract_name:
        raise ModelNotAvailableError("model/contract mismatch")
    t0 = time.perf_counter()
    result = BatchResult()
    if session.get(ModelVersion, bundle.model_version) is None:
        sync_model_versions(session)  # FK target for predictions

    # ---- 1. store transactions (idempotent) --------------------------------------------------
    payloads = [_jsonable(p) for p in payloads]
    ids = [str(p[contract.id_column]) for p in payloads]
    existing = {
        t.transaction_id: t
        for t in session.scalars(
            select(Transaction).where(
                Transaction.workspace_id == workspace_id,
                Transaction.contract == contract_name,
                Transaction.transaction_id.in_(ids),
            )
        )
    }
    txns: list[Transaction] = []
    for p in payloads:
        tid = str(p[contract.id_column])
        h = payload_hash(p)
        txn = existing.get(tid)
        if txn is not None:
            if txn.payload_sha256 != h:
                result.conflicts.append(
                    {"transaction_id": tid, "reason": "transaction_id already used with a different payload"}
                )
                continue
        else:
            event_time, event_seconds = _event_fields(contract, p)
            txn = Transaction(
                workspace_id=workspace_id,
                contract=contract_name,
                transaction_id=tid,
                account_id=p.get(contract.entity_column) if contract.entity_column else None,
                event_time=event_time,
                event_seconds=event_seconds,
                payload=p,
                payload_sha256=h,
                synthetic=contract.provenance == "synthetic",
            )
            session.add(txn)
            existing[tid] = txn
        txns.append(txn)
    session.flush()

    # ---- 2. reuse existing predictions -----------------------------------------------------
    by_txn = {
        p.transaction_pk: p
        for p in session.scalars(
            select(Prediction).where(
                Prediction.transaction_pk.in_([t.id for t in txns]),
                Prediction.model_version == bundle.model_version,
            )
        )
    }
    todo = [t for t in txns if t.id not in by_txn]
    for t in txns:
        if t.id in by_txn:
            result.outcomes.append(ScoreOutcome(by_txn[t.id], replayed=True))
    if not todo:
        return result

    # ---- 3. features with account history from the database --------------------------------
    new_raw = _raw_frame(contract, [t.payload for t in todo])
    if contract.entity_column:
        accounts = sorted({t.account_id for t in todo if t.account_id})
        latest = max(as_utc(t.event_time) for t in todo)  # type: ignore[type-var]
        history = session.scalars(
            select(Transaction).where(
                Transaction.workspace_id == workspace_id,
                Transaction.contract == contract_name,
                Transaction.account_id.in_(accounts),
                Transaction.event_time < latest,
                Transaction.id.not_in([t.id for t in todo]),
            )
        ).all()
        hist_raw = _raw_frame(contract, [h.payload for h in history])
        combined = pd.concat([hist_raw, new_raw], ignore_index=True)
        X = bundle.features(combined).iloc[len(hist_raw) :].reset_index(drop=True)
    else:
        X = bundle.features(new_raw)

    # ---- 4. score + threshold ---------------------------------------------------------------
    scores = bundle.score(X)
    flags = bundle.flag(scores)
    unseen = _unseen_categories(bundle, X)
    per_row_ms = (time.perf_counter() - t0) * 1000 / len(todo)

    explanations: list[Any] = [None] * len(todo)
    if explain:
        explanations = MODEL_STORE.explainer(bundle).explain(X)

    preds: list[Prediction] = []
    for i, t in enumerate(todo):
        status = ExplanationStatus.not_requested
        exp_json = None
        if explain:
            exp_json = explanations[i].to_dict()
            status = ExplanationStatus(explanations[i].status)
        elif flags[i] and batch_job_id is not None:
            status = ExplanationStatus.pending  # worker explains flagged batch rows afterwards
        preds.append(
            Prediction(
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                transaction_pk=t.id,
                model_version=bundle.model_version,
                preprocessing_version=bundle.preprocessing_version,
                score=float(scores[i]),
                score_kind=bundle.score_kind,
                threshold=bundle.threshold,
                flagged=bool(flags[i]),
                recommendation=Recommendation.review if flags[i] else Recommendation.no_action,
                explanation_status=status,
                explanation=exp_json,
                warnings=[f"category unseen in training: {u}" for u in unseen[i]],
                features=_feature_record(X.iloc[i]),
                latency_ms=per_row_ms,
                batch_job_id=batch_job_id,
                created_by=user_id,
            )
        )

    def _reviews(ps: list[Prediction]) -> list[Review]:
        return [
            Review(workspace_id=workspace_id, prediction_id=p.id, status=ReviewStatus.unreviewed)
            for p in ps
            if p.flagged
        ]

    try:
        # Fast path: insert everything in one SAVEPOINT.
        with session.begin_nested():
            session.add_all(preds)
            session.flush()
            session.add_all(_reviews(preds))
            session.flush()
        result.outcomes += [ScoreOutcome(p, replayed=False) for p in preds]
    except IntegrityError:
        # Slow path (rare): a concurrent request or retried job stored some rows first.
        # Retry row by row so only the conflicting rows are replaced by the stored winners.
        for p in preds:
            fresh = Prediction(**{c.key: getattr(p, c.key) for c in Prediction.__table__.columns})
            try:
                with session.begin_nested():
                    session.add(fresh)
                    session.flush()
                    session.add_all(_reviews([fresh]))
                    session.flush()
                result.outcomes.append(ScoreOutcome(fresh, replayed=False))
            except IntegrityError:
                winner = session.scalars(
                    select(Prediction).where(
                        Prediction.transaction_pk == p.transaction_pk,
                        Prediction.model_version == bundle.model_version,
                    )
                ).one()
                result.outcomes.append(ScoreOutcome(winner, replayed=True))

    log.info(
        "transactions_scored",
        contract=contract_name,
        model_version=bundle.model_version,
        n_new=len(todo),
        n_replayed=len(txns) - len(todo),
        n_conflicts=len(result.conflicts),
        n_flagged=int(flags.sum()),
        ms_per_row=round(per_row_ms, 2),
    )
    return result


def explain_prediction(pred: Prediction) -> Prediction:
    """(Re)compute the explanation from the feature vector stored at scoring time."""
    bundle = MODEL_STORE.get(pred.model_version)
    X = features_from_record(bundle, pred.features)
    exp = MODEL_STORE.explainer(bundle).explain(X)[0]
    pred.explanation = exp.to_dict()
    pred.explanation_status = ExplanationStatus(exp.status)
    return pred


def now_utc() -> datetime:
    return datetime.now(UTC)
