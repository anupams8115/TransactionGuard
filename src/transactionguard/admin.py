"""Administrative operations shared by the CLI and tests: migrations, users, demo data."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from transactionguard.api.security import hash_password
from transactionguard.core.config import PROJECT_ROOT, get_settings
from transactionguard.core.logging import get_logger
from transactionguard.data import sources
from transactionguard.data.contract import get_contract
from transactionguard.db.models import ConfirmedLabel, LabelSource, Role, Transaction, User, Workspace
from transactionguard.db.session import session_scope
from transactionguard.inference.service import (
    MODEL_STORE,
    _event_fields,
    _jsonable,
    payload_hash,
    score_transactions,
    sync_model_versions,
)

log = get_logger(__name__)


def upgrade_db(revision: str = "head") -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(cfg, revision)


def get_or_create_workspace(db: Session, name: str) -> Workspace:
    ws = db.scalars(select(Workspace).where(Workspace.name == name)).first()
    if ws is None:
        ws = Workspace(name=name)
        db.add(ws)
        db.flush()
    return ws


def create_user(db: Session, *, email: str, password: str, role: Role, workspace: str) -> User:
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters")
    ws = get_or_create_workspace(db, workspace)
    existing = db.scalars(select(User).where(func.lower(User.email) == email.lower())).first()
    if existing is not None:
        raise ValueError(f"User {email} already exists")
    user = User(workspace_id=ws.id, email=email.lower(), password_hash=hash_password(password), role=role)
    db.add(user)
    db.flush()
    return user


def seed_demo(
    *,
    workspace: str = "demo",
    contract_name: str = "synthetic",
    n_score: int = 2000,
    history_days: int = 30,
    password: str | None = None,
    replay_labels: bool = True,
) -> dict:
    """Create demo users and score the most recent ``n_score`` transactions of a prepared dataset.

    Account history: transactions from the ``history_days`` before the scored window are imported
    first (stored, not scored) so history features match how they were computed in training.
    ``replay_labels`` records the dataset's own labels as confirmed labels with source
    ``dataset_label`` so the monitoring page can demonstrate performance tracking. These are
    clearly marked and never used for training.
    """
    contract = get_contract(contract_name)
    bundle = MODEL_STORE.active(contract_name)
    df = sources.load(contract_name).sort_values([contract.timestamp_column, contract.id_column])
    scored = df.tail(n_score)
    ts = contract.timestamp_column
    window_start = scored[ts].min()
    if contract.field(ts).dtype == "datetime":
        hist = df[(df[ts] < window_start) & (df[ts] >= window_start - pd.Timedelta(days=history_days))]
        if contract.entity_column:
            hist = hist[hist[contract.entity_column].isin(scored[contract.entity_column])]
    else:
        hist = df.iloc[0:0]

    creds = {}
    with session_scope() as db:
        sync_model_versions(db)
        ws = get_or_create_workspace(db, workspace)
        for role in (Role.admin, Role.analyst, Role.viewer):
            email = f"{role.value}@{workspace}.example"
            if db.scalars(select(User).where(User.email == email)).first() is None:
                pw = password or secrets.token_urlsafe(12)
                create_user(db, email=email, password=pw, role=role, workspace=workspace)
                creds[email] = pw

        existing = set(
            db.scalars(
                select(Transaction.transaction_id).where(
                    Transaction.workspace_id == ws.id, Transaction.contract == contract_name
                )
            )
        )
        for row in hist[contract.input_column_names].to_dict(orient="records"):
            p = _jsonable(row)
            if p[contract.id_column] in existing:
                continue
            et, es = _event_fields(contract, p)
            db.add(
                Transaction(
                    workspace_id=ws.id,
                    contract=contract_name,
                    transaction_id=p[contract.id_column],
                    account_id=p.get(contract.entity_column) if contract.entity_column else None,
                    event_time=et,
                    event_seconds=es,
                    payload=p,
                    payload_sha256=payload_hash(p),
                    synthetic=contract.provenance == "synthetic",
                )
            )
        db.flush()
        admin = db.scalars(select(User).where(User.workspace_id == ws.id, User.role == Role.admin)).first()
        ws_id, admin_id = ws.id, admin.id if admin else None

    # One transaction per chunk keeps the session small (no O(n^2) flush overhead) and makes
    # progress durable, exactly like the batch worker.
    payloads = scored[contract.input_column_names].to_dict(orient="records")
    n_flagged = 0
    for start in range(0, len(payloads), 500):
        with session_scope() as db:
            res = score_transactions(
                db,
                workspace_id=ws_id,
                contract_name=contract_name,
                payloads=payloads[start : start + 500],
                bundle=bundle,
                user_id=admin_id,
                explain=True,
            )
            n_flagged += sum(o.prediction.flagged for o in res.outcomes if not o.replayed)

    with session_scope() as db:
        ws = db.get(Workspace, ws_id)
        assert ws is not None
        if replay_labels:
            ids = dict(
                db.execute(
                    select(Transaction.transaction_id, Transaction.id).where(
                        Transaction.workspace_id == ws.id,
                        Transaction.contract == contract_name,
                        Transaction.transaction_id.in_(scored[contract.id_column].tolist()),
                    )
                ).all()
            )
            have = set(
                db.scalars(select(ConfirmedLabel.transaction_pk).where(ConfirmedLabel.workspace_id == ws.id))
            )
            for tid, y in zip(scored[contract.id_column], scored[contract.target_column], strict=True):
                pk = ids.get(tid)
                if pk is not None and pk not in have and admin_id is not None:
                    db.add(
                        ConfirmedLabel(
                            workspace_id=ws.id,
                            transaction_pk=pk,
                            is_fraud=bool(y),
                            source=LabelSource.dataset_label,
                            recorded_by=admin_id,
                            confirmed_at=datetime.now(UTC),
                        )
                    )
    log.info("demo_seeded", workspace=workspace, scored=len(payloads), history=len(hist), flagged=n_flagged)
    return {
        "workspace": workspace,
        "contract": contract_name,
        "scored": len(payloads),
        "history_imported": len(hist),
        "flagged": n_flagged,
        "new_credentials": creds,
        "model_version": bundle.model_version,
    }
