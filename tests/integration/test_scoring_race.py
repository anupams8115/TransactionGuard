"""Duplicate-safety under a race: another writer stores the same prediction between our
"does it already exist?" check and our insert. The unique constraint must reject our copy and
the service must return the stored winner, without failing the other rows in the batch."""

from __future__ import annotations

from sqlalchemy import func, select

from transactionguard.db.models import Prediction, Workspace
from transactionguard.db.session import session_scope


def test_unique_constraint_resolves_race(clients, sample_rows, monkeypatch):
    import transactionguard.inference.service as service

    rows, _ = sample_rows
    batch = rows[:5]
    original = service._unseen_categories
    winner_ids = []

    def race_then_continue(bundle, X):
        # Runs after the existence check, before the insert: simulate a concurrent writer.
        session = race_then_continue.session
        txn = session.scalars(
            select(service.Transaction).where(
                service.Transaction.transaction_id == batch[2]["transaction_id"]
            )
        ).one()
        p = Prediction(
            workspace_id=txn.workspace_id,
            transaction_pk=txn.id,
            model_version=bundle.model_version,
            preprocessing_version=bundle.preprocessing_version,
            score=0.5,
            score_kind=bundle.score_kind,
            threshold=bundle.threshold,
            flagged=False,
            recommendation=service.Recommendation.no_action,
            features={},
            latency_ms=0.0,
        )
        session.add(p)
        session.flush()
        winner_ids.append(p.id)
        return original(bundle, X)

    monkeypatch.setattr(service, "_unseen_categories", race_then_continue)
    with session_scope() as db:
        ws = db.scalars(select(Workspace).where(Workspace.name == "alpha")).one()
        race_then_continue.session = db
        result = service.score_transactions(db, workspace_id=ws.id, contract_name="synthetic", payloads=batch)
        replayed = [o for o in result.outcomes if o.replayed]
        assert len(result.outcomes) == 5
        assert len(replayed) == 1 and replayed[0].prediction.id == winner_ids[0]
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Prediction)) == 5
