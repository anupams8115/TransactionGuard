"""Celery application.

Reliability settings (interview talking points):
  * ``task_acks_late`` + ``task_reject_on_worker_lost``: a message is acknowledged only after the
    task finishes, so a crashed worker's job is redelivered instead of lost (at-least-once).
    Redelivery is safe because scoring is idempotent (DB unique constraints).
  * ``worker_prefetch_multiplier=1``: long batch jobs are not hoarded by one worker.
  * Hard + soft time limits so a stuck job fails visibly instead of hanging forever.
  * Bounded retries with exponential backoff for transient errors only (see tasks.py).

Start a worker:  uv run celery -A transactionguard.worker.celery_app worker -l info -P solo
(``-P solo`` on Windows; Linux/Docker can use the default prefork pool.)
"""

from __future__ import annotations

from celery import Celery

from transactionguard.core.config import get_settings

celery_app = Celery("transactionguard", include=["transactionguard.worker.tasks"])


def configure_celery() -> Celery:
    s = get_settings()
    celery_app.conf.update(
        broker_url=s.redis_url,
        result_backend=s.redis_url,
        task_always_eager=s.celery_eager,
        task_eager_propagates=False,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=s.celery_task_time_limit_s,
        task_soft_time_limit=max(30, s.celery_task_time_limit_s - 30),
        task_serializer="json",  # never pickle messages from the broker
        accept_content=["json"],
        result_expires=24 * 3600,
        broker_connection_retry_on_startup=True,
        timezone="UTC",
    )
    return celery_app


configure_celery()
