"""Hand work to Celery, or run it in-process after the response when Redis isn't available.

``TG_CELERY_EAGER=true`` (default for local development): FastAPI ``BackgroundTasks`` run the task
*after* the HTTP response is sent, so the UI still sees queued -> running -> succeeded.
``TG_CELERY_EAGER=false`` (Docker Compose): tasks go to Redis and a separate worker process.
"""

from __future__ import annotations

from typing import Any

from celery import Task
from fastapi import BackgroundTasks

from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger

log = get_logger(__name__)


def dispatch(task: Task, *args: Any, background: BackgroundTasks | None = None) -> str | None:
    if get_settings().celery_eager:
        if background is not None:
            background.add_task(task.apply, args=list(args))
        else:
            task.apply(args=list(args))
        return None
    result = task.apply_async(args=list(args))
    log.info("task_enqueued", task=task.name, task_id=result.id)
    return result.id
