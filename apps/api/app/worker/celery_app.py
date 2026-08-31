"""Celery application configuration."""

from celery import Celery  # type: ignore[import-untyped]

from app.core.config import Settings

settings = Settings()

celery_app = Celery("deep_research_agent", broker=settings.redis_url)
celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    task_ignore_result=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    accept_content=("json",),
    timezone="UTC",
)
celery_app.autodiscover_tasks(("app.worker",), force=True)
