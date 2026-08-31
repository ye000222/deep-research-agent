"""PostgreSQL outbox dispatcher for Celery research tasks."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import Settings
from app.infrastructure.db.outbox import TaskDispatchOutboxRepository
from app.infrastructure.db.postgres import PostgresRuntime
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


async def dispatch_forever() -> None:
    settings = Settings()
    database = PostgresRuntime(settings.database_url)
    repository = TaskDispatchOutboxRepository(database.session_factory)
    try:
        while True:
            batch = await repository.claim_batch()
            for dispatch in batch:
                try:
                    await asyncio.to_thread(
                        celery_app.send_task,
                        "deep_research.execute_run",
                        args=(str(dispatch.run_id),),
                        task_id=dispatch.dispatch_key,
                    )
                except Exception as exc:
                    error_code = type(exc).__name__
                    logger.warning(
                        "outbox_publish_failed",
                        extra={"outbox_id": str(dispatch.outbox_id), "error_code": error_code},
                    )
                    await repository.mark_retry(
                        dispatch.outbox_id,
                        error_code=error_code,
                    )
                else:
                    await repository.mark_published(dispatch.outbox_id)
            await asyncio.sleep(0.25 if batch else 1.0)
    finally:
        await database.close()


def main() -> None:
    asyncio.run(dispatch_forever())


if __name__ == "__main__":
    main()
