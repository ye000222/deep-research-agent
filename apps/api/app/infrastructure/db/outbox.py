"""Transactional outbox claims for reliable Celery dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.db.run_models import TaskDispatchOutboxRow


@dataclass(frozen=True, slots=True)
class PendingDispatch:
    outbox_id: UUID
    run_id: UUID
    dispatch_key: str


class TaskDispatchOutboxRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def claim_batch(
        self,
        *,
        limit: int = 20,
        stale_after_seconds: int = 30,
    ) -> list[PendingDispatch]:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=stale_after_seconds)
        async with self._sessions() as session, session.begin():
            rows = (
                await session.scalars(
                    select(TaskDispatchOutboxRow)
                    .where(
                        or_(
                            and_(
                                TaskDispatchOutboxRow.status.in_(("pending", "retry")),
                                TaskDispatchOutboxRow.next_attempt_at <= now,
                            ),
                            and_(
                                TaskDispatchOutboxRow.status == "publishing",
                                TaskDispatchOutboxRow.claimed_at <= stale_before,
                            ),
                        )
                    )
                    .order_by(TaskDispatchOutboxRow.created_at)
                    .with_for_update(skip_locked=True)
                    .limit(limit)
                )
            ).all()
            for row in rows:
                row.status = "publishing"
                row.claimed_at = now
                row.attempt_count += 1
            await session.flush()
            return [
                PendingDispatch(
                    outbox_id=row.id,
                    run_id=row.run_id,
                    dispatch_key=row.dispatch_key,
                )
                for row in rows
            ]

    async def mark_published(self, outbox_id: UUID) -> None:
        async with self._sessions() as session, session.begin():
            row = await self._lock(session, outbox_id)
            row.status = "published"
            row.published_at = datetime.now(UTC)
            row.last_error = None

    async def mark_retry(self, outbox_id: UUID, *, error_code: str) -> None:
        async with self._sessions() as session, session.begin():
            row = await self._lock(session, outbox_id)
            delay = min(2 ** max(row.attempt_count - 1, 0), 60)
            row.status = "retry"
            row.next_attempt_at = datetime.now(UTC) + timedelta(seconds=delay)
            row.last_error = error_code[:1000]

    @staticmethod
    async def _lock(session: AsyncSession, outbox_id: UUID) -> TaskDispatchOutboxRow:
        row = await session.scalar(
            select(TaskDispatchOutboxRow)
            .where(TaskDispatchOutboxRow.id == outbox_id)
            .with_for_update()
        )
        if row is None:
            raise LookupError(str(outbox_id))
        return row
