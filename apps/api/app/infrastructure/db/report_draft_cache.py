"""Read/write access for the per-run report draft cache."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.infrastructure.db.report_draft_cache_models import ReportDraftCacheRow


class ReportDraftCacheRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(
        self,
        run_id: UUID,
        *,
        context_hash: str,
        writer_version: str,
    ) -> dict[str, Any] | None:
        row = await self._lookup(run_id, context_hash=context_hash, writer_version=writer_version)
        return dict(row.draft) if row is not None else None

    async def put(
        self,
        run_id: UUID,
        *,
        context_hash: str,
        writer_version: str,
        draft: dict[str, Any],
    ) -> bool:
        """Store a generated draft. Idempotent per (run, context)."""
        if not draft:
            return False
        row = await self._lookup(run_id, context_hash=context_hash, writer_version=writer_version)
        if row is not None:
            return True  # keep the first draft generated for this context
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    ReportDraftCacheRow(
                        id=uuid7(),
                        run_id=run_id,
                        context_hash=context_hash,
                        writer_version=writer_version,
                        draft=draft,
                        created_at=datetime.now(UTC),
                    )
                )
        except IntegrityError:
            # Concurrent writers may race between lookup and insert. The
            # existing winner is an acceptable idempotent result, but do not
            # hide unrelated foreign-key or data-integrity failures.
            winner = await self._lookup(
                run_id,
                context_hash=context_hash,
                writer_version=writer_version,
            )
            if winner is not None:
                return True
            raise
        return True

    async def invalidate_run(self, run_id: UUID) -> int:
        """Delete drafts after evidence, conflict, or citation dependencies change."""
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                delete(ReportDraftCacheRow).where(ReportDraftCacheRow.run_id == run_id)
            )
            return int(getattr(result, "rowcount", 0) or 0)

    async def _lookup(
        self,
        run_id: UUID,
        *,
        context_hash: str,
        writer_version: str,
    ) -> ReportDraftCacheRow | None:
        async with self._sessions() as session:
            return cast(
                ReportDraftCacheRow | None,
                await session.scalar(
                    select(ReportDraftCacheRow).where(
                        ReportDraftCacheRow.run_id == run_id,
                        ReportDraftCacheRow.context_hash == context_hash,
                        ReportDraftCacheRow.writer_version == writer_version,
                    )
                ),
            )
