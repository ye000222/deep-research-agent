"""Read/write access for the evidence extraction cache."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.infrastructure.db.extraction_cache_models import ExtractionCacheRow


class ExtractionCacheRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(
        self,
        run_id: UUID,
        *,
        question_hash: str,
        snapshot_hash: str,
        adapter: str,
        model: str,
        extractor_version: str,
    ) -> list[dict[str, Any]] | None:
        """Return the cached evidence items for a matching key, or None on miss."""
        row = await self._lookup(
            run_id,
            question_hash=question_hash,
            snapshot_hash=snapshot_hash,
            adapter=adapter,
            model=model,
            extractor_version=extractor_version,
        )
        return list(row.evidence) if row is not None else None

    async def put(
        self,
        run_id: UUID,
        *,
        question_hash: str,
        snapshot_hash: str,
        adapter: str,
        model: str,
        extractor_version: str,
        evidence: list[dict[str, Any]],
    ) -> bool:
        """Store a successful, non-empty extraction. Idempotent per key."""
        if not evidence:
            return False  # an empty result is never cached as success
        row = await self._lookup(
            run_id,
            question_hash=question_hash,
            snapshot_hash=snapshot_hash,
            adapter=adapter,
            model=model,
            extractor_version=extractor_version,
        )
        if row is not None:
            return True  # already cached; keep the first successful result
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    ExtractionCacheRow(
                        id=uuid7(),
                        run_id=run_id,
                        question_hash=question_hash,
                        snapshot_hash=snapshot_hash,
                        adapter=adapter,
                        model=model,
                        extractor_version=extractor_version,
                        evidence=evidence,
                        created_at=datetime.now(UTC),
                    )
                )
        except IntegrityError:
            # Another worker won the same-key race. The cache contract is
            # idempotent, so only treat the error as success when that winner
            # is actually visible; unrelated integrity errors must surface.
            winner = await self._lookup(
                run_id,
                question_hash=question_hash,
                snapshot_hash=snapshot_hash,
                adapter=adapter,
                model=model,
                extractor_version=extractor_version,
            )
            if winner is not None:
                return True
            raise
        return True

    async def _lookup(
        self,
        run_id: UUID,
        *,
        question_hash: str,
        snapshot_hash: str,
        adapter: str,
        model: str,
        extractor_version: str,
    ) -> ExtractionCacheRow | None:
        async with self._sessions() as session:
            return cast(
                ExtractionCacheRow | None,
                await session.scalar(
                    select(ExtractionCacheRow).where(
                        ExtractionCacheRow.run_id == run_id,
                        ExtractionCacheRow.question_hash == question_hash,
                        ExtractionCacheRow.snapshot_hash == snapshot_hash,
                        ExtractionCacheRow.adapter == adapter,
                        ExtractionCacheRow.model == model,
                        ExtractionCacheRow.extractor_version == extractor_version,
                    )
                ),
            )
