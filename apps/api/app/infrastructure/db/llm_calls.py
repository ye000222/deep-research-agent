"""Repository for redacted, per-run LLM call diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.infrastructure.db.llm_call_models import LLMCallRow
from app.infrastructure.db.run_models import ResearchRunRow


class LLMCallRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, payload: dict[str, Any]) -> None:
        run_id = UUID(str(payload["run_id"]))
        async with self._sessions() as session, session.begin():
            session.add(
                LLMCallRow(
                    id=uuid7(),
                    run_id=run_id,
                    node=str(payload.get("node", "unknown")),
                    adapter=str(payload.get("adapter", "unknown")),
                    model=str(payload.get("model", "unknown")),
                    strategy=str(payload.get("strategy", "unknown")),
                    provider_request_id=payload.get("provider_request_id"),
                    context_manifest_id=(
                        UUID(str(payload["context_manifest_id"]))
                        if payload.get("context_manifest_id")
                        else None
                    ),
                    finish_reason=payload.get("finish_reason"),
                    status=str(payload.get("status", "unknown")),
                    usage=dict(payload.get("usage", {})),
                    latency_ms=max(0, int(payload.get("latency_ms", 0))),
                    retry_mode=str(payload.get("retry_mode", "none")),
                    error_code=payload.get("error_code"),
                    detail_code=payload.get("detail_code"),
                    diagnostics=dict(payload.get("diagnostics", {})),
                    created_at=datetime.now(UTC),
                )
            )

    async def list_for_run(self, owner_hash: str, run_id: UUID) -> list[dict[str, object]]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                return []
            rows = (
                await session.scalars(
                    select(LLMCallRow)
                    .where(LLMCallRow.run_id == run_id)
                    .order_by(LLMCallRow.created_at, LLMCallRow.id)
                )
            ).all()
            return [
                {
                    "call_id": row.id,
                    "run_id": row.run_id,
                    "node": row.node,
                    "adapter": row.adapter,
                    "model": row.model,
                    "strategy": row.strategy,
                    "provider_request_id": row.provider_request_id,
                    "context_manifest_id": row.context_manifest_id,
                    "finish_reason": row.finish_reason,
                    "status": row.status,
                    "usage": row.usage,
                    "latency_ms": row.latency_ms,
                    "retry_mode": row.retry_mode,
                    "error_code": row.error_code,
                    "detail_code": row.detail_code,
                    "diagnostics": row.diagnostics,
                    "created_at": row.created_at,
                }
                for row in rows
            ]
