"""Repository for saved-profile capability probes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.db.llm_capability_models import LLMCapabilityTestRow


class LLMCapabilityTestRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(
        self,
        *,
        profile_id: UUID,
        credential_version_id: UUID,
        adapter_type: str,
        model: str,
        test_type: str,
        passed: bool,
        capability_matrix: Mapping[str, object],
        selected_fallbacks: Mapping[str, object],
        usage: Mapping[str, object],
        latency_ms: int,
        provider_request_id: str | None,
        error_code: str | None,
        detail_code: str | None,
        ttl: timedelta = timedelta(hours=24),
    ) -> LLMCapabilityTestRow:
        verified_at = datetime.now(UTC)
        row = LLMCapabilityTestRow(
            id=uuid4(),
            profile_id=profile_id,
            credential_version_id=credential_version_id,
            adapter_type=adapter_type,
            model=model,
            test_type=test_type,
            passed=passed,
            capability_matrix=dict(capability_matrix),
            selected_fallbacks=dict(selected_fallbacks),
            usage=dict(usage),
            latency_ms=latency_ms,
            provider_request_id=provider_request_id,
            error_code=error_code,
            detail_code=detail_code,
            verified_at=verified_at,
            expires_at=verified_at + ttl,
        )
        async with self._sessions() as session, session.begin():
            session.add(row)
        return row
