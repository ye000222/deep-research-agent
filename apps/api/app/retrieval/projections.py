"""Build and refresh PostgreSQL retrieval projections."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.memory_models import MemoryItemRow
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.retrieval.models import (
    EvidenceSearchDocumentRow,
    MemorySearchDocumentRow,
    RetrievalConfigVersionRow,
)
from app.retrieval.normalization import normalize_text

_CONFIG_ID = UUID("01a05800-0000-7000-8000-000000000001")


async def _config(session: AsyncSession) -> RetrievalConfigVersionRow:
    row = await session.get(RetrievalConfigVersionRow, _CONFIG_ID)
    if row is None:
        raise RuntimeError("retrieval config lexical-v1 is not installed")
    return row


async def rebuild_evidence(session: AsyncSession, *, run_id: UUID | None = None) -> int:
    config = await _config(session)
    query = (
        select(ResearchEvidenceRow, ResearchSourceRow)
        .join(ResearchSourceRow, ResearchEvidenceRow.source_id == ResearchSourceRow.id)
        .where(ResearchEvidenceRow.accepted.is_(True))
    )
    if run_id is not None:
        query = query.where(ResearchEvidenceRow.run_id == run_id)
    rows = (await session.execute(query)).tuples().all()
    if run_id is not None:
        await session.execute(
            delete(EvidenceSearchDocumentRow).where(EvidenceSearchDocumentRow.run_id == run_id)
        )
    elif rows:
        await session.execute(delete(EvidenceSearchDocumentRow))
    now = datetime.now(UTC)
    for evidence, source in rows:
        normalized = normalize_text(f"{evidence.claim} {evidence.exact_quote} {source.title}")
        values = {
            "evidence_id": evidence.id,
            "run_id": evidence.run_id,
            "question_id": evidence.question_id,
            "dimension_key": None,
            "retrieval_config_version_id": config.id,
            "raw_search_text": normalized.raw,
            "latin_text": normalized.latin_text,
            "cjk_lexemes": normalized.cjk_lexemes,
            "fuzzy_text": normalized.fuzzy_text,
            "search_vector": func.to_tsvector("simple", ""),
            "claim_status": "supported" if evidence.accepted else "candidate",
            "source_owner_key": source.source_owner_key,
            "source_type": source.source_type,
            "evidence_score": evidence.evidence_score,
            "published_at": None,
            "content_hash": evidence.evidence_hash,
            "updated_at": now,
        }
        await session.execute(
            insert(EvidenceSearchDocumentRow)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[EvidenceSearchDocumentRow.evidence_id],
                set_={key: value for key, value in values.items() if key != "evidence_id"},
            )
        )
    if rows:
        await session.execute(
            update(EvidenceSearchDocumentRow)
            .values(
                search_vector=(
                    func.to_tsvector("english", EvidenceSearchDocumentRow.latin_text).op("||")(
                        func.to_tsvector("simple", EvidenceSearchDocumentRow.cjk_lexemes)
                    )
                )
            )
            .where(
                EvidenceSearchDocumentRow.retrieval_config_version_id == config.id,
            )
        )
    return len(rows)


async def rebuild_memory(session: AsyncSession, *, owner_hash: str | None = None) -> int:
    config = await _config(session)
    query = select(MemoryItemRow)
    if owner_hash is not None:
        query = query.where(MemoryItemRow.owner_hash == owner_hash)
    rows = (await session.scalars(query)).all()
    if owner_hash is not None:
        await session.execute(
            delete(MemorySearchDocumentRow).where(
                MemorySearchDocumentRow.scope_id.in_(
                    select(MemoryItemRow.scope_id).where(MemoryItemRow.owner_hash == owner_hash)
                )
            )
        )
    elif rows:
        await session.execute(delete(MemorySearchDocumentRow))
    now = datetime.now(UTC)
    for memory in rows:
        normalized = normalize_text(memory.content_summary)
        values = {
            "memory_id": memory.id,
            "scope_type": memory.scope_type,
            "scope_id": memory.scope_id,
            "memory_type": memory.memory_type,
            "status": memory.status,
            "retrieval_config_version_id": config.id,
            "raw_search_text": normalized.raw,
            "latin_text": normalized.latin_text,
            "cjk_lexemes": normalized.cjk_lexemes,
            "fuzzy_text": normalized.fuzzy_text,
            "search_vector": func.to_tsvector("simple", ""),
            "confidence": memory.confidence,
            "importance": memory.importance,
            "expires_at": memory.expires_at,
            "updated_at": now,
        }
        await session.execute(
            insert(MemorySearchDocumentRow)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[MemorySearchDocumentRow.memory_id],
                set_={key: value for key, value in values.items() if key != "memory_id"},
            )
        )
    if rows:
        await session.execute(
            update(MemorySearchDocumentRow)
            .values(
                search_vector=(
                    func.to_tsvector("english", MemorySearchDocumentRow.latin_text).op("||")(
                        func.to_tsvector("simple", MemorySearchDocumentRow.cjk_lexemes)
                    )
                )
            )
            .where(
                MemorySearchDocumentRow.retrieval_config_version_id == config.id,
            )
        )
    return len(rows)
