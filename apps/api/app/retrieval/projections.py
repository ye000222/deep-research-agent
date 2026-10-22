"""Build and refresh PostgreSQL retrieval projections."""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.stage_timing import stage_timer
from app.infrastructure.db.memory_models import MemoryItemRow
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.retrieval.models import (
    EvidenceSearchDocumentRow,
    MemorySearchDocumentRow,
    RetrievalConfigVersionRow,
)
from app.retrieval.normalization import normalize_text

_CONFIG_ID = UUID("01a05800-0000-7000-8000-000000000001")
_LOG = logging.getLogger(__name__)
_BATCH_SIZE = 500


@dataclass(frozen=True)
class MemoryProjectionChanges:
    requested: int = 0
    loaded: int = 0
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


class MemoryProjectionVersionMismatch(RuntimeError):
    """An explicit maintenance rebuild is required; never scan an owner in capture."""


async def lock_memory_owner(session: AsyncSession, owner_hash: str) -> None:
    """Acquire before changing facts, in the same transaction as their projection."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:owner_hash, 0))"),
        {"owner_hash": owner_hash},
    )


def _owned_memory_ids(owner_hash: str, memory_ids: Collection[UUID] | None = None):
    query = select(MemoryItemRow.id).where(MemoryItemRow.owner_hash == owner_hash)
    if memory_ids is not None:
        query = query.where(MemoryItemRow.id.in_(memory_ids))
    return query


def _searchable(memory: MemoryItemRow, now: datetime) -> bool:
    expiry = memory.expires_at
    if expiry is not None and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return memory.status == "active" and (expiry is None or expiry > now)


def _memory_values(memory: MemoryItemRow, config: RetrievalConfigVersionRow) -> dict:
    normalized = normalize_text(memory.content_summary)
    return {
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
        "confidence": memory.confidence,
        "importance": memory.importance,
        "expires_at": memory.expires_at,
    }


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, datetime) and isinstance(right, datetime):
        left = left.replace(tzinfo=UTC) if left.tzinfo is None else left.astimezone(UTC)
        right = right.replace(tzinfo=UTC) if right.tzinfo is None else right.astimezone(UTC)
    return left == right


async def _upsert_memory_documents(
    session: AsyncSession,
    *,
    owner_hash: str,
    values: list[dict],
    run_id: UUID | None,
) -> None:
    for start in range(0, len(values), _BATCH_SIZE):
        batch = values[start : start + _BATCH_SIZE]
        batch_ids = [value["memory_id"] for value in batch]
        statement = insert(MemorySearchDocumentRow).values(
            [
                {
                    **value,
                    "search_vector": func.to_tsvector("simple", ""),
                    "updated_at": datetime.now(UTC),
                }
                for value in batch
            ]
        )
        with stage_timer("memory.projection.upsert", run_id=run_id):
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[MemorySearchDocumentRow.memory_id],
                    set_={
                        key: getattr(statement.excluded, key)
                        for key in (*batch[0].keys(), "search_vector", "updated_at")
                        if key != "memory_id"
                    },
                    where=MemorySearchDocumentRow.memory_id.in_(
                        _owned_memory_ids(owner_hash, batch_ids)
                    ),
                )
            )
        # FTS is computed by PostgreSQL, only for this batch and this owner.
        with stage_timer("memory.projection.fts_update", run_id=run_id):
            await session.execute(
                update(MemorySearchDocumentRow)
                .where(
                    MemorySearchDocumentRow.memory_id.in_(batch_ids),
                    MemorySearchDocumentRow.memory_id.in_(_owned_memory_ids(owner_hash, batch_ids)),
                )
                .values(
                    search_vector=(
                        func.to_tsvector("english", MemorySearchDocumentRow.latin_text).op("||")(
                            func.to_tsvector("simple", MemorySearchDocumentRow.cjk_lexemes)
                        )
                    )
                )
                .execution_options(synchronize_session=False)
            )


async def sync_memory_projection(
    session: AsyncSession,
    *,
    owner_hash: str,
    changed_ids: Collection[UUID],
    run_id: UUID | None = None,
) -> MemoryProjectionChanges:
    """Sync explicit dirty IDs in the caller's fact transaction, without committing.

    Physical deletion uses the existing ON DELETE CASCADE foreign key. Inactive
    or expired facts lose their projection here. Foreign-owner IDs are ignored.
    Version mismatch on a touched document fails closed for maintenance rebuild.
    Callers acquire lock_memory_owner *before* fact mutations to serialize with
    maintenance and lifecycle updates; the reentrant lock here also protects
    direct use of this function.
    """
    with stage_timer("memory.projection.total", run_id=run_id):
        ids = tuple(set(changed_ids))
        if not ids:
            changes = MemoryProjectionChanges()
        else:
            with stage_timer("memory.projection.owner_lock", run_id=run_id):
                await lock_memory_owner(session, owner_hash)
            config = await _config(session)
            with stage_timer("memory.projection.load_memories", run_id=run_id):
                rows = (
                    await session.scalars(
                        select(MemoryItemRow).where(
                            MemoryItemRow.owner_hash == owner_hash,
                            MemoryItemRow.id.in_(ids),
                        )
                    )
                ).all()
                existing = {
                    row.memory_id: row
                    for row in (
                        await session.scalars(
                            select(MemorySearchDocumentRow).where(
                                MemorySearchDocumentRow.memory_id.in_(ids),
                                MemorySearchDocumentRow.memory_id.in_(
                                    _owned_memory_ids(owner_hash, ids)
                                ),
                            )
                        )
                    ).all()
                }
            with stage_timer("memory.projection.determine_changes", run_id=run_id):
                if any(row.retrieval_config_version_id != config.id for row in existing.values()):
                    raise MemoryProjectionVersionMismatch(
                        "memory projection needs maintenance rebuild"
                    )
                now = datetime.now(UTC)
                values, stale = [], []
                inserted = updated = unchanged = 0
                for memory in rows:
                    previous = existing.get(memory.id)
                    if not _searchable(memory, now):
                        if previous is not None:
                            stale.append(memory.id)
                        else:
                            unchanged += 1
                        continue
                    value = _memory_values(memory, config)
                    if previous is not None and all(
                        _same_value(getattr(previous, key), item) for key, item in value.items()
                    ):
                        unchanged += 1
                        continue
                    values.append(value)
                    inserted += int(previous is None)
                    updated += int(previous is not None)
            if stale:
                with stage_timer("memory.projection.delete_stale", run_id=run_id):
                    await session.execute(
                        delete(MemorySearchDocumentRow).where(
                            MemorySearchDocumentRow.memory_id.in_(stale),
                            MemorySearchDocumentRow.memory_id.in_(
                                _owned_memory_ids(owner_hash, stale)
                            ),
                        )
                    )
            await _upsert_memory_documents(
                session,
                owner_hash=owner_hash,
                values=values,
                run_id=run_id,
            )
            changes = MemoryProjectionChanges(
                requested=len(ids),
                loaded=len(rows),
                inserted=inserted,
                updated=updated,
                deleted=len(stale),
                unchanged=unchanged,
            )
        _LOG.info(
            "memory_projection_changes run_id=%s transaction=pending_commit counts=%s",
            run_id,
            {**asdict(changes), "changed": changes.inserted + changes.updated + changes.deleted},
        )
        return changes


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
    """Explicit maintenance/config-upgrade rebuild; never called by capture.

    Global maintenance (owner_hash=None) requires a quiescent maintenance window.
    Inactive/expired facts are retained as facts but need no searchable document.
    """
    with stage_timer("memory.rebuild.total"):
        config = await _config(session)
        if owner_hash is not None:
            with stage_timer("memory.rebuild.owner_lock"):
                await lock_memory_owner(session, owner_hash)
        with stage_timer("memory.rebuild.load_memories"):
            query = select(MemoryItemRow)
            if owner_hash is not None:
                query = query.where(MemoryItemRow.owner_hash == owner_hash)
            rows = (await session.scalars(query)).all()
        with stage_timer("memory.rebuild.delete_stale"):
            statement = delete(MemorySearchDocumentRow)
            if owner_hash is not None:
                statement = statement.where(
                    MemorySearchDocumentRow.memory_id.in_(_owned_memory_ids(owner_hash))
                )
            await session.execute(statement)
        with stage_timer("memory.rebuild.determine_changes"):
            now = datetime.now(UTC)
            owners: dict[str, list[dict]] = {}
            for memory in rows:
                if _searchable(memory, now):
                    owners.setdefault(memory.owner_hash, []).append(_memory_values(memory, config))
        for owner, values in owners.items():
            await _upsert_memory_documents(session, owner_hash=owner, values=values, run_id=None)
        indexed = sum(len(values) for values in owners.values())
        _LOG.info("memory_rebuild_counts loaded=%d indexed=%d", len(rows), indexed)
        return indexed
