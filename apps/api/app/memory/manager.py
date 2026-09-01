"""Research Memory capture, retrieval, revalidation boundaries, and audit logs."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.domain.memory import (
    MemoryAccessView,
    MemoryItemView,
    MemoryRetrievalResult,
    MemoryStatus,
    MemoryType,
)
from app.domain.state import ResearchState
from app.infrastructure.db.memory_models import MemoryAccessLogRow, MemoryItemRow
from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.run_models import ResearchRunRow
from app.retrieval.models import MemorySearchDocumentRow
from app.retrieval.normalization import normalize_text, reciprocal_rank_fusion
from app.retrieval.projections import rebuild_memory

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*|[\u3400-\u9fff]")


class ResearchMemoryManager:
    """Persist useful state without turning provider chat history into system memory."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def capture_state(
        self,
        state: ResearchState,
        *,
        node_name: str,
    ) -> None:
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            run = await session.get(ResearchRunRow, state.run_id)
            if run is None:
                return
            await session.execute(
                update(MemoryItemRow)
                .where(
                    MemoryItemRow.origin_run_id == state.run_id,
                    MemoryItemRow.memory_type == MemoryType.WORKING.value,
                    MemoryItemRow.status == MemoryStatus.ACTIVE.value,
                )
                .values(status=MemoryStatus.SUPERSEDED.value, updated_at=now)
            )
            working_summary = _working_summary(state)
            await self._insert_if_absent(
                session,
                run=run,
                memory_type=MemoryType.WORKING,
                content_summary=working_summary,
                source_ref_ids=_state_refs(state),
                confidence=1.0,
                importance=1.0,
                fingerprint=_fingerprint(f"working:{state.state_version}:{working_summary}"),
                now=now,
            )
            episode = _episodic_summary(state, node_name=node_name)
            await self._insert_if_absent(
                session,
                run=run,
                memory_type=MemoryType.EPISODIC,
                content_summary=episode,
                source_ref_ids=[f"state:{state.state_version}", f"node:{node_name}"],
                confidence=1.0,
                importance=0.7,
                fingerprint=_fingerprint(f"episodic:{node_name}:{state.state_version}"),
                now=now,
            )
            evidence_rows = (
                await session.scalars(
                    select(ResearchEvidenceRow).where(
                        ResearchEvidenceRow.run_id == state.run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
            ).all()
            for evidence in evidence_rows:
                await self._insert_if_absent(
                    session,
                    run=run,
                    memory_type=MemoryType.SEMANTIC,
                    content_summary=evidence.claim,
                    source_ref_ids=[
                        f"evidence:{evidence.id}",
                        f"claim:{evidence.claim_id}",
                        f"snapshot:{evidence.snapshot_id}",
                    ],
                    confidence=max(0.0, min(1.0, evidence.evidence_score)),
                    importance=0.8,
                    fingerprint=_fingerprint(f"semantic:evidence:{evidence.id}"),
                    now=now,
                )
            # Keep the searchable projection transactionally aligned with memory facts.
            await session.flush()
            await rebuild_memory(session, owner_hash=run.owner_hash)

    async def retrieve_for_run(
        self,
        run_id: UUID,
        *,
        query: str | None = None,
        memory_types: tuple[MemoryType, ...] = (
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
        ),
        top_k: int = 8,
    ) -> MemoryRetrievalResult:
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            run = await session.get(ResearchRunRow, run_id)
            if run is None:
                raise ValueError("research run not found")
            normalized_query = " ".join((query or run.normalized_goal).split())
            normalized = normalize_text(normalized_query)
            query_text = normalized.cjk_lexemes or normalized.latin_text or normalized.raw
            ts_query = func.plainto_tsquery("simple", query_text)
            lexical_score = func.ts_rank_cd(
                MemorySearchDocumentRow.search_vector,
                ts_query,
            )
            fuzzy_score = func.similarity(
                MemorySearchDocumentRow.fuzzy_text,
                normalized.fuzzy_text,
            )
            filters = (
                MemoryItemRow.owner_hash == run.owner_hash,
                MemoryItemRow.status == MemoryStatus.ACTIVE.value,
                MemoryItemRow.memory_type.in_([item.value for item in memory_types]),
                MemorySearchDocumentRow.status == MemoryStatus.ACTIVE.value,
                MemorySearchDocumentRow.expires_at.is_(None)
                | (MemorySearchDocumentRow.expires_at > now),
            )
            lexical_rows = (
                await session.execute(
                    select(MemoryItemRow, lexical_score.label("rank"))
                    .join(
                        MemorySearchDocumentRow,
                        MemorySearchDocumentRow.memory_id == MemoryItemRow.id,
                    )
                    .where(
                        *filters,
                        MemorySearchDocumentRow.search_vector.op("@@")(ts_query),
                    )
                    .order_by(lexical_score.desc(), MemoryItemRow.id)
                    .limit(50)
                )
            ).all()
            fuzzy_rows = (
                await session.execute(
                    select(MemoryItemRow, fuzzy_score.label("rank"))
                    .join(
                        MemorySearchDocumentRow,
                        MemorySearchDocumentRow.memory_id == MemoryItemRow.id,
                    )
                    .where(*filters, fuzzy_score >= 0.05)
                    .order_by(fuzzy_score.desc(), MemoryItemRow.id)
                    .limit(50)
                )
            ).all()
            lexical_ids = tuple(str(row.id) for row, _rank in lexical_rows)
            fuzzy_ids = tuple(str(row.id) for row, _rank in fuzzy_rows)
            fused = reciprocal_rank_fusion(lexical_ids, fuzzy_ids)
            rows_by_id = {
                str(row.id): row
                for row, _rank in [*lexical_rows, *fuzzy_rows]
            }
            maximum_rrf = 2.0 / 61.0
            scored = [
                (
                    rows_by_id[memory_id],
                    min(
                        1.0,
                        0.65 * min(1.0, rrf_score / maximum_rrf)
                        + 0.20 * rows_by_id[memory_id].confidence
                        + 0.10 * rows_by_id[memory_id].importance
                        + (
                            0.05
                            if rows_by_id[memory_id].origin_run_id == run_id
                            else 0.0
                        ),
                    ),
                )
                for memory_id, rrf_score in fused.items()
            ]
            scored.sort(key=lambda item: (-item[1], str(item[0].id)))
            selected = scored[: max(1, min(top_k, 20))]
            access_id = uuid7()
            revalidation_count = sum(1 for row, _score in selected if row.origin_run_id != run_id)
            result = "hit" if selected else "miss"
            session.add(
                MemoryAccessLogRow(
                    id=access_id,
                    run_id=run_id,
                    query=normalized_query,
                    requested_types=[item.value for item in memory_types],
                    candidate_ids=[str(row.id) for row, _score in scored],
                    selected_ids=[str(row.id) for row, _score in selected],
                    score_snapshot={str(row.id): round(score, 6) for row, score in selected},
                    result=result,
                    revalidation_required_count=revalidation_count,
                    created_at=now,
                )
            )
            for row, _score in selected:
                row.last_accessed_at = now
                row.access_count += 1
                row.updated_at = now
            views = tuple(
                self._item_view(row, current_run_id=run_id)
                for row, _score in selected
            )
            return MemoryRetrievalResult(access_id=access_id, items=views, result=result)

    async def apply_lifecycle(self, *, now: datetime | None = None) -> dict[str, int]:
        """Expire stale memory and forget low-value unused items."""
        current = now or datetime.now(UTC)
        decay_cutoff = current - timedelta(days=30)
        async with self._sessions() as session, session.begin():
            stale_result = await session.execute(
                update(MemoryItemRow)
                .where(
                    MemoryItemRow.status == MemoryStatus.ACTIVE.value,
                    MemoryItemRow.expires_at.is_not(None),
                    MemoryItemRow.expires_at <= current,
                )
                .values(status=MemoryStatus.STALE.value, updated_at=current)
            )
            forgotten_result = await session.execute(
                update(MemoryItemRow)
                .where(
                    MemoryItemRow.status == MemoryStatus.ACTIVE.value,
                    MemoryItemRow.memory_type != MemoryType.WORKING.value,
                    MemoryItemRow.importance < 0.35,
                    MemoryItemRow.access_count == 0,
                    MemoryItemRow.updated_at < decay_cutoff,
                )
                .values(status=MemoryStatus.FORGOTTEN.value, updated_at=current)
            )
            return {
                "stale": _affected_rows(stale_result),
                "forgotten": _affected_rows(forgotten_result),
            }

    async def list_items(self, owner_hash: str, run_id: UUID) -> list[MemoryItemView]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ValueError("research run not found")
            rows = (
                await session.scalars(
                    select(MemoryItemRow)
                    .where(MemoryItemRow.origin_run_id == run_id)
                    .order_by(MemoryItemRow.created_at, MemoryItemRow.id)
                )
            ).all()
            return [self._item_view(row, current_run_id=run_id) for row in rows]

    async def list_accesses(self, owner_hash: str, run_id: UUID) -> list[MemoryAccessView]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ValueError("research run not found")
            rows = (
                await session.scalars(
                    select(MemoryAccessLogRow)
                    .where(MemoryAccessLogRow.run_id == run_id)
                    .order_by(MemoryAccessLogRow.created_at, MemoryAccessLogRow.id)
                )
            ).all()
            return [
                MemoryAccessView(
                    access_id=row.id,
                    run_id=row.run_id,
                    query=row.query,
                    requested_types=tuple(MemoryType(item) for item in row.requested_types),
                    candidate_ids=tuple(UUID(item) for item in row.candidate_ids),
                    selected_ids=tuple(UUID(item) for item in row.selected_ids),
                    result=row.result,
                    revalidation_required_count=row.revalidation_required_count,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    @staticmethod
    async def _insert_if_absent(
        session: AsyncSession,
        *,
        run: ResearchRunRow,
        memory_type: MemoryType,
        content_summary: str,
        source_ref_ids: list[str],
        confidence: float,
        importance: float,
        fingerprint: str,
        now: datetime,
    ) -> None:
        existing = await session.scalar(
            select(MemoryItemRow.id).where(
                MemoryItemRow.origin_run_id == run.id,
                MemoryItemRow.memory_type == memory_type.value,
                MemoryItemRow.fingerprint == fingerprint,
            )
        )
        if existing is not None:
            return
        session.add(
            MemoryItemRow(
                id=uuid7(),
                origin_run_id=run.id,
                owner_hash=run.owner_hash,
                scope_type="run",
                scope_id=str(run.id),
                memory_type=memory_type.value,
                content_summary=content_summary,
                source_ref_ids=source_ref_ids,
                keywords=sorted(_terms(content_summary)),
                confidence=confidence,
                importance=importance,
                fingerprint=fingerprint,
                status=MemoryStatus.ACTIVE.value,
                access_count=0,
                utility_count=0,
                expires_at=_expires_at(memory_type, now),
                created_at=now,
                updated_at=now,
            )
        )

    @staticmethod
    def _item_view(row: MemoryItemRow, *, current_run_id: UUID) -> MemoryItemView:
        return MemoryItemView(
            memory_id=row.id,
            origin_run_id=row.origin_run_id,
            memory_type=MemoryType(row.memory_type),
            content_summary=row.content_summary,
            source_ref_ids=tuple(row.source_ref_ids),
            keywords=tuple(row.keywords),
            confidence=row.confidence,
            importance=row.importance,
            status=MemoryStatus(row.status),
            revalidation_required=row.origin_run_id != current_run_id,
            access_count=row.access_count,
            created_at=row.created_at,
            updated_at=row.updated_at,
            expires_at=row.expires_at,
        )


def _expires_at(memory_type: MemoryType, now: datetime) -> datetime | None:
    windows = {
        MemoryType.WORKING: timedelta(days=7),
        MemoryType.EPISODIC: timedelta(days=30),
        MemoryType.SEMANTIC: timedelta(days=90),
    }
    return now + windows[memory_type]

def _working_summary(state: ResearchState) -> str:
    next_action = state.next_action.action_type.value if state.next_action else "none"
    open_gaps = sum(g.status.value in {"open", "resolving"} for g in state.gaps)
    return (
        f"phase={state.phase.value}; iteration={state.iteration}; "
        f"known={len(state.known)}; open_gaps={open_gaps}; "
        f"next_action={next_action}; coverage={state.quality.coverage:.3f}"
    )


def _episodic_summary(state: ResearchState, *, node_name: str) -> str:
    decision = state.next_action.public_decision_summary if state.next_action else "no next action"
    return (
        f"节点 {node_name} 完成于状态版本 {state.state_version}; "
        f"覆盖度 {state.quality.coverage:.1%}, 信息增益 {state.quality.information_gain:.1%}; "
        f"研究决策: {decision}"
    )


def _state_refs(state: ResearchState) -> list[str]:
    refs = [f"state:{state.state_version}"]
    refs.extend(f"claim:{item.claim_id}" for item in state.known)
    refs.extend(f"gap:{item.gap_id}" for item in state.gaps)
    return refs


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _affected_rows(result: object) -> int:
    rowcount = getattr(result, "rowcount", 0)
    return rowcount if isinstance(rowcount, int) else 0


def _terms(value: str) -> set[str]:
    raw = [match.group(0).lower() for match in _WORD_RE.finditer(value)]
    cjk = [item for item in raw if len(item) == 1 and "\u3400" <= item <= "\u9fff"]
    bigrams = {"".join(cjk[index : index + 2]) for index in range(max(0, len(cjk) - 1))}
    return set(raw) | bigrams


def _memory_score(query_terms: set[str], row: MemoryItemRow) -> float:
    memory_terms = set(row.keywords) or _terms(row.content_summary)
    overlap = len(query_terms & memory_terms) / max(1, len(query_terms))
    type_bonus = 0.08 if row.memory_type == MemoryType.EPISODIC.value else 0.05
    return 0.65 * overlap + 0.20 * row.confidence + 0.10 * row.importance + type_bonus
