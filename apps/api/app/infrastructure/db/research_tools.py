"""Transactional store for one bounded Web research iteration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypedDict, cast
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.evaluation import EvaluationScope, EvaluationSnapshot, EvaluationVerdict
from app.domain.evidence_graph import (
    EvidenceGraphClaimEdgeView,
    EvidenceGraphClaimNode,
    EvidenceGraphConflictView,
    EvidenceGraphEvidenceRef,
    EvidenceGraphView,
    build_evidence_chunk,
    claim_fingerprint,
    derive_claim_status,
)
from app.domain.identifiers import uuid7
from app.domain.providers import TokenUsage
from app.domain.research_management import ResearchFactCounts, calculate_information_gain
from app.domain.research_runs import RunPhase, RunStatus
from app.domain.research_tools import EvidenceView, ReadPage, ScoredEvidence, SearchResult
from app.infrastructure.db.evaluation_models import EvaluationSnapshotRow
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimEdgeRow,
    ResearchClaimRow,
    ResearchConflictRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.evidence_graph_relations import (
    RelationRefreshStats,
    refresh_question_relations,
)
from app.infrastructure.db.research_models import (
    ResearchEvidenceRow,
    ResearchGapRow,
    ResearchSourceRow,
    ResearchToolCallRow,
    SearchQueryRow,
    SearchResultRow,
)
from app.infrastructure.db.research_runs import ResearchRunNotFoundError
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow


class ResearchLeaseLostError(RuntimeError):
    pass


_MAX_GAP_ATTEMPTS = 2
_LOW_INFORMATION_GAIN_THRESHOLD = 0.10
_LOW_INFORMATION_GAIN_STREAK_TO_STOP = 2


@dataclass(frozen=True, slots=True)
class ResearchTarget:
    plan_version: int
    question_id: str
    question: str
    query: str
    gap_id: UUID
    tool_call_id: UUID
    source_id_seed: UUID


@dataclass(frozen=True, slots=True)
class IterationEvaluation:
    continue_research: bool
    decision: str
    stop_reason: str | None
    question_status: str
    coverage: float = 0.0
    information_gain: float = 0.0
    low_information_gain_streak: int = 0


class _CoverageMapEntry(TypedDict):
    dimension_key: str
    question: str
    priority: int
    coverage: float
    accepted_evidence: int
    independent_sources: int
    acceptance_criteria: list[str]
    missing_reasons: list[str]


class ResearchToolRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def prepare_target(self, run_id: UUID, *, worker_task_id: str) -> ResearchTarget | None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            if _budget_exhausted(run):
                await self._enter_writing(
                    session,
                    run,
                    reason="research_budget_exhausted",
                    summary="研究预算已耗尽; 使用现有有效证据生成带限制报告。",
                )
                return None
            candidates = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                        ResearchPlanItemRow.status.in_(("pending", "active", "blocked")),
                    )
                    .order_by(ResearchPlanItemRow.priority, ResearchPlanItemRow.question_id)
                    .with_for_update()
                )
            ).all()
            question: ResearchPlanItemRow | None = None
            for candidate in candidates:
                if candidate.status != "blocked":
                    question = candidate
                    break
                candidate_gap = await session.scalar(
                    select(ResearchGapRow).where(
                        ResearchGapRow.run_id == run_id,
                        ResearchGapRow.plan_version == run.plan_version,
                        ResearchGapRow.question_id == candidate.question_id,
                    )
                )
                if (
                    candidate_gap is not None
                    and candidate_gap.resolution_attempts < _MAX_GAP_ATTEMPTS
                ):
                    question = candidate
                    break
            if question is None:
                await self._enter_writing(
                    session,
                    run,
                    reason="quality_met",
                    summary="研究问题已处理完毕; 自动进入证据驱动报告写作。",
                )
                return None
            now = datetime.now(UTC)
            gap = await session.scalar(
                select(ResearchGapRow).where(
                    ResearchGapRow.run_id == run_id,
                    ResearchGapRow.plan_version == run.plan_version,
                    ResearchGapRow.question_id == question.question_id,
                )
            )
            created_gap = gap is None
            if gap is None:
                gap = ResearchGapRow(
                    id=uuid7(),
                    run_id=run_id,
                    plan_version=run.plan_version,
                    question_id=question.question_id,
                    gap_type="missing",
                    description=f"当前缺少对研究问题 {question.question_id} 的可验证证据。",
                    acceptance_criteria="至少获得一条可定位到原网页逐字引文的有效证据。",
                    severity=1.0,
                    status="open",
                    resolution_attempts=0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(gap)
                await session.flush()
            gap.resolution_attempts += 1
            gap.updated_at = now
            question.status = "active"
            attempt_index = max(gap.resolution_attempts - 1, 0)
            query_candidates = [*question.search_hints, question.question]
            query = (
                query_candidates[attempt_index]
                if attempt_index < len(query_candidates)
                else question.question
            )
            normalized_query = " ".join(query.split()).lower()
            duplicate_key = hashlib.sha256(
                f"{run.plan_version}:{question.question_id}:{normalized_query}".encode()
            ).hexdigest()
            tool_call = await session.scalar(
                select(ResearchToolCallRow).where(
                    ResearchToolCallRow.run_id == run_id,
                    ResearchToolCallRow.tool_name == "web_search",
                    ResearchToolCallRow.duplicate_key == duplicate_key,
                )
            )
            if tool_call is None:
                tool_call = ResearchToolCallRow(
                    id=uuid7(),
                    run_id=run_id,
                    question_id=question.question_id,
                    gap_id=gap.id,
                    action_id=uuid7(),
                    tool_name="web_search",
                    duplicate_key=duplicate_key,
                    status="running",
                    arguments={"query": query, "limit": 10},
                    result_refs={},
                    started_at=now,
                )
                session.add(tool_call)
                await session.flush()
            if created_gap:
                await self._append_event(
                    session,
                    run,
                    event_type="gap.opened",
                    public_summary=f"识别到问题 {question.question_id} 的证据缺口。",
                    refs={"gap_id": str(gap.id), "question_id": question.question_id},
                )
            await self._append_event(
                session,
                run,
                event_type="action.selected",
                public_summary="Agent 根据当前缺口选择 Web Search。",
                refs={
                    "action_id": str(tool_call.action_id),
                    "gap_id": str(gap.id),
                    "question_id": question.question_id,
                },
            )
            await self._append_event(
                session,
                run,
                event_type="tool.called",
                public_summary=f"正在搜索: {query[:300]}",
                refs={
                    "tool_call_id": str(tool_call.id),
                    "tool_name": "web_search",
                    "question_id": question.question_id,
                },
            )
            return ResearchTarget(
                plan_version=run.plan_version,
                question_id=question.question_id,
                question=question.question,
                query=query,
                gap_id=gap.id,
                tool_call_id=tool_call.id,
                source_id_seed=uuid7(),
            )

    async def record_search_results(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        results: list[SearchResult],
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            tool_call = await session.get(ResearchToolCallRow, target.tool_call_id)
            if tool_call is None:
                raise ResearchLeaseLostError("tool call disappeared")
            query_hash = hashlib.sha256(" ".join(target.query.split()).lower().encode()).hexdigest()
            query_row = await session.scalar(
                select(SearchQueryRow).where(
                    SearchQueryRow.run_id == run_id,
                    SearchQueryRow.normalized_hash == query_hash,
                )
            )
            if query_row is None:
                now = datetime.now(UTC)
                query_row = SearchQueryRow(
                    id=uuid7(),
                    run_id=run_id,
                    question_id=target.question_id,
                    tool_call_id=tool_call.id,
                    query=target.query,
                    normalized_hash=query_hash,
                    provider="searxng",
                    status="succeeded",
                    result_count=len(results),
                    created_at=now,
                )
                session.add(query_row)
                await session.flush()
                for result in results:
                    session.add(
                        SearchResultRow(
                            id=uuid7(),
                            search_query_id=query_row.id,
                            rank=result.rank,
                            title=result.title,
                            url=result.url,
                            snippet=result.snippet,
                            published_at=result.published_at,
                        )
                    )
            tool_call.status = "succeeded"
            tool_call.result_refs = {"search_query_id": str(query_row.id), "count": len(results)}
            tool_call.finished_at = datetime.now(UTC)
            usage = dict(run.usage_snapshot)
            usage["searches"] = int(usage.get("searches", 0)) + 1
            run.usage_snapshot = usage
            await self._append_event(
                session,
                run,
                event_type="search.completed",
                public_summary=f"搜索完成, 获得 {len(results)} 个候选结果。",
                refs={
                    "search_query_id": str(query_row.id),
                    "question_id": target.question_id,
                },
                metrics={"result_count": len(results)},
            )

    async def record_tool_failure(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        error_code: str,
        retryable: bool,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            tool_call = await session.get(ResearchToolCallRow, target.tool_call_id)
            if tool_call is not None:
                tool_call.status = "failed"
                tool_call.error_code = error_code[:100]
                tool_call.retryable = retryable
                tool_call.finished_at = datetime.now(UTC)
            await self._append_event(
                session,
                run,
                event_type="tool.failed",
                public_summary="Web Search 执行失败, 公开轨迹仅记录安全错误码。",
                refs={"question_id": target.question_id, "error_code": error_code[:100]},
            )

    async def record_extraction_started(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        source_id: UUID,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            await self._append_event(
                session,
                run,
                event_type="evidence.extraction_started",
                public_summary="正在调用模型从网页正文提取可验证证据。",
                refs={
                    "question_id": target.question_id,
                    "source_id": str(source_id),
                },
            )

    async def record_page(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        source_id: UUID,
        page: ReadPage,
        artifact_uri: str,
        evidence: list[ScoredEvidence],
        usage: TokenUsage,
        context_manifest: dict[str, int | bool],
    ) -> tuple[int, int]:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            url_hash = hashlib.sha256(page.final_url.encode()).hexdigest()
            source = await session.scalar(
                select(ResearchSourceRow).where(
                    ResearchSourceRow.run_id == run_id,
                    ResearchSourceRow.url_hash == url_hash,
                )
            )
            reliability = evidence[0].source_reliability if evidence else 0.72
            if source is None:
                source = ResearchSourceRow(
                    id=source_id,
                    run_id=run_id,
                    canonical_url=page.final_url,
                    url_hash=url_hash,
                    domain=(urlsplit(page.final_url).hostname or "unknown")[:255],
                    source_owner_key=(urlsplit(page.final_url).hostname or "unknown")[:255],
                    title=page.title,
                    source_type="webpage",
                    reliability=reliability,
                    artifact_uri=artifact_uri,
                    content_hash=page.content_hash,
                    char_count=len(page.clean_text),
                    fetched_at=page.fetched_at,
                )
                session.add(source)
                await session.flush()
            snapshot = await session.scalar(
                select(ResearchSourceSnapshotRow).where(
                    ResearchSourceSnapshotRow.source_id == source.id,
                    ResearchSourceSnapshotRow.content_hash == page.content_hash,
                )
            )
            if snapshot is None:
                snapshot = ResearchSourceSnapshotRow(
                    id=uuid7(),
                    run_id=run_id,
                    source_id=source.id,
                    final_url=page.final_url,
                    fetched_at=page.fetched_at,
                    published_at=None,
                    content_hash=page.content_hash,
                    parser_version="web-reader-v1",
                    artifact_uri=artifact_uri,
                    char_count=len(page.clean_text),
                )
                session.add(snapshot)
                await session.flush()
            inserted = 0
            accepted = 0
            graph_claim_ids: set[UUID] = set()
            graph_chunk_ids: set[UUID] = set()
            now = datetime.now(UTC)
            for item in evidence:
                candidate = item.candidate
                evidence_hash = hashlib.sha256(
                    (
                        f"{target.question_id}\n{source.id}\n{candidate.claim}\n"
                        f"{candidate.exact_quote}\n{candidate.relation.value}"
                    ).encode()
                ).hexdigest()
                exists = await session.scalar(
                    select(ResearchEvidenceRow.id).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.evidence_hash == evidence_hash,
                    )
                )
                if exists is not None:
                    continue

                claim_hash = claim_fingerprint(candidate.claim)
                claim = await session.scalar(
                    select(ResearchClaimRow).where(
                        ResearchClaimRow.run_id == run_id,
                        ResearchClaimRow.question_id == target.question_id,
                        ResearchClaimRow.claim_hash == claim_hash,
                    )
                )
                chunk_window = build_evidence_chunk(page.clean_text, candidate.exact_quote)
                accepted_by_graph = item.accepted and chunk_window is not None
                rejection_reason = item.rejection_reason
                if item.accepted and chunk_window is None:
                    rejection_reason = "quote_not_located_for_graph"
                if claim is None:
                    claim = ResearchClaimRow(
                        id=uuid7(),
                        run_id=run_id,
                        plan_version=target.plan_version,
                        question_id=target.question_id,
                        dimension_key=target.question_id,
                        atomic_claim=candidate.claim,
                        claim_hash=claim_hash,
                        claim_type="factual",
                        importance=0.8 if accepted_by_graph else 0.5,
                        status=derive_claim_status(
                            has_accepted_evidence=accepted_by_graph,
                            has_refuting_evidence=(
                                accepted_by_graph and candidate.relation.value == "refutes"
                            ),
                            independent_source_count=1 if accepted_by_graph else 0,
                        ),
                        confidence=item.evidence_score if accepted_by_graph else 0.0,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(claim)
                    await session.flush()
                elif accepted_by_graph:
                    claim.importance = max(claim.importance, 0.8)
                    claim.confidence = max(claim.confidence, item.evidence_score)
                    claim.updated_at = now

                chunk: ResearchSourceChunkRow | None = None
                if chunk_window is not None:
                    chunk = await session.scalar(
                        select(ResearchSourceChunkRow).where(
                            ResearchSourceChunkRow.snapshot_id == snapshot.id,
                            ResearchSourceChunkRow.chunk_hash == chunk_window.chunk_hash,
                        )
                    )
                    if chunk is None:
                        chunk = ResearchSourceChunkRow(
                            id=uuid7(),
                            run_id=run_id,
                            snapshot_id=snapshot.id,
                            heading_path=None,
                            char_start=chunk_window.char_start,
                            char_end=chunk_window.char_end,
                            text=chunk_window.text,
                            token_count=chunk_window.token_count,
                            chunk_hash=chunk_window.chunk_hash,
                        )
                        session.add(chunk)
                        await session.flush()
                evidence_row = ResearchEvidenceRow(
                    id=uuid7(),
                    run_id=run_id,
                    plan_version=target.plan_version,
                    question_id=target.question_id,
                    source_id=source.id,
                    claim_id=claim.id,
                    snapshot_id=snapshot.id,
                    chunk_id=chunk.id if chunk is not None else None,
                    claim=candidate.claim,
                    exact_quote=candidate.exact_quote,
                    relation=candidate.relation.value,
                    relevance=candidate.relevance,
                    confidence=candidate.confidence,
                    source_reliability=item.source_reliability,
                    evidence_score=item.evidence_score,
                    accepted=accepted_by_graph,
                    rejection_reason=rejection_reason,
                    evidence_hash=evidence_hash,
                    created_at=now,
                )
                session.add(evidence_row)
                await session.flush()
                independent_source_count = await session.scalar(
                    select(func.count(distinct(ResearchSourceRow.source_owner_key)))
                    .select_from(ResearchEvidenceRow)
                    .join(
                        ResearchSourceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.claim_id == claim.id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                refuting_evidence_count = await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.claim_id == claim.id,
                        ResearchEvidenceRow.accepted.is_(True),
                        ResearchEvidenceRow.relation == "refutes",
                    )
                )
                claim.status = derive_claim_status(
                    has_accepted_evidence=int(independent_source_count or 0) > 0,
                    has_refuting_evidence=int(refuting_evidence_count or 0) > 0,
                    independent_source_count=int(independent_source_count or 0),
                )
                graph_claim_ids.add(claim.id)
                if chunk is not None:
                    graph_chunk_ids.add(chunk.id)
                inserted += 1
                accepted += int(accepted_by_graph)
            relation_stats = RelationRefreshStats()
            if inserted:
                relation_stats = await refresh_question_relations(
                    session,
                    run_id=run_id,
                    question_id=target.question_id,
                )
            usage_snapshot = dict(run.usage_snapshot)
            usage_snapshot["pages"] = int(usage_snapshot.get("pages", 0)) + 1
            usage_snapshot["evidence_input_tokens"] = (
                int(usage_snapshot.get("evidence_input_tokens", 0)) + usage.input_tokens
            )
            usage_snapshot["evidence_output_tokens"] = (
                int(usage_snapshot.get("evidence_output_tokens", 0)) + usage.output_tokens
            )
            usage_snapshot["evidence_total_tokens"] = (
                int(usage_snapshot.get("evidence_total_tokens", 0)) + usage.total_tokens
            )
            run.usage_snapshot = usage_snapshot
            await self._append_event(
                session,
                run,
                event_type="source.read",
                public_summary=f"已读取来源: {page.title[:300]}",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={"clean_chars": len(page.clean_text), "truncated": page.truncated},
            )
            await self._append_event(
                session,
                run,
                event_type="context.assembled",
                public_summary="Context Manager 已选择当前问题与必要网页片段。",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics=cast(dict[str, object], context_manifest),
            )
            await self._append_event(
                session,
                run,
                event_type="evidence.graph_updated",
                public_summary=(
                    "Evidence Graph synchronized atomic claims, source snapshot, and quote chunks."
                ),
                refs={
                    "source_id": str(source.id),
                    "snapshot_id": str(snapshot.id),
                    "question_id": target.question_id,
                },
                metrics={
                    "claim_count": len(graph_claim_ids),
                    "chunk_count": len(graph_chunk_ids),
                    "bound_evidence_count": inserted,
                },
            )
            if any(
                (
                    relation_stats.detected_edges,
                    relation_stats.deleted_edges,
                    relation_stats.detected_conflicts,
                    relation_stats.dismissed_conflicts,
                    relation_stats.reopened_conflicts,
                )
            ):
                await self._append_event(
                    session,
                    run,
                    event_type="evidence.relationships_updated",
                    public_summary=(
                        "Evidence Graph updated Claim relations and disclosed "
                        "independent-source conflicts."
                    ),
                    refs={"question_id": target.question_id},
                    metrics={
                        "examined_pairs": relation_stats.examined_pairs,
                        "detected_edges": relation_stats.detected_edges,
                        "created_edges": relation_stats.created_edges,
                        "deleted_edges": relation_stats.deleted_edges,
                        "detected_conflicts": relation_stats.detected_conflicts,
                        "created_conflicts": relation_stats.created_conflicts,
                        "dismissed_conflicts": relation_stats.dismissed_conflicts,
                        "reopened_conflicts": relation_stats.reopened_conflicts,
                    },
                )
            await self._append_event(
                session,
                run,
                event_type="evidence.extracted",
                public_summary=f"提取 {inserted} 条候选证据, 其中 {accepted} 条通过验证。",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={"candidate_count": inserted, "accepted_count": accepted},
            )
            return inserted, accepted

    async def record_extraction_failure(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        source_id: UUID,
        page: ReadPage,
        artifact_uri: str,
        error_code: str,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            url_hash = hashlib.sha256(page.final_url.encode()).hexdigest()
            source = await session.scalar(
                select(ResearchSourceRow).where(
                    ResearchSourceRow.run_id == run_id,
                    ResearchSourceRow.url_hash == url_hash,
                )
            )
            if source is None:
                source = ResearchSourceRow(
                    id=source_id,
                    run_id=run_id,
                    canonical_url=page.final_url,
                    url_hash=url_hash,
                    domain=(urlsplit(page.final_url).hostname or "unknown")[:255],
                    source_owner_key=(urlsplit(page.final_url).hostname or "unknown")[:255],
                    title=page.title,
                    source_type="webpage",
                    reliability=0.72,
                    artifact_uri=artifact_uri,
                    content_hash=page.content_hash,
                    char_count=len(page.clean_text),
                    fetched_at=page.fetched_at,
                )
                session.add(source)
                await session.flush()
            snapshot = await session.scalar(
                select(ResearchSourceSnapshotRow).where(
                    ResearchSourceSnapshotRow.source_id == source.id,
                    ResearchSourceSnapshotRow.content_hash == page.content_hash,
                )
            )
            if snapshot is None:
                snapshot = ResearchSourceSnapshotRow(
                    id=uuid7(),
                    run_id=run_id,
                    source_id=source.id,
                    final_url=page.final_url,
                    fetched_at=page.fetched_at,
                    published_at=None,
                    content_hash=page.content_hash,
                    parser_version="web-reader-v1",
                    artifact_uri=artifact_uri,
                    char_count=len(page.clean_text),
                )
                session.add(snapshot)
                await session.flush()
            usage_snapshot = dict(run.usage_snapshot)
            usage_snapshot["pages"] = int(usage_snapshot.get("pages", 0)) + 1
            usage_snapshot["evidence_extraction_failures"] = (
                int(usage_snapshot.get("evidence_extraction_failures", 0)) + 1
            )
            run.usage_snapshot = usage_snapshot
            await self._append_event(
                session,
                run,
                event_type="source.read",
                public_summary=f"已读取来源: {page.title[:300]}",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={"clean_chars": len(page.clean_text), "truncated": page.truncated},
            )
            await self._append_event(
                session,
                run,
                event_type="context.assembled",
                public_summary="Context Manager 已选择当前问题与必要网页片段。",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={
                    "source_chars": len(page.clean_text),
                    "selected_chars": min(len(page.clean_text), 14_000),
                    "truncated": len(page.clean_text) > 14_000,
                },
            )
            await self._append_event(
                session,
                run,
                event_type="evidence.failed",
                public_summary="该来源的结构化证据抽取失败; 本轮保留其他已验证证据。",
                refs={
                    "source_id": str(source.id),
                    "question_id": target.question_id,
                    "error_code": error_code[:100],
                },
            )

    async def record_page_failure(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        url: str,
        error_code: str,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            await self._append_event(
                session,
                run,
                event_type="source.rejected",
                public_summary="候选来源未通过安全读取或内容校验。",
                refs={
                    "question_id": target.question_id,
                    "domain": (urlsplit(url).hostname or "unknown")[:255],
                    "error_code": error_code[:100],
                },
            )

    async def finish_iteration(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
    ) -> IterationEvaluation:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            previous_quality = dict(run.quality_snapshot)
            accepted_for_question = int(
                await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.question_id == target.question_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            sources_for_question = int(
                await session.scalar(
                    select(func.count(distinct(ResearchEvidenceRow.source_id))).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.question_id == target.question_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            plan_item = await session.scalar(
                select(ResearchPlanItemRow).where(
                    ResearchPlanItemRow.run_id == run_id,
                    ResearchPlanItemRow.plan_version == target.plan_version,
                    ResearchPlanItemRow.question_id == target.question_id,
                )
            )
            gap = await session.get(ResearchGapRow, target.gap_id)
            attempts = gap.resolution_attempts if gap is not None else _MAX_GAP_ATTEMPTS
            retry_current = (
                accepted_for_question == 0 or sources_for_question < 2
            ) and attempts < _MAX_GAP_ATTEMPTS
            if retry_current:
                question_status = "retrying"
            elif accepted_for_question > 0:
                question_status = "researched"
            else:
                question_status = "blocked"
            if plan_item is not None:
                plan_item.status = "active" if retry_current else question_status
            if gap is not None:
                gap.status = (
                    "open"
                    if retry_current
                    else "resolved"
                    if accepted_for_question > 0
                    else "abandoned"
                )
                gap.updated_at = datetime.now(UTC)

            plan_items = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                    )
                    .order_by(ResearchPlanItemRow.priority, ResearchPlanItemRow.question_id)
                )
            ).all()
            total_questions = len(plan_items)
            question_counts = (
                await session.execute(
                    select(
                        ResearchEvidenceRow.question_id,
                        func.count(ResearchEvidenceRow.id),
                        func.count(distinct(ResearchSourceRow.domain)),
                    )
                    .join(
                        ResearchSourceRow,
                        ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                    .group_by(ResearchEvidenceRow.question_id)
                )
            ).all()
            counts_by_question = {
                question_id: (int(evidence_count), int(question_sources))
                for question_id, evidence_count, question_sources in question_counts
            }
            coverage_map: list[_CoverageMapEntry] = []
            weighted_coverage = 0.0
            total_weight = 0.0
            for item in plan_items:
                evidence_count, independent_sources = counts_by_question.get(
                    item.question_id, (0, 0)
                )
                dimension_coverage = (
                    1.0
                    if evidence_count > 0 and independent_sources >= 2
                    else 0.5
                    if evidence_count > 0
                    else 0.0
                )
                missing_reasons: list[str] = []
                if evidence_count == 0:
                    missing_reasons.append("缺少可验证网页原文证据")
                if independent_sources < 2:
                    missing_reasons.append("缺少第二个独立来源")
                weight = float(4 - item.priority)
                total_weight += weight
                weighted_coverage += dimension_coverage * weight
                coverage_map.append(
                    {
                        "dimension_key": item.question_id,
                        "question": item.question,
                        "priority": item.priority,
                        "coverage": dimension_coverage,
                        "accepted_evidence": evidence_count,
                        "independent_sources": independent_sources,
                        "acceptance_criteria": list(item.evidence_requirements),
                        "missing_reasons": missing_reasons,
                    }
                )
            coverage = round(weighted_coverage / total_weight, 4) if total_weight else 0.0
            covered_questions = sum(
                1 for evidence_count, _ in counts_by_question.values() if evidence_count > 0
            )
            cross_validated_questions = sum(
                1
                for evidence_count, independent_sources in counts_by_question.values()
                if evidence_count > 0 and independent_sources >= 2
            )
            accepted_total = int(
                await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            unique_claims = int(
                await session.scalar(
                    select(func.count(distinct(ResearchEvidenceRow.claim))).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            candidate_total = int(
                await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.run_id == run_id,
                    )
                )
                or 0
            )
            source_count = int(
                await session.scalar(
                    select(func.count(distinct(ResearchEvidenceRow.source_id))).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            domain_count = int(
                await session.scalar(
                    select(func.count(distinct(ResearchSourceRow.domain)))
                    .join(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            source_quality = float(
                await session.scalar(
                    select(func.avg(ResearchEvidenceRow.source_reliability)).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0.0
            )
            cross_validation = (
                round(cross_validated_questions / covered_questions, 4)
                if covered_questions
                else 0.0
            )
            previous_facts = ResearchFactCounts(
                accepted_evidence=int(previous_quality.get("accepted_evidence", 0) or 0),
                unique_claims=int(previous_quality.get("claim_count", 0) or 0),
                independent_sources=int(
                    previous_quality.get(
                        "independent_source_count",
                        previous_quality.get("source_count", 0),
                    )
                    or 0
                ),
                evidence_candidates=int(previous_quality.get("candidate_evidence", 0) or 0),
                coverage=float(previous_quality.get("coverage", 0.0) or 0.0),
            )
            current_facts = ResearchFactCounts(
                accepted_evidence=accepted_total,
                unique_claims=unique_claims,
                independent_sources=domain_count,
                evidence_candidates=candidate_total,
                coverage=coverage,
            )
            information_gain = calculate_information_gain(previous_facts, current_facts)
            previous_low_gain_streak = _as_int(
                previous_quality.get("low_information_gain_streak", 0)
            )
            low_information_gain_streak = (
                previous_low_gain_streak + 1
                if information_gain.score < _LOW_INFORMATION_GAIN_THRESHOLD
                else 0
            )
            critical_gaps = sum(
                1
                for dimension in coverage_map
                if int(dimension["priority"]) == 1 and float(dimension["coverage"]) < 1.0
            )
            priority_one_coverages = [
                float(dimension["coverage"])
                for dimension in coverage_map
                if int(dimension["priority"]) == 1
            ]
            priority_one_coverage = min(priority_one_coverages) if priority_one_coverages else 0.0
            run.quality_snapshot = {
                "coverage": coverage,
                "information_gain": information_gain.score,
                "low_information_gain_streak": low_information_gain_streak,
                "coverage_map": coverage_map,
                "source_quality": round(source_quality, 4),
                "independent_source_count": domain_count,
                "priority_one_coverage": round(priority_one_coverage, 4),
                "source_independence": round(domain_count / source_count, 4)
                if source_count
                else 0.0,
                "cross_validation": cross_validation,
                "accepted_evidence": accepted_total,
                "candidate_evidence": candidate_total,
                "claim_count": unique_claims,
                "source_count": source_count,
                "conflict_count": 0,
                "citation_count": accepted_total,
                "unanswered_questions": max(total_questions - covered_questions, 0),
                "critical_gaps": critical_gaps,
            }
            usage = dict(run.usage_snapshot)
            usage["iterations"] = int(usage.get("iterations", 0)) + 1
            run.usage_snapshot = usage

            if retry_current:
                await self._append_event(
                    session,
                    run,
                    event_type="question.retry_scheduled",
                    public_summary=(
                        f"问题 {target.question_id} 当前有 {accepted_for_question} 条证据、"
                        f"{sources_for_question} 个来源; Evaluator 安排替代检索。"
                    ),
                    refs={
                        "question_id": target.question_id,
                        "status": "retrying",
                        "attempt": attempts,
                    },
                    metrics={
                        "accepted_count": accepted_for_question,
                        "source_count": sources_for_question,
                    },
                )
            else:
                await self._append_event(
                    session,
                    run,
                    event_type="question.researched",
                    public_summary=(
                        f"问题 {target.question_id} 获得 {accepted_for_question} 条有效证据。"
                    ),
                    refs={
                        "question_id": target.question_id,
                        "status": question_status,
                    },
                    metrics={
                        "accepted_count": accepted_for_question,
                        "source_count": sources_for_question,
                    },
                )

            remaining = int(
                await session.scalar(
                    select(func.count(ResearchPlanItemRow.id)).where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                        ResearchPlanItemRow.status.in_(("pending", "active")),
                    )
                )
                or 0
            )
            quality_met = (
                coverage >= 0.85
                and priority_one_coverage >= 0.80
                and source_quality >= 0.75
                and cross_validation >= 0.70
                and critical_gaps == 0
            )
            information_stagnated = (
                coverage >= 0.85
                and critical_gaps == 0
                and low_information_gain_streak >= _LOW_INFORMATION_GAIN_STREAK_TO_STOP
            )
            budget_hit = _budget_exhausted(run)
            if budget_hit:
                decision = "stop_budget"
                stop_reason: str | None = "research_budget_exhausted"
            elif quality_met:
                decision = "ready_to_write"
                stop_reason = "quality_met"
            elif information_stagnated:
                decision = "stop_information_gain"
                stop_reason = "stagnation"
            elif remaining == 0:
                decision = "write_with_limitations"
                stop_reason = "sources_exhausted"
            elif retry_current:
                decision = "retry_current"
                stop_reason = None
            else:
                decision = "continue_plan"
                stop_reason = None

            verdict = (
                "write"
                if decision
                in (
                    "ready_to_write",
                    "write_with_limitations",
                    "stop_budget",
                    "stop_information_gain",
                )
                else "continue"
            )
            session.add(
                EvaluationSnapshotRow(
                    id=uuid7(),
                    run_id=run_id,
                    scope="question",
                    state_version=run.state_version,
                    plan_version=run.plan_version,
                    coverage=coverage,
                    evidence_sufficiency=min(1.0, accepted_for_question / 2.0),
                    source_quality=source_quality,
                    source_diversity=min(1.0, domain_count / 3.0),
                    source_independence=run.quality_snapshot["source_independence"],
                    cross_validation=cross_validation,
                    freshness=1.0,
                    conflict_resolution=1.0,
                    citation_completeness=1.0 if accepted_total else 0.0,
                    citation_support=cross_validation,
                    weak_claim_ids=[],
                    missing_dimension_keys=[
                        d["dimension_key"] for d in coverage_map if d["coverage"] < 1.0
                    ],
                    unresolved_conflict_ids=[],
                    verdict=verdict,
                    created_at=datetime.now(UTC),
                )
            )
            run.phase = RunPhase.EVALUATING.value
            await self._append_event(
                session,
                run,
                event_type="research.information_gain_calculated",
                public_summary=(
                    f"本轮信息增益 {information_gain.score:.2f}: "
                    f"新增 {information_gain.new_evidence} 条有效证据、"
                    f"{information_gain.new_claims} 个 Claim、"
                    f"{information_gain.new_sources} 个独立来源, "
                    f"Coverage 提升 {information_gain.coverage_delta:.0%}。"
                ),
                refs={
                    "question_id": target.question_id,
                    "decision": decision,
                    "low_gain_streak": low_information_gain_streak,
                },
                metrics=information_gain.model_dump(mode="json"),
            )
            await self._append_event(
                session,
                run,
                event_type="evaluation.completed",
                public_summary=_evaluation_summary(decision, target.question_id),
                refs={
                    "question_id": target.question_id,
                    "decision": decision,
                    "reason": stop_reason,
                    "information_gain": information_gain.score,
                    "low_gain_streak": low_information_gain_streak,
                    "critical_gaps": critical_gaps,
                },
                metrics=run.quality_snapshot,
            )
            now = datetime.now(UTC)
            run.updated_at = now
            run.state_version += 1
            if stop_reason is None:
                run.status = RunStatus.RUNNING.value
                run.phase = RunPhase.RESEARCHING.value
                run.termination_reason = None
                run.lease_until = now + timedelta(seconds=300)
                await self._append_event(
                    session,
                    run,
                    event_type="research.continued",
                    public_summary=(
                        f"继续研究: Coverage {coverage:.0%}, "
                        f"信息增益 {information_gain.score:.2f}, "
                        f"仍有 {remaining} 个待处理问题和 {critical_gaps} 个关键缺口。"
                    ),
                    refs={
                        "decision": decision,
                        "information_gain": information_gain.score,
                        "critical_gaps": critical_gaps,
                    },
                    metrics=None,
                )
                return IterationEvaluation(
                    continue_research=True,
                    decision=decision,
                    stop_reason=None,
                    question_status=question_status,
                    coverage=coverage,
                    information_gain=information_gain.score,
                    low_information_gain_streak=low_information_gain_streak,
                )

            run.status = RunStatus.RUNNING.value
            run.phase = RunPhase.WRITING.value
            run.termination_reason = stop_reason
            run.lease_until = now + timedelta(seconds=300)
            await self._append_event(
                session,
                run,
                event_type="report.writing_started",
                public_summary=(
                    "研究预算已停止扩展; 使用现有证据生成带限制报告。"
                    if stop_reason == "research_budget_exhausted"
                    else (
                        "连续两轮边际信息增益过低且无关键缺口; 停止扩展并进入写作。"
                        if stop_reason == "stagnation"
                        else (
                            "计划已遍历但仍有未满足的验收条件; 使用现有证据生成带限制报告。"
                            if stop_reason == "sources_exhausted"
                            else "全部质量门已满足; 自动进入报告写作。"
                        )
                    )
                ),
                refs={"reason": run.termination_reason},
            )
            return IterationEvaluation(
                continue_research=False,
                decision=decision,
                stop_reason=stop_reason,
                question_status=question_status,
                coverage=coverage,
                information_gain=information_gain.score,
                low_information_gain_streak=low_information_gain_streak,
            )

    async def get_evidence_graph(
        self,
        owner_hash: str,
        run_id: UUID,
    ) -> EvidenceGraphView:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ResearchRunNotFoundError(str(run_id))

            rows = (
                await session.execute(
                    select(ResearchClaimRow, ResearchEvidenceRow)
                    .outerjoin(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.claim_id == ResearchClaimRow.id,
                    )
                    .where(ResearchClaimRow.run_id == run_id)
                    .order_by(
                        ResearchClaimRow.question_id,
                        ResearchClaimRow.created_at,
                        ResearchEvidenceRow.evidence_score.desc(),
                    )
                )
            ).tuples()
            claims: dict[UUID, EvidenceGraphClaimNode] = {}
            for claim, evidence in rows:
                node = claims.get(claim.id)
                if node is None:
                    node = EvidenceGraphClaimNode(
                        claim_id=claim.id,
                        question_id=claim.question_id,
                        dimension_key=claim.dimension_key,
                        atomic_claim=claim.atomic_claim,
                        status=claim.status,
                        confidence=claim.confidence,
                    )
                    claims[claim.id] = node
                if evidence is not None:
                    node.evidence.append(
                        EvidenceGraphEvidenceRef(
                            evidence_id=evidence.id,
                            source_id=evidence.source_id,
                            snapshot_id=evidence.snapshot_id,
                            chunk_id=evidence.chunk_id,
                            relation=evidence.relation,
                            accepted=evidence.accepted,
                            evidence_score=evidence.evidence_score,
                        )
                    )

            evidence_count = await session.scalar(
                select(func.count(ResearchEvidenceRow.id)).where(
                    ResearchEvidenceRow.run_id == run_id,
                    ResearchEvidenceRow.claim_id.is_not(None),
                )
            )
            snapshot_count = await session.scalar(
                select(func.count(ResearchSourceSnapshotRow.id)).where(
                    ResearchSourceSnapshotRow.run_id == run_id
                )
            )
            chunk_count = await session.scalar(
                select(func.count(ResearchSourceChunkRow.id)).where(
                    ResearchSourceChunkRow.run_id == run_id
                )
            )
            edge_rows = (
                await session.scalars(
                    select(ResearchClaimEdgeRow)
                    .where(ResearchClaimEdgeRow.run_id == run_id)
                    .order_by(
                        ResearchClaimEdgeRow.relation,
                        ResearchClaimEdgeRow.from_claim_id,
                        ResearchClaimEdgeRow.to_claim_id,
                    )
                )
            ).all()
            conflict_rows = (
                await session.scalars(
                    select(ResearchConflictRow)
                    .where(ResearchConflictRow.run_id == run_id)
                    .order_by(
                        ResearchConflictRow.status,
                        ResearchConflictRow.severity.desc(),
                        ResearchConflictRow.created_at,
                    )
                )
            ).all()
            return EvidenceGraphView(
                run_id=run_id,
                claim_count=len(claims),
                evidence_count=int(evidence_count or 0),
                snapshot_count=int(snapshot_count or 0),
                chunk_count=int(chunk_count or 0),
                edge_count=len(edge_rows),
                conflict_count=len(conflict_rows),
                claims=list(claims.values()),
                edges=[
                    EvidenceGraphClaimEdgeView(
                        edge_id=edge.id,
                        from_claim_id=edge.from_claim_id,
                        to_claim_id=edge.to_claim_id,
                        relation=edge.relation,
                        confidence=edge.confidence,
                    )
                    for edge in edge_rows
                ],
                conflicts=[
                    EvidenceGraphConflictView(
                        conflict_id=conflict.id,
                        question_id=conflict.question_id,
                        entity=conflict.entity,
                        attribute=conflict.attribute,
                        left_evidence_id=conflict.left_evidence_id,
                        right_evidence_id=conflict.right_evidence_id,
                        severity=conflict.severity,
                        status=conflict.status,
                        resolution_summary=conflict.resolution_summary,
                    )
                    for conflict in conflict_rows
                ],
            )

    async def list_evidence(self, owner_hash: str, run_id: UUID) -> list[EvidenceView]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ResearchRunNotFoundError(str(run_id))
            rows = (
                await session.execute(
                    select(ResearchEvidenceRow, ResearchSourceRow)
                    .join(ResearchSourceRow, ResearchEvidenceRow.source_id == ResearchSourceRow.id)
                    .where(ResearchEvidenceRow.run_id == run_id)
                    .order_by(
                        ResearchEvidenceRow.accepted.desc(),
                        ResearchEvidenceRow.evidence_score.desc(),
                        ResearchEvidenceRow.created_at,
                    )
                )
            ).tuples()
            return [
                EvidenceView(
                    evidence_id=evidence.id,
                    claim_id=evidence.claim_id,
                    snapshot_id=evidence.snapshot_id,
                    chunk_id=evidence.chunk_id,
                    question_id=evidence.question_id,
                    claim=evidence.claim,
                    exact_quote=evidence.exact_quote,
                    relation=evidence.relation,
                    source_title=source.title,
                    source_url=source.canonical_url,
                    source_domain=source.domain,
                    source_reliability=evidence.source_reliability,
                    evidence_score=evidence.evidence_score,
                    accepted=evidence.accepted,
                    rejection_reason=evidence.rejection_reason,
                )
                for evidence, source in rows
            ]

    async def list_evaluations(
        self, owner_hash: str, run_id: UUID
    ) -> list[EvaluationSnapshot]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ResearchRunNotFoundError(str(run_id))
            rows = (
                await session.scalars(
                    select(EvaluationSnapshotRow)
                    .where(EvaluationSnapshotRow.run_id == run_id)
                    .order_by(EvaluationSnapshotRow.created_at, EvaluationSnapshotRow.id)
                )
            ).all()
            return [
                EvaluationSnapshot(
                    evaluation_id=row.id,
                    run_id=row.run_id,
                    scope=EvaluationScope(row.scope),
                    state_version=row.state_version,
                    plan_version=row.plan_version,
                    coverage=row.coverage,
                    evidence_sufficiency=row.evidence_sufficiency,
                    source_quality=row.source_quality,
                    source_diversity=row.source_diversity,
                    source_independence=row.source_independence,
                    cross_validation=row.cross_validation,
                    freshness=row.freshness,
                    conflict_resolution=row.conflict_resolution,
                    citation_completeness=row.citation_completeness,
                    citation_support=row.citation_support,
                    weak_claim_ids=tuple(row.weak_claim_ids),
                    missing_dimension_keys=tuple(row.missing_dimension_keys),
                    unresolved_conflict_ids=tuple(row.unresolved_conflict_ids),
                    verdict=EvaluationVerdict(row.verdict),
                )
                for row in rows
            ]

    async def _enter_writing(
        self,
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        reason: str,
        summary: str,
    ) -> None:
        now = datetime.now(UTC)
        run.status = RunStatus.RUNNING.value
        run.phase = RunPhase.WRITING.value
        run.termination_reason = reason
        run.lease_until = now + timedelta(seconds=300)
        run.updated_at = now
        run.state_version += 1
        await self._append_event(
            session,
            run,
            event_type="report.writing_started",
            public_summary=summary,
            refs={"reason": reason},
        )

    @staticmethod
    async def _locked_run(
        session: AsyncSession, run_id: UUID, worker_task_id: str
    ) -> ResearchRunRow:
        run = await session.scalar(
            select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
        )
        if (
            run is None
            or RunStatus(run.status) != RunStatus.RUNNING
            or run.worker_task_id != worker_task_id
        ):
            raise ResearchLeaseLostError(str(run_id))
        return run

    @staticmethod
    async def _append_event(
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        event_type: str,
        public_summary: str,
        refs: dict[str, object],
        metrics: dict[str, object] | None = None,
    ) -> None:
        sequence = run.next_event_seq
        run.next_event_seq += 1
        session.add(
            AgentEventRow(
                run_id=run.id,
                run_seq=sequence,
                schema_version=1,
                phase=run.phase,
                event_type=event_type,
                public_summary=public_summary,
                refs=refs,
                metrics=metrics,
            )
        )
        await session.flush()


def _model_tokens(run: ResearchRunRow) -> int:
    planner = run.usage_snapshot.get("planner")
    planner_tokens = int(planner.get("total_tokens", 0)) if isinstance(planner, dict) else 0
    return planner_tokens + int(run.usage_snapshot.get("evidence_total_tokens", 0))


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _budget_exhausted(run: ResearchRunRow) -> bool:
    budget = run.budget_snapshot
    usage = run.usage_snapshot
    return (
        int(usage.get("iterations", 0)) >= int(budget.get("max_iterations", 0))
        or int(usage.get("searches", 0)) >= int(budget.get("max_searches", 0))
        or int(usage.get("pages", 0)) >= int(budget.get("max_pages", 0))
        or _model_tokens(run) >= int(budget.get("max_tokens", 0))
    )


def _evaluation_summary(decision: str, question_id: str) -> str:
    summaries = {
        "retry_current": f"问题 {question_id} 的证据质量不足; 自动使用替代检索词重试。",
        "continue_plan": f"问题 {question_id} 已完成评估; 自动推进下一个研究问题。",
        "ready_to_write": "全部研究质量门已满足; 进入报告写作。",
        "write_with_limitations": (
            "计划已遍历但部分验收条件仍未满足; 使用现有证据进入限制性写作。"
        ),
        "stop_budget": "研究预算已耗尽; 已保留当前计划、来源、证据与质量快照。",
        "stop_information_gain": (
            "研究覆盖已达到可接受水平且连续两轮信息增益低于阈值; 停止低边际价值检索。"
        ),
    }
    return summaries.get(decision, "Evaluator 已完成本轮研究质量检查。")
