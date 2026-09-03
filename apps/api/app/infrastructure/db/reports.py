"""Transactional report context, persistence, and citation lookup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.domain.providers import TokenUsage
from app.domain.reports import ReportCitationView, ReportSectionView, ReportView
from app.domain.research_runs import RunPhase, RunStatus
from app.infrastructure.db.analysis_models import AnalysisArtifactRow, AnalysisInputRow
from app.infrastructure.db.evidence_graph_models import (
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.report_models import ReportCitationRow, ReportRow, ReportSectionRow
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow


class ReportNotFoundError(LookupError):
    pass


class ReportWritingLeaseLostError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReportQuestion:
    question_id: str
    question: str
    priority: int


@dataclass(frozen=True, slots=True)
class ReportEvidenceCard:
    evidence_id: UUID
    claim_id: UUID
    snapshot_id: UUID
    chunk_id: UUID
    question_id: str
    claim: str
    exact_quote: str
    evidence_score: float
    source_title: str
    source_url: str
    source_domain: str
    source_content_hash: str
    fetched_at: datetime
    analysis_artifact_id: UUID | None = None
    analysis_operation: str | None = None
    analysis_formula: str | None = None
    analysis_result: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ReportContext:
    run_id: UUID
    goal: str
    stop_reason: str | None
    budget_snapshot: dict[str, Any]
    usage_snapshot: dict[str, Any]
    quality_snapshot: dict[str, Any]
    questions: tuple[ReportQuestion, ...]
    evidence: tuple[ReportEvidenceCard, ...]


@dataclass(frozen=True, slots=True)
class PersistedSection:
    section_key: str
    title: str
    markdown: str
    verification_result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PersistedCitation:
    citation_number: int
    evidence_id: UUID
    claim_id: UUID
    snapshot_id: UUID
    chunk_id: UUID
    source_content_hash: str
    url: str
    accessed_at: datetime
    analysis_artifact_id: UUID | None = None


class ReportRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load_for_writing(self, run_id: UUID, *, worker_task_id: str) -> ReportContext:
        async with self._sessions() as session:
            run = self._require_writing_lease(
                await session.get(ResearchRunRow, run_id), worker_task_id
            )
            questions = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                    )
                    .order_by(ResearchPlanItemRow.priority, ResearchPlanItemRow.question_id)
                )
            ).all()
            evidence_rows = (
                (
                    await session.execute(
                        select(
                            ResearchEvidenceRow,
                            ResearchSourceRow,
                            ResearchSourceSnapshotRow,
                            ResearchSourceChunkRow,
                        )
                        .join(
                            ResearchSourceRow,
                            ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                        )
                        .join(
                            ResearchSourceSnapshotRow,
                            ResearchSourceSnapshotRow.id == ResearchEvidenceRow.snapshot_id,
                        )
                        .join(
                            ResearchSourceChunkRow,
                            ResearchSourceChunkRow.id == ResearchEvidenceRow.chunk_id,
                        )
                        .where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.accepted.is_(True),
                            ResearchEvidenceRow.claim_id.is_not(None),
                        )
                        .order_by(
                            ResearchEvidenceRow.question_id,
                            ResearchEvidenceRow.evidence_score.desc(),
                            ResearchEvidenceRow.created_at,
                        )
                    )
                )
                .tuples()
                .all()
            )
            evidence_ids = [evidence.id for evidence, _source, _snapshot, _chunk in evidence_rows]
            analysis_rows = (
                (
                    await session.execute(
                        select(AnalysisInputRow.evidence_id, AnalysisArtifactRow)
                        .join(
                            AnalysisArtifactRow,
                            AnalysisArtifactRow.id == AnalysisInputRow.analysis_artifact_id,
                        )
                        .where(
                            AnalysisArtifactRow.run_id == run_id,
                            AnalysisInputRow.evidence_id.in_(evidence_ids),
                        )
                        .order_by(
                            AnalysisArtifactRow.created_at.desc(),
                            AnalysisArtifactRow.id,
                        )
                    )
                )
                .tuples()
                .all()
                if evidence_ids
                else []
            )
            analysis_by_evidence: dict[UUID, AnalysisArtifactRow] = {}
            for evidence_id, artifact in analysis_rows:
                analysis_by_evidence.setdefault(evidence_id, artifact)
            analysis_ids = {
                evidence_id: artifact.id
                for evidence_id, artifact in analysis_by_evidence.items()
            }
            analysis_operations = {
                evidence_id: artifact.operation
                for evidence_id, artifact in analysis_by_evidence.items()
            }
            analysis_formulas = {
                evidence_id: artifact.formula
                for evidence_id, artifact in analysis_by_evidence.items()
            }
            analysis_results = {
                evidence_id: artifact.result
                for evidence_id, artifact in analysis_by_evidence.items()
            }
            return ReportContext(
                run_id=run.id,
                goal=run.normalized_goal,
                stop_reason=run.termination_reason,
                budget_snapshot=run.budget_snapshot,
                usage_snapshot=run.usage_snapshot,
                quality_snapshot=run.quality_snapshot,
                questions=tuple(
                    ReportQuestion(row.question_id, row.question, row.priority) for row in questions
                ),
                evidence=tuple(
                    ReportEvidenceCard(
                        evidence_id=evidence.id,
                        # The query excludes NULL claim IDs; SQLAlchemy does not
                        # propagate that SQL predicate into the ORM attribute type.
                        claim_id=cast(UUID, evidence.claim_id),
                        snapshot_id=snapshot.id,
                        chunk_id=chunk.id,
                        question_id=evidence.question_id,
                        claim=evidence.claim,
                        exact_quote=evidence.exact_quote,
                        evidence_score=evidence.evidence_score,
                        source_title=source.title,
                        source_url=source.canonical_url,
                        source_domain=source.domain,
                        source_content_hash=snapshot.content_hash,
                        fetched_at=snapshot.fetched_at,
                        analysis_artifact_id=analysis_ids.get(evidence.id),
                        analysis_operation=analysis_operations.get(evidence.id),
                        analysis_formula=analysis_formulas.get(evidence.id),
                        analysis_result=analysis_results.get(evidence.id),
                    )
                    for evidence, source, snapshot, chunk in evidence_rows
                ),
            )

    async def save(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        title: str,
        final_markdown: str,
        limitations: list[str],
        verification_result: dict[str, Any],
        sections: list[PersistedSection],
        citations: list[PersistedCitation],
        usage: TokenUsage | None,
        writer_mode: str,
        completed_with_limitations: bool,
    ) -> ReportView:
        async with self._sessions() as session, session.begin():
            run = self._require_writing_lease(
                await session.scalar(
                    select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
                ),
                worker_task_id,
            )
            version = (
                int(
                    await session.scalar(
                        select(func.coalesce(func.max(ReportRow.version), 0)).where(
                            ReportRow.run_id == run_id
                        )
                    )
                    or 0
                )
                + 1
            )
            now = datetime.now(UTC)
            report = ReportRow(
                id=uuid7(),
                run_id=run_id,
                version=version,
                title=title,
                final_markdown=final_markdown,
                limitations=limitations,
                verification_result=verification_result,
                status="verified",
                created_at=now,
                updated_at=now,
            )
            session.add(report)
            await session.flush()
            for order, section in enumerate(sections, start=1):
                session.add(
                    ReportSectionRow(
                        id=uuid7(),
                        report_id=report.id,
                        outline_order=order,
                        section_key=section.section_key,
                        title=section.title,
                        draft_markdown=section.markdown,
                        status="verified",
                        verification_result=section.verification_result,
                    )
                )
                await self._append_event(
                    session,
                    run,
                    event_type="report.section_completed",
                    public_summary=f"报告章节已生成并完成引用校验: {section.title}",
                    refs={"report_id": str(report.id), "section_key": section.section_key},
                    metrics=section.verification_result,
                )
            for citation in citations:
                session.add(
                    ReportCitationRow(
                        id=uuid7(),
                        report_id=report.id,
                        citation_number=citation.citation_number,
                        analysis_artifact_id=citation.analysis_artifact_id,
                        evidence_id=citation.evidence_id,
                        claim_id=citation.claim_id,
                        snapshot_id=citation.snapshot_id,
                        chunk_id=citation.chunk_id,
                        source_content_hash=citation.source_content_hash,
                        url=citation.url,
                        accessed_at=citation.accessed_at,
                    )
                )
            writer_usage = usage.model_dump(mode="json") if usage is not None else None
            run.usage_snapshot = {
                **run.usage_snapshot,
                "writer": writer_usage,
                "writer_mode": writer_mode,
            }
            run.quality_snapshot = {
                **run.quality_snapshot,
                "citation_completeness": verification_result["citation_completeness"],
                "numeric_citation_rate": verification_result["numeric_citation_rate"],
                "report_verified": verification_result["verified"],
            }
            run.status = (
                RunStatus.COMPLETED_WITH_LIMITATIONS.value
                if completed_with_limitations
                else RunStatus.COMPLETED.value
            )
            run.phase = RunPhase.TERMINAL.value
            run.termination_reason = (
                "completed_with_limitations" if completed_with_limitations else "quality_met"
            )
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.finished_at = now
            run.updated_at = now
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="report.verified",
                public_summary="报告引用已通过确定性校验。",
                refs={"report_id": str(report.id), "version": version},
                metrics=verification_result,
            )
            await self._append_event(
                session,
                run,
                event_type="run.completed",
                public_summary=(
                    "研究报告已生成, 存在明确披露的限制。"
                    if completed_with_limitations
                    else "研究报告已生成并完成验证。"
                ),
                refs={"report_id": str(report.id), "status": run.status},
                metrics=None,
            )
        return await self.get_for_run_by_id(run_id)

    async def fail_no_evidence(self, run_id: UUID, *, worker_task_id: str) -> None:
        async with self._sessions() as session, session.begin():
            run = self._require_writing_lease(
                await session.scalar(
                    select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
                ),
                worker_task_id,
            )
            now = datetime.now(UTC)
            run.status = RunStatus.FAILED.value
            run.phase = RunPhase.TERMINAL.value
            run.termination_reason = "REPORT_NO_ACCEPTED_EVIDENCE"
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.finished_at = now
            run.updated_at = now
            run.state_version += 1
            usage = dict(run.usage_snapshot)
            searches = int(usage.get("searches", 0) or 0)
            pages = int(usage.get("pages", 0) or 0)
            search_failures = int(usage.get("search_provider_failures", 0) or 0)
            extraction_failures = int(usage.get("evidence_extraction_failures", 0) or 0)
            if search_failures and pages == 0 and (
                searches == 0 or search_failures >= searches
            ):
                detail_code = "SEARCH_PROVIDER_EXHAUSTED"
                cause = "search_provider_unavailable"
            elif pages and extraction_failures >= pages:
                detail_code = "EVIDENCE_EXTRACTION_UNAVAILABLE"
                cause = "model_extraction_unavailable"
            elif pages:
                detail_code = "EVIDENCE_VALIDATION_EMPTY"
                cause = "no_candidate_passed_provenance_validation"
            else:
                detail_code = "NO_READABLE_SOURCE"
                cause = "no_readable_source"
            await self._append_event(
                session,
                run,
                event_type="run.failed",
                public_summary="没有可验证证据, 系统拒绝生成看似完整的报告。",
                refs={
                    "reason": "REPORT_NO_ACCEPTED_EVIDENCE",
                    "detail_code": detail_code,
                    "failure_stage": "evidence_evaluation",
                    "failure_cause": cause,
                },
                metrics={
                    "searches": searches,
                    "pages": pages,
                    "search_provider_failures": search_failures,
                    "page_read_failures": int(usage.get("page_read_failures", 0) or 0),
                    "evidence_extraction_failures": extraction_failures,
                    "accepted_evidence": 0,
                },
            )

    async def get_for_run(self, owner_hash: str, run_id: UUID) -> ReportView:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ReportNotFoundError(str(run_id))
            return await self._get_for_run(session, run_id)

    async def get_for_run_by_id(self, run_id: UUID) -> ReportView:
        async with self._sessions() as session:
            return await self._get_for_run(session, run_id)

    async def get_citation(
        self, owner_hash: str, report_id: UUID, citation_number: int
    ) -> ReportCitationView:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        select(
                            ReportCitationRow,
                            ResearchEvidenceRow,
                            ResearchSourceRow,
                            ResearchSourceSnapshotRow,
                            ResearchSourceChunkRow,
                        )
                        .join(ReportRow, ReportRow.id == ReportCitationRow.report_id)
                        .join(ResearchRunRow, ResearchRunRow.id == ReportRow.run_id)
                        .join(
                            ResearchEvidenceRow,
                            ResearchEvidenceRow.id == ReportCitationRow.evidence_id,
                        )
                        .join(
                            ResearchSourceRow,
                            ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                        )
                        .outerjoin(
                            ResearchSourceSnapshotRow,
                            ResearchSourceSnapshotRow.id
                            == func.coalesce(
                                ReportCitationRow.snapshot_id,
                                ResearchEvidenceRow.snapshot_id,
                            ),
                        )
                        .outerjoin(
                            ResearchSourceChunkRow,
                            ResearchSourceChunkRow.id
                            == func.coalesce(
                                ReportCitationRow.chunk_id,
                                ResearchEvidenceRow.chunk_id,
                            ),
                        )
                        .where(
                            ReportCitationRow.report_id == report_id,
                            ReportCitationRow.citation_number == citation_number,
                            ResearchRunRow.owner_hash == owner_hash,
                        )
                    )
                )
                .tuples()
                .first()
            )
            if row is None:
                raise ReportNotFoundError(f"{report_id}:{citation_number}")
            return self._citation_view(*row)

    async def _get_for_run(self, session: AsyncSession, run_id: UUID) -> ReportView:
        report = await session.scalar(
            select(ReportRow)
            .where(ReportRow.run_id == run_id)
            .order_by(ReportRow.version.desc())
            .limit(1)
        )
        if report is None:
            raise ReportNotFoundError(str(run_id))
        section_rows = (
            await session.scalars(
                select(ReportSectionRow)
                .where(ReportSectionRow.report_id == report.id)
                .order_by(ReportSectionRow.outline_order)
            )
        ).all()
        citation_rows = (
            (
                await session.execute(
                    select(
                        ReportCitationRow,
                        ResearchEvidenceRow,
                        ResearchSourceRow,
                        ResearchSourceSnapshotRow,
                        ResearchSourceChunkRow,
                    )
                    .join(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.id == ReportCitationRow.evidence_id,
                    )
                    .join(
                        ResearchSourceRow,
                        ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                    )
                    .outerjoin(
                        ResearchSourceSnapshotRow,
                        ResearchSourceSnapshotRow.id
                        == func.coalesce(
                            ReportCitationRow.snapshot_id,
                            ResearchEvidenceRow.snapshot_id,
                        ),
                    )
                    .outerjoin(
                        ResearchSourceChunkRow,
                        ResearchSourceChunkRow.id
                        == func.coalesce(
                            ReportCitationRow.chunk_id,
                            ResearchEvidenceRow.chunk_id,
                        ),
                    )
                    .where(ReportCitationRow.report_id == report.id)
                    .order_by(ReportCitationRow.citation_number)
                )
            )
            .tuples()
            .all()
        )
        return ReportView(
            report_id=report.id,
            run_id=report.run_id,
            version=report.version,
            title=report.title,
            final_markdown=report.final_markdown,
            limitations=list(report.limitations),
            verification_result=report.verification_result,
            status=report.status,
            created_at=report.created_at,
            sections=[
                ReportSectionView(
                    outline_order=row.outline_order,
                    section_key=row.section_key,
                    title=row.title,
                    draft_markdown=row.draft_markdown,
                    status=row.status,
                    verification_result=row.verification_result,
                )
                for row in section_rows
            ],
            citations=[self._citation_view(*row) for row in citation_rows],
        )

    @staticmethod
    def _citation_view(
        citation: ReportCitationRow,
        evidence: ResearchEvidenceRow,
        source: ResearchSourceRow,
        snapshot: ResearchSourceSnapshotRow | None,
        chunk: ResearchSourceChunkRow | None,
    ) -> ReportCitationView:
        if snapshot is None or chunk is None:
            raise ReportNotFoundError(f"citation provenance missing: {citation.id}")
        return ReportCitationView(
            citation_number=citation.citation_number,
            analysis_artifact_id=citation.analysis_artifact_id,
            evidence_id=evidence.id,
            claim_id=citation.claim_id,
            snapshot_id=citation.snapshot_id,
            chunk_id=citation.chunk_id,
            question_id=evidence.question_id,
            claim=evidence.claim,
            exact_quote=evidence.exact_quote,
            source_title=source.title,
            source_url=citation.url,
            source_domain=source.domain,
            source_content_hash=citation.source_content_hash,
            snapshot_content_hash=snapshot.content_hash,
            chunk_char_start=chunk.char_start,
            chunk_char_end=chunk.char_end,
            accessed_at=citation.accessed_at,
        )

    @staticmethod
    def _require_writing_lease(run: ResearchRunRow | None, worker_task_id: str) -> ResearchRunRow:
        if (
            run is None
            or RunStatus(run.status) != RunStatus.RUNNING
            or RunPhase(run.phase) != RunPhase.WRITING
            or run.worker_task_id != worker_task_id
        ):
            raise ReportWritingLeaseLostError(worker_task_id)
        return run

    @staticmethod
    async def _append_event(
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        event_type: str,
        public_summary: str,
        refs: dict[str, object],
        metrics: dict[str, Any] | None,
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
