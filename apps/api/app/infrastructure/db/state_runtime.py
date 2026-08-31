"""CAS-controlled ResearchState projection used by the LangGraph runtime."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid5

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.reducers.state_reducer import StateVersionConflict, apply_patch
from app.domain.identifiers import uuid7
from app.domain.state import (
    BudgetLimits,
    BudgetUsage,
    CoverageDimensionSnapshot,
    GapStatus,
    GapType,
    KnowledgeStatus,
    KnownClaimRef,
    QualitySnapshot,
    ResearchGap,
    ResearchPhase,
    ResearchState,
    RunStatus,
    StatePatch,
    StopReason,
)
from app.infrastructure.db.research_models import (
    ResearchEvidenceRow,
    ResearchGapRow,
    ResearchSourceRow,
)
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow
from app.infrastructure.db.state_models import ResearchStatePatchRow, ResearchStateSnapshotRow


class StateRuntimeLeaseLostError(RuntimeError):
    pass


class StateSnapshotNotFoundError(LookupError):
    pass


class ResearchStateRuntimeRepository:
    """Keep a small, auditable Agent state separate from full business facts."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def ensure_initialized(self, run_id: UUID) -> ResearchState:
        async with self._sessions() as session, session.begin():
            existing = await session.get(ResearchStateSnapshotRow, run_id)
            if existing is not None:
                return ResearchState.model_validate(existing.state_json)

            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if run is None:
                raise StateSnapshotNotFoundError(str(run_id))
            state = ResearchState(
                run_id=run_id,
                status=_state_status(run.status, run.phase),
                phase=_state_phase(run.phase),
                budget_limits=_budget_limits(run),
                budget_usage=_budget_usage(run),
                quality=_quality(run),
                coverage_map=_coverage_map(run),
                stop_reason=_stop_reason(run.termination_reason),
            )
            state_json = state.model_dump(mode="json")
            state_hash = _state_hash(state_json)
            session.add(
                ResearchStateSnapshotRow(
                    run_id=run_id,
                    state_version=state.state_version,
                    graph_schema_revision=state.graph_schema_revision,
                    state_json=state_json,
                    state_hash=state_hash,
                    updated_at=datetime.now(UTC),
                )
            )
            await self._append_event(
                session,
                run,
                event_type="state.initialized",
                public_summary="LangGraph 轻量 ResearchState 已初始化。",
                refs={"state_version": state.state_version, "state_hash": state_hash},
            )
            return state

    async def get(self, run_id: UUID) -> ResearchState:
        async with self._sessions() as session:
            row = await session.get(ResearchStateSnapshotRow, run_id)
            if row is None:
                raise StateSnapshotNotFoundError(str(run_id))
            return ResearchState.model_validate(row.state_json)

    async def synchronize(
        self,
        run_id: UUID,
        *,
        node_name: str,
        worker_task_id: str | None,
    ) -> ResearchState:
        """Project accepted business facts into State and persist one validated StatePatch."""

        async with self._sessions() as session, session.begin():
            snapshot = await session.scalar(
                select(ResearchStateSnapshotRow)
                .where(ResearchStateSnapshotRow.run_id == run_id)
                .with_for_update()
            )
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if snapshot is None or run is None:
                raise StateSnapshotNotFoundError(str(run_id))
            if worker_task_id is not None and run.worker_task_id != worker_task_id:
                raise StateRuntimeLeaseLostError(str(run_id))

            current = ResearchState.model_validate(snapshot.state_json)
            if current.state_version != snapshot.state_version:
                raise StateVersionConflict("snapshot row and serialized state version diverged")

            evidence_rows = (
                (
                    await session.execute(
                        select(ResearchEvidenceRow, ResearchSourceRow)
                        .join(
                            ResearchSourceRow, ResearchSourceRow.id == ResearchEvidenceRow.source_id
                        )
                        .where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.accepted.is_(True),
                        )
                        .order_by(ResearchEvidenceRow.created_at, ResearchEvidenceRow.id)
                    )
                )
                .tuples()
                .all()
            )
            gap_rows = (
                await session.scalars(
                    select(ResearchGapRow)
                    .where(ResearchGapRow.run_id == run_id)
                    .order_by(ResearchGapRow.created_at, ResearchGapRow.id)
                )
            ).all()

            known = tuple(
                KnownClaimRef(
                    claim_id=evidence.id,
                    question_id=_question_uuid(run_id, evidence.question_id),
                    dimension_key=evidence.question_id,
                    status=KnowledgeStatus.SUPPORTED,
                    confidence=_score(evidence.evidence_score),
                    evidence_ids=(evidence.id,),
                    independent_source_owner_keys=(source.domain,),
                )
                for evidence, source in evidence_rows
            )
            claim_ids_by_question: dict[str, list[UUID]] = {}
            for evidence, _source in evidence_rows:
                claim_ids_by_question.setdefault(evidence.question_id, []).append(evidence.id)
            gaps = tuple(
                _project_gap(run_id, gap, claim_ids_by_question.get(gap.question_id, []))
                for gap in gap_rows
            )
            patch = StatePatch(
                patch_id=uuid7(),
                base_version=current.state_version,
                known_upserts=known,
                gap_upserts=gaps,
                budget_usage=_budget_usage(run),
                quality=_quality(run),
                coverage_map=_coverage_map(run),
                phase=_state_phase(run.phase),
                status=_state_status(run.status, run.phase),
                stop_reason=_stop_reason(run.termination_reason),
                clear_stop_reason=run.termination_reason is None,
            )
            updated = apply_patch(current, patch)
            state_json = updated.model_dump(mode="json")
            state_hash = _state_hash(state_json)
            snapshot.state_version = updated.state_version
            snapshot.graph_schema_revision = updated.graph_schema_revision
            snapshot.state_json = state_json
            snapshot.state_hash = state_hash
            snapshot.updated_at = datetime.now(UTC)
            session.add(
                ResearchStatePatchRow(
                    id=patch.patch_id,
                    run_id=run_id,
                    base_version=patch.base_version,
                    result_version=updated.state_version,
                    node_name=node_name,
                    patch_json=patch.model_dump(mode="json"),
                    result_state_hash=state_hash,
                )
            )
            await self._append_event(
                session,
                run,
                event_type="state.patch_applied",
                public_summary=f"{node_name} 节点已通过 StatePatch 更新 ResearchState。",
                refs={
                    "node": node_name,
                    "base_version": patch.base_version,
                    "result_version": updated.state_version,
                    "state_hash": state_hash,
                },
            )
            return updated

    @staticmethod
    async def _append_event(
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        event_type: str,
        public_summary: str,
        refs: dict[str, object],
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
                metrics=None,
            )
        )
        await session.flush()


def _project_gap(run_id: UUID, row: ResearchGapRow, claim_ids: list[UUID]) -> ResearchGap:
    status = _gap_status(row.status)
    resolved_ids = tuple(claim_ids) if status is GapStatus.RESOLVED else ()
    if status is GapStatus.RESOLVED and not resolved_ids:
        status = GapStatus.ABANDONED
    return ResearchGap(
        gap_id=row.id,
        question_id=_question_uuid(run_id, row.question_id),
        dimension_key=row.question_id,
        gap_type=_gap_type(row.gap_type),
        description=row.description,
        acceptance_criteria=row.acceptance_criteria,
        severity=_score(row.severity),
        resolution_attempts=row.resolution_attempts,
        status=status,
        resolved_by_claim_ids=resolved_ids,
    )


def _budget_limits(run: ResearchRunRow) -> BudgetLimits:
    budget = run.budget_snapshot
    return BudgetLimits(
        max_iterations=max(1, int(budget.get("max_iterations", 8) or 8)),
        max_searches=max(0, int(budget.get("max_searches", 15) or 0)),
        max_pages=max(0, int(budget.get("max_pages", 30) or 0)),
        max_model_tokens=max(1, int(budget.get("max_tokens", 100_000) or 100_000)),
    )


def _budget_usage(run: ResearchRunRow) -> BudgetUsage:
    usage = run.usage_snapshot
    planner = usage.get("planner", {})
    planner_tokens = int(planner.get("total_tokens", 0)) if isinstance(planner, dict) else 0
    writer = usage.get("writer", {})
    writer_tokens = int(writer.get("total_tokens", 0)) if isinstance(writer, dict) else 0
    return BudgetUsage(
        iterations=max(0, int(usage.get("iterations", 0) or 0)),
        searches=max(0, int(usage.get("searches", 0) or 0)),
        pages=max(0, int(usage.get("pages", 0) or 0)),
        model_tokens=max(
            0,
            planner_tokens + writer_tokens + int(usage.get("evidence_total_tokens", 0) or 0),
        ),
    )


def _quality(run: ResearchRunRow) -> QualitySnapshot:
    quality = run.quality_snapshot
    return QualitySnapshot(
        coverage=_score(quality.get("coverage", 0.0)),
        information_gain=_score(quality.get("information_gain", 0.0)),
        low_information_gain_streak=max(0, int(quality.get("low_information_gain_streak", 0) or 0)),
        source_quality=_score(quality.get("source_quality", 0.0)),
        source_independence=_score(quality.get("source_independence", 0.0)),
        cross_validation=_score(quality.get("cross_validation", 0.0)),
        freshness=_score(quality.get("freshness", 0.0)),
        citation_support=_score(quality.get("citation_support", 0.0)),
    )


def _coverage_map(run: ResearchRunRow) -> tuple[CoverageDimensionSnapshot, ...]:
    raw_items = run.quality_snapshot.get("coverage_map", [])
    if not isinstance(raw_items, list):
        return ()
    return tuple(
        CoverageDimensionSnapshot.model_validate(item)
        for item in raw_items
        if isinstance(item, dict)
    )


def _state_status(status: str, phase: str) -> RunStatus:
    if status == "queued":
        return RunStatus.QUEUED
    if status == "completed":
        return RunStatus.COMPLETED
    if status == "completed_with_limitations":
        return RunStatus.COMPLETED_WITH_LIMITATIONS
    if status == "cancelled":
        return RunStatus.CANCELLED
    if status == "credentials_required":
        return RunStatus.CREDENTIALS_REQUIRED
    if status == "interrupted":
        return RunStatus.INTERRUPTED
    if status == "failed":
        return RunStatus.FAILED
    if phase == "writing":
        return RunStatus.WRITING
    if phase == "verifying":
        return RunStatus.VERIFYING
    return RunStatus.RESEARCHING


def _state_phase(phase: str) -> ResearchPhase:
    return {
        "initializing": ResearchPhase.INIT,
        "planning": ResearchPhase.PLAN,
        "researching": ResearchPhase.RESEARCH,
        "evaluating": ResearchPhase.EVALUATE,
        "writing": ResearchPhase.WRITE,
        "verifying": ResearchPhase.VERIFY,
        "terminal": ResearchPhase.FINALIZE,
    }.get(phase, ResearchPhase.INIT)


def _stop_reason(reason: str | None) -> StopReason | None:
    if reason is None:
        return None
    if reason in {"quality_met", "completed"}:
        return StopReason.QUALITY_MET
    if reason == "completed_with_limitations":
        return StopReason.COMPLETED_WITH_LIMITATIONS
    if reason == "research_budget_exhausted":
        return StopReason.BUDGET_EXHAUSTED
    if reason == "stagnation":
        return StopReason.STAGNATION
    if reason == "sources_exhausted":
        return StopReason.SOURCES_EXHAUSTED
    if reason == "user_cancelled":
        return StopReason.CANCELLED
    if reason == "credentials_required":
        return StopReason.CREDENTIALS_REQUIRED
    return StopReason.FATAL_ERROR


def _gap_type(value: str) -> GapType:
    try:
        return GapType(value)
    except ValueError:
        return GapType.MISSING


def _gap_status(value: str) -> GapStatus:
    try:
        return GapStatus(value)
    except ValueError:
        return GapStatus.OPEN


def _question_uuid(run_id: UUID, question_id: str) -> UUID:
    return uuid5(run_id, f"question:{question_id}")


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))


def _state_hash(payload: dict[str, object]) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
