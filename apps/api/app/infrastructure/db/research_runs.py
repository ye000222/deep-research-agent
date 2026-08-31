"""Transactional Research Run, Outbox, and Agent Event Store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.domain.providers import TokenUsage
from app.domain.research_runs import (
    TERMINAL_RUN_STATUSES,
    AgentEventView,
    ResearchRunView,
    RunPhase,
    RunStatus,
)
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow
from app.infrastructure.db.run_models import (
    AgentEventRow,
    ResearchPlanItemRow,
    ResearchRunRow,
    TaskDispatchOutboxRow,
)


class ResearchRunNotFoundError(LookupError):
    pass


class CredentialVersionNotFoundError(LookupError):
    pass


class ResearchPlanNotFoundError(LookupError):
    pass


class InvalidRunTransitionError(ValueError):
    pass


class ResearchRunRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        owner_hash: str,
        *,
        idempotency_key: str,
        original_query: str,
        normalized_goal: str,
        credential_version_id: UUID,
        budget_snapshot: dict[str, object],
    ) -> tuple[ResearchRunView, bool]:
        async with self._sessions() as session, session.begin():
            existing = await session.scalar(
                select(ResearchRunRow).where(
                    ResearchRunRow.owner_hash == owner_hash,
                    ResearchRunRow.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                return self._view(existing), False

            binding = (
                (
                    await session.execute(
                        select(ProviderProfileRow, CredentialVersionRow)
                        .join(
                            CredentialVersionRow,
                            CredentialVersionRow.profile_id == ProviderProfileRow.id,
                        )
                        .where(
                            ProviderProfileRow.owner_hash == owner_hash,
                            ProviderProfileRow.status == "active",
                            ProviderProfileRow.deleted_at.is_(None),
                            CredentialVersionRow.id == credential_version_id,
                            CredentialVersionRow.revoked_at.is_(None),
                            CredentialVersionRow.deleted_at.is_(None),
                        )
                    )
                )
                .tuples()
                .first()
            )
            if binding is None:
                raise CredentialVersionNotFoundError(str(credential_version_id))
            profile, credential = binding

            run = ResearchRunRow(
                id=uuid7(),
                owner_hash=owner_hash,
                idempotency_key=idempotency_key,
                original_query=original_query,
                normalized_goal=normalized_goal,
                status=RunStatus.QUEUED.value,
                phase=RunPhase.INITIALIZING.value,
                credential_status="ready",
                saved_profile_id=profile.id,
                credential_version_id=credential.id,
                llm_config_snapshot={
                    "adapter_type": profile.adapter_type,
                    "base_url": profile.normalized_base_url,
                    "endpoint_host": profile.endpoint_host,
                    "model": profile.model,
                    "profile_version": profile.version,
                    "credential_version": credential.credential_version,
                    "context_window": profile.non_secret_settings.get("context_window"),
                    "max_output_tokens": profile.non_secret_settings.get("max_output_tokens"),
                },
                budget_snapshot=budget_snapshot,
            )
            session.add(run)
            await session.flush()
            await self._append_event(
                session,
                run,
                event_type="run.created",
                public_summary="Research run accepted and queued.",
                refs={"run_id": str(run.id)},
                metrics=None,
            )
            session.add(
                TaskDispatchOutboxRow(
                    id=uuid7(),
                    run_id=run.id,
                    dispatch_type="start",
                    dispatch_key=f"{run.id}:start:1",
                    payload_ref={"run_id": str(run.id)},
                    status="pending",
                )
            )
            await session.flush()
        return self._view(run), True

    async def list_recent(self, owner_hash: str, *, limit: int) -> list[ResearchRunView]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ResearchRunRow)
                    .where(ResearchRunRow.owner_hash == owner_hash)
                    .order_by(ResearchRunRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
            return [self._view(row) for row in rows]

    async def get(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        async with self._sessions() as session:
            run = await session.scalar(
                select(ResearchRunRow).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if run is None:
                raise ResearchRunNotFoundError(str(run_id))
            return self._view(run)

    async def cancel(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        async with self._sessions() as session, session.begin():
            run = await self._lock_run(session, owner_hash, run_id)
            current = RunStatus(run.status)
            if current in TERMINAL_RUN_STATUSES:
                return self._view(run)

            # Mark unpublished dispatches as cancelled in the same transaction.
            # A dispatcher that already claimed a message may still publish it,
            # but the worker lease check below will make that task a no-op.
            pending_dispatches = (
                await session.scalars(
                    select(TaskDispatchOutboxRow)
                    .where(
                        TaskDispatchOutboxRow.run_id == run.id,
                        TaskDispatchOutboxRow.status.in_(("pending", "retry", "publishing")),
                    )
                    .with_for_update()
                )
            ).all()
            for dispatch in pending_dispatches:
                dispatch.status = "cancelled"
                dispatch.last_error = "run_cancelled_by_user"

            previous_status = current.value
            now = datetime.now(UTC)
            run.status = RunStatus.CANCELLED.value
            run.phase = RunPhase.TERMINAL.value
            run.termination_reason = "user_cancelled"
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.finished_at = now
            run.updated_at = now
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.cancelled",
                public_summary=(
                    "Research run cancelled by the user; pending dispatches were revoked."
                ),
                refs={
                    "run_id": str(run.id),
                    "previous_status": previous_status,
                    "cancelled_dispatches": len(pending_dispatches),
                },
                metrics={"cancelled_dispatches": len(pending_dispatches)},
            )
            await session.flush()
        return self._view(run)

    async def resume(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        async with self._sessions() as session, session.begin():
            run = await self._lock_run(session, owner_hash, run_id)
            current = RunStatus(run.status)
            if current == RunStatus.COMPLETED:
                raise InvalidRunTransitionError("completed runs cannot be resumed")
            if current not in {
                RunStatus.CANCELLED,
                RunStatus.FAILED,
                RunStatus.INTERRUPTED,
            }:
                return self._view(run)
            run.status = RunStatus.QUEUED.value
            run.phase = RunPhase.INITIALIZING.value
            run.termination_reason = None
            run.finished_at = None
            run.updated_at = datetime.now(UTC)
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.status_changed",
                public_summary="Research run queued for resume.",
                refs={"run_id": str(run.id), "status": RunStatus.QUEUED.value},
                metrics=None,
            )
            session.add(
                TaskDispatchOutboxRow(
                    id=uuid7(),
                    run_id=run.id,
                    dispatch_type="resume",
                    dispatch_key=f"{run.id}:resume:{run.state_version}",
                    payload_ref={"run_id": str(run.id)},
                    status="pending",
                )
            )
            await session.flush()
        return self._view(run)

    async def list_events(
        self,
        owner_hash: str,
        run_id: UUID,
        *,
        after_seq: int,
        limit: int = 200,
    ) -> list[AgentEventView]:
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
                    select(AgentEventRow)
                    .where(
                        AgentEventRow.run_id == run_id,
                        AgentEventRow.run_seq > after_seq,
                    )
                    .order_by(AgentEventRow.run_seq)
                    .limit(limit)
                )
            ).all()
            return [self._event_view(row) for row in rows]

    async def get_plan(self, owner_hash: str, run_id: UUID) -> ResearchPlan:
        async with self._sessions() as session:
            run = await session.scalar(
                select(ResearchRunRow).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if run is None:
                raise ResearchRunNotFoundError(str(run_id))
            if run.plan_version < 1:
                raise ResearchPlanNotFoundError(str(run_id))
            rows = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                    )
                    .order_by(
                        ResearchPlanItemRow.priority,
                        ResearchPlanItemRow.question_id,
                    )
                )
            ).all()
            constraints = run.constraints
            completion = constraints.get("plan_completion_criteria", [])
            return ResearchPlan(
                goal=str(constraints.get("plan_goal", run.normalized_goal)),
                scope_summary=str(constraints.get("plan_scope_summary", run.normalized_goal)),
                questions=[
                    ResearchQuestion(
                        id=row.question_id,
                        question=row.question,
                        priority=row.priority,
                        rationale=row.rationale,
                        evidence_requirements=row.evidence_requirements,
                        search_hints=row.search_hints,
                    )
                    for row in rows
                ],
                completion_criteria=[str(item) for item in completion]
                if isinstance(completion, list)
                else [],
            )

    async def get_plan_for_execution(self, run_id: UUID) -> ResearchPlan | None:
        async with self._sessions() as session:
            run = await session.get(ResearchRunRow, run_id)
            if run is None or run.plan_version < 1:
                return None
            rows = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                    )
                    .order_by(ResearchPlanItemRow.priority, ResearchPlanItemRow.question_id)
                )
            ).all()
            constraints = run.constraints
            completion = constraints.get("plan_completion_criteria", [])
            return ResearchPlan(
                goal=str(constraints.get("plan_goal", run.normalized_goal)),
                scope_summary=str(constraints.get("plan_scope_summary", run.normalized_goal)),
                questions=[
                    ResearchQuestion(
                        id=row.question_id,
                        question=row.question,
                        priority=row.priority,
                        rationale=row.rationale,
                        evidence_requirements=row.evidence_requirements,
                        search_hints=row.search_hints,
                    )
                    for row in rows
                ],
                completion_criteria=[str(item) for item in completion]
                if isinstance(completion, list)
                else [],
            )

    async def acquire_for_execution(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        lease_seconds: int = 300,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if run is None or RunStatus(run.status) != RunStatus.QUEUED:
                return False
            now = datetime.now(UTC)
            run.status = RunStatus.RUNNING.value
            run.phase = (
                RunPhase.RESEARCHING.value if run.plan_version > 0 else RunPhase.PLANNING.value
            )
            run.started_at = run.started_at or now
            run.updated_at = now
            run.lease_owner = "celery-worker"
            run.lease_until = now + timedelta(seconds=lease_seconds)
            run.worker_task_id = worker_task_id
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.started",
                public_summary="后台 Worker 已领取任务并获得执行租约。",
                refs={"run_id": str(run.id), "phase": run.phase},
                metrics=None,
            )
            return True

    async def save_generated_plan(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        plan: ResearchPlan,
        usage: TokenUsage,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
            ):
                return False
            plan_version = run.plan_version + 1
            for question in plan.questions:
                session.add(
                    ResearchPlanItemRow(
                        id=uuid7(),
                        run_id=run.id,
                        plan_version=plan_version,
                        question_id=question.id,
                        question=question.question,
                        priority=question.priority,
                        rationale=question.rationale,
                        evidence_requirements=question.evidence_requirements,
                        search_hints=question.search_hints,
                        status="pending",
                    )
                )
            run.plan_version = plan_version
            run.constraints = {
                **run.constraints,
                "plan_goal": plan.goal,
                "plan_scope_summary": plan.scope_summary,
                "plan_completion_criteria": plan.completion_criteria,
            }
            run.usage_snapshot = {
                **run.usage_snapshot,
                "planner": usage.model_dump(mode="json"),
            }
            run.phase = RunPhase.RESEARCHING.value
            run.termination_reason = None
            run.updated_at = datetime.now(UTC)
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="plan.generated",
                public_summary=f"Planner 已生成 {len(plan.questions)} 个可验证研究问题。",
                refs={
                    "plan_version": plan_version,
                    "questions": [question.question for question in plan.questions],
                    "question_ids": [question.id for question in plan.questions],
                },
                metrics={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
            )

            return True

    async def interrupt_for_pending_planner(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
            ):
                return
            run.status = RunStatus.INTERRUPTED.value
            run.phase = RunPhase.PLANNING.value
            run.termination_reason = "planner_not_implemented"
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.updated_at = datetime.now(UTC)
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.interrupted",
                public_summary="执行基础设施已验证 - Planner 将在下一开发阶段接入。",
                refs={"reason": "planner_not_implemented"},
                metrics=None,
            )

    async def record_model_retry(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        role: str,
        failed_attempt: int,
        max_attempts: int,
        error_code: str,
        detail_code: str | None,
        delay_seconds: float,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
            ):
                return False
            now = datetime.now(UTC)
            run.updated_at = now
            run.lease_until = now + timedelta(seconds=300)
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="model.retry_scheduled",
                public_summary=(
                    f"{role.title()} 遇到临时模型错误"
                    f" ({detail_code or 'UNKNOWN'}); "
                    f"{delay_seconds:g} 秒后自动重试。"
                ),
                refs={
                    "role": role,
                    "failed_attempt": failed_attempt,
                    "next_attempt": failed_attempt + 1,
                    "max_attempts": max_attempts,
                    "error_code": error_code[:100],
                    "detail_code": (detail_code or "UNKNOWN")[:100],
                },
                metrics={"delay_seconds": delay_seconds},
            )
            return True

    async def fail_execution(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        error_code: str = "WORKER_EXECUTION_FAILED",
        detail_code: str | None = None,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
            ):
                return
            run.status = RunStatus.FAILED.value
            run.phase = RunPhase.TERMINAL.value
            safe_code = error_code if error_code.isupper() else "WORKER_EXECUTION_FAILED"
            run.termination_reason = safe_code[:100]
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            now = datetime.now(UTC)
            run.updated_at = now
            run.finished_at = now
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.failed",
                public_summary="模型调用或后台执行失败 - 敏感异常信息未写入公开事件。",
                refs={
                    "reason": safe_code[:100],
                    "detail_code": (detail_code or "UNKNOWN")[:100],
                },
                metrics=None,
            )

    @staticmethod
    async def _append_event(
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        event_type: str,
        public_summary: str,
        refs: dict[str, object],
        metrics: dict[str, object] | None,
    ) -> AgentEventRow:
        seq = run.next_event_seq
        run.next_event_seq += 1
        event = AgentEventRow(
            run_id=run.id,
            run_seq=seq,
            schema_version=1,
            phase=run.phase,
            event_type=event_type,
            public_summary=public_summary,
            refs=refs,
            metrics=metrics,
        )
        session.add(event)
        await session.flush()
        return event

    @staticmethod
    async def _lock_run(session: AsyncSession, owner_hash: str, run_id: UUID) -> ResearchRunRow:
        run = await session.scalar(
            select(ResearchRunRow)
            .where(
                ResearchRunRow.id == run_id,
                ResearchRunRow.owner_hash == owner_hash,
            )
            .with_for_update()
        )
        if run is None:
            raise ResearchRunNotFoundError(str(run_id))
        return run

    @staticmethod
    def _view(row: ResearchRunRow) -> ResearchRunView:
        return ResearchRunView(
            run_id=row.id,
            original_query=row.original_query,
            normalized_goal=row.normalized_goal,
            status=RunStatus(row.status),
            phase=RunPhase(row.phase),
            state_version=row.state_version,
            plan_version=row.plan_version,
            next_event_seq=row.next_event_seq,
            credential_status=row.credential_status,
            saved_profile_id=row.saved_profile_id,
            credential_version_id=row.credential_version_id,
            llm_config_snapshot=row.llm_config_snapshot,
            budget_snapshot=row.budget_snapshot,
            usage_snapshot=row.usage_snapshot,
            quality_snapshot=row.quality_snapshot,
            termination_reason=row.termination_reason,
            created_at=row.created_at,
            updated_at=row.updated_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )

    @staticmethod
    def _event_view(row: AgentEventRow) -> AgentEventView:
        return AgentEventView(
            global_id=row.global_id,
            run_id=row.run_id,
            seq=row.run_seq,
            schema_version=row.schema_version,
            timestamp=row.created_at,
            phase=row.phase,
            event_type=row.event_type,
            public_summary=row.public_summary,
            refs=row.refs,
            metrics=row.metrics,
        )
