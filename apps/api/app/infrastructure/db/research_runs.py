"""Transactional Research Run, Outbox, and Agent Event Store."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.identifiers import uuid7
from app.domain.planning import (
    ResearchPlan,
    ResearchQuestion,
    append_dynamic_questions,
    build_gap_resolution_hints,
    fit_plan_to_budget,
)
from app.domain.providers import TokenUsage
from app.domain.research_budget import (
    build_resource_pool_snapshot,
    classify_question_risk,
    estimate_question_budgets,
    model_token_pool_limits,
)
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

_MODEL_TRANSPORT_REQUEUE_DELAYS_SECONDS = (30, 120, 600, 1800)
_MODEL_TRANSPORT_RETRY_PENDING = "model_transport_retry_pending"
_SEARCH_TRANSPORT_REQUEUE_DELAYS_SECONDS = (30, 120, 600, 1800)
_SEARCH_TRANSPORT_RETRY_PENDING = "search_transport_retry_pending"


def _reset_plan_scoped_question_budget(
    usage_snapshot: Mapping[str, object],
    *,
    reset_question_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Reset only per-question state explicitly reopened by a replan.

    A question-level budget exhaustion is real run state, not a plan-wide
    switch.  Carrying it across a replan is wrong for reopened gaps, while
    clearing the whole map lets unrelated questions re-enter repeatedly.
    """
    usage = dict(usage_snapshot)
    for field in (
        "question_budget_exhausted_by_question",
        "query_strategy_exhausted_by_question",
    ):
        values = usage.get(field)
        if isinstance(values, dict) and reset_question_ids:
            remaining = dict(values)
            for question_id in reset_question_ids:
                value = remaining.get(question_id)
                hard_exhaustion = (
                    field == "question_budget_exhausted_by_question"
                    and isinstance(value, dict)
                    and value.get("reason") == "hard_question_token_limit"
                )
                if not hard_exhaustion:
                    # Legacy boolean entries were produced by temporary
                    # fairness yields and are safe to clear for a targeted gap.
                    # Proven hard token limits survive every plan version.
                    remaining.pop(question_id, None)
            if remaining:
                usage[field] = remaining
            else:
                usage.pop(field, None)
    return usage


def _reset_replanned_quality_state(
    quality_snapshot: Mapping[str, object],
    *,
    reset_question_ids: tuple[str, ...],
    plan_version: int | None = None,
) -> dict[str, object]:
    """Reopen targeted risk states and clear only their low-gain history."""

    quality = dict(quality_snapshot)
    streaks = quality.get("low_information_gain_streak_by_question")
    if isinstance(streaks, dict):
        remaining = dict(streaks)
        for question_id in reset_question_ids:
            remaining.pop(question_id, None)
        quality["low_information_gain_streak_by_question"] = remaining
    quality["low_information_gain_streak"] = 0
    raw_questions = quality.get("risk_state_by_question", {})
    questions = dict(raw_questions) if isinstance(raw_questions, dict) else {}
    for question_id in reset_question_ids:
        raw_state = questions.get(question_id, {})
        state = dict(raw_state) if isinstance(raw_state, dict) else {}
        reasons_raw = state.get("reasons", [])
        reasons = list(reasons_raw) if isinstance(reasons_raw, list) else []
        if "replan_reopened" not in reasons:
            reasons.append("replan_reopened")
        state.update(
            {
                "gap_open": True,
                "lifecycle": "open",
                "borrow_eligible": bool(state.get("unresolved_high_risk", False)),
                "reasons": reasons,
            }
        )
        questions[question_id] = state
    quality["risk_state_by_question"] = questions
    raw_risk = quality.get("risk_state", {})
    risk = dict(raw_risk) if isinstance(raw_risk, dict) else {}
    risk["version"] = "claim_gap.v2"
    if plan_version is not None:
        risk["plan_version"] = plan_version
    risk["questions"] = questions
    summary_raw = risk.get("summary", {})
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
    summary["unresolved_questions"] = sum(
        bool(state.get("unresolved_high_risk"))
        for state in questions.values()
        if isinstance(state, dict)
    )
    summary["borrow_eligible_questions"] = sum(
        bool(state.get("borrow_eligible"))
        for state in questions.values()
        if isinstance(state, dict)
    )
    risk["summary"] = summary
    quality["risk_state"] = risk
    return quality


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

            allocation = budget_snapshot.get("allocation", {})
            allocation = allocation if isinstance(allocation, dict) else {}
            initial_model_pools: dict[str, object] = {
                name: {
                    "allocated_tokens": limit,
                    "committed_tokens": 0,
                    "reserved_tokens": 0,
                    "remaining_tokens": limit,
                    "status": "available" if limit > 0 else "disabled",
                }
                for name, limit in model_token_pool_limits(allocation).items()
            }
            initial_usage: dict[str, object] = {
                "model_token_pools": initial_model_pools,
            }
            initial_usage["resource_pools"] = build_resource_pool_snapshot(
                budget_snapshot,
                initial_usage,
                model_token_pools=initial_model_pools,
            )

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
                usage_snapshot=initial_usage,
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

    async def pause(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        """Pause a run without conflating it with cancellation or a report limit.

        Pending dispatches are revoked and an already-claimed worker is made a
        no-op by clearing its lease.  The run remains ``interrupted`` and can be
        resumed through the existing checkpoint path; no report is finalized.
        """
        async with self._sessions() as session, session.begin():
            run = await self._lock_run(session, owner_hash, run_id)
            current = RunStatus(run.status)
            if current in TERMINAL_RUN_STATUSES:
                return self._view(run)
            pending_dispatches = (
                await session.scalars(
                    select(TaskDispatchOutboxRow)
                    .where(
                        TaskDispatchOutboxRow.run_id == run.id,
                        TaskDispatchOutboxRow.status.in_(
                            ("pending", "retry", "publishing")
                        ),
                    )
                    .with_for_update()
                )
            ).all()
            for dispatch in pending_dispatches:
                dispatch.status = "cancelled"
                dispatch.last_error = "run_paused_by_user"
            now = datetime.now(UTC)
            run.status = RunStatus.INTERRUPTED.value
            run.termination_reason = "user_paused"
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.finished_at = None
            run.updated_at = now
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.paused",
                public_summary="研究任务已暂停; 当前 Checkpoint 将在恢复时继续。",
                refs={
                    "run_id": str(run.id),
                    "previous_status": current.value,
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
            if run is None:
                return False
            now = datetime.now(UTC)
            current_status = RunStatus(run.status)
            expired_takeover = (
                current_status == RunStatus.RUNNING
                and run.lease_until is not None
                and run.lease_until <= now
            )
            if current_status != RunStatus.QUEUED and not expired_takeover:
                return False
            run.status = RunStatus.RUNNING.value
            if not expired_takeover and RunPhase(run.phase) == RunPhase.INITIALIZING:
                run.phase = (
                    RunPhase.RESEARCHING.value if run.plan_version > 0 else RunPhase.PLANNING.value
                )
            # A run may be resumed after an older Worker already recorded more
            # Provider usage than the configured ceiling. Preserve that truthful
            # telemetry, but mark it guarded before State synchronization so the
            # graph can converge to deterministic limited-report writing without
            # issuing another model call.
            usage = dict(run.usage_snapshot)
            model_tokens = _model_token_total(usage)
            maximum_tokens = int(run.budget_snapshot.get("max_tokens", 0) or 0)
            if maximum_tokens > 0 and model_tokens >= maximum_tokens:
                usage["model_tokens"] = model_tokens
                usage["model_budget_guarded"] = True
            usage.pop("model_retry_after_seconds", None)
            usage.pop("model_retry_not_before", None)
            usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot, usage
            )
            run.usage_snapshot = usage
            if run.termination_reason in {
                _MODEL_TRANSPORT_RETRY_PENDING,
                _SEARCH_TRANSPORT_RETRY_PENDING,
            }:
                run.termination_reason = None
            run.started_at = run.started_at or now
            run.updated_at = now
            run.lease_owner = "celery-worker"
            run.lease_until = now + timedelta(seconds=lease_seconds)
            run.worker_task_id = worker_task_id
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="run.recovered" if expired_takeover else "run.started",
                public_summary=(
                    "检测到过期 Worker 租约; Celery 重投任务已接管并从 Checkpoint 恢复。"
                    if expired_takeover
                    else "后台 Worker 已领取任务并获得执行租约。"
                ),
                refs={
                    "run_id": str(run.id),
                    "phase": run.phase,
                    "recovered_from_expired_lease": expired_takeover,
                },
                metrics=None,
            )
            return True

    async def reconcile_expired_leases(self, *, limit: int = 100) -> list[UUID]:
        """Requeue runs whose worker lease expired without a terminal result.

        The reconciler is deliberately idempotent: the row lock prevents two
        reconcilers from creating duplicate resume dispatches, while the unique
        dispatch key makes a retry safe after a process crash.
        """

        if limit < 1:
            return []
        recovered: list[UUID] = []
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            rows = (
                await session.scalars(
                    select(ResearchRunRow)
                    .where(
                        ResearchRunRow.status == RunStatus.RUNNING.value,
                        ResearchRunRow.lease_until.is_not(None),
                        ResearchRunRow.lease_until <= now,
                    )
                    .order_by(ResearchRunRow.lease_until)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for run in rows:
                run.status = RunStatus.QUEUED.value
                run.phase = RunPhase.INITIALIZING.value
                run.termination_reason = "stale_worker_lease_recovered"
                run.lease_owner = None
                run.lease_until = None
                run.worker_task_id = None
                run.updated_at = now
                run.state_version += 1
                dispatch_key = f"{run.id}:reconcile:{run.state_version}"
                session.add(
                    TaskDispatchOutboxRow(
                        id=uuid7(),
                        run_id=run.id,
                        dispatch_type="resume",
                        dispatch_key=dispatch_key,
                        payload_ref={"run_id": str(run.id), "reason": "stale_lease"},
                        status="pending",
                    )
                )
                await self._append_event(
                    session,
                    run,
                    event_type="run.requeued",
                    public_summary="检测到过期 Worker 租约; 任务已重新排队等待 Checkpoint 恢复。",
                    refs={
                        "run_id": str(run.id),
                        "reason": "stale_worker_lease_recovered",
                        "dispatch_key": dispatch_key,
                    },
                    metrics=None,
                )
                recovered.append(run.id)
        return recovered

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
            plan = fit_plan_to_budget(
                plan,
                max_iterations=int(run.budget_snapshot.get("max_iterations", 0) or 0),
            )
            allocation = dict(run.budget_snapshot.get("allocation", {}))
            pool_limits = model_token_pool_limits(allocation)
            writer_reserve = pool_limits["writer"]
            research_pool = pool_limits["research"]
            question_budgets = estimate_question_budgets(
                [question.model_dump(mode="json") for question in plan.questions],
                max_tokens=research_pool,
            )
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
            allocation.update(
                {
                    "writer_tokens_initial": writer_reserve,
                    "question_token_budgets": {
                        item.question_id: item.target_tokens for item in question_budgets
                    },
                    "question_token_floors": {
                        item.question_id: item.minimum_tokens for item in question_budgets
                    },
                    "question_token_budget_total": sum(
                        item.target_tokens for item in question_budgets
                    ),
                    "budget_estimate_version": "question_weighted.v2",
                    "question_budget_plan_version": plan_version,
                }
            )
            run.budget_snapshot = {**run.budget_snapshot, "allocation": allocation}
            initial_risk_states = {
                question.id: classify_question_risk(
                    question_id=question.id,
                    priority=question.priority,
                    coverage=0.0,
                    requirements=question.evidence_requirements,
                    gap_open=True,
                    open_dimension_keys=tuple(
                        f"{question.id}:d{index}"
                        for index, _criterion in enumerate(
                            question.evidence_requirements, start=1
                        )
                    ),
                ).as_dict()
                for question in plan.questions
            }
            run.quality_snapshot = {
                **run.quality_snapshot,
                "risk_state": {
                    "version": "claim_gap.v2",
                    "plan_version": plan_version,
                    "questions": initial_risk_states,
                    "claims": {},
                    "gaps": {},
                    "conflicts": {},
                    "summary": {
                        "unresolved_questions": sum(
                            bool(state.get("unresolved_high_risk"))
                            for state in initial_risk_states.values()
                        ),
                        "unresolved_claims": 0,
                        "open_gaps": len(plan.questions),
                        "open_conflicts": 0,
                        "borrow_eligible_questions": sum(
                            bool(state.get("borrow_eligible"))
                            for state in initial_risk_states.values()
                        ),
                    },
                },
                "risk_state_by_question": initial_risk_states,
            }
            updated_usage = {
                **run.usage_snapshot,
                "planner": usage.model_dump(mode="json"),
            }
            updated_usage["model_tokens"] = _model_token_total(updated_usage)
            ledger_raw = updated_usage.get("budget_ledger", [])
            budget_ledger = list(ledger_raw) if isinstance(ledger_raw, list) else []
            budget_ledger.extend(
                [
                    {
                        "node": "planner",
                        "phase": "planning",
                        "allocated_tokens": pool_limits["planner"],
                        "actual_tokens": usage.total_tokens,
                        "status": "settled",
                    },
                    {
                        "node": "evidence_extractor",
                        "phase": "researching",
                        "allocated_tokens": research_pool,
                        "actual_tokens": 0,
                        "status": "available",
                    },
                    {
                        "node": "writer",
                        "phase": "writing",
                        "allocated_tokens": writer_reserve,
                        "actual_tokens": 0,
                        "status": "protected",
                    },
                    {
                        "node": "verification",
                        "phase": "writing",
                        "allocated_tokens": _safe_int(allocation.get("verification_tokens", 0)),
                        "actual_tokens": 0,
                        "status": "protected",
                    },
                    {
                        "node": "safety",
                        "phase": "all",
                        "allocated_tokens": _safe_int(allocation.get("safety_tokens", 0)),
                        "actual_tokens": 0,
                        "status": "protected",
                    },
                ]
            )
            updated_usage["budget_ledger"] = budget_ledger[-20:]
            updated_usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot,
                updated_usage,
            )
            run.usage_snapshot = updated_usage
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

    async def save_gap_resolution_plan(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        gaps: list[tuple[str, tuple[str, ...]]],
    ) -> bool:
        """Replan original questions in place so new evidence closes parent gaps."""

        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
                or run.plan_version < 1
            ):
                return False
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
            targets = {question_id: reasons for question_id, reasons in gaps[:3]}
            if not targets:
                return False
            plan_version = run.plan_version + 1
            reset_ids: list[str] = []
            for row in rows:
                requirements = [str(item) for item in row.evidence_requirements]
                hints = [str(item) for item in row.search_hints]
                status = row.status
                reasons = targets.get(row.question_id)
                if reasons is not None:
                    question = ResearchQuestion(
                        id=row.question_id,
                        question=row.question,
                        priority=row.priority,
                        rationale=row.rationale,
                        evidence_requirements=requirements,
                        search_hints=hints,
                    )
                    hints = build_gap_resolution_hints(question, reasons)
                    status = "pending"
                    reset_ids.append(row.question_id)
                session.add(
                    ResearchPlanItemRow(
                        id=uuid7(),
                        run_id=run.id,
                        plan_version=plan_version,
                        question_id=row.question_id,
                        question=row.question,
                        priority=row.priority,
                        rationale=row.rationale,
                        evidence_requirements=requirements,
                        search_hints=hints,
                        status=status,
                    )
                )
            if not reset_ids:
                return False
            run.plan_version = plan_version
            run.phase = RunPhase.RESEARCHING.value
            run.termination_reason = None
            run.quality_snapshot = _reset_replanned_quality_state(
                run.quality_snapshot,
                reset_question_ids=tuple(reset_ids),
                plan_version=plan_version,
            )
            usage = _reset_plan_scoped_question_budget(
                run.usage_snapshot,
                reset_question_ids=tuple(reset_ids),
            )
            replans = usage.get("replans", 0)
            usage["replans"] = (
                int(replans) + 1 if isinstance(replans, (int, float, str)) else 1
            )
            # A question-level budget yield belongs to the old plan version.
            # Replan creates fresh pending work for the selected gaps; carrying
            # the old map forward would make prepare_target skip those new
            # items and incorrectly conclude that sources are exhausted.
            usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot, usage
            )
            run.usage_snapshot = usage
            run.updated_at = datetime.now(UTC)
            run.state_version += 1
            await self._append_event(
                session,
                run,
                event_type="plan.revised",
                public_summary=(
                    f"REPLAN 已创建计划版本 {plan_version}; "
                    f"对 {len(reset_ids)} 个原问题执行定向补证。"
                ),
                refs={
                    "plan_version": plan_version,
                    "gap_resolution_question_ids": reset_ids,
                    "question_count": len(rows),
                    "coverage_denominator_changed": False,
                },
                metrics=None,
            )
            return True

    async def save_replanned_questions(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        additions: list[ResearchQuestion],
    ) -> bool:
        """Create a new immutable plan version while preserving completed question states."""

        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
                or run.plan_version < 1
            ):
                return False
            old_rows = (
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
            raw_completion = constraints.get("plan_completion_criteria", [])
            current = ResearchPlan(
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
                    for row in old_rows
                ],
                completion_criteria=(
                    [str(item) for item in raw_completion]
                    if isinstance(raw_completion, list)
                    else []
                ),
            )
            revised = append_dynamic_questions(current, additions)
            old_ids = {row.question_id for row in old_rows}
            added = [item for item in revised.questions if item.id not in old_ids]
            if not added:
                return False
            old_status = {row.question_id: row.status for row in old_rows}
            plan_version = run.plan_version + 1
            for question in revised.questions:
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
                        status=old_status.get(question.id, "pending"),
                    )
                )
            run.plan_version = plan_version
            run.phase = RunPhase.RESEARCHING.value
            run.termination_reason = None
            allocation = dict(run.budget_snapshot.get("allocation", {}))
            research_pool = model_token_pool_limits(allocation)["research"]
            question_budgets = estimate_question_budgets(
                [question.model_dump(mode="json") for question in revised.questions],
                max_tokens=research_pool,
            )
            allocation.update(
                {
                    "question_token_budgets": {
                        item.question_id: item.target_tokens for item in question_budgets
                    },
                    "question_token_floors": {
                        item.question_id: item.minimum_tokens for item in question_budgets
                    },
                    "question_token_budget_total": sum(
                        item.target_tokens for item in question_budgets
                    ),
                    "budget_estimate_version": "question_weighted.v3_rebalanced",
                    "question_budget_plan_version": plan_version,
                }
            )
            run.budget_snapshot = {**run.budget_snapshot, "allocation": allocation}
            updated_usage = _reset_plan_scoped_question_budget(
                run.usage_snapshot,
                reset_question_ids=tuple(question.id for question in additions),
            )
            updated_usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot, updated_usage
            )
            run.usage_snapshot = updated_usage
            quality = _reset_replanned_quality_state(
                run.quality_snapshot,
                reset_question_ids=tuple(question.id for question in added),
                plan_version=plan_version,
            )
            risk_by_question_raw = quality.get("risk_state_by_question", {})
            risk_by_question = (
                dict(risk_by_question_raw)
                if isinstance(risk_by_question_raw, dict)
                else {}
            )
            for question in added:
                risk_by_question[question.id] = classify_question_risk(
                    question_id=question.id,
                    priority=question.priority,
                    coverage=0.0,
                    requirements=question.evidence_requirements,
                    gap_open=True,
                    open_dimension_keys=tuple(
                        f"{question.id}:d{index}"
                        for index, _criterion in enumerate(
                            question.evidence_requirements, start=1
                        )
                    ),
                ).as_dict()
            quality["risk_state_by_question"] = risk_by_question
            risk_raw = quality.get("risk_state", {})
            risk = dict(risk_raw) if isinstance(risk_raw, dict) else {}
            risk["questions"] = risk_by_question
            risk["plan_version"] = plan_version
            risk["version"] = "claim_gap.v2"
            summary_raw = risk.get("summary", {})
            summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
            summary["unresolved_questions"] = sum(
                bool(state.get("unresolved_high_risk"))
                for state in risk_by_question.values()
                if isinstance(state, dict)
            )
            summary["borrow_eligible_questions"] = sum(
                bool(state.get("borrow_eligible"))
                for state in risk_by_question.values()
                if isinstance(state, dict)
            )
            risk["summary"] = summary
            quality["risk_state"] = risk
            run.quality_snapshot = quality
            run.updated_at = datetime.now(UTC)
            run.state_version += 1
            for question in added:
                await self._append_event(
                    session,
                    run,
                    event_type="plan.question_added",
                    public_summary=f"Evaluator 缺口触发动态问题 {question.id}.",
                    refs={
                        "plan_version": plan_version,
                        "question_id": question.id,
                        "question": question.question,
                        "created_reason": "evaluation_gap",
                    },
                    metrics=None,
                )
            await self._append_event(
                session,
                run,
                event_type="plan.revised",
                public_summary=(f"REPLAN 已创建计划版本 {plan_version}, 新增 {len(added)} 个问题."),
                refs={
                    "plan_version": plan_version,
                    "added_question_ids": [item.id for item in added],
                    "question_count": len(revised.questions),
                },
                metrics=None,
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

    async def defer_retryable_model_error(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        error_code: str,
        detail_code: str | None,
    ) -> int | None:
        """Persist a delayed retry after the short in-process retry window fails.

        Network outages often last longer than the Planner's bounded 3/10 second
        backoff. Turning that temporary condition into a terminal failed Run loses
        useful checkpoints and forces the user to resume manually. This transition
        releases the Worker lease and creates an Outbox message in the same
        transaction, so Dispatcher/Celery can retry after a wider recovery window.

        The returned integer is the selected delay in seconds. ``None`` means the
        retry budget was exhausted or the caller no longer owns the Run lease.
        """

        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
            ):
                return None

            usage = dict(run.usage_snapshot)
            retry_phase = run.phase
            previous_retry_phase = str(usage.get("model_transport_retry_phase", ""))
            completed_requeues = (
                max(0, int(usage.get("model_transport_requeues", 0) or 0))
                if previous_retry_phase == retry_phase
                else 0
            )
            if completed_requeues >= len(_MODEL_TRANSPORT_REQUEUE_DELAYS_SECONDS):
                return None

            delay_seconds = _MODEL_TRANSPORT_REQUEUE_DELAYS_SECONDS[completed_requeues]
            requeue_attempt = completed_requeues + 1
            now = datetime.now(UTC)
            retry_at = now + timedelta(seconds=delay_seconds)
            usage["model_transport_requeues"] = requeue_attempt
            usage["model_transport_retry_phase"] = retry_phase
            usage["model_retry_after_seconds"] = delay_seconds
            usage["model_retry_not_before"] = retry_at.isoformat()
            usage["last_model_error_code"] = error_code[:100]
            usage["last_model_detail_code"] = (detail_code or "UNKNOWN")[:100]
            # A model error may occur after a page slot was reserved but before
            # the page result was committed. A new Worker must never inherit it.
            usage["page_slots_reserved"] = 0
            usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot, usage
            )
            run.usage_snapshot = usage
            run.status = RunStatus.QUEUED.value
            run.termination_reason = _MODEL_TRANSPORT_RETRY_PENDING
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.finished_at = None
            run.updated_at = now
            run.state_version += 1

            dispatch_key = f"{run.id}:model-retry:{run.state_version}"
            session.add(
                TaskDispatchOutboxRow(
                    id=uuid7(),
                    run_id=run.id,
                    dispatch_type="resume",
                    dispatch_key=dispatch_key,
                    payload_ref={
                        "run_id": str(run.id),
                        "reason": _MODEL_TRANSPORT_RETRY_PENDING,
                    },
                    status="pending",
                    next_attempt_at=retry_at,
                )
            )
            await self._append_event(
                session,
                run,
                event_type="model.retry_deferred",
                public_summary=(
                    "模型网络在短时重试窗口内仍不可用; "
                    f"任务状态已保留, 将在 {delay_seconds} 秒后自动恢复。"
                ),
                refs={
                    "error_code": error_code[:100],
                    "detail_code": (detail_code or "UNKNOWN")[:100],
                    "requeue_attempt": requeue_attempt,
                    "max_requeue_attempts": len(_MODEL_TRANSPORT_REQUEUE_DELAYS_SECONDS),
                    "dispatch_key": dispatch_key,
                },
                metrics={"delay_seconds": delay_seconds},
            )
            return delay_seconds

    async def defer_retryable_search_error(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        error_code: str,
        detail_code: str | None,
    ) -> int | None:
        """Persist a delayed resume when the shared Search Provider is unavailable.

        Individual webpage failures remain isolated by the research loop. This
        path is only used after SearXNG's internal strategies and circuit breaker
        have reported a retryable run-wide dependency failure.
        """

        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
            )
            if (
                run is None
                or RunStatus(run.status) != RunStatus.RUNNING
                or run.worker_task_id != worker_task_id
            ):
                return None

            usage = dict(run.usage_snapshot)
            retry_phase = run.phase
            previous_retry_phase = str(usage.get("search_transport_retry_phase", ""))
            completed_requeues = (
                max(0, int(usage.get("search_transport_requeues", 0) or 0))
                if previous_retry_phase == retry_phase
                else 0
            )
            if completed_requeues >= len(_SEARCH_TRANSPORT_REQUEUE_DELAYS_SECONDS):
                return None

            delay_seconds = _SEARCH_TRANSPORT_REQUEUE_DELAYS_SECONDS[completed_requeues]
            requeue_attempt = completed_requeues + 1
            now = datetime.now(UTC)
            retry_at = now + timedelta(seconds=delay_seconds)
            usage["search_transport_requeues"] = requeue_attempt
            usage["search_transport_retry_phase"] = retry_phase
            usage["search_retry_after_seconds"] = delay_seconds
            usage["search_retry_not_before"] = retry_at.isoformat()
            usage["last_search_error_code"] = error_code[:100]
            usage["last_search_detail_code"] = (detail_code or "UNKNOWN")[:100]
            usage["page_slots_reserved"] = 0
            usage = _record_search_transport_retry(usage)
            usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot, usage
            )
            run.usage_snapshot = usage
            run.status = RunStatus.QUEUED.value
            run.termination_reason = _SEARCH_TRANSPORT_RETRY_PENDING
            run.lease_owner = None
            run.lease_until = None
            run.worker_task_id = None
            run.finished_at = None
            run.updated_at = now
            run.state_version += 1

            dispatch_key = f"{run.id}:search-retry:{run.state_version}"
            session.add(
                TaskDispatchOutboxRow(
                    id=uuid7(),
                    run_id=run.id,
                    dispatch_type="resume",
                    dispatch_key=dispatch_key,
                    payload_ref={
                        "run_id": str(run.id),
                        "reason": _SEARCH_TRANSPORT_RETRY_PENDING,
                    },
                    status="pending",
                    next_attempt_at=retry_at,
                )
            )
            await self._append_event(
                session,
                run,
                event_type="search.retry_deferred",
                public_summary=(
                    "搜索服务在内部回退后仍暂时不可用; "
                    f"任务状态已保留, 将在 {delay_seconds} 秒后自动恢复。"
                ),
                refs={
                    "error_code": error_code[:100],
                    "detail_code": (detail_code or "UNKNOWN")[:100],
                    "requeue_attempt": requeue_attempt,
                    "max_requeue_attempts": len(_SEARCH_TRANSPORT_REQUEUE_DELAYS_SECONDS),
                    "dispatch_key": dispatch_key,
                },
                metrics={"delay_seconds": delay_seconds},
            )
            return delay_seconds

    async def fail_execution(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        error_code: str = "WORKER_EXECUTION_FAILED",
        detail_code: str | None = None,
        diagnostics: Mapping[str, str | int] | None = None,
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
            failure_phase = run.phase
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
            # A failed task must not leave a batch reservation blocking a later
            # explicit resume. Committed usage remains untouched and truthful.
            usage = dict(run.usage_snapshot)
            usage["page_slots_reserved"] = 0
            usage["resource_pools"] = build_resource_pool_snapshot(
                run.budget_snapshot, usage
            )
            run.usage_snapshot = usage
            run.state_version += 1
            allowed_diagnostics = {
                "structured_output_strategy",
                "finish_reason",
                "output_tokens",
                "max_output_tokens",
                "response_length",
                "provider_request_id",
                "retry_mode",
                "first_error_code",
                "attempt_stage",
                "compact_trigger",
                "failure_node",
                "budget_type",
                "budget_limit",
                "budget_used",
                "budget_overrun",
            }
            safe_diagnostics = {
                key: value[:200] if isinstance(value, str) else value
                for key, value in (diagnostics or {}).items()
                if key in allowed_diagnostics and isinstance(value, str | int)
            }
            safe_diagnostics.setdefault("failure_node", failure_phase[:100])
            if detail_code == "PAGE_BUDGET_OVERRUN":
                limit = int(run.budget_snapshot.get("max_pages", 0) or 0)
                used = int(run.usage_snapshot.get("pages", 0) or 0)
                safe_diagnostics.update(
                    {
                        "failure_node": "research_iteration",
                        "budget_type": "pages",
                        "budget_limit": limit,
                        "budget_used": used,
                        "budget_overrun": max(0, used - limit),
                    }
                )
            await self._append_event(
                session,
                run,
                event_type="run.failed",
                public_summary="模型调用或后台执行失败 - 敏感异常信息未写入公开事件。",
                refs={
                    "reason": safe_code[:100],
                    "detail_code": (detail_code or "UNKNOWN")[:100],
                    "model_diagnostics": safe_diagnostics,
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


def _model_token_total(usage: dict[str, object]) -> int:
    total = _safe_int(usage.get("evidence_total_tokens", 0))
    for key in ("planner", "writer"):
        item = usage.get(key)
        if isinstance(item, dict):
            total += _safe_int(item.get("total_tokens", 0))
    return total


def _record_search_transport_retry(usage: dict[str, object]) -> dict[str, object]:
    """Count a durable provider recovery as a technical retry.

    Retryable search failures bypass the normal iteration evaluator, so this
    counter must be updated at the durable requeue boundary.
    """

    updated = dict(usage)
    updated["technical_retries"] = _safe_int(updated.get("technical_retries", 0)) + 1
    return updated


def _safe_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
