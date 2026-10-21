"""Durable V3 Work scheduling over the shared ResearchLoop executor."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from app.domain.research_work import (
    LegacyResearchObjective,
    PropositionGapResearchObjective,
    ResearchWork,
    ResearchWorkKind,
    WorkAttempt,
    WorkAttemptState,
)
from app.domain.run_state import (
    GapProjectionStatus,
    ResearchRunState,
    SemanticExecutionStatus,
)
from app.domain.semantic_control_projection import (
    SemanticGapControlArtifact,
    SemanticIntentControlArtifact,
)
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.research_work import ResearchWorkRepository
from app.services.research_loop import ResearchIterationResult, ResearchLoopService
from app.services.run_state_publisher import RunStatePublisher

ResearchExecutorKind = Literal["V2_LEGACY", "V3"]


def research_executor_for_version(version: str) -> ResearchExecutorKind:
    """Pure dispatch contract shared by the graph and deterministic tests."""

    if version == "V2_LEGACY":
        return "V2_LEGACY"
    if version == "V3":
        return "V3"
    raise ValueError("unsupported_orchestrator_version")


class ResearchOrchestrator:
    """Selects and reserves V3 work; it never evaluates semantic truth."""

    def __init__(
        self,
        *,
        runs: ResearchRunRepository,
        work: ResearchWorkRepository,
        run_states: RunStatePublisher,
        loop: ResearchLoopService,
        tools: ResearchToolRepository,
        semantic_recovery: object | None = None,
        gap_work_scheduling_enabled: bool = False,
    ) -> None:
        self._runs = runs
        self._work = work
        self._run_states = run_states
        self._loop = loop
        self._tools = tools
        self._semantic_recovery = semantic_recovery
        self._gap_work_scheduling_enabled = gap_work_scheduling_enabled

    async def run_next_work(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> ResearchIterationResult:
        inputs = await self._runs.get_run_state_inputs(run_id)
        if research_executor_for_version(str(inputs["orchestrator_version"])) != "V3":
            raise ValueError("research_orchestrator_requires_v3_run")
        run_lease_token = inputs.get("lease_owner")
        if not isinstance(run_lease_token, str) or not run_lease_token:
            raise RuntimeError("research_execution_lease_identity_unavailable")
        state = await self._run_states.load_current(run_id)
        if state is None:
            raise RuntimeError("canonical_v3_run_state_unavailable")
        plan = await self._runs.get_plan_for_execution(run_id)
        if plan is None:
            raise RuntimeError("v3_legacy_plan_unavailable")

        await self._work.materialize_legacy_plan(state, plan)
        execution_owner_token = uuid4().hex
        reservation_identity = f"v3:{run_id}:{worker_task_id}:{execution_owner_token}"
        reserved = await self._work.reserve_next(
            run_id,
            run_state=state,
            reservation_identity=reservation_identity,
        )
        gap_projection: tuple[
            SemanticGapControlArtifact, SemanticIntentControlArtifact
        ] | None = None
        if reserved is None and self._gap_work_scheduling_enabled:
            state, gap_projection, reserved = await self._reserve_gap_work(
                run_id,
                worker_task_id=worker_task_id,
                state=state,
                reservation_identity=reservation_identity,
            )
        if reserved is None:
            replan_requested = await self._tools.replan_requested(
                run_id,
                worker_task_id=worker_task_id,
            )
            decision = "replan" if replan_requested else "no_pending_question"
            return ResearchIterationResult(
                outcome=f"research_stopped:{decision}:iterations=0:pages=0:accepted=0",
                continue_research=replan_requested,
                decision=decision,
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )

        work_item, attempt = reserved
        if not await self._work.claim_attempt_execution_ownership(
            run_id,
            attempt.attempt_ref,
            worker_task_id=worker_task_id,
            run_lease_token=run_lease_token,
            execution_owner_token=execution_owner_token,
        ):
            return ResearchIterationResult(
                outcome="research_waiting:work_already_owned",
                continue_research=True,
                decision="work_already_owned",
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )

        async def settle_owned_attempt(
            *,
            state: WorkAttemptState,
            query_refs: tuple[UUID, ...] = (),
            tool_call_refs: tuple[UUID, ...] = (),
            produced_evidence_refs: tuple[UUID, ...] = (),
            reused_evidence_refs: tuple[UUID, ...] = (),
            budget_refs: tuple[str, ...] = (),
            failure_classification: str | None = None,
            usage_summary: dict[str, object] | None = None,
        ) -> None:
            await self._work.settle_attempt(
                attempt.attempt_ref,
                run_ref=run_id,
                worker_task_id=worker_task_id,
                run_lease_token=run_lease_token,
                execution_owner_token=execution_owner_token,
                state=state,
                query_refs=query_refs,
                tool_call_refs=tool_call_refs,
                produced_evidence_refs=produced_evidence_refs,
                reused_evidence_refs=reused_evidence_refs,
                budget_refs=budget_refs,
                failure_classification=failure_classification,
                usage_summary=usage_summary,
            )

        is_gap_work = isinstance(work_item.objective, PropositionGapResearchObjective)
        objective_kind_matches = (
            is_gap_work
            and work_item.work_kind is ResearchWorkKind.PROPOSITION_GAP
        ) or (
            not is_gap_work
            and work_item.work_kind is ResearchWorkKind.LEGACY_RESEARCH
        )
        if not objective_kind_matches:
            await settle_owned_attempt(
                state=WorkAttemptState.BLOCKED,
                failure_classification="WORK_OBJECTIVE_KIND_MISMATCH",
            )
            raise RuntimeError("v3_work_objective_kind_mismatch")
        if not is_gap_work and not isinstance(work_item.objective, LegacyResearchObjective):
            await settle_owned_attempt(
                state=WorkAttemptState.BLOCKED,
                failure_classification="UNSUPPORTED_WORK_OBJECTIVE_KIND",
            )
            raise RuntimeError("legacy_orchestrator_rejected_non_legacy_objective")
        try:
            if is_gap_work:
                if gap_projection is None:
                    raise RuntimeError("gap_work_authority_projection_unavailable")
                gap_artifact, intent_artifact = gap_projection
                target = await self._tools.prepare_target_for_work(
                    run_id,
                    worker_task_id=worker_task_id,
                    work_item=work_item,
                    attempt=attempt,
                    run_state=state,
                    gap_artifact=gap_artifact,
                    intent_artifact=intent_artifact,
                )
            else:
                assert isinstance(work_item.objective, LegacyResearchObjective)
                target = await self._tools.prepare_target_for_work(
                    run_id,
                    worker_task_id=worker_task_id,
                    question_id=work_item.objective.question_id,
                    work_item=work_item,
                    attempt=attempt,
                )
        except Exception:
            await settle_owned_attempt(
                state=WorkAttemptState.BLOCKED,
                failure_classification="TARGET_PREPARATION_FAILED",
            )
            raise
        if target is None:
            await settle_owned_attempt(
                state=WorkAttemptState.BLOCKED,
                failure_classification="BLOCKED_TARGET_UNAVAILABLE",
            )
            return ResearchIterationResult(
                outcome="work_blocked:target_unavailable",
                continue_research=True,
                decision="work_blocked",
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )

        if (
            isinstance(work_item.objective, LegacyResearchObjective)
            and target.question_id != work_item.objective.question_id
        ):
            await settle_owned_attempt(
                state=WorkAttemptState.BLOCKED,
                failure_classification="BLOCKED_TARGET_IDENTITY_MISMATCH",
            )
            raise RuntimeError("v3_work_target_identity_mismatch")

        if attempt.state is WorkAttemptState.RESERVED:
            await self._work.mark_attempt_executing(
                attempt.attempt_ref,
                run_ref=run_id,
                worker_task_id=worker_task_id,
                run_lease_token=run_lease_token,
                execution_owner_token=execution_owner_token,
            )
        if not await self._work.assert_attempt_execution_ownership(
            run_id,
            attempt.attempt_ref,
            worker_task_id=worker_task_id,
            run_lease_token=run_lease_token,
            execution_owner_token=execution_owner_token,
        ):
            return ResearchIterationResult(
                outcome="research_waiting:execution_ownership_lost",
                continue_research=True,
                decision="execution_ownership_lost",
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )
        try:
            result = await self._loop.execute_target(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
            )
        except Exception:
            # Never persist raw exception text: it may contain credentials or
            # provider request data. Attempt identity and lifecycle are enough
            # for the bounded retry seam in this batch.
            if not await self._work.assert_attempt_execution_ownership(
                run_id,
                attempt.attempt_ref,
                worker_task_id=worker_task_id,
                run_lease_token=run_lease_token,
                execution_owner_token=execution_owner_token,
            ):
                return ResearchIterationResult(
                    outcome="research_waiting:execution_ownership_lost",
                    continue_research=True,
                    decision="execution_ownership_lost",
                    pages_read=0,
                    accepted_evidence=0,
                    coverage=0.0,
                    information_gain=0.0,
                    low_information_gain_streak=0,
                )
            await self._tools.project_work_failure(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                failure_classification="EXECUTION_ERROR",
            )
            await settle_owned_attempt(
                state=WorkAttemptState.FAILED,
                query_refs=(target.query_execution_id,)
                if target.query_execution_id is not None
                else (),
                tool_call_refs=(target.tool_call_id,),
                failure_classification="EXECUTION_ERROR",
            )
            if is_gap_work:
                await self._recover_gap_semantics(run_id)
            return ResearchIterationResult(
                outcome="work_execution_failed",
                continue_research=True,
                decision="work_retryable",
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )

        if not await self._work.assert_attempt_execution_ownership(
            run_id,
            attempt.attempt_ref,
            worker_task_id=worker_task_id,
            run_lease_token=run_lease_token,
            execution_owner_token=execution_owner_token,
        ):
            return ResearchIterationResult(
                outcome="research_waiting:execution_ownership_lost",
                continue_research=True,
                decision="execution_ownership_lost",
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )
        evaluation = await self._loop.project_work_iteration(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
            attempt_outcome=result.outcome,
        )
        await settle_owned_attempt(
            state=WorkAttemptState.COMPLETED,
            query_refs=(target.query_execution_id,)
            if target.query_execution_id is not None
            else (),
            tool_call_refs=(target.tool_call_id,),
            produced_evidence_refs=result.evidence_refs,
            reused_evidence_refs=result.reused_evidence_refs,
            usage_summary={
                "budget_scope": "RUN_LEVEL",
                "work_budget_attribution": dict(work_item.budget_attribution),
                "execution_bounds": dict(work_item.execution_bounds),
                "outcome": result.outcome,
                "pages_read": result.pages_read,
                "accepted_evidence": result.accepted_evidence,
            },
        )
        if is_gap_work:
            # Completion only records the Attempt.  The same stable Work is
            # reconciled against a newly authoritative semantic projection.
            await self._recover_gap_semantics(run_id)
            return ResearchIterationResult(
                outcome=(
                    f"research_stopped:gap_reassessment_pending:iterations=1:"
                    f"pages={result.pages_read}:accepted={result.accepted_evidence}"
                ),
                continue_research=True,
                decision="gap_reassessment_pending",
                pages_read=result.pages_read,
                accepted_evidence=result.accepted_evidence,
                coverage=evaluation.coverage,
                information_gain=evaluation.information_gain,
                low_information_gain_streak=evaluation.low_information_gain_streak,
            )
        return ResearchIterationResult(
            outcome=(
                f"research_stopped:{evaluation.decision}:iterations=1:"
                f"pages={result.pages_read}:accepted={result.accepted_evidence}"
            ),
            continue_research=evaluation.continue_research,
            decision=evaluation.decision,
            pages_read=result.pages_read,
            accepted_evidence=result.accepted_evidence,
            coverage=evaluation.coverage,
            information_gain=evaluation.information_gain,
            low_information_gain_streak=evaluation.low_information_gain_streak,
        )

    async def _reserve_gap_work(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        state: ResearchRunState,
        reservation_identity: str,
    ) -> tuple[
        ResearchRunState,
        tuple[SemanticGapControlArtifact, SemanticIntentControlArtifact] | None,
        tuple[ResearchWork, WorkAttempt] | None,
    ]:
        """Reserve Gap Work only from current, complete RunState authority."""

        if self._semantic_recovery is None:
            return state, None, None
        replay = getattr(self._semantic_recovery, "recover_semantic_feedback_for_run", None)
        if not callable(replay):
            raise RuntimeError("gap_semantic_recovery_unavailable")
        await replay(run_id)
        current_state = await self._run_states.load_current(run_id)
        if (
            current_state is None
            or current_state.run_ref != run_id
            or current_state.gap_projection_status is not GapProjectionStatus.DERIVED
            or current_state.semantic_execution_status is not SemanticExecutionStatus.COMPLETE
        ):
            raise RuntimeError("gap_projection_not_current_and_complete")
        state = current_state
        gap_artifact, intent_artifact = await self._run_states.load_control_projection(state)
        if gap_artifact is None or intent_artifact is None:
            raise RuntimeError("gap_control_projection_unavailable")
        if intent_artifact.issues:
            raise RuntimeError("gap_intent_projection_has_issues")
        budget_exhausted = await self._tools.run_budget_exhausted(run_id)
        await self._work.reconcile_proposition_gap_work_after_semantic_update(
            state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
            budget_exhausted=budget_exhausted,
        )
        if budget_exhausted:
            return state, (gap_artifact, intent_artifact), None
        await self._work.materialize_proposition_gap_work(
            state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
        )
        reserved = await self._work.reserve_next(
            run_id,
            run_state=state,
            reservation_identity=reservation_identity,
            work_kind=ResearchWorkKind.PROPOSITION_GAP,
        )
        return state, (gap_artifact, intent_artifact), reserved

    async def _recover_gap_semantics(self, run_id: UUID) -> None:
        replay = getattr(self._semantic_recovery, "recover_semantic_feedback_for_run", None)
        if not callable(replay):
            raise RuntimeError("gap_semantic_recovery_unavailable_after_attempt")
        await replay(run_id)
