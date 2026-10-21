from __future__ import annotations

from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from app.core.config import Settings
from app.domain.proposition_research_gap import (
    DesiredEvidenceRelation,
    GapSearchability,
    PropositionGapLifecycle,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
)
from app.domain.research_work import (
    GapResearchExecutionContext,
    LegacyResearchObjective,
    PropositionGapResearchObjective,
    ResearchWork,
    ResearchWorkKind,
    ResearchWorkStatus,
    WorkAttempt,
    WorkAttemptState,
    legacy_objective_hash,
    legacy_work_identity,
    proposition_gap_objective_hash,
    proposition_gap_work_identity,
)
from app.domain.run_state import GapProjectionStatus, SemanticExecutionStatus
from app.infrastructure.db.research_tools import IterationEvaluation, ResearchTarget
from app.services.research_loop import ResearchAttemptResult
from app.services.research_orchestrator import ResearchOrchestrator

RUN = UUID("71000000-0000-4000-8000-000000000001")
GROUP = "a" * 64
GAP = "b" * 64
INTENT = "c" * 64
PROP = UUID("72000000-0000-4000-8000-000000000001")
ATTEMPT_REF = UUID("73000000-0000-4000-8000-000000000001")
TOOL_REF = UUID("74000000-0000-4000-8000-000000000001")


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run the immediate fake-boundary coroutine without pytest's event loop."""

    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("fake orchestration coroutine unexpectedly yielded")


def _state(*, version: int = 1, derived: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        run_ref=RUN,
        state_ref=f"runs/state-{version}.json",
        state_hash=str(version) * 64,
        state_version=version,
        gap_projection_status=(
            GapProjectionStatus.DERIVED if derived else GapProjectionStatus.INCOMPLETE
        ),
        semantic_execution_status=SemanticExecutionStatus.COMPLETE,
    )


def _work(state: SimpleNamespace, *, gap_ref: str = GAP) -> ResearchWork:
    required = RequiredEvidenceCharacteristics(
        target_group_ref=GROUP,
        target_proposition_refs=(PROP,),
        desired_relations=(DesiredEvidenceRelation.SUPPORTS,),
    )
    objective = PropositionGapResearchObjective(
        gap_ref=gap_ref,
        intent_ref=INTENT,
        objective_scope_ref="scope:group",
        target_group_ref=GROUP,
        target_proposition_refs=(PROP,),
        gap_type=PropositionResearchGapType.MISSING_SUPPORT,
        required_evidence=required,
        source_semantic_snapshot_ref=f"runs/semantic-{state.state_version}.json",
        source_semantic_snapshot_hash=str(state.state_version) * 64,
        source_gap_projection_hash="d" * 64,
    )
    objective_hash = proposition_gap_objective_hash(objective)
    return ResearchWork(
        work_ref=proposition_gap_work_identity(
            run_ref=RUN, gap_ref=gap_ref, objective_hash=objective_hash
        ),
        run_ref=RUN,
        producer_kind="SEMANTIC_GAP_PROJECTOR",
        producer_ref=f"gap:{gap_ref}:objective:{objective_hash}",
        work_kind=ResearchWorkKind.PROPOSITION_GAP,
        objective=objective,
        objective_hash=objective_hash,
        priority=1,
        execution_bounds={"max_attempts": 2},
        budget_attribution={"scope": "RUN_LEVEL"},
        source_run_state_ref=state.state_ref,
        source_run_state_hash=state.state_hash,
        source_run_state_version=state.state_version,
        status=ResearchWorkStatus.ELIGIBLE,
    )


def _attempt(work: ResearchWork, index: int = 1) -> WorkAttempt:
    return WorkAttempt(
        attempt_ref=ATTEMPT_REF,
        work_ref=work.work_ref,
        attempt_index=index,
        state=WorkAttemptState.RESERVED,
        reservation_identity=f"reservation-{index}",
        source_run_state_ref=work.source_run_state_ref,
        source_run_state_version=work.source_run_state_version,
    )


def _target(work: ResearchWork, attempt: WorkAttempt) -> ResearchTarget:
    objective = work.objective
    assert isinstance(objective, PropositionGapResearchObjective)
    context = GapResearchExecutionContext(
        gap_ref=objective.gap_ref,
        intent_ref=objective.intent_ref,
        gap_type=objective.gap_type,
        searchability=GapSearchability.SEARCHABLE,
        lifecycle_state=PropositionGapLifecycle.OPEN,
        target_group_ref=objective.target_group_ref,
        target_proposition_refs=objective.target_proposition_refs,
        parent_claim_refs=objective.parent_claim_refs,
        required_evidence=objective.required_evidence,
        objective_scope_ref=objective.objective_scope_ref,
        source_semantic_snapshot_ref=objective.source_semantic_snapshot_ref,
        source_semantic_snapshot_hash=objective.source_semantic_snapshot_hash,
        source_gap_projection_hash=objective.source_gap_projection_hash,
        attempt_index=1,
    )
    return ResearchTarget(
        plan_version=1,
        question_id="q-lineage",
        question="Resolve the open proposition gap.",
        query="source-backed query",
        gap_id=UUID("75000000-0000-4000-8000-000000000001"),
        tool_call_id=TOOL_REF,
        source_id_seed=UUID("76000000-0000-4000-8000-000000000001"),
        execution_work_ref=work.work_ref,
        execution_attempt_ref=attempt.attempt_ref,
        gap_execution_context=context,
    )


class _Runs:
    async def get_run_state_inputs(self, _run_id: UUID) -> dict[str, str]:
        return {"orchestrator_version": "V3", "lease_owner": "lease-1"}

    async def get_plan_for_execution(self, _run_id: UUID) -> object:
        return object()


class _Publisher:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state
        self.projection = SimpleNamespace(issues=(), artifact_ref="gap-artifact")
        self.intent = SimpleNamespace(issues=(), artifact_ref="intent-artifact")
        self.loads = 0
        self.projection_error: str | None = None

    async def load_current(self, _run_id: UUID) -> SimpleNamespace:
        self.loads += 1
        return self.state

    async def load_control_projection(
        self, state: SimpleNamespace
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        assert state is self.state
        if self.projection_error is not None:
            raise ValueError(self.projection_error)
        return self.projection, self.intent


class _Recovery:
    def __init__(self, calls: list[str], on_recover: object | None = None) -> None:
        self.calls = calls
        self.on_recover = on_recover

    async def recover_semantic_feedback_for_run(self, _run_id: UUID) -> int:
        self.calls.append("semantic_recovery")
        if callable(self.on_recover):
            self.on_recover()
        return 0


class _Work:
    def __init__(
        self,
        state: SimpleNamespace,
        calls: list[str],
        *,
        legacy: tuple[ResearchWork, WorkAttempt] | None = None,
        gap: tuple[ResearchWork, WorkAttempt] | None = None,
    ) -> None:
        self.state = state
        self.calls = calls
        self.legacy = legacy
        self.gap = gap
        self.reserved_kinds: list[ResearchWorkKind] = []
        self.settlements: list[dict[str, object]] = []
        self.materialized_gap_inputs: list[tuple[object, object, object]] = []
        self.reconciliations: list[dict[str, object]] = []
        self.gap_available = True
        self.gap_reserve_count = 0

    async def materialize_legacy_plan(self, _state: object, _plan: object) -> tuple[()]:
        self.calls.append("materialize_legacy")
        return ()

    async def reserve_next(
        self, _run_id: UUID, *, work_kind: ResearchWorkKind = ResearchWorkKind.LEGACY_RESEARCH,
        **_kwargs: object,
    ) -> tuple[ResearchWork, WorkAttempt] | None:
        self.calls.append(f"reserve:{work_kind.value}")
        self.reserved_kinds.append(work_kind)
        if work_kind is ResearchWorkKind.LEGACY_RESEARCH:
            return self.legacy
        self.gap_reserve_count += 1
        return self.gap if self.gap_available else None

    async def reconcile_proposition_gap_work_after_semantic_update(
        self, state: object, **kwargs: object
    ) -> tuple[()]:
        self.calls.append("reconcile")
        self.reconciliations.append({"state": state, **kwargs})
        return ()

    async def claim_attempt_execution_ownership(self, *_args: object, **_kwargs: object) -> bool:
        return True

    async def assert_attempt_execution_ownership(self, *_args: object, **_kwargs: object) -> bool:
        return True

    async def materialize_proposition_gap_work(
        self, state: object, *, gap_artifact: object, intent_artifact: object
    ) -> tuple[ResearchWork, ...]:
        self.calls.append("materialize_gap")
        self.materialized_gap_inputs.append((state, gap_artifact, intent_artifact))
        return (self.gap[0],) if self.gap is not None and self.gap_available else ()

    async def mark_attempt_executing(self, _attempt_ref: UUID, **_kwargs: object) -> WorkAttempt:
        self.calls.append("mark_executing")
        selected = self.gap or self.legacy
        assert selected is not None
        return selected[1].model_copy(update={"state": WorkAttemptState.EXECUTING})

    async def settle_attempt(self, _attempt_ref: UUID, **kwargs: object) -> WorkAttempt:
        self.calls.append("settle")
        self.settlements.append(kwargs)
        selected = self.gap or self.legacy
        assert selected is not None
        return selected[1].model_copy(update={"state": kwargs["state"]})


class _Tools:
    def __init__(self, calls: list[str], *, budget_exhausted: bool = False) -> None:
        self.calls = calls
        self.budget_exhausted = budget_exhausted
        self.target: ResearchTarget | None = None
        self.replan = False

    async def run_budget_exhausted(self, _run_id: UUID) -> bool:
        self.calls.append("budget")
        return self.budget_exhausted

    async def prepare_target_for_work(self, _run_id: UUID, **kwargs: object) -> ResearchTarget:
        self.calls.append("prepare_target")
        assert "question_id" not in kwargs
        assert kwargs["run_state"] is not None
        assert kwargs["gap_artifact"] is not None
        assert kwargs["intent_artifact"] is not None
        assert self.target is not None
        return self.target

    async def project_work_failure(self, *_args: object, **_kwargs: object) -> None:
        self.calls.append("project_failure")

    async def replan_requested(self, _run_id: UUID, *, worker_task_id: str) -> bool:
        return self.replan


class _LegacyTools(_Tools):
    async def prepare_target_for_work(self, _run_id: UUID, **kwargs: object) -> ResearchTarget:
        self.calls.append("prepare_legacy_target")
        assert kwargs["question_id"] == "legacy-q"
        assert "run_state" not in kwargs
        assert self.target is not None
        return self.target


class _Loop:
    def __init__(self, calls: list[str], *, fail_execution: bool = False) -> None:
        self.calls = calls
        self.fail_execution = fail_execution

    async def execute_target(self, _run_id: UUID, **_kwargs: object) -> ResearchAttemptResult:
        self.calls.append("execute_shared_target")
        if self.fail_execution:
            raise RuntimeError("fake bounded execution failure")
        return ResearchAttemptResult(
            pages_read=1,
            accepted_evidence=1,
            outcome="evidence_gained",
            evidence_refs=(UUID("77000000-0000-4000-8000-000000000001"),),
        )

    async def project_work_iteration(self, _run_id: UUID, **_kwargs: object) -> IterationEvaluation:
        self.calls.append("project_iteration")
        return IterationEvaluation(
            continue_research=False,
            decision="plan_complete",
            stop_reason=None,
            question_status="complete",
            coverage=1.0,
            information_gain=0.0,
        )


def _orchestrator(
    *,
    state: SimpleNamespace,
    calls: list[str],
    work: _Work,
    publisher: _Publisher | None = None,
    recovery: _Recovery | None = None,
    tools: _Tools | None = None,
    loop: _Loop | None = None,
    enabled: bool = True,
) -> ResearchOrchestrator:
    tools = tools or _Tools(calls)
    publisher = publisher or _Publisher(state)
    if work.gap is not None:
        tools.target = _target(*work.gap)
    return ResearchOrchestrator(
        runs=_Runs(),  # type: ignore[arg-type]
        work=work,  # type: ignore[arg-type]
        run_states=publisher,  # type: ignore[arg-type]
        loop=loop or _Loop(calls),  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
        semantic_recovery=recovery,
        gap_work_scheduling_enabled=enabled,
    )


def _legacy_pair() -> tuple[ResearchWork, WorkAttempt]:
    objective = LegacyResearchObjective(
        question_id="legacy-q",
        question="Legacy plan question remains executable.",
        rationale="Preserve existing plan-driven behavior.",
        requirements=("source-backed evidence",),
        priority=1,
        plan_version=1,
    )
    objective_hash = legacy_objective_hash(objective)
    work_item = ResearchWork(
        work_ref=legacy_work_identity(
            run_ref=RUN,
            plan_version=1,
            question_id="legacy-q",
            objective_hash=objective_hash,
        ),
        run_ref=RUN,
        producer_kind="LEGACY_PLANNER",
        producer_ref="plan:1:question:legacy-q",
        work_kind=ResearchWorkKind.LEGACY_RESEARCH,
        objective=objective,
        objective_hash=objective_hash,
        priority=1,
        source_run_state_ref="runs/legacy-state.json",
        source_run_state_hash="a" * 64,
        source_run_state_version=1,
        status=ResearchWorkStatus.ELIGIBLE,
    )
    attempt = WorkAttempt(
        attempt_ref=ATTEMPT_REF,
        work_ref=work_item.work_ref,
        attempt_index=1,
        state=WorkAttemptState.RESERVED,
        reservation_identity="legacy-reservation",
        source_run_state_ref=work_item.source_run_state_ref,
        source_run_state_version=1,
    )
    return work_item, attempt


def test_gap_work_uses_authoritative_projection_and_reassesses_after_attempt() -> None:
    state = _state()
    work_item = _work(state)
    attempt = _attempt(work_item)
    calls: list[str] = []
    publisher = _Publisher(state)
    repo = _Work(state, calls, gap=(work_item, attempt))
    recovery = _Recovery(calls)
    orchestrator = _orchestrator(
        state=state, calls=calls, work=repo, publisher=publisher, recovery=recovery
    )

    result = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert repo.reserved_kinds == [
        ResearchWorkKind.LEGACY_RESEARCH,
        ResearchWorkKind.PROPOSITION_GAP,
    ]
    assert repo.materialized_gap_inputs == [(state, publisher.projection, publisher.intent)]
    assert calls.index("prepare_target") < calls.index("execute_shared_target")
    assert calls.index("settle") < max(
        index for index, call in enumerate(calls) if call == "semantic_recovery"
    )
    assert repo.settlements[0]["state"] is WorkAttemptState.COMPLETED
    assert result.continue_research is True
    assert result.decision == "gap_reassessment_pending"


def test_legacy_work_keeps_original_execution_path_and_result() -> None:
    state = _state(derived=False)
    calls: list[str] = []
    legacy = _legacy_pair()
    repo = _Work(state, calls, legacy=legacy)
    tools = _LegacyTools(calls)
    tools.target = ResearchTarget(
        plan_version=1,
        question_id="legacy-q",
        question="Legacy plan question remains executable.",
        query="legacy query",
        gap_id=UUID("78000000-0000-4000-8000-000000000001"),
        tool_call_id=TOOL_REF,
        source_id_seed=UUID("79000000-0000-4000-8000-000000000001"),
    )
    orchestrator = _orchestrator(
        state=state,
        calls=calls,
        work=repo,
        tools=tools,
        enabled=False,
    )

    result = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert result.decision == "plan_complete"
    assert result.continue_research is False
    assert calls == [
        "materialize_legacy",
        "reserve:LEGACY_RESEARCH",
        "prepare_legacy_target",
        "mark_executing",
        "execute_shared_target",
        "project_iteration",
        "settle",
    ]
    assert repo.settlements[0]["state"] is WorkAttemptState.COMPLETED


def test_gap_scheduling_gate_off_preserves_legacy_only_reservation_path() -> None:
    state = _state(derived=False)
    calls: list[str] = []
    repo = _Work(state, calls)
    orchestrator = _orchestrator(state=state, calls=calls, work=repo, enabled=False)

    result = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert result.decision == "no_pending_question"
    assert repo.reserved_kinds == [ResearchWorkKind.LEGACY_RESEARCH]
    assert "semantic_recovery" not in calls
    assert "materialize_gap" not in calls


def test_production_gap_scheduling_default_remains_disabled() -> None:
    assert Settings.model_fields["proposition_gap_work_scheduling_enabled"].default is False


def test_incomplete_or_stale_projection_fails_closed_before_gap_reservation() -> None:
    state = _state(derived=False)
    calls: list[str] = []
    repo = _Work(state, calls)
    recovery = _Recovery(calls)
    orchestrator = _orchestrator(
        state=state, calls=calls, work=repo, recovery=recovery
    )

    with pytest.raises(RuntimeError, match="gap_projection_not_current_and_complete"):
        _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert repo.reserved_kinds == [ResearchWorkKind.LEGACY_RESEARCH]
    assert "materialize_gap" not in calls
    assert "execute_shared_target" not in calls


def test_stale_projection_artifact_fails_closed_before_gap_reservation() -> None:
    state = _state()
    calls: list[str] = []
    repo = _Work(state, calls)
    publisher = _Publisher(state)
    publisher.projection_error = "run_state_semantic_pointer_mismatch"
    orchestrator = _orchestrator(
        state=state,
        calls=calls,
        work=repo,
        publisher=publisher,
        recovery=_Recovery(calls),
    )

    with pytest.raises(ValueError, match="run_state_semantic_pointer_mismatch"):
        _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert repo.reserved_kinds == [ResearchWorkKind.LEGACY_RESEARCH]
    assert "materialize_gap" not in calls
    assert "execute_shared_target" not in calls


def test_exhausted_budget_reconciles_but_never_materializes_or_reserves_gap() -> None:
    state = _state()
    calls: list[str] = []
    repo = _Work(state, calls)
    tools = _Tools(calls, budget_exhausted=True)
    orchestrator = _orchestrator(
        state=state,
        calls=calls,
        work=repo,
        tools=tools,
        recovery=_Recovery(calls),
    )

    result = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert result.continue_research is False
    assert "reconcile" in calls
    assert "materialize_gap" not in calls
    assert repo.reserved_kinds == [ResearchWorkKind.LEGACY_RESEARCH]


def test_same_gap_work_identity_is_used_for_attempt_and_recovery() -> None:
    state = _state()
    work_item = _work(state)
    attempt = _attempt(work_item, index=2)
    calls: list[str] = []
    repo = _Work(state, calls, gap=(work_item, attempt))
    orchestrator = _orchestrator(
        state=state,
        calls=calls,
        work=repo,
        recovery=_Recovery(calls),
    )

    _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert repo.gap is not None
    assert repo.gap[0].work_ref == work_item.work_ref
    assert repo.gap[1].attempt_index == 2
    assert repo.settlements[0]["produced_evidence_refs"] == (
        UUID("77000000-0000-4000-8000-000000000001"),
    )


def test_failed_gap_attempt_is_semantically_reassessed_before_retry() -> None:
    state = _state()
    work_item = _work(state)
    attempt = _attempt(work_item)
    calls: list[str] = []
    repo = _Work(state, calls, gap=(work_item, attempt))
    recovery = _Recovery(calls)
    orchestrator = _orchestrator(
        state=state,
        calls=calls,
        work=repo,
        recovery=recovery,
        loop=_Loop(calls, fail_execution=True),
    )

    result = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert result.decision == "work_retryable"
    assert repo.settlements[0]["state"] is WorkAttemptState.FAILED
    assert calls.index("settle") < max(
        index for index, call in enumerate(calls) if call == "semantic_recovery"
    )


def test_new_gap_is_materialized_from_the_new_current_projection() -> None:
    first = _state(version=1)
    fresh = _state(version=2)
    fresh_work = _work(fresh, gap_ref="e" * 64)
    calls: list[str] = []
    publisher = _Publisher(first)

    def advance_projection() -> None:
        publisher.state = fresh

    repo = _Work(first, calls, gap=(fresh_work, _attempt(fresh_work)))
    orchestrator = _orchestrator(
        state=first,
        calls=calls,
        work=repo,
        publisher=publisher,
        recovery=_Recovery(calls, on_recover=advance_projection),
    )

    result = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert repo.materialized_gap_inputs[0][0] is fresh
    assert repo.gap is not None and repo.gap[0].objective.gap_ref == "e" * 64
    assert result.decision == "gap_reassessment_pending"


def test_closed_gap_is_not_reexecuted_on_repeated_orchestrator_call() -> None:
    state = _state(version=1)
    fresh = _state(version=2)
    work_item = _work(state)
    calls: list[str] = []
    publisher = _Publisher(state)
    repo = _Work(state, calls, gap=(work_item, _attempt(work_item)))
    recovery_count = 0

    def close_gap_after_execution() -> None:
        nonlocal recovery_count
        recovery_count += 1
        if recovery_count == 2:
            publisher.state = fresh
            repo.gap_available = False

    orchestrator = _orchestrator(
        state=state,
        calls=calls,
        work=repo,
        publisher=publisher,
        recovery=_Recovery(calls, on_recover=close_gap_after_execution),
    )

    first = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))
    second = _run(orchestrator.run_next_work(RUN, worker_task_id="worker"))

    assert first.decision == "gap_reassessment_pending"
    assert second.decision == "no_pending_question"
    assert calls.count("execute_shared_target") == 1
    assert repo.gap_reserve_count == 2
