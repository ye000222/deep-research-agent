from collections.abc import Coroutine, Generator
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from app.domain.research_work import (
    LegacyResearchObjective,
    ResearchWork,
    ResearchWorkKind,
    ResearchWorkStatus,
    WorkAttempt,
    WorkAttemptState,
    deterministic_attempt_ref,
    legacy_objective_hash,
    legacy_work_identity,
)
from app.infrastructure.db.research_tools import IterationEvaluation, ResearchTarget
from app.services.research_loop import ResearchAttemptResult
from app.services.research_orchestrator import (
    ResearchOrchestrator,
    research_executor_for_version,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Drive this all-fake async test without platform event-loop fixtures."""

    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("fake orchestrator coroutine unexpectedly yielded")


def test_executor_selection_is_explicit_and_closed() -> None:
    assert research_executor_for_version("V2_LEGACY") == "V2_LEGACY"
    assert research_executor_for_version("V3") == "V3"
    with pytest.raises(ValueError, match="unsupported_orchestrator_version"):
        research_executor_for_version("V4")


class _Runs:
    def __init__(self, run_id: UUID, plan: object) -> None:
        self.run_id = run_id
        self.plan = plan

    async def get_run_state_inputs(self, _run_id: UUID) -> dict[str, object]:
        return {"orchestrator_version": "V3", "lease_owner": "lease-1"}

    async def get_plan_for_execution(self, _run_id: UUID) -> object:
        return self.plan


class _StatePublisher:
    def __init__(self, state: object) -> None:
        self.state = state

    async def load_current(self, _run_id: UUID) -> object:
        return self.state


class _WorkRepository:
    def __init__(self, work_item: ResearchWork, attempt: WorkAttempt, calls: list[str]) -> None:
        self.work_item = work_item
        self.attempt = attempt
        self.calls = calls
        self.settlement: dict[str, object] = {}

    async def materialize_legacy_plan(
        self, _state: object, _plan: object
    ) -> tuple[ResearchWork, ...]:
        self.calls.append("materialize")
        return (self.work_item,)

    async def reserve_next(
        self, _run_id: UUID, **_kwargs: object
    ) -> tuple[ResearchWork, WorkAttempt]:
        self.calls.append("reserve")
        return self.work_item, self.attempt

    async def claim_attempt_execution_ownership(self, *_args: object, **_kwargs: object) -> bool:
        return True

    async def assert_attempt_execution_ownership(self, *_args: object, **_kwargs: object) -> bool:
        return True

    async def mark_attempt_executing(self, _attempt_ref: UUID, **_kwargs: object) -> WorkAttempt:
        self.calls.append("executing")
        return self.attempt.model_copy(update={"state": WorkAttemptState.EXECUTING})

    async def settle_attempt(self, _attempt_ref: UUID, **kwargs: object) -> WorkAttempt:
        self.calls.append("settle")
        self.settlement = kwargs
        return self.attempt.model_copy(update={"state": kwargs["state"]})


class _Tools:
    def __init__(self, target: ResearchTarget, calls: list[str]) -> None:
        self.target = target
        self.calls = calls

    async def prepare_target_for_work(self, _run_id: UUID, **kwargs: object) -> ResearchTarget:
        self.calls.append("target")
        assert kwargs["question_id"] == "q1"
        assert kwargs["work_item"] is not None
        assert kwargs["attempt"] is not None
        return self.target

    async def replan_requested(self, _run_id: UUID, *, worker_task_id: str) -> bool:
        return False


class _Loop:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def execute_target(self, _run_id: UUID, **_kwargs: object) -> ResearchAttemptResult:
        self.calls.append("execute_target")
        return ResearchAttemptResult(
            pages_read=1,
            accepted_evidence=1,
            outcome="evidence_gained",
            evidence_refs=(UUID("33333333-3333-4333-8333-333333333333"),),
        )

    async def project_work_iteration(self, _run_id: UUID, **_kwargs: object) -> IterationEvaluation:
        self.calls.append("project")
        return IterationEvaluation(
            continue_research=True,
            decision="continue_plan",
            stop_reason=None,
            question_status="active",
            coverage=0.25,
            information_gain=0.1,
        )


@pytest.mark.parametrize(
    ("attempt_state", "expected_execution_prefix"),
    [
        (WorkAttemptState.RESERVED, ["executing"]),
        (WorkAttemptState.EXECUTING, []),
    ],
)
def test_v3_reserves_before_target_and_resumes_active_attempt(
    attempt_state: WorkAttemptState,
    expected_execution_prefix: list[str],
) -> None:
    run_id = UUID("11111111-1111-4111-8111-111111111111")
    objective = LegacyResearchObjective(
        question_id="q1",
        question="Find source-backed evidence for the first dimension.",
        rationale="The planned dimension requires source evidence.",
        requirements=("an exact public source",),
        priority=1,
        search_hints=("official source",),
        acceptance_context=(("q1:d1", "an exact public source"),),
        plan_version=1,
        bounds={"max_attempts": 3},
    )
    work_ref = legacy_work_identity(
        run_ref=run_id,
        plan_version=1,
        question_id="q1",
        objective_hash=legacy_objective_hash(objective),
    )
    work_item = ResearchWork(
        work_ref=work_ref,
        run_ref=run_id,
        producer_kind="LEGACY_PLANNER",
        producer_ref="plan:1:question:q1",
        work_kind=ResearchWorkKind.LEGACY_RESEARCH,
        objective=objective,
        objective_hash=legacy_objective_hash(objective),
        priority=1,
        source_run_state_ref="runs/test/run-state/hash.json",
        source_run_state_hash="a" * 64,
        source_run_state_version=1,
        status=ResearchWorkStatus.ELIGIBLE,
        execution_bounds={"max_attempts": 3},
    )
    attempt = WorkAttempt(
        attempt_ref=deterministic_attempt_ref(work_ref, 1),
        work_ref=work_ref,
        attempt_index=1,
        state=attempt_state,
        reservation_identity="reservation-1",
        source_run_state_ref=work_item.source_run_state_ref,
        source_run_state_version=1,
    )
    tool_call_id = UUID("22222222-2222-4222-8222-222222222222")
    target = ResearchTarget(
        plan_version=1,
        question_id="q1",
        question=objective.question,
        query="official source query",
        gap_id=UUID("44444444-4444-4444-8444-444444444444"),
        tool_call_id=tool_call_id,
        source_id_seed=UUID("55555555-5555-4555-8555-555555555555"),
        priority=1,
        acceptance_dimensions=objective.acceptance_context,
    )
    calls: list[str] = []
    work_repo = _WorkRepository(work_item, attempt, calls)
    orchestrator = ResearchOrchestrator(
        runs=_Runs(run_id, object()),  # type: ignore[arg-type]
        work=work_repo,  # type: ignore[arg-type]
        run_states=_StatePublisher(SimpleNamespace(run_ref=run_id)),  # type: ignore[arg-type]
        loop=_Loop(calls),  # type: ignore[arg-type]
        tools=_Tools(target, calls),  # type: ignore[arg-type]
    )

    result = _run(orchestrator.run_next_work(run_id, worker_task_id="worker-1"))

    assert calls == [
        "materialize",
        "reserve",
        "target",
        *expected_execution_prefix,
        "execute_target",
        "project",
        "settle",
    ]
    assert result.decision == "continue_plan"
    assert work_repo.settlement["state"] is WorkAttemptState.COMPLETED
    assert work_repo.settlement["tool_call_refs"] == (tool_call_id,)
    assert work_repo.settlement["produced_evidence_refs"] == (
        UUID("33333333-3333-4333-8333-333333333333"),
    )


def test_concurrent_callers_execute_one_reused_active_attempt_once() -> None:
    run_id = UUID("11111111-1111-4111-8111-111111111111")
    objective = LegacyResearchObjective(
        question_id="q1",
        question="Find evidence for the planned dimension.",
        rationale="The dimension needs source-backed evidence.",
        requirements=("one primary source",),
        priority=1,
        plan_version=1,
    )
    objective_hash = legacy_objective_hash(objective)
    work_item = ResearchWork(
        work_ref=legacy_work_identity(
            run_ref=run_id,
            plan_version=1,
            question_id="q1",
            objective_hash=objective_hash,
        ),
        run_ref=run_id,
        producer_kind="LEGACY_PLANNER",
        producer_ref="plan:1:question:q1",
        work_kind=ResearchWorkKind.LEGACY_RESEARCH,
        objective=objective,
        objective_hash=objective_hash,
        priority=1,
        source_run_state_ref="runs/test/run-state/current.json",
        source_run_state_hash="a" * 64,
        source_run_state_version=1,
        status=ResearchWorkStatus.IN_PROGRESS,
    )
    attempt = WorkAttempt(
        attempt_ref=deterministic_attempt_ref(work_item.work_ref, 1),
        work_ref=work_item.work_ref,
        attempt_index=1,
        state=WorkAttemptState.RESERVED,
        reservation_identity="same-durable-attempt",
        source_run_state_ref=work_item.source_run_state_ref,
        source_run_state_version=1,
    )
    target = ResearchTarget(
        plan_version=1,
        question_id="q1",
        question=objective.question,
        query="planned query",
        gap_id=UUID("44444444-4444-4444-8444-444444444444"),
        tool_call_id=UUID("22222222-2222-4222-8222-222222222222"),
        source_id_seed=UUID("55555555-5555-4555-8555-555555555555"),
    )
    calls: list[str] = []

    class _RacingWork(_WorkRepository):
        def __init__(self) -> None:
            super().__init__(work_item, attempt, calls)
            self.owner: str | None = None

        async def claim_attempt_execution_ownership(
            self, *_args: object, **kwargs: object
        ) -> bool:
            await _YieldAtOwnershipClaim()
            if self.owner is not None:
                return False
            self.owner = str(kwargs["execution_owner_token"])
            return True

        async def assert_attempt_execution_ownership(
            self, *_args: object, **kwargs: object
        ) -> bool:
            return self.owner == kwargs["execution_owner_token"]

    work_repo = _RacingWork()
    orchestrator = ResearchOrchestrator(
        runs=_Runs(run_id, object()),  # type: ignore[arg-type]
        work=work_repo,  # type: ignore[arg-type]
        run_states=_StatePublisher(SimpleNamespace(run_ref=run_id)),  # type: ignore[arg-type]
        loop=_Loop(calls),  # type: ignore[arg-type]
        tools=_Tools(target, calls),  # type: ignore[arg-type]
    )

    first_coro = orchestrator.run_next_work(run_id, worker_task_id="same-worker")
    second_coro = orchestrator.run_next_work(run_id, worker_task_id="same-worker")
    assert first_coro.send(None) == "ownership-claim"
    assert second_coro.send(None) == "ownership-claim"
    results: list[object] = []
    for coroutine in (first_coro, second_coro):
        try:
            coroutine.send(None)
        except StopIteration as completed:
            results.append(completed.value)
        else:
            raise AssertionError("fake orchestration yielded beyond the claim barrier")
    first, second = results
    decisions = {first.decision, second.decision}  # type: ignore[attr-defined]
    assert "work_already_owned" in decisions
    assert calls.count("execute_target") == 1
    assert calls.count("settle") == 1


class _YieldAtOwnershipClaim:
    def __await__(self) -> Generator[str, None, None]:
        yield "ownership-claim"
