from __future__ import annotations

from collections.abc import Coroutine
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.domain.research_work import (
    WorkAttemptState,
    WorkResumePoint,
    deterministic_attempt_ref,
    resolve_work_resume_point,
)
from app.infrastructure.db.research_tools import ResearchTarget, ResearchToolRepository
from app.infrastructure.db.research_work import ResearchWorkRepository
from app.services.research_orchestrator import ResearchOrchestrator

_WORK = UUID("22222222-2222-4222-8222-222222222222")
_RUN = UUID("11111111-1111-4111-8111-111111111111")
_QUERY = UUID("33333333-3333-4333-8333-333333333333")
_EVIDENCE = UUID("44444444-4444-4444-8444-444444444444")


def _run_immediate(coroutine: Coroutine[object, object, object]) -> object:
    try:
        return coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("fake recovery persistence unexpectedly suspended")


class _DurableBacking:
    """Serialized row fixture; no Python domain object crosses instances."""

    def __init__(self) -> None:
        self.work: dict[str, object] = {
            "work_ref": _WORK,
            "run_id": _RUN,
            "producer_kind": "LEGACY_PLANNER",
            "producer_ref": "plan:1:question:q1",
            "work_kind": "LEGACY_RESEARCH",
            "objective": {
                "question_id": "q1",
                "question": "Persisted question",
                "rationale": "Persisted rationale",
                "requirements": ["primary evidence"],
                "priority": 1,
                "search_hints": [],
                "acceptance_context": [["q1:d1", "primary evidence"]],
                "exclusions": [],
                "plan_version": 1,
                "bounds": {"max_attempts": 3},
            },
            "objective_hash": "a" * 64,
            "priority": 1,
            "constraints": {},
            "execution_bounds": {"max_attempts": 3},
            "budget_attribution": {},
            "source_run_state_ref": "runs/r/state/1.json",
            "source_run_state_hash": "b" * 64,
            "source_run_state_version": 1,
            "status": "IN_PROGRESS",
            "schema_version": "research-work-v1",
            "created_at": datetime(2026, 10, 1, tzinfo=UTC),
        }
        self.attempt: dict[str, object] = {
            "attempt_ref": deterministic_attempt_ref(_WORK, 1),
            "work_ref": _WORK,
            "attempt_index": 1,
            "state": "RESERVED",
            "reservation_identity": "reservation-1",
            "source_run_state_ref": "runs/r/state/1.json",
            "source_run_state_version": 1,
            "query_refs": [],
            "tool_call_refs": [],
            "produced_evidence_refs": [],
            "reused_evidence_refs": [],
            "budget_refs": [],
            "usage_summary": {},
            "failure_classification": None,
            "resume_metadata": {},
            "started_at": None,
            "completed_at": None,
            "schema_version": "work-attempt-v1",
        }


class _RepositoryInstance:
    def __init__(self, durable: _DurableBacking) -> None:
        # Each repository rehydrates from serialized durable rows afresh.
        self.work = ResearchWorkRepository._work_view(SimpleNamespace(**durable.work))
        self.attempt = ResearchWorkRepository._attempt_view(
            SimpleNamespace(**durable.attempt)
        )


def test_new_repository_instance_rehydrates_reserved_attempt_identity() -> None:
    durable = _DurableBacking()
    instance_a = _RepositoryInstance(durable)
    assert instance_a.attempt.state is WorkAttemptState.RESERVED
    attempt_ref = instance_a.attempt.attempt_ref
    del instance_a

    instance_b = _RepositoryInstance(durable)
    assert instance_b.work.work_ref == _WORK
    assert instance_b.attempt.attempt_index == 1
    assert instance_b.attempt.attempt_ref == attempt_ref
    assert resolve_work_resume_point(instance_b.attempt) is WorkResumePoint.TARGET_NOT_STARTED


def test_new_repository_instance_derives_executing_resume_point_from_durable_refs() -> None:
    durable = _DurableBacking()
    durable.attempt.update(
        {
            "state": "EXECUTING",
            "query_refs": [str(_QUERY)],
            "produced_evidence_refs": [str(_EVIDENCE)],
            "resume_metadata": {
                "last_durable_checkpoint": "EVIDENCE_DURABLE",
                "search_result_durable": True,
                "evidence_durable": True,
            },
        }
    )
    instance_a = _RepositoryInstance(durable)
    assert instance_a.attempt.attempt_ref == deterministic_attempt_ref(_WORK, 1)
    del instance_a

    instance_b = _RepositoryInstance(durable)
    assert instance_b.attempt.query_refs == (_QUERY,)
    assert instance_b.attempt.produced_evidence_refs == (_EVIDENCE,)
    assert resolve_work_resume_point(instance_b.attempt) is WorkResumePoint.EVIDENCE_DURABLE


def test_new_instance_failed_retry_uses_next_deterministic_attempt_index() -> None:
    durable = _DurableBacking()
    durable.attempt["state"] = "FAILED"
    instance_a = _RepositoryInstance(durable)
    assert instance_a.attempt.attempt_index == 1
    del instance_a

    instance_b = _RepositoryInstance(durable)
    next_attempt_ref = deterministic_attempt_ref(instance_b.work.work_ref, 2)
    assert next_attempt_ref != instance_b.attempt.attempt_ref
    assert instance_b.attempt.attempt_index + 1 == 2


def test_new_orchestrator_instance_does_not_execute_completed_work() -> None:
    durable = _DurableBacking()
    durable.work["status"] = "COMPLETED"
    durable.attempt["state"] = "COMPLETED"
    instance_a = _RepositoryInstance(durable)
    assert instance_a.work.status.value == "COMPLETED"
    del instance_a
    instance_b = _RepositoryInstance(durable)
    calls: list[str] = []

    class _Runs:
        async def get_run_state_inputs(self, _run_id: UUID) -> dict[str, object]:
            return {"orchestrator_version": "V3", "lease_owner": "lease-1"}

        async def get_plan_for_execution(self, _run_id: UUID) -> object:
            return object()

    class _StatePublisher:
        async def load_current(self, _run_id: UUID) -> object:
            return SimpleNamespace(run_ref=_RUN)

    class _WorkRepo:
        async def materialize_legacy_plan(self, *_args: object) -> tuple[object, ...]:
            calls.append("materialize_existing")
            return (instance_b.work,)

        async def reserve_next(self, *_args: object, **_kwargs: object) -> None:
            calls.append("no_eligible_work")
            return None

    class _Tools:
        async def replan_requested(self, *_args: object, **_kwargs: object) -> bool:
            return False

    class _Loop:
        async def execute_target(self, *_args: object, **_kwargs: object) -> None:
            calls.append("UNEXPECTED_EXECUTION")

    orchestrator = ResearchOrchestrator(
        runs=_Runs(),  # type: ignore[arg-type]
        work=_WorkRepo(),  # type: ignore[arg-type]
        run_states=_StatePublisher(),  # type: ignore[arg-type]
        loop=_Loop(),  # type: ignore[arg-type]
        tools=_Tools(),  # type: ignore[arg-type]
    )
    result = _run_immediate(orchestrator.run_next_work(_RUN, worker_task_id="worker"))
    assert result.decision == "no_pending_question"
    assert calls == ["materialize_existing", "no_eligible_work"]


def test_second_caller_reuses_active_attempt_after_run_state_only_advance() -> None:
    work_ref = UUID("55555555-5555-4555-8555-555555555555")
    durable = _DurableBacking()
    durable.work.update({"work_ref": work_ref, "status": "IN_PROGRESS"})
    durable.attempt.update(
        {
            "work_ref": work_ref,
            "attempt_ref": deterministic_attempt_ref(work_ref, 1),
            "state": "EXECUTING",
            "source_run_state_ref": "runs/r/state/1.json",
            "source_run_state_version": 1,
            "produced_evidence_refs": [str(_EVIDENCE)],
        }
    )
    run_row = SimpleNamespace(
        id=_RUN,
        orchestrator_version="V3",
        plan_version=1,
        latest_run_state_ref="runs/r/state/2.json",
        latest_run_state_hash="c" * 64,
        latest_run_state_version=2,
    )
    active_row = SimpleNamespace(**durable.attempt)
    work_row = SimpleNamespace(**durable.work)

    class _Session:
        def __init__(self) -> None:
            self.scalar_calls = 0
            self.added = 0

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> _Session:
            return self

        async def scalar(self, _statement: object) -> object:
            self.scalar_calls += 1
            return run_row if self.scalar_calls == 1 else active_row

        async def get(self, _model: object, _identity: UUID) -> object:
            return work_row

        def add(self, _row: object) -> None:
            self.added += 1

        async def flush(self) -> None:
            return None

    sessions: list[_Session] = []

    def _new_session() -> _Session:
        session = _Session()
        sessions.append(session)
        return session

    run_state = SimpleNamespace(
        run_ref=_RUN,
        state_ref="runs/r/state/2.json",
        state_hash="c" * 64,
        state_version=2,
    )
    first = _run_immediate(
        ResearchWorkRepository(_new_session).reserve_next(
            _RUN, run_state=run_state, reservation_identity="caller-a"
        )
    )
    second = _run_immediate(
        ResearchWorkRepository(_new_session).reserve_next(
            _RUN, run_state=run_state, reservation_identity="caller-b"
        )
    )

    assert first[1].attempt_ref == second[1].attempt_ref
    assert first[1].attempt_index == second[1].attempt_index == 1
    assert first[1].source_run_state_version == 1
    assert all(session.added == 0 for session in sessions)


def test_stale_eligible_view_conflict_creates_only_one_attempt() -> None:
    durable = _DurableBacking()
    durable.work["status"] = "ELIGIBLE"
    work_row = SimpleNamespace(**durable.work)
    attempts: list[SimpleNamespace] = []
    run_row = SimpleNamespace(
        id=_RUN,
        orchestrator_version="V3",
        plan_version=1,
        latest_run_state_ref="runs/r/state/1.json",
        latest_run_state_hash="b" * 64,
        latest_run_state_version=1,
    )

    class _Session:
        def __init__(self) -> None:
            self.scalar_index = 0

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> _Session:
            return self

        async def scalar(self, _statement: object) -> object:
            self.scalar_index += 1
            if self.scalar_index == 1:
                return run_row
            if self.scalar_index == 2:
                return next(
                    (
                        attempt
                        for attempt in attempts
                        if attempt.state in ("RESERVED", "EXECUTING")
                    ),
                    None,
                )
            if self.scalar_index == 3:
                return work_row if work_row.status == "ELIGIBLE" else None
            return max((item.attempt_index for item in attempts), default=None)

        async def get(self, _model: object, _identity: UUID) -> object:
            return work_row

        def add(self, attempt: object) -> None:
            attempt.query_refs = []  # type: ignore[attr-defined]
            attempt.tool_call_refs = []  # type: ignore[attr-defined]
            attempt.produced_evidence_refs = []  # type: ignore[attr-defined]
            attempt.reused_evidence_refs = []  # type: ignore[attr-defined]
            attempt.budget_refs = []  # type: ignore[attr-defined]
            attempt.usage_summary = {}  # type: ignore[attr-defined]
            attempt.resume_metadata = {}  # type: ignore[attr-defined]
            attempts.append(attempt)  # type: ignore[arg-type]
            work_row.status = "IN_PROGRESS"

        async def flush(self) -> None:
            return None

    run_state = SimpleNamespace(
        run_ref=_RUN,
        state_ref="runs/r/state/1.json",
        state_hash="b" * 64,
        state_version=1,
    )
    # Both callers observed the same stale eligible snapshot before Caller A
    # committed; reservation itself rechecks durable state under the repo seam.
    stale_eligible_snapshot = (work_row.work_ref, "ELIGIBLE")
    assert stale_eligible_snapshot[1] == "ELIGIBLE"
    first = _run_immediate(
        ResearchWorkRepository(lambda: _Session()).reserve_next(
            _RUN, run_state=run_state, reservation_identity="caller-a"
        )
    )
    second = _run_immediate(
        ResearchWorkRepository(lambda: _Session()).reserve_next(
            _RUN, run_state=run_state, reservation_identity="caller-b"
        )
    )
    assert first is not None and second is not None
    assert first[1].attempt_ref == second[1].attempt_ref
    assert first[1].attempt_index == second[1].attempt_index == 1
    assert len(attempts) == 1
    assert work_row.status == "IN_PROGRESS"


def test_plan_supersede_makes_old_active_work_non_executable() -> None:
    durable = _DurableBacking()
    old_work = SimpleNamespace(**durable.work)
    durable.attempt["state"] = "EXECUTING"
    active_attempt = SimpleNamespace(**durable.attempt)
    current_questions = [
        ResearchQuestion(
            id=f"q{index}",
            question=f"Research question {index}",
            priority=1,
            rationale="A current plan rationale",
            evidence_requirements=["one reliable source"],
            search_hints=[],
        )
        for index in range(1, 6)
    ]
    current_rows = [
        SimpleNamespace(
            question_id=question.id,
            question=question.question,
            rationale=question.rationale,
            evidence_requirements=question.evidence_requirements,
            priority=question.priority,
            search_hints=[],
            plan_version=2,
        )
        for question in current_questions
    ]
    run_row = SimpleNamespace(
        id=_RUN,
        orchestrator_version="V3",
        plan_version=2,
        latest_run_state_ref="runs/r/state/2.json",
        latest_run_state_hash="c" * 64,
        latest_run_state_version=2,
    )

    class _Rows:
        def __init__(self, rows: list[object]) -> None:
            self.rows = rows

        def all(self) -> list[object]:
            return self.rows

    class _Session:
        def __init__(self) -> None:
            self.scalar_index = 0
            self.scalars_index = 0
            self.added: list[object] = []

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> _Session:
            return self

        async def scalar(self, _statement: object) -> object:
            self.scalar_index += 1
            return run_row if self.scalar_index == 1 else None

        async def scalars(self, _statement: object) -> _Rows:
            self.scalars_index += 1
            return _Rows(current_rows if self.scalars_index == 1 else [old_work])

        def add(self, row: object) -> None:
            if getattr(row, "created_at", None) is None:
                row.created_at = datetime(2026, 10, 17, tzinfo=UTC)
            self.added.append(row)

        async def flush(self) -> None:
            return None

    run_state = SimpleNamespace(
        run_ref=_RUN,
        state_ref="runs/r/state/2.json",
        state_hash="c" * 64,
        state_version=2,
    )
    plan = ResearchPlan(
        goal="Research the current question set",
        scope_summary="Current plan supersedes earlier work",
        questions=current_questions,
        completion_criteria=["Evidence collected", "Evaluation complete"],
    )
    session = _Session()
    materialized = _run_immediate(
        ResearchWorkRepository(lambda: session).materialize_legacy_plan(run_state, plan)
    )
    assert old_work.status == "SUPERSEDED"
    assert active_attempt.state == "EXECUTING"  # Attempt lifecycle remains unchanged.
    assert len(materialized) == 5
    assert all(item.objective.plan_version == 2 for item in materialized)
    assert old_work not in session.added


@pytest.mark.parametrize(
    ("state", "metadata", "refs", "expected"),
    [
        (
            "EXECUTING",
            {"search_result_durable": True},
            {"query_refs": [str(_QUERY)]},
            WorkResumePoint.SEARCH_RESULT_DURABLE,
        ),
        (
            "EXECUTING",
            {"reader_artifact_ref": "runs/r/sources/s.txt"},
            {},
            WorkResumePoint.READER_RESULT_DURABLE,
        ),
        (
            "COMPLETED",
            {},
            {},
            WorkResumePoint.ATTEMPT_ALREADY_COMPLETE,
        ),
    ],
)
def test_rehydrated_durable_facts_control_resume_point(
    state: str,
    metadata: dict[str, object],
    refs: dict[str, list[str]],
    expected: WorkResumePoint,
) -> None:
    durable = _DurableBacking()
    durable.attempt.update({"state": state, "resume_metadata": metadata, **refs})
    assert resolve_work_resume_point(_RepositoryInstance(durable).attempt) is expected


def test_budget_reservation_settlement_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    durable_row = {
        "status": "reserved",
        "reserved_total": 100,
        "actual_total": 0,
        "usage_estimated": False,
        "settled_at": None,
        "question_id": "q1",
    }
    durable_run = {
        "id": _RUN,
        "budget_snapshot": {},
        "usage_snapshot": {"settled_reservations": 0},
    }

    async def no_refresh(_session: object, _run: object) -> None:
        return None

    async def no_event(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        ResearchToolRepository,
        "_refresh_model_pool_usage",
        staticmethod(no_refresh),
    )
    monkeypatch.setattr(
        ResearchToolRepository,
        "_append_event",
        staticmethod(no_event),
    )

    # Instance A settles from its own row views, then only serialized durable
    # fields are carried into the fresh instance.
    row_a = SimpleNamespace(**durable_row)
    run_a = SimpleNamespace(**durable_run)

    class _Session:
        def __init__(self, row: SimpleNamespace) -> None:
            self.row = row

        async def scalar(self, _statement: object) -> SimpleNamespace:
            return self.row

    first = _run_immediate(
        ResearchToolRepository._settle_reservation_row(
            _Session(row_a), run_a, attempt_id=deterministic_attempt_ref(_WORK, 1),
            actual_total=37, usage_estimated=False,
        )
    )
    durable_row.update(vars(row_a))
    durable_run.update(vars(run_a))
    del row_a, run_a
    row_b = SimpleNamespace(**durable_row)
    run_b = SimpleNamespace(**durable_run)
    second = _run_immediate(
        ResearchToolRepository._settle_reservation_row(
            _Session(row_b), run_b, attempt_id=deterministic_attempt_ref(_WORK, 1),
            actual_total=37, usage_estimated=False,
        )
    )
    assert first is True
    assert second is False
    assert row_b.actual_total == 37
    assert run_b.usage_snapshot["settled_reservations"] == 1


def test_duplicate_projection_reads_the_committed_attempt_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        "decision": "continue_plan",
        "stop_reason": None,
        "question_status": "active",
        "continue_research": True,
        "coverage": 0.5,
        "information_gain": 0.2,
        "low_information_gain_streak": 0,
    }
    event = SimpleNamespace(metrics=metrics)

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> _Session:
            return self

        async def scalar(self, _statement: object) -> SimpleNamespace:
            return event

    run = SimpleNamespace(quality_snapshot={})

    async def locked_run(_session: object, _run_id: UUID, _worker: str) -> object:
        return run

    monkeypatch.setattr(
        ResearchToolRepository,
        "_locked_run",
        staticmethod(locked_run),
    )
    def project_with_new_repository_instance() -> object:
        repository = ResearchToolRepository(lambda: _Session())  # type: ignore[arg-type]
        return _run_immediate(
            repository._project_iteration(
            _RUN,
            worker_task_id="worker",
            target=ResearchTarget(
                plan_version=1,
                question_id="q1",
                question="question",
                query="query",
                gap_id=UUID("55555555-5555-4555-8555-555555555555"),
                tool_call_id=UUID("66666666-6666-4666-8666-666666666666"),
                source_id_seed=UUID("77777777-7777-4777-8777-777777777777"),
                execution_attempt_ref=deterministic_attempt_ref(_WORK, 1),
            ),
            attempt_outcome="evidence_gained",
            legacy_control_flow=False,
        )
        )

    result = project_with_new_repository_instance()
    result_after_restart = project_with_new_repository_instance()
    assert result.coverage == 0.5
    assert result_after_restart == result
    assert result.information_gain == 0.2
    assert run.quality_snapshot == {}


def test_budget_projection_is_durable_and_idempotent_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_ref = deterministic_attempt_ref(_WORK, 1)
    durable = {
        "run": SimpleNamespace(
            id=_RUN,
            quality_snapshot={"coverage": 0.25, "low_information_gain_streak": 2},
            usage_snapshot={},
            budget_snapshot={},
            plan_version=1,
            next_event_seq=1,
            phase="RESEARCHING",
            status="RUNNING",
            termination_reason=None,
        ),
        "plan_item": SimpleNamespace(status="active"),
        "events": [],
    }

    async def locked_run(_session: object, _run_id: UUID, _worker: str) -> object:
        return durable["run"]

    monkeypatch.setattr(
        ResearchToolRepository,
        "_locked_run",
        staticmethod(locked_run),
    )

    class _Rows:
        def __init__(self, rows: list[object]) -> None:
            self.rows = rows

        def all(self) -> list[object]:
            return self.rows

    class _Session:
        def __init__(self) -> None:
            self.scalar_index = 0

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> _Session:
            return self

        async def scalar(self, _statement: object) -> object:
            self.scalar_index += 1
            if self.scalar_index == 1:
                return next(
                    (
                        event
                        for event in reversed(durable["events"])
                        if event.event_type == "research.work_evaluation_projected"
                        and event.refs.get("attempt_ref") == str(attempt_ref)
                    ),
                    None,
                )
            if self.scalar_index == 2:
                return durable["plan_item"]
            return 0

        async def get(self, _model: object, _identity: UUID) -> None:
            return None

        def add(self, event: object) -> None:
            durable["events"].append(event)

        async def flush(self) -> None:
            return None

    def apply_from_fresh_repository() -> object:
        repository = ResearchToolRepository(lambda: _Session())  # type: ignore[arg-type]
        return _run_immediate(
            repository._project_iteration(
                _RUN,
                worker_task_id="worker",
                target=ResearchTarget(
                    plan_version=1,
                    question_id="q1",
                    question="question",
                    query="query",
                    gap_id=UUID("55555555-5555-4555-8555-555555555555"),
                    tool_call_id=UUID("66666666-6666-4666-8666-666666666666"),
                    source_id_seed=UUID("77777777-7777-4777-8777-777777777777"),
                    execution_attempt_ref=attempt_ref,
                ),
                attempt_outcome="question_budget_exhausted",
                legacy_control_flow=False,
            )
        )

    first_result = apply_from_fresh_repository()
    first_snapshot = (
        durable["plan_item"].status,
        dict(durable["run"].quality_snapshot),
        dict(durable["run"].usage_snapshot),
        durable["run"].status,
        durable["run"].phase,
        durable["run"].termination_reason,
        len(durable["events"]),
    )
    assert first_snapshot[0] == "blocked"
    assert any(
        event.event_type == "research.work_evaluation_projected"
        for event in durable["events"]
    )

    # Simulate process restart by serializing/reconstructing durable records.
    durable["run"] = SimpleNamespace(
        id=_RUN,
        quality_snapshot=dict(durable["run"].quality_snapshot),
        usage_snapshot=dict(durable["run"].usage_snapshot),
        budget_snapshot=dict(durable["run"].budget_snapshot),
        plan_version=1,
        next_event_seq=durable["run"].next_event_seq,
        phase=durable["run"].phase,
        status=durable["run"].status,
        termination_reason=durable["run"].termination_reason,
    )
    durable["plan_item"] = SimpleNamespace(status=durable["plan_item"].status)
    second_result = apply_from_fresh_repository()
    second_snapshot = (
        durable["plan_item"].status,
        dict(durable["run"].quality_snapshot),
        dict(durable["run"].usage_snapshot),
        durable["run"].status,
        durable["run"].phase,
        durable["run"].termination_reason,
        len(durable["events"]),
    )
    assert second_result == first_result
    assert second_snapshot == first_snapshot
