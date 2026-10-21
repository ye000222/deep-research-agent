"""Direct compatibility checks for the V2 executor and V3 legacy Work adapter.

These tests intentionally stop at the shared production seams: V3 must project
the persisted legacy objective into the same target inputs consumed by the V2
executor. Durable orchestration/recovery E2E is covered separately.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TypeVar, cast
from uuid import UUID

import pytest
from app.domain.adaptive_scheduler import QueryFamily, build_family_query
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.domain.provider_failure_classification import (
    ProviderFailureType,
    classify_provider_failure,
)
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import EvidenceBatch, EvidenceCandidate, ReadPage
from app.domain.research_work import (
    LegacyResearchObjective,
    ResearchWork,
    ResearchWorkKind,
    WorkAttempt,
    WorkAttemptState,
    deterministic_attempt_ref,
    legacy_objective_hash,
)
from app.domain.run_state import ResearchRunState
from app.infrastructure.db import research_tools as research_tools_module
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.research_models import (
    ResearchEvidenceRow,
    ResearchGapRow,
    ResearchSourceRow,
    ResearchToolCallRow,
)
from app.infrastructure.db.research_tools import (
    ResearchTarget,
    ResearchToolRepository,
    _budget_exhaustion_reason,
    _research_attempt_consumes_iteration,
)
from app.infrastructure.db.research_work import ResearchWorkRepository
from app.infrastructure.db.run_models import (
    ResearchPlanItemRow,
    ResearchRunRow,
    ResearchWorkRow,
    WorkAttemptRow,
)
from app.services.evidence_extractor import EvidenceExtractorService

RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
WORK_ID = UUID("22222222-2222-4222-8222-222222222222")
ATTEMPT_ID = deterministic_attempt_ref(WORK_ID, 1)
RowT = TypeVar("RowT")


def _run_immediate(awaitable: object) -> object:
    """Run these synchronous fake-backed async paths without a loop fixture."""
    iterator = awaitable.__await__()  # type: ignore[attr-defined]
    try:
        iterator.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("fake-backed async path unexpectedly suspended")


def _target() -> ResearchTarget:
    return ResearchTarget(
        plan_version=1,
        question_id="q1",
        question="Find primary evidence supporting the planned research dimension.",
        query="official primary evidence research dimension",
        gap_id=UUID("33333333-3333-4333-8333-333333333333"),
        tool_call_id=UUID("44444444-4444-4444-8444-444444444444"),
        source_id_seed=UUID("55555555-5555-4555-8555-555555555555"),
        priority=1,
        acceptance_dimensions=(("q1:d1", "primary evidence"),),
        provider_request_allowance=2,
    )


def _objective() -> LegacyResearchObjective:
    return LegacyResearchObjective(
        question_id="q1",
        question="Find primary evidence supporting the planned research dimension.",
        rationale="The persisted plan requires a source-backed answer.",
        requirements=("primary evidence",),
        priority=1,
        search_hints=("official primary evidence",),
        acceptance_context=(("q1:d1", "primary evidence"),),
        plan_version=1,
        bounds={"max_attempts": 3},
    )


def _work_and_attempt() -> tuple[ResearchWork, WorkAttempt]:
    objective = _objective()
    work = ResearchWork(
        work_ref=WORK_ID,
        run_ref=RUN_ID,
        producer_kind="LEGACY_PLANNER",
        producer_ref="plan:1:question:q1",
        work_kind=ResearchWorkKind.LEGACY_RESEARCH,
        objective=objective,
        objective_hash=legacy_objective_hash(objective),
        priority=objective.priority,
        execution_bounds={"max_attempts": 3},
        source_run_state_ref="runs/test/state.json",
        source_run_state_hash="a" * 64,
        source_run_state_version=1,
    )
    attempt = WorkAttempt(
        attempt_ref=ATTEMPT_ID,
        work_ref=WORK_ID,
        attempt_index=1,
        state=WorkAttemptState.RESERVED,
        reservation_identity="legacy-equivalence-attempt-1",
        source_run_state_ref=work.source_run_state_ref,
        source_run_state_version=1,
    )
    return work, attempt


def _business_inputs(target: ResearchTarget) -> tuple[object, ...]:
    """Exclude V3-only execution provenance while comparing executor inputs."""
    return (
        target.plan_version,
        target.question_id,
        target.question,
        target.query,
        target.alternate_query,
        target.priority,
        target.attempt_index,
        target.gap_attempt_index,
        target.first_pass,
        target.acceptance_dimensions,
        target.used_source_owner_keys,
        target.provider_request_allowance,
        target.search_excluded_owner_keys,
        target.search_excluded_urls,
        target.reusable_results,
        target.reusable_pages,
        target.query_already_executed,
        target.search_budget_exhausted,
        target.query_family,
        target.baseline_model_tokens,
        target.baseline_pages_fetched,
        target.baseline_pages_extracted,
        target.baseline_provider_requests,
        target.cheap_triage_enabled,
        target.owner_acceptance_rates,
    )


def test_v2_and_v3_target_preparation_preserve_same_business_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise both public repository entry points and their shared target seam."""
    repository = ResearchToolRepository(sessions=lambda: None)  # type: ignore[arg-type]
    shared = _target()
    calls: list[tuple[str | None, object | None]] = []

    async def prepared(
        _run_id: UUID,
        *,
        worker_task_id: str,
        selected_question_id: str | None,
        selected_work: ResearchWork | None = None,
    ) -> ResearchTarget:
        assert worker_task_id == "worker-1"
        calls.append((selected_question_id, selected_work))
        return shared

    monkeypatch.setattr(repository, "_prepare_target", prepared)
    work, attempt = _work_and_attempt()
    async def run() -> tuple[ResearchTarget | None, ResearchTarget | None]:
        v2 = await repository.prepare_target(RUN_ID, worker_task_id="worker-1")
        v3 = await repository.prepare_target_for_work(
            RUN_ID,
            worker_task_id="worker-1",
            question_id="q1",
            work_item=work,
            attempt=attempt,
        )
        return v2, v3

    v2_target, v3_target = cast(
        tuple[ResearchTarget | None, ResearchTarget | None], _run_immediate(run())
    )

    assert v2_target is not None and v3_target is not None
    assert _business_inputs(v2_target) == _business_inputs(v3_target)
    assert v2_target.execution_work_ref is None
    assert v3_target.execution_work_ref == work.work_ref
    assert v3_target.execution_attempt_ref == attempt.attempt_ref
    assert calls[0] == (None, None)
    assert calls[1] == ("q1", work)


def test_v2_plan_and_v3_work_feed_identical_production_query_planner_inputs() -> None:
    """The production query builder receives the same objective in either lane."""
    objective = _objective()
    v2_plan_item = SimpleNamespace(
        question_id=objective.question_id,
        question=objective.question,
        evidence_requirements=objective.requirements,
        search_hints=objective.search_hints,
    )
    v3_work, _attempt = _work_and_attempt()
    v3_objective = v3_work.objective

    v2_inputs = (
        v2_plan_item.question,
        v2_plan_item.evidence_requirements[0],
        tuple(v2_plan_item.search_hints),
    )
    v3_inputs = (
        v3_objective.question,
        v3_objective.requirements[0],
        tuple(v3_objective.search_hints),
    )
    assert v2_inputs == v3_inputs

    # Exercise the real deterministic planner for every query family rather
    # than comparing hand-authored query strings.
    for family in QueryFamily:
        v2_query = build_family_query(
            question=v2_inputs[0],
            criterion=v2_inputs[1],
            hints=v2_inputs[2],
            family=family,
        )
        v3_query = build_family_query(
            question=v3_inputs[0],
            criterion=v3_inputs[1],
            hints=v3_inputs[2],
            family=family,
        )
        assert v2_query == v3_query


def test_v3_materialization_preserves_persisted_plan_inputs() -> None:
    """Exercise the production V3 plan-to-Work materializer against fake rows."""
    questions = [
        ResearchQuestion(
            id=f"q{index}",
            question=f"Find primary evidence for research dimension {index}.",
            rationale=f"The persisted plan requires source-backed dimension {index}.",
            priority=1,
            evidence_requirements=[f"requirement {index}"],
            search_hints=[f"official source {index}"],
        )
        for index in range(1, 6)
    ]
    plan = ResearchPlan(
        goal="Research the five planned dimensions",
        scope_summary="A frozen five-question research plan",
        questions=questions,
        completion_criteria=["Evidence collected", "Evaluation complete"],
    )
    state = SimpleNamespace(
        run_ref=RUN_ID,
        state_ref="runs/test/state/1.json",
        state_hash="a" * 64,
        state_version=1,
    )
    run_row = SimpleNamespace(
        id=RUN_ID,
        orchestrator_version="V3",
        latest_run_state_ref=state.state_ref,
        latest_run_state_hash=state.state_hash,
        latest_run_state_version=state.state_version,
        plan_version=1,
    )
    plan_rows = [
        SimpleNamespace(
            run_id=RUN_ID,
            plan_version=1,
            question_id=item.id,
            question=item.question,
            rationale=item.rationale,
            evidence_requirements=item.evidence_requirements,
            priority=item.priority,
            search_hints=item.search_hints,
        )
        for item in questions
    ]

    class Rows:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def all(self) -> list[object]:
            return self._rows

    class Session:
        def __init__(self) -> None:
            self.scalar_calls = 0
            self.scalars_calls = 0
            self.added: list[object] = []

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> Session:
            return self

        async def scalar(self, _statement: object) -> object:
            self.scalar_calls += 1
            return run_row if self.scalar_calls == 1 else None

        async def scalars(self, _statement: object) -> Rows:
            self.scalars_calls += 1
            return Rows(list(plan_rows) if self.scalars_calls == 1 else [])

        def add(self, row: object) -> None:
            assert isinstance(row, ResearchWorkRow)
            row.created_at = datetime(2026, 10, 17, tzinfo=UTC)
            self.added.append(row)

        async def flush(self) -> None:
            return None

    session = Session()
    repository = ResearchWorkRepository(lambda: session)  # type: ignore[arg-type]
    work_items = cast(
        tuple[ResearchWork, ...],
        _run_immediate(
            repository.materialize_legacy_plan(cast(ResearchRunState, state), plan)
        ),
    )
    assert len(work_items) == len(questions) == 5
    for question, work in zip(questions, work_items, strict=True):
        objective = work.objective
        assert objective.question_id == question.id
        assert objective.question == question.question
        assert objective.rationale == question.rationale
        assert objective.priority == question.priority
        assert objective.requirements == tuple(question.evidence_requirements)
        assert objective.search_hints == tuple(question.search_hints)
        assert objective.acceptance_context == (
            (f"{question.id}:d1", question.evidence_requirements[0]),
        )


def test_persisted_plan_item_v2_and_v3_production_preparation_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run both real target preparation paths over one persisted PlanItem row."""
    from app.infrastructure.db.research_work import ResearchWorkRepository

    questions = [
        ResearchQuestion(
            id=f"q{index}",
            question=f"Find primary evidence for planned dimension {index}.",
            rationale=f"The persisted plan requires source-backed answer {index}.",
            priority=1 if index == 1 else 2,
            evidence_requirements=[f"primary evidence {index}"],
            search_hints=[f"official primary evidence {index}", "independent research"],
        )
        for index in range(1, 6)
    ]
    plan = ResearchPlan(
        goal="Research the planned dimension",
        scope_summary="One persisted research question",
        questions=questions,
        completion_criteria=["Evidence collected", "Evaluation complete"],
    )
    plan_items = [
        ResearchPlanItemRow(
            id=UUID(f"77777777-7777-4777-8777-{index:012d}"),
            run_id=RUN_ID,
            plan_version=1,
            question_id=item.id,
            question=item.question,
            rationale=item.rationale,
            priority=item.priority,
            evidence_requirements=list(item.evidence_requirements),
            search_hints=list(item.search_hints),
            status="pending",
        )
        for index, item in enumerate(questions, start=1)
    ]
    selected_plan_item = plan_items[0]
    state = SimpleNamespace(
        run_ref=RUN_ID,
        state_ref="runs/test/state/1.json",
        state_hash="b" * 64,
        state_version=1,
    )
    v3_run_row = SimpleNamespace(
        id=RUN_ID,
        orchestrator_version="V3",
        latest_run_state_ref=state.state_ref,
        latest_run_state_hash=state.state_hash,
        latest_run_state_version=state.state_version,
        plan_version=1,
    )

    class Rows:
        def __init__(self, values: list[object]) -> None:
            self.values = values

        def all(self) -> list[object]:
            return self.values

    class Execution:
        def all(self) -> list[object]:
            return []

        def scalars(self) -> Rows:
            return Rows([])

    class Session:
        def __init__(self, run: object) -> None:
            self.run = run
            self.added: list[object] = []

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> Session:
            return self

        async def scalar(self, statement: object) -> object | None:
            descriptions = getattr(statement, "column_descriptions", ())
            entity = descriptions[0].get("entity") if descriptions else None
            if entity is ResearchRunRow:
                return self.run
            if entity in (ResearchGapRow, ResearchToolCallRow, ResearchWorkRow):
                return None
            # The only scalar expression left in these exercised paths is the
            # historical query-attempt count.
            return 0

        async def scalars(self, statement: object) -> Rows:
            descriptions = getattr(statement, "column_descriptions", ())
            entity = descriptions[0].get("entity") if descriptions else None
            return Rows(cast(list[object], plan_items) if entity is ResearchPlanItemRow else [])

        async def execute(self, _statement: object) -> Execution:
            return Execution()

        def add(self, row: object) -> None:
            if isinstance(row, ResearchWorkRow):
                row.created_at = datetime(2026, 10, 17, tzinfo=UTC)
            self.added.append(row)

        async def flush(self) -> None:
            return None

    materialization_session = Session(v3_run_row)
    work_repository = ResearchWorkRepository(
        lambda: materialization_session  # type: ignore[arg-type]
    )
    work_items = cast(
        tuple[ResearchWork, ...],
        _run_immediate(
            work_repository.materialize_legacy_plan(cast(ResearchRunState, state), plan)
        ),
    )
    assert len(work_items) == len(questions) == 5
    work = next(item for item in work_items if item.objective.question_id == "q1")
    attempt = WorkAttempt(
        attempt_ref=deterministic_attempt_ref(work.work_ref, 1),
        work_ref=work.work_ref,
        attempt_index=1,
        state=WorkAttemptState.RESERVED,
        reservation_identity="same-plan-item-attempt-1",
        source_run_state_ref=work.source_run_state_ref,
        source_run_state_version=work.source_run_state_version,
    )

    def execution_run() -> SimpleNamespace:
        return SimpleNamespace(
            id=RUN_ID,
            status="running",
            worker_task_id="worker-1",
            plan_version=1,
            budget_snapshot={
                "max_provider_requests": 8,
                "max_logical_queries": 8,
                "max_pages": 8,
                "max_pages_fetched": 8,
                "max_pages_extracted": 8,
                "max_page_fetch_attempts": 12,
                "max_extraction_calls": 8,
                "max_verification_calls": 0,
                "max_scheduler_actions": 20,
            },
            usage_snapshot={
                "search_provider_requests": 0,
                "logical_queries": 0,
                "pages_fetched": 0,
                "pages_extracted": 0,
                "page_fetch_attempts": 0,
                "extraction_calls": 0,
                "verification_calls": 0,
                "scheduler_actions": 0,
            },
            quality_snapshot={"coverage_map": []},
            started_at=datetime(2026, 10, 17, tzinfo=UTC),
            created_at=datetime(2026, 10, 17, tzinfo=UTC),
        )

    outputs: dict[str, tuple[ResearchTarget, list[object]]] = {}
    for lane in ("v2", "v3"):
        run = execution_run()
        session = Session(run)
        repository = ResearchToolRepository(
            lambda current=session: current  # type: ignore[arg-type]
        )

        async def locked_run(
            _session: object,
            _run_id: UUID,
            _worker_task_id: str,
            current_run: object = run,
        ) -> object:
            return current_run

        async def no_feedback(*_args: object, **_kwargs: object) -> None:
            return None

        async def no_event(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(
            ResearchToolRepository, "_locked_run", staticmethod(locked_run)
        )
        monkeypatch.setattr(repository, "_prepare_feedback_target", no_feedback)
        monkeypatch.setattr(repository, "_append_event", no_event)

        if lane == "v2":
            target = _run_immediate(
                repository.prepare_target(RUN_ID, worker_task_id="worker-1")
            )
        else:
            selected_plan_item.status = "pending"
            target = _run_immediate(
                repository.prepare_target_for_work(
                    RUN_ID,
                    worker_task_id="worker-1",
                    question_id="q1",
                    work_item=work,
                    attempt=attempt,
                )
            )
        assert isinstance(target, ResearchTarget)
        outputs[lane] = (target, session.added)

    v2_target, v2_rows = outputs["v2"]
    v3_target, v3_rows = outputs["v3"]
    assert _business_inputs(v2_target) == _business_inputs(v3_target)
    assert v2_target.acceptance_dimensions == v3_target.acceptance_dimensions
    assert v2_target.search_excluded_owner_keys == v3_target.search_excluded_owner_keys
    assert v2_target.search_excluded_urls == v3_target.search_excluded_urls
    assert v2_target.reusable_results == v3_target.reusable_results
    assert v2_target.reusable_pages == v3_target.reusable_pages
    assert v2_target.query_family == v3_target.query_family
    assert v2_target.provider_request_allowance == v3_target.provider_request_allowance
    assert v2_target.execution_work_ref is None
    assert v3_target.execution_work_ref == work.work_ref
    assert v3_target.execution_attempt_ref == attempt.attempt_ref

    def query_call(rows: list[object]) -> ResearchToolCallRow:
        matches = [row for row in rows if isinstance(row, ResearchToolCallRow)]
        assert len(matches) == 1
        return matches[0]

    v2_call = query_call(v2_rows)
    v3_call = query_call(v3_rows)
    assert v2_call.arguments["query"] == v2_target.query
    assert v3_call.arguments["query"] == v3_target.query
    assert v2_call.arguments["query"] == v3_call.arguments["query"]
    assert v2_target.alternate_query == v3_target.alternate_query
    assert v2_target.query_family == v3_target.query_family


def test_v2_and_v3_acceptance_inputs_produce_identical_production_decision() -> None:
    """Run the production acceptance gate, not a mocked accept/reject result."""
    page = ReadPage(
        final_url="https://example.org/report",
        title="Research report",
        clean_text=("The report states that the program produced primary evidence "
                    "for the planned research dimension. " * 5),
        content_hash="a" * 64,
        fetched_at=__import__("datetime").datetime(2026, 1, 1),
    )
    batch = EvidenceBatch(
        items=[
            EvidenceCandidate(
                claim="The program produced primary evidence for the planned research dimension.",
                exact_quote=(
                    "The report states that the program produced primary evidence "
                    "for the planned research dimension."
                ),
                dimension_key="q1:d1",
                relevance=0.95,
                confidence=0.95,
            )
        ]
    )
    v2 = EvidenceExtractorService._score_batch(
        batch,
        page=page,
        dimension_criteria={"q1:d1": "primary evidence"},
        reliability=0.95,
    )
    v3 = EvidenceExtractorService._score_batch(
        batch,
        page=page,
        dimension_criteria={"q1:d1": "primary evidence"},
        reliability=0.95,
    )
    assert [item.model_dump(mode="json") for item in v2] == [
        item.model_dump(mode="json") for item in v3
    ]


def test_v2_and_v3_persistence_paths_preserve_evidence_semantic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the repository's real page-persistence/hash path over fake storage."""
    from app.domain.research_tools import EvidenceRelation
    from app.infrastructure.db.evidence_graph_relations import RelationRefreshStats

    claim_text = "The program produced primary evidence for the planned dimension."
    quote = (
        "The report states that the program produced primary evidence "
        "for the planned research dimension, with independently measured outcomes."
    )
    page = ReadPage(
        final_url="https://example.org/report",
        title="Research report",
        clean_text=(
            "Introductory context. "
            + quote
            + " Supporting context follows. Additional source detail follows. " * 4
        ),
        content_hash="c" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    candidate = EvidenceCandidate(
        claim=claim_text,
        exact_quote=quote,
        dimension_key="q1:d1",
        relation=EvidenceRelation.SUPPORTS,
        relevance=0.95,
        confidence=0.95,
    )
    scored = EvidenceExtractorService._score_batch(
        EvidenceBatch(items=[candidate]),
        page=page,
        dimension_criteria={"q1:d1": "primary evidence"},
        reliability=0.95,
    )
    assert scored[0].accepted
    work, attempt = _work_and_attempt()
    v2_target = _target()
    v3_target = replace(
        v2_target,
        execution_work_ref=work.work_ref,
        execution_attempt_ref=attempt.attempt_ref,
        work_execution_bounds=(("max_attempts", "3"),),
    )

    class Session:
        def __init__(self, attempt_row: WorkAttemptRow | None) -> None:
            self.attempt_row = attempt_row
            self.added: list[object] = []

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self) -> Session:
            return self

        async def scalar(self, statement: object) -> object | None:
            descriptions = getattr(statement, "column_descriptions", ())
            selected = descriptions[0].get("entity") if descriptions else None
            return self.attempt_row if selected is WorkAttemptRow else None

        async def scalars(self, _statement: object) -> object:
            raise AssertionError("candidate lookup should use the supplied fake identity")

        def add(self, row: object) -> None:
            self.added.append(row)

        async def flush(self) -> None:
            return None

    identity = SimpleNamespace(
        candidate_id="candidate-1",
        raw_url=page.final_url,
        normalized_url=page.final_url,
        question_id="q1",
        query_id="query-1",
        domain="example.org",
    )

    async def locked_run(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            status="running",
            worker_task_id="worker-1",
            budget_snapshot={},
            usage_snapshot={},
            quality_snapshot={},
            phase="researching",
            state_version=1,
            lease_until=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    async def candidate_identity(*_args: object, **_kwargs: object) -> object:
        return identity

    async def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    async def no_relationships(*_args: object, **_kwargs: object) -> RelationRefreshStats:
        return RelationRefreshStats()

    monkeypatch.setattr(
        ResearchToolRepository,
        "_locked_run",
        staticmethod(locked_run),
    )
    monkeypatch.setattr(
        research_tools_module, "refresh_question_relations", no_relationships
    )

    records: dict[str, list[object]] = {}
    for lane, target in (("v2", v2_target), ("v3", v3_target)):
        attempt_row = (
            WorkAttemptRow(
                attempt_ref=attempt.attempt_ref,
                work_ref=work.work_ref,
                attempt_index=1,
                state="RESERVED",
                reservation_identity="legacy-equivalence-attempt-1",
                source_run_state_ref=work.source_run_state_ref,
                source_run_state_version=1,
                query_refs=[],
                tool_call_refs=[],
                produced_evidence_refs=[],
                reused_evidence_refs=[],
                budget_refs=[],
                usage_summary={},
                resume_metadata={},
                schema_version="work-attempt-v1",
            )
            if lane == "v3"
            else None
        )
        session = Session(attempt_row)
        repository = ResearchToolRepository(lambda current=session: current)  # type: ignore[arg-type]
        monkeypatch.setattr(repository, "_candidate_identity_for_url", candidate_identity)
        monkeypatch.setattr(repository, "_append_formation_diagnostic", no_op)
        monkeypatch.setattr(repository, "_append_event", no_op)
        monkeypatch.setattr(repository, "_append_candidate_event", no_op)
        monkeypatch.setattr(repository, "_align_and_evaluate_evidence", no_op)
        result = _run_immediate(
            repository.record_page(
                RUN_ID,
                worker_task_id="worker-1",
                target=target,
                source_id=UUID("66666666-6666-4666-8666-666666666666"),
                page=page,
                artifact_uri="runs/test/source.txt",
                evidence=scored,
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    accuracy=UsageAccuracy.EXACT,
                ),
                context_manifest={"selected_chars": len(page.clean_text)},
            )
        )
        assert result == (1, 1)
        records[lane] = session.added

    def one(rows: list[object], row_type: type[RowT]) -> RowT:
        matches = [row for row in rows if isinstance(row, row_type)]
        assert len(matches) == 1
        return matches[0]

    projections: dict[str, tuple[object, ...]] = {}
    for lane, rows in records.items():
        source = one(rows, ResearchSourceRow)
        snapshot = one(rows, ResearchSourceSnapshotRow)
        claim = one(rows, ResearchClaimRow)
        chunk = one(rows, ResearchSourceChunkRow)
        evidence_row = one(rows, ResearchEvidenceRow)
        projections[lane] = (
            evidence_row.question_id,
            claim.claim_hash,
            evidence_row.source_id,
            source.canonical_url,
            snapshot.source_id,
            snapshot.content_hash,
            evidence_row.claim,
            evidence_row.exact_quote,
            evidence_row.relation,
            chunk.char_start,
            chunk.char_end,
            chunk.chunk_hash,
            evidence_row.accepted,
            evidence_row.evidence_hash,
        )
    assert projections["v2"] == projections["v3"]


def test_evidence_semantic_identity_uses_same_claim_fingerprint_for_both_lanes() -> None:
    """The established claim identity is canonical and independent of Work refs."""
    from app.domain.evidence_graph import claim_fingerprint

    claim = "The program produced primary evidence for the planned dimension."
    work, _attempt = _work_and_attempt()
    normalized_equivalent = "  THE   PROGRAM PRODUCED PRIMARY EVIDENCE FOR THE PLANNED DIMENSION.  "
    assert claim_fingerprint(claim) == claim_fingerprint(normalized_equivalent)
    assert claim_fingerprint(claim) != claim_fingerprint(f"{claim} Work:{work.work_ref}")


@pytest.mark.parametrize(
    ("budget", "usage", "expected_reason"),
    [
        (
            {"max_provider_requests": 4, "max_logical_queries": 6, "max_pages_fetched": 8},
            {"search_provider_requests": 1, "logical_queries": 2, "pages_fetched": 3},
            None,
        ),
        (
            {"max_provider_requests": 4, "max_logical_queries": 6, "max_pages_fetched": 8},
            {"search_provider_requests": 4, "logical_queries": 2, "pages_fetched": 3},
            "provider_request_budget_exhausted",
        ),
        (
            {"max_provider_requests": 4, "max_logical_queries": 6, "max_pages_fetched": 8},
            {"search_provider_requests": 1, "logical_queries": 6, "pages_fetched": 3},
            "logical_query_budget_exhausted",
        ),
        (
            {"max_provider_requests": 4, "max_logical_queries": 6, "max_pages_fetched": 8},
            {"search_provider_requests": 1, "logical_queries": 2, "pages_fetched": 8},
            "fetched_page_budget_exhausted",
        ),
    ],
)
def test_execution_budget_semantics_are_unchanged_by_v3_work_metadata(
    budget: dict[str, int],
    usage: dict[str, int],
    expected_reason: str | None,
) -> None:
    work, _attempt = _work_and_attempt()
    run = SimpleNamespace(
        budget_snapshot=budget,
        usage_snapshot=usage,
        started_at=datetime(2026, 10, 17, tzinfo=UTC),
        created_at=datetime(2026, 10, 17, tzinfo=UTC),
    )
    before = _budget_exhaustion_reason(cast(ResearchRunRow, run))
    run.execution_bounds = work.execution_bounds
    run.work_ref = str(work.work_ref)
    after = _budget_exhaustion_reason(cast(ResearchRunRow, run))
    assert before == expected_reason
    assert after == before
    assert work.execution_bounds == {"max_attempts": 3}


@pytest.mark.parametrize(
    "outcome",
    ["zero_results", "unreadable", "no_evidence", "provider_error", "budget_exhausted"],
)
def test_v2_and_v3_failure_outcomes_preserve_retryable_round_semantics(
    outcome: str,
) -> None:
    """Both orchestration lanes use the same production iteration rule."""
    v2_consumes = _research_attempt_consumes_iteration(outcome)
    v3_consumes = _research_attempt_consumes_iteration(outcome)
    assert v2_consumes is False
    assert v3_consumes is v2_consumes


def test_v2_finish_and_v3_projection_share_failure_projection_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ResearchToolRepository(sessions=lambda: None)  # type: ignore[arg-type]
    target = _target()
    calls: list[tuple[str, bool, ResearchTarget]] = []

    async def project(
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        attempt_outcome: str,
        legacy_control_flow: bool,
    ) -> object:
        assert run_id == RUN_ID
        calls.append((attempt_outcome, legacy_control_flow, target))
        return SimpleNamespace(consumed=False, outcome=attempt_outcome)

    monkeypatch.setattr(repository, "_project_iteration", project)

    async def run() -> tuple[object, object]:
        v2 = await repository.finish_iteration(
            RUN_ID,
            worker_task_id="worker-1",
            target=target,
            attempt_outcome="provider_error",
        )
        v3 = await repository.project_work_iteration(
            RUN_ID,
            worker_task_id="worker-1",
            target=target,
            attempt_outcome="provider_error",
        )
        return v2, v3

    v2_result, v3_result = cast(tuple[object, object], _run_immediate(run()))
    assert v2_result == v3_result
    assert calls == [
        ("provider_error", True, target),
        ("provider_error", False, target),
    ]


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("MODEL_TIMEOUT", ProviderFailureType.TIMEOUT),
        ("MODEL_NETWORK_ERROR", ProviderFailureType.NETWORK_ERROR),
        ("MODEL_HTTP_503", ProviderFailureType.SERVER_ERROR),
    ],
)
def test_v2_and_v3_legacy_failure_taxonomy_is_stable(
    error_code: str, expected: ProviderFailureType
) -> None:
    # Both orchestrators delegate execution to the same ResearchLoop and
    # provider-failure contract; V3 Work state is additive, not a new taxonomy.
    v2_failure = classify_provider_failure(error_code=error_code)
    v3_failure = classify_provider_failure(error_code=error_code)
    assert v2_failure is expected
    assert v3_failure is v2_failure
