"""Quick first-evidence regressions; SQLite memory only, no external boundaries."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.domain.research_budget import allocate_research_call, estimate_question_budgets
from app.domain.research_management import allocate_budget_shares
from app.domain.research_runs import RunPhase, RunStatus
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_tools import ResearchToolRepository, _writing_start_summary
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.research_loop import ResearchLoopService
from app.tools.errors import ToolExecutionError
from sqlalchemy import func, select

from tests.integration.test_v3_deterministic_e2e import (
    _new_store,
    _seed_run,
    _sqlite_utc_datetime_compatibility,  # noqa: F401 -- imported SQLite fixture
)
from tests.unit.test_research_loop import (
    FakeArtifacts,
    FakeExtractor,
    FakeReader,
    FakeRepository,
    FakeSearch,
)


@pytest.fixture
def store(tmp_path):
    factory, _artifacts = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, "Offline first evidence budget regression")
    pools = allocate_budget_shares(
        max_tokens=30000,
        max_iterations=8,
        max_searches=6,
        max_pages=5,
    )
    targets = estimate_question_budgets(
        [{"id": f"q{i}"} for i in range(1, 8)],
        max_tokens=pools["research_tokens"],
    )
    pools["question_token_budgets"] = {q.question_id: q.target_tokens for q in targets}
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        run.status = RunStatus.RUNNING.value
        run.phase = RunPhase.RESEARCHING.value
        run.worker_task_id = "offline-worker"
        run.lease_until = datetime.now(UTC) + timedelta(minutes=5)
        run.plan_version = 1
        run.budget_snapshot = {
            **run.budget_snapshot,
            "max_tokens": 30000,
            "allocation": pools,
            "max_evidence_call_tokens": 12000,
        }
        for i in range(1, 8):
            session.add(
                ResearchPlanItemRow(
                    id=uuid4(),
                    run_id=run_id,
                    plan_version=1,
                    question_id=f"q{i}",
                    question=f"What is the source statement for question {i}?",
                    priority=1,
                    rationale="Preserve first pass fairness",
                    evidence_requirements=["A direct quotation"],
                    search_hints=[],
                    status="pending",
                )
            )
        session.commit()
    return factory, run_id


def add_event(store, event_type, question="q1"):
    factory, run_id = store
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        session.add(
            AgentEventRow(
                run_id=run_id,
                run_seq=run.next_event_seq,
                schema_version=1,
                created_at=datetime.now(UTC),
                phase=run.phase,
                event_type=event_type,
                public_summary="Synthetic offline event",
                refs={"question_id": question},
            )
        )
        run.next_event_seq += 1
        session.commit()


async def reserve(store, amount=3125, question="q1", attempt=None):
    factory, run_id = store
    return await ResearchToolRepository(factory).reserve_model_tokens(
        run_id,
        worker_task_id="offline-worker",
        question_id=question,
        node="evidence_extractor",
        attempt_id=attempt or uuid4(),
        estimated_input=amount - 800,
        max_output=800,
        minimum_call_tokens=2700,
    )


@pytest.mark.asyncio
async def test_readable_underfunded_question_gets_real_atomic_first_reservation(store):
    factory, run_id = store
    add_event(store, "source.readable")
    budget = await ResearchToolRepository(factory).evidence_model_budget(
        run_id,
        worker_task_id="offline-worker",
        question_id="q1",
        minimum_call=2700,
    )
    assert budget.allowed and budget.max_call_tokens == 3125
    attempt = uuid4()
    result = await reserve(store, budget.max_call_tokens, attempt=attempt)
    assert result.granted
    with factory._sync_factory() as session:
        rows = session.scalars(select(ModelBudgetReservationRow)).all()
        assert len(rows) == 1 and rows[0].reserved_total == 3125
        assert rows[0].status == "reserved"
        run = session.get(ResearchRunRow, run_id)
        assert run.budget_snapshot["max_tokens"] == 30000
        assert run.budget_snapshot["allocation"]["question_token_budgets"]["q1"] == 1785
        event = session.scalar(
            select(AgentEventRow).where(
                AgentEventRow.event_type == "budget.question_first_pass_allocated",
            )
        )
        assert event.metrics["future_question_reserve_tokens"] == 8100
    # The same attempt is idempotent, not another reservation.
    assert (await reserve(store, attempt=attempt)).granted
    with factory._sync_factory() as session:
        assert session.scalar(select(func.count(ModelBudgetReservationRow.id))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [12500, 3126])
async def test_first_call_cannot_take_other_affordable_seats(store, amount):
    add_event(store, "source.readable")
    assert not (await reserve(store, amount=amount)).granted


@pytest.mark.asyncio
async def test_global_budget_is_still_hard(store):
    factory, run_id = store
    add_event(store, "source.readable")
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        run.usage_snapshot = {"model_tokens": 29000}
        session.commit()
    assert not (await reserve(store)).granted


@pytest.mark.asyncio
async def test_affordable_continuation_reserves_only_remaining_nominal_target(store):
    factory, run_id = store
    add_event(store, "source.readable")
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        allocation = dict(run.budget_snapshot["allocation"])
        allocation["question_token_budgets"] = {
            **allocation["question_token_budgets"],
            "q1": 4000,
        }
        run.budget_snapshot = {**run.budget_snapshot, "allocation": allocation}
        session.commit()
    first = uuid4()
    assert (await reserve(store, attempt=first)).granted
    actual = ResearchToolRepository(factory)
    await actual.settle_model_reservation(
        run_id,
        worker_task_id="offline-worker",
        attempt_id=first,
        actual_total=698,
        usage_estimated=False,
    )
    budget = await actual.evidence_model_budget(
        run_id,
        worker_task_id="offline-worker",
        question_id="q1",
        minimum_call=2700,
    )
    assert budget.allowed and 2700 <= budget.max_call_tokens <= 4000 - 698
    assert (await reserve(store, amount=budget.max_call_tokens)).granted
    with factory._sync_factory() as session:
        rows = session.scalars(select(ModelBudgetReservationRow)).all()
        committed = sum(
            r.actual_total if r.actual_total is not None else r.reserved_total for r in rows
        )
        assert committed <= 4000  # no repeat first-pass borrowing


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["deadline_exhausted", "quality_met", "token_budget_exhausted", "stagnation"]
)
async def test_every_writing_entry_recomputes_summary_from_real_evidence_state(store, reason):
    factory, run_id = store
    repository = ResearchToolRepository(factory)
    async with factory() as session, session.begin():
        run = await session.get(ResearchRunRow, run_id)
        await repository._enter_writing(
            session,
            run,
            reason=reason,
            summary="全部质量门已满足; 使用现有有效证据生成报告。",
        )
    with factory._sync_factory() as session:
        writing = session.scalar(
            select(AgentEventRow).where(AgentEventRow.event_type == "report.writing_started")
        )
        assert "全部质量门已满足" not in writing.public_summary
        assert "使用现有有效证据" not in writing.public_summary
        assert writing.refs["reason"] == reason


@pytest.mark.asyncio
async def test_research_pool_cannot_overrun(store):
    for i in range(1, 5):
        add_event(store, "source.readable", f"q{i}")
        assert (await reserve(store, question=f"q{i}")).granted
    add_event(store, "source.readable", "q5")
    assert not (await reserve(store, question="q5")).granted
    factory, _ = store
    with factory._sync_factory() as session:
        assert session.scalar(select(func.sum(ModelBudgetReservationRow.reserved_total))) == 12500


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["reserved", "uncertain", "settled", "released"])
async def test_no_repeat_first_call_allowance_even_after_unknown_or_release(store, status):
    factory, _ = store
    add_event(store, "source.readable")
    assert (await reserve(store)).granted
    with factory._sync_factory() as session:
        row = session.scalar(select(ModelBudgetReservationRow))
        row.status = status
        row.actual_total = 0 if status in ("settled", "released") else None
        session.commit()
    assert not (await reserve(store)).granted


@pytest.mark.asyncio
async def test_without_readable_event_no_first_call_exception(store):
    assert not (await reserve(store)).granted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([], "NO_READABLE_SOURCE"),
        (["source.fetch_failed"], "NO_READABLE_SOURCE"),
        (["source.readable"], "EVIDENCE_EXTRACTION_NOT_COMPLETED"),
        (["source.readable", "evidence.extraction_started"], "EVIDENCE_EXTRACTION_NOT_COMPLETED"),
        (
            ["source.readable", "evidence.extraction_started", "evidence.extracted"],
            "EVIDENCE_VALIDATION_EMPTY",
        ),
    ],
)
async def test_no_evidence_diagnostics_use_read_events_not_only_page_counter(
    store,
    events,
    expected,
):
    factory, run_id = store
    for event in events:
        add_event(store, event)
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        run.phase = RunPhase.WRITING.value
        run.termination_reason = "deadline_exhausted"
        session.commit()
    await ReportRepository(factory).fail_no_evidence(run_id, worker_task_id="offline-worker")
    with factory._sync_factory() as session:
        event = session.scalar(
            select(AgentEventRow).where(AgentEventRow.event_type == "run.failed")
        )
        assert event.refs["detail_code"] == expected
        assert event.refs["research_stop_reason"] == "deadline_exhausted"
        assert event.metrics["accepted_evidence"] == 0
        assert session.get(ResearchRunRow, run_id).status == "failed"


@pytest.mark.parametrize("reason", ["deadline_exhausted", "token_budget_exhausted", "other", None])
@pytest.mark.parametrize("accepted", [0, 1])
def test_stop_messages_never_invent_quality_pass(reason, accepted):
    summary = _writing_start_summary(reason, accepted_evidence=accepted)
    assert "全部质量门已满足" not in summary
    if reason == "deadline_exhausted":
        assert "截止时间" in summary


def test_quality_message_requires_evidence_and_quality_stop():
    assert "全部质量门已满足" not in _writing_start_summary("quality_met", accepted_evidence=0)
    assert "全部质量门已满足" in _writing_start_summary("quality_met", accepted_evidence=1)


def test_real_extractor_minimum_and_quick_pool_formula():
    extractor = object.__new__(EvidenceExtractorService)
    minimum = extractor.estimate_minimum_request_tokens(
        question="北斗三号全球卫星导航系统何时正式开通服务?",
    )
    assert minimum.min_source_tokens == 1000 and minimum.min_output_tokens == 768
    assert minimum.total_tokens > 1785
    fair = allocate_research_call(
        available=12500,
        eligible_questions=7,
        other_unattempted=0,
        current_attempts=0,
        minimum_call=minimum.total_tokens,
        maximum_call=12000,
    )
    assert fair.call_tokens == 3125
    assert fair.call_tokens + fair.future_reserve <= 12500


@pytest.mark.asyncio
async def test_real_loop_reaches_extractor_after_atomic_first_reservation(store):
    factory, run_id = store
    actual = ResearchToolRepository(factory)
    repository = FakeRepository()
    repository.page_budget_granted = 1
    repository.evidence_model_budget = actual.evidence_model_budget
    repository.reserve_model_tokens = actual.reserve_model_tokens
    reader = FakeReader()
    original_read = reader.read

    async def read(url):
        page = await original_read(url)
        add_event(store, "source.readable")
        return page

    reader.read = read
    extractor = FakeExtractor()
    extractor.minimum_total = 2700
    # No pretend successful evidence: exercise an extraction that returns none.
    repository.record_page_result = (0, 0)
    result = await ResearchLoopService(
        repository,
        FakeSearch(),
        reader,
        extractor,
        FakeArtifacts(),
    ).run_one_iteration(run_id, worker_task_id="offline-worker")
    assert extractor.calls == 1
    assert result.accepted_evidence == 0
    with factory._sync_factory() as session:
        assert session.scalar(select(ModelBudgetReservationRow)).reserved_total == 3125


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["403", "short", "deadline"])
async def test_unusable_source_and_expired_deadline_never_reach_extractor(failure):
    repository = FakeRepository()
    repository.page_budget_granted = 1
    extractor = FakeExtractor()
    reader = FakeReader()
    search = FakeSearch()
    if failure == "403":
        reader.read = AsyncMock(
            side_effect=ToolExecutionError(
                "WEBPAGE_REQUEST_REJECTED",
                retryable=False,
            )
        )
    elif failure == "short":
        original_read = reader.read

        async def short_read(url):
            return (await original_read(url)).model_copy(update={"clean_text": "Short body."})

        reader.read = short_read
        repository.target = replace(repository.target, cheap_triage_enabled=True)
    else:
        repository.target = replace(
            repository.target,
            deadline_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    result = await ResearchLoopService(
        repository,
        search,
        reader,
        extractor,
        FakeArtifacts(),
    ).run_one_iteration("offline-run", worker_task_id="offline-worker")
    assert extractor.calls == 0 and result.accepted_evidence == 0
    assert not repository.token_reservations
    if failure == "deadline":
        assert repository.last_attempt_outcome == "deadline_exhausted"
        assert search.calls == 0 and reader.calls == 0
    elif failure == "403":
        assert repository.page_failures > 0
    else:
        assert ("selection_skipped", "low_evidence_value") in repository.selection_events
