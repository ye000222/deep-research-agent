"""Offline reliability checks: scope, timing, bounded read-only-source continuation."""

import asyncio
import hashlib
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.core import stage_timing
from app.core.stage_timing import stage_timer, timed_call
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.domain.planning_scope import PlanScopeError, enforce_plan_scope, question_scope_error
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import SearchResult
from app.infrastructure.checkpoints.lifecycle import TimedPostgresSaver
from app.infrastructure.db.research_tools import EvidenceModelBudget
from app.llm.adapters import ModelGatewayError
from app.services.planner import PlannerService
from app.services.research_loop import ResearchLoopService
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from tests.unit.test_planner_service import (
    FakeBindingRepository,
    FakeGateway,
    dependencies,
    model_result,
)
from tests.unit.test_research_loop import FakeArtifacts, FakeExtractor, FakeReader, FakeRepository

GOAL = "北斗三号何时正式开通服务? 请区分建成、开通与投入运营并提供原始引用。"


def question(text, *, qid="q1"):
    return ResearchQuestion(
        id=qid,
        question=text,
        priority=1,
        rationale="以原始来源核验用户要求的主体和事件状态。",
        evidence_requirements=["必须提供对应原文引用"],
        search_hints=["原始来源事件日期"],
    )


def plan(goal, questions):
    return ResearchPlan(
        goal=goal,
        scope_summary=goal,
        questions=questions,
        completion_criteria=["主体和事件正确", "引用可追溯到原文"],
    )


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("北斗三号正式开通服务的日期是哪一天?", None),
        ("北斗三号建成与开通的官方表述有什么区别?", None),
        ("北斗三号投入运营与开通的日期是否一致?", None),
        ("哪些独立来源能够交叉核验该日期?", None),
        ("北斗三号开始建设的日期是哪一天?", "requested_event_replaced"),
        ("北斗二号正式开通服务的日期是哪一天?", "subject_identity_changed"),
        ("GPS正式开通服务的日期是哪一天?", "subject_identity_missing"),
        ("北斗三号的定位性能和授时精度是多少?", "unrequested_fact_type"),
        ("北斗三号未来升级计划是什么?", "unrequested_fact_type"),
    ],
)
def test_narrow_question_keeps_event_and_subject_semantics(text, error):
    assert question_scope_error(GOAL, question(text)) == error


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("北斗三号在中国境内2019年的正式开通日期是什么?", None),
        ("北斗三号在欧盟境内2019年的正式开通日期是什么?", "region_changed"),
        ("北斗三号在中国境内2024年的正式开通日期是什么?", "time_scope_changed"),
        ("北斗三号2017与2019年开通日期有什么区别?", None),
    ],
)
def test_explicit_region_and_time_scope(text, error):
    goal = "北斗三号在中国境内2018至2020年何时正式开通服务?"
    assert question_scope_error(goal, question(text)) == error


def test_completion_is_not_automatically_operation():
    assert (
        question_scope_error("北斗三号何时投入运营?", question("北斗三号什么时候建成?"))
        == "requested_event_replaced"
    )


def test_scope_repair_removes_drift_without_padding_or_fabricating_facts():
    good = question("北斗三号正式开通服务的日期是哪一天?")
    bad = question("北斗三号的性能指标是多少?", qid="q2")
    scoped = enforce_plan_scope(plan(GOAL, [good, bad]), GOAL)
    assert scoped.questions == [good]
    assert scoped.goal == GOAL
    assert "性能" not in scoped.scope_summary
    with pytest.raises(PlanScopeError, match="no_in_scope_question"):
        enforce_plan_scope(plan(GOAL, [bad]), GOAL)


def test_completion_criteria_cannot_reintroduce_removed_drift():
    valid = plan(GOAL, [question("北斗三号正式开通服务的日期是哪一天?")])
    with pytest.raises(PlanScopeError, match="completion_criterion_out_of_scope"):
        enforce_plan_scope(
            valid.model_copy(
                update={
                    "completion_criteria": ["比较北斗三号性能指标", "引用可追溯原文"],
                }
            ),
            GOAL,
        )
    repaired = enforce_plan_scope(
        valid.model_copy(update={"scope_summary": "研究北斗三号性能"}), GOAL
    )
    assert repaired.scope_summary == GOAL
    assert repaired.completion_criteria == valid.completion_criteria


@pytest.mark.asyncio
async def test_planner_rejects_scope_failure_without_extra_model_call():
    binding, cipher = dependencies()
    binding = replace(binding, goal=GOAL)
    payload = {
        "scope_summary": "A bounded outline",
        "questions": [question("北斗三号的未来升级计划是什么?").model_dump(exclude={"id"})],
        "completion_criteria": ["保留原始来源", "不混淆事件状态"],
    }
    gateway = FakeGateway([model_result(payload, input_tokens=10, output_tokens=20)])
    with pytest.raises(ModelGatewayError, match="PLANNER_SCOPE_INVALID"):
        await PlannerService(FakeBindingRepository(binding), cipher, gateway).generate(
            binding.run_id
        )
    assert len(gateway.calls) == 1


def test_timer_uses_monotonic_clock_not_wall_time(monkeypatch, caplog):
    ticks = iter([100.0, 100.125])
    monkeypatch.setattr(stage_timing.time, "perf_counter", lambda: next(ticks))
    caplog.set_level(logging.INFO)
    with stage_timer("offline.test", run_id=uuid4()):
        pass
    metrics = caplog.records[-1].stage_timing
    assert metrics["elapsed_ms"] == 125
    assert metrics["clock"] == "perf_counter" and metrics["outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError, asyncio.CancelledError, RuntimeError])
async def test_timer_preserves_exception_and_redacts_message(exception, caplog):
    caplog.set_level(logging.INFO)

    async def fail():
        raise exception("SECRET_DO_NOT_LOG")

    with pytest.raises(exception):
        await timed_call("offline.failed", fail())
    assert "SECRET_DO_NOT_LOG" not in caplog.text
    assert caplog.records[-1].stage_timing["error_type"] == exception.__name__


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["aget_tuple", "aput", "aput_writes"])
async def test_checkpoint_timer_delegates_exact_operation_without_database(
    monkeypatch, operation, caplog
):
    caplog.set_level(logging.INFO)
    base = AsyncMock(return_value="unchanged-result")
    monkeypatch.setattr(AsyncPostgresSaver, operation, base)
    saver = object.__new__(TimedPostgresSaver)
    config = {"configurable": {"thread_id": str(uuid4())}}
    assert await getattr(saver, operation)(config, example="metadata") == "unchanged-result"
    base.assert_awaited_once_with(config, example="metadata")
    assert caplog.records[-1].stage_timing["run_id"] == config["configurable"]["thread_id"]


@pytest.mark.asyncio
async def test_graph_timing_covers_nodes_memory_and_transition_without_services(caplog):
    from types import SimpleNamespace

    from app.services.research_graph import ResearchGraphService
    from langgraph.checkpoint.memory import InMemorySaver

    from tests.unit.test_research_graph import (
        FakePlanner,
        FakeReportWriter,
        FakeResearchLoop,
        FakeRuns,
        FakeStates,
    )

    caplog.set_level(logging.INFO)
    run_id = uuid4()
    memories = SimpleNamespace(
        retrieve_for_run=AsyncMock(return_value=SimpleNamespace(items=())),
        capture_state=AsyncMock(),
    )
    service = ResearchGraphService(
        FakeRuns(),
        FakeStates(run_id),
        FakePlanner(),
        FakeResearchLoop(),
        FakeReportWriter(),
        memories=memories,
    )
    await service.execute(run_id, worker_task_id="offline", checkpointer=InMemorySaver())
    stages = {r.stage_timing["stage"] for r in caplog.records if hasattr(r, "stage_timing")}
    assert {
        "graph.node.planner",
        "graph.node.research_iteration",
        "graph.node.report_writer",
        "state.synchronize.research",
        "memory.capture.research",
        "memory.capture.writer",
        "report.write_and_verify",
    } <= stages
    assert "research_transition" in caplog.text


class ReadSearch:
    async def search(self, query, *, limit):
        return [
            SearchResult(
                title=f"Industrial inspection evidence {i}", url=f"https://example.org/{i}", rank=i
            )
            for i in range(1, 4)
        ]


class QualifiedReader(FakeReader):
    async def read(self, url):
        page = await super().read(url)
        text = "Industrial inspection uses traceable methods and independent evaluation. " * 8
        return page.model_copy(
            update={
                "clean_text": text,
                "content_hash": hashlib.sha256((text + url).encode()).hexdigest(),
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcomes", "accepted"),
    [
        ([(0, 0), (1, 1)], 1),
        ([(0, 0), (0, 0)], 0),
        ([(0, 0), (1, 0)], 0),
    ],
)
async def test_empty_first_page_uses_only_bounded_prefetched_opportunities(outcomes, accepted):
    repo, reader = FakeRepository(), QualifiedReader()
    repo.target = replace(repo.target, original_goal="Research industrial inspection methods")
    repo.record_page = AsyncMock(side_effect=outcomes)
    extractor = FakeExtractor()
    extractor.extract = AsyncMock(
        return_value=(
            [],
            TokenUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                accuracy=UsageAccuracy.EXACT,
            ),
            {},
        )
    )
    result = await ResearchLoopService(
        repo,
        ReadSearch(),
        reader,
        extractor,
        FakeArtifacts(),
        parallel_reads_enabled=True,
    ).run_one_iteration(uuid4(), worker_task_id="worker-1")
    assert reader.calls == 2  # existing per-question page allowance, no refetch
    assert extractor.extract.await_count == 2
    assert result.accepted_evidence == accepted
    assert repo.record_page.await_count == 2
    assert len(repo.token_reservations) == 2


@pytest.mark.asyncio
async def test_prefetched_aliases_do_not_extract_same_page_twice():
    repo, reader = FakeRepository(), QualifiedReader()
    original = reader.read

    async def alias(url):
        page = await original(url)
        return page.model_copy(
            update={"final_url": "https://example.org/same", "content_hash": "a" * 64}
        )

    reader.read = alias
    repo.record_page_result = (0, 0)
    extractor = FakeExtractor()
    result = await ResearchLoopService(
        repo,
        ReadSearch(),
        reader,
        extractor,
        FakeArtifacts(),
        parallel_reads_enabled=True,
    ).run_one_iteration(uuid4(), worker_task_id="worker-1")
    assert extractor.calls == 1 and result.accepted_evidence == 0
    assert ("selection_skipped", "already_attempted") in repo.selection_events


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["budget", "deadline"])
async def test_continuation_cannot_bypass_budget_or_deadline(stop):
    repo, reader, extractor = FakeRepository(), QualifiedReader(), FakeExtractor()
    repo.record_page_result = (0, 0)
    if stop == "budget":
        original_budget = repo.evidence_model_budget
        count = 0

        async def budget(*args, **kwargs):
            nonlocal count
            count += 1
            if count > 2:
                return EvidenceModelBudget(False, 0, 0, 4500, outcome="stop_run")
            return await original_budget(*args, **kwargs)

        repo.evidence_model_budget = budget
    else:

        async def expire(*args, **kwargs):
            # The immutable target is replaced on the repository; expire the
            # clock used by the loop without touching the production deadline.
            raise TimeoutError

        extractor.extract = expire
    result = await ResearchLoopService(
        repo,
        ReadSearch(),
        reader,
        extractor,
        FakeArtifacts(),
        parallel_reads_enabled=True,
    ).run_one_iteration(uuid4(), worker_task_id="worker-1")
    assert result.accepted_evidence == 0
    assert len(repo.token_reservations) == 1


@pytest.mark.asyncio
async def test_deadline_expiring_after_empty_extraction_prevents_second_request(monkeypatch):
    from app.services import research_loop

    repo, reader, extractor = FakeRepository(), QualifiedReader(), FakeExtractor()
    repo.target = replace(repo.target, deadline_at=datetime.now(UTC) + timedelta(seconds=180))
    repo.record_page_result = (0, 0)
    expired = False

    async def persist(*args, **kwargs):
        nonlocal expired
        expired = True
        return 0, 0

    repo.record_page = persist
    monkeypatch.setattr(research_loop, "_deadline_expired", lambda target: expired)
    result = await ResearchLoopService(
        repo,
        ReadSearch(),
        reader,
        extractor,
        FakeArtifacts(),
        parallel_reads_enabled=True,
    ).run_one_iteration(uuid4(), worker_task_id="worker-1")
    assert result.accepted_evidence == 0 and extractor.calls == 1
    assert repo.last_attempt_outcome == "deadline_exhausted"
    assert len(repo.token_reservations) == 1
