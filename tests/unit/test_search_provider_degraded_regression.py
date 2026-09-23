"""Fail-first regression coverage for provider degradation after quality progress.

This test intentionally asserts the post-fix contract.  It must fail against the
current implementation because a retryable provider error escapes the graph before
the graph can route to the report writer.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import SearchResult
from app.infrastructure.db.research_tools import IterationEvaluation
from app.services.research_graph import ResearchGraphService
from app.services.research_loop import ResearchLoopService
from app.tools.errors import ToolExecutionError
from app.worker import tasks as worker_tasks
from langgraph.checkpoint.memory import InMemorySaver
from test_research_graph import FakePlanner, FakeRuns, FakeStates
from test_research_loop import (
    FakeArtifacts,
    FakeExtractor,
    FakeReader,
    FakeRepository,
)


class QualityThenDegradedRepository(FakeRepository):
    """Record the first quality evaluation before the next search fails."""

    def __init__(self) -> None:
        super().__init__()
        self.quality_met = False
        self.quality_snapshot = {
            "coverage": 0.95,
            "priority_one_coverage": 1.0,
            "cross_validation": 0.9,
            "critical_gaps": 0,
            "source_quality": 0.9,
            "numeric_scope_consistency": 1.0,
            "source_role_fit": 1.0,
            "freshness": 1.0,
        }
        self.tool_failures: list[str] = []

    async def record_tool_failure(self, *args: object, **kwargs: object) -> None:
        self.tool_failures.append(str(kwargs["error_code"]))

    async def finish_iteration(self, *args: object, **kwargs: object) -> IterationEvaluation:
        self.finish_calls += 1
        self.finished = True
        self.last_attempt_outcome = str(kwargs.get("attempt_outcome"))
        self.quality_met = True
        provider_degraded = kwargs.get("attempt_outcome") == "provider_error"
        return IterationEvaluation(
            continue_research=not provider_degraded,
            decision="ready_to_write" if provider_degraded else "continue_plan",
            stop_reason="quality_met" if provider_degraded else None,
            question_status="researched",
            coverage=0.95,
            information_gain=0.2,
            low_information_gain_streak=0,
        )


class FirstSearchThenDegraded:
    """Produce usable candidates once, then reproduce provider degradation."""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        self.calls += 1
        if self.calls > 1:
            raise ToolExecutionError("SEARCH_PROVIDER_DEGRADED", retryable=True)
        return [
            SearchResult(
                title=f"Industrial inspection evidence {rank}",
                url=f"https://example.com/industrial-inspection/{rank}",
                rank=rank,
            )
            for rank in (1, 2, 3)
        ]


class StableExtractor(FakeExtractor):
    async def extract(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        self.calls += 1
        return (
            [],
            TokenUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                accuracy=UsageAccuracy.EXACT,
            ),
            {"source_chars": 200, "selected_chars": 200, "truncated": False},
        )


class RecordingReportWriter:
    def __init__(self) -> None:
        self.calls = 0

    async def write(self, run_id: UUID, *, worker_task_id: str) -> str:
        self.calls += 1
        return "report_completed:completed:citations=3"


class ExhaustedRetryRepository:
    """Worker-boundary persistence probe with the durable retry budget exhausted."""

    def __init__(self, _session_factory: object) -> None:
        self.search_transport_requeues = 4
        self.status = "RUNNING"
        self.termination_reason: str | None = None
        self.failure_detail: str | None = None

    async def acquire_for_execution(self, *args: object, **kwargs: object) -> bool:
        return True

    async def defer_retryable_search_error(self, *args: object, **kwargs: object) -> None:
        # Four durable retries already occurred before this worker invocation.
        return None

    async def fail_execution(self, *args: object, **kwargs: object) -> None:
        self.status = "FAILED"
        self.termination_reason = str(kwargs["error_code"])
        self.failure_detail = str(kwargs["detail_code"])


class RecordingStateRuntime:
    def __init__(self, _session_factory: object) -> None:
        self.nodes: list[str] = []

    async def synchronize(self, *args: object, **kwargs: object) -> None:
        self.nodes.append(str(kwargs.get("node_name")))


class FakeDatabase:
    def __init__(self, _database_url: str) -> None:
        self.session_factory = object()

    async def close(self) -> None:
        return None


class FakeCheckpointRuntime:
    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    async def open(self) -> object:
        return InMemorySaver()

    async def close(self) -> None:
        return None


class FakeHttpClient:
    async def aclose(self) -> None:
        return None


class FakeLLMCalls:
    async def record(self, *args: object, **kwargs: object) -> None:
        return None


class FailingGraph:
    async def execute(self, *args: object, **kwargs: object) -> str:
        raise ToolExecutionError("SEARCH_PROVIDER_DEGRADED", retryable=True)


@pytest.mark.asyncio
async def test_quality_met_provider_degraded_routes_to_report_after_retry_exhaustion() -> None:
    """Fail-first contract: provider degradation must not bypass completion logic."""

    run_id = uuid4()
    repository = QualityThenDegradedRepository()
    search = FirstSearchThenDegraded()
    research_loop = ResearchLoopService(
        repository,  # type: ignore[arg-type]
        search,  # type: ignore[arg-type]
        FakeReader(),
        StableExtractor(),
        FakeArtifacts(),
    )
    writer = RecordingReportWriter()
    graph = ResearchGraphService(
        FakeRuns(),  # type: ignore[arg-type]
        FakeStates(run_id),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        research_loop,  # type: ignore[arg-type]
        writer,  # type: ignore[arg-type]
    )

    observed_error: ToolExecutionError | None = None
    outcome: str | None = None
    try:
        outcome = await graph.execute(
            run_id,
            worker_task_id="provider-degraded-regression",
            checkpointer=InMemorySaver(),
        )
    except ToolExecutionError as exc:
        # Do not hide the failure: convert it into an explicit assertion below
        # so the test reports the current bypass as a regression.
        observed_error = exc

    assert repository.quality_met is True
    assert repository.finish_calls == 2
    assert repository.last_attempt_outcome == "provider_error"
    assert repository.tool_failures == ["SEARCH_PROVIDER_DEGRADED"]
    assert search.calls == 2
    assert observed_error is None, (
        "SEARCH_PROVIDER_DEGRADED escaped the Research Graph before completion "
        f"routing: {observed_error}"
    )
    assert writer.calls == 1
    assert outcome == "report_completed:completed:citations=3"


@pytest.mark.asyncio
async def test_exhausted_provider_retry_persists_current_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document the current worker outcome after durable search retries are exhausted."""

    repository = ExhaustedRetryRepository(object())
    state_runtime = RecordingStateRuntime(object())

    monkeypatch.setattr(
        worker_tasks,
        "Settings",
        lambda: SimpleNamespace(
            database_url="unused",
            checkpoint_database_uri="unused",
            checkpoint_pool_min_size=1,
            checkpoint_pool_max_size=1,
            model_http_proxy=None,
            public_web_http_proxy=None,
            searxng_base_url="http://unused",
            artifact_root="artifacts",
            evidence_aware_context_enabled=True,
        ),
    )
    monkeypatch.setattr(worker_tasks, "PostgresRuntime", FakeDatabase)
    monkeypatch.setattr(worker_tasks, "CheckpointRuntime", FakeCheckpointRuntime)
    monkeypatch.setattr(
        worker_tasks.httpx,
        "AsyncClient",
        lambda *args, **kwargs: FakeHttpClient(),
    )
    monkeypatch.setattr(worker_tasks, "ResearchRunRepository", lambda *_: repository)
    monkeypatch.setattr(worker_tasks, "ResearchStateRuntimeRepository", lambda *_: state_runtime)
    monkeypatch.setattr(
        worker_tasks,
        "ResearchGraphService",
        lambda *args, **kwargs: FailingGraph(),
    )
    monkeypatch.setattr(worker_tasks, "LLMCallRepository", lambda *_: FakeLLMCalls())
    monkeypatch.setattr(
        worker_tasks, "ResearchToolRepository", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(worker_tasks, "ReportRepository", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "RunProviderBindingRepository", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "ContextBudgetManager", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "ExtractionCacheRepository", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "ReportDraftCacheRepository", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "ResearchMemoryManager", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "ControlledToolGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "SearchEvidenceTool", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "AnalyzeDataTool", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "SecretCipher", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "load_or_create_master_key", lambda *_: object())
    monkeypatch.setattr(worker_tasks, "LLMGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "PlannerService", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "SearXNGSearchProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "PublicWebReader", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "EvidenceExtractorService", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "LocalArtifactStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_tasks, "ReportWriterService", lambda *args, **kwargs: object())

    result = await worker_tasks._execute(uuid4(), "provider-degraded-worker")

    assert result == "failed:SEARCH_PROVIDER_DEGRADED"
    assert repository.status == "FAILED"
    assert repository.termination_reason == "SEARCH_PROVIDER_DEGRADED"
    assert repository.failure_detail == "TOOL_EXECUTION_REJECTED"
    assert state_runtime.nodes == ["failure_boundary"]
