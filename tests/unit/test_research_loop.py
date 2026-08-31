from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import ReadPage, SearchResult
from app.infrastructure.db.research_tools import IterationEvaluation, ResearchTarget
from app.llm.adapters import ModelGatewayError
from app.services.research_loop import ResearchLoopService


class FakeRepository:
    def __init__(self) -> None:
        self.target = ResearchTarget(
            plan_version=1,
            question_id="q1",
            question="Which routes are used for industrial inspection?",
            query="industrial inspection technology routes",
            gap_id=uuid4(),
            tool_call_id=uuid4(),
            source_id_seed=uuid4(),
        )
        self.extraction_failures: list[str] = []
        self.finished = False
        self.finish_calls = 0
        self.stop_after = 1

    async def prepare_target(self, *args: object, **kwargs: object) -> ResearchTarget:
        return self.target

    async def record_search_results(self, *args: object, **kwargs: object) -> None:
        return None

    async def record_tool_failure(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("search must not fail")

    async def record_extraction_started(self, *args: object, **kwargs: object) -> None:
        return None

    async def record_page_failure(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("reader must not fail")

    async def record_extraction_failure(self, *args: object, **kwargs: object) -> None:
        self.extraction_failures.append(str(kwargs["error_code"]))

    async def record_page(self, *args: object, **kwargs: object) -> tuple[int, int]:
        return 1, 1

    async def finish_iteration(
        self, *args: object, **kwargs: object
    ) -> IterationEvaluation:
        self.finished = True
        self.finish_calls += 1
        should_continue = self.finish_calls < self.stop_after
        return IterationEvaluation(
            continue_research=should_continue,
            decision="continue_plan" if should_continue else "ready_to_write",
            stop_reason=None if should_continue else "writer_not_implemented",
            question_status="researched",
        )


class FakeSearch:
    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        return [
            SearchResult(title=f"Source {rank}", url=f"https://example.com/{rank}", rank=rank)
            for rank in (1, 2)
        ]


class FakeReader:
    async def read(self, url: str) -> ReadPage:
        return ReadPage(
            final_url=url,
            title=url,
            clean_text="Evidence-bearing public page content. " * 5,
            content_hash="a" * 64,
            fetched_at=datetime.now(UTC),
        )


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0
        self.error_code = "EVIDENCE_OUTPUT_SCHEMA_INVALID"

    async def extract(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        self.calls += 1
        if self.calls == 2:
            raise ModelGatewayError(
                self.error_code,
                retryable=self.error_code != "EVIDENCE_OUTPUT_SCHEMA_INVALID",
            )
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


class FakeArtifacts:
    async def save_page(self, run_id: object, source_id: object, text: str) -> str:
        return "runs/test/source.txt"


@pytest.mark.asyncio
async def test_schema_failure_is_isolated_to_page_and_iteration_finishes() -> None:
    repository = FakeRepository()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=1:pages=2:accepted=1"
    assert repository.extraction_failures == ["EVIDENCE_OUTPUT_SCHEMA_INVALID"]
    assert repository.finished is True

@pytest.mark.asyncio
async def test_evaluator_continues_without_manual_resume() -> None:
    repository = FakeRepository()
    repository.stop_after = 2
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=2:pages=4:accepted=3"
    assert repository.finish_calls == 2

@pytest.mark.asyncio
async def test_retryable_model_failure_is_isolated_to_page() -> None:
    repository = FakeRepository()
    extractor = FakeExtractor()
    extractor.error_code = "MODEL_TIMEOUT"
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=1:pages=2:accepted=1"
    assert repository.extraction_failures == ["MODEL_TIMEOUT"]
