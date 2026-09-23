"""Fail-first regression coverage for Candidate Dispatch lifecycle events.

These tests execute the real ResearchLoopService result-selection path with a
small repository probe.  The probe records the existing fetch boundary and
provides the event sink that the production implementation must call once
Dispatch lifecycle telemetry exists.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import ReadPage, SearchResult
from app.domain.source_policy import normalize_source_url
from app.infrastructure.db.research_tools import (
    EvidenceModelBudget,
    IterationEvaluation,
    ModelTokenReservation,
    PageBudgetReservation,
    ResearchTarget,
)
from app.services.research_loop import ResearchLoopService

Event = dict[str, Any]


class DispatchSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        del query, limit
        return self.results


class DispatchReader:
    async def read(self, url: str) -> ReadPage:
        return ReadPage(
            final_url=url,
            title="Industrial inspection source",
            clean_text="Industrial defect inspection evidence. " * 5,
            content_hash="a" * 64,
            fetched_at=datetime.now(UTC),
        )


class DispatchExtractor:
    def estimate_minimum_request_tokens(self, **kwargs: object) -> Any:
        del kwargs
        from app.domain.research_budget import MinimumCallEstimate

        return MinimumCallEstimate(1_000, 1_000, 768, 1_232, 4_000)

    async def extract(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
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


class DispatchArtifacts:
    async def save_page(self, run_id: object, source_id: object, text: str) -> str:
        del run_id, source_id, text
        return "runs/test/source.txt"


class DispatchRepository:
    """Repository probe for the real ResearchLoopService dispatch path."""

    def __init__(self, results: list[SearchResult], *, page_budget: int = 1) -> None:
        self.page_budget = page_budget
        self.events: list[Event] = []
        self._candidate_ids = {
            normalize_source_url(result.url): f"candidate-{index}"
            for index, result in enumerate(results, start=1)
        }
        self.target = ResearchTarget(
            plan_version=1,
            question_id="q1",
            question="Which routes are used for industrial inspection?",
            query="industrial defect inspection",
            gap_id=uuid4(),
            tool_call_id=uuid4(),
            source_id_seed=uuid4(),
            acceptance_dimensions=(("d1", "industrial defect inspection"),),
            used_source_owner_keys=(),
        )

    async def prepare_target(self, *args: object, **kwargs: object) -> ResearchTarget:
        del args, kwargs
        return self.target

    async def evidence_model_budget(self, *args: object, **kwargs: object) -> EvidenceModelBudget:
        del args, kwargs
        return EvidenceModelBudget(
            allowed=True,
            max_call_tokens=18_000,
            remaining_tokens=25_000,
            writer_reserve_tokens=10_000,
        )

    async def record_search_results(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def record_candidate_dispatch_event(self, *args: object, **kwargs: object) -> None:
        del args
        url = normalize_source_url(str(kwargs["url"]))
        candidate_id = self._candidate_ids[url]
        stage = str(kwargs["stage"])
        self.events.append(
            {
                "event_type": f"candidate.{stage}",
                "candidate_id": candidate_id,
                "reason": kwargs.get("reason"),
            }
        )

    async def reserve_page_slots(self, *args: object, **kwargs: object) -> PageBudgetReservation:
        del args, kwargs
        return PageBudgetReservation(granted=self.page_budget, remaining=0)

    async def record_page_fetched(self, *args: object, **kwargs: object) -> None:
        del args
        url = normalize_source_url(str(kwargs["url"]))
        candidate_id = self._candidate_ids.get(url)
        if candidate_id is not None:
            self.events.append(
                {"event_type": "candidate.fetch_started", "candidate_id": candidate_id}
            )

    async def record_page_failure(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def release_page_slots(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def page_already_processed(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return False

    async def reserve_extraction_slot(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return True

    async def release_extraction_slot(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def reserve_model_tokens(self, *args: object, **kwargs: object) -> ModelTokenReservation:
        del args, kwargs
        return ModelTokenReservation(True, 4_000, "reserved")

    async def settle_model_reservation(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return True

    async def release_model_reservation(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return True

    async def mark_model_reservation_uncertain(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return True

    async def record_extraction_started(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def record_extraction_failure(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def record_page(self, *args: object, **kwargs: object) -> tuple[int, int]:
        del args, kwargs
        return 0, 0

    async def finish_iteration(self, *args: object, **kwargs: object) -> IterationEvaluation:
        del args, kwargs
        return IterationEvaluation(
            continue_research=False,
            decision="ready_to_write",
            stop_reason=None,
            question_status="researched",
        )


def _events_for(repository: DispatchRepository, candidate_id: str) -> list[Event]:
    return [event for event in repository.events if event.get("candidate_id") == candidate_id]


def _service(
    repository: DispatchRepository,
    results: list[SearchResult],
) -> ResearchLoopService:
    return ResearchLoopService(
        repository,  # type: ignore[arg-type]
        DispatchSearch(results),  # type: ignore[arg-type]
        DispatchReader(),  # type: ignore[arg-type]
        DispatchExtractor(),  # type: ignore[arg-type]
        DispatchArtifacts(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_dispatch_selected_precedes_fetch_started() -> None:
    results = [
        SearchResult(
            title="Industrial defect inspection route",
            url="https://source.example/industrial-inspection",
            snippet="Industrial defect inspection technology route.",
            rank=1,
        )
    ]
    repository = DispatchRepository(results)

    await _service(repository, results).run_one_iteration(uuid4(), worker_task_id="worker-1")

    events = _events_for(repository, "candidate-1")
    assert [event["event_type"] for event in events] == [
        "candidate.dispatch_started",
        "candidate.dispatch_selected",
        "candidate.fetch_started",
    ]


@pytest.mark.asyncio
async def test_page_budget_produces_dispatch_skip_not_early_candidate_skip() -> None:
    results = [
        SearchResult(
            title="Industrial defect inspection route",
            url="https://source.example/industrial-inspection",
            snippet="Industrial defect inspection technology route.",
            rank=1,
        )
    ]
    repository = DispatchRepository(results, page_budget=0)

    await _service(repository, results).run_one_iteration(uuid4(), worker_task_id="worker-1")

    events = _events_for(repository, "candidate-1")
    assert [event["event_type"] for event in events] == [
        "candidate.dispatch_started",
        "candidate.dispatch_skipped",
    ]
    assert events[-1]["reason"] == "page_budget"
    assert all(event["event_type"] != "candidate.skipped" for event in events)


@pytest.mark.asyncio
async def test_source_diversity_rejection_is_dispatch_skip() -> None:
    results = [
        SearchResult(
            title="Industrial inspection source A",
            url="https://same.example/a",
            snippet="Industrial defect inspection technology route.",
            rank=1,
        ),
        SearchResult(
            title="Industrial inspection source B",
            url="https://same.example/b",
            snippet="Industrial defect inspection technology route.",
            rank=2,
        ),
        SearchResult(
            title="Independent industrial inspection source",
            url="https://other.example/report",
            snippet="Industrial defect inspection technology route.",
            rank=3,
        ),
    ]
    repository = DispatchRepository(results)
    repository.target = replace(
        repository.target,
        used_source_owner_keys=("same.example",),
    )

    await _service(repository, results).run_one_iteration(uuid4(), worker_task_id="worker-1")

    events = _events_for(repository, "candidate-1")
    assert [event["event_type"] for event in events] == [
        "candidate.dispatch_started",
        "candidate.dispatch_skipped",
    ]
    assert events[-1]["reason"] == "source_diversity"


@pytest.mark.asyncio
async def test_low_relevance_candidate_is_dispatch_skip() -> None:
    results = [
        SearchResult(
            title="Unrelated cooking recipe",
            url="https://source.example/recipe",
            snippet="A recipe with ingredients and cooking instructions.",
            rank=1,
        )
    ]
    repository = DispatchRepository(results)

    await _service(repository, results).run_one_iteration(uuid4(), worker_task_id="worker-1")

    events = _events_for(repository, "candidate-1")
    assert [event["event_type"] for event in events] == [
        "candidate.dispatch_started",
        "candidate.dispatch_skipped",
    ]
    assert events[-1]["reason"] == "relevance_rank"
