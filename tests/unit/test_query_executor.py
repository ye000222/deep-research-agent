from __future__ import annotations

from uuid import UUID

from app.domain.query_execution import (
    QueryExecutionEligibility,
    QueryExecutionRequest,
)
from app.domain.query_executor import (
    QueryExecutionEventType,
    QueryExecutionResultStatus,
    QueryExecutor,
    SearchResultSet,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000009")
QUESTION_ID = "q7"
GAP_ID = UUID("00000000-0000-0000-0000-000000000010")
CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000011")


class FakeProvider:
    provider_name = "fake-provider"

    def __init__(self, result: SearchResultSet | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def search(self, query_text: str, constraints: tuple[str, ...]) -> SearchResultSet:
        self.calls.append((query_text, constraints))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _request() -> QueryExecutionRequest:
    return QueryExecutionRequest(
        execution_id=UUID("00000000-0000-0000-0000-000000000012"),
        run_id=RUN_ID,
        question_id=QUESTION_ID,
        gap_id=GAP_ID,
        dimension_key="q7:d2",
        candidate_id=CANDIDATE_ID,
        ranking_score=0.9,
        query_text="industrial vision market report",
        source_constraints=("exclude_existing_sources=true",),
        requirement_type="independent_source",
        execution_reason="high gap alignment",
    )


def test_successful_execution_normalizes_results_and_emits_events() -> None:
    provider = FakeProvider(
        result=SearchResultSet(
            provider="fake-provider",
            sources=(" https://a.example ", "", "https://a.example", "https://b.example"),
        )
    )

    result = QueryExecutor.execute(_request(), provider_adapter=provider)

    assert result.status is QueryExecutionResultStatus.SUCCESS
    assert result.provider == "fake-provider"
    assert result.result_count == 2
    assert result.candidate_sources == ("https://a.example", "https://b.example")
    assert [event.event_type for event in result.events] == [
        QueryExecutionEventType.STARTED,
        QueryExecutionEventType.COMPLETED,
    ]
    assert len(provider.calls) == 1


def test_blocked_execution_does_not_call_provider() -> None:
    provider = FakeProvider(result=SearchResultSet("fake-provider", ("https://a.example",)))

    result = QueryExecutor.execute(
        _request(),
        provider_adapter=provider,
        eligibility=QueryExecutionEligibility.BLOCKED,
    )

    assert result.status is QueryExecutionResultStatus.BLOCKED
    assert result.error_reason == "execution_blocked"
    assert provider.calls == []


def test_provider_timeout_returns_failed_result() -> None:
    provider = FakeProvider(error=TimeoutError("provider timeout"))

    result = QueryExecutor.execute(_request(), provider_adapter=provider)

    assert result.status is QueryExecutionResultStatus.FAILED
    assert result.error_reason == "timeout"
    assert result.events[-1].event_type is QueryExecutionEventType.FAILED
    assert result.events[-1].error_reason == "timeout"


def test_empty_provider_result_returns_empty() -> None:
    provider = FakeProvider(result=SearchResultSet("fake-provider", ()))

    result = QueryExecutor.execute(_request(), provider_adapter=provider)

    assert result.status is QueryExecutionResultStatus.EMPTY
    assert result.error_reason == "empty_result"
    assert result.result_count == 0
    assert result.events[-1].event_type is QueryExecutionEventType.FAILED
