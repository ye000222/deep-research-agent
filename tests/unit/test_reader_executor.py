from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.reader_executor import (
    ReaderExecutionEventType,
    ReaderExecutionStatus,
    ReaderExecutor,
    ReaderParseStatus,
    ReaderProviderResult,
)
from app.domain.reader_request import (
    ReaderDispatchState,
    ReaderExecutionRequest,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000027")
ROUTING_ID = UUID("00000000-0000-0000-0000-000000000028")
ALIGNMENT_ID = UUID("00000000-0000-0000-0000-000000000029")
GAP_ID = UUID("00000000-0000-0000-0000-000000000030")
CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000031")
QUERY_EXECUTION_ID = UUID("00000000-0000-0000-0000-000000000032")


class FakeReader:
    provider_name = "fake-reader"

    def __init__(self, result: ReaderProviderResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def read(self, source: str) -> ReaderProviderResult:
        self.calls.append(source)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _request(state: ReaderDispatchState = ReaderDispatchState.READY) -> ReaderExecutionRequest:
    return ReaderExecutionRequest(
        routing_id=ROUTING_ID,
        alignment_id=ALIGNMENT_ID,
        source_id="https://industry-report.example/report",
        run_id=RUN_ID,
        question_id="q7",
        gap_id=GAP_ID,
        dimension_key="q7:d2",
        requirement_type="independent_source",
        execution_reason="aligned source",
        state=state,
        query_execution_id=QUERY_EXECUTION_ID,
    )


def test_successful_reader_execution_normalizes_content_and_events() -> None:
    reader = FakeReader(ReaderProviderResult("  readable body  "))

    result = ReaderExecutor.execute(
        _request(),
        provider_adapter=reader,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.status is ReaderExecutionStatus.SUCCESS
    assert result.content_available is True
    assert result.content_length == len("readable body")
    assert result.parse_status is ReaderParseStatus.SUCCESS
    assert result.failure_reason is None
    assert result.query_execution_id == QUERY_EXECUTION_ID
    assert [event.event_type for event in result.events] == [
        ReaderExecutionEventType.STARTED,
        ReaderExecutionEventType.COMPLETED,
    ]
    assert result.events[0].reader_execution_id == result.execution_id
    assert reader.calls == [_request().source_id]


def test_fetch_timeout_returns_failed_result() -> None:
    reader = FakeReader(error=TimeoutError("reader timeout"))

    result = ReaderExecutor.execute(_request(), provider_adapter=reader)

    assert result.status is ReaderExecutionStatus.FAILED
    assert result.failure_reason == "timeout"
    assert result.events[-1].event_type is ReaderExecutionEventType.FAILED


def test_empty_content_returns_empty_result() -> None:
    reader = FakeReader(ReaderProviderResult("   "))

    result = ReaderExecutor.execute(_request(), provider_adapter=reader)

    assert result.status is ReaderExecutionStatus.EMPTY
    assert result.content_available is False
    assert result.content_length == 0
    assert result.failure_reason == "empty_content"
    assert result.events[-1].event_type is ReaderExecutionEventType.COMPLETED


def test_parse_failure_returns_failed_result() -> None:
    reader = FakeReader(
        ReaderProviderResult("body", parse_status=ReaderParseStatus.FAILED)
    )

    result = ReaderExecutor.execute(_request(), provider_adapter=reader)

    assert result.status is ReaderExecutionStatus.FAILED
    assert result.parse_status is ReaderParseStatus.FAILED
    assert result.failure_reason == "parse_error"


def test_blocked_request_does_not_call_reader() -> None:
    reader = FakeReader(ReaderProviderResult("body"))

    result = ReaderExecutor.execute(
        _request(ReaderDispatchState.BLOCKED),
        provider_adapter=reader,
    )

    assert result.status is ReaderExecutionStatus.BLOCKED
    assert result.failure_reason == "reader_request_not_ready"
    assert result.events[0].event_type is ReaderExecutionEventType.FAILED
    assert reader.calls == []
