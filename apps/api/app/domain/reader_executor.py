"""Execute Reader requests through an injected, provider-neutral adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from app.domain.reader_request import (
    ReaderDispatchState,
    ReaderExecutionRequest,
)


class ReaderExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    EMPTY = "empty"
    BLOCKED = "blocked"


class ReaderParseStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ReaderQualityStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"


class ReaderExecutionEventType(StrEnum):
    STARTED = "reader.execution.started"
    COMPLETED = "reader.execution.completed"
    FAILED = "reader.execution.failed"


@dataclass(frozen=True, slots=True)
class ReaderProviderResult:
    """Provider-neutral fetch and parse result."""

    content: str
    parse_status: ReaderParseStatus = ReaderParseStatus.SUCCESS


class ReaderProviderAdapter(Protocol):
    """Minimal Reader contract used by ReaderExecutor."""

    provider_name: str

    def read(self, source: str) -> ReaderProviderResult: ...


@dataclass(frozen=True, slots=True)
class ReaderExecutionEvent:
    """One observable Reader execution lifecycle event."""

    event_type: ReaderExecutionEventType
    run_id: UUID
    question_id: str
    gap_id: UUID
    source_id: str
    routing_id: UUID
    alignment_id: UUID
    reader_execution_id: UUID
    reason: str
    timestamp: datetime
    query_execution_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ReaderExecutionResult:
    """Normalized result of one Reader execution attempt."""

    execution_id: UUID
    routing_id: UUID
    alignment_id: UUID
    source_id: str
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    status: ReaderExecutionStatus
    content_available: bool
    content_length: int
    parse_status: ReaderParseStatus
    quality_status: ReaderQualityStatus
    failure_reason: str | None
    query_execution_id: UUID | None = None
    events: tuple[ReaderExecutionEvent, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_id": str(self.execution_id),
            "routing_id": str(self.routing_id),
            "alignment_id": str(self.alignment_id),
            "source_id": self.source_id,
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "status": self.status.value,
            "content_available": self.content_available,
            "content_length": self.content_length,
            "parse_status": self.parse_status.value,
            "quality_status": self.quality_status.value,
            "failure_reason": self.failure_reason,
            "query_execution_id": (
                str(self.query_execution_id) if self.query_execution_id else None
            ),
            "events": [
                {
                    "event_type": event.event_type.value,
                    "run_id": str(event.run_id),
                    "question_id": event.question_id,
                    "gap_id": str(event.gap_id),
                    "source_id": event.source_id,
                    "routing_id": str(event.routing_id),
                    "alignment_id": str(event.alignment_id),
                    "reader_execution_id": str(event.reader_execution_id),
                    "reason": event.reason,
                    "timestamp": event.timestamp.isoformat(),
                    "query_execution_id": (
                        str(event.query_execution_id)
                        if event.query_execution_id
                        else None
                    ),
                }
                for event in self.events
            ],
        }


class ReaderExecutor:
    """Consume READY requests without invoking Evidence or Gap logic."""

    @staticmethod
    def execute(
        request: ReaderExecutionRequest,
        *,
        provider_adapter: ReaderProviderAdapter,
        now: datetime | None = None,
    ) -> ReaderExecutionResult:
        execution_id = uuid4()
        timestamp = now or datetime.now(UTC)
        if request.state is not ReaderDispatchState.READY:
            reason = "reader_request_not_ready"
            event = _event(
                ReaderExecutionEventType.FAILED,
                request=request,
                execution_id=execution_id,
                reason=reason,
                timestamp=timestamp,
            )
            return _result(
                request,
                execution_id=execution_id,
                status=ReaderExecutionStatus.BLOCKED,
                content="",
                parse_status=ReaderParseStatus.UNKNOWN,
                failure_reason=reason,
                events=(event,),
            )

        started = _event(
            ReaderExecutionEventType.STARTED,
            request=request,
            execution_id=execution_id,
            reason=request.execution_reason,
            timestamp=timestamp,
        )
        try:
            provider_result = provider_adapter.read(request.source_id)
        except Exception as exc:  # Reader boundary: normalize adapter failures.
            reason = _reader_error_reason(exc)
            failed = _event(
                ReaderExecutionEventType.FAILED,
                request=request,
                execution_id=execution_id,
                reason=reason,
                timestamp=timestamp,
            )
            return _result(
                request,
                execution_id=execution_id,
                status=ReaderExecutionStatus.FAILED,
                content="",
                parse_status=ReaderParseStatus.UNKNOWN,
                failure_reason=reason,
                events=(started, failed),
            )

        content = provider_result.content.strip()
        if provider_result.parse_status is ReaderParseStatus.FAILED:
            failed = _event(
                ReaderExecutionEventType.FAILED,
                request=request,
                execution_id=execution_id,
                reason="parse_error",
                timestamp=timestamp,
            )
            return _result(
                request,
                execution_id=execution_id,
                status=ReaderExecutionStatus.FAILED,
                content=content,
                parse_status=ReaderParseStatus.FAILED,
                failure_reason="parse_error",
                events=(started, failed),
            )
        if not content:
            completed = _event(
                ReaderExecutionEventType.COMPLETED,
                request=request,
                execution_id=execution_id,
                reason="empty_content",
                timestamp=timestamp,
            )
            return _result(
                request,
                execution_id=execution_id,
                status=ReaderExecutionStatus.EMPTY,
                content=content,
                parse_status=provider_result.parse_status,
                failure_reason="empty_content",
                events=(started, completed),
            )

        completed = _event(
            ReaderExecutionEventType.COMPLETED,
            request=request,
            execution_id=execution_id,
            reason="reader_success",
            timestamp=timestamp,
        )
        return _result(
            request,
            execution_id=execution_id,
            status=ReaderExecutionStatus.SUCCESS,
            content=content,
            parse_status=provider_result.parse_status,
            failure_reason=None,
            events=(started, completed),
        )


def _result(
    request: ReaderExecutionRequest,
    *,
    execution_id: UUID,
    status: ReaderExecutionStatus,
    content: str,
    parse_status: ReaderParseStatus,
    failure_reason: str | None,
    events: tuple[ReaderExecutionEvent, ...],
) -> ReaderExecutionResult:
    return ReaderExecutionResult(
        execution_id=execution_id,
        routing_id=request.routing_id,
        alignment_id=request.alignment_id,
        source_id=request.source_id,
        run_id=request.run_id,
        question_id=request.question_id,
        gap_id=request.gap_id,
        dimension_key=request.dimension_key,
        status=status,
        content_available=bool(content),
        content_length=len(content),
        parse_status=parse_status,
        quality_status=ReaderQualityStatus.NOT_EVALUATED,
        failure_reason=failure_reason,
        query_execution_id=request.query_execution_id,
        events=events,
    )


def _event(
    event_type: ReaderExecutionEventType,
    *,
    request: ReaderExecutionRequest,
    execution_id: UUID,
    reason: str,
    timestamp: datetime,
) -> ReaderExecutionEvent:
    return ReaderExecutionEvent(
        event_type=event_type,
        run_id=request.run_id,
        question_id=request.question_id,
        gap_id=request.gap_id,
        source_id=request.source_id,
        routing_id=request.routing_id,
        alignment_id=request.alignment_id,
        reader_execution_id=execution_id,
        reason=reason,
        timestamp=timestamp,
        query_execution_id=request.query_execution_id,
    )


def _reader_error_reason(error: Exception) -> str:
    message = str(error).casefold()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    return error.__class__.__name__.lower()
