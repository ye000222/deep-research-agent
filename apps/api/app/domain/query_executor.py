"""Execute prepared query requests through an injected provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from app.domain.query_execution import (
    QueryExecutionEligibility,
    QueryExecutionRequest,
)


class QueryExecutionResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    EMPTY = "empty"
    BLOCKED = "blocked"


class QueryExecutionEventType(StrEnum):
    STARTED = "query.execution.started"
    COMPLETED = "query.execution.completed"
    FAILED = "query.execution.failed"


@dataclass(frozen=True, slots=True)
class SearchResultSet:
    """Provider-neutral search results returned by an adapter."""

    provider: str
    sources: tuple[str, ...]


class SearchProviderAdapter(Protocol):
    """Minimal provider contract used by QueryExecutor."""

    provider_name: str

    def search(
        self,
        query_text: str,
        constraints: tuple[str, ...],
    ) -> SearchResultSet: ...


@dataclass(frozen=True, slots=True)
class QueryExecutionEvent:
    """One observable execution lifecycle event."""

    event_type: QueryExecutionEventType
    run_id: UUID
    question_id: str
    gap_id: UUID
    candidate_id: UUID
    execution_id: UUID
    query_text: str
    timestamp: datetime
    error_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QueryExecutionResult:
    """Normalized result of one provider execution attempt."""

    execution_id: UUID
    request_id: UUID
    status: QueryExecutionResultStatus
    provider: str
    query_text: str
    result_count: int
    candidate_sources: tuple[str, ...]
    error_reason: str | None
    events: tuple[QueryExecutionEvent, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_id": str(self.execution_id),
            "request_id": str(self.request_id),
            "status": self.status.value,
            "provider": self.provider,
            "query_text": self.query_text,
            "result_count": self.result_count,
            "candidate_sources": list(self.candidate_sources),
            "error_reason": self.error_reason,
            "events": [
                {
                    "event_type": event.event_type.value,
                    "run_id": str(event.run_id),
                    "question_id": event.question_id,
                    "gap_id": str(event.gap_id),
                    "candidate_id": str(event.candidate_id),
                    "execution_id": str(event.execution_id),
                    "query_text": event.query_text,
                    "timestamp": event.timestamp.isoformat(),
                    "error_reason": event.error_reason,
                }
                for event in self.events
            ],
        }


class QueryExecutor:
    """Run one request through an explicitly supplied SearchProviderAdapter."""

    @staticmethod
    def execute(
        request: QueryExecutionRequest,
        *,
        provider_adapter: SearchProviderAdapter,
        eligibility: QueryExecutionEligibility = QueryExecutionEligibility.READY,
        now: datetime | None = None,
    ) -> QueryExecutionResult:
        timestamp = now or datetime.now(UTC)
        execution_id = uuid4()
        if eligibility is not QueryExecutionEligibility.READY:
            reason = _eligibility_reason(eligibility)
            event = _event(
                QueryExecutionEventType.FAILED,
                request=request,
                execution_id=execution_id,
                timestamp=timestamp,
                error_reason=reason,
            )
            return QueryExecutionResult(
                execution_id=execution_id,
                request_id=request.execution_id,
                status=QueryExecutionResultStatus.BLOCKED,
                provider=provider_adapter.provider_name,
                query_text=request.query_text,
                result_count=0,
                candidate_sources=(),
                error_reason=reason,
                events=(event,),
            )

        started = _event(
            QueryExecutionEventType.STARTED,
            request=request,
            execution_id=execution_id,
            timestamp=timestamp,
        )
        try:
            result_set = provider_adapter.search(
                request.query_text,
                request.source_constraints,
            )
        except Exception as exc:  # Provider boundary: normalize adapter failures.
            reason = _provider_error_reason(exc)
            failed = _event(
                QueryExecutionEventType.FAILED,
                request=request,
                execution_id=execution_id,
                timestamp=timestamp,
                error_reason=reason,
            )
            return QueryExecutionResult(
                execution_id=execution_id,
                request_id=request.execution_id,
                status=QueryExecutionResultStatus.FAILED,
                provider=provider_adapter.provider_name,
                query_text=request.query_text,
                result_count=0,
                candidate_sources=(),
                error_reason=reason,
                events=(started, failed),
            )

        sources = _normalize_sources(result_set.sources)
        if not sources:
            failed = _event(
                QueryExecutionEventType.FAILED,
                request=request,
                execution_id=execution_id,
                timestamp=timestamp,
                error_reason="empty_result",
            )
            return QueryExecutionResult(
                execution_id=execution_id,
                request_id=request.execution_id,
                status=QueryExecutionResultStatus.EMPTY,
                provider=result_set.provider,
                query_text=request.query_text,
                result_count=0,
                candidate_sources=(),
                error_reason="empty_result",
                events=(started, failed),
            )

        completed = _event(
            QueryExecutionEventType.COMPLETED,
            request=request,
            execution_id=execution_id,
            timestamp=timestamp,
        )
        return QueryExecutionResult(
            execution_id=execution_id,
            request_id=request.execution_id,
            status=QueryExecutionResultStatus.SUCCESS,
            provider=result_set.provider,
            query_text=request.query_text,
            result_count=len(sources),
            candidate_sources=sources,
            error_reason=None,
            events=(started, completed),
        )


def _event(
    event_type: QueryExecutionEventType,
    *,
    request: QueryExecutionRequest,
    execution_id: UUID,
    timestamp: datetime,
    error_reason: str | None = None,
) -> QueryExecutionEvent:
    return QueryExecutionEvent(
        event_type=event_type,
        run_id=request.run_id,
        question_id=request.question_id,
        gap_id=request.gap_id,
        candidate_id=request.candidate_id,
        execution_id=execution_id,
        query_text=request.query_text,
        timestamp=timestamp,
        error_reason=error_reason,
    )


def _normalize_sources(sources: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for source in sources:
        value = source.strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _provider_error_reason(error: Exception) -> str:
    message = str(error).casefold()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    return error.__class__.__name__.lower()


def _eligibility_reason(eligibility: QueryExecutionEligibility) -> str:
    return {
        QueryExecutionEligibility.BLOCKED: "execution_blocked",
        QueryExecutionEligibility.REJECTED: "execution_rejected",
        QueryExecutionEligibility.UNKNOWN: "execution_eligibility_unknown",
        QueryExecutionEligibility.READY: "",
    }[eligibility]
