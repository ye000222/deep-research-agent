"""Execute candidate-evidence extraction through an injected adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from app.domain.evidence_extraction_request import EvidenceExtractionRequest
from app.domain.reader_executor import ReaderExecutionResult, ReaderExecutionStatus


class EvidenceExtractionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    EMPTY = "empty"
    BLOCKED = "blocked"


class EvidenceExtractionEventType(StrEnum):
    STARTED = "evidence.extraction.started"
    COMPLETED = "evidence.extraction.completed"
    FAILED = "evidence.extraction.failed"


@dataclass(frozen=True, slots=True)
class EvidenceExtractorProviderResult:
    """Provider-neutral candidate evidence returned by an adapter."""

    extracted_claims: tuple[str, ...]
    evidence_items: tuple[str, ...]


class EvidenceExtractorAdapter(Protocol):
    """Minimal extractor contract, independent of acceptance rules."""

    def extract(self, content: str) -> EvidenceExtractorProviderResult: ...


@dataclass(frozen=True, slots=True)
class EvidenceExtractionEvent:
    """One observable extraction lifecycle event."""

    event_type: EvidenceExtractionEventType
    run_id: UUID
    question_id: str
    gap_id: UUID
    source_id: str
    reader_execution_id: UUID
    extraction_id: UUID
    reason: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class EvidenceExtractionResult:
    """Normalized candidate-evidence extraction result."""

    extraction_id: UUID
    reader_execution_id: UUID
    source_id: str
    question_id: str
    gap_id: UUID
    dimension_key: str
    status: EvidenceExtractionStatus
    candidate_evidence_count: int
    extracted_claims: tuple[str, ...]
    evidence_items: tuple[str, ...]
    failure_reason: str | None
    routing_id: UUID
    alignment_id: UUID
    events: tuple[EvidenceExtractionEvent, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "extraction_id": str(self.extraction_id),
            "reader_execution_id": str(self.reader_execution_id),
            "source_id": self.source_id,
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "status": self.status.value,
            "candidate_evidence_count": self.candidate_evidence_count,
            "extracted_claims": list(self.extracted_claims),
            "evidence_items": list(self.evidence_items),
            "failure_reason": self.failure_reason,
            "routing_id": str(self.routing_id),
            "alignment_id": str(self.alignment_id),
            "events": [
                {
                    "event_type": event.event_type.value,
                    "run_id": str(event.run_id),
                    "question_id": event.question_id,
                    "gap_id": str(event.gap_id),
                    "source_id": event.source_id,
                    "reader_execution_id": str(event.reader_execution_id),
                    "extraction_id": str(event.extraction_id),
                    "reason": event.reason,
                    "timestamp": event.timestamp.isoformat(),
                }
                for event in self.events
            ],
        }


class EvidenceExtractor:
    """Consume extraction requests without applying acceptance or Coverage."""

    @staticmethod
    def execute(
        request: EvidenceExtractionRequest,
        *,
        reader_result: ReaderExecutionResult,
        provider_adapter: EvidenceExtractorAdapter,
        now: datetime | None = None,
    ) -> EvidenceExtractionResult:
        extraction_id = uuid4()
        timestamp = now or datetime.now(UTC)
        _validate_reader_link(request, reader_result)
        if reader_result.status is not ReaderExecutionStatus.SUCCESS:
            reason = "reader_result_not_success"
            event = _event(
                EvidenceExtractionEventType.FAILED,
                request=request,
                extraction_id=extraction_id,
                reason=reason,
                timestamp=timestamp,
            )
            return _result(
                request,
                extraction_id=extraction_id,
                status=EvidenceExtractionStatus.BLOCKED,
                extracted_claims=(),
                evidence_items=(),
                failure_reason=reason,
                events=(event,),
            )

        started = _event(
            EvidenceExtractionEventType.STARTED,
            request=request,
            extraction_id=extraction_id,
            reason=request.execution_reason,
            timestamp=timestamp,
        )
        if not request.content_reference.strip():
            completed = _event(
                EvidenceExtractionEventType.COMPLETED,
                request=request,
                extraction_id=extraction_id,
                reason="empty_content_reference",
                timestamp=timestamp,
            )
            return _result(
                request,
                extraction_id=extraction_id,
                status=EvidenceExtractionStatus.EMPTY,
                extracted_claims=(),
                evidence_items=(),
                failure_reason="empty_content_reference",
                events=(started, completed),
            )

        try:
            provider_result = provider_adapter.extract(request.content_reference)
        except Exception as exc:  # Extractor boundary: normalize adapter failures.
            reason = _extractor_error_reason(exc)
            failed = _event(
                EvidenceExtractionEventType.FAILED,
                request=request,
                extraction_id=extraction_id,
                reason=reason,
                timestamp=timestamp,
            )
            return _result(
                request,
                extraction_id=extraction_id,
                status=EvidenceExtractionStatus.FAILED,
                extracted_claims=(),
                evidence_items=(),
                failure_reason=reason,
                events=(started, failed),
            )

        claims = _normalize_items(provider_result.extracted_claims)
        evidence = _normalize_items(provider_result.evidence_items)
        if not evidence:
            completed = _event(
                EvidenceExtractionEventType.COMPLETED,
                request=request,
                extraction_id=extraction_id,
                reason="no_candidate_evidence",
                timestamp=timestamp,
            )
            return _result(
                request,
                extraction_id=extraction_id,
                status=EvidenceExtractionStatus.EMPTY,
                extracted_claims=claims,
                evidence_items=evidence,
                failure_reason="no_candidate_evidence",
                events=(started, completed),
            )

        completed = _event(
            EvidenceExtractionEventType.COMPLETED,
            request=request,
            extraction_id=extraction_id,
            reason="candidate_evidence_extracted",
            timestamp=timestamp,
        )
        return _result(
            request,
            extraction_id=extraction_id,
            status=EvidenceExtractionStatus.SUCCESS,
            extracted_claims=claims,
            evidence_items=evidence,
            failure_reason=None,
            events=(started, completed),
        )


def _validate_reader_link(
    request: EvidenceExtractionRequest,
    reader_result: ReaderExecutionResult,
) -> None:
    if request.reader_execution_id != reader_result.execution_id:
        raise ValueError("extraction request and Reader result do not match")
    if request.source_id != reader_result.source_id:
        raise ValueError("extraction request and Reader source do not match")


def _result(
    request: EvidenceExtractionRequest,
    *,
    extraction_id: UUID,
    status: EvidenceExtractionStatus,
    extracted_claims: tuple[str, ...],
    evidence_items: tuple[str, ...],
    failure_reason: str | None,
    events: tuple[EvidenceExtractionEvent, ...],
) -> EvidenceExtractionResult:
    return EvidenceExtractionResult(
        extraction_id=extraction_id,
        reader_execution_id=request.reader_execution_id,
        source_id=request.source_id,
        question_id=request.question_id,
        gap_id=request.gap_id,
        dimension_key=request.dimension_key,
        status=status,
        candidate_evidence_count=len(evidence_items),
        extracted_claims=extracted_claims,
        evidence_items=evidence_items,
        failure_reason=failure_reason,
        routing_id=request.routing_id,
        alignment_id=request.alignment_id,
        events=events,
    )


def _event(
    event_type: EvidenceExtractionEventType,
    *,
    request: EvidenceExtractionRequest,
    extraction_id: UUID,
    reason: str,
    timestamp: datetime,
) -> EvidenceExtractionEvent:
    return EvidenceExtractionEvent(
        event_type=event_type,
        run_id=request.run_id,
        question_id=request.question_id,
        gap_id=request.gap_id,
        source_id=request.source_id,
        reader_execution_id=request.reader_execution_id,
        extraction_id=extraction_id,
        reason=reason,
        timestamp=timestamp,
    )


def _normalize_items(items: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = item.strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _extractor_error_reason(error: Exception) -> str:
    message = str(error).casefold()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    return error.__class__.__name__.lower()
