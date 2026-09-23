"""Build evidence-extraction requests from Reader execution results."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.domain.reader_executor import ReaderExecutionResult, ReaderExecutionStatus


@dataclass(frozen=True, slots=True)
class EvidenceExtractionRequest:
    """A request linking source content to a future extraction attempt."""

    extraction_request_id: UUID
    reader_execution_id: UUID
    routing_id: UUID
    alignment_id: UUID
    source_id: str
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    content_reference: str
    execution_reason: str
    reader_status: ReaderExecutionStatus

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("extraction source must not be empty")
        if not self.execution_reason.strip():
            raise ValueError("extraction reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "extraction_request_id": str(self.extraction_request_id),
            "reader_execution_id": str(self.reader_execution_id),
            "routing_id": str(self.routing_id),
            "alignment_id": str(self.alignment_id),
            "source_id": self.source_id,
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "content_reference": self.content_reference,
            "execution_reason": self.execution_reason,
            "reader_status": self.reader_status.value,
        }


class EvidenceExtractionRequestFactory:
    """Create extraction requests without invoking an extractor."""

    @staticmethod
    def create(
        reader_result: ReaderExecutionResult,
        *,
        extraction_request_id: UUID,
        requirement_type: str,
        content_reference: str,
        execution_reason: str = "reader execution completed",
    ) -> EvidenceExtractionRequest:
        return EvidenceExtractionRequest(
            extraction_request_id=extraction_request_id,
            reader_execution_id=reader_result.execution_id,
            routing_id=reader_result.routing_id,
            alignment_id=reader_result.alignment_id,
            source_id=reader_result.source_id,
            run_id=reader_result.run_id,
            question_id=reader_result.question_id,
            gap_id=reader_result.gap_id,
            dimension_key=reader_result.dimension_key,
            requirement_type=requirement_type,
            content_reference=content_reference,
            execution_reason=execution_reason,
            reader_status=reader_result.status,
        )
