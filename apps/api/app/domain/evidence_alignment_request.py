"""Create evidence-to-Gap alignment requests from extraction results."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.evidence_extractor import (
    EvidenceExtractionResult,
    EvidenceExtractionStatus,
)
from app.domain.gap_closure import GapClosureStatus, GapRequirement


@dataclass(frozen=True, slots=True)
class EvidenceAlignmentRequest:
    """One candidate-evidence item awaiting alignment classification."""

    alignment_request_id: UUID
    extraction_id: UUID
    reader_execution_id: UUID
    source_id: str
    evidence_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    claim_id: str | None
    candidate_evidence: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.candidate_evidence.strip():
            raise ValueError("alignment request source and evidence must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "alignment_request_id": str(self.alignment_request_id),
            "extraction_id": str(self.extraction_id),
            "reader_execution_id": str(self.reader_execution_id),
            "source_id": self.source_id,
            "evidence_id": str(self.evidence_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "claim_id": self.claim_id,
            "candidate_evidence": self.candidate_evidence,
        }


class EvidenceAlignmentRequestFactory:
    """Expand extraction evidence items into immutable alignment requests."""

    @staticmethod
    def create(
        extraction_result: EvidenceExtractionResult,
        *,
        requirement: GapRequirement,
    ) -> tuple[EvidenceAlignmentRequest, ...]:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        if extraction_result.status is not EvidenceExtractionStatus.SUCCESS:
            return ()
        if extraction_result.gap_id != requirement.gap_id:
            raise ValueError("extraction and requirement gaps do not match")
        if extraction_result.question_id != requirement.question_id:
            raise ValueError("extraction and requirement questions do not match")
        if extraction_result.dimension_key != requirement.dimension_key:
            raise ValueError("extraction and requirement dimensions do not match")

        return tuple(
            EvidenceAlignmentRequest(
                alignment_request_id=uuid5(
                    NAMESPACE_URL,
                    f"evidence-alignment-request:{extraction_result.extraction_id}:{index}",
                ),
                extraction_id=extraction_result.extraction_id,
                reader_execution_id=extraction_result.reader_execution_id,
                source_id=extraction_result.source_id,
                evidence_id=uuid5(
                    NAMESPACE_URL,
                    f"candidate-evidence:{extraction_result.extraction_id}:{index}",
                ),
                run_id=extraction_result.events[0].run_id
                if extraction_result.events
                else requirement.run_id,
                question_id=requirement.question_id,
                gap_id=requirement.gap_id,
                dimension_key=requirement.dimension_key,
                requirement_type=requirement.requirement_type.value,
                claim_id=(
                    extraction_result.extracted_claims[index]
                    if index < len(extraction_result.extracted_claims)
                    else None
                ),
                candidate_evidence=evidence,
            )
            for index, evidence in enumerate(extraction_result.evidence_items)
        )
