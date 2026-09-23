"""Classify candidate evidence against GapRequirements without closing them."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from app.domain.evidence_alignment import EvidenceAlignment
from app.domain.evidence_alignment_request import EvidenceAlignmentRequest
from app.domain.gap_closure import GapClosureStatus, GapRequirement


class EvidenceAlignmentEventType(StrEnum):
    STARTED = "evidence.alignment.started"
    COMPLETED = "evidence.alignment.completed"
    FAILED = "evidence.alignment.failed"


@dataclass(frozen=True, slots=True)
class EvidenceAlignmentEvent:
    event_type: EvidenceAlignmentEventType
    run_id: UUID
    question_id: str
    gap_id: UUID
    evidence_id: UUID
    extraction_id: UUID
    alignment_id: UUID
    reason: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class EvidenceAlignmentExecution:
    """Alignment result plus lifecycle observations."""

    alignment: EvidenceAlignment | None
    events: tuple[EvidenceAlignmentEvent, ...]


class EvidenceAlignmentExecutor:
    """Use existing EvidenceAlignment rules without Acceptance or Gap writes."""

    @staticmethod
    def execute(
        request: EvidenceAlignmentRequest,
        *,
        requirement: GapRequirement,
        evidence_dimension_key: str | None = None,
        evidence_quality_passed: bool = True,
        independent_source_count: int | None = None,
        claim_verified: bool = False,
        now: datetime | None = None,
    ) -> EvidenceAlignmentExecution:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return EvidenceAlignmentExecution(alignment=None, events=())
        _validate_request(request, requirement)
        timestamp = now or datetime.now(UTC)
        alignment_id = uuid4()
        started = EvidenceAlignmentEvent(
            event_type=EvidenceAlignmentEventType.STARTED,
            run_id=request.run_id,
            question_id=request.question_id,
            gap_id=request.gap_id,
            evidence_id=request.evidence_id,
            extraction_id=request.extraction_id,
            alignment_id=alignment_id,
            reason="alignment_started",
            timestamp=timestamp,
        )
        alignment = EvidenceAlignment.evaluate(
            requirement,
            evidence_id=request.evidence_id,
            evidence_dimension_key=evidence_dimension_key or request.dimension_key,
            evidence_quality_passed=evidence_quality_passed,
            independent_source_count=(
                independent_source_count
                if independent_source_count is not None
                else requirement.current_independent_sources
            ),
            claim_verified=claim_verified,
            state_version=requirement.state_version,
            created_at=timestamp,
        )
        alignment = replace(
            alignment,
            alignment_id=alignment_id,
            extraction_id=request.extraction_id,
            reader_execution_id=request.reader_execution_id,
            alignment_request_id=request.alignment_request_id,
        )
        completed = EvidenceAlignmentEvent(
            event_type=EvidenceAlignmentEventType.COMPLETED,
            run_id=request.run_id,
            question_id=request.question_id,
            gap_id=request.gap_id,
            evidence_id=request.evidence_id,
            extraction_id=request.extraction_id,
            alignment_id=alignment_id,
            reason=alignment.alignment_status.value,
            timestamp=timestamp,
        )
        return EvidenceAlignmentExecution(
            alignment=alignment,
            events=(started, completed),
        )


def _validate_request(
    request: EvidenceAlignmentRequest,
    requirement: GapRequirement,
) -> None:
    if request.gap_id != requirement.gap_id:
        raise ValueError("alignment request and requirement gaps do not match")
    if request.question_id != requirement.question_id:
        raise ValueError("alignment request and requirement questions do not match")
    if request.dimension_key != requirement.dimension_key:
        raise ValueError("alignment request and requirement dimensions do not match")
