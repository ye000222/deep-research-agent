from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.domain.closure_evaluation_context import ClosureEvaluationContext
from app.domain.evidence_alignment_executor import (
    EvidenceAlignmentEventType,
    EvidenceAlignmentExecutor,
)
from app.domain.evidence_alignment_request import EvidenceAlignmentRequestFactory
from app.domain.evidence_extractor import (
    EvidenceExtractionResult,
    EvidenceExtractionStatus,
)
from app.domain.gap_closure import (
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirementType,
    VerificationStatus,
    project_gap_requirements,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000039")
READER_ID = UUID("00000000-0000-0000-0000-000000000040")
ROUTING_ID = UUID("00000000-0000-0000-0000-000000000041")
ALIGNMENT_ID = UUID("00000000-0000-0000-0000-000000000042")
EXTRACTION_ID = UUID("00000000-0000-0000-0000-000000000043")
GAP_ID = UUID("00000000-0000-0000-0000-000000000044")


def _requirement(
    question_id: str,
    dimension_key: str,
    requirement_type: GapRequirementType,
    *,
    required_sources: int = 1,
):
    return project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=4,
        coverage_map=[
            {
                "dimension_key": question_id,
                "requirement_statuses": [
                    {
                        "dimension_key": dimension_key,
                        "coverage": 0.5,
                        "accepted_evidence": 2,
                        "independent_sources": 1,
                        "required_sources": required_sources,
                    }
                ],
            }
        ],
        claim_states=(
            {"claim": {"dimension_key": dimension_key, "unresolved": True}}
            if requirement_type is GapRequirementType.CLAIM_VERIFICATION
            else None
        ),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )[0]


def _extraction(
    requirement,
    *,
    evidence: tuple[str, ...] = ("candidate evidence",),
    claims: tuple[str, ...] = ("claim",),
):
    return EvidenceExtractionResult(
        extraction_id=EXTRACTION_ID,
        reader_execution_id=READER_ID,
        source_id="https://industry-report.example/report",
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        dimension_key=requirement.dimension_key,
        status=EvidenceExtractionStatus.SUCCESS,
        candidate_evidence_count=len(evidence),
        extracted_claims=claims,
        evidence_items=evidence,
        failure_reason=None,
        routing_id=ROUTING_ID,
        alignment_id=ALIGNMENT_ID,
    )


def _request(requirement, extraction):
    return EvidenceAlignmentRequestFactory.create(
        extraction,
        requirement=requirement,
    )[0]


def test_matching_evidence_is_aligned() -> None:
    requirement = _requirement(
        "q7",
        "q7:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
        required_sources=2,
    )
    extraction = _extraction(requirement)
    execution = EvidenceAlignmentExecutor.execute(
        _request(requirement, extraction),
        requirement=requirement,
        independent_source_count=2,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert execution.alignment is not None
    assert execution.alignment.alignment_status.value == "aligned"
    assert execution.alignment.extraction_id == EXTRACTION_ID
    assert execution.events[-1].event_type is EvidenceAlignmentEventType.COMPLETED


def test_q7_missing_independent_source_is_partial() -> None:
    requirement = _requirement(
        "q7",
        "q7:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
        required_sources=2,
    )
    extraction = _extraction(requirement)
    execution = EvidenceAlignmentExecutor.execute(
        _request(requirement, extraction),
        requirement=requirement,
        independent_source_count=1,
    )

    assert execution.alignment is not None
    assert execution.alignment.alignment_status.value == "partial"
    assert execution.alignment.rejection_reason.value == "missing_independent_source"


def test_claim_requirement_with_wrong_dimension_is_not_aligned() -> None:
    requirement = _requirement(
        "q4",
        "q4:d1",
        GapRequirementType.CLAIM_VERIFICATION,
        required_sources=2,
    )
    extraction = _extraction(requirement)
    execution = EvidenceAlignmentExecutor.execute(
        _request(requirement, extraction),
        requirement=requirement,
        evidence_dimension_key="q4:unrelated",
        claim_verified=False,
    )

    assert execution.alignment is not None
    assert execution.alignment.alignment_status.value == "not_aligned"
    assert execution.alignment.rejection_reason.value == "dimension_mismatch"


def test_closed_gap_generates_no_alignment() -> None:
    requirement = _requirement(
        "q7",
        "q7:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    closed = replace(requirement, closure_status=GapClosureStatus.CLOSED)
    extraction = _extraction(closed)

    assert EvidenceAlignmentRequestFactory.create(extraction, requirement=closed) == ()
    # No request means the alignment executor cannot be invoked for a closed Gap.


def test_closure_context_counts_alignments_without_closing_gap() -> None:
    requirement = _requirement(
        "q7",
        "q7:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
        required_sources=2,
    )
    extraction = _extraction(requirement)
    execution = EvidenceAlignmentExecutor.execute(
        _request(requirement, extraction),
        requirement=requirement,
        independent_source_count=1,
    )
    assert execution.alignment is not None

    context = ClosureEvaluationContext.from_alignments(
        gap_id=requirement.gap_id,
        before_snapshot=ClosureSnapshot(
            evidence_count=requirement.current_evidence_count,
            independent_sources=requirement.current_independent_sources,
            verification_status=VerificationStatus.NOT_EVALUATED,
        ),
        evidence_alignments=(execution.alignment,),
        independent_source_count=1,
    )

    assert context.new_evidence_count == 1
    assert context.partial_evidence_count == 1
    assert context.aligned_evidence_count == 0
    assert requirement.closure_status is GapClosureStatus.PARTIAL
