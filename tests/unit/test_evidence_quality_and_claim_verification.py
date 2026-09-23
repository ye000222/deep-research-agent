"""Phase 14.0-B/C/D tests: quality analysis, claim verification, metadata.

Covers the EvidenceQualityAnalyzer status model, the three claim-support
cases, and the additive ``ClosureFeedback.metadata`` integration: the
dispatcher chain (feedback -> need -> action) must keep producing identical
artifacts whether or not analysis metadata is attached.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.claim_verification_analyzer import (
    ClaimSupportLevel,
    ClaimVerificationAnalyzer,
)
from app.domain.closure_feedback import ClosureFeedbackReason
from app.domain.closure_feedback_completeness import generate_completeness_feedback
from app.domain.closure_feedback_dispatcher import (
    ClosureFeedbackDispatcher,
    FeedbackDispatchStatus,
)
from app.domain.evidence_alignment import (
    EvidenceAlignment,
    EvidenceAlignmentReason,
    EvidenceAlignmentStatus,
)
from app.domain.evidence_failure_classification import (
    EvidenceFailureReason,
    MissingEvidenceType,
)
from app.domain.evidence_quality_analyzer import (
    EvidenceQualityAnalyzer,
    EvidenceQualityStatus,
    build_feedback_analysis,
)
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.research_action import SuggestedResearchActionType
from app.domain.research_need import ResearchNeedType

RUN_ID = UUID("00000000-0000-0000-0000-000000000606")
NOW = datetime(2026, 4, 1, tzinfo=UTC)


def _requirement(
    *,
    status: GapClosureStatus = GapClosureStatus.PARTIAL,
    question_id: str = "alpha",
    dimension_key: str = "dim-core",
    requirement_type: GapRequirementType = GapRequirementType.EVIDENCE_QUALITY,
    required_evidence: int = 2,
    current_evidence: int = 2,
    required_sources: int = 1,
    current_sources: int = 1,
    verification: VerificationStatus = VerificationStatus.VERIFIED,
) -> GapRequirement:
    return GapRequirement(
        gap_id=uuid4(),
        run_id=RUN_ID,
        question_id=question_id,
        dimension_key=dimension_key,
        requirement_type=requirement_type,
        criterion="generic criterion",
        required_evidence_count=required_evidence,
        required_independent_sources=required_sources,
        current_evidence_count=current_evidence,
        current_independent_sources=current_sources,
        verification_status=verification,
        closure_status=status,
        created_at=NOW,
        updated_at=NOW,
        state_version=3,
    )


def _alignment(
    requirement: GapRequirement,
    status: EvidenceAlignmentStatus,
    reason: EvidenceAlignmentReason | None = None,
) -> EvidenceAlignment:
    return EvidenceAlignment(
        alignment_id=uuid4(),
        evidence_id=uuid4(),
        gap_id=requirement.gap_id,
        question_id=requirement.question_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type,
        satisfies_requirement=status is EvidenceAlignmentStatus.ALIGNED,
        alignment_status=status,
        rejection_reason=reason if status is not EvidenceAlignmentStatus.ALIGNED else None,
        created_at=NOW,
        state_version=requirement.state_version,
    )


# --------------------------------------------------------------------------
# B. Evidence quality analyzer
# --------------------------------------------------------------------------


def test_zero_evidence_assessment_is_insufficient() -> None:
    requirement = _requirement(
        status=GapClosureStatus.OPEN, current_evidence=0, current_sources=0,
        required_sources=0,
    )
    assessment = EvidenceQualityAnalyzer().assess(requirement)
    assert assessment.quality_status is EvidenceQualityStatus.INSUFFICIENT
    assert EvidenceFailureReason.NO_PRIMARY_SOURCE in assessment.failure_reasons
    assert assessment.missing_evidence_type is MissingEvidenceType.PRIMARY_SOURCE


def test_unverified_claim_assessment_points_at_verification_support() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.REQUIRED,
    )
    assessment = EvidenceQualityAnalyzer().assess(
        requirement,
        claim_id=uuid4(),
    )
    assert assessment.quality_status is EvidenceQualityStatus.DEGRADED
    assert assessment.missing_evidence_type is MissingEvidenceType.VERIFICATION_SUPPORT
    assert assessment.recommended_need_type is ResearchNeedType.CLAIM_VERIFICATION
    assert "benchmark" in assessment.query_hints


def test_closed_requirement_assessment_is_sufficient() -> None:
    requirement = _requirement(status=GapClosureStatus.CLOSED)
    assessment = EvidenceQualityAnalyzer().assess(requirement)
    assert assessment.quality_status is EvidenceQualityStatus.SUFFICIENT


def test_assessment_as_dict_is_json_serializable() -> None:
    requirement = _requirement(required_sources=2, current_sources=1)
    payload = EvidenceQualityAnalyzer().assess(requirement).as_dict()
    json.dumps(payload)
    assert payload["gap_id"] == str(requirement.gap_id)
    assert "no_independent_validation" in payload["failure_reasons"]


# --------------------------------------------------------------------------
# C. Claim verification analyzer (three cases from the spec)
# --------------------------------------------------------------------------


def test_case1_direct_support_when_verified_and_independent() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.VERIFIED,
        required_sources=2,
        current_sources=2,
    )
    facts = build_feedback_analysis(requirement, ())
    assert facts["support_level"] == ClaimSupportLevel.DIRECT_SUPPORT.value
    result = ClaimVerificationAnalyzer().analyze(requirement)
    assert result.support_level is ClaimSupportLevel.DIRECT_SUPPORT


def test_case2_relevant_only_when_evidence_cannot_prove_claim() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.REQUIRED,
    )
    result = ClaimVerificationAnalyzer().analyze(requirement)
    assert result.support_level is ClaimSupportLevel.RELEVANT_ONLY
    assert result.missing_support_type is MissingEvidenceType.VERIFICATION_SUPPORT


def test_case3_single_source_needs_independent_validation() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.VERIFIED,
        required_sources=2,
        current_sources=1,
    )
    result = ClaimVerificationAnalyzer().analyze(requirement)
    assert result.support_level is ClaimSupportLevel.SINGLE_SOURCE
    assert result.missing_support_type is MissingEvidenceType.INDEPENDENT_VALIDATION


def test_unsupported_claim_without_any_evidence() -> None:
    requirement = _requirement(
        status=GapClosureStatus.OPEN, current_evidence=0, current_sources=0,
        required_sources=0,
    )
    result = ClaimVerificationAnalyzer().analyze(requirement)
    assert result.support_level is ClaimSupportLevel.UNSUPPORTED
    assert result.missing_support_type is MissingEvidenceType.SUPPORTING_EVIDENCE


def test_claim_result_as_dict_roundtrip() -> None:
    requirement = _requirement(required_sources=2, current_sources=1)
    claim_id = uuid4()
    payload = ClaimVerificationAnalyzer().analyze(
        requirement, claim_id=claim_id
    ).as_dict()
    json.dumps(payload)
    assert payload["claim_id"] == str(claim_id)
    assert payload["verification_status"] == "verified"


# --------------------------------------------------------------------------
# D. ClosureFeedback metadata integration (additive, chain-compatible)
# --------------------------------------------------------------------------


def test_feedback_defaults_to_empty_metadata_for_old_paths() -> None:
    requirement = _requirement(required_sources=2, current_sources=1)
    feedback = generate_completeness_feedback(requirement)
    assert feedback is not None
    assert feedback.metadata == {}


def test_metadata_survives_serialization_and_keeps_legacy_keys() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.REQUIRED,
        required_sources=2,
        current_sources=1,
    )
    alignment = _alignment(
        requirement,
        EvidenceAlignmentStatus.PARTIAL,
        EvidenceAlignmentReason.CLAIM_NOT_VERIFIED,
    )
    feedback = generate_completeness_feedback(requirement)
    assert feedback is not None
    metadata = build_feedback_analysis(requirement, (alignment,))
    enriched = replace(feedback, metadata=metadata)
    payload = enriched.as_dict()
    json.dumps(payload)
    # Legacy keys required by closure.feedback.generated stay present.
    for key in (
        "feedback_id",
        "gap_id",
        "failure_reason",
        "missing_requirement",
        "recommended_need_type",
    ):
        assert key in payload
    assert payload["metadata"]["primary_failure_reason"] == "claim_not_supported"
    assert payload["metadata"]["missing_support_type"] == "verification_support"
    assert "benchmark" in payload["metadata"]["query_hints"]


def test_dispatcher_chain_is_unchanged_by_metadata() -> None:
    requirement = _requirement(required_sources=2, current_sources=1)
    plain = generate_completeness_feedback(requirement)
    assert plain is not None
    assert plain.failure_reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE
    alignment = _alignment(
        requirement,
        EvidenceAlignmentStatus.PARTIAL,
        EvidenceAlignmentReason.MISSING_INDEPENDENT_SOURCE,
    )
    enriched = replace(
        plain, metadata=build_feedback_analysis(requirement, (alignment,))
    )
    dispatch_plain = ClosureFeedbackDispatcher().dispatch(plain, requirement)
    dispatch_enriched = ClosureFeedbackDispatcher().dispatch(
        enriched, requirement, alignments=(alignment,)
    )
    assert dispatch_plain.dispatch_status is FeedbackDispatchStatus.CREATED
    assert dispatch_enriched.dispatch_status is FeedbackDispatchStatus.CREATED
    # Same ids and same artifacts: metadata is purely explanatory.
    assert dispatch_plain.research_need_id == dispatch_enriched.research_need_id
    assert dispatch_enriched.research_needs[0].need_type is (
        ResearchNeedType.INDEPENDENT_SOURCE
    )
    assert dispatch_enriched.suggested_actions[0].action_type is (
        SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE
    )
