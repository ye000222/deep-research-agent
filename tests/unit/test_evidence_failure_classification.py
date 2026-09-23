"""Phase 14.0-A/E tests: evidence failure classification and need refinement.

These tests prove the classification is driven purely by generic state:
identical evidence/requirement status must produce identical classifications
regardless of question identity or research domain, and the implementation
contains no benchmark-coupled identifiers.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.evidence_failure_classification import (
    OUTDATED_SOURCE_DAYS,
    EvidenceFailureReason,
    EvidenceItemFacts,
    classify_evidence_failure,
    missing_type_for_reason,
)
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.research_need import ResearchNeedType
from app.domain.research_need_refinement import (
    RefinedNeedType,
    refine_research_need,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000505")
NOW = datetime(2026, 4, 1, tzinfo=UTC)


def _requirement(
    *,
    status: GapClosureStatus = GapClosureStatus.PARTIAL,
    question_id: str = "alpha",
    dimension_key: str = "dim-core",
    gap_id: UUID | None = None,
    requirement_type: GapRequirementType = GapRequirementType.EVIDENCE_QUALITY,
    required_evidence: int = 2,
    current_evidence: int = 2,
    required_sources: int = 1,
    current_sources: int = 1,
    verification: VerificationStatus = VerificationStatus.VERIFIED,
) -> GapRequirement:
    return GapRequirement(
        gap_id=gap_id or uuid4(),
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


def _fact(
    status: EvidenceAlignmentStatus = EvidenceAlignmentStatus.ALIGNED,
    *,
    source_reliability: float | None = None,
    confidence: float | None = None,
    is_primary_source: bool | None = None,
    has_quantitative_data: bool | None = None,
    has_method_description: bool | None = None,
    age_days: float | None = None,
) -> EvidenceItemFacts:
    return EvidenceItemFacts(
        evidence_id=uuid4(),
        alignment_status=status,
        accepted=True,
        source_reliability=source_reliability,
        confidence=confidence,
        is_primary_source=is_primary_source,
        has_quantitative_data=has_quantitative_data,
        has_method_description=has_method_description,
        age_days=age_days,
    )


# --------------------------------------------------------------------------
# Classification rules (generic state only)
# --------------------------------------------------------------------------


def test_zero_evidence_classifies_missing_primary_source() -> None:
    requirement = _requirement(
        status=GapClosureStatus.OPEN, current_evidence=0, current_sources=0,
        required_sources=0,
    )
    verdict = classify_evidence_failure(requirement)
    assert EvidenceFailureReason.NO_PRIMARY_SOURCE in verdict.reasons
    assert verdict.gap_id == requirement.gap_id


def test_missing_independent_sources_is_classified() -> None:
    requirement = _requirement(required_sources=2, current_sources=1)
    verdict = classify_evidence_failure(requirement, (_fact(),))
    assert EvidenceFailureReason.NO_INDEPENDENT_VALIDATION in verdict.reasons


def test_unverified_claim_requirement_is_claim_not_supported() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.REQUIRED,
    )
    verdict = classify_evidence_failure(requirement, (_fact(),))
    assert verdict.primary_reason is EvidenceFailureReason.CLAIM_NOT_SUPPORTED


def test_all_misaligned_facts_are_dimension_mismatch() -> None:
    requirement = _requirement()
    facts = (
        _fact(EvidenceAlignmentStatus.NOT_ALIGNED),
        _fact(EvidenceAlignmentStatus.NOT_ALIGNED),
    )
    verdict = classify_evidence_failure(requirement, facts)
    assert verdict.primary_reason is EvidenceFailureReason.DIMENSION_MISMATCH


def test_low_reliability_facts_are_source_quality_low() -> None:
    requirement = _requirement()
    verdict = classify_evidence_failure(
        requirement, (_fact(source_reliability=0.1),)
    )
    assert EvidenceFailureReason.SOURCE_QUALITY_LOW in verdict.reasons


def test_weak_confidence_facts_are_insufficient_detail() -> None:
    requirement = _requirement()
    verdict = classify_evidence_failure(requirement, (_fact(confidence=0.1),))
    assert EvidenceFailureReason.INSUFFICIENT_DETAIL in verdict.reasons


def test_observed_absence_of_quantitative_data_is_classified() -> None:
    requirement = _requirement()
    verdict = classify_evidence_failure(
        requirement, (_fact(has_quantitative_data=False),)
    )
    assert EvidenceFailureReason.NO_QUANTITATIVE_SUPPORT in verdict.reasons


def test_observed_absence_of_method_description_is_classified() -> None:
    requirement = _requirement()
    verdict = classify_evidence_failure(
        requirement, (_fact(has_method_description=False),)
    )
    assert EvidenceFailureReason.NO_METHOD_DESCRIPTION in verdict.reasons


def test_stale_sources_are_classified_outdated() -> None:
    requirement = _requirement()
    verdict = classify_evidence_failure(
        requirement, (_fact(age_days=OUTDATED_SOURCE_DAYS + 100),)
    )
    assert EvidenceFailureReason.OUTDATED_SOURCE in verdict.reasons


def test_unobserved_signals_never_fabricate_defects() -> None:
    """Tri-state metadata: absent signals must not produce defect reasons."""

    requirement = _requirement()
    verdict = classify_evidence_failure(requirement, (_fact(),))
    assert verdict.reasons == (EvidenceFailureReason.UNKNOWN,)
    assert verdict.primary_reason is EvidenceFailureReason.UNKNOWN


def test_classification_as_dict_is_stable_and_json_safe() -> None:
    requirement = _requirement(required_sources=2, current_sources=1)
    payload = classify_evidence_failure(requirement, (_fact(),)).as_dict()
    assert payload["gap_id"] == str(requirement.gap_id)
    assert "no_independent_validation" in payload["failure_reasons"]
    assert isinstance(payload["primary_failure_reason"], str)


# --------------------------------------------------------------------------
# Need refinement mapping (Task E)
# --------------------------------------------------------------------------


def test_claim_not_supported_refines_to_verification_need() -> None:
    refinement = refine_research_need(EvidenceFailureReason.CLAIM_NOT_SUPPORTED)
    assert refinement.refined_need_type is RefinedNeedType.CLAIM_VERIFICATION
    assert refinement.persisted_need_type is ResearchNeedType.CLAIM_VERIFICATION
    assert set(refinement.query_hints) >= {"benchmark", "evaluation", "paper"}


def test_missing_independence_refines_to_third_party_validation() -> None:
    refinement = refine_research_need(
        EvidenceFailureReason.NO_INDEPENDENT_VALIDATION
    )
    assert refinement.persisted_need_type is ResearchNeedType.INDEPENDENT_SOURCE
    assert set(refinement.query_hints) >= {
        "official documentation",
        "research paper",
        "third party analysis",
    }


def test_missing_quantitative_refines_with_measurement_hints() -> None:
    refinement = refine_research_need(
        EvidenceFailureReason.NO_QUANTITATIVE_SUPPORT
    )
    assert refinement.refined_need_type is RefinedNeedType.QUANTITATIVE_EVIDENCE
    assert set(refinement.query_hints) >= {
        "performance",
        "accuracy",
        "benchmark",
        "measurement",
    }


def test_every_failure_reason_refines_to_a_persisted_need_type() -> None:
    for reason in EvidenceFailureReason:
        refinement = refine_research_need(reason)
        assert isinstance(refinement.persisted_need_type, ResearchNeedType)
        assert missing_type_for_reason(reason) is not None


# --------------------------------------------------------------------------
# Genericity (Task F)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("question_id", ["q1", "q5", "q999"])
def test_identity_independence_same_state_same_classification(
    question_id: str,
) -> None:
    requirement = _requirement(
        question_id=question_id,
        gap_id=uuid4(),
        dimension_key=f"dim-{question_id}",
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification=VerificationStatus.REQUIRED,
        required_sources=3,
        current_sources=1,
    )
    facts = (
        _fact(EvidenceAlignmentStatus.PARTIAL, has_quantitative_data=False),
    )
    verdict = classify_evidence_failure(requirement, facts)
    assert verdict.reasons == (
        EvidenceFailureReason.CLAIM_NOT_SUPPORTED,
        EvidenceFailureReason.NO_INDEPENDENT_VALIDATION,
        EvidenceFailureReason.NO_QUANTITATIVE_SUPPORT,
    )
    assert verdict.primary_reason is EvidenceFailureReason.CLAIM_NOT_SUPPORTED


@pytest.mark.parametrize(
    "dimension_key",
    ["industrial-quality", "medical-outcome", "finance-risk", "software-reliability"],
)
def test_domain_independence_same_state_same_verdict(dimension_key: str) -> None:
    requirement = _requirement(
        gap_id=uuid4(),
        dimension_key=dimension_key,
        required_sources=2,
        current_sources=1,
    )
    facts = (_fact(),)
    verdict = classify_evidence_failure(requirement, facts)
    baseline = classify_evidence_failure(
        _requirement(required_sources=2, current_sources=1), facts
    )
    assert verdict.reasons == baseline.reasons
    assert verdict.primary_reason is baseline.primary_reason


def test_implementation_has_no_benchmark_or_identity_coupling() -> None:
    """Scan Phase 14.0 domain modules for forbidden identity tokens."""

    domain_root = Path(__file__).resolve().parents[2] / "apps" / "api" / "app" / "domain"
    targets = [
        "evidence_failure_classification.py",
        "evidence_quality_analyzer.py",
        "claim_verification_analyzer.py",
        "research_need_refinement.py",
        "closure_feedback.py",
    ]
    forbidden_patterns = (
        re.compile(r"\bq\d+\b", re.IGNORECASE),
        re.compile(r"industrial|vision[- ]?defect", re.IGNORECASE),
        re.compile(r"v1_benchmark|benchmark_suite|golden\.json", re.IGNORECASE),
    )
    for name in targets:
        source = (domain_root / name).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert not pattern.search(source), f"{name} matches {pattern.pattern}"
