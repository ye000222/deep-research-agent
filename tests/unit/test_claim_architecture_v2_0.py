from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from app.core.config import Settings
from app.domain.claim_architecture import (
    CanonicalClaim,
    CanonicalClaimType,
    ClaimDiagnosticCode,
    ClaimEvidenceLink,
    ClaimEvidenceRelation,
    ClaimRelationType,
    ClaimVerificationResult,
    ClaimVerificationState,
    deterministic_relation_candidate,
    exact_proposition_key,
)
from app.domain.research_tools import EvidenceCandidate
from app.infrastructure.db.evidence_graph_models import ResearchClaimRow
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow

RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
CLAIM_ID = UUID("00000000-0000-0000-0000-000000000102")
EVIDENCE_ID = UUID("00000000-0000-0000-0000-000000000103")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000104")


def claim_from_fixture(value: dict[str, object], *, claim_id: UUID) -> CanonicalClaim:
    return CanonicalClaim.from_text(
        claim_id=claim_id,
        run_id=RUN_ID,
        question_id="q-test",
        dimension_key="d-test",
        original_text=str(value["text"]),
        claim_type=CanonicalClaimType.QUANTITATIVE,
        subject=str(value["subject"]),
        predicate=str(value["predicate"]),
        object_or_value=str(value["value"]),
        qualifiers=value.get("qualifiers", {}),
        polarity=value.get("polarity"),
    )


def test_golden_claim_semantics_fixtures() -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "claim_semantics_v2_0.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert len(fixtures) == 7
    for index, fixture in enumerate(fixtures):
        left = claim_from_fixture(fixture["left"], claim_id=UUID(int=1000 + index * 2))
        right = claim_from_fixture(fixture["right"], claim_id=UUID(int=1001 + index * 2))
        relation = deterministic_relation_candidate(left, right)
        assert relation.value == fixture["v2_0_result"], fixture["case"]


def test_claim_keeps_original_text_and_all_qualifiers() -> None:
    claim = CanonicalClaim.from_text(
        claim_id=CLAIM_ID,
        run_id=RUN_ID,
        question_id="q1",
        dimension_key="revenue",
        original_text=" Revenue  was $20B in 2025 ",
        claim_type=CanonicalClaimType.QUANTITATIVE,
        qualifiers={
            "time": "2025",
            "date_range": "FY2025",
            "location": "US",
            "unit": "USD",
            "version": "v2",
            "population_scope": "enterprise customers",
            "methodology": "reported revenue",
            "condition": "continuing operations",
        },
    )
    assert claim.original_text == " Revenue  was $20B in 2025 "
    assert claim.normalized_text == "Revenue was $20B in 2025"
    assert claim.qualifiers["time"] == "2025"
    assert claim.qualifiers["version"] == "v2"
    assert len(claim.qualifiers) == 8


def test_claim_record_identity_is_separate_from_provisional_text_key() -> None:
    left = CanonicalClaim.from_text(
        claim_id=CLAIM_ID,
        run_id=RUN_ID,
        question_id="q1",
        dimension_key="d1",
        original_text="The same proposition.",
    )
    right = left.model_copy(update={"claim_id": SOURCE_ID})
    assert left.claim_id != right.claim_id
    assert exact_proposition_key(left) == exact_proposition_key(right)


def test_claim_evidence_link_integrity_and_source_owner_propagation() -> None:
    claim = CanonicalClaim.from_text(
        claim_id=CLAIM_ID,
        run_id=RUN_ID,
        question_id="q1",
        dimension_key="d1",
        original_text="A sufficiently specific factual proposition.",
    )
    link = ClaimEvidenceLink(
        claim_id=CLAIM_ID,
        evidence_id=EVIDENCE_ID,
        relation=ClaimEvidenceRelation.SUPPORTS,
        confidence=0.8,
        source_id=SOURCE_ID,
        source_owner_key="example.org",
        quote_reference="chunk:12-42",
        created_by="fixture",
    )
    link.validate_scope(
        claim=claim,
        run_id=RUN_ID,
        question_id="q1",
        dimension_key="d1",
        evidence_source_id=SOURCE_ID,
        evidence_owner_key="example.org",
    )
    with pytest.raises(ValueError, match="source_owner_key"):
        link.validate_scope(
            claim=claim,
            run_id=RUN_ID,
            question_id="q1",
            dimension_key="d1",
            evidence_source_id=SOURCE_ID,
            evidence_owner_key="other.org",
        )
    with pytest.raises(ValueError, match="dimension"):
        link.validate_scope(
            claim=claim,
            run_id=RUN_ID,
            question_id="q1",
            dimension_key="other-dimension",
            evidence_source_id=SOURCE_ID,
            evidence_owner_key="example.org",
        )


def test_verification_serialization_preserves_unknown() -> None:
    result = ClaimVerificationResult(
        claim_id=CLAIM_ID,
        state=ClaimVerificationState.UNKNOWN,
        reason="historical per-claim verification was not persisted",
    )
    assert result.model_dump(mode="json")["state"] == "unknown"
    assert ClaimDiagnosticCode.VERIFICATION_NOT_COMPUTABLE.value == "VERIFICATION_NOT_COMPUTABLE"
    assert "verification_state" not in ResearchClaimRow.__table__.c


def test_claim_relation_contract_rejects_self_edges() -> None:
    from app.domain.claim_architecture import ClaimRelation

    with pytest.raises(ValueError, match="distinct"):
        ClaimRelation(
            left_claim_id=CLAIM_ID,
            right_claim_id=CLAIM_ID,
            relation=ClaimRelationType.UNKNOWN,
        )


def test_legacy_evidence_schema_remains_nullable_and_compatible() -> None:
    assert ResearchEvidenceRow.__table__.c.claim_id.nullable is True
    assert ResearchEvidenceRow.__table__.c.claim_id.foreign_keys
    assert "atomic_claim" in ResearchClaimRow.__table__.c
    assert "source_owner_key" in ResearchSourceRow.__table__.c
    legacy_candidate = EvidenceCandidate(
        claim="A legacy evidence claim is sufficiently descriptive.",
        exact_quote="A sufficiently long exact quote from an existing source.",
        relevance=0.5,
        confidence=0.5,
    )
    assert legacy_candidate.dimension_key is None


def test_v1_frozen_feature_defaults_are_unchanged_and_v2_is_not_wired() -> None:
    settings = Settings(_env_file=None)
    assert settings.evidence_aware_context_enabled is False
    assert settings.independent_source_targeting_enabled is True
    assert settings.evidence_input_quality_enabled is False
    assert "claim_architecture_v2_enabled" not in Settings.model_fields
