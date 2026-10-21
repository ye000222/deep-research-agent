from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.domain.claim_architecture import CanonicalClaim, CanonicalClaimType, ClaimRelationType
from app.domain.semantic_claim_equivalence import (
    MATCHER_VERSION,
    QualifierCompatibility,
    SemanticClaimMatcher,
    SemanticMatchReason,
    normalize_safe_text,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "claim_semantics_v2_1.json"


def _claim(data: dict[str, object], *, run_id: UUID | None = None) -> CanonicalClaim:
    text = str(data["text"])
    return CanonicalClaim.from_text(
        claim_id=uuid4(),
        run_id=run_id or UUID("00000000-0000-0000-0000-000000000001"),
        question_id="q1",
        dimension_key="q1:d1",
        original_text=text,
        claim_type=CanonicalClaimType.QUANTITATIVE,
        subject=data.get("subject"),
        predicate=data.get("predicate"),
        object_or_value=data.get("value"),
        qualifiers=data.get("qualifiers", {}),
        polarity=data.get("polarity"),
    )


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    ids=lambda case: case["case"],
)
def test_golden_semantic_relations(case: dict[str, object]) -> None:
    result = SemanticClaimMatcher().compare(_claim(case["left"]), _claim(case["right"]))
    assert result.relation.value == case["relation"]
    assert result.rationale
    assert result.reason_codes
    assert result.method


def test_text_normalization_preserves_negation_and_qualifier_tokens() -> None:
    normalized = normalize_safe_text("  Model X v1 — NOT supported in 2024 ")
    assert normalized == "model x v1 - not supported in 2024"


def test_scope_and_run_are_not_equivalence_scopes() -> None:
    same = {
        "text": "Revenue was $20B.",
        "subject": "revenue",
        "predicate": "amount",
        "value": "20 billion",
        "qualifiers": {"unit": "USD"},
    }
    left = _claim(same)
    other_question = left.model_copy(update={"question_id": "q2"})
    assert (
        SemanticClaimMatcher().compare(left, other_question).relation
        is ClaimRelationType.RELATED
    )
    other_run = left.model_copy(update={"run_id": uuid4()})
    assert SemanticClaimMatcher().compare(left, other_run).relation is ClaimRelationType.UNKNOWN


def test_unknown_result_is_conservative_and_explainable() -> None:
    left = _claim({"text": "Market conditions shifted.", "qualifiers": {}})
    right = _claim({"text": "Market dynamics changed.", "qualifiers": {}})
    result = SemanticClaimMatcher().compare(left, right)
    assert result.relation is ClaimRelationType.UNKNOWN
    assert result.method == "fallback_unknown"
    assert SemanticMatchReason.INSUFFICIENT_INFORMATION in result.reason_codes
    assert result.qualifier_compatibility is QualifierCompatibility.COMPATIBLE


def test_matcher_version_is_new_and_research_contract_is_passive() -> None:
    assert MATCHER_VERSION == "claim-equivalence-v2-v1"
    # The matcher only accepts immutable CanonicalClaim values and returns a
    # comparison; it does not mutate claims or create persistence side effects.
    claim = _claim(
        {
            "text": "The company was founded in 2018.",
            "subject": "company",
            "predicate": "founded",
            "value": "2018",
            "qualifiers": {"time": "2018"},
        }
    )
    before = claim.model_dump(mode="json")
    SemanticClaimMatcher().compare(claim, claim.model_copy(update={"claim_id": uuid4()}))
    assert claim.model_dump(mode="json") == before
