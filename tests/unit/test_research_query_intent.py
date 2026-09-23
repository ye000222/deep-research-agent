from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import project_gap_requirements
from app.domain.research_action import ResearchActionGenerator
from app.domain.research_need import ResearchNeedGenerator
from app.domain.research_query_intent import (
    ResearchQueryIntentGenerator,
    ResearchQueryIntentStatus,
    ResearchQueryIntentType,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000002")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _requirement(
    question_id: str,
    dimension_key: str,
    *,
    coverage: float,
    sources: int,
    required_sources: int,
    unresolved_claim: bool = False,
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
                        "coverage": coverage,
                        "accepted_evidence": 4,
                        "independent_sources": sources,
                        "required_sources": required_sources,
                    }
                ],
            }
        ],
        claim_states=(
            {"claim": {"dimension_key": dimension_key, "unresolved": True}}
            if unresolved_claim
            else None
        ),
        now=NOW,
    )[0]


def _intent_for(
    question_id: str,
    dimension_key: str,
    *,
    sources: int,
    required_sources: int,
    unresolved_claim: bool = False,
):
    requirement = _requirement(
        question_id,
        dimension_key,
        coverage=0.5,
        sources=sources,
        required_sources=required_sources,
        unresolved_claim=unresolved_claim,
    )
    needs = ResearchNeedGenerator.generate(requirement, now=NOW)
    actions = ResearchActionGenerator.generate(
        needs[0],
        requirement=requirement,
        now=NOW,
    )
    intents = ResearchQueryIntentGenerator.generate(
        actions[0],
        requirement=requirement,
        now=NOW,
    )
    return requirement, actions[0], intents[0]


def test_q7_maps_to_independent_source_intent() -> None:
    requirement, action, intent = _intent_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    assert intent.intent_type is ResearchQueryIntentType.FIND_INDEPENDENT_SOURCE
    assert intent.status is ResearchQueryIntentStatus.SUGGESTED
    assert intent.run_id == requirement.run_id
    assert intent.question_id == "q7"
    assert intent.dimension_key == "q7:d2"
    assert intent.suggested_action_id == action.action_id
    assert "government" in intent.preferred_source_types


def test_q1_preserves_dimension_and_source_intent() -> None:
    _, _, intent = _intent_for("q1", "q1:d2", sources=1, required_sources=2)

    assert intent.intent_type is ResearchQueryIntentType.FIND_INDEPENDENT_SOURCE
    assert intent.question_id == "q1"
    assert intent.dimension_key == "q1:d2"


def test_q4_maps_to_claim_verification_intent() -> None:
    _, _, intent = _intent_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )

    assert intent.intent_type is ResearchQueryIntentType.VERIFY_CLAIM
    assert "official" in intent.preferred_source_types


def test_closed_gap_does_not_generate_query_intent() -> None:
    requirement = _requirement(
        "q5",
        "q5:d1",
        coverage=1.0,
        sources=2,
        required_sources=2,
    )

    assert ResearchNeedGenerator.generate(requirement) == ()


def test_intent_generation_is_observation_only() -> None:
    requirement, action, intent = _intent_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    assert intent.status is ResearchQueryIntentStatus.SUGGESTED
    assert action.status.value == "suggested"
    assert requirement.closure_status.value == "partial"
    assert intent.intent_type is not ResearchQueryIntentType.UNKNOWN
