from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import project_gap_requirements
from app.domain.research_action import (
    ResearchActionGenerator,
    SuggestedResearchActionStatus,
    SuggestedResearchActionType,
)
from app.domain.research_need import ResearchNeedGenerator

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
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


def _action_for(question_id: str, dimension_key: str, *, sources: int, required_sources: int):
    requirement = _requirement(
        question_id,
        dimension_key,
        coverage=0.5,
        sources=sources,
        required_sources=required_sources,
        unresolved_claim=question_id == "q4",
    )
    needs = ResearchNeedGenerator.generate(requirement, now=NOW)
    actions = ResearchActionGenerator.generate(
        needs[0],
        requirement=requirement,
        now=NOW,
    )
    return requirement, actions[0]


def test_q7_maps_to_find_independent_source() -> None:
    requirement, action = _action_for("q7", "q7:d2", sources=1, required_sources=2)

    assert action.action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE
    assert action.status is SuggestedResearchActionStatus.SUGGESTED
    assert action.gap_id == requirement.gap_id
    assert "q7:d2" in action.description


def test_q1_maps_to_find_independent_source() -> None:
    _, action = _action_for("q1", "q1:d2", sources=1, required_sources=2)

    assert action.action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE


def test_q4_maps_to_verify_claim() -> None:
    _, action = _action_for("q4", "q4:d1", sources=1, required_sources=2)

    assert action.action_type is SuggestedResearchActionType.VERIFY_CLAIM


def test_closed_q5_does_not_generate_open_action() -> None:
    requirement = _requirement("q5", "q5:d1", coverage=1.0, sources=2, required_sources=2)

    assert ResearchNeedGenerator.generate(requirement) == ()


def test_action_mapping_does_not_change_gap_state_or_research_execution() -> None:
    requirement, action = _action_for("q7", "q7:d2", sources=1, required_sources=2)

    assert action.status is SuggestedResearchActionStatus.SUGGESTED
    assert requirement.closure_status.value == "partial"
    assert action.action_type is not SuggestedResearchActionType.UNKNOWN
