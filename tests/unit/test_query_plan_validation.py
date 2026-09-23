from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import project_gap_requirements
from app.domain.query_plan_validation import (
    QueryPlanAlignmentType,
    QueryPlanValidationStatus,
    QueryPlanValidator,
)
from app.domain.research_action import ResearchActionGenerator
from app.domain.research_need import ResearchNeedGenerator
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.research_query_plan import (
    ResearchQueryPlanGenerator,
    ResearchQueryPlanStrategy,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000004")
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


def _plan_for(
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
    plans = ResearchQueryPlanGenerator.generate(
        intents[0],
        requirement=requirement,
        now=NOW,
    )
    return requirement, plans[0]


def test_q7_plan_is_valid_for_independent_source_gap() -> None:
    requirement, plan = _plan_for("q7", "q7:d2", sources=1, required_sources=2)

    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]

    assert validation.validation_status is QueryPlanValidationStatus.VALID
    assert validation.alignment_type is QueryPlanAlignmentType.DIRECT_ALIGNMENT
    assert validation.gap_id == requirement.gap_id
    assert validation.question_id == "q7"


def test_q1_plan_is_valid_for_independent_source_gap() -> None:
    requirement, plan = _plan_for("q1", "q1:d2", sources=1, required_sources=2)

    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]

    assert validation.validation_status is QueryPlanValidationStatus.VALID


def test_q4_plan_is_valid_for_claim_verification_gap() -> None:
    requirement, plan = _plan_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )

    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]

    assert validation.validation_status is QueryPlanValidationStatus.VALID
    assert validation.alignment_type is QueryPlanAlignmentType.DIRECT_ALIGNMENT


def test_claim_verification_with_data_collection_is_invalid() -> None:
    requirement, plan = _plan_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )
    wrong_plan = replace(
        plan,
        search_strategy_type=ResearchQueryPlanStrategy.DATA_COLLECTION,
    )

    validation = QueryPlanValidator.validate(wrong_plan, requirement=requirement)[0]

    assert validation.validation_status is QueryPlanValidationStatus.INVALID
    assert validation.alignment_type is QueryPlanAlignmentType.CLAIM_MISMATCH
    assert "query_strategy_cannot_directly_validate_claim" in validation.issues


def test_closed_gap_does_not_generate_validation() -> None:
    requirement = _requirement(
        "q5",
        "q5:d1",
        coverage=1.0,
        sources=2,
        required_sources=2,
    )

    assert QueryPlanValidator.validate(  # type: ignore[arg-type]
        object(),
        requirement=requirement,
    ) == ()


def test_validation_is_observation_only() -> None:
    requirement, plan = _plan_for("q7", "q7:d2", sources=1, required_sources=2)
    before = plan

    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]

    assert validation.validation_status is QueryPlanValidationStatus.VALID
    assert plan == before
    assert requirement.closure_status.value == "partial"
