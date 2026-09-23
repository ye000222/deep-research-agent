from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import project_gap_requirements
from app.domain.query_candidate import (
    QueryCandidateGenerator,
    QueryCandidateStatus,
)
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_action import ResearchActionGenerator
from app.domain.research_need import ResearchNeedGenerator
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.research_query_plan import (
    ResearchQueryPlanGenerator,
    ResearchQueryPlanStrategy,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000005")
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


def _candidate_for(
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
    plan = plans[0]
    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]
    candidates = QueryCandidateGenerator.generate(
        plan,
        requirement=requirement,
        validation=validation,
    )
    return requirement, plan, validation, candidates


def test_q7_generates_independent_source_candidate() -> None:
    requirement, plan, validation, candidates = _candidate_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    candidate = candidates[0]
    assert candidate.status is QueryCandidateStatus.GENERATED
    assert candidate.question_id == "q7"
    assert candidate.dimension_key == "q7:d2"
    assert candidate.gap_id == requirement.gap_id
    assert candidate.research_query_plan_id == plan.query_plan_id
    assert candidate.query_purpose == "寻找满足独立来源要求的行业数据"
    assert "third party analysis" in candidate.query_text
    assert str(validation.validation_id) in candidate.generation_reason


def test_q1_generates_independent_source_candidate() -> None:
    _, _, _, candidates = _candidate_for(
        "q1",
        "q1:d2",
        sources=1,
        required_sources=2,
    )

    assert candidates[0].question_id == "q1"
    assert "official statistics" in candidates[0].query_text


def test_q4_generates_claim_validation_candidate() -> None:
    _, _, _, candidates = _candidate_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )

    assert "claim" in candidates[0].query_text
    assert candidates[0].query_purpose == "寻找能够验证或反驳 Claim 的官方或第三方依据"


def test_closed_gap_does_not_generate_candidate() -> None:
    requirement = _requirement(
        "q5",
        "q5:d1",
        coverage=1.0,
        sources=2,
        required_sources=2,
    )

    assert ResearchNeedGenerator.generate(requirement) == ()


def test_invalid_query_plan_does_not_generate_candidate() -> None:
    requirement, plan, _, _ = _candidate_for(
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

    assert (
        QueryCandidateGenerator.generate(
            wrong_plan,
            requirement=requirement,
            validation=validation,
        )
        == ()
    )


def test_candidate_generation_is_observation_only() -> None:
    requirement, plan, _, candidates = _candidate_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    assert candidates[0].status is QueryCandidateStatus.GENERATED
    assert requirement.closure_status.value == "partial"
    assert plan.status.value == "suggested"
