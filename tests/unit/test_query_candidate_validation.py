from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import GapClosureStatus, project_gap_requirements
from app.domain.query_candidate import QueryCandidateGenerator
from app.domain.query_candidate_validation import (
    QueryCandidateAlignmentType,
    QueryCandidateValidationStatus,
    QueryCandidateValidator,
)
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_action import ResearchActionGenerator
from app.domain.research_need import ResearchNeedGenerator
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.research_query_plan import ResearchQueryPlanGenerator

RUN_ID = UUID("00000000-0000-0000-0000-000000000006")
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
    plan_validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]
    candidates = QueryCandidateGenerator.generate(
        plan,
        requirement=requirement,
        validation=plan_validation,
    )
    return requirement, plan, candidates[0]


def test_q7_candidate_is_gap_aligned() -> None:
    requirement, plan, candidate = _candidate_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    validation = QueryCandidateValidator.validate(
        candidate,
        plan=plan,
        requirement=requirement,
    )[0]

    assert validation.validation_status is QueryCandidateValidationStatus.VALID
    assert validation.alignment_type is QueryCandidateAlignmentType.GAP_ALIGNED
    assert validation.query_candidate_id == candidate.query_candidate_id


def test_q1_candidate_is_valid_for_independent_source_gap() -> None:
    requirement, plan, candidate = _candidate_for(
        "q1",
        "q1:d2",
        sources=1,
        required_sources=2,
    )

    validation = QueryCandidateValidator.validate(
        candidate,
        plan=plan,
        requirement=requirement,
    )[0]

    assert validation.validation_status is QueryCandidateValidationStatus.VALID


def test_q4_candidate_is_valid_for_claim_verification() -> None:
    requirement, plan, candidate = _candidate_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )

    validation = QueryCandidateValidator.validate(
        candidate,
        plan=plan,
        requirement=requirement,
    )[0]

    assert validation.validation_status is QueryCandidateValidationStatus.VALID
    assert validation.alignment_type is QueryCandidateAlignmentType.GAP_ALIGNED


def test_claim_gap_with_market_statistics_query_is_invalid() -> None:
    requirement, plan, candidate = _candidate_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )
    wrong_candidate = replace(
        candidate,
        query_text="market size statistics annual data",
        preferred_source_types=("dataset",),
    )

    validation = QueryCandidateValidator.validate(
        wrong_candidate,
        plan=plan,
        requirement=requirement,
    )[0]

    assert validation.validation_status is QueryCandidateValidationStatus.INVALID
    assert validation.alignment_type is QueryCandidateAlignmentType.SOURCE_MISMATCH


def test_claim_gap_with_non_claim_topic_is_topic_mismatch() -> None:
    requirement, plan, candidate = _candidate_for(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )
    wrong_candidate = replace(
        candidate,
        query_text="market size statistics annual data",
    )

    validation = QueryCandidateValidator.validate(
        wrong_candidate,
        plan=plan,
        requirement=requirement,
    )[0]

    assert validation.validation_status is QueryCandidateValidationStatus.INVALID
    assert validation.alignment_type is QueryCandidateAlignmentType.TOPIC_MISMATCH


def test_closed_gap_does_not_generate_validation() -> None:
    requirement, plan, candidate = _candidate_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )
    closed_requirement = replace(
        requirement,
        closure_status=GapClosureStatus.CLOSED,
    )

    assert (
        QueryCandidateValidator.validate(
            candidate,
            plan=plan,
            requirement=closed_requirement,
        )
        == ()
    )


def test_candidate_validation_is_observation_only() -> None:
    requirement, plan, candidate = _candidate_for(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    validation = QueryCandidateValidator.validate(
        candidate,
        plan=plan,
        requirement=requirement,
    )[0]

    assert validation.validation_status is QueryCandidateValidationStatus.VALID
    assert candidate.query_text.startswith("q7:d2")
    assert plan.status.value == "suggested"
    assert requirement.closure_status.value == "partial"
