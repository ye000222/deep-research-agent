from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, project_gap_requirements
from app.domain.query_candidate import QueryCandidateGenerator, QueryCandidateStatus
from app.domain.query_candidate_ranking import QueryCandidateRanker
from app.domain.query_candidate_validation import QueryCandidateValidator
from app.domain.query_execution import (
    QueryExecutionAdapter,
    QueryExecutionEligibility,
)
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_action import ResearchActionGenerator
from app.domain.research_need import ResearchNeedGenerator
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.research_query_plan import ResearchQueryPlanGenerator

RUN_ID = UUID("00000000-0000-0000-0000-000000000008")
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


def _execution_inputs(
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
    candidate = QueryCandidateGenerator.generate(
        plan,
        requirement=requirement,
        validation=plan_validation,
    )[0]
    candidate_validation = QueryCandidateValidator.validate(
        candidate,
        plan=plan,
        requirement=requirement,
    )[0]
    rankings = QueryCandidateRanker.rank(
        [candidate],
        validations={candidate.query_candidate_id: candidate_validation},
        requirement=requirement,
    )
    return requirement, candidate, candidate_validation, rankings[0]


def test_q7_valid_candidate_with_ranking_is_ready() -> None:
    requirement, candidate, validation, ranking = _execution_inputs(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    decision = QueryExecutionAdapter.evaluate(
        candidate,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
    )
    requests = QueryExecutionAdapter.create_request(
        candidate,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
    )

    assert decision.eligibility is QueryExecutionEligibility.READY
    assert requests[0].candidate_id == candidate.query_candidate_id
    assert requests[0].ranking_score == ranking.score
    assert requests[0].query_text == candidate.query_text
    assert requests[0].gap_id == requirement.gap_id


def test_q4_claim_validation_candidate_is_ready() -> None:
    requirement, candidate, validation, ranking = _execution_inputs(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )

    decision = QueryExecutionAdapter.evaluate(
        candidate,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
    )

    assert decision.eligibility is QueryExecutionEligibility.READY


def test_closed_gap_is_blocked() -> None:
    requirement, candidate, validation, ranking = _execution_inputs(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )
    closed_requirement = replace(requirement, closure_status=GapClosureStatus.CLOSED)

    decision = QueryExecutionAdapter.evaluate(
        candidate,
        validation=validation,
        ranking=ranking,
        requirement=closed_requirement,
    )

    assert decision.eligibility is QueryExecutionEligibility.BLOCKED
    assert decision.reason == "gap_already_closed"
    assert (
        QueryExecutionAdapter.create_request(
            candidate,
            validation=validation,
            ranking=ranking,
            requirement=closed_requirement,
        )
        == ()
    )


def test_invalid_candidate_is_rejected() -> None:
    requirement, candidate, validation, ranking = _execution_inputs(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )
    invalid_candidate = replace(
        candidate,
        query_candidate_id=uuid5(NAMESPACE_URL, "invalid-query-candidate"),
        status=QueryCandidateStatus.REJECTED,
    )

    decision = QueryExecutionAdapter.evaluate(
        invalid_candidate,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
    )

    assert decision.eligibility is QueryExecutionEligibility.REJECTED
    assert decision.reason == "invalid_query_candidate"


def test_missing_ranking_is_blocked_without_provider_call() -> None:
    requirement, candidate, validation, _ = _execution_inputs(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )

    decision = QueryExecutionAdapter.evaluate(
        candidate,
        validation=validation,
        ranking=None,
        requirement=requirement,
    )

    assert decision.eligibility is QueryExecutionEligibility.BLOCKED
    assert decision.reason == "candidate_ranking_missing"
