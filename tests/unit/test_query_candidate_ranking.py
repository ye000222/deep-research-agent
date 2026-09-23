from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, project_gap_requirements
from app.domain.query_candidate import QueryCandidateGenerator
from app.domain.query_candidate_ranking import QueryCandidateRanker
from app.domain.query_candidate_validation import QueryCandidateValidator
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_action import ResearchActionGenerator
from app.domain.research_need import ResearchNeedGenerator
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.research_query_plan import ResearchQueryPlanGenerator

RUN_ID = UUID("00000000-0000-0000-0000-000000000007")
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


def _candidate_set(
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
    return requirement, plan, candidate, candidate_validation


def test_q7_independent_source_candidate_ranks_above_introduction() -> None:
    requirement, plan, candidate, validation = _candidate_set(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )
    weaker = replace(
        candidate,
        query_candidate_id=uuid5(NAMESPACE_URL, "q7-weaker-candidate"),
        query_text="AI vision introduction",
        target_topics=("AI vision introduction",),
    )
    weaker_validation = QueryCandidateValidator.validate(
        weaker,
        plan=plan,
        requirement=requirement,
    )[0]
    rankings = QueryCandidateRanker.rank(
        [candidate, weaker],
        validations={
            candidate.query_candidate_id: validation,
            weaker.query_candidate_id: weaker_validation,
        },
        requirement=requirement,
    )

    assert rankings[0].candidate_id == candidate.query_candidate_id
    assert rankings[0].rank == 1
    assert rankings[0].score > rankings[1].score
    assert any("independent_source" in reason for reason in rankings[0].reasons)


def test_q4_claim_verification_ranks_above_market_statistics() -> None:
    requirement, plan, candidate, validation = _candidate_set(
        "q4",
        "q4:d1",
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )
    weaker = replace(
        candidate,
        query_candidate_id=uuid5(NAMESPACE_URL, "q4-weaker-candidate"),
        query_text="collect market statistics",
        target_topics=("market statistics",),
        preferred_source_types=("dataset",),
    )
    weaker_validation = QueryCandidateValidator.validate(
        weaker,
        plan=plan,
        requirement=requirement,
    )[0]
    rankings = QueryCandidateRanker.rank(
        [candidate, weaker],
        validations={
            candidate.query_candidate_id: validation,
            weaker.query_candidate_id: weaker_validation,
        },
        requirement=requirement,
    )

    assert rankings[0].candidate_id == candidate.query_candidate_id
    assert rankings[0].score > rankings[1].score


def test_closed_gap_does_not_generate_ranking() -> None:
    requirement, _, candidate, validation = _candidate_set(
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
        QueryCandidateRanker.rank(
            [candidate],
            validations={candidate.query_candidate_id: validation},
            requirement=closed_requirement,
        )
        == ()
    )


def test_ranking_is_observation_only() -> None:
    requirement, plan, candidate, validation = _candidate_set(
        "q7",
        "q7:d2",
        sources=1,
        required_sources=2,
    )
    original_text = candidate.query_text

    rankings = QueryCandidateRanker.rank(
        [candidate],
        validations={candidate.query_candidate_id: validation},
        requirement=requirement,
    )

    assert rankings[0].score_details.candidate_id == candidate.query_candidate_id
    assert candidate.query_text == original_text
    assert plan.status.value == "suggested"
    assert requirement.closure_status.value == "partial"
