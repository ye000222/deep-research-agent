from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.domain.feedback_execution import (
    FeedbackExecutionContext,
    feedback_execution_allowed,
)
from app.domain.gap_closure import GapClosureStatus, GapRequirement, GapRequirementType
from app.domain.query_candidate import QueryCandidate, QueryCandidateStatus
from app.domain.query_candidate_ranking import QueryCandidateRanker
from app.domain.query_candidate_validation import QueryCandidateValidator
from app.domain.query_execution import QueryExecutionAdapter, QueryExecutionEligibility
from app.domain.research_query_plan import (
    ResearchQueryPlan,
    ResearchQueryPlanStatus,
    ResearchQueryPlanStrategy,
)


def _fixture() -> tuple[GapRequirement, ResearchQueryPlan, QueryCandidate]:
    now = datetime.now(UTC)
    run_id = uuid4()
    gap_id = uuid4()
    requirement = GapRequirement(
        gap_id=gap_id,
        run_id=run_id,
        question_id="q7",
        dimension_key="q7:d2",
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
        criterion="two independent sources",
        required_evidence_count=1,
        required_independent_sources=2,
        current_evidence_count=1,
        current_independent_sources=1,
        verification_status="not_evaluated",
        closure_status=GapClosureStatus.PARTIAL,
        created_at=now,
        updated_at=now,
        state_version=2,
    )
    plan = ResearchQueryPlan(
        query_plan_id=uuid4(),
        run_id=run_id,
        question_id="q7",
        gap_id=gap_id,
        dimension_key="q7:d2",
        requirement_type="independent_source",
        research_query_intent_id=uuid4(),
        objective="find another source",
        search_strategy_type=ResearchQueryPlanStrategy.INDEPENDENT_SOURCE_DISCOVERY,
        target_topics=("industry report",),
        preferred_source_types=("industry_report", "government"),
        exclusion_constraints=("exclude_existing_sources=true",),
        priority=2,
        status=ResearchQueryPlanStatus.SUGGESTED,
        created_at=now,
    )
    candidate = QueryCandidate(
        query_candidate_id=uuid4(),
        run_id=run_id,
        question_id="q7",
        gap_id=gap_id,
        dimension_key="q7:d2",
        requirement_type="independent_source",
        research_query_plan_id=plan.query_plan_id,
        query_text="q7:d2 industrial vision market report official statistics",
        query_purpose="find independent source",
        target_topics=("industry report",),
        preferred_source_types=("industry_report", "government"),
        constraints=("exclude_existing_sources=true",),
        generation_reason="feedback",
        status=QueryCandidateStatus.GENERATED,
        feedback_id=uuid4(),
        avoid_previous_failure_reason="insufficient_evidence",
    )
    return requirement, plan, candidate


def test_feedback_context_produces_isolated_execution_request() -> None:
    requirement, plan, candidate = _fixture()
    validation = QueryCandidateValidator.validate(
        candidate, plan=plan, requirement=requirement
    )[0]
    ranking = QueryCandidateRanker.rank(
        (candidate,),
        validations={candidate.query_candidate_id: validation},
        requirement=requirement,
    )[0]
    context = FeedbackExecutionContext.create(
        feedback_id=candidate.feedback_id,
        run_id=candidate.run_id,
        question_id=candidate.question_id,
        gap_id=candidate.gap_id,
        dimension_key=candidate.dimension_key,
        requirement_type=candidate.requirement_type,
        research_need_id=uuid4(),
        query_candidate_id=candidate.query_candidate_id,
    )
    request = QueryExecutionAdapter.create_request(
        candidate,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
        feedback_context=context,
    )[0]

    assert request.feedback_id == candidate.feedback_id
    assert request.feedback_execution_id == context.feedback_execution_id
    assert request.trigger_reason == "insufficient_evidence"
    assert QueryExecutionAdapter.evaluate(
        candidate,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
    ).eligibility is QueryExecutionEligibility.READY


def test_feedback_execution_limit_stops_repeated_handoffs() -> None:
    assert feedback_execution_allowed(execution_count=0, limit=2)
    assert feedback_execution_allowed(execution_count=1, limit=2)
    assert not feedback_execution_allowed(execution_count=2, limit=2)


def test_normal_candidate_has_no_feedback_execution_context() -> None:
    requirement, plan, candidate = _fixture()
    normal = replace(
        candidate,
        feedback_id=None,
        avoid_previous_failure_reason=None,
    )
    validation = QueryCandidateValidator.validate(
        normal, plan=plan, requirement=requirement
    )[0]
    ranking = QueryCandidateRanker.rank(
        (normal,),
        validations={normal.query_candidate_id: validation},
        requirement=requirement,
    )[0]
    request = QueryExecutionAdapter.create_request(
        normal,
        validation=validation,
        ranking=ranking,
        requirement=requirement,
    )[0]
    assert request.feedback_id is None
    assert request.feedback_execution_id is None
