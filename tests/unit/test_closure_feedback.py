from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.closure_evaluation_context import ClosureEvaluationContext
from app.domain.closure_evaluator import ClosureEvaluator
from app.domain.closure_feedback import (
    ClosureFeedbackGenerator,
    ClosureFeedbackReason,
)
from app.domain.evidence_alignment_executor import EvidenceAlignmentExecutor
from app.domain.evidence_alignment_request import EvidenceAlignmentRequest
from app.domain.gap_closure import (
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.query_candidate import QueryCandidateGenerator
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_action import (
    ResearchActionGenerator,
    SuggestedResearchActionType,
)
from app.domain.research_need import ResearchNeedGenerator, ResearchNeedType
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.research_query_plan import ResearchQueryPlanGenerator

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _requirement(*, required_sources: int = 2) -> GapRequirement:
    return GapRequirement(
        gap_id=uuid4(),
        run_id=RUN_ID,
        question_id="q7",
        dimension_key="q7:d2",
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
        criterion="需要独立来源",
        required_evidence_count=1,
        required_independent_sources=required_sources,
        current_evidence_count=0,
        current_independent_sources=0,
        verification_status=VerificationStatus.REQUIRED,
        closure_status=GapClosureStatus.OPEN,
        created_at=NOW,
        updated_at=NOW,
        state_version=1,
    )


def _partial_closure(requirement: GapRequirement):
    request = EvidenceAlignmentRequest(
        alignment_request_id=uuid4(),
        extraction_id=uuid4(),
        reader_execution_id=uuid4(),
        source_id=str(uuid4()),
        evidence_id=uuid4(),
        run_id=RUN_ID,
        question_id="q7",
        gap_id=requirement.gap_id,
        dimension_key="q7:d2",
        requirement_type="independent_source",
        claim_id=None,
        candidate_evidence="industrial vision market report",
    )
    alignment = EvidenceAlignmentExecutor.execute(
        request,
        requirement=requirement,
        evidence_dimension_key="q7:d2",
        evidence_quality_passed=True,
        independent_source_count=1,
    ).alignment
    assert alignment is not None
    context = ClosureEvaluationContext.from_alignments(
        gap_id=requirement.gap_id,
        before_snapshot=ClosureSnapshot(),
        evidence_alignments=(alignment,),
        independent_source_count=1,
    )
    return ClosureEvaluator.evaluate(context, requirement=requirement)


def test_partial_closure_generates_independent_source_feedback() -> None:
    requirement = _requirement()
    closure = _partial_closure(requirement)

    feedback = ClosureFeedbackGenerator.generate(closure, now=NOW)

    assert len(feedback) == 1
    assert feedback[0].failure_reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE
    assert feedback[0].recommended_need_type is ResearchNeedType.INDEPENDENT_SOURCE
    assert feedback[0].missing_requirement == "independent_source"


def test_closed_gap_generates_no_feedback() -> None:
    requirement = _requirement(required_sources=1)
    closed = _partial_closure(requirement)
    assert closed.requirement.closure_status is GapClosureStatus.CLOSED
    assert ClosureFeedbackGenerator.generate(closed) == ()


def test_feedback_flows_to_need_action_and_query_candidate() -> None:
    requirement = _requirement()
    closure = _partial_closure(requirement)
    feedback = ClosureFeedbackGenerator.generate(closure, now=NOW)[0]

    need = ResearchNeedGenerator.generate(
        closure.requirement,
        feedback=feedback,
        now=NOW,
    )[0]
    action = ResearchActionGenerator.generate(
        need,
        requirement=closure.requirement,
        feedback=feedback,
        now=NOW,
    )[0]
    intent = ResearchQueryIntentGenerator.generate(
        action,
        requirement=closure.requirement,
        now=NOW,
    )[0]
    plan = ResearchQueryPlanGenerator.generate(
        intent,
        requirement=closure.requirement,
        now=NOW,
    )[0]
    validation = QueryPlanValidator.validate(plan, requirement=closure.requirement)[0]
    candidate = QueryCandidateGenerator.generate(
        plan,
        requirement=closure.requirement,
        validation=validation,
        feedback=feedback,
    )[0]

    assert need.feedback_id == feedback.feedback_id
    assert action.source_feedback_id == feedback.feedback_id
    assert action.action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE
    assert candidate.feedback_id == feedback.feedback_id
    assert candidate.avoid_previous_failure_reason == "missing_independent_source"
