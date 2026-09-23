from uuid import uuid4

import pytest
from app.domain.feedback_execution import FeedbackExecutionContext
from app.domain.feedback_query_executor_adapter import FeedbackQueryExecutionContext
from app.domain.query_execution import QueryExecutionRequest


def test_feedback_adapter_preserves_existing_execution_context() -> None:
    feedback_id = uuid4()
    context = FeedbackExecutionContext.create(
        feedback_id=feedback_id,
        run_id=uuid4(),
        question_id="q7",
        gap_id=uuid4(),
        dimension_key="q7:d2",
        requirement_type="independent_source",
        research_need_id=uuid4(),
        query_candidate_id=uuid4(),
    )
    request = QueryExecutionRequest(
        execution_id=uuid4(),
        run_id=context.run_id,
        question_id=context.question_id,
        gap_id=context.gap_id,
        dimension_key=context.dimension_key,
        candidate_id=context.query_candidate_id,
        ranking_score=0.9,
        query_text="find independent source",
        source_constraints=("industry_report",),
        requirement_type=context.requirement_type,
        execution_reason="aligned",
        feedback_id=feedback_id,
        feedback_execution_id=context.feedback_execution_id,
        trigger_reason="missing_independent_source",
    )
    adapted = FeedbackQueryExecutionContext.from_request(context, request)
    assert adapted.feedback_id == feedback_id
    assert adapted.query_execution_id == request.execution_id
    assert adapted.trigger_reason == "missing_independent_source"


def test_feedback_adapter_rejects_cross_context_request() -> None:
    context = FeedbackExecutionContext.create(
        feedback_id=uuid4(),
        run_id=uuid4(),
        question_id="q7",
        gap_id=uuid4(),
        dimension_key="q7:d2",
        requirement_type="independent_source",
        research_need_id=uuid4(),
        query_candidate_id=uuid4(),
    )
    request = QueryExecutionRequest(
        execution_id=uuid4(),
        run_id=context.run_id,
        question_id=context.question_id,
        gap_id=context.gap_id,
        dimension_key=context.dimension_key,
        candidate_id=context.query_candidate_id,
        ranking_score=0.9,
        query_text="find independent source",
        source_constraints=("industry_report",),
        requirement_type=context.requirement_type,
        execution_reason="aligned",
        feedback_id=uuid4(),
        feedback_execution_id=context.feedback_execution_id,
    )
    with pytest.raises(ValueError, match="feedback request"):
        FeedbackQueryExecutionContext.from_request(context, request)
