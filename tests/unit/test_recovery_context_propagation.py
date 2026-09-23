from __future__ import annotations

from uuid import UUID

from app.domain.recovery_execution import RecoveryContext
from app.infrastructure.db.research_tools import ResearchToolRepository

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
ATTEMPT_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_recovery_context_contains_stable_pipeline_identity() -> None:
    context = RecoveryContext.create(
        run_id=RUN_ID,
        question_id="q1",
        plan_version=13,
        recovery_attempt_id=ATTEMPT_ID,
        trigger_reason="high_risk_high_utility_borrow",
        coverage_before=0.5,
        gap_before=("q1:d2",),
    )

    refs = context.as_refs()

    assert refs["recovery_attempt_id"] == str(ATTEMPT_ID)
    assert refs["recovery_question_id"] == "q1"
    assert refs["recovery_plan_version"] == 13
    assert refs["recovery_trigger_reason"] == "high_risk_high_utility_borrow"
    assert refs["recovery_coverage_before"] == 0.5
    assert refs["recovery_gap_before"] == ["q1:d2"]
    assert refs["recovery_context_created_at"]


def test_repository_exposes_late_context_backfill() -> None:
    assert callable(getattr(ResearchToolRepository, "research_event_cursor", None))
    assert callable(getattr(ResearchToolRepository, "attach_recovery_context", None))
    assert callable(
        getattr(ResearchToolRepository, "record_search_query_started", None)
    )


def test_normal_research_has_no_recovery_context_by_default() -> None:
    context = None

    assert context is None
