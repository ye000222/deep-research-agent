from __future__ import annotations

from uuid import UUID

import pytest
from app.domain.recovery_trace_collector import (
    RecoveryTraceCollector,
    RecoveryTraceError,
)

RUN_ID = "00000000-0000-0000-0000-000000000061"
ATTEMPT_ID = "00000000-0000-0000-0000-000000000062"
GAP_ID = "00000000-0000-0000-0000-000000000063"
QUERY_ID = "00000000-0000-0000-0000-000000000064"
READER_ID = "00000000-0000-0000-0000-000000000065"
EXTRACTION_ID = "00000000-0000-0000-0000-000000000066"
ALIGNMENT_ID = "00000000-0000-0000-0000-000000000067"
EVALUATION_ID = "00000000-0000-0000-0000-000000000068"


def _base_event(event: str) -> dict[str, object]:
    return {
        "event": event,
        "run_id": RUN_ID,
        "question_id": "q5",
        "gap_id": GAP_ID,
        "recovery_attempt_id": ATTEMPT_ID,
    }


def _successful_events() -> list[dict[str, object]]:
    started = _base_event("recovery.started")
    started.update({"coverage_before": 0.0, "gap_before": ["q5:d1"]})
    query = _base_event("query.execution.completed")
    query["execution_id"] = QUERY_ID
    reader = _base_event("reader.execution.completed")
    reader["reader_execution_id"] = READER_ID
    extraction = _base_event("evidence.extraction.completed")
    extraction["extraction_id"] = EXTRACTION_ID
    alignment = _base_event("evidence.alignment.completed")
    alignment["alignment_id"] = ALIGNMENT_ID
    closure = _base_event("gap.closure.transition")
    closure.update(
        {
            "evaluation_id": EVALUATION_ID,
            "after_status": "closed",
            "transition_reason": "requirement_satisfied",
        }
    )
    completed = _base_event("recovery.completed")
    completed.update(
        {
            "coverage_before": 0.0,
            "coverage_after": 1.0,
            "gap_before": ["q5:d1"],
            "gap_after": [],
            "outcome_type": "successful",
        }
    )
    return [started, query, reader, extraction, alignment, closure, completed]


def test_successful_recovery_forms_a_complete_closure_trace() -> None:
    trace = RecoveryTraceCollector.collect(_successful_events())

    assert trace.recovery_attempt_id == UUID(ATTEMPT_ID)
    assert trace.query_execution_ids == (UUID(QUERY_ID),)
    assert trace.reader_execution_ids == (UUID(READER_ID),)
    assert trace.extraction_ids == (UUID(EXTRACTION_ID),)
    assert trace.alignment_ids == (UUID(ALIGNMENT_ID),)
    assert trace.closure_evaluation_ids == (UUID(EVALUATION_ID),)
    assert trace.before_gap_state == ("q5:d1",)
    assert trace.after_gap_state == ()
    assert trace.outcome_type == "successful"
    assert trace.closure_transition_reason == "requirement_satisfied"


def test_partial_recovery_preserves_open_gap_and_partial_outcome() -> None:
    events = _successful_events()
    for event in events:
        event["question_id"] = "q7"
    events[-2]["after_status"] = "partial"
    events[-2]["transition_reason"] = "missing_independent_source"
    events[-1].update(
        {
            "coverage_after": 0.7895,
            "gap_before": ["q7:d2"],
            "gap_after": ["q7:d2"],
            "outcome_type": "partial",
        }
    )
    events[0]["coverage_before"] = 0.25
    events[0]["gap_before"] = ["q7:d2"]

    trace = RecoveryTraceCollector.collect(events)

    assert trace.question_id == "q7"
    assert trace.after_gap_state == ("q7:d2",)
    assert trace.before_coverage == 0.25
    assert trace.after_coverage == 0.7895
    assert trace.outcome_type == "partial"
    assert trace.closure_transition_reason == "missing_independent_source"


def test_failed_recovery_stops_at_query_failure_and_keeps_gap_open() -> None:
    started = _base_event("recovery.started")
    started.update({"coverage_before": 0.0, "gap_before": ["q7:d2"]})
    query = _base_event("query.execution.failed")
    query.update({"execution_id": QUERY_ID, "error_reason": "timeout"})
    failed = _base_event("recovery.failed")
    failed.update(
        {
            "coverage_before": 0.0,
            "gap_before": ["q7:d2"],
            "reason": "timeout",
            "outcome_type": "failed",
        }
    )

    trace = RecoveryTraceCollector.collect([started, query, failed])

    assert trace.query_execution_ids == (UUID(QUERY_ID),)
    assert trace.reader_execution_ids == ()
    assert trace.after_gap_state == ("q7:d2",)
    assert trace.after_coverage == 0.0
    assert trace.outcome_type == "failed"


def test_normal_research_event_cannot_be_attached_to_recovery_trace() -> None:
    events = _successful_events()
    del events[2]["recovery_attempt_id"]

    with pytest.raises(RecoveryTraceError, match="another attempt"):
        RecoveryTraceCollector.collect(events)


def test_mixed_question_events_are_rejected() -> None:
    events = _successful_events()
    events[1]["question_id"] = "q7"

    with pytest.raises(RecoveryTraceError, match="question_id"):
        RecoveryTraceCollector.collect(events)


def test_evidence_gain_without_alignment_is_not_a_successful_trace() -> None:
    events = _successful_events()
    del events[4]
    events[-1]["accepted_evidence_before"] = 0
    events[-1]["accepted_evidence_after"] = 1

    with pytest.raises(RecoveryTraceError, match=r"evidence\.alignment\.completed"):
        RecoveryTraceCollector.collect(events)
