from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.domain.evidence_routing import (
    EvidenceRoutingDecision,
    EvidenceRoutingStatus,
)
from app.domain.gap_closure import GapClosureStatus, project_gap_requirements
from app.domain.reader_request import (
    ReaderDispatchEventType,
    ReaderDispatchState,
    ReaderRequestAdapter,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000024")
ALIGNMENT_ID = UUID("00000000-0000-0000-0000-000000000025")
ROUTING_ID = UUID("00000000-0000-0000-0000-000000000026")


def _requirement(question_id: str, dimension_key: str):
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
                        "coverage": 0.5,
                        "accepted_evidence": 2,
                        "independent_sources": 1,
                        "required_sources": 2,
                    }
                ],
            }
        ],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )[0]


def _decision(
    requirement,
    status: EvidenceRoutingStatus,
    source_id: str = "https://industry-report.example/report",
) -> EvidenceRoutingDecision:
    return EvidenceRoutingDecision(
        routing_id=ROUTING_ID,
        alignment_id=ALIGNMENT_ID,
        source_id=source_id,
        gap_id=requirement.gap_id,
        question_id=requirement.question_id,
        decision=status,
        reason=(
            "source quality unknown"
            if status is EvidenceRoutingStatus.DEFER
            else "test routing"
        ),
    )


def test_route_creates_ready_reader_request() -> None:
    requirement = _requirement("q7", "q7:d2")

    request = ReaderRequestAdapter.create(
        _decision(requirement, EvidenceRoutingStatus.ROUTE),
        requirement=requirement,
    )[0]

    assert request.state is ReaderDispatchState.READY
    assert request.alignment_id == ALIGNMENT_ID
    assert request.routing_id == ROUTING_ID
    assert request.events[0].event_type is ReaderDispatchEventType.READY


def test_defer_creates_blocked_reader_request() -> None:
    requirement = _requirement("q1", "q1:d2")

    request = ReaderRequestAdapter.create(
        _decision(requirement, EvidenceRoutingStatus.DEFER),
        requirement=requirement,
    )[0]

    assert request.state is ReaderDispatchState.BLOCKED
    assert request.execution_reason == "source quality unknown"
    assert request.events[0].event_type is ReaderDispatchEventType.BLOCKED


def test_skip_creates_skipped_reader_request() -> None:
    requirement = _requirement("q7", "q7:d2")

    request = ReaderRequestAdapter.create(
        _decision(requirement, EvidenceRoutingStatus.SKIP),
        requirement=requirement,
    )[0]

    assert request.state is ReaderDispatchState.SKIPPED
    assert request.events[0].event_type is ReaderDispatchEventType.SKIPPED


def test_closed_gap_creates_no_reader_request() -> None:
    requirement = _requirement("q7", "q7:d2")
    closed_requirement = replace(
        requirement,
        closure_status=GapClosureStatus.CLOSED,
    )

    assert (
        ReaderRequestAdapter.create(
            _decision(requirement, EvidenceRoutingStatus.ROUTE),
            requirement=closed_requirement,
        )
        == ()
    )


def test_question_and_gap_associations_are_preserved() -> None:
    requirement = _requirement("q1", "q1:d2")
    request = ReaderRequestAdapter.create(
        _decision(requirement, EvidenceRoutingStatus.ROUTE),
        requirement=requirement,
    )[0]

    assert request.question_id == "q1"
    assert request.gap_id == requirement.gap_id
    assert request.events[0].alignment_id == ALIGNMENT_ID
