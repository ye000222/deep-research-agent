from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.infrastructure.db.research_tools import (
    CandidateIdentity,
    evidence_selection_lifecycle_event,
)


@pytest.fixture
def candidate() -> CandidateIdentity:
    return CandidateIdentity(
        candidate_id="candidate-1",
        raw_url="https://example.com/article?utm_source=test",
        normalized_url="https://example.com/article",
        question_id="q1",
        query_id="query-1",
        domain="example.com",
    )


def test_selection_started_and_selected_preserve_candidate_identity(
    candidate: CandidateIdentity,
) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)

    started = evidence_selection_lifecycle_event(
        candidate,
        "selection_started",
        timestamp=timestamp,
    )
    selected = evidence_selection_lifecycle_event(
        candidate,
        "selection_selected",
        source_id=uuid4(),
        timestamp=timestamp,
    )

    assert started["event_type"] == "evidence.selection_started"
    assert selected["event_type"] == "evidence.selection_selected"
    assert started["refs"]["candidate_id"] == candidate.candidate_id
    assert selected["refs"]["normalized_url"] == candidate.normalized_url
    assert selected["refs"]["stage"] == "selection_selected"
    assert selected["refs"]["source_id"]


@pytest.mark.parametrize(
    "reason",
    (
        "already_processed",
        "duplicate_source",
        "extraction_budget",
        "question_budget",
        "token_budget",
        "source_diversity",
        "low_evidence_value",
        "question_mismatch",
        "page_limit",
    ),
)
def test_selection_skipped_accepts_required_reasons(
    candidate: CandidateIdentity,
    reason: str,
) -> None:
    event = evidence_selection_lifecycle_event(
        candidate,
        "selection_skipped",
        reason=reason,
    )

    assert event["event_type"] == "evidence.selection_skipped"
    assert event["refs"]["reason"] == reason


def test_selection_skipped_rejects_unknown_reason(candidate: CandidateIdentity) -> None:
    with pytest.raises(ValueError, match="unsupported evidence selection skip reason"):
        evidence_selection_lifecycle_event(
            candidate,
            "selection_skipped",
            reason="unclassified",
        )
