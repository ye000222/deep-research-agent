"""Regression coverage for Candidate identity and lifecycle events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest
from app.domain.source_policy import normalize_source_url
from app.infrastructure.db.research_tools import (
    CandidateIdentity,
    candidate_lifecycle_event,
)

Event = dict[str, Any]

TERMINAL_STAGES = {
    "readable",
    "triage_rejected",
    "extraction_failed",
    "candidate_skipped",
    "fetch_failed",
}

SKIP_REASONS = {
    "duplicate",
    "low_quality",
    "budget_limit",
    "domain_restricted",
    "already_processed",
}


def _identity(candidate_id: str, raw_url: str, *, query_id: str = "query-1") -> CandidateIdentity:
    normalized_url = normalize_source_url(raw_url)
    return CandidateIdentity(
        candidate_id=candidate_id,
        raw_url=raw_url,
        normalized_url=normalized_url,
        question_id="question-1",
        query_id=query_id,
        domain="example.com",
    )


def _event(
    identity: CandidateIdentity,
    stage: str,
    *,
    reason: str | None = None,
    failure_reason: str | None = None,
) -> Event:
    return candidate_lifecycle_event(
        identity,
        stage,
        reason=reason,
        failure_reason=failure_reason,
    )


def _refs(event: Event) -> dict[str, Any]:
    refs = event.get("refs")
    assert isinstance(refs, dict)
    return refs


def _events_for_candidate(events: Sequence[Event], candidate_id: str) -> list[Event]:
    return [event for event in events if _refs(event).get("candidate_id") == candidate_id]


def _stages(events: Sequence[Event]) -> list[str]:
    return [str(_refs(event)["stage"]) for event in events]


def _assert_complete_candidate_trace(
    events: Sequence[Event],
    candidate_id: str,
) -> list[str]:
    candidate_events = _events_for_candidate(events, candidate_id)
    assert candidate_events
    stages = _stages(candidate_events)
    assert stages[0] == "candidate_created"
    assert stages[-1] in TERMINAL_STAGES
    for event in candidate_events:
        refs = _refs(event)
        assert refs["candidate_id"] == candidate_id
        assert refs["timestamp"]
        assert refs["raw_url"]
        assert refs["normalized_url"]
        assert refs["question_id"]
        assert refs["query_id"]
        assert refs["domain"]
    return stages


def test_duplicate_candidates_have_terminal_skip_and_only_one_fetch() -> None:
    """Duplicate normalized URLs are skipped before the duplicate is fetched."""

    raw_url_a = "https://example.com/report?utm_source=search"
    raw_url_b = "https://example.com/report#overview"
    assert normalize_source_url(raw_url_a) == normalize_source_url(raw_url_b)
    first = _identity("candidate-1", raw_url_a)
    duplicate = _identity("candidate-2", raw_url_b)

    events = [
        _event(first, "candidate_created"),
        _event(first, "url_normalized"),
        _event(first, "duplicate_checked"),
        _event(first, "fetch_started"),
        _event(first, "fetch_success"),
        _event(first, "extraction_started"),
        _event(first, "readable"),
        _event(duplicate, "candidate_created"),
        _event(duplicate, "url_normalized"),
        _event(duplicate, "duplicate_checked"),
        _event(duplicate, "candidate_skipped", reason="duplicate"),
    ]

    first_stages = _assert_complete_candidate_trace(events, first.candidate_id)
    duplicate_stages = _assert_complete_candidate_trace(events, duplicate.candidate_id)

    assert duplicate_stages[-1] == "candidate_skipped"
    assert _refs(_events_for_candidate(events, duplicate.candidate_id)[-1])["reason"] == (
        "duplicate"
    )
    assert first_stages.count("fetch_started") == 1
    assert duplicate_stages.count("fetch_started") == 0


def test_fetch_failure_has_candidate_linked_terminal_event() -> None:
    """A failed fetch closes the Candidate lifecycle with its failure reason."""

    identity = _identity("candidate-failure", "https://example.com/failing")
    events = [
        _event(identity, "candidate_created"),
        _event(identity, "url_normalized"),
        _event(identity, "duplicate_checked"),
        _event(identity, "fetch_started"),
        _event(identity, "fetch_failed", failure_reason="WEBPAGE_REQUEST_REJECTED"),
    ]

    stages = _assert_complete_candidate_trace(events, identity.candidate_id)
    failure_refs = _refs(events[-1])
    assert stages[-1] == "fetch_failed"
    assert failure_refs["failure_reason"] == "WEBPAGE_REQUEST_REJECTED"
    assert failure_refs["raw_url"] == "https://example.com/failing"
    assert failure_refs["normalized_url"] == normalize_source_url(identity.raw_url)


def test_fetch_success_cannot_leave_candidate_without_a_follow_up_state() -> None:
    """A successful fetch must continue to extraction or an explicit terminal state."""

    identity = _identity("candidate-fetched", "https://example.com/fetched")
    events = [
        _event(identity, "candidate_created"),
        _event(identity, "url_normalized"),
        _event(identity, "duplicate_checked"),
        _event(identity, "fetch_started"),
        _event(identity, "fetch_success"),
        _event(identity, "extraction_started"),
        _event(identity, "readable"),
    ]

    stages = _assert_complete_candidate_trace(events, identity.candidate_id)
    assert stages[-2:] == ["extraction_started", "readable"]


@pytest.mark.parametrize("reason", sorted(SKIP_REASONS))
def test_candidate_skipped_is_a_formal_terminal_state(reason: str) -> None:
    """All supported skip reasons are represented by a terminal event."""

    identity = _identity(f"candidate-{reason}", f"https://example.com/{reason}")
    events = [
        _event(identity, "candidate_created"),
        _event(identity, "url_normalized"),
        _event(identity, "duplicate_checked"),
        _event(identity, "candidate_skipped", reason=reason),
    ]

    stages = _assert_complete_candidate_trace(events, identity.candidate_id)
    assert stages[-1] == "candidate_skipped"
    assert _refs(events[-1])["reason"] == reason


def test_candidate_id_links_search_fetch_triage_extraction_and_evidence() -> None:
    """Every pipeline stage is joined by candidate_id, not only by URL."""

    identity = _identity(str(uuid4()), "https://example.com/trace")
    events = [
        _event(identity, "candidate_created"),
        _event(identity, "url_normalized"),
        _event(identity, "duplicate_checked"),
        _event(identity, "fetch_started"),
        _event(identity, "fetch_success"),
        _event(identity, "extraction_started"),
        _event(identity, "readable"),
    ]

    stages = _assert_complete_candidate_trace(events, identity.candidate_id)
    assert stages == [
        "candidate_created",
        "url_normalized",
        "duplicate_checked",
        "fetch_started",
        "fetch_success",
        "extraction_started",
        "readable",
    ]
    assert {_refs(event)["candidate_id"] for event in events} == {identity.candidate_id}
