from __future__ import annotations

from uuid import UUID

from app.domain.evidence_routing import (
    EvidenceRoutingRouter,
    EvidenceRoutingStatus,
)
from app.domain.search_result_alignment import (
    SearchResultAlignment,
    SearchResultAlignmentStatus,
)

EXECUTION_ID = UUID("00000000-0000-0000-0000-000000000020")
ALIGNMENT_ID = UUID("00000000-0000-0000-0000-000000000023")
CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000021")
GAP_ID = UUID("00000000-0000-0000-0000-000000000022")


def _alignment(
    status: SearchResultAlignmentStatus,
    source_id: str,
) -> SearchResultAlignment:
    return SearchResultAlignment(
        alignment_id=ALIGNMENT_ID,
        execution_id=EXECUTION_ID,
        query_candidate_id=CANDIDATE_ID,
        gap_id=GAP_ID,
        question_id="q7",
        source_id=source_id,
        alignment_status=status,
        alignment_reason="test alignment",
    )


def test_aligned_industry_report_routes_to_reader() -> None:
    decision = EvidenceRoutingRouter.route(
        _alignment(
            SearchResultAlignmentStatus.ALIGNED,
            "https://industry-report.example/report",
        )
    )

    assert decision.decision is EvidenceRoutingStatus.ROUTE
    assert decision.source_id.endswith("/report")


def test_partial_vendor_blog_is_deferred() -> None:
    decision = EvidenceRoutingRouter.route(
        _alignment(
            SearchResultAlignmentStatus.PARTIAL,
            "https://vendor.example/blog",
        )
    )

    assert decision.decision is EvidenceRoutingStatus.DEFER


def test_partial_high_potential_source_routes() -> None:
    decision = EvidenceRoutingRouter.route(
        _alignment(SearchResultAlignmentStatus.PARTIAL, "https://unknown.example/source"),
        evidence_potential=0.9,
    )

    assert decision.decision is EvidenceRoutingStatus.ROUTE


def test_topic_mismatch_is_skipped() -> None:
    decision = EvidenceRoutingRouter.route(
        _alignment(
            SearchResultAlignmentStatus.NOT_ALIGNED,
            "https://research.example/tutorial",
        )
    )

    assert decision.decision is EvidenceRoutingStatus.SKIP


def test_empty_alignment_set_produces_no_routing() -> None:
    assert EvidenceRoutingRouter.route_many(()) == ()
