from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import project_gap_requirements
from app.domain.query_execution import QueryExecutionRequest
from app.domain.query_executor import (
    QueryExecutionResult,
    QueryExecutionResultStatus,
    QueryExecutor,
    SearchResultSet,
)
from app.domain.search_result_alignment import (
    SearchResultAlignmentGenerator,
    SearchResultAlignmentStatus,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000013")
GAP_ID = UUID("00000000-0000-0000-0000-000000000014")
CANDIDATE_ID = UUID("00000000-0000-0000-0000-000000000015")


class FakeProvider:
    provider_name = "fake-provider"

    def __init__(self, sources: tuple[str, ...]):
        self.sources = sources

    def search(self, query_text: str, constraints: tuple[str, ...]) -> SearchResultSet:
        return SearchResultSet(self.provider_name, self.sources)


def _requirement(question_id: str, dimension_key: str, *, claim: bool = False):
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
        claim_states=(
            {"claim": {"dimension_key": dimension_key, "unresolved": True}}
            if claim
            else None
        ),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )[0]


def _result(
    query_text: str,
    sources: tuple[str, ...],
    *,
    question_id: str = "q7",
    gap_id: UUID = GAP_ID,
    dimension_key: str = "q7:d2",
) -> QueryExecutionResult:
    request = QueryExecutionRequest(
        execution_id=UUID("00000000-0000-0000-0000-000000000016"),
        run_id=RUN_ID,
        question_id=question_id,
        gap_id=gap_id,
        dimension_key=dimension_key,
        candidate_id=CANDIDATE_ID,
        ranking_score=0.9,
        query_text=query_text,
        source_constraints=(),
        requirement_type="independent_source",
        execution_reason="validated candidate",
    )
    return QueryExecutor.execute(
        request,
        provider_adapter=FakeProvider(sources),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_q7_industry_report_is_aligned() -> None:
    requirement = _requirement("q7", "q7:d2")
    result = _result(
        "industrial vision market CAGR report",
        ("https://industry-report.example/industrial-vision-market-cagr",),
        gap_id=requirement.gap_id,
    )

    alignments = SearchResultAlignmentGenerator.generate(
        result,
        requirement=requirement,
    )

    assert alignments[0].alignment_status is SearchResultAlignmentStatus.ALIGNED
    assert alignments[0].alignment_id
    assert alignments[0].query_candidate_id == CANDIDATE_ID
    assert alignments[0].gap_id == requirement.gap_id


def test_q1_vendor_blog_is_partial() -> None:
    requirement = _requirement("q1", "q1:d2")
    result = _result(
        "industrial vision market report independent source",
        ("https://vendor.example/blog/vision-overview",),
        question_id="q1",
        gap_id=requirement.gap_id,
        dimension_key="q1:d2",
    )

    alignments = SearchResultAlignmentGenerator.generate(
        result,
        requirement=requirement,
    )

    assert alignments[0].alignment_status is SearchResultAlignmentStatus.PARTIAL


def test_q4_official_statement_is_aligned() -> None:
    requirement = _requirement("q4", "q4:d1", claim=True)
    request = QueryExecutionRequest(
        execution_id=UUID("00000000-0000-0000-0000-000000000017"),
        run_id=RUN_ID,
        question_id="q4",
        gap_id=requirement.gap_id,
        dimension_key="q4:d1",
        candidate_id=CANDIDATE_ID,
        ranking_score=0.9,
        query_text="verify company claim official statement",
        source_constraints=(),
        requirement_type="claim_verification",
        execution_reason="validated candidate",
    )
    result = QueryExecutor.execute(
        request,
        provider_adapter=FakeProvider(("https://gov.example/official-statement",)),
    )

    alignments = SearchResultAlignmentGenerator.generate(
        result,
        requirement=requirement,
    )

    assert alignments[0].alignment_status is SearchResultAlignmentStatus.ALIGNED


def test_topic_mismatch_is_not_aligned() -> None:
    requirement = _requirement("q7", "q7:d2")
    result = _result(
        "computer vision tutorial",
        ("https://research.example/tutorial",),
        gap_id=requirement.gap_id,
    )

    alignments = SearchResultAlignmentGenerator.generate(
        result,
        requirement=requirement,
    )

    assert alignments[0].alignment_status is SearchResultAlignmentStatus.NOT_ALIGNED


def test_failed_execution_produces_no_alignment_or_evidence() -> None:
    requirement = _requirement("q7", "q7:d2")
    failed = QueryExecutionResult(
        execution_id=UUID("00000000-0000-0000-0000-000000000018"),
        request_id=UUID("00000000-0000-0000-0000-000000000019"),
        status=QueryExecutionResultStatus.FAILED,
        provider="fake-provider",
        query_text="industrial vision market report",
        result_count=0,
        candidate_sources=(),
        error_reason="timeout",
    )

    assert SearchResultAlignmentGenerator.generate(failed, requirement=requirement) == ()
