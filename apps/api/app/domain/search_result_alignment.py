"""Explain how executed search sources relate to a canonical GapRequirement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapRequirement, GapRequirementType
from app.domain.query_executor import (
    QueryExecutionResult,
    QueryExecutionResultStatus,
)


class SearchResultAlignmentStatus(StrEnum):
    ALIGNED = "aligned"
    PARTIAL = "partial"
    NOT_ALIGNED = "not_aligned"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SearchResultAlignment:
    """A read-only explanation linking one source to an execution Gap."""

    alignment_id: UUID
    execution_id: UUID
    query_candidate_id: UUID
    gap_id: UUID
    question_id: str
    source_id: str
    alignment_status: SearchResultAlignmentStatus
    alignment_reason: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("aligned search source must not be empty")
        if not self.alignment_reason.strip():
            raise ValueError("search alignment reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "alignment_id": str(self.alignment_id),
            "execution_id": str(self.execution_id),
            "query_candidate_id": str(self.query_candidate_id),
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "source_id": self.source_id,
            "alignment_status": self.alignment_status.value,
            "alignment_reason": self.alignment_reason,
        }


class SearchResultAlignmentGenerator:
    """Map normalized execution sources to a Gap without creating Evidence."""

    @staticmethod
    def generate(
        result: QueryExecutionResult,
        *,
        requirement: GapRequirement,
    ) -> tuple[SearchResultAlignment, ...]:
        if result.status in {
            QueryExecutionResultStatus.BLOCKED,
            QueryExecutionResultStatus.FAILED,
            QueryExecutionResultStatus.EMPTY,
        }:
            return ()
        candidate_id = _candidate_id_from_result(result)
        if result.events[0].run_id != requirement.run_id:
            raise ValueError("execution and requirement runs do not match")
        if result.events[0].question_id != requirement.question_id:
            raise ValueError("execution and requirement questions do not match")
        if result.events[0].gap_id != requirement.gap_id:
            raise ValueError("execution and requirement gaps do not match")

        return tuple(
            SearchResultAlignment(
                alignment_id=uuid5(
                    NAMESPACE_URL,
                    f"search-result-alignment:{result.execution_id}:{source}",
                ),
                execution_id=result.execution_id,
                query_candidate_id=candidate_id,
                gap_id=requirement.gap_id,
                question_id=requirement.question_id,
                source_id=source,
                alignment_status=status,
                alignment_reason=reason,
            )
            for source in result.candidate_sources
            for status, reason in (
                _evaluate_source(
                    result.query_text,
                    source,
                    requirement.requirement_type,
                ),
            )
        )


def _candidate_id_from_result(result: QueryExecutionResult) -> UUID:
    candidate_ids = {event.candidate_id for event in result.events}
    if len(candidate_ids) != 1:
        raise ValueError("execution result must identify exactly one query candidate")
    return next(iter(candidate_ids))


def _evaluate_source(
    query_text: str,
    source: str,
    requirement_type: GapRequirementType,
) -> tuple[SearchResultAlignmentStatus, str]:
    query = query_text.casefold()
    source_text = source.casefold()
    if requirement_type is GapRequirementType.INDEPENDENT_SOURCE:
        topic_match = any(
            term in query
            for term in ("market", "report", "cagr", "statistics", "independent")
        )
        source_is_suitable = any(
            term in source_text
            for term in (
                "industry",
                "report",
                "government",
                ".gov",
                "statistics",
                "research",
                "institution",
                "official",
            )
        )
        source_is_weak = any(
            term in source_text
            for term in ("blog", "forum", "aggregator", "vendor", "marketing")
        )
        if not topic_match:
            return (
                SearchResultAlignmentStatus.NOT_ALIGNED,
                "query topic does not target the independent-source requirement",
            )
        if source_is_weak and not source_is_suitable:
            return (
                SearchResultAlignmentStatus.PARTIAL,
                "source may be relevant but is not an independent high-quality source",
            )
        if source_is_suitable:
            return (
                SearchResultAlignmentStatus.ALIGNED,
                "query topic and source characteristics match the independent-source requirement",
            )
        return (
            SearchResultAlignmentStatus.PARTIAL,
            "query topic matches but source suitability is unknown",
        )

    if requirement_type is GapRequirementType.CLAIM_VERIFICATION:
        topic_match = any(
            term in query
            for term in (
                "claim",
                "verify",
                "verification",
                "official",
                "statement",
                "documentation",
            )
        )
        source_is_suitable = any(
            term in source_text
            for term in ("official", ".gov", "technical", "documentation", "research")
        )
        if not topic_match:
            return (
                SearchResultAlignmentStatus.NOT_ALIGNED,
                "query topic does not target Claim verification",
            )
        if source_is_suitable:
            return (
                SearchResultAlignmentStatus.ALIGNED,
                "query topic and official or technical source match Claim verification",
            )
        return (
            SearchResultAlignmentStatus.PARTIAL,
            "query targets Claim verification but source suitability is unknown",
        )

    return (
        SearchResultAlignmentStatus.UNKNOWN,
        "no search-result alignment rule exists for this Requirement type",
    )
