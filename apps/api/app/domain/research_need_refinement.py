"""Phase 14.0-E Research Need refinement from evidence failure reasons.

Upgrades the ``Gap -> Research Need`` step from "needs more evidence" to a
concrete information direction: third-party validation, primary reference,
quantitative metrics, method description, and so on.

The refinement is pure domain data.  It never executes search and never
changes the persisted ``ResearchNeed``/``SuggestedResearchAction`` chain: each
refined direction also carries the closest existing ``ResearchNeedType`` so
feedback dispatch keeps flowing through the unchanged Phase 13.2 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.domain.evidence_failure_classification import EvidenceFailureReason
from app.domain.research_need import ResearchNeedType


class RefinedNeedType(StrEnum):
    PRIMARY_REFERENCE = "primary_reference"
    INDEPENDENT_VALIDATION = "independent_validation"
    CLAIM_VERIFICATION = "claim_verification"
    QUANTITATIVE_EVIDENCE = "quantitative_evidence"
    METHOD_EVIDENCE = "method_evidence"
    DETAILED_EVIDENCE = "detailed_evidence"
    HIGH_QUALITY_SOURCE = "high_quality_source"
    CURRENT_SOURCE = "current_source"
    DIMENSION_TARGETED_EVIDENCE = "dimension_targeted_evidence"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NeedRefinement:
    """A concrete research direction derived from one evidence failure."""

    refined_need_type: RefinedNeedType
    persisted_need_type: ResearchNeedType
    query_hints: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "refined_need_type": self.refined_need_type.value,
            "persisted_need_type": self.persisted_need_type.value,
            "query_hints": list(self.query_hints),
        }


_REFINEMENTS: Final[dict[EvidenceFailureReason, NeedRefinement]] = {
    EvidenceFailureReason.NO_PRIMARY_SOURCE: NeedRefinement(
        refined_need_type=RefinedNeedType.PRIMARY_REFERENCE,
        persisted_need_type=ResearchNeedType.ADDITIONAL_SOURCE,
        query_hints=("official documentation", "primary publication", "original data"),
    ),
    EvidenceFailureReason.NO_INDEPENDENT_VALIDATION: NeedRefinement(
        refined_need_type=RefinedNeedType.INDEPENDENT_VALIDATION,
        persisted_need_type=ResearchNeedType.INDEPENDENT_SOURCE,
        query_hints=(
            "official documentation",
            "research paper",
            "third party analysis",
        ),
    ),
    EvidenceFailureReason.CLAIM_NOT_SUPPORTED: NeedRefinement(
        refined_need_type=RefinedNeedType.CLAIM_VERIFICATION,
        persisted_need_type=ResearchNeedType.CLAIM_VERIFICATION,
        query_hints=("benchmark", "evaluation", "paper", "experiment"),
    ),
    EvidenceFailureReason.INSUFFICIENT_DETAIL: NeedRefinement(
        refined_need_type=RefinedNeedType.DETAILED_EVIDENCE,
        persisted_need_type=ResearchNeedType.EVIDENCE_QUALITY,
        query_hints=("full text", "detailed analysis", "case study"),
    ),
    EvidenceFailureReason.NO_QUANTITATIVE_SUPPORT: NeedRefinement(
        refined_need_type=RefinedNeedType.QUANTITATIVE_EVIDENCE,
        persisted_need_type=ResearchNeedType.MISSING_DATA,
        query_hints=("performance", "accuracy", "benchmark", "measurement"),
    ),
    EvidenceFailureReason.NO_METHOD_DESCRIPTION: NeedRefinement(
        refined_need_type=RefinedNeedType.METHOD_EVIDENCE,
        persisted_need_type=ResearchNeedType.MISSING_DATA,
        query_hints=("method", "experimental setup", "procedure"),
    ),
    EvidenceFailureReason.SOURCE_QUALITY_LOW: NeedRefinement(
        refined_need_type=RefinedNeedType.HIGH_QUALITY_SOURCE,
        persisted_need_type=ResearchNeedType.EVIDENCE_QUALITY,
        query_hints=("peer-reviewed", "authoritative source", "official report"),
    ),
    EvidenceFailureReason.OUTDATED_SOURCE: NeedRefinement(
        refined_need_type=RefinedNeedType.CURRENT_SOURCE,
        persisted_need_type=ResearchNeedType.EVIDENCE_QUALITY,
        query_hints=("latest", "recent", "updated"),
    ),
    EvidenceFailureReason.DIMENSION_MISMATCH: NeedRefinement(
        refined_need_type=RefinedNeedType.DIMENSION_TARGETED_EVIDENCE,
        persisted_need_type=ResearchNeedType.DIMENSION_COVERAGE,
        query_hints=("topic specific study", "dimension relevant report"),
    ),
    EvidenceFailureReason.UNKNOWN: NeedRefinement(
        refined_need_type=RefinedNeedType.UNKNOWN,
        persisted_need_type=ResearchNeedType.UNKNOWN,
        query_hints=(),
    ),
}


def refine_research_need(reason: EvidenceFailureReason) -> NeedRefinement:
    """Map one evidence failure onto a concrete, generic need direction."""

    return _REFINEMENTS[reason]
