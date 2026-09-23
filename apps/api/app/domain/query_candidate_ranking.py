"""Rank query candidates without executing or scheduling them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import prod
from uuid import UUID

from app.domain.gap_closure import GapClosureStatus, GapRequirement, GapRequirementType
from app.domain.query_candidate import QueryCandidate
from app.domain.query_candidate_validation import (
    QueryCandidateAlignmentType,
    QueryCandidateValidation,
    QueryCandidateValidationStatus,
)


class QueryCandidateRankingStatus(StrEnum):
    RANKED = "ranked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryCandidateScore:
    """Explainable component scores for one query candidate."""

    candidate_id: UUID
    alignment_score: float
    requirement_match_score: float
    source_potential_score: float
    evidence_gain_score: float
    cost_score: float
    total_score: float
    ranking_reason: tuple[str, ...]

    def __post_init__(self) -> None:
        scores = (
            self.alignment_score,
            self.requirement_match_score,
            self.source_potential_score,
            self.evidence_gain_score,
            self.cost_score,
            self.total_score,
        )
        if any(score < 0 or score > 1 for score in scores):
            raise ValueError("query candidate scores must be between 0 and 1")
        if not self.ranking_reason:
            raise ValueError("query candidate ranking reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": str(self.candidate_id),
            "alignment_score": self.alignment_score,
            "requirement_match_score": self.requirement_match_score,
            "source_potential_score": self.source_potential_score,
            "evidence_gain_score": self.evidence_gain_score,
            "cost_score": self.cost_score,
            "total_score": self.total_score,
            "ranking_reason": list(self.ranking_reason),
        }


@dataclass(frozen=True, slots=True)
class QueryCandidateRanking:
    """A deterministic rank for a candidate, with its score breakdown."""

    candidate_id: UUID
    rank: int
    score: float
    reasons: tuple[str, ...]
    score_details: QueryCandidateScore
    status: QueryCandidateRankingStatus = QueryCandidateRankingStatus.RANKED

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": str(self.candidate_id),
            "rank": self.rank,
            "score": self.score,
            "reasons": list(self.reasons),
            "score_details": self.score_details.as_dict(),
            "status": self.status.value,
        }


class QueryCandidateRanker:
    """Create an explainable ranking without changing candidate inputs."""

    @staticmethod
    def rank(
        candidates: tuple[QueryCandidate, ...] | list[QueryCandidate],
        *,
        validations: dict[UUID, QueryCandidateValidation],
        requirement: GapRequirement,
    ) -> tuple[QueryCandidateRanking, ...]:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()

        scored = [
            _score_candidate(
                candidate,
                validation=validations.get(candidate.query_candidate_id),
                requirement=requirement,
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item.total_score, str(item.candidate_id)))
        return tuple(
            QueryCandidateRanking(
                candidate_id=item.candidate_id,
                rank=index,
                score=item.total_score,
                reasons=item.ranking_reason,
                score_details=item,
            )
            for index, item in enumerate(scored, start=1)
        )


def _score_candidate(
    candidate: QueryCandidate,
    *,
    validation: QueryCandidateValidation | None,
    requirement: GapRequirement,
) -> QueryCandidateScore:
    alignment = _alignment_score(validation)
    requirement_match = _requirement_match_score(candidate, requirement)
    source_potential = _source_potential_score(candidate)
    evidence_gain = _evidence_gain_score(candidate, requirement)
    cost = _cost_score(candidate)
    total = round(
        prod((alignment, requirement_match, source_potential, evidence_gain, cost)) ** 0.2,
        4,
    )
    reasons = _ranking_reasons(
        candidate,
        validation=validation,
        requirement=requirement,
        alignment=alignment,
        requirement_match=requirement_match,
        source_potential=source_potential,
        evidence_gain=evidence_gain,
        cost=cost,
    )
    return QueryCandidateScore(
        candidate_id=candidate.query_candidate_id,
        alignment_score=alignment,
        requirement_match_score=requirement_match,
        source_potential_score=source_potential,
        evidence_gain_score=evidence_gain,
        cost_score=cost,
        total_score=total,
        ranking_reason=reasons,
    )


def _alignment_score(validation: QueryCandidateValidation | None) -> float:
    if validation is None:
        return 0.0
    if validation.validation_status is QueryCandidateValidationStatus.VALID:
        if validation.alignment_type is QueryCandidateAlignmentType.GAP_ALIGNED:
            return 1.0
        if validation.alignment_type is QueryCandidateAlignmentType.PLAN_ALIGNED:
            return 0.9
        return 0.8
    if validation.validation_status is QueryCandidateValidationStatus.PARTIAL:
        return 0.5
    if validation.validation_status is QueryCandidateValidationStatus.UNKNOWN:
        return 0.25
    return 0.0


def _requirement_match_score(
    candidate: QueryCandidate,
    requirement: GapRequirement,
) -> float:
    text = candidate.query_text.casefold()
    terms = {
        GapRequirementType.INDEPENDENT_SOURCE: (
            "industry",
            "report",
            "statistics",
            "third party",
            "analysis",
        ),
        GapRequirementType.CLAIM_VERIFICATION: (
            "claim",
            "verification",
            "official",
            "documentation",
            "technical",
        ),
        GapRequirementType.EVIDENCE_QUALITY: (
            "primary",
            "official",
            "authoritative",
        ),
        GapRequirementType.DIMENSION_COVERAGE: (candidate.dimension_key.casefold(),),
    }.get(requirement.requirement_type, ("data", "dataset", "statistics"))
    matches = sum(term in text for term in terms)
    return min(1.0, 0.25 + matches / max(1, len(terms)))


def _source_potential_score(candidate: QueryCandidate) -> float:
    source_types = set(candidate.preferred_source_types)
    high_value = {
        "government",
        "official",
        "official_documentation",
        "research_institution",
        "industry_report",
        "independent_analysis",
        "primary_source",
        "authoritative_report",
    }
    low_value = {"blog", "forum", "aggregator", "marketing_page"}
    high_matches = len(source_types.intersection(high_value))
    low_matches = len(source_types.intersection(low_value))
    if low_matches and not high_matches:
        return 0.15
    if high_matches >= 2 and not low_matches:
        return 1.0
    if high_matches:
        return 0.75
    return 0.35


def _evidence_gain_score(
    candidate: QueryCandidate,
    requirement: GapRequirement,
) -> float:
    text = candidate.query_text.casefold()
    target_terms = " ".join(candidate.target_topics).casefold()
    combined = f"{text} {target_terms}"
    terms = {
        GapRequirementType.INDEPENDENT_SOURCE: ("report", "statistics", "analysis"),
        GapRequirementType.CLAIM_VERIFICATION: ("claim", "verification", "official"),
        GapRequirementType.EVIDENCE_QUALITY: ("primary", "authoritative", "official"),
        GapRequirementType.DIMENSION_COVERAGE: (candidate.dimension_key.casefold(),),
    }.get(requirement.requirement_type, ("data", "dataset", "statistics"))
    matches = sum(term in combined for term in terms)
    return min(1.0, 0.25 + matches / max(1, len(terms)))


def _cost_score(candidate: QueryCandidate) -> float:
    text = candidate.query_text.casefold()
    word_count = len(text.split())
    score = 1.0
    if word_count > 10:
        score -= 0.15
    if "pdf" in text or "download" in text:
        score -= 0.2
    if "report" in text or "technical" in text:
        score -= 0.1
    if "official" in text or "documentation" in text:
        score += 0.05
    return max(0.1, min(1.0, score))


def _ranking_reasons(
    candidate: QueryCandidate,
    *,
    validation: QueryCandidateValidation | None,
    requirement: GapRequirement,
    alignment: float,
    requirement_match: float,
    source_potential: float,
    evidence_gain: float,
    cost: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if validation is None:
        reasons.append("missing candidate validation")
    elif validation.validation_status is QueryCandidateValidationStatus.VALID:
        reasons.append("validated against the Gap and Query Plan")
    elif validation.validation_status is QueryCandidateValidationStatus.INVALID:
        reasons.append("candidate validation failed")
    if alignment >= 0.9 and requirement_match >= 0.7:
        reasons.append(f"matches {requirement.requirement_type.value} requirement")
    elif requirement_match < 0.5:
        reasons.append("weak requirement match")
    if source_potential >= 0.9:
        reasons.append("high-quality or independent source potential")
    elif source_potential < 0.4:
        reasons.append("limited source potential")
    if evidence_gain >= 0.8:
        reasons.append(f"high evidence gain potential for {candidate.dimension_key}")
    elif evidence_gain < 0.5:
        reasons.append("low evidence gain potential")
    if cost >= 0.85:
        reasons.append("relatively low acquisition cost")
    elif cost < 0.6:
        reasons.append("higher acquisition cost")
    return tuple(reasons)
