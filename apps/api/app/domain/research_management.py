"""Deterministic metrics for coverage-driven research management."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResearchFactCounts(BaseModel):
    """Small fact snapshot captured before and after one research iteration."""

    model_config = ConfigDict(extra="forbid")

    accepted_evidence: int = Field(ge=0)
    unique_claims: int = Field(ge=0)
    independent_sources: int = Field(ge=0)
    evidence_candidates: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)


class InformationGainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    new_evidence: int = Field(ge=0)
    new_claims: int = Field(ge=0)
    new_sources: int = Field(ge=0)
    new_candidates: int = Field(ge=0)
    coverage_delta: float = Field(ge=0.0, le=1.0)
    duplicate_or_low_value_ratio: float = Field(ge=0.0, le=1.0)


def calculate_information_gain(
    previous: ResearchFactCounts,
    current: ResearchFactCounts,
) -> InformationGainResult:
    """Calculate a bounded, replayable marginal information-gain score.

    Normalizers express the expected useful output of one bounded V1 iteration:
    five accepted evidence items, three distinct claims, two independent sources,
    or a 20 percentage-point coverage increase each saturate their component.
    """

    new_evidence = max(current.accepted_evidence - previous.accepted_evidence, 0)
    new_claims = max(current.unique_claims - previous.unique_claims, 0)
    new_sources = max(current.independent_sources - previous.independent_sources, 0)
    new_candidates = max(current.evidence_candidates - previous.evidence_candidates, 0)
    coverage_delta = max(current.coverage - previous.coverage, 0.0)
    low_value_items = max(new_candidates - new_evidence, 0)
    low_value_ratio = min(low_value_items / new_candidates, 1.0) if new_candidates else 0.0

    score = (
        0.35 * min(new_evidence / 5.0, 1.0)
        + 0.25 * min(new_claims / 3.0, 1.0)
        + 0.20 * min(new_sources / 2.0, 1.0)
        + 0.20 * min(coverage_delta / 0.20, 1.0)
        - 0.20 * low_value_ratio
    )
    return InformationGainResult(
        score=round(min(max(score, 0.0), 1.0), 4),
        new_evidence=new_evidence,
        new_claims=new_claims,
        new_sources=new_sources,
        new_candidates=new_candidates,
        coverage_delta=round(coverage_delta, 4),
        duplicate_or_low_value_ratio=round(low_value_ratio, 4),
    )
