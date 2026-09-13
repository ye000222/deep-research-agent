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


def allocate_budget_shares(
    *,
    max_iterations: int,
    max_searches: int,
    max_pages: int,
    max_tokens: int,
    max_logical_queries: int | None = None,
    max_provider_requests: int | None = None,
    max_pages_fetched: int | None = None,
    max_pages_extracted: int | None = None,
    max_extraction_calls: int | None = None,
    max_verification_calls: int | None = None,
    max_scheduler_actions: int | None = None,
) -> dict[str, object]:
    """Freeze executable V2 pool limits in the run snapshot."""

    explicit_limits = (
        max_iterations,
        max_searches,
        max_pages,
        max_tokens,
        *(
            value
            for value in (
                max_logical_queries,
                max_provider_requests,
                max_pages_fetched,
                max_pages_extracted,
                max_extraction_calls,
                max_verification_calls,
                max_scheduler_actions,
            )
            if value is not None
        ),
    )
    if min(explicit_limits) < 0:
        raise ValueError("budget values must be non-negative")
    report_iterations = max(1, (max_iterations * 15 + 99) // 100)
    validation_iterations = max(1, (max_iterations * 10 + 99) // 100)
    reserved_iterations = min(max_iterations, report_iterations + validation_iterations)
    # The Planner reserves 1.5k input + up to 4k output before its provider
    # request.  Small tiers therefore need an absolute floor in addition to the
    # percentage target, otherwise a correctly enforced planner pool would make
    # every quick run impossible to start.
    planner_tokens = min(max_tokens, max(5_500, int(max_tokens * 0.05))) if max_tokens else 0
    remaining_tokens = max(0, max_tokens - planner_tokens)
    writer_tokens_initial = (
        min(remaining_tokens, 15_000, max(3_000, int(max_tokens * 0.15)))
        if max_tokens
        else 0
    )
    # ``writer_tokens`` was exposed by the V1 dashboard as a 10k-compatible
    # field. Keep that read-only compatibility alias while the executable
    # reserve uses the dynamic V2 value below.
    writer_tokens = min(10_000, writer_tokens_initial)
    remaining_tokens -= writer_tokens_initial
    verification_tokens = min(remaining_tokens, int(max_tokens * 0.15))
    remaining_tokens -= verification_tokens
    safety_tokens = min(remaining_tokens, int(max_tokens * 0.10))
    research_tokens = max(0, remaining_tokens - safety_tokens)
    executable_limits = {
        "logical_queries": max(
            0,
            max_searches if max_logical_queries is None else max_logical_queries,
        ),
        "provider_requests": max(
            0,
            max_searches if max_provider_requests is None else max_provider_requests,
        ),
        "pages_fetched": max(0, max_pages if max_pages_fetched is None else max_pages_fetched),
        "pages_extracted": max(
            0,
            max_pages if max_pages_extracted is None else max_pages_extracted,
        ),
        "extraction_calls": max(
            0,
            max_pages if max_extraction_calls is None else max_extraction_calls,
        ),
        "verification_calls": max(
            0,
            validation_iterations
            if max_verification_calls is None
            else max_verification_calls,
        ),
        "scheduler_actions": max(
            0,
            max_iterations if max_scheduler_actions is None else max_scheduler_actions,
        ),
        "productive_iterations": max_iterations,
    }
    return {
        "planner_tokens": planner_tokens,
        "research_tokens": research_tokens,
        "verification_tokens": verification_tokens,
        "writer_tokens_initial": writer_tokens_initial,
        "safety_tokens": safety_tokens,
        "token_pool_version": 3,
        "executable_resource_limits": executable_limits,
        # These values are retained only as an explicitly non-executable V1
        # dashboard projection. They are not advertised as pools and no
        # scheduler decision reads them.
        "derived_display": {
            "research_iterations": max_iterations - reserved_iterations,
            "report_iterations": report_iterations,
            "validation_iterations": validation_iterations,
            "research_searches": max(0, (max_searches * 85) // 100),
            "report_pages": max(0, (max_pages * 10) // 100),
            "writer_tokens": writer_tokens,
        },
    }


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
