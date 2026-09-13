import pytest
from app.domain.research_management import (
    ResearchFactCounts,
    allocate_budget_shares,
    calculate_information_gain,
)


def _facts(
    *,
    evidence: int,
    claims: int,
    sources: int,
    candidates: int,
    coverage: float,
) -> ResearchFactCounts:
    return ResearchFactCounts(
        accepted_evidence=evidence,
        unique_claims=claims,
        independent_sources=sources,
        evidence_candidates=candidates,
        coverage=coverage,
    )


def test_budget_shares_are_deterministic_and_sum_to_iteration_limit() -> None:
    allocation = allocate_budget_shares(
        max_iterations=20,
        max_searches=20,
        max_pages=30,
        max_tokens=100_000,
    )
    display = allocation["derived_display"]
    assert isinstance(display, dict)
    assert display["research_iterations"] + display["report_iterations"] + display[
        "validation_iterations"
    ] == 20
    assert display["writer_tokens"] == 10_000
    assert (
        allocation["planner_tokens"]
        + allocation["research_tokens"]
        + allocation["verification_tokens"]
        + allocation["writer_tokens_initial"]
        + allocation["safety_tokens"]
        == 100_000
    )
    assert "research_searches" not in allocation
    assert "report_pages" not in allocation
    limits = allocation["executable_resource_limits"]
    assert isinstance(limits, dict)
    assert limits == {
        "logical_queries": 20,
        "provider_requests": 20,
        "pages_fetched": 30,
        "pages_extracted": 30,
        "extraction_calls": 30,
        "verification_calls": 2,
        "scheduler_actions": 20,
        "productive_iterations": 20,
    }


def test_budget_shares_reject_negative_limits() -> None:
    with pytest.raises(ValueError):
        allocate_budget_shares(
            max_iterations=-1,
            max_searches=1,
            max_pages=1,
            max_tokens=1,
        )


def test_information_gain_rewards_new_independent_knowledge() -> None:
    result = calculate_information_gain(
        _facts(evidence=2, claims=2, sources=1, candidates=2, coverage=0.2),
        _facts(evidence=7, claims=5, sources=3, candidates=7, coverage=0.4),
    )

    assert result.score == 1.0
    assert result.new_evidence == 5
    assert result.new_claims == 3
    assert result.new_sources == 2
    assert result.coverage_delta == 0.2


def test_information_gain_is_zero_for_repeated_or_low_value_results() -> None:
    result = calculate_information_gain(
        _facts(evidence=5, claims=4, sources=3, candidates=6, coverage=0.7),
        _facts(evidence=5, claims=4, sources=3, candidates=10, coverage=0.7),
    )

    assert result.score == 0.0
    assert result.new_candidates == 4
    assert result.duplicate_or_low_value_ratio == 1.0


def test_information_gain_never_becomes_negative_when_nothing_changes() -> None:
    facts = _facts(evidence=0, claims=0, sources=0, candidates=0, coverage=0.0)

    result = calculate_information_gain(facts, facts)

    assert result.score == 0.0
    assert result.coverage_delta == 0.0
