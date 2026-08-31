from app.domain.research_management import ResearchFactCounts, calculate_information_gain


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
