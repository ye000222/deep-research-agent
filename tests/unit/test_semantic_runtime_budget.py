from __future__ import annotations

import pytest
from app.domain.semantic_runtime_budget import (
    SemanticRuntimeBudget,
    SemanticRuntimeBudgetState,
)
from pydantic import ValidationError


def _budget(**changes: int) -> SemanticRuntimeBudget:
    values = {
        "max_ap_decomposition_calls": 0,
        "max_source_semantic_extraction_calls": 0,
        "max_identity_resolution_calls": 0,
        "max_evidence_inputs": 0,
        "max_propositions_per_evidence": 0,
        "max_source_assertions_per_evidence": 0,
        "max_candidate_pairs": 0,
        "max_direction_assessments": 0,
    }
    values.update(changes)
    return SemanticRuntimeBudget(**values)


def test_budget_requires_explicit_limits_and_is_immutable() -> None:
    with pytest.raises(ValidationError):
        SemanticRuntimeBudget()
    budget = _budget(max_candidate_pairs=4)
    with pytest.raises(ValidationError):
        budget.max_candidate_pairs = 5  # type: ignore[misc]


def test_budget_separates_provider_and_deterministic_work() -> None:
    budget = _budget(
        max_ap_decomposition_calls=3,
        max_candidate_pairs=7,
        max_direction_assessments=7,
    )
    state = SemanticRuntimeBudgetState()
    state, allowed = state.consume(budget, "provider_ap_decomposition_calls")
    assert allowed
    state, allowed = state.consume(budget, "candidate_pairs", 2)
    assert allowed
    assert state.provider_ap_decomposition_calls == 1
    assert state.candidate_pairs == 2
    assert state.direction_assessments == 0
    assert state.can_consume_direction(budget)


def test_zero_limits_are_disabled_and_cannot_be_consumed() -> None:
    assert not SemanticRuntimeBudgetState().can_consume_direction(_budget())
    with pytest.raises(ValidationError):
        _budget(max_candidate_pairs=-1)


def test_budget_counter_names_are_closed() -> None:
    with pytest.raises(ValueError, match="invalid_semantic_runtime_budget_counter"):
        SemanticRuntimeBudgetState().consume(_budget(), "not_a_counter")


def test_provider_and_direction_consumption_stop_at_explicit_limits() -> None:
    budget = _budget(
        max_source_semantic_extraction_calls=1,
        max_candidate_pairs=1,
        max_direction_assessments=1,
    )
    state, allowed = SemanticRuntimeBudgetState().consume(
        budget, "provider_source_semantic_extraction_calls"
    )
    assert allowed
    unchanged, allowed = state.consume(budget, "provider_source_semantic_extraction_calls")
    assert not allowed
    assert unchanged == state
    state, allowed = state.consume_direction_pair(budget)
    assert allowed
    unchanged, allowed = state.consume_direction_pair(budget)
    assert not allowed
    assert unchanged == state
