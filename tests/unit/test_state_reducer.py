from uuid import uuid4

import pytest
from app.agent.reducers.state_reducer import StateVersionConflict, apply_patch
from app.domain.state import BudgetLimits, BudgetUsage, ResearchState, StatePatch
from pydantic import ValidationError


def test_patch_uses_compare_and_swap_version() -> None:
    state = ResearchState(run_id=uuid4(), state_version=2)
    patch = StatePatch(patch_id=uuid4(), base_version=1)

    with pytest.raises(StateVersionConflict):
        apply_patch(state, patch)


def test_patch_revalidates_budget_invariants() -> None:
    state = ResearchState(
        run_id=uuid4(),
        budget_limits=BudgetLimits(max_searches=1),
    )
    patch = StatePatch(
        patch_id=uuid4(),
        base_version=0,
        budget_usage=BudgetUsage(searches=2),
    )

    with pytest.raises(ValidationError, match="search budget exceeded"):
        apply_patch(state, patch)


def test_patch_increments_version_without_mutating_input() -> None:
    state = ResearchState(run_id=uuid4())
    patch = StatePatch(patch_id=uuid4(), base_version=0)

    updated = apply_patch(state, patch)

    assert state.state_version == 0
    assert updated.state_version == 1
