import pytest
from app.domain.state import StopReason
from app.infrastructure.db.state_runtime import _stop_reason


@pytest.mark.parametrize(
    "reason",
    [
        "research_budget_exhausted",
        "page_budget_exhausted",
        "search_budget_exhausted",
        "token_budget_exhausted",
        "iteration_budget_exhausted",
        "RESEARCH_BUDGET_EXHAUSTED",
    ],
)
def test_precise_budget_stop_reasons_project_to_budget_exhausted(reason: str) -> None:
    assert _stop_reason(reason) is StopReason.BUDGET_EXHAUSTED
