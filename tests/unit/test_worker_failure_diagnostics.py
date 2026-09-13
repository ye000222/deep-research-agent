from uuid import uuid4

from app.domain.state import BudgetLimits, BudgetUsage, ResearchState, RunStatus, StopReason
from app.infrastructure.db.state_runtime import _stop_reason
from app.worker.tasks import _unexpected_failure_codes
from pydantic import ValidationError


def test_model_budget_validation_error_is_not_reported_as_unknown() -> None:
    try:
        ResearchState(
            run_id=uuid4(),
            budget_limits=BudgetLimits(max_model_tokens=100),
            budget_usage=BudgetUsage(model_tokens=101),
        )
    except ValidationError as exc:
        error_code, detail_code = _unexpected_failure_codes(exc)
    else:  # pragma: no cover - the state invariant is the test setup
        raise AssertionError("expected model budget validation error")

    assert error_code == "RESEARCH_BUDGET_EXHAUSTED"
    assert detail_code == "MODEL_TOKEN_BUDGET_OVERRUN"


def test_unexpected_failure_uses_sanitized_exception_type() -> None:
    error_code, detail_code = _unexpected_failure_codes(RuntimeError("secret text"))

    assert error_code == "WORKER_EXECUTION_FAILED"
    assert detail_code == "RUNTIMEERROR"
    assert "secret" not in detail_code.casefold()


def test_non_budget_validation_error_has_a_specific_safe_diagnostic() -> None:
    try:
        BudgetLimits(max_searches=-1)
    except ValidationError as exc:
        error_code, detail_code = _unexpected_failure_codes(exc)
    else:  # pragma: no cover - the validation error is the test setup
        raise AssertionError("expected budget field validation error")

    assert error_code == "STATE_VALIDATION_FAILED"
    assert detail_code == "MAX_SEARCHES_GREATER_THAN_EQUAL"
    assert "-1" not in detail_code


def test_page_budget_validation_error_is_not_reported_as_unknown() -> None:
    try:
        ResearchState(
            run_id=uuid4(),
            budget_limits=BudgetLimits(max_pages=1),
            budget_usage=BudgetUsage(pages=2),
        )
    except ValidationError as exc:
        error_code, detail_code = _unexpected_failure_codes(exc)
    else:  # pragma: no cover - the state invariant is the test setup
        raise AssertionError("expected page budget validation error")

    assert error_code == "RESEARCH_BUDGET_EXHAUSTED"
    assert detail_code == "PAGE_BUDGET_OVERRUN"


def test_terminal_budget_overrun_can_be_projected_without_hiding_usage() -> None:
    state = ResearchState(
        run_id=uuid4(),
        status=RunStatus.FAILED,
        stop_reason=StopReason.BUDGET_EXHAUSTED,
        budget_limits=BudgetLimits(max_pages=1),
        budget_usage=BudgetUsage(pages=2),
    )
    assert state.budget_usage.pages == 2


def test_durable_model_retry_is_not_projected_as_a_terminal_failure() -> None:
    assert _stop_reason("model_transport_retry_pending") is None


def test_durable_search_retry_is_not_projected_as_a_terminal_failure() -> None:
    assert _stop_reason("search_transport_retry_pending") is None
