from app.domain.provider_failure_classification import (
    ProviderFailureType,
    RetryDecision,
    classify_provider_failure,
    retry_decision,
)


def test_failure_types_are_preserved() -> None:
    assert classify_provider_failure(
        error_code="SEARCH_PROVIDER_DEGRADED", details={"failure_type": "timeout"}
    ) is ProviderFailureType.TIMEOUT
    assert classify_provider_failure(error_code="HTTP_429") is ProviderFailureType.RATE_LIMIT
    assert classify_provider_failure(
        error_code="SEARCH_PROVIDER_DEGRADED", details={"failure_type": "server_error"}
    ) is ProviderFailureType.SERVER_ERROR
    assert classify_provider_failure(
        error_code="SEARCH_PROVIDER_DEGRADED", details={"failure_type": "empty_response"}
    ) is ProviderFailureType.EMPTY_RESPONSE


def test_retry_decisions_are_bounded_and_class_specific() -> None:
    assert retry_decision(ProviderFailureType.TIMEOUT) is RetryDecision.RETRY
    assert retry_decision(ProviderFailureType.CIRCUIT_OPEN) is RetryDecision.FALLBACK
    assert retry_decision(ProviderFailureType.EMPTY_RESPONSE) is RetryDecision.FALLBACK
    assert retry_decision(ProviderFailureType.UNKNOWN) is RetryDecision.STOP
