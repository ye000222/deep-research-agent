"""Provider failure taxonomy and bounded continuation decisions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class ProviderFailureType(StrEnum):
    TIMEOUT = "timeout"
    UNRESPONSIVE = "unresponsive"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    EMPTY_RESPONSE = "empty_response"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class RetryDecision(StrEnum):
    RETRY = "retry"
    FALLBACK = "fallback"
    STOP = "stop"


def classify_provider_failure(
    *, error_code: str, details: Mapping[str, object] | None = None
) -> ProviderFailureType:
    payload = details or {}
    failure_type = payload.get("failure_type")
    if isinstance(failure_type, str):
        if failure_type == "fallback_failure":
            metrics = payload.get("metrics")
            if (
                isinstance(metrics, Mapping)
                and int(metrics.get("empty_response_count", 0) or 0) > 0
            ):
                return ProviderFailureType.EMPTY_RESPONSE
            return ProviderFailureType.CIRCUIT_OPEN
        try:
            return ProviderFailureType(failure_type)
        except ValueError:
            pass
    normalized = error_code.casefold()
    if "timeout" in normalized:
        return ProviderFailureType.TIMEOUT
    if "rate" in normalized or "429" in normalized:
        return ProviderFailureType.RATE_LIMIT
    if "network" in normalized or "connect" in normalized:
        return ProviderFailureType.NETWORK_ERROR
    if "empty" in normalized:
        return ProviderFailureType.EMPTY_RESPONSE
    if "circuit" in normalized:
        return ProviderFailureType.CIRCUIT_OPEN
    if "server" in normalized or "http" in normalized or "5" in normalized:
        return ProviderFailureType.SERVER_ERROR
    return ProviderFailureType.UNKNOWN


def retry_decision(failure_type: ProviderFailureType) -> RetryDecision:
    if failure_type in {
        ProviderFailureType.TIMEOUT,
        ProviderFailureType.UNRESPONSIVE,
        ProviderFailureType.RATE_LIMIT,
        ProviderFailureType.NETWORK_ERROR,
        ProviderFailureType.SERVER_ERROR,
    }:
        return RetryDecision.RETRY
    if failure_type in {
        ProviderFailureType.CIRCUIT_OPEN,
        ProviderFailureType.EMPTY_RESPONSE,
    }:
        return RetryDecision.FALLBACK
    return RetryDecision.STOP
