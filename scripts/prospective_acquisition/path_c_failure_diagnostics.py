"""PATH-C-only safe observations; never retain requests, secrets or exception text.

This does not enable dispatch or define a new attempt identity. Observations are
written by the generation state machine only after its durable reservation.
"""

from __future__ import annotations

import json
import re
import socket
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

import httpx
from app.llm.adapters import LLMGateway, ModelGatewayError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from scripts.prospective_acquisition.control import AcquisitionBlocked

# These are output allowlists, not classifiers. Classification uses native
# exceptions, Gateway contract codes and HTTP status, never message phrases.
GATEWAY_CODES = frozenset(
    {
        "MODEL_TIMEOUT",
        "MODEL_NETWORK_ERROR",
        "MODEL_AUTHENTICATION_FAILED",
        "MODEL_RATE_LIMITED",
        "MODEL_PROVIDER_UNAVAILABLE",
        "MODEL_REQUEST_INVALID",
        "MODEL_RESPONSE_INVALID",
        "MODEL_OUTPUT_INVALID",
        "MODEL_OUTPUT_TRUNCATED",
        "PROVIDER_REDIRECT_BLOCKED",
        "UNSUPPORTED_ADAPTER",
    }
)
DETAIL_CODES = frozenset(
    {
        "CONNECT_TIMEOUT",
        "READ_TIMEOUT",
        "WRITE_TIMEOUT",
        "POOL_TIMEOUT",
        "CONNECT_ERROR",
        "READ_ERROR",
        "WRITE_ERROR",
        "REMOTE_PROTOCOL_ERROR",
        "LOCAL_PROTOCOL_ERROR",
        "PROVIDER_LENGTH_STOP",
        "PARSE_EMPTY_RESPONSE",
        "PARSE_TRUNCATED_OBJECT",
        "PARSE_TRAILING_COMMA",
        "PARSE_NON_OBJECT_JSON",
        "STRUCTURED_OUTPUT_FAILURE",
        "DECODING_ERROR",
        "STREAM_CONSUMED",
        "STREAM_CLOSED",
        "JSON_DECODE_ERROR",
    }
)
PROVIDER_CODES = frozenset(
    {
        "invalid_api_key",
        "authentication_error",
        "invalid_request_error",
        "invalid_request",
        "rate_limit_error",
        "rate_limit_exceeded",
        "insufficient_quota",
        "context_length_exceeded",
        "model_not_found",
        "server_error",
        "invalid_parameter",
        "unsupported_parameter",
        "permission_error",
        "permission_denied",
    }
)
EXCEPTION_CODES = frozenset(
    {
        "GATEWAY_ERROR",
        "CONNECT_TIMEOUT",
        "READ_TIMEOUT",
        "WRITE_TIMEOUT",
        "POOL_TIMEOUT",
        "TIMEOUT",
        "CONNECT_ERROR",
        "NETWORK_ERROR",
        "DNS_ERROR",
        "TLS_ERROR",
        "CONNECTION_RESET",
        "JSON_DECODE_ERROR",
        "DECODING_ERROR",
        "STREAM_CONSUMED",
        "STREAM_CLOSED",
        "VALIDATION_ERROR",
        "BUDGET_GUARD",
        "OS_ERROR",
        "UNKNOWN_EXCEPTION",
    }
)


def _safe_detail(value: str | None) -> str | None:
    if value in DETAIL_CODES:
        return value
    # Accept only the Gateway's generated code grammar with enumerated finish
    # and parse symbols. Arbitrary Provider finish strings are not safe codes.
    if isinstance(value, str) and re.fullmatch(
        r"OUTPUT_INVALID_(JSON_MODE|PROMPT_JSON|NATIVE_JSON_SCHEMA)"
        r"_FINISH_(STOP|LENGTH|MAX_TOKENS|MAX_OUTPUT_TOKENS|UNKNOWN)_"
        r"(PROVIDER_LENGTH_STOP|PARSE_EMPTY_RESPONSE|PARSE_TRUNCATED_OBJECT|"
        r"PARSE_TRAILING_COMMA|PARSE_NON_OBJECT_JSON|PARSE_EXPECTING_VALUE_AT_[0-9]+)"
        r"_CHARS_[0-9]+",
        value,
    ):
        return value
    return None


class FailureCategory(StrEnum):
    AUTHENTICATION = "AUTHENTICATION"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    NETWORK = "NETWORK"
    REQUEST_INVALID = "REQUEST_INVALID"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RESPONSE_INVALID = "RESPONSE_INVALID"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    PROVIDER_BINDING_CHANGED = "PROVIDER_BINDING_CHANGED"
    CREDENTIAL_RESOLUTION = "CREDENTIAL_RESOLUTION"
    REQUEST_CONSTRUCTION = "REQUEST_CONSTRUCTION"
    OTHER = "OTHER"


class FailurePhase(StrEnum):
    PREFLIGHT = "PREFLIGHT"
    PROVIDER_RESOLUTION = "PROVIDER_RESOLUTION"
    CREDENTIAL_RESOLUTION = "CREDENTIAL_RESOLUTION"
    REQUEST_CONSTRUCTION = "REQUEST_CONSTRUCTION"
    GATEWAY = "GATEWAY"
    OUTPUT_VALIDATION = "OUTPUT_VALIDATION"


class PathCFailureDiagnostics(BaseModel):
    """Typed allowlist: no free-form exception/header/request/body fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    diagnostic_version: Literal["path-c-safe-failure-diagnostics.v1"] = (
        "path-c-safe-failure-diagnostics.v1"
    )
    failure_class: Literal[
        "MODEL_GATEWAY_ERROR",
        "TRANSPORT_ERROR",
        "GENERATION_CONTRACT_ERROR",
        "LOCAL_RUNTIME_ERROR",
    ]
    failure_category: FailureCategory
    failure_phase: FailurePhase
    gateway_error_code: str | None = None
    gateway_detail_code: str | None = None
    gateway_detail_redacted: bool = False
    provider_http_status: int | None = Field(default=None, ge=100, le=599)
    safe_provider_error_code: str | None = None
    safe_provider_error_type: str | None = None
    provider_error_fields_redacted: bool = False
    safe_exception_chain: tuple[str, ...] = ()
    reservation_persisted_at_utc: datetime | None = None
    gateway_entered_at_utc: datetime | None = None
    outbound_attempted_at_utc: datetime | None = None
    outbound_completed_at_utc: datetime | None = None
    failure_recorded_at_utc: datetime
    response_received: bool | None = None
    response_body_present: bool | None = None
    raw_response_preserved: bool = False
    response_body_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response_body_size: int | None = Field(default=None, ge=0)
    raw_capture_policy: Literal[
        "NO_RESPONSE",
        "EXACT_SUCCESS_RESPONSE",
        "ERROR_RESPONSE_HASH_ONLY",
        "SENSITIVE_RESPONSE_HASH_ONLY",
        "CAPTURE_FAILED",
    ] = "NO_RESPONSE"
    physical_outbound_count: int = Field(ge=0)
    blocked_additional_outbound_count: int = Field(ge=0)

    @field_validator(
        "gateway_error_code",
        "gateway_detail_code",
        "safe_provider_error_code",
        "safe_provider_error_type",
    )
    @classmethod
    def validate_safe_code(cls, value: str | None, info: object) -> str | None:
        field_name = getattr(info, "field_name", "")
        if field_name == "gateway_detail_code" and value is not None:
            if _safe_detail(value) is None:
                raise ValueError("diagnostic detail is not allowlisted")
            return value
        allowed = (
            GATEWAY_CODES
            if field_name == "gateway_error_code"
            else DETAIL_CODES
            if field_name == "gateway_detail_code"
            else PROVIDER_CODES
        )
        if value is not None and value not in allowed:
            raise ValueError("diagnostic code is not allowlisted")
        return value

    @field_validator("safe_exception_chain")
    @classmethod
    def validate_chain(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(item not in EXCEPTION_CODES for item in value):
            raise ValueError("exception category is not allowlisted")
        return value

    @field_validator(
        "reservation_persisted_at_utc",
        "gateway_entered_at_utc",
        "outbound_attempted_at_utc",
        "outbound_completed_at_utc",
        "failure_recorded_at_utc",
    )
    @classmethod
    def validate_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("diagnostic timestamp must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None


def _exception_category(exc: BaseException) -> str:
    for exception_type, category in (
        (httpx.DecodingError, "DECODING_ERROR"),
        (httpx.StreamConsumed, "STREAM_CONSUMED"),
        (httpx.StreamClosed, "STREAM_CLOSED"),
        (httpx.ConnectTimeout, "CONNECT_TIMEOUT"),
        (httpx.ReadTimeout, "READ_TIMEOUT"),
        (httpx.WriteTimeout, "WRITE_TIMEOUT"),
        (httpx.PoolTimeout, "POOL_TIMEOUT"),
        (httpx.TimeoutException, "TIMEOUT"),
        (httpx.ConnectError, "CONNECT_ERROR"),
        (httpx.RequestError, "NETWORK_ERROR"),
        (socket.gaierror, "DNS_ERROR"),
        (ssl.SSLError, "TLS_ERROR"),
        (ConnectionResetError, "CONNECTION_RESET"),
        (ModelGatewayError, "GATEWAY_ERROR"),
        (json.JSONDecodeError, "JSON_DECODE_ERROR"),
        (ValidationError, "VALIDATION_ERROR"),
        (AcquisitionBlocked, "BUDGET_GUARD"),
        (OSError, "OS_ERROR"),
    ):
        if isinstance(exc, exception_type):
            return category
    return "UNKNOWN_EXCEPTION"


def _chain(exc: BaseException) -> tuple[BaseException, ...]:
    result: list[BaseException] = []
    seen: set[int] = set()
    while id(exc) not in seen and len(result) < 12:
        seen.add(id(exc))
        result.append(exc)
        next_exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        if next_exc is None:
            break
        exc = next_exc
    return tuple(result)


def _gateway_category(code: str | None) -> FailureCategory | None:
    return {
        "MODEL_AUTHENTICATION_FAILED": FailureCategory.AUTHENTICATION,
        "MODEL_RATE_LIMITED": FailureCategory.RATE_LIMIT,
        "MODEL_TIMEOUT": FailureCategory.TIMEOUT,
        "MODEL_NETWORK_ERROR": FailureCategory.NETWORK,
        "MODEL_REQUEST_INVALID": FailureCategory.REQUEST_INVALID,
        "MODEL_PROVIDER_UNAVAILABLE": FailureCategory.PROVIDER_UNAVAILABLE,
        "MODEL_RESPONSE_INVALID": FailureCategory.RESPONSE_INVALID,
        "MODEL_OUTPUT_INVALID": FailureCategory.STRUCTURED_OUTPUT_INVALID,
        "MODEL_OUTPUT_TRUNCATED": FailureCategory.STRUCTURED_OUTPUT_INVALID,
    }.get(code or "")


@dataclass
class FailureDiagnosticsRecorder:
    """Mutable in-memory observations shared by the PATH-C entry and transport."""

    phase: FailurePhase = FailurePhase.PREFLIGHT
    reservation_persisted_at_utc: datetime | None = None
    gateway_entered_at_utc: datetime | None = None
    outbound_attempted_at_utc: datetime | None = None
    outbound_completed_at_utc: datetime | None = None
    provider_http_status: int | None = None
    response_received: bool | None = None
    response_body_present: bool | None = None
    raw_response_preserved: bool = False
    response_body_sha256: str | None = None
    response_body_size: int | None = None
    safe_provider_error_code: str | None = None
    safe_provider_error_type: str | None = None
    provider_error_fields_redacted: bool = False
    raw_capture_policy: Literal[
        "NO_RESPONSE",
        "EXACT_SUCCESS_RESPONSE",
        "ERROR_RESPONSE_HASH_ONLY",
        "SENSITIVE_RESPONSE_HASH_ONLY",
        "CAPTURE_FAILED",
    ] = "NO_RESPONSE"
    transport_exception_categories: tuple[str, ...] = ()
    gateway_error_code: str | None = None
    gateway_detail_code: str | None = None
    gateway_detail_redacted: bool = False
    gateway_exception_categories: tuple[str, ...] = ()
    physical_outbound_count: int = 0
    blocked_additional_outbound_count: int = 0

    def observe_gateway_error(self, exc: ModelGatewayError) -> None:
        if self.gateway_error_code is not None:
            return
        self.gateway_error_code = exc.code if exc.code in GATEWAY_CODES else None
        self.gateway_detail_code = _safe_detail(exc.detail_code)
        self.gateway_detail_redacted = exc.detail_code is not None and (
            self.gateway_detail_code is None
        )
        if self.gateway_detail_redacted and exc.code in {
            "MODEL_OUTPUT_INVALID",
            "MODEL_OUTPUT_TRUNCATED",
        }:
            self.gateway_detail_code = "STRUCTURED_OUTPUT_FAILURE"
        self.gateway_exception_categories = tuple(_exception_category(e) for e in _chain(exc))

    def snapshot(
        self,
        exc: BaseException,
        *,
        physical_outbound_count: int,
        blocked_additional_outbound_count: int,
        recorded_at: datetime | None = None,
    ) -> PathCFailureDiagnostics:
        self.physical_outbound_count = physical_outbound_count
        self.blocked_additional_outbound_count = blocked_additional_outbound_count
        chain = _chain(exc)
        for item in chain:
            if isinstance(item, ModelGatewayError):
                self.observe_gateway_error(item)
        categories = tuple(
            dict.fromkeys(
                (
                    *self.transport_exception_categories,
                    *self.gateway_exception_categories,
                    *(_exception_category(e) for e in chain),
                )
            )
        )
        category = _gateway_category(self.gateway_error_code)
        status = self.provider_http_status
        if category is None and status is not None:
            if status in {401, 403}:
                category = FailureCategory.AUTHENTICATION
            elif status == 429:
                category = FailureCategory.RATE_LIMIT
            elif status >= 500:
                category = FailureCategory.PROVIDER_UNAVAILABLE
            elif status >= 400:
                category = FailureCategory.REQUEST_INVALID
        if category is None and any("TIMEOUT" in item for item in categories):
            category = FailureCategory.TIMEOUT
        if category is None and any(
            item in categories
            for item in (
                "NETWORK_ERROR",
                "CONNECT_ERROR",
                "DNS_ERROR",
                "TLS_ERROR",
                "CONNECTION_RESET",
            )
        ):
            category = FailureCategory.NETWORK
        code = getattr(exc, "code", None)
        if category is None and code == "EXECUTION_PROVIDER_BINDING_CHANGED":
            category = FailureCategory.PROVIDER_BINDING_CHANGED
        if category is None and self.phase == FailurePhase.CREDENTIAL_RESOLUTION:
            category = FailureCategory.CREDENTIAL_RESOLUTION
        if category is None and self.phase == FailurePhase.REQUEST_CONSTRUCTION:
            category = FailureCategory.REQUEST_CONSTRUCTION
        if category is None and code in {
            "PROVIDER_JSON_OBJECT_INVALID",
            "PROVIDER_TASK_RESPONSE_INVALID",
            "TASK_COUNT_INVALID",
            "TASK_TEXT_EMPTY",
            "STRUCTURAL_DUPLICATE_TASK",
            "FROZEN_TASK_MANIFEST_SCHEMA_INVALID",
        }:
            category = FailureCategory.STRUCTURED_OUTPUT_INVALID
        if category is None and any(
            item in categories
            for item in ("JSON_DECODE_ERROR", "DECODING_ERROR", "STREAM_CONSUMED", "STREAM_CLOSED")
        ):
            category = FailureCategory.RESPONSE_INVALID
        if (
            category is None
            and physical_outbound_count == 0
            and self.phase == FailurePhase.PREFLIGHT
        ):
            category = FailureCategory.CONTRACT_INVALID
        failure_class: Literal[
            "MODEL_GATEWAY_ERROR",
            "TRANSPORT_ERROR",
            "GENERATION_CONTRACT_ERROR",
            "LOCAL_RUNTIME_ERROR",
        ] = "LOCAL_RUNTIME_ERROR"
        if self.gateway_error_code is not None or any(
            isinstance(e, ModelGatewayError) for e in chain
        ):
            failure_class = "MODEL_GATEWAY_ERROR"
        elif self.transport_exception_categories:
            failure_class = "TRANSPORT_ERROR"
        elif code is not None:
            failure_class = "GENERATION_CONTRACT_ERROR"
        return PathCFailureDiagnostics(
            failure_class=failure_class,
            failure_category=category or FailureCategory.OTHER,
            failure_phase=self.phase,
            gateway_error_code=self.gateway_error_code,
            gateway_detail_code=self.gateway_detail_code,
            gateway_detail_redacted=self.gateway_detail_redacted,
            provider_http_status=status,
            safe_provider_error_code=self.safe_provider_error_code,
            safe_provider_error_type=self.safe_provider_error_type,
            provider_error_fields_redacted=self.provider_error_fields_redacted,
            safe_exception_chain=categories,
            reservation_persisted_at_utc=self.reservation_persisted_at_utc,
            gateway_entered_at_utc=self.gateway_entered_at_utc,
            outbound_attempted_at_utc=self.outbound_attempted_at_utc,
            outbound_completed_at_utc=self.outbound_completed_at_utc,
            failure_recorded_at_utc=recorded_at or datetime.now(UTC),
            response_received=self.response_received,
            response_body_present=self.response_body_present,
            raw_response_preserved=self.raw_response_preserved,
            response_body_sha256=self.response_body_sha256,
            response_body_size=self.response_body_size,
            raw_capture_policy=self.raw_capture_policy,
            physical_outbound_count=physical_outbound_count,
            blocked_additional_outbound_count=blocked_additional_outbound_count,
        )


class ObservedPathCGateway(LLMGateway):
    """Observe the original HTTP error before a blocked Gateway fallback masks it.

    Request construction, parsing and fallback decisions remain in LLMGateway.
    The existing single-outbound transport guard still rejects every extra call.
    """

    def __init__(self, client: httpx.AsyncClient, observations: FailureDiagnosticsRecorder) -> None:
        super().__init__(client)
        self._observations = observations

    async def _post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            return await super()._post_json(url, headers=headers, body=body)
        except ModelGatewayError as exc:
            self._observations.observe_gateway_error(exc)
            raise
