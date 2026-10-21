"""Guarded one-Claim Canary entry point; no implicit DB, credentials, or network setup."""

from __future__ import annotations

import argparse
import errno
import json
import re
import socket
import ssl
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
CANARY_DIAGNOSTIC_ARTIFACT_DIR = ROOT / "artifacts"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.domain.atomic_proposition import ATOMIC_PROPOSITION_SCHEMA_VERSION  # noqa: E402
from app.domain.atomic_proposition_contracts import (  # noqa: E402
    AtomicPropositionExecutionMode,
    resolve_atomic_proposition_contract,
)
from app.llm.adapters import ModelGatewayError  # noqa: E402

_SAFE_GATEWAY_CODES = frozenset(
    {
        "MODEL_AUTHENTICATION_FAILED",
        "MODEL_REQUEST_INVALID",
        "MODEL_RESPONSE_INVALID",
        "MODEL_OUTPUT_INVALID",
        "MODEL_OUTPUT_TRUNCATED",
        "MODEL_TIMEOUT",
        "MODEL_NETWORK_ERROR",
        "MODEL_RATE_LIMITED",
        "MODEL_PROVIDER_UNAVAILABLE",
        "PROVIDER_REDIRECT_BLOCKED",
        "UNSUPPORTED_ADAPTER",
    }
)
_SAFE_DETAIL_CODES = frozenset(
    {
        "PROVIDER_LENGTH_STOP",
        "CONNECT_TIMEOUT",
        "READ_TIMEOUT",
        "WRITE_TIMEOUT",
        "POOL_TIMEOUT",
        "CONNECT_ERROR",
        "READ_ERROR",
        "WRITE_ERROR",
        "REMOTE_PROTOCOL_ERROR",
        "LOCAL_PROTOCOL_ERROR",
        "REQUEST_ERROR",
    }
)


@dataclass(frozen=True)
class SafeGatewayFailure:
    classification: str
    code: str | None = None
    detail_code: str | None = None

    def safe_dict(self) -> dict[str, str | None]:
        return {
            "classification": self.classification,
            "code": self.code,
            "detail_code": self.detail_code,
        }


@dataclass(frozen=True)
class CanaryTransportRuntime:
    proxy_mode: str
    trust_env: bool
    connect_timeout: float | None
    read_timeout: float | None
    write_timeout: float | None
    pool_timeout: float | None
    transport_retry_count: int

    def safe_dict(self) -> dict[str, object]:
        return {
            "proxy_mode": self.proxy_mode,
            "trust_env": self.trust_env,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
            "pool_timeout": self.pool_timeout,
            "transport_retry_count": self.transport_retry_count,
        }


@dataclass
class CanaryTransportDiagnostics:
    runtime: CanaryTransportRuntime
    request_scheme: str | None = None
    request_host: str | None = None
    request_port: int | None = None
    request_path_template: str | None = None
    exception_chain_categories: tuple[str, ...] = ()
    outbound_request_count: int = 0
    response_received: bool = False
    stage_markers: list[str] | None = None

    def capture_request(self, request: httpx.Request) -> None:
        self.request_scheme = request.url.scheme
        self.request_host = request.url.host
        self.request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
        self.request_path_template = _safe_path_template(request.url.path)

    def capture_exception(self, exc: BaseException) -> None:
        self.exception_chain_categories = _safe_exception_chain_categories(exc)

    def mark(self, stage: str) -> None:
        if self.stage_markers is None:
            self.stage_markers = []
        self.stage_markers.append(stage)

    def safe_dict(self) -> dict[str, object]:
        return {
            "request_scheme": self.request_scheme,
            "request_host": self.request_host,
            "request_port": self.request_port,
            "request_path_template": self.request_path_template,
            **self.runtime.safe_dict(),
            "exception_chain_categories": list(self.exception_chain_categories),
            "outbound_attempted": self.outbound_request_count > 0,
            "outbound_count": self.outbound_request_count,
            "response_received": self.response_received,
            "stage_markers": list(self.stage_markers or ()),
        }


class OneRequestLimitTransport(httpx.AsyncBaseTransport):
    """Fail closed before a second outbound HTTP request reaches the network."""

    def __init__(
        self,
        delegate: httpx.AsyncBaseTransport,
        *,
        runtime: CanaryTransportRuntime,
    ) -> None:
        self._delegate = delegate
        self.outbound_request_count = 0
        self.diagnostics = CanaryTransportDiagnostics(runtime)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.outbound_request_count >= 1:
            raise httpx.RequestError("canary outbound request limit reached", request=request)
        self.outbound_request_count += 1
        self.diagnostics.outbound_request_count = self.outbound_request_count
        self.diagnostics.mark("TRANSPORT_ENTER")
        self.diagnostics.capture_request(request)
        try:
            response = await self._delegate.handle_async_request(request)
            self.diagnostics.response_received = True
            self.diagnostics.mark("TRANSPORT_EXIT")
            return response
        except Exception as exc:
            self.diagnostics.capture_exception(exc)
            raise

    async def aclose(self) -> None:
        await self._delegate.aclose()


def build_canary_http_client(
    *,
    timeout: httpx.Timeout,
    proxy: str | None = None,
    delegate_transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[httpx.AsyncClient, OneRequestLimitTransport]:
    """Build the Canary-only HTTPX client with env proxies and transport retries disabled."""
    delegate = (
        delegate_transport
        if delegate_transport is not None
        else httpx.AsyncHTTPTransport(retries=0, proxy=proxy)
    )
    runtime = CanaryTransportRuntime(
        proxy_mode="PROXY" if proxy else "DIRECT",
        trust_env=False,
        connect_timeout=timeout.connect,
        read_timeout=timeout.read,
        write_timeout=timeout.write,
        pool_timeout=timeout.pool,
        transport_retry_count=0,
    )
    guarded_transport = OneRequestLimitTransport(delegate, runtime=runtime)
    client = httpx.AsyncClient(timeout=timeout, transport=guarded_transport, trust_env=False)
    return client, guarded_transport


def _safe_path_template(path: str) -> str:
    if path in {"/chat/completions", "/v1/chat/completions"}:
        return path
    if path.endswith("/chat/completions"):
        return "/<base-path>/chat/completions"
    return "/<redacted-path>" if path != "/" else "/"


def _safe_exception_chain_categories(exc: BaseException) -> tuple[str, ...]:
    categories: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(categories) < 8:
        seen.add(id(current))
        category = _safe_exception_category(current)
        if category and (not categories or categories[-1] != category):
            categories.append(category)
        current = current.__cause__ or current.__context__
    if not categories:
        return ("UNKNOWN_CONNECT_ERROR",)
    return tuple(categories)


def _safe_exception_category(exc: BaseException) -> str | None:
    if isinstance(exc, httpx.ConnectTimeout):
        return "CONNECT_TIMEOUT"
    if isinstance(exc, httpx.PoolTimeout):
        return "POOL_TIMEOUT"
    if isinstance(exc, httpx.ReadTimeout):
        return "READ_TIMEOUT"
    if isinstance(exc, httpx.WriteTimeout):
        return "WRITE_TIMEOUT"
    if isinstance(exc, httpx.ProxyError) or type(exc).__name__ == "ProxyError":
        return "PROXY_CONNECT_ERROR"
    if isinstance(exc, (ssl.SSLError, ssl.CertificateError)):
        return "TLS_ERROR"
    if isinstance(exc, socket.gaierror):
        return "DNS_ERROR"
    if isinstance(exc, (ConnectionResetError,)) or getattr(exc, "errno", None) == errno.ECONNRESET:
        return "CONNECTION_RESET"
    if isinstance(exc, (httpx.RemoteProtocolError,)):
        return "REMOTE_CLOSE"
    if isinstance(exc, httpx.ConnectError):
        return "UNKNOWN_CONNECT_ERROR"
    if isinstance(exc, OSError):
        return "OS_NETWORK_ERROR"
    if isinstance(exc, httpx.TimeoutException):
        return "CONNECT_TIMEOUT"
    return None


class CanaryGateway:
    """Diagnostics-only wrapper; delegates request construction to the normal Gateway."""

    def __init__(
        self,
        gateway: Any,
        *,
        transport_diagnostics: CanaryTransportDiagnostics | OneRequestLimitTransport,
    ) -> None:
        self._gateway = gateway
        self._transport_diagnostics = (
            transport_diagnostics.diagnostics
            if isinstance(transport_diagnostics, OneRequestLimitTransport)
            else transport_diagnostics
        )
        self.safe_failure: SafeGatewayFailure | None = None
        self.safe_exception_type: str | None = None
        self.failure_stage: str | None = None
        self.gateway_call_count = 0
        self.regeneration_options: list[bool] = []

    @property
    def safe_transport_diagnostics(self) -> dict[str, object] | None:
        if self._transport_diagnostics is None:
            return None
        return self._transport_diagnostics.safe_dict()

    async def generate_structured(self, **kwargs: Any) -> Any:
        self.safe_failure = None
        self.safe_exception_type = None
        self.failure_stage = None
        self.gateway_call_count += 1
        self.regeneration_options.append(bool(kwargs.get("allow_regeneration", True)))
        if self._transport_diagnostics is not None:
            self._transport_diagnostics.mark("GATEWAY_REQUEST_BUILD")
        try:
            response = await self._gateway.generate_structured(**kwargs)
            if self._transport_diagnostics is not None:
                self._transport_diagnostics.mark("GATEWAY_RESPONSE")
            return response
        except ModelGatewayError as exc:
            code = exc.code if exc.code in _SAFE_GATEWAY_CODES else None
            detail = exc.detail_code if exc.detail_code in _SAFE_DETAIL_CODES else None
            self.safe_failure = SafeGatewayFailure(_classify_gateway_code(code), code, detail)
            self.safe_exception_type = type(exc).__name__
            if (
                self._transport_diagnostics is not None
                and self._transport_diagnostics.response_received
            ):
                self._transport_diagnostics.mark("GATEWAY_RESPONSE")
            self.failure_stage = self._current_failure_stage()
            raise
        except Exception as exc:
            # Never serialize exception text, provider payloads, headers, or credentials.
            self.safe_failure = SafeGatewayFailure("UNKNOWN")
            self.safe_exception_type = type(exc).__name__
            self.failure_stage = self._current_failure_stage()
            raise

    def _current_failure_stage(self) -> str:
        if self._transport_diagnostics is not None:
            if self._transport_diagnostics.response_received:
                return "GATEWAY_RESPONSE"
            stages = self._transport_diagnostics.stage_markers or []
            if stages and stages[-1] in {"TRANSPORT_ENTER", "TRANSPORT_EXIT"}:
                return stages[-1]
        return "GATEWAY_REQUEST_BUILD"


class CanaryDecomposer:
    """Expose Canary-safe Gateway diagnostics alongside the unchanged decomposer."""

    def __init__(self, decomposer: OneClaimDecomposer, gateway: CanaryGateway) -> None:
        self._decomposer = decomposer
        self._gateway = gateway

    @property
    def provider_profile_id(self) -> str | None:
        return getattr(self._decomposer, "provider_profile_id", None)

    @property
    def safe_gateway_failure(self) -> SafeGatewayFailure | None:
        return self._gateway.safe_failure

    @property
    def safe_failure_stage(self) -> str | None:
        return self._gateway.failure_stage

    @property
    def safe_exception_type(self) -> str | None:
        return self._gateway.safe_exception_type

    @property
    def safe_transport_diagnostics(self) -> dict[str, object] | None:
        return self._gateway.safe_transport_diagnostics

    async def decompose_claim(
        self,
        claim_text: str,
        context_metadata: Mapping[str, object] | None = None,
        *,
        claim_id: str | None = None,
        schema_version: str = ATOMIC_PROPOSITION_SCHEMA_VERSION,
        prompt_version: str | None = None,
        execution_mode: AtomicPropositionExecutionMode = AtomicPropositionExecutionMode.SHADOW,
    ) -> Any:
        return await self._decomposer.decompose_claim(
            claim_text,
            context_metadata,
            claim_id=claim_id,
            schema_version=schema_version,
            prompt_version=prompt_version,
            execution_mode=execution_mode,
        )


def _classify_gateway_code(code: str | None) -> str:
    if code == "MODEL_AUTHENTICATION_FAILED":
        return "AUTH"
    if code in {"INVALID_BASE_URL", "PROVIDER_NOT_CONFIGURED"}:
        return "CONFIG"
    if code == "UNSUPPORTED_ADAPTER":
        return "ADAPTER"
    if code == "MODEL_REQUEST_INVALID":
        return "REQUEST_FORMAT"
    if code in {
        "MODEL_RESPONSE_INVALID",
        "MODEL_OUTPUT_INVALID",
        "MODEL_OUTPUT_TRUNCATED",
        "MODEL_RATE_LIMITED",
        "MODEL_PROVIDER_UNAVAILABLE",
        "PROVIDER_REDIRECT_BLOCKED",
    }:
        return "PROVIDER_RESPONSE"
    return "UNKNOWN"


class OneClaimDecomposer(Protocol):
    async def decompose_claim(
        self,
        claim_text: str,
        context_metadata: Mapping[str, object] | None = None,
        *,
        claim_id: str | None = None,
        schema_version: str = ATOMIC_PROPOSITION_SCHEMA_VERSION,
        prompt_version: str | None = None,
        execution_mode: AtomicPropositionExecutionMode = AtomicPropositionExecutionMode.SHADOW,
    ) -> Any: ...


@dataclass(frozen=True)
class CanaryRequest:
    claim_id: str | None
    claim_text: str
    provider_profile_id: str | None
    profile_ready: bool
    credential_ready: bool
    schema_version: str
    maximum_calls: int
    holdout_exclusion_established: bool
    excluded_claim_ids: frozenset[str]
    context_metadata: Mapping[str, object]
    execution_mode: AtomicPropositionExecutionMode = AtomicPropositionExecutionMode.SHADOW
    prompt_version: str | None = None


@dataclass(frozen=True)
class CanaryResult:
    status: str
    reason: str | None
    decomposition: Any | None = None
    gateway_failure: SafeGatewayFailure | None = None
    transport_diagnostics: dict[str, object] | None = None
    claim: dict[str, str | None] | None = None
    failure_stage: str | None = None
    safe_exception_type: str | None = None
    outbound_attempted: bool = False
    outbound_count: int = 0
    response_received: bool = False
    stage_markers: tuple[str, ...] = ()
    payload_fidelity_status: str | None = None
    contract_resolution: dict[str, str | bool | None] | None = None

    def safe_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"status": self.status, "reason": self.reason}
        data["failure_stage"] = self.failure_stage
        data["safe_exception_type"] = self.safe_exception_type
        data["outbound_attempted"] = self.outbound_attempted
        data["outbound_count"] = self.outbound_count
        data["response_received"] = self.response_received
        data["stage_markers"] = list(self.stage_markers)
        data["payload_fidelity_status"] = self.payload_fidelity_status
        if self.contract_resolution is not None:
            data["contract_resolution"] = self.contract_resolution
        if self.claim is not None:
            data["claim"] = self.claim
        if self.gateway_failure is not None:
            data["gateway_failure"] = self.gateway_failure.safe_dict()
        if self.transport_diagnostics is not None:
            data["transport_diagnostics"] = self.transport_diagnostics
        if self.decomposition is not None:
            payload = self.decomposition.model_dump(mode="json")
            safe_provenance = {
                key: value
                for key, value in payload["provenance"].items()
                if key
                in {
                    "claim_id",
                    "evidence_id",
                    "source_id",
                    "source_owner_key",
                    "run_id",
                    "question_id",
                    "dimension_id",
                }
            }
            data["decomposition"] = {
                "claim_id": payload["claim_id"],
                "provenance": safe_provenance,
                "schema_version": payload["schema_version"],
                "prompt_version": payload["prompt_version"],
                "status": payload["status"],
                "proposition_count": len(payload["propositions"]),
                "pre_validation_proposition_count": payload.get("pre_validation_proposition_count"),
                "validation_issue_codes": [issue["code"] for issue in payload["validation_issues"]],
                "model_call_count": payload["model_call_count"],
                "input_tokens": payload["input_tokens"],
                "output_tokens": payload["output_tokens"],
                "latency_ms": payload["latency_ms"],
                "model_metadata": payload["model_metadata"],
                "model_structured_result": payload.get("model_structured_result"),
                "parser_result": payload.get("parser_result", {}),
                "validator_result": payload.get("validator_result", {}),
                "status_transition_trace": payload.get("status_transition_trace", []),
                "final_shadow_result": {
                    "status": payload["status"],
                    "proposition_count": len(payload["propositions"]),
                    "validation_issue_codes": [
                        issue["code"] for issue in payload["validation_issues"]
                    ],
                },
            }
            data["provider_model"] = payload["model_metadata"]
            data["tokens"] = {
                "input": payload["input_tokens"],
                "output": payload["output_tokens"],
                "total": payload["input_tokens"] + payload["output_tokens"],
            }
            data["latency_ms"] = payload["latency_ms"]
        return data

    def safe_json(self) -> str:
        """Serialize the safe result and identify serialization failures without details."""
        try:
            data = self.safe_dict()
        except Exception as exc:
            return json.dumps(
                {
                    "status": "FAILED",
                    "reason": "result_build_failed",
                    "failure_stage": "CANARY_RESULT_BUILD",
                    "safe_exception_type": type(exc).__name__,
                    "outbound_attempted": self.outbound_attempted,
                    "outbound_count": self.outbound_count,
                    "response_received": self.response_received,
                    "stage_markers": [*self.stage_markers, "CANARY_RESULT_BUILD"],
                },
                ensure_ascii=False,
            )
        stage_values = data.get("stage_markers", [])
        stages = [str(stage) for stage in stage_values] if isinstance(stage_values, list) else []
        stages.append("CANARY_RESULT_SERIALIZE")
        data["stage_markers"] = stages
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            safe_failure = {
                "status": "FAILED",
                "reason": "result_serialization_failed",
                "failure_stage": "CANARY_RESULT_SERIALIZE",
                "safe_exception_type": type(exc).__name__,
                "outbound_attempted": self.outbound_attempted,
                "outbound_count": self.outbound_count,
                "response_received": self.response_received,
                "stage_markers": stages,
            }
            return json.dumps(safe_failure, ensure_ascii=False)


_ARTIFACT_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s,;\"']+"),
    re.compile(
        r"(?i)\b(?:authorization|x-api-key|api[_ -]?key|cookie|master[_ -]?key|credential|"
        r"secret[_ -]?header|x-[\w-]*secret[\w-]*)\b[\"']?\s*[:=]\s*(?:Bearer\s+)?[^\s,;\"']+"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_ARTIFACT_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(authorization|api.?key|credential|master.?key|secret|cookie|password|access.?token)"
)


def _sanitize_artifact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_artifact_value(item)
            for key, item in value.items()
            if not _ARTIFACT_SECRET_KEY_PATTERN.search(str(key))
        }
    if isinstance(value, list):
        return [_sanitize_artifact_value(item) for item in value]
    if isinstance(value, str):
        sanitized = value[:12_000]
        for pattern in _ARTIFACT_SECRET_VALUE_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized
    return value


def persist_safe_canary_artifact(
    result: CanaryResult,
    *,
    output_dir: Path | None = None,
    artifact_path: Path | None = None,
) -> Path:
    """Persist only the safe, redacted Canary result for post-session diagnosis."""
    output_dir = output_dir or CANARY_DIAGNOSTIC_ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_path = artifact_path or (
        output_dir / f"v2_1r_semantic_canary_{timestamp}_{uuid4().hex[:8]}.json"
    )
    safe_payload = _sanitize_artifact_value(json.loads(result.safe_json()))
    artifact_path.write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path


async def run_one_claim_canary(
    request: CanaryRequest,
    *,
    decomposer: OneClaimDecomposer | None,
    diagnostic_artifact_dir: Path | None = None,
) -> CanaryResult:
    """Run exactly one caller-supplied Claim after all explicit preflight guards pass."""

    resolution = resolve_atomic_proposition_contract(
        schema_version=request.schema_version,
        prompt_version=request.prompt_version,
        execution_mode=request.execution_mode,
    )
    preflight_artifact_path: Path | None = None

    def finish(result: CanaryResult) -> CanaryResult:
        result = replace(result, contract_resolution=resolution.safe_dict())
        persist_safe_canary_artifact(
            result,
            output_dir=diagnostic_artifact_dir,
            artifact_path=preflight_artifact_path,
        )
        return result

    if not request.claim_text.strip():
        return finish(CanaryResult("BLOCKED", "claim_missing"))
    if not request.provider_profile_id:
        return finish(CanaryResult("BLOCKED", "provider_profile_missing"))
    if not request.profile_ready:
        return finish(CanaryResult("BLOCKED", "provider_profile_not_ready"))
    if not request.credential_ready:
        return finish(CanaryResult("BLOCKED", "credential_not_ready"))
    if resolution.status != "RESOLVED" or resolution.contract is None:
        return finish(CanaryResult("BLOCKED", resolution.reason or "contract_resolution_failed"))
    if request.maximum_calls != 1:
        return finish(CanaryResult("BLOCKED", "call_budget_must_equal_one"))
    if not request.holdout_exclusion_established:
        return finish(CanaryResult("BLOCKED", "holdout_exclusion_not_established"))
    if request.claim_id is not None and request.claim_id in request.excluded_claim_ids:
        return finish(CanaryResult("BLOCKED", "claim_is_excluded"))
    if decomposer is None:
        return finish(CanaryResult("BLOCKED", "decomposer_unavailable"))
    if getattr(decomposer, "provider_profile_id", None) != request.provider_profile_id:
        return finish(CanaryResult("BLOCKED", "provider_profile_mismatch"))

    preflight_artifact_path = persist_safe_canary_artifact(
        CanaryResult(
            "READY_TO_INVOKE",
            None,
            stage_markers=("CANARY_PRECHECK", "CONTRACT_RESOLVED"),
            contract_resolution=resolution.safe_dict(),
        ),
        output_dir=diagnostic_artifact_dir,
    )
    stages = ["CANARY_PRECHECK", "CLAIM_PAYLOAD_BUILD"]
    try:
        result = await decomposer.decompose_claim(
            request.claim_text,
            request.context_metadata,
            claim_id=request.claim_id,
            schema_version=resolution.contract.schema_version,
            prompt_version=resolution.contract.prompt_version,
            execution_mode=request.execution_mode,
        )
    except Exception as exc:
        diagnostics = getattr(decomposer, "safe_transport_diagnostics", None)
        if isinstance(diagnostics, dict):
            for stage in diagnostics.get("stage_markers", []):
                if stage not in stages:
                    stages.append(stage)
        failure_stage = getattr(decomposer, "safe_failure_stage", None)
        failure_stage = failure_stage or (stages[-1] if stages else "CLAIM_PAYLOAD_BUILD")
        return finish(
            CanaryResult(
                "FAILED",
                "decomposer_error",
                gateway_failure=getattr(decomposer, "safe_gateway_failure", None),
                transport_diagnostics=diagnostics,
                failure_stage=failure_stage,
                safe_exception_type=getattr(decomposer, "safe_exception_type", None)
                or type(exc).__name__,
                outbound_attempted=bool((diagnostics or {}).get("outbound_attempted", False)),
                outbound_count=int((diagnostics or {}).get("outbound_count", 0)),
                response_received=bool((diagnostics or {}).get("response_received", False)),
                stage_markers=tuple(stages),
            )
        )
    diagnostics = getattr(decomposer, "safe_transport_diagnostics", None)
    if isinstance(diagnostics, dict):
        for stage in diagnostics.get("stage_markers", []):
            if stage not in stages:
                stages.append(stage)
    decomposition_trace = getattr(result, "status_transition_trace", ())
    trace_stages = {
        entry.get("stage"): entry for entry in decomposition_trace if isinstance(entry, Mapping)
    }
    parsed = trace_stages.get("PARSED_RESULT", {})
    if parsed.get("status") != "NOT_REACHED":
        stages.append("DECOMPOSER_PARSE")
    validated = trace_stages.get("VALIDATED_RESULT", {})
    if validated.get("status") != "NOT_REACHED":
        stages.append("DECOMPOSER_VALIDATE")
    stages.append("CANARY_RESULT_BUILD")
    diagnostic_values = diagnostics if isinstance(diagnostics, dict) else {}
    gateway_failure = getattr(decomposer, "safe_gateway_failure", None)
    failure_stage = getattr(decomposer, "safe_failure_stage", None)
    safe_exception_type = getattr(decomposer, "safe_exception_type", None)
    outbound_attempted = bool(diagnostic_values.get("outbound_attempted", False))
    outbound_count = int(diagnostic_values.get("outbound_count", 0))
    response_received = bool(diagnostic_values.get("response_received", False))
    stage_markers = tuple(stages)
    if getattr(result, "model_call_count", None) != 1:
        return finish(
            CanaryResult(
                "FAILED",
                "decomposer_call_count_violation",
                decomposition=result,
                gateway_failure=gateway_failure,
                transport_diagnostics=diagnostics,
                failure_stage=failure_stage,
                safe_exception_type=safe_exception_type,
                outbound_attempted=outbound_attempted,
                outbound_count=outbound_count,
                response_received=response_received,
                stage_markers=stage_markers,
            )
        )
    return finish(
        CanaryResult(
            "COMPLETED",
            None,
            result,
            gateway_failure,
            transport_diagnostics=diagnostics,
            claim={"claim_id": request.claim_id, "claim_text": request.claim_text},
            payload_fidelity_status=("PASS" if len(request.claim_text) <= 12_000 else "TRUNCATED"),
            failure_stage=failure_stage,
            safe_exception_type=safe_exception_type,
            outbound_attempted=outbound_attempted,
            outbound_count=outbound_count,
            response_received=response_received,
            stage_markers=stage_markers,
        )
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-json", type=Path, required=True)
    parser.add_argument("--provider-profile-id", required=True)
    parser.add_argument("--maximum-calls", type=int, default=1)
    args = parser.parse_args()
    try:
        payload = json.loads(args.claim_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"input_invalid:{type(exc).__name__}"}))
        return 2
    if not isinstance(payload, dict) or not isinstance(payload.get("claim_text"), str):
        print(json.dumps({"status": "BLOCKED", "reason": "claim_missing"}))
        return 2
    # The CLI intentionally has no credential/profile loader or network factory.
    # A reviewed caller must inject the already-approved Gateway into the function API.
    print(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": "explicit_runtime_injection_required",
                "provider_profile_id": args.provider_profile_id,
                "maximum_calls": args.maximum_calls,
            },
            ensure_ascii=False,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
