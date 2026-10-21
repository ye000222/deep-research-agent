"""Persist the first raw Provider response before the Gateway parses it."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    PROVIDER_CODES,
    FailureDiagnosticsRecorder,
    _chain,
    _exception_category,
)
from scripts.prospective_acquisition.path_c_generation import _exclusive_write


def _contains_sensitive_response(raw_body: bytes, request: httpx.Request) -> bool:
    # Never persist a reflected credential, even inside a nominal success body.
    authorization = request.headers.get("Authorization", "")
    secret = authorization.partition(" ")[2]
    if secret and secret.encode("utf-8") in raw_body:
        return True
    try:
        decoded: object = json.loads(raw_body)
    except (ValueError, RecursionError):
        return False
    forbidden = {
        "authorization",
        "api_key",
        "credential",
        "credentials",
        "cookie",
        "cookies",
        "owner_hash",
        "master_key",
        "password",
        "access_token",
        "secret",
    }

    def contains(value: object) -> bool:
        if isinstance(value, str):
            return bool(secret and secret in value)
        if isinstance(value, dict):
            return any(
                str(key).casefold() in forbidden or contains(item) for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    try:
        return contains(decoded)
    except RecursionError:
        return True


class RawResponseCaptureTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        destination: Path,
        observations: FailureDiagnosticsRecorder | None = None,
    ) -> None:
        self._transport = transport
        self._destination = destination
        self.observations = observations or FailureDiagnosticsRecorder()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        observations = self.observations
        observations.physical_outbound_count += 1
        observations.outbound_attempted_at_utc = datetime.now(UTC)
        observations.response_received = False
        response: httpx.Response | None = None
        try:
            response = await self._transport.handle_async_request(request)
            observations.response_received = True
            observations.provider_http_status = response.status_code
            raw_body = await response.aread()
            observations.response_body_present = bool(raw_body)
            observations.response_body_sha256 = hashlib.sha256(raw_body).hexdigest()
            observations.response_body_size = len(raw_body)
            try:
                payload: object = json.loads(raw_body)
            except (ValueError, RecursionError):
                payload = None
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                for name, target in (
                    ("code", "safe_provider_error_code"),
                    ("type", "safe_provider_error_type"),
                ):
                    value = error.get(name)
                    if isinstance(value, str) and value in PROVIDER_CODES:
                        setattr(observations, target, value)
                    elif value is not None:
                        observations.provider_error_fields_redacted = True
            is_error = response.status_code >= 300 or error is not None
            sensitive = _contains_sensitive_response(raw_body, request)
            # Create-once persistence happens before returning any response bytes to
            # LLMGateway. Error/credential-bearing bodies get only a hash and size;
            # their arbitrary messages and reflected request fields are not retained.
            if is_error or sensitive:
                observations.raw_capture_policy = (
                    "ERROR_RESPONSE_HASH_ONLY" if is_error else "SENSITIVE_RESPONSE_HASH_ONLY"
                )
            else:
                try:
                    _exclusive_write(self._destination, raw_body)
                except Exception:
                    observations.raw_capture_policy = "CAPTURE_FAILED"
                    raise
                observations.raw_response_preserved = True
                observations.raw_capture_policy = "EXACT_SUCCESS_RESPONSE"
            # aread() returns HTTPX-decoded entity bytes, not compressed wire
            # bytes. Retaining Content-Encoding here makes Response(content=...)
            # decode the same payload again during construction. Normalize only
            # representation/framing headers; never alter the decoded content.
            headers = httpx.Headers(response.headers)
            for name in ("content-encoding", "content-length", "transfer-encoding"):
                if name in headers:
                    del headers[name]
            return httpx.Response(
                status_code=response.status_code,
                headers=headers,
                content=raw_body,
                extensions=dict(response.extensions),
                request=request,
            )
        except BaseException as exc:
            observations.transport_exception_categories = tuple(
                _exception_category(item) for item in _chain(exc)
            )
            raise
        finally:
            observations.outbound_completed_at_utc = datetime.now(UTC)
            if response is not None:
                await response.aclose()

    async def aclose(self) -> None:
        await self._transport.aclose()
