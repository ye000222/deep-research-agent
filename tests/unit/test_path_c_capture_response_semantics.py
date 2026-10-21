"""Synthetic encoded response reproduction; never inspect prospective output."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib.util
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from app.domain.providers import AdapterType
from pydantic import SecretStr

from scripts.prospective_acquisition import path_c_generation as generation
from scripts.prospective_acquisition.capture_transport import RawResponseCaptureTransport
from scripts.prospective_acquisition.control import CallBudget
from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    FailureDiagnosticsRecorder,
    ObservedPathCGateway,
)
from scripts.prospective_acquisition.path_c_generation_entry import build_path_c_model_client
from tests.unit.test_path_c_generation import _context, _copy_frozen_root, _raw_response, _tasks


def encode(body: bytes, encoding: str) -> bytes:
    if encoding == "gzip":
        return gzip.compress(body, mtime=0)
    if encoding == "br":
        name = "brotli" if importlib.util.find_spec("brotli") else "brotlicffi"
        if importlib.util.find_spec(name) is None:
            pytest.skip("optional brotli support not installed; no dependency installation")
        module = __import__(name)
        return bytes(module.compress(body))
    return body


class WireStream(httpx.AsyncByteStream):
    def __init__(self, wire: bytes, state: SimpleNamespace, fault: str | None = None) -> None:
        self.wire = wire
        self.state = state
        self.fault = fault

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.fault == "read":
            raise httpx.ReadError("synthetic secret-bearing read error")
        yield self.wire

    async def aclose(self) -> None:
        self.state.closed = True
        if self.fault == "close":
            raise httpx.ReadError("synthetic secret-bearing close error")


class WrongSyncStream(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b"synthetic"


@pytest.mark.parametrize("encoding", ["plain", "gzip", "br"])
def test_same_capture_response_consumable_and_exact_decoded_hash(
    tmp_path: Path, encoding: str
) -> None:
    body = _raw_response(_tasks())
    wire = encode(body, encoding)
    state = SimpleNamespace(closed=False)
    seen: list[httpx.Response] = []
    observations = FailureDiagnosticsRecorder()

    async def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": "application/json", "content-length": str(len(wire))}
        if encoding != "plain":
            headers["content-encoding"] = encoding
        response = httpx.Response(
            200, headers=headers, request=request, stream=WireStream(wire, state),
            extensions={"http_version": b"HTTP/1.1", "network_stream": state},
        )
        seen.append(response)
        return response

    async def run() -> None:
        capture = RawResponseCaptureTransport(
            httpx.MockTransport(handler), tmp_path / "synthetic.raw.json", observations,
        )
        client = build_path_c_model_client(CallBudget(1), capture)
        try:
            response = await client.post("https://example.invalid/chat/completions", json={})
            assert await response.aread() == body
            assert response.content == body
            assert response.json() == json.loads(body)
            assert response.request.url.host == "example.invalid"
            assert "content-encoding" not in response.headers
            assert int(response.headers["content-length"]) == len(body)
            assert response.extensions["http_version"] == b"HTTP/1.1"
            assert state.closed and seen[0].is_closed
            await response.aclose()
            await response.aclose()
            # AsyncClient installs BoundAsyncStream: sync close is misuse,
            # independently of capture; normal downstream uses aclose().
            with pytest.raises(RuntimeError):
                response.close()
            assert response.is_closed
            assert (tmp_path / "synthetic.raw.json").read_bytes() == body
            assert observations.response_body_sha256 == hashlib.sha256(body).hexdigest()
            assert observations.raw_response_preserved
            assert observations.physical_outbound_count == 1
        finally:
            await client.aclose()

    asyncio.run(run())


def test_exact_old_reconstruction_statement_raises_decoding_error() -> None:
    decoded = b'{"choices":[]}'
    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers={"content-encoding": "gzip"}, content=decoded)


def test_old_brotli_reconstruction_statement_if_supported() -> None:
    decoded = b'{"choices":[]}'
    encode(decoded, "br")  # skip if optional decoder unavailable
    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers={"content-encoding": "br"}, content=decoded)


@pytest.mark.parametrize("encoding", ["plain", "gzip", "br"])
def test_encoded_envelope_through_same_gateway_and_generation(
    tmp_path: Path, encoding: str
) -> None:
    checkout = _copy_frozen_root(tmp_path)
    out = tmp_path / "synthetic-attempt"
    body = _raw_response(_tasks())
    wire = encode(body, encoding)
    observations = FailureDiagnosticsRecorder()
    budget = CallBudget(1)
    context, binding = _context()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        assert "seed" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        assert "40" in payload["messages"][0]["content"]
        headers = {"content-type": "application/json"}
        if encoding != "plain":
            headers["content-encoding"] = encoding
        return httpx.Response(
            200, headers=headers, stream=WireStream(wire, SimpleNamespace(closed=False)),
            request=request,
        )

    async def run() -> None:
        client = build_path_c_model_client(
            budget, RawResponseCaptureTransport(
                httpx.MockTransport(handler), out / "provider_response.raw.json", observations,
            ),
        )
        async def recheck() -> None:
            return None
        try:
            result = await generation._execute_guarded_core(
                root=checkout, output_dir=out, context=context, provider_binding=binding,
                binding_recheck=recheck, gateway=ObservedPathCGateway(client, observations),
                api_key=SecretStr("fake-test-key"), base_url="https://example.invalid/v1",
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                context_manifest_id=UUID(generation.EXECUTION_CONTEXT_ID), physical_budget=budget,
                raw_response_path=out / "provider_response.raw.json",
                failure_diagnostics=observations,
            )
            assert result.state == "SUCCEEDED_AND_FROZEN"
            assert budget.dispatched == calls == observations.physical_outbound_count == 1
            assert budget.blocked_attempts == 0
            assert (out / "provider_response.raw.json").read_bytes() == body
            assert json.loads((out / "path_c_task_manifest.json").read_bytes())["task_count"] == 40
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("case,exception", [
    ("consumed", httpx.StreamConsumed), ("closed", httpx.StreamClosed),
    ("read", httpx.ReadError), ("close", httpx.ReadError),
    ("sync_stream", RuntimeError),
])
def test_lifecycle_failures_have_concrete_types(
    tmp_path: Path, case: str, exception: type[Exception]
) -> None:
    body = _raw_response(_tasks())
    observations = FailureDiagnosticsRecorder()

    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if case == "sync_stream":
                return httpx.Response(200, stream=WrongSyncStream(), request=request)
            response = httpx.Response(
                200, request=request,
                stream=WireStream(body, SimpleNamespace(closed=False), case),
            )
            if case == "consumed":
                async for _ in response.aiter_raw():
                    pass
            elif case == "closed":
                await response.aclose()
            return response

        transport = RawResponseCaptureTransport(
            httpx.MockTransport(handler), tmp_path / "synthetic.raw.json", observations,
        )
        request = httpx.Request("POST", "https://example.invalid")
        with pytest.raises(exception):
            await transport.handle_async_request(request)
        assert observations.physical_outbound_count == 1
        assert not observations.raw_response_preserved
        await transport.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("case,category,code,detail", [
    ("decoding", "RESPONSE_INVALID", "MODEL_RESPONSE_INVALID", "DECODING_ERROR"),
    ("consumed", "RESPONSE_INVALID", "MODEL_RESPONSE_INVALID", "STREAM_CONSUMED"),
    ("closed", "RESPONSE_INVALID", "MODEL_RESPONSE_INVALID", "STREAM_CLOSED"),
    ("json", "RESPONSE_INVALID", "MODEL_RESPONSE_INVALID", "JSON_DECODE_ERROR"),
    ("network", "NETWORK", "MODEL_NETWORK_ERROR", "READ_ERROR"),
])
def test_generic_taxonomy_redaction_one_call(
    tmp_path: Path, case: str, category: str, code: str, detail: str,
) -> None:
    from tests.unit.test_path_c_failure_observability import _pipeline

    checkout = _copy_frozen_root(tmp_path)
    out = tmp_path / "synthetic-attempt"
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if case == "json":
            return httpx.Response(200, content=b"synthetic-invalid-json", request=request)
        if case == "consumed":
            raise httpx.StreamConsumed()
        if case == "closed":
            raise httpx.StreamClosed()
        errors: dict[str, type[Exception]] = {
            "decoding": httpx.DecodingError, "network": httpx.ReadError,
        }
        raise errors[case]("secret-bearing synthetic exception message")

    error, budget = asyncio.run(_pipeline(checkout, out, handler))
    assert calls == budget.dispatched == 1
    assert budget.blocked_attempts == 0
    assert error.diagnostics is not None
    safe = error.diagnostics.model_dump(mode="json")
    assert safe["failure_category"] == category
    assert safe["gateway_error_code"] == code
    assert safe["gateway_detail_code"] == detail
    assert safe["gateway_detail_redacted"] is False
    assert "secret-bearing" not in json.dumps(safe)
    assert json.loads((out / "generation_state.json").read_bytes())["state"] == "CONSUMED_FAILED"


def test_invalid_extension_is_not_a_gateway_network_error(tmp_path: Path) -> None:
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=WireStream(b'{"choices":[]}', SimpleNamespace(closed=False)),
                request=request, extensions={"http_version": 123},
            )
        transport = RawResponseCaptureTransport(
            httpx.MockTransport(handler), tmp_path / "synthetic.raw.json",
        )
        response = await transport.handle_async_request(httpx.Request("GET", "https://example.invalid"))
        assert response.json() == {"choices": []}
        with pytest.raises(AttributeError):
            _ = response.http_version
        await response.aclose()
        await transport.aclose()

    asyncio.run(run())
