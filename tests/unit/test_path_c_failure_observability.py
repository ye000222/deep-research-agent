"""Same operator pipeline, fake HTTP only; no real generation authorization."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import httpx
import pytest
from app.domain import providers as provider_domain
from app.llm.adapters import ModelGatewayError
from pydantic import SecretStr, ValidationError

from scripts.prospective_acquisition import path_c_generation as generation
from scripts.prospective_acquisition import path_c_generation_entry as entry
from scripts.prospective_acquisition.capture_transport import RawResponseCaptureTransport
from scripts.prospective_acquisition.control import CallBudget, build_guarded_provider_client
from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    FailureDiagnosticsRecorder,
    FailurePhase,
    ObservedPathCGateway,
    PathCFailureDiagnostics,
)
from tests.unit.test_path_c_generation import _context, _copy_frozen_root, _tasks

SECRET = "fixture-secret-never-persist-this"
OWNER = "a" * 64


def _response(content: str, finish: str = "stop") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


async def _pipeline(
    root: Path,
    out: Path,
    handler: Any,
    *,
    observations: FailureDiagnosticsRecorder | None = None,
    binding_ok: bool = True,
    base_url: str = "https://api.deepseek.com/v1",
) -> tuple[generation.PathCGenerationBlocked, CallBudget]:
    observations = observations or FailureDiagnosticsRecorder()
    budget = CallBudget(1)
    capture = RawResponseCaptureTransport(
        httpx.MockTransport(handler),
        out / "provider_response.raw.json",
        observations,
    )
    client = build_guarded_provider_client(budget, transport=capture)
    context, binding = _context()

    async def recheck() -> None:
        if not binding_ok:
            raise RuntimeError(SECRET)

    try:
        with pytest.raises(generation.PathCGenerationBlocked) as raised:
            await generation._execute_guarded_core(
                root=root,
                output_dir=out,
                context=context,
                provider_binding=binding,
                binding_recheck=recheck,
                gateway=ObservedPathCGateway(client, observations),
                api_key=SecretStr(SECRET),
                base_url=base_url,
                adapter_type=provider_domain.AdapterType.OPENAI_COMPATIBLE_CHAT,
                context_manifest_id=UUID(generation.EXECUTION_CONTEXT_ID),
                physical_budget=budget,
                raw_response_path=out / "provider_response.raw.json",
                failure_diagnostics=observations,
            )
        return raised.value, budget
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "case,category,gateway_code,detail,status",
    [
        ("connect", "TIMEOUT", "MODEL_TIMEOUT", "CONNECT_TIMEOUT", None),
        ("read", "TIMEOUT", "MODEL_TIMEOUT", "READ_TIMEOUT", None),
        ("network", "NETWORK", "MODEL_NETWORK_ERROR", "CONNECT_ERROR", None),
        ("401", "AUTHENTICATION", "MODEL_AUTHENTICATION_FAILED", None, 401),
        ("403", "AUTHENTICATION", "MODEL_AUTHENTICATION_FAILED", None, 403),
        ("429", "RATE_LIMIT", "MODEL_RATE_LIMITED", None, 429),
        ("400", "REQUEST_INVALID", "MODEL_REQUEST_INVALID", None, 400),
        ("503", "PROVIDER_UNAVAILABLE", "MODEL_PROVIDER_UNAVAILABLE", None, 503),
        ("body_json", "RESPONSE_INVALID", "MODEL_RESPONSE_INVALID", "JSON_DECODE_ERROR", 200),
        ("shape", "RESPONSE_INVALID", "MODEL_RESPONSE_INVALID", None, 200),
        (
            "model_json",
            "STRUCTURED_OUTPUT_INVALID",
            "MODEL_OUTPUT_INVALID",
            "STRUCTURED_OUTPUT_FAILURE",
            200,
        ),
        (
            "length",
            "STRUCTURED_OUTPUT_INVALID",
            "MODEL_OUTPUT_TRUNCATED",
            "STRUCTURED_OUTPUT_FAILURE",
            200,
        ),
        ("schema", "STRUCTURED_OUTPUT_INVALID", None, None, 200),
    ],
)
def test_same_path_failures_persist_safe_diagnostics_and_never_redispatch(
    tmp_path: Path,
    case: str,
    category: str,
    gateway_code: str | None,
    detail: str | None,
    status: int | None,
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        state = json.loads((out / "generation_state.json").read_bytes())
        assert state["state"] == "DISPATCH_RESERVED"
        assert "seed" not in json.loads(request.content)
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        error_types = {
            "connect": httpx.ConnectTimeout,
            "read": httpx.ReadTimeout,
            "network": httpx.ConnectError,
        }
        if case in error_types:
            raise error_types[case](SECRET, request=request)
        if case.isdigit():
            return httpx.Response(
                int(case),
                request=request,
                json={
                    "error": {
                        "code": "invalid_request_error",
                        "message": SECRET,
                        "owner_hash": OWNER,
                    }
                },
                headers={"Set-Cookie": SECRET, "Authorization": SECRET},
            )
        if case == "body_json":
            return httpx.Response(200, content=b"not-json", request=request)
        if case == "shape":
            return httpx.Response(200, json={"choices": []}, request=request)
        content = '{"tasks":[' if case == "model_json" else json.dumps({"tasks": _tasks(39)})
        return httpx.Response(
            200,
            json=_response(content, "length" if case == "length" else "stop"),
            request=request,
        )

    error, budget = asyncio.run(_pipeline(root, out, handler))
    assert admitted == budget.dispatched == 1
    state_bytes = (out / "generation_state.json").read_bytes()
    state = json.loads(state_bytes)
    assert state["state"] == "CONSUMED_FAILED"
    assert state["physical_outbound_count"] == 1
    diagnostic = state["failure_diagnostics"]
    PathCFailureDiagnostics.model_validate_json(json.dumps(diagnostic))
    assert error.diagnostics is not None
    assert diagnostic["failure_category"] == category
    assert diagnostic["gateway_error_code"] == gateway_code
    if detail == "STRUCTURED_OUTPUT_FAILURE":
        assert diagnostic["gateway_detail_code"].startswith("OUTPUT_INVALID_JSON_MODE_FINISH_")
        assert diagnostic["gateway_detail_redacted"] is False
    else:
        assert diagnostic["gateway_detail_code"] == detail
    assert diagnostic["provider_http_status"] == status
    assert diagnostic["physical_outbound_count"] == 1
    assert diagnostic["response_received"] is (status is not None)
    stamps = [
        datetime.fromisoformat(diagnostic[key].replace("Z", "+00:00"))
        for key in (
            "reservation_persisted_at_utc",
            "gateway_entered_at_utc",
            "outbound_attempted_at_utc",
            "outbound_completed_at_utc",
            "failure_recorded_at_utc",
        )
    ]
    assert stamps == sorted(stamps)
    assert all(stamp.utcoffset() == timedelta(0) for stamp in stamps)
    assert not (out / "path_c_task_manifest.json").exists()
    if case.isdigit():
        assert diagnostic["raw_capture_policy"] == "ERROR_RESPONSE_HASH_ONLY"
        assert diagnostic["response_body_sha256"] is not None
        assert diagnostic["response_body_size"] > 0
        assert not (out / "provider_response.raw.json").exists()
    if case in {"connect", "read", "network"}:
        assert not (out / "provider_response.raw.json").exists()
        assert diagnostic["response_body_sha256"] is None
    for path in out.iterdir():
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes()
            assert OWNER.encode() not in path.read_bytes()
            assert b"Set-Cookie" not in path.read_bytes()
    _, rerun_budget = asyncio.run(_pipeline(root, out, handler))
    assert admitted == 1 and rerun_budget.dispatched == 0
    assert (out / "generation_state.json").read_bytes() == state_bytes


def test_origin_connect_timeout_cannot_hide_first_error_behind_blocked_fallback(
    tmp_path: Path,
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        raise httpx.ConnectTimeout(SECRET, request=request)

    _, budget = asyncio.run(_pipeline(root, out, handler, base_url="https://api.deepseek.com"))
    diagnostic = json.loads((out / "generation_state.json").read_bytes())["failure_diagnostics"]
    assert admitted == budget.dispatched == 1
    assert budget.blocked_attempts == 1
    assert diagnostic["gateway_error_code"] == "MODEL_TIMEOUT"
    assert diagnostic["gateway_detail_code"] == "CONNECT_TIMEOUT"
    assert "BUDGET_GUARD" in diagnostic["safe_exception_chain"]


@pytest.mark.parametrize("case", ["contract", "provider_drift", "request_construction"])
def test_pre_dispatch_failures_have_zero_physical_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        raise AssertionError("preflight must not reach transport")

    if case == "contract":
        contract = root / "artifacts/v2_1r_d4r_a51/generic_task_generation_contract.json"
        contract.write_bytes(contract.read_bytes() + b" ")
    if case == "request_construction":

        def invalid_request(**kwargs: object) -> object:
            raise ValueError(SECRET)

        monkeypatch.setattr(provider_domain, "CanonicalModelRequest", invalid_request)
    error, budget = asyncio.run(_pipeline(root, out, handler, binding_ok=case != "provider_drift"))
    assert admitted == budget.dispatched == 0
    assert not (out / "generation_state.json").exists()
    assert error.diagnostics is not None
    diagnostic = error.diagnostics.model_dump(mode="json")
    assert diagnostic["physical_outbound_count"] == 0
    assert diagnostic["gateway_entered_at_utc"] is None
    assert diagnostic["outbound_attempted_at_utc"] is None
    assert (
        diagnostic["failure_category"]
        == {
            "contract": "CONTRACT_INVALID",
            "provider_drift": "PROVIDER_BINDING_CHANGED",
            "request_construction": "REQUEST_CONSTRUCTION",
        }[case]
    )
    assert SECRET not in json.dumps(diagnostic)


def test_real_entry_credential_resolution_failure_is_safe_and_pre_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, binding = _context()
    disposed = False
    # Selection itself is exercised with real synthetic authorization files in
    # test_path_c_attempts. This credential test isolates the unchanged entry
    # failure stage with an in-memory authorized target, before any writer.
    fake_attempt = SimpleNamespace(location=tmp_path / "unused-attempt")
    monkeypatch.setattr(
        entry,
        "resolve_operator_attempt",
        lambda *a: SimpleNamespace(
            authorized=fake_attempt,
        ),
    )
    monkeypatch.setattr(entry, "validate_authorized_attempt", lambda *a: None)

    class FakeEngine:
        async def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    class FakeProfileService:
        async def get_test_material(self, *args: object) -> object:
            raise RuntimeError(SECRET)

    async def owner_scope(*args: object) -> str:
        return OWNER

    async def resolve_binding(*args: object) -> object:
        return binding

    monkeypatch.setattr(
        entry,
        "Settings",
        lambda: SimpleNamespace(
            artifact_root=tmp_path,
            database_url="fake-offline-database",
            allow_insecure_provider_endpoints=False,
        ),
    )
    monkeypatch.setattr(
        entry,
        "_safe_json",
        lambda path: {
            "provider_binding": {
                "profile_id": binding.profile_id,
            }
        },
    )
    monkeypatch.setattr(entry, "create_async_engine", lambda *args, **kwargs: FakeEngine())
    monkeypatch.setattr(entry, "_read_owner_scope", owner_scope)
    monkeypatch.setattr(entry, "_build_context", lambda *args: context)
    monkeypatch.setattr(entry, "_load_existing_master_key", lambda settings: b"x" * 32)
    monkeypatch.setattr(entry, "ProviderProfileService", lambda *a, **k: FakeProfileService())
    monkeypatch.setattr(entry, "resolve_provider_binding", resolve_binding)
    assert entry.main() == 3
    output = capsys.readouterr().err
    diagnostic = json.loads(output.split("PATH_C_SAFE_FAILURE_DIAGNOSTICS ")[1])
    assert diagnostic["failure_category"] == "CREDENTIAL_RESOLUTION"
    assert diagnostic["physical_outbound_count"] == 0
    assert diagnostic["outbound_attempted_at_utc"] is None
    assert SECRET not in output and OWNER not in output
    assert disposed is True
    assert not list(tmp_path.rglob("generation_state.json"))


def test_historical_consumed_attempt_is_not_rewritten_or_enriched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "legacy-attempt"
    out.mkdir()
    original = (
        b'{"state":"CONSUMED_FAILED","failure_code":"MODELGATEWAYERROR",'
        b'"physical_outbound_count":1,"blocked_additional_outbound_count":0}'
    )
    (out / "generation_state.json").write_bytes(original)

    async def no_transport(request: httpx.Request) -> httpx.Response:
        raise AssertionError("historical attempt must not redispatch")

    _, budget = asyncio.run(_pipeline(tmp_path, out, no_transport))
    assert budget.dispatched == 0
    assert (out / "generation_state.json").read_bytes() == original
    assert list(out.iterdir()) == [out / "generation_state.json"]
    canonical = (
        tmp_path
        / "v2_1r_d4r_path_c_generation"
        / (f"{generation.POPULATION_ID}__{generation.BATCH_ID}")
    )
    canonical.mkdir(parents=True)
    (canonical / "generation_state.json").write_bytes(original)
    monkeypatch.setattr(entry, "Settings", lambda: SimpleNamespace(artifact_root=tmp_path))
    assert entry.main() == 2
    assert (canonical / "generation_state.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["snapshot", "state_write"])
def test_diagnostic_or_terminal_write_failure_never_restores_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        raise httpx.ReadTimeout(SECRET, request=request)

    def broken(*args: object, **kwargs: object) -> Any:
        raise OSError(SECRET)

    if failure == "snapshot":
        monkeypatch.setattr(FailureDiagnosticsRecorder, "snapshot", broken)
    else:
        monkeypatch.setattr(generation, "_replace_json", broken)
    asyncio.run(_pipeline(root, out, handler))
    original = (out / "generation_state.json").read_bytes()
    state = json.loads(original)
    assert state["state"] == ("CONSUMED_FAILED" if failure == "snapshot" else "DISPATCH_RESERVED")
    if failure == "snapshot":
        assert state["failure_diagnostics_status"] == "UNAVAILABLE"
    _, budget = asyncio.run(_pipeline(root, out, handler))
    assert admitted == 1 and budget.dispatched == 0
    assert (out / "generation_state.json").read_bytes() == original
    assert SECRET.encode() not in original


def test_untrusted_gateway_fields_and_error_messages_cannot_be_serialized() -> None:
    observations = FailureDiagnosticsRecorder(phase=FailurePhase.GATEWAY)
    exc = ModelGatewayError(
        SECRET,
        retryable=False,
        detail_code=SECRET,
        diagnostics={
            "Authorization": SECRET,
            "owner_hash": OWNER,
            "provider_request_id": SECRET,
        },
    )
    diagnostic = observations.snapshot(
        exc,
        physical_outbound_count=0,
        blocked_additional_outbound_count=0,
    )
    assert SECRET not in diagnostic.model_dump_json()
    assert OWNER not in diagnostic.model_dump_json()
    assert diagnostic.gateway_error_code is None and diagnostic.gateway_detail_code is None
    assert diagnostic.gateway_detail_redacted is True
    with pytest.raises(ValidationError):
        PathCFailureDiagnostics.model_validate({**diagnostic.model_dump(), "api_key": SECRET})


def test_error_body_hash_is_exact_but_unknown_codes_headers_and_body_are_not_persisted(
    tmp_path: Path,
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"
    raw = json.dumps({"error": {"code": SECRET, "type": SECRET, "message": SECRET}}).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=raw, headers={"Set-Cookie": SECRET}, request=request)

    asyncio.run(_pipeline(root, out, handler))
    state = json.loads((out / "generation_state.json").read_bytes())
    diagnostic = state["failure_diagnostics"]
    assert diagnostic["response_body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert diagnostic["response_body_size"] == len(raw)
    assert diagnostic["provider_error_fields_redacted"] is True
    assert diagnostic["safe_provider_error_code"] is None
    assert not (out / "provider_response.raw.json").exists()
    assert SECRET not in json.dumps(state)


def test_success_status_cannot_persist_a_reflected_credential(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"

    async def handler(request: httpx.Request) -> httpx.Response:
        tasks = _tasks()
        tasks[0] = SECRET
        return httpx.Response(200, json=_response(json.dumps({"tasks": tasks})), request=request)

    asyncio.run(_pipeline(root, out, handler))
    state = json.loads((out / "generation_state.json").read_bytes())
    assert state["failure_diagnostics"]["raw_capture_policy"] == "SENSITIVE_RESPONSE_HASH_ONLY"
    assert SECRET not in json.dumps(state)
    assert not (out / "provider_response.raw.json").exists()
    assert not (out / "path_c_task_manifest.json").exists()


def test_http_response_can_be_received_before_a_body_read_network_failure(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "fake-attempt"

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            raise httpx.ReadError(SECRET)
            yield b""  # pragma: no cover -- establishes the async iterator protocol

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenBody(), request=request)

    _, budget = asyncio.run(_pipeline(root, out, handler))
    diagnostic = json.loads((out / "generation_state.json").read_bytes())["failure_diagnostics"]
    assert budget.dispatched == 1
    assert diagnostic["response_received"] is True
    assert diagnostic["provider_http_status"] == 200
    assert diagnostic["response_body_sha256"] is None
    assert diagnostic["gateway_error_code"] == "MODEL_NETWORK_ERROR"
    assert diagnostic["gateway_detail_code"] == "READ_ERROR"
    assert not (out / "provider_response.raw.json").exists()
    assert SECRET not in json.dumps(diagnostic)


def test_provider_controlled_finish_string_cannot_enter_gateway_detail() -> None:
    observations = FailureDiagnosticsRecorder(phase=FailurePhase.GATEWAY)
    exc = ModelGatewayError(
        "MODEL_OUTPUT_INVALID",
        retryable=False,
        detail_code=f"OUTPUT_INVALID_JSON_MODE_FINISH_{SECRET}_PARSE_EMPTY_RESPONSE_CHARS_0",
    )
    diagnostic = observations.snapshot(
        exc,
        physical_outbound_count=1,
        blocked_additional_outbound_count=0,
    )
    assert diagnostic.gateway_detail_code == "STRUCTURED_OUTPUT_FAILURE"
    assert diagnostic.gateway_detail_redacted is True
    assert SECRET not in diagnostic.model_dump_json()
