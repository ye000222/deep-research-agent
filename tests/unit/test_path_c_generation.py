from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from app.domain.providers import (
    AdapterType,
    CanonicalModelRequest,
    CanonicalModelResult,
    ContentPart,
    TokenUsage,
)
from app.llm.adapters import LLMGateway
from pydantic import SecretStr

from scripts.prospective_acquisition.capture_transport import RawResponseCaptureTransport
from scripts.prospective_acquisition.control import CallBudget, build_guarded_provider_client
from scripts.prospective_acquisition.execution_identity import (
    FrozenProviderBinding,
    ProspectiveAcquisitionExecutionContext,
)
from scripts.prospective_acquisition.path_c_generation import (
    EXECUTION_CONTEXT_ID,
    PathCGenerationBlocked,
)
from scripts.prospective_acquisition.path_c_generation import (
    _execute_guarded_core as execute_path_c_generation,
)
from scripts.prospective_acquisition.seed_amendment import resolve_seed_amendment

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "artifacts/v2_1r_d4r_a51"
AMENDMENT_DIR = ROOT / "artifacts/v2_1r_d4r_path_c_seed_amendment"
PROVIDER_FINGERPRINT = "32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2"


def _context() -> tuple[ProspectiveAcquisitionExecutionContext, FrozenProviderBinding]:
    binding = FrozenProviderBinding(
        profile_id="49a9ae5c-b062-4617-805f-dc587b706ce1",
        adapter="openai_compatible_chat",
        model="deepseek-flash",
        endpoint_family="https://api.deepseek.com",
        profile_version=23,
        credential_version_id="499faa00-0000-4000-8000-000000000004",
        credential_version=4,
        credential_present=True,
        fingerprint_contract="provider-safe-config-fingerprint.v1",
        provider_safe_config_fingerprint=PROVIDER_FINGERPRINT,
    )
    context = ProspectiveAcquisitionExecutionContext(
        execution_context_id=EXECUTION_CONTEXT_ID,
        population_id="v2_1r_d4r_prospective_population_001",
        batch_id="v2_1r_d4r_prospective_population_001_batch_001",
        owner_hash="a" * 64,
        owner_scope_source="AUTHENTICATED_CLIENT_SESSION",
        provider_resolution_policy="FROZEN_DEFAULT_PROFILE_V1",
        protocol_version="prospective-development-population.v1",
        created_at="2026-10-12T08:55:13Z",
        provider_binding=binding,
        context_hash="",
    )
    context = context.with_provider_binding(binding)
    return context, binding


def _copy_frozen_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "artifacts").mkdir(parents=True)
    shutil.copytree(BASE_DIR, root / "artifacts/v2_1r_d4r_a51")
    shutil.copytree(AMENDMENT_DIR, root / "artifacts/v2_1r_d4r_path_c_seed_amendment")
    (root / "scripts/prospective_acquisition").mkdir(parents=True)
    shutil.copy2(
        ROOT / "scripts/prospective_acquisition/seed_amendment.py",
        root / "scripts/prospective_acquisition/seed_amendment.py",
    )
    for filename in (
        "path_c_generation.py",
        "path_c_generation_entry.py",
        "capture_transport.py",
        "path_c_failure_diagnostics.py",
        "path_c_attempts.py",
        "control.py",
    ):
        shutil.copy2(
            ROOT / "scripts/prospective_acquisition" / filename,
            root / "scripts/prospective_acquisition" / filename,
        )
    api_adapter = root / "apps/api/app/llm/adapters.py"
    api_adapter.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "apps/api/app/llm/adapters.py", api_adapter)
    (root / "tests/unit").mkdir(parents=True)
    shutil.copy2(
        ROOT / "tests/unit/test_path_c_seed_contract_amendment.py",
        root / "tests/unit/test_path_c_seed_contract_amendment.py",
    )
    return root


def _raw_response(tasks: list[str]) -> bytes:
    payload = {
        "id": "fake-request-id",
        "choices": [
            {"message": {"content": json.dumps({"tasks": tasks})}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 300, "total_tokens": 310},
    }
    return json.dumps(payload, separators=(",", ":")).encode()


class FakeResult(CanonicalModelResult):
    def __init__(self, parsed: object) -> None:
        super().__init__(
            parsed_object=parsed if isinstance(parsed, dict) else None,
            text=json.dumps(parsed),
            usage=TokenUsage(
                input_tokens=10, output_tokens=300, total_tokens=310, accuracy="exact"
            ),
            provider_request_id="fake-request-id",
            finish_reason="stop",
        )


class FakeGateway:
    def __init__(
        self, raw_path: Path, *, parsed: object | None = None, error: Exception | None = None
    ) -> None:
        self.raw_path = raw_path
        self.parsed = parsed
        self.error = error
        self.calls = 0
        self.request: CanonicalModelRequest | None = None

    async def generate_structured(self, **kwargs: Any) -> FakeResult:
        self.calls += 1
        self.request = kwargs["request"]
        reserved = json.loads((self.raw_path.parent / "generation_state.json").read_text())
        assert reserved["state"] == "DISPATCH_RESERVED"
        assert reserved["dispatch_count"] == 1
        await asyncio.sleep(0)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.error, ValueError):
            self.raw_path.write_bytes(b"{")
            raise self.error
        if self.error:
            raise self.error
        assert isinstance(self.parsed, dict)
        self.raw_path.write_bytes(_raw_response(self.parsed["tasks"]))
        budget = kwargs["request"].metadata.get("test_budget")
        del budget
        return FakeResult(self.parsed)


class FakeBudget:
    def __init__(self, dispatched: int = 0, blocked_attempts: int = 0) -> None:
        self.dispatched = dispatched
        self.blocked_attempts = blocked_attempts


def _tasks(count: int = 40) -> list[str]:
    return [
        f"Investigate an open research question about topic {index} and compare available evidence."
        for index in range(count)
    ]


async def _execute(
    root: Path,
    out: Path,
    gateway: FakeGateway,
    *,
    budget: FakeBudget | None = None,
    binding_ok: bool = True,
    application_root: Path | None = None,
) -> Any:
    context, binding = _context()

    async def recheck() -> None:
        if not binding_ok:
            raise RuntimeError("binding changed")

    return await execute_path_c_generation(
        root=root,
        application_root=application_root,
        output_dir=out,
        context=context,
        provider_binding=binding,
        binding_recheck=recheck,
        gateway=gateway,
        api_key=SecretStr("fake-never-emitted"),
        base_url="https://api.deepseek.com",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        context_manifest_id=UUID(EXECUTION_CONTEXT_ID),
        physical_budget=budget or FakeBudget(1),
        raw_response_path=out / "provider_response.raw.json",
    )


def test_valid_success_freezes_once_and_omits_seed(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})
    budget = FakeBudget(1)
    result = asyncio.run(_execute(root, out, gateway, budget=budget))
    assert result.state == "SUCCEEDED_AND_FROZEN"
    assert result.seed_capability == "SEED_SUPPORT_NOT_ESTABLISHED"
    assert result.seed_sent is False and result.deterministic_generation_claim is False
    assert gateway.calls == 1 and budget.dispatched == 1
    assert gateway.request is not None
    request_dump = gateway.request.model_dump(mode="json")
    assert "seed" not in json.dumps(request_dump).casefold()
    assert json.loads((out / "path_c_task_manifest.json").read_text())["task_count"] == 40
    assert (out / "provider_response.raw.json").is_file()
    metadata = json.loads((out / "generation_metadata.json").read_text())
    assert metadata["search_count"] == 0
    assert metadata["reader_count"] == 0
    assert metadata["evidence_extractor_count"] == 0
    assert metadata["database_writes"] == 0
    assert metadata["holdout_access"] == 0
    serialized_outputs = " ".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in out.iterdir()
        if path.is_file()
    )
    assert "a" * 64 not in serialized_outputs
    assert "fake-never-emitted" not in serialized_outputs
    state = json.loads((out / "generation_state.json").read_text())
    assert state["state"] == "SUCCEEDED_AND_FROZEN"


def test_provider_drift_blocks_before_dispatch(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})
    with pytest.raises(PathCGenerationBlocked, match="EXECUTION_PROVIDER_BINDING_CHANGED"):
        asyncio.run(_execute(root, out, gateway, binding_ok=False))
    assert gateway.calls == 0
    assert not (out / "generation_state.json").exists()


@pytest.mark.parametrize(
    "mutate,code",
    [
        ("base", "BASE_BUNDLE_FILE_HASH_MISMATCH"),
        ("amendment", "AMENDMENT_HASH_MISMATCH"),
        ("schema", "BASE_BUNDLE_FILE_HASH_MISMATCH"),
    ],
)
def test_frozen_hash_mismatches_block_before_dispatch(
    tmp_path: Path, mutate: str, code: str
) -> None:
    root = _copy_frozen_root(tmp_path)
    if mutate == "base":
        path = root / "artifacts/v2_1r_d4r_a51/generic_task_generation_contract.json"
    elif mutate == "schema":
        path = root / "artifacts/v2_1r_d4r_a51/task_manifest_schema.json"
    else:
        path = root / "artifacts/v2_1r_d4r_path_c_seed_amendment/seed_policy_amendment.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})
    with pytest.raises(PathCGenerationBlocked, match=code):
        asyncio.run(_execute(root, out, gateway))
    assert gateway.calls == 0


@pytest.mark.parametrize(
    "relative",
    [
        "apps/api/app/llm/adapters.py",
        "scripts/prospective_acquisition/schema_validation.py",
    ],
)
def test_missing_output_validation_runtime_file_blocks_before_reservation(
    tmp_path: Path, relative: str
) -> None:
    root = _copy_frozen_root(tmp_path)
    (root / relative).unlink()
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})

    with pytest.raises(PathCGenerationBlocked, match="GENERATION_RUNTIME_FILE_UNAVAILABLE"):
        asyncio.run(_execute(root, out, gateway))

    assert gateway.calls == 0
    assert not (out / "generation_state.json").exists()


def test_generation_resolves_api_source_from_canonical_application_mount(
    tmp_path: Path,
) -> None:
    root = _copy_frozen_root(tmp_path)
    (root / "apps/api/app/llm/adapters.py").unlink()
    application_root = tmp_path / "container-app"
    mounted_adapter = application_root / "apps/api/app/llm/adapters.py"
    mounted_adapter.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "apps/api/app/llm/adapters.py", mounted_adapter)
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})

    result = asyncio.run(_execute(root, out, gateway, application_root=application_root))

    metadata = json.loads((out / "generation_metadata.json").read_text(encoding="utf-8"))
    assert result.state == "SUCCEEDED_AND_FROZEN"
    assert gateway.calls == 1
    assert metadata["implementation_sha256"]["apps/api/app/llm/adapters.py"] == (
        hashlib.sha256(mounted_adapter.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("kind", ["wrong_count", "malformed", "schema_invalid"])
def test_invalid_provider_output_is_consumed_without_manifest_or_retry(
    tmp_path: Path, kind: str
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    if kind == "wrong_count":
        parsed: object = {"tasks": _tasks(39)}
    elif kind == "schema_invalid":
        parsed = {"tasks": _tasks(), "unexpected": True}
    else:
        parsed = None
    gateway = FakeGateway(out / "provider_response.raw.json", parsed=parsed)
    if kind == "malformed":
        gateway.error = ValueError("invalid JSON")
    budget = FakeBudget(1)
    with pytest.raises(PathCGenerationBlocked):
        asyncio.run(_execute(root, out, gateway, budget=budget))
    assert gateway.calls == 1 and budget.dispatched == 1
    assert (out / "provider_response.raw.json").exists()
    assert not (out / "path_c_task_manifest.json").exists()
    assert json.loads((out / "generation_state.json").read_text())["state"] == "CONSUMED_FAILED"
    with pytest.raises(PathCGenerationBlocked, match="GENERATION_AUTHORIZATION_ALREADY_CONSUMED"):
        asyncio.run(_execute(root, out, gateway, budget=budget))
    assert gateway.calls == 1


def test_transport_failure_is_reserved_and_never_retried(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", error=OSError("offline"))
    budget = FakeBudget(1)
    with pytest.raises(PathCGenerationBlocked):
        asyncio.run(_execute(root, out, gateway, budget=budget))
    state = json.loads((out / "generation_state.json").read_text())
    assert state["state"] == "CONSUMED_FAILED" and state["dispatch_count"] == 1
    with pytest.raises(PathCGenerationBlocked):
        asyncio.run(_execute(root, out, gateway, budget=budget))
    assert gateway.calls == 1


def test_rerun_after_success_returns_frozen_manifest_without_dispatch(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})
    first = asyncio.run(_execute(root, out, gateway))
    second = asyncio.run(_execute(root, out, gateway))
    assert first.manifest_sha256 == second.manifest_sha256
    assert second.reused_existing is True
    assert gateway.calls == 1


def test_preexisting_dispatch_reserved_blocks_without_dispatch(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    out.mkdir()
    (out / "generation_state.json").write_text('{"state":"DISPATCH_RESERVED"}', encoding="utf-8")
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})
    with pytest.raises(PathCGenerationBlocked, match="GENERATION_AUTHORIZATION_ALREADY_CONSUMED"):
        asyncio.run(_execute(root, out, gateway))
    assert gateway.calls == 0


def test_concurrent_operator_entries_share_one_durable_reservation(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    gateway = FakeGateway(out / "provider_response.raw.json", parsed={"tasks": _tasks()})

    async def recheck_factory() -> None:
        results = await asyncio.gather(
            _execute(root, out, gateway),
            _execute(root, out, gateway),
            return_exceptions=True,
        )
        assert sum(not isinstance(item, BaseException) for item in results) == 1
        assert sum(isinstance(item, PathCGenerationBlocked) for item in results) == 1

    asyncio.run(recheck_factory())
    assert gateway.calls == 1
    assert (
        json.loads((out / "generation_state.json").read_text())["state"] == "SUCCEEDED_AND_FROZEN"
    )


def test_seed_resolver_is_not_dispatch_authority() -> None:
    result = resolve_seed_amendment(
        BASE_DIR / "generic_task_generation_contract.json",
        AMENDMENT_DIR / "seed_policy_amendment.json",
        "SEED_SUPPORT_NOT_ESTABLISHED",
    )
    assert result.send_seed is False
    assert result.deterministic_generation_claim is False
    assert result.provider_dispatch_authorized is False


def test_operator_generation_request_omits_seed_on_real_gateway_path(tmp_path: Path) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / "generation"
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        response = {
            "id": "offline-id",
            "choices": [
                {"message": {"content": json.dumps({"tasks": _tasks()})}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 40, "total_tokens": 41},
        }
        return httpx.Response(200, json=response, request=request)

    async def run() -> None:
        budget = CallBudget(maximum=1)
        inner = httpx.MockTransport(handler)
        capture = RawResponseCaptureTransport(inner, out / "provider_response.raw.json")
        client = build_guarded_provider_client(budget, transport=capture)
        try:
            context, binding = _context()

            async def recheck() -> None:
                return None

            result = await execute_path_c_generation(
                root=root,
                output_dir=out,
                context=context,
                provider_binding=binding,
                binding_recheck=recheck,
                gateway=LLMGateway(client),
                api_key=SecretStr("never-print-this"),
                base_url="https://api.deepseek.com/v1",
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                context_manifest_id=UUID(EXECUTION_CONTEXT_ID),
                physical_budget=budget,
                raw_response_path=out / "provider_response.raw.json",
            )
            assert result.state == "SUCCEEDED_AND_FROZEN"
        finally:
            await client.aclose()
        assert budget.dispatched == 1 and budget.blocked_attempts == 0

    asyncio.run(run())
    assert len(captured) == 1
    assert "seed" not in captured[0]
    assert captured[0]["response_format"] == {"type": "json_object"}


def test_raw_provider_response_is_frozen_before_gateway_returns(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.response"
    provider_body = {
        "id": "safe-id",
        "choices": [
            {"message": {"content": json.dumps({"tasks": _tasks()})}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 40, "total_tokens": 41},
    }
    raw_bytes = json.dumps(provider_body, separators=(",", ":")).encode()
    observed = False

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_bytes, request=request)

    async def run() -> None:
        nonlocal observed
        budget = CallBudget(maximum=1)
        inner = httpx.MockTransport(handler)
        capture = RawResponseCaptureTransport(inner, raw_path)
        client = build_guarded_provider_client(budget, transport=capture)
        try:
            request = CanonicalModelRequest(
                task_kind="path_c_generic_task_generation",
                role="research_task_generation",
                model="deepseek-flash",
                instructions="Generate exactly 40 tasks.",
                content_parts=(ContentPart(kind="text", value="Generate generic tasks."),),
                response_contract={"type": "object", "properties": {"tasks": {"type": "array"}}},
                generation_parameters={"temperature": 0},
                max_output_tokens=8192,
                stream=False,
                context_manifest_id=UUID(EXECUTION_CONTEXT_ID),
            )
            result = await LLMGateway(client).generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://api.deepseek.com/v1",
                api_key=SecretStr("never-print-this"),
                request=request,
                allow_regeneration=False,
            )
            observed = raw_path.is_file() and raw_path.read_bytes() == raw_bytes
            assert result.parsed_object is not None
        finally:
            await client.aclose()
        assert observed is True
        assert budget.dispatched == 1

    asyncio.run(run())


def test_gateway_fallback_is_blocked_before_second_http_transport(tmp_path: Path) -> None:
    outbound_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal outbound_count
        outbound_count += 1
        return httpx.Response(
            400, json={"error": {"message": "response format rejected"}}, request=request
        )

    async def run() -> tuple[CallBudget, BaseException | None]:
        budget = CallBudget(maximum=1)
        client = build_guarded_provider_client(budget, transport=httpx.MockTransport(handler))
        error: BaseException | None = None
        try:
            request = CanonicalModelRequest(
                task_kind="path_c_generic_task_generation",
                role="research_task_generation",
                model="deepseek-flash",
                instructions="Generate exactly 40 tasks.",
                content_parts=(ContentPart(kind="text", value="Generate generic tasks."),),
                response_contract={"type": "object", "properties": {"tasks": {"type": "array"}}},
                generation_parameters={"temperature": 0},
                max_output_tokens=8192,
                stream=False,
                context_manifest_id=UUID(EXECUTION_CONTEXT_ID),
            )
            await LLMGateway(client).generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://api.deepseek.com/v1",
                api_key=SecretStr("never-print-this"),
                request=request,
                allow_regeneration=False,
            )
        except BaseException as exc:
            error = exc
        finally:
            await client.aclose()
        return budget, error

    budget, error = asyncio.run(run())
    assert error is not None
    assert outbound_count == 1
    assert budget.dispatched == 1
    assert budget.blocked_attempts == 1
