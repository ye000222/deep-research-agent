"""Synthetic governance grants in temporary directories; real outbound is zero."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from app.domain.providers import AdapterType
from pydantic import SecretStr

from scripts.prospective_acquisition import path_c_generation as generation
from scripts.prospective_acquisition import path_c_generation_entry as entry
from scripts.prospective_acquisition.capture_transport import RawResponseCaptureTransport
from scripts.prospective_acquisition.control import CallBudget, build_guarded_provider_client
from scripts.prospective_acquisition.path_c_attempts import (
    AuthorizedAttempt,
    FrozenAttemptBinding,
    GenerationAttemptError,
    HistoryScope,
    PathCAttemptAuthorization,
    attempt_id_for,
    canonical_bytes,
    digest,
    initialize_attempt,
    read_history,
    resolve_operator_attempt,
)
from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    FailureDiagnosticsRecorder,
    ObservedPathCGateway,
)
from tests.unit.test_path_c_generation import _context, _copy_frozen_root, _tasks

SECRET = "fake-key-never-persist"


def _scope(binding: FrozenAttemptBinding) -> HistoryScope:
    return HistoryScope(**{key: getattr(binding, key) for key in HistoryScope.model_fields})


def _history_fixture(tmp_path: Path) -> tuple[Path, Path, FrozenAttemptBinding, bytes]:
    root = _copy_frozen_root(tmp_path)
    artifact_root = tmp_path / "isolated-artifacts"
    batch = (
        artifact_root
        / "v2_1r_d4r_path_c_generation"
        / (f"{generation.POPULATION_ID}__{generation.BATCH_ID}")
    )
    batch.mkdir(parents=True)
    context, provider = _context()
    binding = generation.frozen_attempt_binding(context, provider)
    # This fixture reflects supplied historical accounting; it fabricates no
    # missing Gateway diagnostics or historical authorization UUID.
    legacy = {
        "state_version": "path-c-generation-state.v1",
        "state": "CONSUMED_FAILED",
        "population_id": generation.POPULATION_ID,
        "batch_id": generation.BATCH_ID,
        "execution_context_id": generation.EXECUTION_CONTEXT_ID,
        "contract_version": generation.GENERATION_CONTRACT_VERSION,
        "dispatch_count": 1,
        "failure_code": "MODELGATEWAYERROR",
        "physical_outbound_count": 1,
        "blocked_additional_outbound_count": 0,
        "reserved_at_utc": "2026-10-12T11:20:50+00:00",
        "failed_at_utc": "2026-10-12T11:21:00+00:00",
        "seed_sent": False,
    }
    raw = canonical_bytes(legacy) + b"\n"
    (batch / "generation_state.json").write_bytes(raw)
    return root, batch, binding, raw


def _synthetic_grant(
    batch: Path,
    binding: FrozenAttemptBinding,
    authorization_id: UUID | None = None,
) -> PathCAttemptAuthorization:
    previous = read_history(batch, _scope(binding))[-1]
    number = previous.number + 1
    authorization_id = authorization_id or uuid4()
    return PathCAttemptAuthorization(
        **binding.model_dump(),
        authorization_id=authorization_id,
        attempt_number=number,
        attempt_id=attempt_id_for(_scope(binding), number, authorization_id),
        previous_attempt_number=previous.number,
        previous_attempt_id=previous.attempt_id,
        previous_state_sha256=previous.state_sha256,
        created_at_utc=datetime.now(UTC),
    ).with_integrity_hash()


def _publish_fake(batch: Path, auth: PathCAttemptAuthorization, **extra: Any) -> Path:
    directory = batch / "authorizations"
    directory.mkdir(exist_ok=True)
    path = directory / f"attempt_{auth.attempt_number:03d}.authorization.json"
    with path.open("xb") as stream:
        stream.write(canonical_bytes({**auth.model_dump(mode="json"), **extra}) + b"\n")
    return path


async def _dispatch(
    root: Path,
    attempt: AuthorizedAttempt,
    handler: Any,
    *,
    yield_preflight: bool = False,
    drift: bool = False,
) -> tuple[generation.GenerationOutcome, CallBudget]:
    context, provider = _context()
    observations = FailureDiagnosticsRecorder()
    budget = CallBudget(1)
    client = build_guarded_provider_client(
        budget,
        transport=RawResponseCaptureTransport(
            httpx.MockTransport(handler),
            attempt.location / "provider_response.raw.json",
            observations,
        ),
    )

    async def recheck() -> None:
        if drift:
            raise RuntimeError(SECRET)
        if yield_preflight:
            await asyncio.sleep(0)

    try:
        outcome = await generation.execute_path_c_generation(
            root=root,
            output_dir=attempt.location,
            context=context,
            provider_binding=provider,
            binding_recheck=recheck,
            gateway=ObservedPathCGateway(client, observations),
            api_key=SecretStr(SECRET),
            base_url="https://api.deepseek.com/v1",
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            context_manifest_id=UUID(generation.EXECUTION_CONTEXT_ID),
            physical_budget=budget,
            raw_response_path=attempt.location / "provider_response.raw.json",
            failure_diagnostics=observations,
            authorized_attempt=attempt,
        )
        return outcome, budget
    finally:
        await client.aclose()


def _assert_legacy(batch: Path, original: bytes) -> None:
    actual = (batch / "generation_state.json").read_bytes()
    assert actual == original and digest(actual) == digest(original)
    assert b"failure_diagnostics" not in actual


def test_legacy_recognized_and_operator_no_grant_stops_before_db_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, batch, binding, original = _history_fixture(tmp_path)
    history = read_history(batch, _scope(binding))
    assert len(history) == 1 and history[0].number == 1
    assert history[0].state == "CONSUMED_FAILED" and history[0].consumed is True
    assert history[0].location == batch and history[0].authorization_id is None
    monkeypatch.setattr(
        entry, "Settings", lambda: SimpleNamespace(artifact_root=batch.parent.parent)
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no-grant entry must not resolve credentials or create a DB engine")

    monkeypatch.setattr(entry, "create_async_engine", forbidden)
    monkeypatch.setattr(entry, "_safe_json", forbidden)
    assert entry.main() == 2
    output = capsys.readouterr().err
    assert "NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT" in output
    assert "PATH_C_PROVIDER_HTTP_COUNT 0" in output
    assert not (batch / "attempt_002").exists()
    assert not (batch / "authorizations").exists()
    _assert_legacy(batch, original)


@pytest.mark.parametrize(
    "case",
    [
        "malformed",
        "missing_field",
        "hash",
        "attempt_id",
        "attempt_id_reused",
        "skipped",
        "parent",
        "parent_hash",
        "previous_state",
        "population",
        "batch",
        "execution_context",
        "context_hash",
        "contract_version",
        "contract_hash",
        "schema_hash",
        "amendment_version",
        "amendment_hash",
        "provider_fingerprint",
        "provider_binding",
        "provider_override",
        "key",
        "owner_hash",
        "model_override",
        "endpoint_override",
        "retry",
        "dispatch_limit",
        "boolean_dispatch_limit",
        "authorization_status",
        "authorization_source",
        "authorization_purpose",
        "duplicate_file",
        "duplicate_number_alias",
    ],
)
def test_invalid_authorization_never_creates_attempt_or_dispatches(
    tmp_path: Path,
    case: str,
) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    auth = _synthetic_grant(batch, binding)
    mutations: dict[str, dict[str, Any]] = {
        "hash": {"authorization_hash": "f" * 64},
        "attempt_id": {"attempt_id": "f" * 64},
        "attempt_id_reused": {"attempt_id": auth.previous_attempt_id},
        "parent": {"previous_attempt_id": "f" * 64},
        "parent_hash": {"previous_state_sha256": "f" * 64},
        "previous_state": {"previous_terminal_state": "SUCCEEDED_AND_FROZEN"},
        "population": {"population_id": "wrong_population"},
        "batch": {"batch_id": "wrong_batch"},
        "execution_context": {"execution_context_id": uuid4()},
        "context_hash": {"execution_context_hash": "f" * 64},
        "contract_version": {"generation_contract_version": "wrong-contract"},
        "contract_hash": {"generation_contract_sha256": "f" * 64},
        "schema_hash": {"task_schema_sha256": "f" * 64},
        "amendment_version": {"seed_amendment_version": "wrong-amendment"},
        "amendment_hash": {"seed_amendment_sha256": "f" * 64},
        "provider_fingerprint": {"provider_safe_config_fingerprint": "f" * 64},
        "provider_binding": {"provider_binding_sha256": "f" * 64},
        "retry": {"retry": 1},
        "dispatch_limit": {"maximum_provider_generation_dispatches": 2},
        "boolean_dispatch_limit": {"maximum_provider_generation_dispatches": True},
        "authorization_status": {"authorization_status": "CONSUMED"},
        "authorization_source": {"authorization_source": "PROFILE_DISCOVERY"},
        "authorization_purpose": {"purpose": "AUTOMATIC_RETRY"},
    }
    if case in mutations:
        auth = auth.model_copy(update=mutations[case])
        if case != "hash":
            auth = auth.with_integrity_hash()
    if case == "skipped":
        auth = auth.model_copy(
            update={
                "attempt_number": 3,
                "previous_attempt_number": 2,
                "attempt_id": attempt_id_for(_scope(binding), 3, auth.authorization_id),
            }
        ).with_integrity_hash()
    extras: dict[str, Any] = {}
    if case in {"provider_override", "key", "owner_hash", "model_override", "endpoint_override"}:
        extras[
            {
                "provider_override": "provider_profile_id",
                "key": "api_key",
                "model_override": "model",
                "endpoint_override": "base_url",
            }.get(case, case)
        ] = SECRET
    path = _publish_fake(batch, auth, **extras)
    if case == "malformed":
        path.write_bytes(b"{")
    if case == "missing_field":
        data = json.loads(path.read_bytes())
        del data["purpose"]
        path.write_bytes(canonical_bytes(data))
    if case == "duplicate_file":
        (path.parent / "copy.authorization.json").write_bytes(path.read_bytes())
    if case == "duplicate_number_alias":
        (path.parent / "attempt_0002.authorization.json").write_bytes(path.read_bytes())
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        raise AssertionError("invalid grant reached Provider transport")

    with pytest.raises(GenerationAttemptError):
        selection = resolve_operator_attempt(batch, _scope(binding))
        assert selection.authorized is not None
        asyncio.run(_dispatch(root, selection.authorized, handler))
    assert admitted == 0
    assert not (batch / "attempt_002").exists()
    assert not (batch / "attempt_003").exists()
    _assert_legacy(batch, original)


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_separate_authorization_one_dispatch_isolation_diagnostics_and_no_next_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    auth = _synthetic_grant(batch, binding)
    grant_path = _publish_fake(batch, auth)
    grant_bytes = grant_path.read_bytes()
    selection = resolve_operator_attempt(batch, _scope(binding))
    assert selection.authorized is not None
    attempt = selection.authorized
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        state = json.loads((attempt.location / "generation_state.json").read_bytes())
        assert state["state"] == "DISPATCH_RESERVED"
        assert state["authorization_id"] == str(auth.authorization_id)
        assert state["attempt_id"] == auth.attempt_id and state["attempt_number"] == 2
        assert (attempt.location / "dispatch_reservation.json").exists()
        assert "seed" not in json.loads(request.content)
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        if outcome == "failure":
            raise httpx.ReadTimeout(SECRET, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps({"tasks": _tasks()})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    if outcome == "failure":
        with pytest.raises(generation.PathCGenerationBlocked):
            asyncio.run(_dispatch(root, attempt, handler))
    else:
        result, budget = asyncio.run(_dispatch(root, attempt, handler))
        assert result.state == "SUCCEEDED_AND_FROZEN" and budget.dispatched == 1
    assert admitted == 1
    state_bytes = (attempt.location / "generation_state.json").read_bytes()
    state = json.loads(state_bytes)
    assert state["state"] == ("SUCCEEDED_AND_FROZEN" if outcome == "success" else "CONSUMED_FAILED")
    assert state["attempt_id"] == auth.attempt_id
    assert state["attempt_number"] == 2
    assert state["authorization_id"] == str(auth.authorization_id)
    if outcome == "failure":
        assert state["failure_diagnostics"]["physical_outbound_count"] == 1
        assert state["failure_diagnostics"]["gateway_detail_code"] == "READ_TIMEOUT"
        assert not (attempt.location / "path_c_task_manifest.json").exists()
    else:
        assert (attempt.location / "provider_response.raw.json").is_file()
        assert (attempt.location / "path_c_task_manifest.json").is_file()
        metadata = json.loads((attempt.location / "generation_metadata.json").read_bytes())
        assert metadata["attempt_id"] == auth.attempt_id
        assert metadata["authorization_id"] == str(auth.authorization_id)
    with pytest.raises(GenerationAttemptError):
        asyncio.run(_dispatch(root, attempt, handler))
    monkeypatch.setattr(
        entry, "Settings", lambda: SimpleNamespace(artifact_root=batch.parent.parent)
    )
    assert entry.main() == (0 if outcome == "success" else 2)
    captured = capsys.readouterr()
    if outcome == "success":
        assert json.loads(captured.out)["provider_http_count"] == 0
    else:
        assert "NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT" in captured.err
    assert admitted == 1
    assert (attempt.location / "generation_state.json").read_bytes() == state_bytes
    assert grant_path.read_bytes() == grant_bytes
    assert not (batch / "attempt_003").exists()
    assert not (batch / "authorizations/attempt_003.authorization.json").exists()
    for file in attempt.location.iterdir():
        assert SECRET.encode() not in file.read_bytes()
    _assert_legacy(batch, original)


def test_initialized_unconsumed_attempt_resumes_same_identity_once(tmp_path: Path) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None
    initial = initialize_attempt(attempt, binding)
    assert initial["state"] == "AUTHORIZED_NOT_DISPATCHED"
    identity_bytes = (attempt.location / "attempt_identity.json").read_bytes()
    resumed = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert resumed == attempt
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        raise httpx.ReadTimeout("fake timeout", request=request)

    with pytest.raises(generation.PathCGenerationBlocked):
        asyncio.run(_dispatch(root, resumed, handler))
    assert admitted == 1
    assert (attempt.location / "attempt_identity.json").read_bytes() == identity_bytes
    _assert_legacy(batch, original)


def test_authorization_cannot_be_reformatted_or_rehashed_after_identity_creation(
    tmp_path: Path,
) -> None:
    _, batch, binding, original = _history_fixture(tmp_path)
    auth = _synthetic_grant(batch, binding)
    path = _publish_fake(batch, auth)
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None
    initialize_attempt(attempt, binding)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(
        GenerationAttemptError, match="AUTHORIZATION_CHANGED_AFTER_ATTEMPT_CREATION"
    ):
        resolve_operator_attempt(batch, _scope(binding))
    _assert_legacy(batch, original)


def test_dispatch_receipt_blocks_resume_if_reservation_state_replacement_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise OSError(SECRET)

    monkeypatch.setattr(generation, "_replace_json", crash)

    async def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("interrupted reservation must not reach HTTP")

    with pytest.raises(generation.PathCGenerationBlocked):
        asyncio.run(_dispatch(root, attempt, forbidden))
    assert json.loads((attempt.location / "generation_state.json").read_bytes())["state"] == (
        "AUTHORIZED_NOT_DISPATCHED"
    )
    assert (attempt.location / "dispatch_reservation.json").is_file()
    with pytest.raises(GenerationAttemptError, match="GENERATION_AUTHORIZATION_ALREADY_CONSUMED"):
        resolve_operator_attempt(batch, _scope(binding))
    _assert_legacy(batch, original)


def test_duplicate_authorization_uuid_cannot_authorize_a_later_attempt(tmp_path: Path) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    auth = _synthetic_grant(batch, binding)
    _publish_fake(batch, auth)
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None

    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fake", request=request)

    with pytest.raises(generation.PathCGenerationBlocked):
        asyncio.run(_dispatch(root, attempt, timeout))
    _publish_fake(batch, _synthetic_grant(batch, binding, auth.authorization_id))
    with pytest.raises(GenerationAttemptError, match="DUPLICATE_ATTEMPT_AUTHORIZATION"):
        resolve_operator_attempt(batch, _scope(binding))
    assert not (batch / "attempt_003").exists()
    _assert_legacy(batch, original)


def test_duplicate_directory_number_and_partial_identity_fail_closed(tmp_path: Path) -> None:
    _, batch, binding, original = _history_fixture(tmp_path)
    alias = batch / "attempt_0002"
    alias.mkdir()
    with pytest.raises(GenerationAttemptError, match="DUPLICATE_ATTEMPT_NUMBER"):
        read_history(batch, _scope(binding))
    # Independent temporary fixture, not a repair/reset of the rejected history.
    second = tmp_path / "independent-fixture"
    second.mkdir()
    _, batch2, binding2, _ = _history_fixture(second)
    (batch2 / "attempt_002").mkdir()
    with pytest.raises(GenerationAttemptError, match="ATTEMPT_ARTIFACT_UNREADABLE"):
        read_history(batch2, _scope(binding2))
    _assert_legacy(batch, original)


def test_concurrent_same_authorization_admits_one_transport(tmp_path: Path) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        await asyncio.sleep(0)
        raise httpx.ReadTimeout("fake", request=request)

    async def run() -> None:
        results = await asyncio.gather(
            _dispatch(root, attempt, handler, yield_preflight=True),
            _dispatch(root, attempt, handler, yield_preflight=True),
            return_exceptions=True,
        )
        assert all(isinstance(result, Exception) for result in results)

    asyncio.run(run())
    assert admitted == 1
    _assert_legacy(batch, original)


def test_public_generation_cannot_be_called_without_authorization(tmp_path: Path) -> None:
    root, batch, _, original = _history_fixture(tmp_path)
    context, provider = _context()

    async def run() -> None:
        with pytest.raises(GenerationAttemptError, match="SEPARATE_ATTEMPT_AUTHORIZATION_REQUIRED"):
            await generation.execute_path_c_generation(
                root=root,
                output_dir=batch / "attempt_002",
                context=context,
                provider_binding=provider,
                binding_recheck=None,
                gateway=cast_fake_gateway(),
                api_key=SecretStr(SECRET),
                base_url="https://api.deepseek.com/v1",
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                context_manifest_id=UUID(generation.EXECUTION_CONTEXT_ID),
                physical_budget=CallBudget(1),
                raw_response_path=batch / "attempt_002/provider_response.raw.json",
            )

    asyncio.run(run())
    assert not (batch / "attempt_002").exists()
    _assert_legacy(batch, original)


def cast_fake_gateway() -> Any:
    return SimpleNamespace()


@pytest.mark.parametrize("failure", ["frozen_contract", "live_provider_drift"])
def test_valid_grant_cannot_create_state_after_contract_or_live_drift_failure(
    tmp_path: Path,
    failure: str,
) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None
    if failure == "frozen_contract":
        path = root / "artifacts/v2_1r_d4r_a51/generic_task_generation_contract.json"
        path.write_bytes(path.read_bytes() + b" ")

    async def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("pre-dispatch failure admitted HTTP")

    with pytest.raises(generation.PathCGenerationBlocked):
        asyncio.run(_dispatch(root, attempt, forbidden, drift=failure == "live_provider_drift"))
    assert not attempt.location.exists()
    _assert_legacy(batch, original)


def test_numbered_attempt_blocks_fallback_before_second_transport(tmp_path: Path) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None
    admitted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        return httpx.Response(400, request=request, json={"error": {"message": SECRET}})

    with pytest.raises(generation.PathCGenerationBlocked):
        asyncio.run(_dispatch(root, attempt, handler))
    state_bytes = (attempt.location / "generation_state.json").read_bytes()
    state = json.loads(state_bytes)
    assert admitted == state["physical_outbound_count"] == 1
    assert state["blocked_additional_outbound_count"] == 1
    assert state["authorization_id"] == str(attempt.authorization.authorization_id)
    assert state["failure_diagnostics"]["gateway_error_code"] == "MODEL_REQUEST_INVALID"
    with pytest.raises(GenerationAttemptError):
        asyncio.run(_dispatch(root, attempt, handler))
    assert admitted == 1
    assert (attempt.location / "generation_state.json").read_bytes() == state_bytes
    _assert_legacy(batch, original)


def test_real_operator_entry_resolves_exact_synthetic_authorization_and_fake_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    auth = _synthetic_grant(batch, binding)
    _publish_fake(batch, auth)
    context, provider = _context()
    admitted = 0
    drift_checks = 0
    disposed = False

    class FakeEngine:
        async def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    class FakeProfileService:
        async def get_test_material(self, owner: str, profile_id: UUID) -> Any:
            assert owner == context.owner_hash and str(profile_id) == provider.profile_id
            return SimpleNamespace(
                api_key=SecretStr(SECRET),
                base_url="https://api.deepseek.com/v1",
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT.value,
            )

    async def read_owner(*args: Any) -> str:
        return context.owner_hash

    async def resolve_provider(*args: Any) -> Any:
        return provider

    async def live_drift_check(*args: Any) -> None:
        nonlocal drift_checks
        drift_checks += 1

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        state = json.loads((batch / "attempt_002/generation_state.json").read_bytes())
        assert state["attempt_id"] == auth.attempt_id
        assert state["authorization_id"] == str(auth.authorization_id)
        assert "seed" not in json.loads(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps({"tasks": _tasks()})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    actual_execute = generation.execute_path_c_generation

    async def execute_with_test_mount(**kwargs: Any) -> generation.GenerationOutcome:
        # Substitute only the /opt frozen-file mount location, not authorization,
        # context validation, generation or budget logic.
        kwargs["root"] = root
        return await actual_execute(**kwargs)

    monkeypatch.setattr(
        entry,
        "Settings",
        lambda: SimpleNamespace(
            artifact_root=batch.parent.parent,
            database_url="fake-offline-database",
            allow_insecure_provider_endpoints=False,
            model_http_proxy=None,
        ),
    )
    monkeypatch.setattr(entry, "_safe_json", lambda path: context.safe_artifact())
    monkeypatch.setattr(entry, "create_async_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(entry, "_read_owner_scope", read_owner)
    monkeypatch.setattr(entry, "_load_existing_master_key", lambda settings: b"x" * 32)
    monkeypatch.setattr(entry, "ProviderProfileService", lambda *a, **k: FakeProfileService())
    monkeypatch.setattr(entry, "resolve_provider_binding", resolve_provider)
    monkeypatch.setattr(entry, "assert_provider_binding_unchanged", live_drift_check)
    monkeypatch.setattr(entry, "execute_path_c_generation", execute_with_test_mount)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler))
    assert entry.main() == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["attempt_number"] == 2 and result["attempt_id"] == auth.attempt_id
    assert result["authorization_id"] == str(auth.authorization_id)
    assert admitted == drift_checks == 1 and disposed is True
    assert SECRET not in output.out + output.err
    assert not (batch / "attempt_003").exists()
    _assert_legacy(batch, original)


@pytest.mark.parametrize("case", ["maximum_two", "parent_exhausted", "wrong_output_directory"])
def test_attempt_scope_and_available_one_call_budget_are_checked_before_creation(
    tmp_path: Path,
    case: str,
) -> None:
    root, batch, binding, original = _history_fixture(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, _scope(binding)).authorized
    assert attempt is not None
    context, provider = _context()
    budget = CallBudget(2) if case == "maximum_two" else CallBudget(1, parent=CallBudget(0))

    async def run() -> None:
        with pytest.raises(GenerationAttemptError):
            await generation.execute_path_c_generation(
                root=root,
                output_dir=batch / "attempt_003"
                if case == "wrong_output_directory"
                else attempt.location,
                context=context,
                provider_binding=provider,
                binding_recheck=None,
                gateway=cast_fake_gateway(),
                api_key=SecretStr(SECRET),
                base_url="unused",
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                context_manifest_id=UUID(generation.EXECUTION_CONTEXT_ID),
                physical_budget=budget,
                raw_response_path=attempt.location / "provider_response.raw.json",
                authorized_attempt=attempt,
            )

    asyncio.run(run())
    assert budget.dispatched == 0
    assert not attempt.location.exists()
    assert not (batch / "attempt_003").exists()
    _assert_legacy(batch, original)
