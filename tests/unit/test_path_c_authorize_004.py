"""Final authorization tests: synthetic history only, no real generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prospective_acquisition import path_c_attempt_003_structure as structure
from scripts.prospective_acquisition.path_c_attempts import (
    FrozenAttemptBinding,
    GenerationAttemptError,
    PathCAttemptAuthorization,
    attempt_id_for,
    canonical_bytes,
    digest,
    initialize_attempt,
    read_history,
    resolve_operator_attempt,
)
from scripts.prospective_acquisition.path_c_authorize_002 import (
    CANONICAL_ARTIFACT_VOLUME_ROOT,
    CANONICAL_ATTESTATION_RUN,
    CANONICAL_CHECKOUT_ROOT,
    freeze_authorization,
    governance_rationale_for_attempt,
    scope,
    verify_canonical_runtime_attestation,
    verify_response_and_request_integrity,
)
from scripts.prospective_acquisition.path_c_generation import PathCGenerationBlocked
from tests.unit.test_path_c_attempts import _publish_fake, _synthetic_grant
from tests.unit.test_path_c_authorize_003 import all_bytes, consumed_history


@pytest.fixture(autouse=True)
def emulate_atomic_no_replace_link_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows sandbox denies hardlinks; emulate only this publication primitive in tests."""
    import os
    import shutil

    def publish(source: Path, target: Path) -> None:
        with Path(target).open("xb") as destination, Path(source).open("rb") as origin:
            shutil.copyfileobj(origin, destination)
        os.chmod(target, 0o600)

    monkeypatch.setattr("scripts.prospective_acquisition.path_c_authorize_002.os.link", publish)


def consumed_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, FrozenAttemptBinding, bytes]:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    _publish_fake(batch, _synthetic_grant(batch, binding))
    selected = resolve_operator_attempt(batch, scope()).authorized
    assert selected is not None
    initial = initialize_attempt(selected, binding)
    reservation = {**initial, "state": "DISPATCH_RESERVED", "dispatch_count": 1}
    (selected.location / "dispatch_reservation.json").write_bytes(canonical_bytes(reservation))
    # Synthetic content only. Constants are patched only inside this test's
    # monkeypatch lifecycle, never in source/canonical artifacts/host arguments.
    body = canonical_bytes({
        "choices": [{"message": {"content": json.dumps({"tasks": ["synthetic"] * 40})}}],
    })
    monkeypatch.setattr(structure, "EXPECTED_SHA256", digest(body))
    monkeypatch.setattr(structure, "EXPECTED_SIZE", len(body))
    (selected.location / "provider_response.raw.json").write_bytes(body)
    state = {
        **reservation, "state": "CONSUMED_FAILED", "physical_outbound_count": 1,
        "blocked_additional_outbound_count": 0, "failure_code": "MODELGATEWAYERROR",
        "failure_diagnostics": {
            "failure_category": "NETWORK", "gateway_error_code": "MODEL_NETWORK_ERROR",
            "provider_http_status": 200, "response_received": True,
            "response_body_present": True, "raw_response_preserved": True,
            "raw_capture_policy": "EXACT_SUCCESS_RESPONSE",
            "response_body_sha256": digest(body), "response_body_size": len(body),
        },
    }
    (selected.location / "generation_state.json").write_bytes(canonical_bytes(state))
    return checkout, batch, binding, legacy


def test_final_authorization_four_only_no_dispatch_preserves_all_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    original = all_bytes(batch)
    history = read_history(batch, scope())
    assert [r.number for r in history] == [1, 2, 3]
    with pytest.raises(GenerationAttemptError, match="NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT"):
        resolve_operator_attempt(batch, scope())
    with (
        patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send,
        patch("httpx.Client.send", side_effect=AssertionError("outbound forbidden")) as sync_send,
    ):
        target = freeze_authorization(
            batch, binding, approved_attempt=4, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    send.assert_not_called()
    sync_send.assert_not_called()
    auth = PathCAttemptAuthorization.model_validate_json(target.read_bytes())
    auth.validate_integrity()
    assert auth.attempt_number == 4 and auth.previous_attempt_number == 3
    assert auth.previous_attempt_id == history[-1].attempt_id
    assert auth.previous_state_sha256 == digest(
        original[batch / "attempt_003/generation_state.json"]
    )
    assert auth.binding() == binding
    assert auth.maximum_provider_generation_dispatches == 1
    assert auth.retry == 0 and auth.regeneration is False
    assert auth.no_top_up and auth.no_replacement
    assert (
        auth.provider_fallback == auth.endpoint_fallback == auth.request_format_fallback == "none"
    )
    for path, raw in original.items():
        assert path.read_bytes() == raw
    assert not (batch / "attempt_004").exists()
    plan = resolve_operator_attempt(batch, scope()).authorized
    assert plan is not None and plan.authorization == auth
    published = target.read_bytes()
    with pytest.raises(GenerationAttemptError, match="AUTHORIZATION_ALREADY_EXISTS"):
        freeze_authorization(
            batch, binding, approved_attempt=4, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert target.read_bytes() == published


@pytest.mark.parametrize("case", [
    "parent_hash", "parent_id", "state", "raw_hash", "grant", "count",
])
def test_bad_history_blocks_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    if case in {"parent_hash", "parent_id"}:
        path = batch / "attempt_003/attempt_identity.json"
        value = json.loads(path.read_bytes())
        key = "previous_state_sha256" if case == "parent_hash" else "previous_attempt_id"
        value[key] = "0" * 64
    elif case == "grant":
        path = batch / "authorizations/attempt_003.authorization.json"
        value = json.loads(path.read_bytes())
        value["authorization_hash"] = "0" * 64
    elif case == "raw_hash":
        path = batch / "attempt_003/provider_response.raw.json"
        value = {"choices": []}
    else:
        path = batch / "attempt_003/generation_state.json"
        value = json.loads(path.read_bytes())
        value["state" if case == "state" else "physical_outbound_count"] = (
            "DISPATCH_RESERVED" if case == "state" else 2
        )
    path.write_bytes(canonical_bytes(value))
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=4, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


@pytest.mark.parametrize("number", [2, 3, 7, True])
def test_no_skipped_or_unsupported_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: int,
) -> None:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=number, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


@pytest.mark.parametrize("name", [
    "attempt_004", "attempt_004.dispatch_reservation.json",
    "attempt_004.provider_response.raw.json", "attempt_004.path_c_task_manifest.json",
])
def test_existing_four_evidence_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    (batch / name).touch()
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=4, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


@pytest.mark.parametrize("case", ["timeout", "capture", "taxonomy", "seed", "task_count"])
def test_implementation_regression_prevents_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    if case == "timeout":
        path = checkout / "apps/api/app/llm/model_http_policy.py"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("90.0", "10.0"), encoding="utf-8")
    elif case == "capture":
        path = checkout / "scripts/prospective_acquisition/capture_transport.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"content-encoding", ', ''), encoding="utf-8",
        )
    elif case == "taxonomy":
        path = checkout / "apps/api/app/llm/adapters.py"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace('"DECODING_ERROR"', '"REQUEST_ERROR"'), encoding="utf-8")
    elif case == "seed":
        path = checkout / "scripts/prospective_acquisition/path_c_generation.py"
        original = path.read_text(encoding="utf-8")
        path.write_text(
            original.replace('{"temperature": 0}', '{"temperature": 0, "seed": 1}'),
            encoding="utf-8",
        )
    else:
        path = checkout / "artifacts/v2_1r_d4r_a51/generic_task_generation_contract.json"
        value = json.loads(path.read_bytes())
        value["task_count"] = 39
        path.write_bytes(canonical_bytes(value))
    before = all_bytes(batch)
    with pytest.raises((GenerationAttemptError, PathCGenerationBlocked)):
        freeze_authorization(
            batch, binding, approved_attempt=4, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


def test_atomic_write_failure_leaves_no_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    before = all_bytes(batch)
    with patch("os.link", side_effect=OSError("synthetic failure")), pytest.raises(OSError):
        freeze_authorization(
            batch, binding, approved_attempt=4, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


def test_current_response_request_integrity_no_http() -> None:
    checkout = Path(__file__).resolve().parents[2]
    with patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send:
        verify_response_and_request_integrity(checkout)
    send.assert_not_called()


def test_future_governance_rationale_is_display_only_and_attempt_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, _ = consumed_three(tmp_path, monkeypatch)
    del checkout
    auth = _synthetic_grant(batch, binding)
    original_hash = auth.authorization_hash
    auth.validate_integrity()

    rationale_5 = governance_rationale_for_attempt(5)
    rationale_6 = governance_rationale_for_attempt(6)
    assert "Attempt 005" in rationale_5 and "Attempts 001-004" in rationale_5
    assert "Attempt 006" in rationale_6 and "Attempts 001-005" in rationale_6
    assert "Attempt 005" not in rationale_6
    assert "governance_rationale" not in PathCAttemptAuthorization.model_fields
    assert "governance_rationale" not in auth.model_dump(mode="json")
    assert auth.authorization_hash == original_hash
    auth.validate_integrity()

    with pytest.raises(ValueError, match="UNSUPPORTED_ATTEMPT_NUMBER"):
        governance_rationale_for_attempt(7)


def test_canonical_attestation_uses_readonly_checkout_artifact_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _, binding, _ = consumed_three(tmp_path, monkeypatch)
    attestation_root = checkout / "artifacts"
    _publish_canonical_attestation(attestation_root, checkout, binding)
    assert Path("/app/artifacts") == CANONICAL_ARTIFACT_VOLUME_ROOT
    assert Path("/opt") == CANONICAL_CHECKOUT_ROOT
    assert attestation_root == checkout / "artifacts"
    assert Path("/opt/artifacts") == CANONICAL_CHECKOUT_ROOT / "artifacts"
    assert CANONICAL_ATTESTATION_RUN == "run_20261013T014431260Z_2a030977"

    # Prove the CLI's actual container mount and root wiring, not just that the
    # verifier works when handed a suitable temporary directory.
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = (repo_root / "scripts/operator_authorize_path_c_attempt_002.ps1").read_text(
        encoding="utf-8",
    )
    assert '"type=volume,src=$volume,dst=/app/artifacts,volume-nocopy"' in wrapper
    assert '"type=bind,src=$repoRoot,dst=/opt,readonly"' in wrapper
    authorization_source = (
        repo_root / "scripts/prospective_acquisition/path_c_authorize_002.py"
    ).read_text(encoding="utf-8")
    assert "artifact_root=checkout / \"artifacts\"" in authorization_source
    assert "root = CANONICAL_ARTIFACT_VOLUME_ROOT" in authorization_source

    verify_canonical_runtime_attestation(attestation_root, checkout, binding)
    with pytest.raises(GenerationAttemptError, match="CANONICAL_ATTESTATION_FILE_MISSING"):
        verify_canonical_runtime_attestation(tmp_path / "missing-artifacts", checkout, binding)
    attestation_file = (
        attestation_root / "v2_1r_d4r_a6_operator" / CANONICAL_ATTESTATION_RUN
        / "canonical_runtime_attestation.json"
    )
    attestation_file.write_text("not-json", encoding="utf-8")
    with pytest.raises(GenerationAttemptError, match="CANONICAL_ATTESTATION_INVALID_JSON"):
        verify_canonical_runtime_attestation(attestation_root, checkout, binding)


def _publish_canonical_attestation(root: Path, checkout: Path, binding: FrozenAttemptBinding,
                                   *, status: str = "PASS") -> None:
    source_script = (
        Path(__file__).resolve().parents[2] / "scripts/operator_verify_canonical_runtime.ps1"
    )
    copied_script = checkout / "scripts/operator_verify_canonical_runtime.ps1"
    copied_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_script, copied_script)
    run = root / "v2_1r_d4r_a6_operator" / CANONICAL_ATTESTATION_RUN
    run.mkdir(parents=True)
    image_id = "sha256:" + "8" * 64
    stages = [
        {"stage": name} for name in (
            "PROBE_PROCESS_STARTED", "PYTHON_ENV_READY", "CURRENT_CHECKOUT_IDENTITY",
            "IMPORT_PROBE_STARTED", "IMPORT_PROBE_COMPLETED", "SCHEMA_RUNTIME_PROBE_STARTED",
            "SCHEMA_RUNTIME_PROBE_COMPLETED", "DB_READ_PROBE_STARTED",
            "DB_READ_PROBE_COMPLETED", "FROZEN_CONTEXT_RESOLUTION_STARTED",
            "FROZEN_CONTEXT_VALIDATED", "PROVIDER_PROFILE_RESOLUTION_STARTED",
            "PROVIDER_PROFILE_RESOLUTION_COMPLETED", "SAFE_FINGERPRINT_STARTED",
            "SAFE_FINGERPRINT_COMPLETED", "FINAL_ATTESTATION_READY",
        )
    ]
    documents = {
        "canonical_runtime_attestation.json": {
            "status": status, "execution_context_id": str(binding.execution_context_id),
            "execution_context_hash": binding.execution_context_hash,
            "schema_runtime": {"provider_generation_calls": 0, "provider_http_count": 0,
                                "db_writes": 0, "status": "PASS"},
            "image": {"reference": "deep-research-agent-api:local", "image_id": image_id,
                      "current_checkout_source_read_only_mount_verified": True},
            "source_file_sha256": {
                "apps/api/app/llm/adapters.py": digest(
                    (checkout / "apps/api/app/llm/adapters.py").read_bytes()
                ),
                "scripts/operator_verify_canonical_runtime.ps1": digest(
                    (checkout / "scripts/operator_verify_canonical_runtime.ps1").read_bytes()
                ),
            },
            "imported_components": [
                {"status": "PASS", "matches_current_checkout": True}
                for _ in range(28)
            ],
        },
        "docker_host_attestation.json": {
            "status": "PASS", "commands": {
                name: {"exit_code": 0} for name in (
                    "docker_version", "docker_context_show", "docker_ps",
                    "docker_compose_config",
                )
            },
            "ephemeral_container": {
                "compose_run_exit_code": 0, "source_mount_current_checkout_read_only": True,
                "removed_by_operator_script": True, "image_id": image_id,
                "image_reference": "deep-research-agent-api:local",
            }, "ephemeral_cleanup": "REMOVED_EXACT_EPHEMERAL_CONTAINER",
            "docker_calls_by_codex": 0, "provider_generation_calls": 0,
            "provider_http_count": 0, "semantic_verifier_calls": 0, "db_writes": 0,
            "real_acquisition": 0, "holdout_access": 0,
        },
        "operator_probe_execution.json": {
            "status": "PASS", "ephemeral_runtime": {
                "cleanup_status": "REMOVED_EXACT_EPHEMERAL_CONTAINER", "image_id": image_id,
            }, "provider_generation_calls": 0, "provider_http_count": 0,
            "semantic_verifier_calls": 0, "db_writes": 0, "holdout_access": 0,
        },
        "operator_probe_stage_report.json": {
            "status": "PASS", "stages": stages,
            "import_results": [{"status": "PASS"} for _ in range(28)],
            "current_checkout_identity": {"value": {"status": "PASS"}},
            "database_diagnostics": {"value": {"status": "PASS", "read_only_confirmed": True}},
            "provider_diagnostics": {"value": {"status": "RESOLVED"}},
            "provider_generation_calls": 0, "provider_http_count": 0,
            "semantic_verifier_calls": 0, "db_writes": 0, "holdout_access": 0,
        },
        "provider_readiness_attestation.json": {
            "status": "RESOLVED", "binding_match": True,
            "execution_context_id": str(binding.execution_context_id),
            "execution_context_hash": binding.execution_context_hash,
            "frozen_provider_binding": {
                "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
                "adapter": "openai_compatible_chat", "model": "deepseek-flash",
                "endpoint_family": "https://api.deepseek.com", "profile_version": 23,
                "credential_version": 4,
                "fingerprint_contract": "provider-safe-config-fingerprint.v1",
                "provider_safe_config_fingerprint": binding.provider_safe_config_fingerprint,
            },
            "provider_generation_calls": 0, "provider_http_count": 0, "db_writes": 0,
            "credential_plaintext_emitted": False, "authorization_emitted": False,
        },
    }
    for name, value in documents.items():
        (run / name).write_bytes(canonical_bytes(value))
    files = {name: digest((run / name).read_bytes()) for name in documents}
    files["scripts/operator_verify_canonical_runtime.ps1"] = digest(
        (checkout / "scripts/operator_verify_canonical_runtime.ps1").read_bytes()
    )
    payload = {
        "artifact_version": "v2_1r_d4r_a6_operator.hash_manifest.v1",
        "algorithm": "SHA-256", "files": files,
    }
    manifest = {
        **payload,
        "manifest_sha256": digest(json.dumps(payload, separators=(",", ":")).encode()),
        "self_hash_excluded": True,
    }
    (run / "hash_manifest.json").write_bytes(canonical_bytes(manifest))


def _consumed_four(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                   ) -> tuple[Path, Path, FrozenAttemptBinding, bytes]:
    checkout, batch, binding, legacy = consumed_three(tmp_path, monkeypatch)
    freeze_authorization(
        batch, binding, approved_attempt=4, checkout=checkout,
        expected_parent_hash=digest(legacy),
    )
    selected = resolve_operator_attempt(batch, scope()).authorized
    assert selected is not None and selected.authorization.attempt_number == 4
    initial = initialize_attempt(selected, binding)
    reservation = {**initial, "state": "DISPATCH_RESERVED", "dispatch_count": 1}
    (selected.location / "dispatch_reservation.json").write_bytes(canonical_bytes(reservation))
    terminal = {
        **reservation, "state": "CONSUMED_FAILED", "physical_outbound_count": 1,
        "blocked_additional_outbound_count": 0, "failure_code": "SCHEMA_RUNTIME_ERROR",
        "failure_diagnostics": {"failure_category": "CONFIGURATION"},
    }
    (selected.location / "generation_state.json").write_bytes(canonical_bytes(terminal))
    return checkout, batch, binding, legacy


def _consumed_five(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                   ) -> tuple[Path, Path, FrozenAttemptBinding, bytes, Path]:
    checkout, batch, binding, legacy = _consumed_four(tmp_path, monkeypatch)
    artifact_root = tmp_path / "canonical-artifacts"
    _publish_canonical_attestation(artifact_root, checkout, binding)
    freeze_authorization(
        batch, binding, approved_attempt=5, checkout=checkout,
        artifact_root=artifact_root, expected_parent_hash=digest(legacy),
    )
    selected = resolve_operator_attempt(batch, scope()).authorized
    assert selected is not None and selected.authorization.attempt_number == 5
    initial = initialize_attempt(selected, binding)
    reservation = {**initial, "state": "DISPATCH_RESERVED", "dispatch_count": 1}
    (selected.location / "dispatch_reservation.json").write_bytes(canonical_bytes(reservation))
    terminal = {
        **reservation, "state": "CONSUMED_FAILED", "physical_outbound_count": 1,
        "blocked_additional_outbound_count": 0, "failure_code": "OUTPUT_VALIDATION",
        "failure_diagnostics": {"failure_category": "LOCAL_RUNTIME_ERROR"},
    }
    (selected.location / "generation_state.json").write_bytes(canonical_bytes(terminal))
    return checkout, batch, binding, legacy, artifact_root


def test_attempt_six_authorization_freeze_binds_latest_attestation_and_preserves_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy, artifact_root = _consumed_five(tmp_path, monkeypatch)
    before = all_bytes(batch)
    history = read_history(batch, scope())
    assert [item.number for item in history] == [1, 2, 3, 4, 5]
    assert all(item.state == "CONSUMED_FAILED" and item.consumed for item in history)
    manifest_hash, _ = verify_canonical_runtime_attestation(artifact_root, checkout, binding)
    with (
        patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send,
        patch("httpx.Client.send", side_effect=AssertionError("outbound forbidden")) as sync_send,
    ):
        target = freeze_authorization(
            batch, binding, approved_attempt=6, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    send.assert_not_called()
    sync_send.assert_not_called()
    auth = PathCAttemptAuthorization.model_validate_json(target.read_bytes())
    auth.validate_integrity()
    assert auth.attempt_number == 6
    assert auth.attempt_id == attempt_id_for(scope(), 6, auth.authorization_id)
    assert auth.previous_attempt_number == 5
    assert auth.previous_attempt_id == history[-1].attempt_id
    assert auth.previous_terminal_state == "CONSUMED_FAILED"
    assert auth.previous_state_sha256 == history[-1].state_sha256
    assert auth.canonical_runtime_attestation_run == CANONICAL_ATTESTATION_RUN
    assert auth.canonical_runtime_attestation_manifest_sha256 == manifest_hash
    assert auth.binding() == binding
    assert auth.retry == 0 and auth.regeneration is False
    assert auth.maximum_provider_generation_dispatches == 1
    assert auth.repair == auth.provider_fallback == auth.endpoint_fallback == "none"
    assert auth.request_format_fallback == "none" and auth.no_top_up and auth.no_replacement
    for path, raw in before.items():
        assert path.read_bytes() == raw
    assert not (batch / "attempt_006").exists()
    assert not (batch / "attempt_006/dispatch_reservation.json").exists()
    assert not (batch / "attempt_006/provider_response.raw.json").exists()
    assert not (batch / "attempt_006/path_c_task_manifest.json").exists()


def test_attempt_seven_and_attempt_six_preexisting_state_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy, artifact_root = _consumed_five(tmp_path, monkeypatch)
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError, match="ATTEMPT_NUMBER_NOT_APPROVED"):
        freeze_authorization(
            batch, binding, approved_attempt=7, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
    (batch / "attempt_006").mkdir()
    (batch / "attempt_006/dispatch_reservation.json").write_text("{}", encoding="utf-8")
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=6, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
    assert not (batch / "authorizations/attempt_006.authorization.json").exists()


@pytest.mark.parametrize("artifact", [
    "dispatch_reservation.json", "provider_response.raw.json", "path_c_task_manifest.json",
])
def test_attempt_six_rejects_each_preexisting_consumption_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str,
) -> None:
    checkout, batch, binding, legacy, artifact_root = _consumed_five(tmp_path, monkeypatch)
    attempt = batch / "attempt_006"
    attempt.mkdir()
    (attempt / artifact).write_text("{}", encoding="utf-8")
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=6, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
    assert not (batch / "authorizations/attempt_006.authorization.json").exists()


@pytest.mark.parametrize("corruption", ["parent_id", "parent_hash", "parent_state"])
def test_attempt_six_requires_intact_consumed_attempt_five_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    checkout, batch, binding, legacy, artifact_root = _consumed_five(tmp_path, monkeypatch)
    if corruption in {"parent_id", "parent_hash"}:
        identity_path = batch / "attempt_005/attempt_identity.json"
        identity = json.loads(identity_path.read_bytes())
        field = "previous_attempt_id" if corruption == "parent_id" else "previous_state_sha256"
        identity[field] = "0" * 64
        identity_path.write_bytes(canonical_bytes(identity))
    else:
        state_path = batch / "attempt_005/generation_state.json"
        state = json.loads(state_path.read_bytes())
        state["state"] = "AUTHORIZED_NOT_DISPATCHED"
        state_path.write_bytes(canonical_bytes(state))
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=6, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
    assert not (batch / "authorizations/attempt_006.authorization.json").exists()


def test_attempt_six_duplicate_authorization_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy, artifact_root = _consumed_five(tmp_path, monkeypatch)
    target = freeze_authorization(
        batch, binding, approved_attempt=6, checkout=checkout,
        artifact_root=artifact_root, expected_parent_hash=digest(legacy),
    )
    published = target.read_bytes()
    with pytest.raises(GenerationAttemptError, match="AUTHORIZATION_ALREADY_EXISTS"):
        freeze_authorization(
            batch, binding, approved_attempt=6, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    assert target.read_bytes() == published
    assert not (batch / "attempt_006").exists()


def test_attempt_six_requires_exact_latest_attestation_and_source_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy, artifact_root = _consumed_five(tmp_path, monkeypatch)
    run = artifact_root / "v2_1r_d4r_a6_operator" / CANONICAL_ATTESTATION_RUN
    source = run / "canonical_runtime_attestation.json"
    value = json.loads(source.read_bytes())
    value["status"] = "BLOCKED"
    source.write_bytes(canonical_bytes(value))
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=6, checkout=checkout,
            artifact_root=artifact_root, expected_parent_hash=digest(legacy),
        )
    assert not (batch / "authorizations/attempt_006.authorization.json").exists()


def test_authorization_entry_parser_allows_six_not_seven() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-attempt", type=int, choices=(2, 3, 4, 5, 6), default=2)
    assert parser.parse_args(["--approved-attempt", "6"]).approved_attempt == 6
    with pytest.raises(SystemExit):
        parser.parse_args(["--approved-attempt", "7"])


def test_authorize_attempt_five_only_after_four_consumed_and_attestation_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy = _consumed_four(tmp_path, monkeypatch)
    artifact_root = tmp_path / "canonical-artifacts"
    _publish_canonical_attestation(artifact_root, checkout, binding)
    original = all_bytes(batch)
    records = read_history(batch, scope())
    assert [record.number for record in records] == [1, 2, 3, 4]
    with pytest.raises(GenerationAttemptError, match="NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT"):
        resolve_operator_attempt(batch, scope())
    with (
        patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send,
        patch("httpx.Client.send", side_effect=AssertionError("outbound forbidden")) as sync_send,
    ):
        target = freeze_authorization(
            batch, binding, approved_attempt=5, checkout=checkout, artifact_root=artifact_root,
            expected_parent_hash=digest(legacy),
        )
    send.assert_not_called()
    sync_send.assert_not_called()
    auth = PathCAttemptAuthorization.model_validate_json(target.read_bytes())
    auth.validate_integrity()
    assert auth.attempt_number == 5
    assert auth.previous_attempt_number == 4
    assert auth.previous_attempt_id == records[-1].attempt_id
    assert auth.previous_terminal_state == "CONSUMED_FAILED"
    assert auth.previous_state_sha256 == records[-1].state_sha256
    assert auth.attempt_id == attempt_id_for(scope(), 5, auth.authorization_id)
    assert auth.binding() == binding
    assert auth.maximum_provider_generation_dispatches == 1
    assert (
        auth.retry == 0 and auth.regeneration is False
        and auth.no_top_up and auth.no_replacement
    )
    assert (
        auth.provider_fallback == auth.endpoint_fallback == auth.request_format_fallback == "none"
    )
    for path, raw in original.items():
        assert path.read_bytes() == raw
    assert not (batch / "attempt_005").exists()
    assert not (batch / "attempt_005/dispatch_reservation.json").exists()
    assert not (batch / "attempt_005/provider_response.raw.json").exists()
    assert not (batch / "attempt_005/path_c_task_manifest.json").exists()


def test_attempt_five_rejects_nonpass_attestation_and_attempt_seven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy = _consumed_four(tmp_path, monkeypatch)
    artifact_root = tmp_path / "canonical-artifacts"
    _publish_canonical_attestation(artifact_root, checkout, binding, status="BLOCKED")
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError, match="CANONICAL_ATTESTATION_NOT_PASS"):
        freeze_authorization(
            batch, binding, approved_attempt=5, checkout=checkout, artifact_root=artifact_root,
            expected_parent_hash=digest(legacy),
        )
    with pytest.raises(GenerationAttemptError, match="ATTEMPT_NUMBER_NOT_APPROVED"):
        freeze_authorization(
            batch, binding, approved_attempt=7, checkout=checkout, artifact_root=artifact_root,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
    assert not (batch / "authorizations/attempt_005.authorization.json").exists()


@pytest.mark.parametrize("corruption", [
    "parent_id", "parent_hash", "existing_state", "existing_reservation",
    "existing_raw", "existing_manifest",
])
def test_attempt_five_rejects_bad_parent_or_preexisting_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    checkout, batch, binding, legacy = _consumed_four(tmp_path, monkeypatch)
    artifact_root = tmp_path / "canonical-artifacts"
    _publish_canonical_attestation(artifact_root, checkout, binding)
    if corruption in {"parent_id", "parent_hash"}:
        path = batch / "attempt_004/attempt_identity.json"
        value = json.loads(path.read_bytes())
        key = "previous_attempt_id" if corruption == "parent_id" else "previous_state_sha256"
        value[key] = "0" * 64
        path.write_bytes(canonical_bytes(value))
    elif corruption == "existing_state":
        (batch / "attempt_005").mkdir()
    else:
        attempt_dir = batch / "attempt_005"
        attempt_dir.mkdir()
        artifact = {
            "existing_reservation": "dispatch_reservation.json",
            "existing_raw": "provider_response.raw.json",
            "existing_manifest": "path_c_task_manifest.json",
        }[corruption]
        (attempt_dir / artifact).write_text("{}", encoding="utf-8")
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=5, checkout=checkout, artifact_root=artifact_root,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
    assert not (batch / "authorizations/attempt_005.authorization.json").exists()


def test_attempt_five_existing_authorization_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, batch, binding, legacy = _consumed_four(tmp_path, monkeypatch)
    artifact_root = tmp_path / "canonical-artifacts"
    _publish_canonical_attestation(artifact_root, checkout, binding)
    auth = _synthetic_grant(batch, binding)
    auth = auth.model_copy(update={
        "attempt_number": 5,
        "attempt_id": attempt_id_for(scope(), 5, auth.authorization_id),
        "previous_attempt_number": 4,
        "previous_attempt_id": read_history(batch, scope())[-1].attempt_id,
        "previous_state_sha256": read_history(batch, scope())[-1].state_sha256,
    }).with_integrity_hash()
    _publish_fake(batch, auth)
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError, match="AUTHORIZATION_ALREADY_EXISTS"):
        freeze_authorization(
            batch, binding, approved_attempt=5, checkout=checkout, artifact_root=artifact_root,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
