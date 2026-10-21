"""Synthetic consumed history only; authorization never dispatches a Provider."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from scripts.prospective_acquisition.path_c_attempts import (
    FrozenAttemptBinding,
    GenerationAttemptError,
    PathCAttemptAuthorization,
    canonical_bytes,
    digest,
    initialize_attempt,
    read_history,
    resolve_operator_attempt,
)
from scripts.prospective_acquisition.path_c_authorize_002 import (
    freeze_authorization,
    scope,
    verify_timeout_policy,
)
from tests.unit.test_path_c_attempts import _history_fixture, _publish_fake, _synthetic_grant

ROOT = Path(__file__).resolve().parents[2]


def consumed_history(tmp_path: Path) -> tuple[Path, Path, FrozenAttemptBinding, bytes]:
    checkout, batch, binding, legacy = _history_fixture(tmp_path)
    # Fixture only, never an authorization in the canonical artifacts storage.
    _publish_fake(batch, _synthetic_grant(batch, binding))
    attempt = resolve_operator_attempt(batch, scope()).authorized
    assert attempt is not None
    initial = initialize_attempt(attempt, binding)
    reservation = {**initial, "state": "DISPATCH_RESERVED", "dispatch_count": 1}
    (attempt.location / "dispatch_reservation.json").write_bytes(canonical_bytes(reservation))
    terminal = {
        **reservation, "state": "CONSUMED_FAILED", "physical_outbound_count": 1,
        "blocked_additional_outbound_count": 0, "failure_code": "MODELGATEWAYERROR",
        "failure_diagnostics": {
            "failure_class": "MODEL_GATEWAY_ERROR", "failure_category": "TIMEOUT",
            "gateway_error_code": "MODEL_TIMEOUT", "gateway_detail_code": "READ_TIMEOUT",
            "provider_http_status": 200, "response_received": True,
        },
    }
    (attempt.location / "generation_state.json").write_bytes(canonical_bytes(terminal))
    for name in ("apps/api/app/llm/model_http_policy.py", "apps/api/app/worker/tasks.py"):
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return checkout, batch, binding, legacy


def all_bytes(batch: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in batch.rglob("*") if p.is_file()}


def test_authorize_exact_next_without_dispatch_and_preserve_history(tmp_path: Path) -> None:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    before = all_bytes(batch)
    records = read_history(batch, scope())
    assert [r.number for r in records] == [1, 2]
    with pytest.raises(GenerationAttemptError, match="NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT"):
        resolve_operator_attempt(batch, scope())
    assert not (batch / "attempt_003").exists()
    with (
        patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send,
        patch("httpx.Client.send", side_effect=AssertionError("outbound forbidden")) as sync_send,
    ):
        target = freeze_authorization(
            batch, binding, approved_attempt=3, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    send.assert_not_called()
    sync_send.assert_not_called()
    auth = PathCAttemptAuthorization.model_validate_json(target.read_bytes())
    auth.validate_integrity()
    assert auth.attempt_number == 3 and auth.previous_attempt_number == 2
    assert auth.previous_attempt_id == records[-1].attempt_id
    assert auth.previous_state_sha256 == records[-1].state_sha256
    assert auth.previous_state_sha256 == digest(before[batch / "attempt_002/generation_state.json"])
    assert auth.authorization_id != records[-1].authorization_id
    assert auth.binding() == binding
    assert auth.maximum_provider_generation_dispatches == 1
    assert auth.retry == 0 and auth.regeneration is False
    for path, raw in before.items():
        assert path.read_bytes() == raw
    assert not (batch / "attempt_003").exists()
    selected = resolve_operator_attempt(batch, scope()).authorized
    assert selected is not None and selected.authorization == auth
    published = target.read_bytes()
    with pytest.raises(GenerationAttemptError, match="AUTHORIZATION_ALREADY_EXISTS"):
        freeze_authorization(
            batch, binding, approved_attempt=3, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert target.read_bytes() == published


@pytest.mark.parametrize("change", [
    "parent_id", "parent_hash", "state", "count", "diagnostic", "authorization_hash",
    "provider_binding", "legacy_hash", "receipt",
])
def test_parent_integrity_failure_leaves_no_grant(tmp_path: Path, change: str) -> None:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    identity = batch / "attempt_002/attempt_identity.json"
    state = batch / "attempt_002/generation_state.json"
    grant = batch / "authorizations/attempt_002.authorization.json"
    expected = digest(legacy)
    if change in {"parent_id", "parent_hash"}:
        value = json.loads(identity.read_bytes())
        key = "previous_attempt_id" if change == "parent_id" else "previous_state_sha256"
        value[key] = "0" * 64
        identity.write_bytes(canonical_bytes(value))
    elif change == "authorization_hash":
        value = json.loads(grant.read_bytes())
        value["authorization_hash"] = "0" * 64
        grant.write_bytes(canonical_bytes(value))
    elif change == "provider_binding":
        binding = binding.model_copy(update={"provider_safe_config_fingerprint": "f" * 64})
    elif change == "legacy_hash":
        expected = "0" * 64
    elif change == "receipt":
        value = json.loads((batch / "attempt_002/dispatch_reservation.json").read_bytes())
        value["authorization_id"] = "00000000-0000-0000-0000-000000000000"
        (batch / "attempt_002/dispatch_reservation.json").write_bytes(canonical_bytes(value))
    else:
        value = json.loads(state.read_bytes())
        if change == "state":
            value["state"] = "DISPATCH_RESERVED"
        elif change == "count":
            value["physical_outbound_count"] = 0
        else:
            value["failure_diagnostics"]["gateway_detail_code"] = "CONNECT_TIMEOUT"
        state.write_bytes(canonical_bytes(value))
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=3, checkout=checkout, expected_parent_hash=expected,
        )
    assert not (batch / "authorizations/attempt_003.authorization.json").exists()
    assert all_bytes(batch) == before


@pytest.mark.parametrize("number", [2, 4, 0, True])
def test_no_skipped_or_unapproved_number(tmp_path: Path, number: int) -> None:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=number, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


@pytest.mark.parametrize("name", [
    "attempt_003", "attempt_003.dispatch_reservation.json",
    "attempt_003.provider_response.raw.json",
])
def test_existing_execution_evidence_blocks(tmp_path: Path, name: str) -> None:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    (batch / name).touch()
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(
            batch, binding, approved_attempt=3, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


def test_timeout_policy_required_and_failure_blocks(tmp_path: Path) -> None:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    before = all_bytes(batch)
    with pytest.raises(GenerationAttemptError, match="TIMEOUT_POLICY_PROOF_REQUIRED"):
        freeze_authorization(
            batch, binding, approved_attempt=3, expected_parent_hash=digest(legacy),
        )
    with (
        patch(
            "app.llm.model_http_policy.model_generation_timeout",
            return_value=httpx.Timeout(10.0),
        ),
        pytest.raises(GenerationAttemptError, match="MODEL_TIMEOUT_POLICY_INTEGRITY_MISMATCH"),
    ):
        freeze_authorization(
            batch, binding, approved_attempt=3, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before


@pytest.mark.parametrize("case", ["operator_override", "worker_override", "stale_policy"])
def test_timeout_source_wiring_drift_blocks(tmp_path: Path, case: str) -> None:
    checkout, _, _, _ = consumed_history(tmp_path)
    target = checkout / (
        "scripts/prospective_acquisition/path_c_generation_entry.py"
        if case == "operator_override" else "apps/api/app/worker/tasks.py"
        if case == "worker_override" else "apps/api/app/llm/model_http_policy.py"
    )
    original = target.read_text()
    changed = original.replace("model_generation_timeout()", "httpx.Timeout(10.0)")
    if case == "stale_policy":
        changed = original.replace("90.0", "10.0")
    target.write_text(changed)
    with pytest.raises(GenerationAttemptError):
        verify_timeout_policy(checkout)


def test_atomic_publication_failure_no_partial_grant(tmp_path: Path) -> None:
    checkout, batch, binding, legacy = consumed_history(tmp_path)
    before = all_bytes(batch)
    with patch("os.link", side_effect=OSError("fake write failure")), pytest.raises(OSError):
        freeze_authorization(
            batch, binding, approved_attempt=3, checkout=checkout,
            expected_parent_hash=digest(legacy),
        )
    assert all_bytes(batch) == before
