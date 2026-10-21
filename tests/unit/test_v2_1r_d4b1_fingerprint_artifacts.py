from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.provider_fingerprint_contract import canonical_safe_payload_bytes

ROOT = Path(__file__).resolve().parents[2]


def _json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_d4a_artifact_bytes_remain_pinned_but_do_not_claim_preimage() -> None:
    manifest = _json("artifacts/v2_1r_d4a_hash_manifest.json")
    provider_artifact_path = ROOT / "artifacts/v2_1r_d4a_provider_runtime_identity.json"
    actual = hashlib.sha256(provider_artifact_path.read_bytes()).hexdigest()
    assert actual == manifest["provider_identity_artifact_sha256"]
    provider_artifact = _json("artifacts/v2_1r_d4a_provider_runtime_identity.json")
    assert "safe_config_fingerprint" in provider_artifact
    assert "fingerprint_payload" not in provider_artifact
    assert "fingerprint_algorithm_version" not in provider_artifact


def test_declared_d4a_six_field_contract_recomputes_to_beb5() -> None:
    payload = {
        "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
        "provider_type": "openai_compatible_chat",
        "model": "deepseek-flash",
        "endpoint_host": "api.deepseek.com",
        "profile_version": 23,
        "credential_version": 4,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert digest == "beb5bd2977b662c052e51551fa7566d264187576b982f716d581b763e954d996"
    assert digest != "a85a047a776b6c8cad13c8c846abbfcbe6b38e6b8c3d34f99c6a0550ff79dcca"


def test_d4a1_payload_hash_and_authorization_boundary() -> None:
    readiness = _json("artifacts/v2_1r_d4a1_provider_readiness.json")
    payload = readiness["safe_payload"]
    assert isinstance(payload, dict)
    digest = hashlib.sha256(canonical_safe_payload_bytes(payload)).hexdigest()
    assert digest == readiness["safe_config_fingerprint"]
    authorization = readiness["authorization"]
    assert isinstance(authorization, dict)
    assert authorization["d4b_execution_authorized_by_this_artifact"] is False
    assert authorization["requires_explicit_reauthorization_against_d4a1"] is True
    rendered = json.dumps(payload, ensure_ascii=False)
    for forbidden in ("api_key", "Authorization", "plaintext", "master_key", "secret"):
        assert forbidden.lower() not in rendered.lower()


def test_zero_call_blocked_attempt_and_new_preflight_are_preserved() -> None:
    blocked = _json("artifacts/v2_1r_d4b_preflight_block.json")
    preflight = _json("artifacts/v2_1r_d4b1_zero_call_preflight.json")
    assert blocked["provider_attempts"] == 0
    assert preflight["preserved_attempt"] == "D4B_EXECUTION_ATTEMPT_1"
    assert preflight["preserved_attempt_status"] == "BLOCKED_BEFORE_PROVIDER_CALL"
    assert preflight["d4b1_provider_calls"] == 0
    assert preflight["d4a_artifacts_overwritten"] is False
    assert preflight["d4b_attempt_overwritten"] is False


def test_d4b1_hash_manifest_matches_all_recorded_files() -> None:
    manifest = _json("artifacts/v2_1r_d4b1_hash_manifest.json")
    groups = (
        manifest["d4a_original_artifacts_preserved"],
        manifest["d4b1_artifacts"],
        manifest["readiness_code_and_tests"],
    )
    for group in groups:
        assert isinstance(group, dict)
        for relative_path, expected_hash in group.items():
            actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            assert actual == expected_hash, relative_path
