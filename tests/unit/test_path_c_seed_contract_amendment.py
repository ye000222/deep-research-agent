from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.prospective_acquisition.seed_amendment import (
    SeedAmendmentIntegrityError,
    resolve_seed_amendment,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "artifacts" / "v2_1r_d4r_a51"
AMENDMENT_DIR = ROOT / "artifacts" / "v2_1r_d4r_path_c_seed_amendment"
BASE_CONTRACT = BASE_DIR / "generic_task_generation_contract.json"
BASE_SCHEMA = BASE_DIR / "task_manifest_schema.json"
AMENDMENT = AMENDMENT_DIR / "seed_policy_amendment.json"
HASH_MANIFEST = AMENDMENT_DIR / "hash_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_a51_contract_and_schema_hashes_are_unchanged() -> None:
    base_manifest = _read_json(BASE_DIR / "hash_manifest.json")
    assert _sha256(BASE_CONTRACT) == (
        "b8f8dea0c4ad84d1d17aa25e73b8e53f3ecdde402cd8ca6e8cab8b0cc68f6af3"
    )
    assert _sha256(BASE_SCHEMA) == (
        "f908b6397240f5216746edccd26825582223a3d0747b73c9e340c73a1b6d53d3"
    )
    assert base_manifest["files"]["generic_task_generation_contract.json"] == _sha256(
        BASE_CONTRACT
    )
    assert base_manifest["files"]["task_manifest_schema.json"] == _sha256(BASE_SCHEMA)


def test_entire_original_a51_bundle_still_matches_its_frozen_hashes() -> None:
    base_manifest = _read_json(BASE_DIR / "hash_manifest.json")
    for relative_path, expected in base_manifest["files"].items():
        assert _sha256(BASE_DIR / relative_path) == expected
    payload = {
        relative_path: _read_json(BASE_DIR / relative_path)
        for relative_path in base_manifest["files"]
        if relative_path not in {"hash_manifest.json", "readiness_report.md"}
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == base_manifest["artifact_json_bundle_sha256"]


def test_amendment_hash_manifest_matches_files_and_base_identity() -> None:
    manifest = _read_json(HASH_MANIFEST)
    amendment = _read_json(AMENDMENT)
    assert manifest["base_contract_sha256"] == _sha256(BASE_CONTRACT)
    assert manifest["base_a51_hash_manifest_sha256"] == _sha256(
        BASE_DIR / "hash_manifest.json"
    )
    assert manifest["base_a51_bundle_sha256"] == _read_json(
        BASE_DIR / "hash_manifest.json"
    )["artifact_json_bundle_sha256"]
    assert amendment["base_contract"]["sha256"] == _sha256(BASE_CONTRACT)

    file_hashes = manifest["files"]
    assert isinstance(file_hashes, dict)
    for relative_path, expected in file_hashes.items():
        assert _sha256(AMENDMENT_DIR / relative_path) == expected
    implementation_hashes = manifest["implementation_files"]
    assert isinstance(implementation_hashes, dict)
    for relative_path, expected in implementation_hashes.items():
        assert _sha256(ROOT / relative_path) == expected
    canonical = json.dumps(
        {"artifacts": file_hashes, "implementation_files": implementation_hashes},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == manifest["amendment_bundle_sha256"]


def test_only_seed_behavior_clause_is_overridden() -> None:
    base = _read_json(BASE_CONTRACT)
    result = resolve_seed_amendment(BASE_CONTRACT, AMENDMENT, "SEED_SUPPORT_NOT_ESTABLISHED")
    effective = result.effective_contract
    for key, value in base.items():
        if key != "unsupported_seed_behavior":
            assert effective[key] == value
    assert effective["unsupported_seed_behavior"] != base["unsupported_seed_behavior"]


@pytest.mark.parametrize(
    ("capability", "expected_send", "expected_seed"),
    [
        ("SEED_SUPPORTED", True, "d4r-a51-taskgen-40-v1"),
        ("SEED_SUPPORT_NOT_ESTABLISHED", False, None),
        ("SEED_UNSUPPORTED", False, None),
    ],
)
def test_capability_resolves_seed_without_determinism_claim(
    capability: str, expected_send: bool, expected_seed: str | None
) -> None:
    result = resolve_seed_amendment(BASE_CONTRACT, AMENDMENT, cast(Any, capability))
    assert result.send_seed is expected_send
    assert result.seed_value == expected_seed
    assert result.record_seed_sent_on_dispatch is expected_send
    assert result.deterministic_generation_claim is False
    assert result.provider_dispatch_authorized is False


def test_one_call_and_no_retry_top_up_replacement_or_fallback_remain_frozen() -> None:
    contract = _read_json(BASE_CONTRACT)
    amendment = _read_json(AMENDMENT)
    assert contract["task_count"] == 40
    assert contract["generation_calls_max"] == 1
    assert contract["retry"] == 0
    assert contract["regeneration"] is False
    assert contract["fallback"] == "none"
    assert contract["no_replacement"] is True
    unchanged = amendment["unchanged_clauses"]
    assert unchanged["no_top_up"] is True
    assert unchanged["no_replacement"] is True
    assert unchanged["no_truncation"] is True
    assert unchanged["raw_provider_output_freeze"] is True
    assert unchanged["invalid_json_or_schema"] == "CONSUMED_TERMINAL_FAILURE"
    assert unchanged["wrong_task_count"] == "CONSUMED_TERMINAL_FAILURE"
    assert unchanged["provider_binding"] == "UNCHANGED"
    assert unchanged["execution_context"] == "UNCHANGED"


def test_base_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    altered = tmp_path / "altered_base.json"
    altered.write_text(BASE_CONTRACT.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SeedAmendmentIntegrityError, match="base_contract_hash_mismatch"):
        resolve_seed_amendment(altered, AMENDMENT, "SEED_SUPPORT_NOT_ESTABLISHED")


def test_amendment_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    altered = tmp_path / "altered_amendment.json"
    altered.write_text(AMENDMENT.read_text(encoding="utf-8") + " ", encoding="utf-8")
    manifest = _read_json(HASH_MANIFEST)
    manifest["files"] = {"altered_amendment.json": manifest["files"]["seed_policy_amendment.json"]}
    (tmp_path / "hash_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedAmendmentIntegrityError, match="amendment_artifact_hash_mismatch"):
        resolve_seed_amendment(
            BASE_CONTRACT,
            altered,
            "SEED_SUPPORT_NOT_ESTABLISHED",
        )


def test_resolver_never_authorizes_or_performs_provider_dispatch() -> None:
    amendment = _read_json(AMENDMENT)
    assert amendment["provider_dispatch_authorized"] is False
    result = resolve_seed_amendment(BASE_CONTRACT, AMENDMENT, "SEED_SUPPORT_NOT_ESTABLISHED")
    assert result.provider_dispatch_authorized is False
    assert not hasattr(result, "dispatch")
