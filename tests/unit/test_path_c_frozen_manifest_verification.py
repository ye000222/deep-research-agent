"""Deterministic, content-blind tests for the host manifest verifier."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from scripts.prospective_acquisition import verify_frozen_manifest as verifier
from scripts.prospective_acquisition.path_c_attempts import (
    AttemptIdentity,
    PathCAttemptAuthorization,
    attempt_id_for,
    canonical_bytes,
    digest,
)
from scripts.prospective_acquisition.path_c_authorize_002 import CANONICAL_ATTESTATION_RUN
from scripts.prospective_acquisition.path_c_generation import (
    BATCH_ID,
    EXECUTION_CONTEXT_ID,
    GENERATION_IMPLEMENTATION_FILES,
    POPULATION_ID,
    frozen_attempt_binding,
)
from tests.unit.test_path_c_generation import _context


def _fixture(tmp_path: Path, *, count: int = 40) -> tuple[Path, Path, list[str]]:
    assert set(GENERATION_IMPLEMENTATION_FILES) == set(
        verifier.ATTEMPT_006_GENERATION_IMPLEMENTATION_FILES
    )
    artifact_root = tmp_path / "volume"
    checkout_root = Path(__file__).resolve().parents[2]
    api_root = checkout_root / "apps/api"
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    attempt = batch / "attempt_006"
    attempt.mkdir(parents=True)
    tasks = [
        {"task_id": f"task-{number:03d}", "task_text": f"Synthetic manifest item {number:03d}"}
        for number in range(1, count + 1)
    ]
    texts = [task["task_text"] for task in tasks]
    manifest_bytes = canonical_bytes({
        "manifest_version": "prospective-generic-task-manifest.v1",
        "task_count": 40,
        "tasks": tasks,
    }) + b"\n"
    (attempt / "path_c_task_manifest.json").write_bytes(manifest_bytes)

    context, provider = _context()
    binding = frozen_attempt_binding(context, provider)
    scope = verifier._attempt_scope()
    authorization_id = UUID(verifier.AUTHORIZATION_ID)
    auth = PathCAttemptAuthorization(
        **binding.model_dump(),
        authorization_id=authorization_id,
        attempt_number=6,
        attempt_id=attempt_id_for(scope, 6, authorization_id),
        previous_attempt_number=5,
        previous_attempt_id="a" * 64,
        previous_state_sha256="b" * 64,
        canonical_runtime_attestation_run=CANONICAL_ATTESTATION_RUN,
        canonical_runtime_attestation_manifest_sha256="c" * 64,
        created_at_utc=datetime.now(UTC),
    ).with_integrity_hash()
    auth_bytes = canonical_bytes(auth.model_dump(mode="json", exclude_none=True)) + b"\n"
    (batch / "authorizations").mkdir()
    (batch / "authorizations/attempt_006.authorization.json").write_bytes(auth_bytes)
    identity = AttemptIdentity(
        binding=binding,
        attempt_number=6,
        attempt_id=auth.attempt_id,
        authorization_id=authorization_id,
        authorization_hash=auth.authorization_hash,
        authorization_file_sha256=digest(auth_bytes),
        previous_attempt_id=auth.previous_attempt_id,
        previous_state_sha256=auth.previous_state_sha256,
        created_at_utc=datetime.now(UTC),
    )
    identity_bytes = canonical_bytes(identity.model_dump(mode="json")) + b"\n"
    (attempt / "attempt_identity.json").write_bytes(identity_bytes)
    manifest_sha = digest(manifest_bytes)
    raw_response = json.dumps({
        "choices": [{"message": {"content": json.dumps({"tasks": texts})}}],
    }, separators=(",", ":")).encode()
    (attempt / "provider_response.raw.json").write_bytes(raw_response)
    implementation_hashes = {}
    for relative in GENERATION_IMPLEMENTATION_FILES:
        root = api_root if relative.startswith("apps/api/") else checkout_root
        suffix = (
            relative.removeprefix("apps/api/")
            if relative.startswith("apps/api/")
            else relative
        )
        implementation_hashes[relative] = digest((root / suffix).read_bytes())
    metadata = {
        "attempt_number": 6, "attempt_id": auth.attempt_id,
        "authorization_id": str(authorization_id), "authorization_hash": auth.authorization_hash,
        "authorization_file_sha256": digest(auth_bytes),
        "attempt_identity_sha256": digest(identity_bytes),
        "population_id": POPULATION_ID, "batch_id": BATCH_ID,
        "execution_context_id": EXECUTION_CONTEXT_ID,
        "manifest_sha256": manifest_sha, "task_count": 40,
        "generation_dispatch_count": 1,
        "seed_capability": "SEED_SUPPORT_NOT_ESTABLISHED", "seed_sent": False,
        "deterministic_generation_claim": False, "acquisition_started": False,
        "search_count": 0, "reader_count": 0, "evidence_extractor_count": 0,
        "database_writes": 0, "holdout_access": 0,
        "implementation_sha256": implementation_hashes,
    }
    metadata_bytes = canonical_bytes(metadata) + b"\n"
    (attempt / "generation_metadata.json").write_bytes(metadata_bytes)
    hash_manifest_bytes = canonical_bytes({
        "files": {
            "path_c_task_manifest.json": manifest_sha,
            "provider_response.raw.json": digest(raw_response),
            "generation_metadata.json": digest(metadata_bytes),
        }
    }) + b"\n"
    (attempt / "hash_manifest.json").write_bytes(hash_manifest_bytes)
    reservation = {
        "state": "DISPATCH_RESERVED", "attempt_number": 6,
        "attempt_id": auth.attempt_id, "authorization_id": str(authorization_id),
        "authorization_hash": auth.authorization_hash,
        "authorization_file_sha256": digest(auth_bytes),
        "attempt_identity_sha256": digest(identity_bytes), "dispatch_count": 1,
        "seed_sent": False, "deterministic_generation_claim": False,
    }
    (attempt / "dispatch_reservation.json").write_bytes(canonical_bytes(reservation) + b"\n")
    state = {
        **reservation, "state": "SUCCEEDED_AND_FROZEN",
        "population_id": POPULATION_ID, "batch_id": BATCH_ID,
        "execution_context_id": EXECUTION_CONTEXT_ID,
        "manifest_sha256": manifest_sha,
        "metadata_sha256": digest(metadata_bytes),
        "raw_provider_response_sha256": digest(raw_response),
        "hash_manifest_sha256": digest(hash_manifest_bytes),
    }
    (attempt / "generation_state.json").write_bytes(canonical_bytes(state) + b"\n")
    schema = checkout_root / verifier.SCHEMA_RELATIVE_PATH
    return artifact_root, schema, texts


def _mock_history(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        SimpleNamespace(
            state="CONSUMED_FAILED" if number < 6 else "SUCCEEDED_AND_FROZEN",
            consumed=True,
            attempt_id=verifier.ATTEMPT_ID if number == 6 else f"{number:064x}",
        )
        for number in range(1, 7)
    ]
    monkeypatch.setattr(verifier, "read_history", lambda *_: records)


def _rewrite_metadata_hash_chain(batch: Path, metadata: dict[str, object]) -> None:
    attempt = batch / "attempt_006"
    metadata_bytes = canonical_bytes(metadata) + b"\n"
    (attempt / "generation_metadata.json").write_bytes(metadata_bytes)
    hash_manifest_path = attempt / "hash_manifest.json"
    hash_manifest = json.loads(hash_manifest_path.read_bytes())
    hash_manifest["files"]["generation_metadata.json"] = digest(metadata_bytes)
    hash_manifest_bytes = canonical_bytes(hash_manifest) + b"\n"
    hash_manifest_path.write_bytes(hash_manifest_bytes)
    state_path = attempt / "generation_state.json"
    state = json.loads(state_path.read_bytes())
    state["metadata_sha256"] = digest(metadata_bytes)
    state["hash_manifest_sha256"] = digest(hash_manifest_bytes)
    state_path.write_bytes(canonical_bytes(state) + b"\n")


def test_valid_attempt_006_manifest_passes_without_exposing_task_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, schema, texts = _fixture(tmp_path)
    _mock_history(monkeypatch)
    manifest = (
        artifact_root / "v2_1r_d4r_path_c_generation"
        / f"{POPULATION_ID}__{BATCH_ID}"
        / "attempt_006/path_c_task_manifest.json"
    )
    result = verifier.verify_frozen_manifest(
        artifact_root, schema, expected_manifest_sha256=digest(manifest.read_bytes()),
        checkout_root=Path(__file__).resolve().parents[2],
        api_root=Path(__file__).resolve().parents[2] / "apps/api",
    )
    assert result["status"] == "PATH_C_FROZEN_MANIFEST_VERIFIED"
    assert result["task_count"] == 40 and result["content_exposed"] is False
    assert all(text not in json.dumps(result) for text in texts)


def test_post_generation_source_drift_does_not_invalidate_frozen_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    checkout_root = tmp_path / "checkout"
    api_root = tmp_path / "api"
    for relative in GENERATION_IMPLEMENTATION_FILES:
        is_api_source = relative.startswith("apps/api/")
        root = api_root if is_api_source else checkout_root
        suffix = relative.removeprefix("apps/api/") if is_api_source else relative
        path = root / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"post-generation acquisition change")

    manifest = (
        artifact_root / "v2_1r_d4r_path_c_generation"
        / f"{POPULATION_ID}__{BATCH_ID}"
        / "attempt_006/path_c_task_manifest.json"
    )
    result = verifier.verify_frozen_manifest(
        artifact_root, schema, expected_manifest_sha256=digest(manifest.read_bytes()),
        checkout_root=checkout_root, api_root=api_root,
    )

    assert result["status"] == "PATH_C_FROZEN_MANIFEST_VERIFIED"
    assert result["generation_implementation_provenance_recorded"] is True
    assert "generation_implementation_hashes_match" not in result


def test_malformed_historical_implementation_hashes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    metadata_path = batch / "attempt_006/generation_metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["implementation_sha256"][GENERATION_IMPLEMENTATION_FILES[0]] = "not-a-sha"
    _rewrite_metadata_hash_chain(batch, metadata)
    with pytest.raises(
        verifier.ManifestVerificationError,
        match="ATTEMPT_006_IMPLEMENTATION_HASHES_INVALID",
    ):
        manifest_path = batch / "attempt_006/path_c_task_manifest.json"
        verifier.verify_frozen_manifest(
            artifact_root, schema, expected_manifest_sha256=digest(manifest_path.read_bytes()),
            checkout_root=Path(__file__).resolve().parents[2],
            api_root=Path(__file__).resolve().parents[2] / "apps/api",
        )


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_historical_implementation_hash_file_set_must_be_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    metadata_path = batch / "attempt_006/generation_metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    if change == "missing":
        metadata["implementation_sha256"].pop(GENERATION_IMPLEMENTATION_FILES[0])
    else:
        metadata["implementation_sha256"]["scripts/unexpected.py"] = "0" * 64
    _rewrite_metadata_hash_chain(batch, metadata)

    with pytest.raises(
        verifier.ManifestVerificationError,
        match="ATTEMPT_006_IMPLEMENTATION_HASHES_INVALID",
    ):
        manifest_path = batch / "attempt_006/path_c_task_manifest.json"
        verifier.verify_frozen_manifest(
            artifact_root, schema, expected_manifest_sha256=digest(manifest_path.read_bytes()),
            checkout_root=Path(__file__).resolve().parents[2],
            api_root=Path(__file__).resolve().parents[2] / "apps/api",
        )


def test_wrong_sha_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    with pytest.raises(verifier.ManifestVerificationError, match="SHA256_MISMATCH"):
        verifier.verify_frozen_manifest(artifact_root, schema, expected_manifest_sha256="0" * 64)


@pytest.mark.parametrize("count", [39, 41])
def test_non_40_task_manifest_fails_schema_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path, count=count)
    _mock_history(monkeypatch)
    manifest = (
        artifact_root / "v2_1r_d4r_path_c_generation"
        / f"{POPULATION_ID}__{BATCH_ID}"
        / "attempt_006/path_c_task_manifest.json"
    )
    with pytest.raises(verifier.ManifestVerificationError):
        verifier.verify_frozen_manifest(
            artifact_root, schema, expected_manifest_sha256=digest(manifest.read_bytes()),
            checkout_root=Path(__file__).resolve().parents[2],
            api_root=Path(__file__).resolve().parents[2] / "apps/api",
        )


def test_duplicate_task_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    manifest_path = batch / "attempt_006/path_c_task_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["tasks"][1]["task_id"] = manifest["tasks"][0]["task_id"]
    raw = canonical_bytes(manifest) + b"\n"
    manifest_path.write_bytes(raw)
    with pytest.raises(verifier.ManifestVerificationError, match="TASK_IDENTITY_INVALID"):
        verifier.verify_frozen_manifest(
            artifact_root, schema, expected_manifest_sha256=digest(raw),
            checkout_root=Path(__file__).resolve().parents[2],
            api_root=Path(__file__).resolve().parents[2] / "apps/api",
        )


def test_attempt_005_manifest_is_not_a_fallback_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    attempt_006 = batch / "attempt_006"
    attempt_005 = batch / "attempt_005"
    attempt_005.mkdir()
    (attempt_005 / "path_c_task_manifest.json").write_bytes(
        (attempt_006 / "path_c_task_manifest.json").read_bytes()
    )
    (attempt_006 / "path_c_task_manifest.json").unlink()
    with pytest.raises(verifier.ManifestVerificationError, match="FROZEN_MANIFEST_MISSING"):
        verifier.verify_frozen_manifest(artifact_root, schema)


def test_wrong_attempt_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, schema, _ = _fixture(tmp_path)
    _mock_history(monkeypatch)
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    identity_path = batch / "attempt_006/attempt_identity.json"
    identity = json.loads(identity_path.read_bytes())
    identity["attempt_id"] = "0" * 64
    identity_path.write_bytes(canonical_bytes(identity) + b"\n")
    with pytest.raises(
        verifier.ManifestVerificationError,
        match="ATTEMPT_006_PROVENANCE_IDENTITY_MISMATCH",
    ):
        manifest_path = batch / "attempt_006/path_c_task_manifest.json"
        verifier.verify_frozen_manifest(
            artifact_root,
            schema,
            expected_manifest_sha256=digest(manifest_path.read_bytes()),
            checkout_root=Path(__file__).resolve().parents[2],
            api_root=Path(__file__).resolve().parents[2] / "apps/api",
        )
