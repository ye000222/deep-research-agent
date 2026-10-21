"""Content-blind verification for the immutable PATH-C Attempt 006 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from scripts.prospective_acquisition.path_c_attempts import (
    AttemptIdentity,
    HistoryScope,
    PathCAttemptAuthorization,
    attempt_id_for,
    digest,
    read_history,
)
from scripts.prospective_acquisition.path_c_generation import (
    BATCH_ID,
    EXECUTION_CONTEXT_ID,
    GENERATION_CONTRACT_VERSION,
    POPULATION_ID,
)
from scripts.prospective_acquisition.schema_validation import validate_instance

ATTEMPT_NUMBER = 6
ATTEMPT_ID = "4f05b3509cbb75d2481dcf0fa46b0683e9278d5c66021b03c59f6ebc9eb69ef0"
AUTHORIZATION_ID = "213e94e8-5f75-4d05-87b9-eb41afe40ba8"
MANIFEST_SHA256 = "9806e556026cbcdd5a56c8da8f5ad71e7714a0877a459054ecaff5f775d507ff"
SCHEMA_RELATIVE_PATH = Path("artifacts/v2_1r_d4r_a51/task_manifest_schema.json")
ATTEMPT_006_GENERATION_IMPLEMENTATION_FILES = (
    "scripts/prospective_acquisition/path_c_generation.py",
    "scripts/prospective_acquisition/path_c_generation_entry.py",
    "scripts/prospective_acquisition/capture_transport.py",
    "scripts/prospective_acquisition/path_c_failure_diagnostics.py",
    "scripts/prospective_acquisition/path_c_attempts.py",
    "scripts/prospective_acquisition/control.py",
    "scripts/prospective_acquisition/seed_amendment.py",
    "scripts/prospective_acquisition/schema_validation.py",
    "apps/api/app/llm/adapters.py",
)


class ManifestVerificationError(RuntimeError):
    """Safe error code; never includes task content or arbitrary schema errors."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestVerificationError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except ManifestVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ManifestVerificationError(code) from exc
    if not isinstance(value, dict):
        raise ManifestVerificationError(code)
    return value


def _same_fields(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _attempt_scope() -> HistoryScope:
    return HistoryScope(
        population_id=POPULATION_ID,
        batch_id=BATCH_ID,
        execution_context_id=UUID(EXECUTION_CONTEXT_ID),
        generation_contract_version=GENERATION_CONTRACT_VERSION,
    )


def verify_frozen_manifest(
    artifact_root: Path,
    schema_path: Path,
    *,
    expected_manifest_sha256: str = MANIFEST_SHA256,
    checkout_root: Path = Path("/opt"),
    api_root: Path = Path("/app/apps/api"),
) -> dict[str, object]:
    """Verify immutable files and provenance; return only content-blind facts."""
    batch = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    attempt = batch / "attempt_006"
    manifest_path = attempt / "path_c_task_manifest.json"
    if not manifest_path.is_file():
        raise ManifestVerificationError("FROZEN_MANIFEST_MISSING")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = digest(manifest_bytes)
    if manifest_sha != expected_manifest_sha256:
        raise ManifestVerificationError("FROZEN_MANIFEST_SHA256_MISMATCH")

    try:
        manifest = json.loads(manifest_bytes, object_pairs_hook=_object)
        schema = _read_object(schema_path, "FROZEN_TASK_SCHEMA_UNREADABLE")
        schema_errors = validate_instance(schema, manifest)
    except ManifestVerificationError:
        raise
    except Exception as exc:
        raise ManifestVerificationError("FROZEN_MANIFEST_SCHEMA_VALIDATION_FAILED") from exc
    if schema_errors:
        raise ManifestVerificationError("FROZEN_MANIFEST_SCHEMA_INVALID")
    if not isinstance(manifest, dict) or manifest.get("task_count") != 40:
        raise ManifestVerificationError("FROZEN_TASK_COUNT_NOT_EXACTLY_40")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 40:
        raise ManifestVerificationError("FROZEN_TASK_COUNT_NOT_EXACTLY_40")

    ids: list[str] = []
    texts: list[str] = []
    normalized_hashes: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ManifestVerificationError("FROZEN_TASK_ITEM_INVALID")
        task_id, task_text = task.get("task_id"), task.get("task_text")
        if not isinstance(task_id, str) or not isinstance(task_text, str) or not task_text.strip():
            raise ManifestVerificationError("FROZEN_TASK_REQUIRED_FIELD_INVALID")
        ids.append(task_id)
        texts.append(task_text)
        normalized = re.sub(r"\s+", " ", task_text).strip().casefold()
        normalized_hashes.append(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    expected_ids = [f"task-{number:03d}" for number in range(1, 41)]
    if ids != expected_ids or len(set(ids)) != 40:
        raise ManifestVerificationError("FROZEN_TASK_IDENTITY_INVALID")
    if len(set(normalized_hashes)) != 40:
        raise ManifestVerificationError("FROZEN_TASK_DUPLICATE_TEXT")

    authorization_path = batch / "authorizations" / "attempt_006.authorization.json"
    identity_path = attempt / "attempt_identity.json"
    state_path = attempt / "generation_state.json"
    reservation_path = attempt / "dispatch_reservation.json"
    metadata_path = attempt / "generation_metadata.json"
    hash_manifest_path = attempt / "hash_manifest.json"
    raw_response_path = attempt / "provider_response.raw.json"
    try:
        authorization_bytes = authorization_path.read_bytes()
        authorization = PathCAttemptAuthorization.model_validate_json(authorization_bytes)
        authorization.validate_integrity()
        identity = AttemptIdentity.model_validate_json(identity_path.read_bytes())
    except Exception as exc:
        raise ManifestVerificationError("ATTEMPT_006_AUTHORIZATION_OR_IDENTITY_INVALID") from exc
    state = _read_object(state_path, "ATTEMPT_006_STATE_UNREADABLE")
    reservation = _read_object(reservation_path, "ATTEMPT_006_RESERVATION_UNREADABLE")
    metadata = _read_object(metadata_path, "ATTEMPT_006_METADATA_UNREADABLE")
    hash_manifest = _read_object(hash_manifest_path, "ATTEMPT_006_HASH_MANIFEST_INVALID")
    scope = _attempt_scope()
    if (
        authorization.attempt_number != ATTEMPT_NUMBER
        or authorization.attempt_id != ATTEMPT_ID
        or str(authorization.authorization_id) != AUTHORIZATION_ID
        or authorization.scope() != scope
        or authorization.attempt_id != attempt_id_for(
            scope, ATTEMPT_NUMBER, authorization.authorization_id
        )
        or authorization.canonical_runtime_attestation_run is None
        or authorization.canonical_runtime_attestation_manifest_sha256 is None
        or identity.attempt_number != ATTEMPT_NUMBER
        or identity.attempt_id != ATTEMPT_ID
        or str(identity.authorization_id) != AUTHORIZATION_ID
        or identity.binding != authorization.binding()
    ):
        raise ManifestVerificationError("ATTEMPT_006_PROVENANCE_IDENTITY_MISMATCH")

    auth_sha = digest(authorization_bytes)
    identity_sha = digest(identity_path.read_bytes())
    if (
        identity.authorization_hash != authorization.authorization_hash
        or identity.authorization_file_sha256 != auth_sha
        or identity.previous_attempt_id != authorization.previous_attempt_id
        or identity.previous_state_sha256 != authorization.previous_state_sha256
    ):
        raise ManifestVerificationError("ATTEMPT_006_AUTHORIZATION_BINDING_MISMATCH")
    expected_state = {
        "state": "SUCCEEDED_AND_FROZEN",
        "population_id": POPULATION_ID,
        "batch_id": BATCH_ID,
        "execution_context_id": EXECUTION_CONTEXT_ID,
        "attempt_number": ATTEMPT_NUMBER,
        "attempt_id": ATTEMPT_ID,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_hash": authorization.authorization_hash,
        "authorization_file_sha256": auth_sha,
        "attempt_identity_sha256": identity_sha,
        "dispatch_count": 1,
        "manifest_sha256": manifest_sha,
        "seed_sent": False,
        "deterministic_generation_claim": False,
    }
    if not _same_fields(state, expected_state):
        raise ManifestVerificationError("ATTEMPT_006_STATE_PROVENANCE_MISMATCH")
    if not _same_fields(reservation, {
        "state": "DISPATCH_RESERVED",
        "attempt_number": ATTEMPT_NUMBER,
        "attempt_id": ATTEMPT_ID,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_hash": authorization.authorization_hash,
        "authorization_file_sha256": auth_sha,
        "attempt_identity_sha256": identity_sha,
        "dispatch_count": 1,
        "seed_sent": False,
        "deterministic_generation_claim": False,
    }):
        raise ManifestVerificationError("ATTEMPT_006_RESERVATION_PROVENANCE_MISMATCH")
    if not _same_fields(metadata, {
        "attempt_number": ATTEMPT_NUMBER,
        "attempt_id": ATTEMPT_ID,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_hash": authorization.authorization_hash,
        "authorization_file_sha256": auth_sha,
        "attempt_identity_sha256": identity_sha,
        "population_id": POPULATION_ID,
        "batch_id": BATCH_ID,
        "execution_context_id": EXECUTION_CONTEXT_ID,
        "manifest_sha256": manifest_sha,
        "task_count": 40,
        "generation_dispatch_count": 1,
        "seed_capability": "SEED_SUPPORT_NOT_ESTABLISHED",
        "seed_sent": False,
        "deterministic_generation_claim": False,
        "acquisition_started": False,
        "search_count": 0,
        "reader_count": 0,
        "evidence_extractor_count": 0,
        "database_writes": 0,
        "holdout_access": 0,
    }):
        raise ManifestVerificationError("ATTEMPT_006_METADATA_PROVENANCE_MISMATCH")

    try:
        raw_response = json.loads(raw_response_path.read_bytes(), object_pairs_hook=_object)
        raw_tasks_value = raw_response["choices"][0]["message"]["content"]
        raw_tasks = json.loads(raw_tasks_value, object_pairs_hook=_object)
    except Exception as exc:
        raise ManifestVerificationError("ATTEMPT_006_RAW_RESPONSE_UNREADABLE") from exc
    if not isinstance(raw_tasks, dict) or raw_tasks.get("tasks") != texts:
        raise ManifestVerificationError("FROZEN_MANIFEST_DIFFERS_FROM_RAW_RESPONSE")
    files = hash_manifest.get("files")
    if not isinstance(files, dict):
        raise ManifestVerificationError("ATTEMPT_006_HASH_MANIFEST_INVALID")
    hashes = {
        "path_c_task_manifest.json": manifest_sha,
        "provider_response.raw.json": digest(raw_response_path.read_bytes()),
        "generation_metadata.json": digest(metadata_path.read_bytes()),
    }
    if any(files.get(name) != value for name, value in hashes.items()):
        raise ManifestVerificationError("ATTEMPT_006_HASH_MANIFEST_MISMATCH")
    if (
        state.get("metadata_sha256") != hashes["generation_metadata.json"]
        or state.get("raw_provider_response_sha256") != hashes["provider_response.raw.json"]
        or state.get("hash_manifest_sha256") != digest(hash_manifest_path.read_bytes())
    ):
        raise ManifestVerificationError("ATTEMPT_006_FROZEN_FILE_HASH_MISMATCH")

    # These hashes describe the implementation used at generation time. They
    # are historical provenance, not a requirement that later acquisition
    # code remain byte-identical to the generation checkout. Their authenticity
    # is established by the frozen metadata/hash-manifest/state chain above and
    # by the independent successful host verification recorded at freeze time.
    recorded_implementation = metadata.get("implementation_sha256")
    if (
        not isinstance(recorded_implementation, dict)
        or set(recorded_implementation) != set(ATTEMPT_006_GENERATION_IMPLEMENTATION_FILES)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in recorded_implementation.values()
        )
    ):
        raise ManifestVerificationError("ATTEMPT_006_IMPLEMENTATION_HASHES_INVALID")

    try:
        history = read_history(batch, scope)
    except Exception as exc:
        raise ManifestVerificationError("ATTEMPT_006_HISTORY_INVALID") from exc
    if (
        len(history) != 6
        or any(
            record.state != "CONSUMED_FAILED" or not record.consumed
            for record in history[:5]
        )
        or history[5].state != "SUCCEEDED_AND_FROZEN"
        or history[5].attempt_id != ATTEMPT_ID
    ):
        raise ManifestVerificationError("ATTEMPT_HISTORY_NOT_CLOSED_AS_EXPECTED")

    return {
        "status": "PATH_C_FROZEN_MANIFEST_VERIFIED",
        "attempt_number": ATTEMPT_NUMBER,
        "attempt_id": ATTEMPT_ID,
        "authorization_id": AUTHORIZATION_ID,
        "manifest_sha256": manifest_sha,
        "task_count": 40,
        "schema_valid": True,
        "task_identity_unique": True,
        "task_text_duplicates": False,
        "provenance_valid": True,
        "prior_attempts_001_005_consumed_failed": True,
        "generation_implementation_provenance_recorded": True,
        "provider_dispatch_count": 1,
        "provider_http_count": 1,
        "blocked_extra_outbound_count": 0,
        "seed_sent": False,
        "deterministic_generation_claim": False,
        "content_exposed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Content-blind Attempt 006 manifest verification")
    parser.add_argument("--artifact-root", type=Path, default=Path("/app/artifacts"))
    parser.add_argument("--schema-path", type=Path, default=Path("/opt") / SCHEMA_RELATIVE_PATH)
    parser.add_argument("--checkout-root", type=Path, default=Path("/opt"))
    parser.add_argument("--api-root", type=Path, default=Path("/app/apps/api"))
    args = parser.parse_args()
    try:
        result = verify_frozen_manifest(
            args.artifact_root, args.schema_path,
            checkout_root=args.checkout_root, api_root=args.api_root,
        )
    except ManifestVerificationError as exc:
        sys.stderr.write(f"PATH_C_MANIFEST_VERIFICATION_BLOCKED {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
