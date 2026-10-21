"""One-shot PATH-C task-manifest generation and durable artifact freezing.

This module is shared by the host operator entry and fake tests.  It performs
no acquisition work; callers must provide an already-resolved frozen runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

from app.domain.providers import (
    AdapterType,
    CanonicalModelRequest,
    CanonicalModelResult,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from scripts.prospective_acquisition.path_c_attempts import (
    AuthorizedAttempt,
    FrozenAttemptBinding,
    GenerationAttemptError,
    initialize_attempt,
    validate_authorized_attempt,
)
from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    FailureDiagnosticsRecorder,
    FailurePhase,
    PathCFailureDiagnostics,
)
from scripts.prospective_acquisition.seed_amendment import (
    SeedCapability,
    resolve_seed_amendment,
)

BASE_CONTRACT_SHA256 = "b8f8dea0c4ad84d1d17aa25e73b8e53f3ecdde402cd8ca6e8cab8b0cc68f6af3"
BASE_BUNDLE_SHA256 = "2deb55cdf4a6eb8505681af7bd81b0b28b06a85f479c5867c7894d9a865f555b"
BASE_HASH_MANIFEST_SHA256 = "365ccdc6c723bd242590edcd22c6d5748cf80caa010524de69d7e95487c4c497"
SCHEMA_SHA256 = "f908b6397240f5216746edccd26825582223a3d0747b73c9e340c73a1b6d53d3"
AMENDMENT_SHA256 = "0deb0cc4d213fc73b24c02e81f2bfccfc592720556b5bdd24a7a5550ce428333"
AMENDMENT_BUNDLE_SHA256 = "e924bfa2e03e557dd8fc926a691d0456a2249dfd8baa8d3ebfefc54ddd85b825"
EXECUTION_CONTEXT_ID = "8862f286-e707-461e-876f-ec57207802e8"
POPULATION_ID = "v2_1r_d4r_prospective_population_001"
BATCH_ID = "v2_1r_d4r_prospective_population_001_batch_001"
GENERATION_STATE_VERSION = "path-c-generation-state.v1"
GENERATION_CONTRACT_VERSION = "v2_1r_d4r_a51.generic_task_generation.v1+path-c-seed-policy.v1"
GENERATION_IMPLEMENTATION_FILES = (
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


class PathCGenerationBlocked(RuntimeError):
    """A safe fail-closed generation error; ``code`` contains no secret data."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.diagnostics: PathCFailureDiagnostics | None = None


class ProviderTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tasks: list[str] = Field(min_length=40, max_length=40)


class Gateway(Protocol):
    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = ...,
    ) -> CanonicalModelResult: ...


@dataclass(frozen=True)
class GenerationOutcome:
    state: str
    manifest_path: str
    manifest_sha256: str
    provider_dispatch_count: int
    seed_capability: str
    seed_sent: bool
    deterministic_generation_claim: bool
    reused_existing: bool = False


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PathCGenerationBlocked("CONTRACT_ARTIFACT_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise PathCGenerationBlocked("CONTRACT_ARTIFACT_NOT_OBJECT")
    return cast(dict[str, Any], value)


def _verify_a51_bundle(base_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = base_dir / "hash_manifest.json"
    if _sha256(manifest_path.read_bytes()) != BASE_HASH_MANIFEST_SHA256:
        raise PathCGenerationBlocked("BASE_HASH_MANIFEST_MISMATCH")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PathCGenerationBlocked("BASE_BUNDLE_MANIFEST_INVALID")
    payload: dict[str, object] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise PathCGenerationBlocked("BASE_BUNDLE_MANIFEST_INVALID")
        path = (base_dir / relative).resolve()
        if base_dir.resolve() not in path.parents:
            raise PathCGenerationBlocked("BASE_BUNDLE_PATH_INVALID")
        raw = path.read_bytes()
        if _sha256(raw) != expected:
            raise PathCGenerationBlocked("BASE_BUNDLE_FILE_HASH_MISMATCH")
        if relative not in {"hash_manifest.json", "readiness_report.md"}:
            try:
                payload[relative] = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PathCGenerationBlocked("BASE_BUNDLE_JSON_INVALID") from exc
    bundle_hash = _sha256(_canonical_json(payload))
    if (
        bundle_hash != BASE_BUNDLE_SHA256
        or manifest.get("artifact_json_bundle_sha256") != bundle_hash
    ):
        raise PathCGenerationBlocked("BASE_BUNDLE_DIGEST_MISMATCH")
    contract_path = base_dir / "generic_task_generation_contract.json"
    schema_path = base_dir / "task_manifest_schema.json"
    contract_hash = _sha256(contract_path.read_bytes())
    schema_hash = _sha256(schema_path.read_bytes())
    if contract_hash != BASE_CONTRACT_SHA256:
        raise PathCGenerationBlocked("BASE_CONTRACT_HASH_MISMATCH")
    if schema_hash != SCHEMA_SHA256:
        raise PathCGenerationBlocked("SCHEMA_HASH_MISMATCH")
    contract = _read_json(contract_path)
    schema = _read_json(schema_path)
    return contract, schema, bundle_hash


def verify_frozen_contracts(
    root: Path, capability: SeedCapability
) -> tuple[dict[str, Any], dict[str, Any], Any, str]:
    base_dir = root / "artifacts" / "v2_1r_d4r_a51"
    amendment_dir = root / "artifacts" / "v2_1r_d4r_path_c_seed_amendment"
    contract, schema, base_bundle_hash = _verify_a51_bundle(base_dir)
    amendment_path = amendment_dir / "seed_policy_amendment.json"
    amendment_manifest_path = amendment_dir / "hash_manifest.json"
    if _sha256(amendment_path.read_bytes()) != AMENDMENT_SHA256:
        raise PathCGenerationBlocked("AMENDMENT_HASH_MISMATCH")
    amendment_manifest = _read_json(amendment_manifest_path)
    if amendment_manifest.get("amendment_bundle_sha256") != AMENDMENT_BUNDLE_SHA256:
        raise PathCGenerationBlocked("AMENDMENT_BUNDLE_DIGEST_MISMATCH")
    if (
        amendment_manifest.get("artifact_version")
        != "v2_1r_d4r_path_c_seed_amendment.hash_manifest.v1"
        or amendment_manifest.get("hash_algorithm") != "SHA-256"
        or amendment_manifest.get("effective_amendment_version") != "path-c-seed-policy.v1"
    ):
        raise PathCGenerationBlocked("AMENDMENT_HASH_MANIFEST_IDENTITY_MISMATCH")
    amendment_files = amendment_manifest.get("files")
    implementation_files = amendment_manifest.get("implementation_files")
    if not isinstance(amendment_files, dict) or not isinstance(implementation_files, dict):
        raise PathCGenerationBlocked("AMENDMENT_BUNDLE_MANIFEST_INVALID")
    for relative, expected in amendment_files.items():
        path = (amendment_dir / str(relative)).resolve()
        if amendment_dir.resolve() not in path.parents or _sha256(path.read_bytes()) != expected:
            raise PathCGenerationBlocked("AMENDMENT_BUNDLE_FILE_HASH_MISMATCH")
    for relative, expected in implementation_files.items():
        path = (root / str(relative)).resolve()
        if root.resolve() not in path.parents or _sha256(path.read_bytes()) != expected:
            raise PathCGenerationBlocked("AMENDMENT_IMPLEMENTATION_HASH_MISMATCH")
    canonical_amendment = _canonical_json(
        {"artifacts": amendment_files, "implementation_files": implementation_files}
    )
    if _sha256(canonical_amendment) != AMENDMENT_BUNDLE_SHA256:
        raise PathCGenerationBlocked("AMENDMENT_BUNDLE_DIGEST_MISMATCH")
    if amendment_manifest.get("base_a51_bundle_sha256") != base_bundle_hash:
        raise PathCGenerationBlocked("AMENDMENT_BASE_BUNDLE_LINK_MISMATCH")
    if amendment_manifest.get("base_a51_hash_manifest_sha256") != BASE_HASH_MANIFEST_SHA256:
        raise PathCGenerationBlocked("AMENDMENT_BASE_MANIFEST_LINK_MISMATCH")
    amendment_value = _read_json(amendment_path)
    if (
        amendment_value.get("effective_amendment_version") != "path-c-seed-policy.v1"
        or amendment_value.get("current_provider_seed_capability") != "SEED_SUPPORT_NOT_ESTABLISHED"
    ):
        raise PathCGenerationBlocked("SEED_AMENDMENT_POLICY_IDENTITY_MISMATCH")
    try:
        resolution = resolve_seed_amendment(
            base_dir / "generic_task_generation_contract.json", amendment_path, capability
        )
    except (ValueError, OSError) as exc:
        raise PathCGenerationBlocked("SEED_AMENDMENT_RESOLUTION_FAILED") from exc
    return contract, schema, resolution, base_bundle_hash


def _exclusive_write(path: Path, data: bytes) -> None:
    """Create a file exactly once, with file data flushed before visibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise
    _fsync_directory(path.parent)


def _replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - canonical container is Linux
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_provider_identity(binding: object) -> dict[str, Any]:
    if hasattr(binding, "safe_dict"):
        value = binding.safe_dict()
    elif isinstance(binding, dict):
        value = binding
    else:
        raise PathCGenerationBlocked("PROVIDER_BINDING_INVALID")
    required = {
        "profile_id",
        "adapter",
        "model",
        "endpoint_family",
        "profile_version",
        "credential_version_id",
        "credential_version",
        "credential_present",
        "fingerprint_contract",
        "provider_safe_config_fingerprint",
    }
    if not required.issubset(value):
        raise PathCGenerationBlocked("PROVIDER_BINDING_INVALID")
    if value.get("fingerprint_contract") != "provider-safe-config-fingerprint.v1":
        raise PathCGenerationBlocked("PROVIDER_FINGERPRINT_CONTRACT_MISMATCH")
    if value.get("credential_present") is not True:
        raise PathCGenerationBlocked("PROVIDER_CREDENTIAL_UNAVAILABLE")
    fingerprint = value.get("provider_safe_config_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise PathCGenerationBlocked("PROVIDER_FINGERPRINT_INVALID")
    return {key: value[key] for key in sorted(required)}


def _assert_frozen_context(context: object, provider_identity: dict[str, Any]) -> None:
    from scripts.prospective_acquisition.execution_identity import (
        ProspectiveAcquisitionExecutionContext,
        validate_frozen_context,
    )

    try:
        if not isinstance(context, ProspectiveAcquisitionExecutionContext):
            raise TypeError("execution context runtime type mismatch")
        validate_frozen_context(context)
    except Exception as exc:
        raise PathCGenerationBlocked("EXECUTION_CONTEXT_INVALID") from exc
    if (
        getattr(context, "execution_context_id", None) != EXECUTION_CONTEXT_ID
        or getattr(context, "population_id", None) != POPULATION_ID
        or getattr(context, "batch_id", None) != BATCH_ID
        or getattr(context, "owner_scope_source", None) != "AUTHENTICATED_CLIENT_SESSION"
    ):
        raise PathCGenerationBlocked("EXECUTION_CONTEXT_IDENTITY_MISMATCH")
    binding = getattr(context, "provider_binding", None)
    if binding is None:
        raise PathCGenerationBlocked("EXECUTION_CONTEXT_PROVIDER_BINDING_MISSING")
    frozen = _safe_provider_identity(binding)
    if frozen != provider_identity:
        raise PathCGenerationBlocked("EXECUTION_PROVIDER_BINDING_CHANGED")


def _generation_prompt(contract: dict[str, Any]) -> str:
    prompt_lines = contract.get("prompt_contract")
    if (
        not isinstance(prompt_lines, list)
        or not prompt_lines
        or not all(isinstance(item, str) for item in prompt_lines)
    ):
        raise PathCGenerationBlocked("GENERATION_PROMPT_CONTRACT_INVALID")
    return "\n".join(cast(list[str], prompt_lines))


def _provider_response_schema() -> dict[str, object]:
    # Frozen prompt contract requests 40 strings; task IDs are assigned locally.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tasks"],
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": 40,
                "maxItems": 40,
                "items": {"type": "string", "minLength": 1},
            }
        },
    }


def _structural_task_validation(tasks: list[str]) -> None:
    if len(tasks) != 40:
        raise PathCGenerationBlocked("TASK_COUNT_INVALID")
    normalized = [re.sub(r"\s+", " ", text).strip().casefold() for text in tasks]
    if any(not item for item in normalized):
        raise PathCGenerationBlocked("TASK_TEXT_EMPTY")
    if len(set(normalized)) != 40:
        raise PathCGenerationBlocked("STRUCTURAL_DUPLICATE_TASK")


def _read_existing_success(state_path: Path, manifest_path: Path) -> GenerationOutcome:
    state = _read_json(state_path)
    if state.get("state") != "SUCCEEDED_AND_FROZEN":
        raise PathCGenerationBlocked("GENERATION_AUTHORIZATION_ALREADY_CONSUMED")
    if not manifest_path.is_file():
        raise PathCGenerationBlocked("FROZEN_SUCCESS_MANIFEST_MISSING")
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = _sha256(manifest_bytes)
    if manifest_hash != state.get("manifest_sha256"):
        raise PathCGenerationBlocked("FROZEN_MANIFEST_HASH_MISMATCH")
    output_dir = state_path.parent
    for filename, key in (
        ("provider_response.raw.json", "raw_provider_response_sha256"),
        ("generation_metadata.json", "metadata_sha256"),
        ("hash_manifest.json", "hash_manifest_sha256"),
    ):
        path = output_dir / filename
        if not path.is_file() or _sha256(path.read_bytes()) != state.get(key):
            raise PathCGenerationBlocked("FROZEN_GENERATION_ARTIFACT_HASH_MISMATCH")
    hashes = _read_json(output_dir / "hash_manifest.json").get("files")
    if not isinstance(hashes, dict):
        raise PathCGenerationBlocked("FROZEN_GENERATION_HASH_MANIFEST_INVALID")
    if (
        hashes.get("path_c_task_manifest.json") != manifest_hash
        or hashes.get("provider_response.raw.json") != state.get("raw_provider_response_sha256")
        or hashes.get("generation_metadata.json") != state.get("metadata_sha256")
    ):
        raise PathCGenerationBlocked("FROZEN_GENERATION_HASH_MANIFEST_MISMATCH")
    return GenerationOutcome(
        state="SUCCEEDED_AND_FROZEN",
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_hash,
        provider_dispatch_count=1,
        seed_capability=str(state.get("seed_capability")),
        seed_sent=bool(state.get("seed_sent")),
        deterministic_generation_claim=bool(state.get("deterministic_generation_claim")),
        reused_existing=True,
    )


def return_existing_frozen_success(output_dir: Path) -> GenerationOutcome:
    """Recover a verified success without Provider/DB access or redispatch."""
    return _read_existing_success(
        output_dir / "generation_state.json",
        output_dir / "path_c_task_manifest.json",
    )


async def execute_path_c_generation(
    *,
    root: Path,
    application_root: Path | None = None,
    output_dir: Path,
    context: object,
    provider_binding: object,
    binding_recheck: Any,
    gateway: Gateway,
    api_key: Any,
    base_url: str,
    adapter_type: Any,
    context_manifest_id: UUID,
    physical_budget: Any,
    raw_response_path: Path,
    failure_diagnostics: FailureDiagnosticsRecorder | None = None,
    authorized_attempt: AuthorizedAttempt | None = None,
) -> GenerationOutcome:
    """Only an exact disk-backed, separately authorized numbered attempt may run."""
    if authorized_attempt is None:
        raise GenerationAttemptError("SEPARATE_ATTEMPT_AUTHORIZATION_REQUIRED")
    if output_dir != authorized_attempt.location:
        raise GenerationAttemptError("ATTEMPT_OUTPUT_LOCATION_MISMATCH")
    binding = frozen_attempt_binding(context, provider_binding)
    validate_authorized_attempt(authorized_attempt, binding)
    if (
        getattr(physical_budget, "maximum", None) != 1
        or getattr(physical_budget, "dispatched", None) != 0
        or getattr(physical_budget, "blocked_attempts", None) != 0
        or getattr(physical_budget, "remaining", None) != 1
    ):
        raise GenerationAttemptError("ATTEMPT_SINGLE_CALL_BUDGET_INVALID")
    return await _execute_guarded_core(
        root=root,
        application_root=application_root,
        output_dir=output_dir,
        context=context,
        provider_binding=provider_binding,
        binding_recheck=binding_recheck,
        gateway=gateway,
        api_key=api_key,
        base_url=base_url,
        adapter_type=adapter_type,
        context_manifest_id=context_manifest_id,
        physical_budget=physical_budget,
        raw_response_path=raw_response_path,
        failure_diagnostics=failure_diagnostics,
        authorized_attempt=authorized_attempt,
    )


def frozen_attempt_binding(context: object, provider_binding: object) -> FrozenAttemptBinding:
    identity = _safe_provider_identity(provider_binding)
    _assert_frozen_context(context, identity)
    return FrozenAttemptBinding(
        population_id=POPULATION_ID,
        batch_id=BATCH_ID,
        execution_context_id=UUID(EXECUTION_CONTEXT_ID),
        execution_context_hash=str(getattr(context, "context_hash", "")),
        generation_contract_version=GENERATION_CONTRACT_VERSION,
        generation_contract_sha256=BASE_CONTRACT_SHA256,
        task_schema_sha256=SCHEMA_SHA256,
        seed_amendment_sha256=AMENDMENT_SHA256,
        provider_safe_config_fingerprint=str(identity["provider_safe_config_fingerprint"]),
        provider_binding_sha256=_sha256(_canonical_json(identity)),
    )


async def _execute_guarded_core(
    *,
    root: Path,
    application_root: Path | None = None,
    output_dir: Path,
    context: object,
    provider_binding: object,
    binding_recheck: Any,
    gateway: Gateway,
    api_key: Any,
    base_url: str,
    adapter_type: Any,
    context_manifest_id: UUID,
    physical_budget: Any,
    raw_response_path: Path,
    failure_diagnostics: FailureDiagnosticsRecorder | None = None,
    authorized_attempt: AuthorizedAttempt | None = None,
) -> GenerationOutcome:
    """Shared generation core; unnumbered mode exists only for isolated legacy tests."""
    # A historical reservation wins before any new preflight or diagnostic work.
    # This also protects legacy Attempt 001, which has no detailed diagnostics.
    if authorized_attempt is None and (output_dir / "generation_state.json").exists():
        return return_existing_frozen_success(output_dir)
    observations = failure_diagnostics or FailureDiagnosticsRecorder()
    try:
        return await _execute_path_c_generation(
            root=root,
            application_root=application_root,
            output_dir=output_dir,
            context=context,
            provider_binding=provider_binding,
            binding_recheck=binding_recheck,
            gateway=gateway,
            api_key=api_key,
            base_url=base_url,
            adapter_type=adapter_type,
            context_manifest_id=context_manifest_id,
            physical_budget=physical_budget,
            raw_response_path=raw_response_path,
            observations=observations,
            authorized_attempt=authorized_attempt,
        )
    except Exception as exc:
        if isinstance(exc, GenerationAttemptError):
            raise
        blocked = (
            exc
            if isinstance(exc, PathCGenerationBlocked)
            else PathCGenerationBlocked(type(exc).__name__.upper())
        )
        if blocked.diagnostics is None:
            # Reservation safety must never depend on diagnostic serialization.
            with suppress(Exception):
                blocked.diagnostics = observations.snapshot(
                    exc,
                    physical_outbound_count=getattr(physical_budget, "dispatched", 0),
                    blocked_additional_outbound_count=getattr(
                        physical_budget, "blocked_attempts", 0
                    ),
                )
        if blocked is exc:
            raise
        raise blocked from exc


async def _execute_path_c_generation(
    *,
    root: Path,
    application_root: Path | None,
    output_dir: Path,
    context: object,
    provider_binding: object,
    binding_recheck: Any,
    gateway: Gateway,
    api_key: Any,
    base_url: str,
    adapter_type: Any,
    context_manifest_id: UUID,
    physical_budget: Any,
    raw_response_path: Path,
    observations: FailureDiagnosticsRecorder,
    authorized_attempt: AuthorizedAttempt | None,
) -> GenerationOutcome:
    """Validate, reserve once, dispatch once, and freeze a valid 40-task manifest."""
    provider_identity = _safe_provider_identity(provider_binding)
    _assert_frozen_context(context, provider_identity)
    contract, schema, seed_resolution, base_bundle_hash = verify_frozen_contracts(
        root, "SEED_SUPPORT_NOT_ESTABLISHED"
    )
    if seed_resolution.send_seed or seed_resolution.seed_value is not None:
        raise PathCGenerationBlocked("SEED_POLICY_RESOLUTION_UNSAFE")
    if contract.get("task_count") != 40 or contract.get("generation_calls_max") != 1:
        raise PathCGenerationBlocked("BASE_GENERATION_LIMIT_MISMATCH")
    app_root = application_root if application_root is not None else root
    implementation_hashes: dict[str, str] = {}
    try:
        for relative in GENERATION_IMPLEMENTATION_FILES:
            source_root = app_root if relative.startswith("apps/api/") else root
            implementation_hashes[relative] = _sha256((source_root / relative).read_bytes())
    except OSError as exc:
        # All implementation files used by output validation/metadata must be
        # visible before a one-shot Provider reservation is consumed.
        raise PathCGenerationBlocked("GENERATION_RUNTIME_FILE_UNAVAILABLE") from exc
    amendment_dir = root / "artifacts" / "v2_1r_d4r_path_c_seed_amendment"
    amendment_manifest = _read_json(amendment_dir / "hash_manifest.json")
    amendment_hash = _sha256((amendment_dir / "seed_policy_amendment.json").read_bytes())
    schema_hash = _sha256((root / "artifacts/v2_1r_d4r_a51/task_manifest_schema.json").read_bytes())
    prompt = _generation_prompt(contract)
    prompt_hash = _sha256(prompt.encode("utf-8"))
    state_path = output_dir / "generation_state.json"
    raw_response_path = output_dir / "provider_response.raw.json"
    manifest_path = output_dir / "path_c_task_manifest.json"

    if authorized_attempt is None and state_path.exists():
        return _read_existing_success(state_path, manifest_path)
    if any(
        (output_dir / name).exists()
        for name in (
            "provider_response.raw.json",
            "path_c_task_manifest.json",
            "generation_metadata.json",
            "hash_manifest.json",
        )
    ):
        raise PathCGenerationBlocked("GENERATION_PARTIAL_ARTIFACTS_WITHOUT_STATE")

    # Construct and validate the complete local request before consuming the
    # durable one-shot reservation. No request field is sourced from operator
    # input, and the effective seed policy intentionally contributes no seed.
    from app.domain.providers import CanonicalModelRequest, ContentPart

    observations.phase = FailurePhase.REQUEST_CONSTRUCTION
    request = CanonicalModelRequest(
        task_kind="path_c_generic_task_generation",
        role="research_task_generation",
        model=str(provider_identity["model"]),
        instructions=prompt,
        content_parts=(
            ContentPart(kind="text", value="Generate the requested generic research tasks."),
        ),
        response_contract=_provider_response_schema(),
        generation_parameters={"temperature": 0},
        max_output_tokens=8192,
        stream=False,
        context_manifest_id=context_manifest_id,
        metadata={"node": "path_c_task_generation", "retry_mode": "none"},
    )

    # This callback performs the final live owner-scoped Provider binding check.
    observations.phase = FailurePhase.PROVIDER_RESOLUTION
    try:
        await binding_recheck()
    except Exception as exc:
        raise PathCGenerationBlocked("EXECUTION_PROVIDER_BINDING_CHANGED") from exc

    attempt_state: dict[str, Any] = {}
    if authorized_attempt is not None:
        attempt_state = initialize_attempt(
            authorized_attempt, frozen_attempt_binding(context, provider_binding)
        )

    reservation = {
        **attempt_state,
        "state_version": GENERATION_STATE_VERSION,
        "state": "DISPATCH_RESERVED",
        "dispatch_count": 1,
        "execution_context_id": EXECUTION_CONTEXT_ID,
        "population_id": POPULATION_ID,
        "batch_id": BATCH_ID,
        "contract_version": GENERATION_CONTRACT_VERSION,
        "seed_capability": seed_resolution.capability,
        "seed_sent": False,
        "deterministic_generation_claim": False,
        "provider_binding": provider_identity,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "amendment_sha256": amendment_hash,
        "reserved_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        if authorized_attempt is None:
            _exclusive_write(state_path, _canonical_json(reservation) + b"\n")
        else:
            # Create-once receipt consumes THIS authorization before any mutable
            # state replacement or HTTP work. A crash here cannot reopen it.
            _exclusive_write(
                output_dir / "dispatch_reservation.json", _canonical_json(reservation) + b"\n"
            )
            _replace_json(state_path, reservation)
    except FileExistsError as exc:
        raise PathCGenerationBlocked("GENERATION_AUTHORIZATION_ALREADY_CONSUMED") from exc
    observations.reservation_persisted_at_utc = datetime.now(UTC)

    failure_code = "GENERATION_FAILED"
    try:
        observations.phase = FailurePhase.GATEWAY
        observations.gateway_entered_at_utc = datetime.now(UTC)
        result = await gateway.generate_structured(
            adapter_type=adapter_type,
            base_url=base_url,
            api_key=api_key,
            request=request,
            allow_regeneration=False,
        )
        observations.phase = FailurePhase.OUTPUT_VALIDATION
        if getattr(physical_budget, "dispatched", None) != 1:
            raise PathCGenerationBlocked("PHYSICAL_PROVIDER_OUTBOUND_COUNT_INVALID")
        if getattr(physical_budget, "blocked_attempts", 0) != 0:
            raise PathCGenerationBlocked("PROVIDER_FALLBACK_ATTEMPT_BLOCKED")
        if not raw_response_path.is_file():
            raise PathCGenerationBlocked("RAW_PROVIDER_RESPONSE_NOT_FROZEN")
        raw_hash = _sha256(raw_response_path.read_bytes())

        parsed = getattr(result, "parsed_object", None)
        if not isinstance(parsed, dict):
            raise PathCGenerationBlocked("PROVIDER_JSON_OBJECT_INVALID")
        try:
            provider_output = ProviderTaskResponse.model_validate(parsed)
        except ValidationError as exc:
            raise PathCGenerationBlocked("PROVIDER_TASK_RESPONSE_INVALID") from exc
        tasks = provider_output.tasks
        _structural_task_validation(tasks)
        manifest = {
            "manifest_version": "prospective-generic-task-manifest.v1",
            "task_count": 40,
            "tasks": [
                {"task_id": f"task-{index:03d}", "task_text": task}
                for index, task in enumerate(tasks, start=1)
            ],
        }
        from scripts.prospective_acquisition.schema_validation import validate_instance

        errors = validate_instance(schema, manifest)
        if errors:
            raise PathCGenerationBlocked("FROZEN_TASK_MANIFEST_SCHEMA_INVALID")
        manifest_bytes = _canonical_json(manifest) + b"\n"
        manifest_hash = _sha256(manifest_bytes)
        _exclusive_write(manifest_path, manifest_bytes)
        metadata = {
            **{
                key: attempt_state[key]
                for key in (
                    "attempt_number",
                    "attempt_id",
                    "authorization_id",
                    "authorization_hash",
                    "authorization_file_sha256",
                    "attempt_identity_sha256",
                )
                if key in attempt_state
            },
            "artifact_version": "path-c-generation-metadata.v1",
            "execution_context_id": EXECUTION_CONTEXT_ID,
            "population_id": POPULATION_ID,
            "batch_id": BATCH_ID,
            "contract_version": GENERATION_CONTRACT_VERSION,
            "base_a51_bundle_sha256": base_bundle_hash,
            "seed_amendment_sha256": amendment_hash,
            "seed_amendment_bundle_sha256": amendment_manifest["amendment_bundle_sha256"],
            "seed_capability": seed_resolution.capability,
            "seed_sent": False,
            "deterministic_generation_claim": False,
            "provider_binding": provider_identity,
            "implementation_sha256": {
                relative: implementation_hashes[relative]
                for relative in GENERATION_IMPLEMENTATION_FILES
            },
            "generation_prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
            "raw_provider_response_sha256": raw_hash,
            "manifest_sha256": manifest_hash,
            "task_count": 40,
            "generation_dispatch_count": 1,
            "max_output_tokens": 8192,
            "provider_request_id": getattr(result, "provider_request_id", None),
            "usage": getattr(getattr(result, "usage", None), "model_dump", lambda **_: {})(
                mode="json"
            ),
            "finish_reason": getattr(result, "finish_reason", None),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "acquisition_started": False,
            "search_count": 0,
            "reader_count": 0,
            "evidence_extractor_count": 0,
            "database_writes": 0,
            "holdout_access": 0,
        }
        metadata_path = output_dir / "generation_metadata.json"
        metadata_bytes = _canonical_json(metadata) + b"\n"
        _exclusive_write(metadata_path, metadata_bytes)
        hash_manifest = {
            "artifact_version": "path-c-generation-hash-manifest.v1",
            "hash_algorithm": "SHA-256",
            "files": {
                "provider_response.raw.json": raw_hash,
                "path_c_task_manifest.json": manifest_hash,
                "generation_metadata.json": _sha256(metadata_bytes),
            },
        }
        hash_manifest_bytes = _canonical_json(hash_manifest) + b"\n"
        _exclusive_write(output_dir / "hash_manifest.json", hash_manifest_bytes)
        success = {
            **reservation,
            "state": "SUCCEEDED_AND_FROZEN",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "raw_provider_response_sha256": raw_hash,
            "manifest_sha256": manifest_hash,
            "metadata_sha256": _sha256(metadata_bytes),
            "hash_manifest_sha256": _sha256(hash_manifest_bytes),
        }
        _replace_json(state_path, success)
        return GenerationOutcome(
            state="SUCCEEDED_AND_FROZEN",
            manifest_path=str(manifest_path),
            manifest_sha256=manifest_hash,
            provider_dispatch_count=1,
            seed_capability=seed_resolution.capability,
            seed_sent=False,
            deterministic_generation_claim=False,
        )
    except BaseException as exc:
        if isinstance(exc, PathCGenerationBlocked):
            failure_code = exc.code
        elif isinstance(exc, Exception):
            failure_code = type(exc).__name__.upper()
        failure_time = datetime.now(UTC)
        failed: dict[str, object] = {
            **reservation,
            "state": "CONSUMED_FAILED",
            "failure_code": failure_code,
            "failed_at_utc": failure_time.isoformat(),
            "physical_outbound_count": getattr(physical_budget, "dispatched", 0),
            "blocked_additional_outbound_count": getattr(physical_budget, "blocked_attempts", 0),
        }
        diagnostic_snapshot: PathCFailureDiagnostics | None = None
        try:
            diagnostic_snapshot = observations.snapshot(
                exc,
                physical_outbound_count=getattr(physical_budget, "dispatched", 0),
                blocked_additional_outbound_count=getattr(physical_budget, "blocked_attempts", 0),
                recorded_at=failure_time,
            )
            failed["failure_diagnostics"] = diagnostic_snapshot.model_dump(mode="json")
        except Exception:
            failed["failure_diagnostics_status"] = "UNAVAILABLE"
        # The existing fsync + atomic replace commits diagnostics and terminal
        # state together. If this fails, DISPATCH_RESERVED still forbids rerun.
        _replace_json(state_path, failed)
        if isinstance(exc, PathCGenerationBlocked):
            exc.diagnostics = diagnostic_snapshot
            raise
        blocked = PathCGenerationBlocked(failure_code)
        blocked.diagnostics = diagnostic_snapshot
        raise blocked from exc
