"""Host-only authorization publication; never imports an execution entry."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from app.llm import model_http_policy

from scripts.prospective_acquisition import path_c_attempt_003_structure as structure
from scripts.prospective_acquisition.path_c_attempts import (
    AttemptRecord,
    FrozenAttemptBinding,
    GenerationAttemptError,
    HistoryScope,
    PathCAttemptAuthorization,
    _read,
    _read_authorizations,
    attempt_id_for,
    canonical_bytes,
    digest,
    read_history,
)
from scripts.prospective_acquisition.path_c_generation import (
    AMENDMENT_SHA256,
    BASE_CONTRACT_SHA256,
    BATCH_ID,
    EXECUTION_CONTEXT_ID,
    GENERATION_CONTRACT_VERSION,
    POPULATION_ID,
    SCHEMA_SHA256,
    _provider_response_schema,
    _safe_provider_identity,
    verify_frozen_contracts,
)

PARENT_SHA256 = "7d9b0da465a30c39a4b0733d1d851fc959dd99c55c7fa916d5078dd95497114c"
PROVIDER_SHA256 = "32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2"
CANONICAL_ATTESTATION_RUN: Literal["run_20261013T014431260Z_2a030977"] = (
    "run_20261013T014431260Z_2a030977"
)
CANONICAL_ARTIFACT_VOLUME_ROOT = Path("/app/artifacts")
CANONICAL_CHECKOUT_ROOT = Path("/opt")


def scope() -> HistoryScope:
    return HistoryScope(
        population_id=POPULATION_ID,
        batch_id=BATCH_ID,
        execution_context_id=UUID(EXECUTION_CONTEXT_ID),
        generation_contract_version=GENERATION_CONTRACT_VERSION,
    )


def load_binding(artifact_root: Path, checkout: Path) -> FrozenAttemptBinding:
    """Bind the authoritative frozen safe artifact, not a discovered Provider."""
    verify_frozen_contracts(checkout, "SEED_SUPPORT_NOT_ESTABLISHED")
    key = hashlib.sha256(
        json.dumps([POPULATION_ID, BATCH_ID], separators=(",", ":")).encode()
    ).hexdigest()
    path = artifact_root / "v2_1r_d4r_a6_closure" / "execution_contexts" / f"{key}.json"
    if path.is_symlink():
        raise GenerationAttemptError("CONTEXT_SYMLINK_REJECTED")
    artifact, _ = _read(path)
    if any(artifact.get(k) != v for k, v in {
        "execution_context_id": EXECUTION_CONTEXT_ID,
        "population_id": POPULATION_ID,
        "batch_id": BATCH_ID,
        "owner_scope_source": "AUTHENTICATED_CLIENT_SESSION",
    }.items()):
        raise GenerationAttemptError("FROZEN_CONTEXT_IDENTITY_MISMATCH")
    provider = _safe_provider_identity(artifact["provider_binding"])
    if any(provider.get(k) != v for k, v in {
        "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
        "adapter": "openai_compatible_chat",
        "model": "deepseek-flash",
        "endpoint_family": "https://api.deepseek.com",
        "profile_version": 23,
        "credential_version": 4,
        "provider_safe_config_fingerprint": PROVIDER_SHA256,
    }.items()):
        raise GenerationAttemptError("FROZEN_PROVIDER_BINDING_MISMATCH")
    return FrozenAttemptBinding(
        **scope().model_dump(),
        execution_context_hash=artifact["context_hash"],
        generation_contract_sha256=BASE_CONTRACT_SHA256,
        task_schema_sha256=SCHEMA_SHA256,
        seed_amendment_sha256=AMENDMENT_SHA256,
        provider_safe_config_fingerprint=PROVIDER_SHA256,
        provider_binding_sha256=digest(canonical_bytes(provider)),
    )


def verify_canonical_runtime_attestation(
    artifact_root: Path, checkout: Path, binding: FrozenAttemptBinding,
) -> tuple[str, str]:
    """Require the exact successful operator attestation and verify its manifest."""
    run = artifact_root / "v2_1r_d4r_a6_operator" / CANONICAL_ATTESTATION_RUN
    names = (
        "canonical_runtime_attestation.json",
        "docker_host_attestation.json",
        "operator_probe_execution.json",
        "operator_probe_stage_report.json",
        "provider_readiness_attestation.json",
        "hash_manifest.json",
    )
    paths = {name: run / name for name in names}
    if any(path.is_symlink() for path in (run, *paths.values())):
        raise GenerationAttemptError("CANONICAL_ATTESTATION_SYMLINK")
    documents: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise GenerationAttemptError("CANONICAL_ATTESTATION_FILE_MISSING") from exc
        except OSError as exc:
            raise GenerationAttemptError("CANONICAL_ATTESTATION_FILE_UNREADABLE") from exc
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GenerationAttemptError("CANONICAL_ATTESTATION_INVALID_JSON") from exc
        if not isinstance(document, dict):
            raise GenerationAttemptError("CANONICAL_ATTESTATION_DOCUMENT_NOT_OBJECT")
        documents[name] = document
    attestation = documents["canonical_runtime_attestation.json"]
    docker_host = documents["docker_host_attestation.json"]
    probe_execution = documents["operator_probe_execution.json"]
    stage_report = documents["operator_probe_stage_report.json"]
    provider = documents["provider_readiness_attestation.json"]
    manifest = documents["hash_manifest.json"]
    image = attestation.get("image", {})
    host_container = docker_host.get("ephemeral_container", {})
    runtime = probe_execution.get("ephemeral_runtime", {})
    stage_names = {item.get("stage") for item in stage_report.get("stages", [])
                   if isinstance(item, dict)}
    expected_stages = {
        "PROBE_PROCESS_STARTED", "PYTHON_ENV_READY", "CURRENT_CHECKOUT_IDENTITY",
        "IMPORT_PROBE_STARTED", "IMPORT_PROBE_COMPLETED", "SCHEMA_RUNTIME_PROBE_STARTED",
        "SCHEMA_RUNTIME_PROBE_COMPLETED", "DB_READ_PROBE_STARTED",
        "DB_READ_PROBE_COMPLETED", "FROZEN_CONTEXT_RESOLUTION_STARTED",
        "FROZEN_CONTEXT_VALIDATED", "PROVIDER_PROFILE_RESOLUTION_STARTED",
        "PROVIDER_PROFILE_RESOLUTION_COMPLETED", "SAFE_FINGERPRINT_STARTED",
        "SAFE_FINGERPRINT_COMPLETED", "FINAL_ATTESTATION_READY",
    }
    command_results = docker_host.get("commands", {})
    source_hashes = attestation.get("source_file_sha256", {})
    imported_components = attestation.get("imported_components", [])
    required_docker_commands = {
        "docker_version", "docker_context_show", "docker_ps", "docker_compose_config",
    }
    provider_binding = provider.get("frozen_provider_binding", {})
    if (
        attestation.get("status") != "PASS"
        or attestation.get("execution_context_id") != EXECUTION_CONTEXT_ID
        or attestation.get("execution_context_hash") != binding.execution_context_hash
        or attestation.get("schema_runtime", {}).get("provider_generation_calls") != 0
        or attestation.get("schema_runtime", {}).get("provider_http_count") != 0
        or attestation.get("schema_runtime", {}).get("db_writes") != 0
        or attestation.get("schema_runtime", {}).get("status") != "PASS"
        or image.get("image_id") != host_container.get("image_id")
        or image.get("reference") != host_container.get("image_reference")
        or image.get("current_checkout_source_read_only_mount_verified") is not True
        or host_container.get("compose_run_exit_code") != 0
        or host_container.get("source_mount_current_checkout_read_only") is not True
        or host_container.get("removed_by_operator_script") is not True
        or docker_host.get("ephemeral_cleanup") != "REMOVED_EXACT_EPHEMERAL_CONTAINER"
        or any(docker_host.get(key) != 0 for key in (
            "docker_calls_by_codex", "provider_generation_calls", "provider_http_count",
            "semantic_verifier_calls", "db_writes", "real_acquisition", "holdout_access",
        ))
        or not isinstance(command_results, dict)
        or not required_docker_commands.issubset(command_results)
        or any(not isinstance(value, dict) or value.get("exit_code") != 0
               for value in command_results.values())
        or probe_execution.get("status") != "PASS"
        or runtime.get("cleanup_status") != "REMOVED_EXACT_EPHEMERAL_CONTAINER"
        or runtime.get("image_id") != image.get("image_id")
        or any(probe_execution.get(key) != 0 for key in (
            "provider_generation_calls", "provider_http_count", "semantic_verifier_calls",
            "db_writes", "holdout_access",
        ))
        or stage_report.get("status") != "PASS"
        or not expected_stages.issubset(stage_names)
        or len(stage_report.get("import_results", [])) != 28
        or any(not isinstance(item, dict) or item.get("status") != "PASS"
               for item in stage_report.get("import_results", []))
        or len(imported_components) != 28
        or any(not isinstance(item, dict) or item.get("status") != "PASS"
               or item.get("matches_current_checkout") is not True
               for item in imported_components)
        or stage_report.get("current_checkout_identity", {}).get("value", {}).get(
            "status"
        ) != "PASS"
        or stage_report.get("database_diagnostics", {}).get("value", {}).get(
            "status"
        ) != "PASS"
        or stage_report.get("database_diagnostics", {}).get("value", {}).get(
            "read_only_confirmed"
        ) is not True
        or stage_report.get("provider_diagnostics", {}).get("value", {}).get("status") != "RESOLVED"
        or any(stage_report.get(key) != 0 for key in (
            "provider_generation_calls", "provider_http_count", "semantic_verifier_calls",
            "db_writes", "holdout_access",
        ))
        or not isinstance(source_hashes, dict)
        or not source_hashes
        or provider.get("status") != "RESOLVED"
        or provider.get("binding_match") is not True
        or provider.get("execution_context_id") != EXECUTION_CONTEXT_ID
        or provider.get("execution_context_hash") != binding.execution_context_hash
        or any(provider_binding.get(key) != value for key, value in {
            "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
            "adapter": "openai_compatible_chat",
            "model": "deepseek-flash",
            "endpoint_family": "https://api.deepseek.com",
            "profile_version": 23,
            "credential_version": 4,
            "fingerprint_contract": "provider-safe-config-fingerprint.v1",
            "provider_safe_config_fingerprint": binding.provider_safe_config_fingerprint,
        }.items())
        or provider.get("provider_generation_calls") != 0
        or provider.get("provider_http_count") != 0
        or provider.get("db_writes") != 0
        or provider.get("credential_plaintext_emitted") is not False
        or provider.get("authorization_emitted") is not False
        or manifest.get("artifact_version") != "v2_1r_d4r_a6_operator.hash_manifest.v1"
        or manifest.get("algorithm") != "SHA-256"
        or manifest.get("self_hash_excluded") is not True
        or not isinstance(manifest.get("files"), dict)
    ):
        raise GenerationAttemptError("CANONICAL_ATTESTATION_NOT_PASS_OR_BINDING_MISMATCH")
    files = manifest["files"]
    required = {
        "canonical_runtime_attestation.json",
        "docker_host_attestation.json",
        "operator_probe_execution.json",
        "operator_probe_stage_report.json",
        "provider_readiness_attestation.json",
        "scripts/operator_verify_canonical_runtime.ps1",
    }
    if not required.issubset(files):
        raise GenerationAttemptError("CANONICAL_ATTESTATION_MANIFEST_INCOMPLETE")
    for relative, expected in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise GenerationAttemptError("CANONICAL_ATTESTATION_MANIFEST_PATH_INVALID")
        candidate = (checkout / rel) if rel.parts[:1] == ("scripts",) else (run / rel)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or digest(candidate.read_bytes()) != expected
        ):
            raise GenerationAttemptError("CANONICAL_ATTESTATION_HASH_MISMATCH")
    for relative, expected in source_hashes.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise GenerationAttemptError("CANONICAL_SOURCE_HASH_PATH_INVALID")
        candidate = checkout / rel
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or digest(candidate.read_bytes()) != expected
        ):
            raise GenerationAttemptError("CANONICAL_SOURCE_HASH_MISMATCH")
    payload = {
        "artifact_version": manifest["artifact_version"],
        "algorithm": manifest["algorithm"],
        "files": files,
    }
    manifest_hash = digest(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    if manifest_hash != manifest.get("manifest_sha256"):
        raise GenerationAttemptError("CANONICAL_ATTESTATION_MANIFEST_HASH_MISMATCH")
    return manifest_hash, digest(paths["canonical_runtime_attestation.json"].read_bytes())


def verify_timeout_policy(checkout: Path) -> None:
    """Integrity check only: no timeout tuning, HTTP client or dispatch."""
    expected = {"connect": 10.0, "read": 90.0, "write": 90.0, "pool": 90.0}
    policy_path = checkout / "apps/api/app/llm/model_http_policy.py"
    imported_path = Path(str(model_http_policy.__file__))
    if (
        digest(policy_path.read_bytes()) != digest(imported_path.read_bytes())
        or model_http_policy.model_generation_timeout().as_dict() != expected
    ):
        raise GenerationAttemptError("MODEL_TIMEOUT_POLICY_INTEGRITY_MISMATCH")
    entry_tree = ast.parse(
        (checkout / "scripts/prospective_acquisition/path_c_generation_entry.py").read_text(
            encoding="utf-8"
        )
    )
    functions = {
        node.name: node for node in entry_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    builder = functions["build_path_c_model_client"]
    assignments = [
        node for node in ast.walk(builder) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "timeout" for t in node.targets)
    ]
    if len(assignments) != 1 or ast.unparse(assignments[0].value) != "model_generation_timeout()":
        raise GenerationAttemptError("PATH_C_TIMEOUT_OVERRIDE_DETECTED")
    run_assignments = [node for node in ast.walk(functions["_run"])
                       if isinstance(node, ast.Assign)]
    if any(any(isinstance(t, ast.Attribute) and t.attr == "timeout" for t in node.targets)
           for node in run_assignments):
        raise GenerationAttemptError("PATH_C_TIMEOUT_OVERRIDE_DETECTED")
    if not any(
        isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "build_path_c_model_client"
        for node in run_assignments
    ):
        raise GenerationAttemptError("PATH_C_MODEL_TIMEOUT_WIRING_MISSING")
    worker_tree = ast.parse(
        (checkout / "apps/api/app/worker/tasks.py").read_text(encoding="utf-8")
    )
    model_clients = [
        node.value for node in ast.walk(worker_tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model_client" for t in node.targets)
    ]
    if len(model_clients) != 1 or not isinstance(model_clients[0], ast.Call) or not any(
        k.arg == "timeout" and ast.unparse(k.value) == "model_generation_timeout()"
        for k in model_clients[0].keywords
    ):
        raise GenerationAttemptError("WORKER_MODEL_TIMEOUT_WIRING_MISSING")


def verify_response_and_request_integrity(checkout: Path) -> None:
    """Inspect actual code/contracts only: no Gateway execution or HTTP request."""
    from app.llm import adapters

    from scripts.prospective_acquisition import capture_transport, path_c_generation

    for relative, module in (
        ("apps/api/app/llm/adapters.py", adapters),
        ("scripts/prospective_acquisition/capture_transport.py", capture_transport),
        ("scripts/prospective_acquisition/path_c_generation.py", path_c_generation),
    ):
        if digest((checkout / relative).read_bytes()) != digest(
            Path(str(module.__file__)).read_bytes()
        ):
            raise GenerationAttemptError("CURRENT_CHECKOUT_IMPLEMENTATION_MISMATCH")
    capture_tree = ast.parse(
        (checkout / "scripts/prospective_acquisition/capture_transport.py").read_text(
            encoding="utf-8"
        )
    )
    capture_class = next(
        n for n in capture_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RawResponseCaptureTransport"
    )
    handler = next(
        n for n in capture_class.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_async_request"
    )
    required = (
        "headers = httpx.Headers(response.headers)",
        "for name in ('content-encoding', 'content-length', 'transfer-encoding'):\n"
        "    if name in headers:\n        del headers[name]",
    )
    nodes = {ast.dump(n, include_attributes=False) for n in ast.walk(handler)}
    if any(ast.dump(ast.parse(s).body[0], include_attributes=False) not in nodes for s in required):
        raise GenerationAttemptError("CAPTURE_RECONSTRUCTION_INTEGRITY_MISMATCH")
    replacements = [
        n.value for n in ast.walk(handler) if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "httpx.Response"
    ]
    if len(replacements) != 1:
        raise GenerationAttemptError("CAPTURE_RECONSTRUCTION_INTEGRITY_MISMATCH")
    representation = {k.arg: ast.unparse(k.value) for k in replacements[0].keywords}
    if representation.get("headers") != "headers" or representation.get("content") != "raw_body":
        raise GenerationAttemptError("CAPTURE_RECONSTRUCTION_INTEGRITY_MISMATCH")

    adapter_tree = ast.parse(
        (checkout / "apps/api/app/llm/adapters.py").read_text(encoding="utf-8")
    )
    gateway = next(
        n for n in adapter_tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMGateway"
    )
    post = next(
        n for n in gateway.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_post_json"
    )
    handlers = [n for n in ast.walk(post) if isinstance(n, ast.ExceptHandler)]
    expected = {
        "httpx.DecodingError": ("MODEL_RESPONSE_INVALID", {"DECODING_ERROR"}),
        "(httpx.StreamConsumed, httpx.StreamClosed)": (
            "MODEL_RESPONSE_INVALID", {"STREAM_CONSUMED", "STREAM_CLOSED"},
        ),
        "(json.JSONDecodeError, UnicodeDecodeError)": (
            "MODEL_RESPONSE_INVALID", {"JSON_DECODE_ERROR"},
        ),
        "httpx.RequestError": ("MODEL_NETWORK_ERROR", set()),
    }
    for error_type, (code, details) in expected.items():
        matched = [h for h in handlers if h.type is not None and ast.unparse(h.type) == error_type]
        if len(matched) != 1:
            raise GenerationAttemptError("RESPONSE_TAXONOMY_INTEGRITY_MISMATCH")
        calls = [
            n for n in ast.walk(matched[0]) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "ModelGatewayError"
        ]
        if len(calls) != 1 or not calls[0].args or ast.literal_eval(calls[0].args[0]) != code:
            raise GenerationAttemptError("RESPONSE_TAXONOMY_INTEGRITY_MISMATCH")
        symbols = {n.value for n in ast.walk(calls[0]) if isinstance(n, ast.Constant)}
        if not details.issubset(symbols):
            raise GenerationAttemptError("RESPONSE_TAXONOMY_INTEGRITY_MISMATCH")
    types = [ast.unparse(h.type) for h in handlers if h.type is not None]
    if types.index("httpx.DecodingError") > types.index("httpx.RequestError"):
        raise GenerationAttemptError("RESPONSE_TAXONOMY_INTEGRITY_MISMATCH")

    contract, _, seed, _ = verify_frozen_contracts(checkout, "SEED_SUPPORT_NOT_ESTABLISHED")
    tasks = _provider_response_schema()["properties"]
    if (
        contract.get("task_count") != 40 or contract.get("generation_calls_max") != 1
        or seed.send_seed or seed.seed_value is not None
        or not isinstance(tasks, dict) or tasks["tasks"].get("minItems") != 40
        or tasks["tasks"].get("maxItems") != 40
    ):
        raise GenerationAttemptError("TASK_COUNT_OR_SEED_POLICY_INTEGRITY_MISMATCH")
    generation_tree = ast.parse(
        (checkout / "scripts/prospective_acquisition/path_c_generation.py").read_text(
            encoding="utf-8"
        )
    )
    builds = [
        n for n in ast.walk(generation_tree) if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "CanonicalModelRequest"
    ]
    if len(builds) != 1:
        raise GenerationAttemptError("GENERATION_REQUEST_INTEGRITY_MISMATCH")
    arguments = {k.arg: k.value for k in builds[0].keywords}
    if "seed" in arguments or ast.literal_eval(arguments["generation_parameters"]) != {
        "temperature": 0,
    }:
        raise GenerationAttemptError("GENERATION_REQUEST_INTEGRITY_MISMATCH")
    calls = [
        n for n in ast.walk(generation_tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "generate_structured"
    ]
    if len(calls) != 1 or not any(
        k.arg == "allow_regeneration" and isinstance(k.value, ast.Constant)
        and k.value.value is False
        for k in calls[0].keywords
    ):
        raise GenerationAttemptError("GENERATION_REQUEST_INTEGRITY_MISMATCH")


def check_history(
    batch: Path, expected_parent_hash: str, approved_attempt: int,
    binding: FrozenAttemptBinding,
) -> tuple[AttemptRecord, ...]:
    if type(approved_attempt) is not int or approved_attempt not in {2, 3, 4, 5, 6}:
        raise GenerationAttemptError("ATTEMPT_NUMBER_NOT_APPROVED")
    if any(p.is_symlink() for p in (batch, *batch.parents)):
        raise GenerationAttemptError("HISTORY_DIRECTORY_SYMLINK")
    records = read_history(batch, scope())
    # The supplied frozen hash always anchors legacy Attempt 001. Attempt 002's
    # parent hash is read from canonical bytes in this SAME execution, not CLI.
    if records[0].state_sha256 != expected_parent_hash:
        raise GenerationAttemptError("PARENT_HISTORY_HASH_MISMATCH")
    if len(records) != approved_attempt - 1:
        raise GenerationAttemptError("NEXT_ATTEMPT_HISTORY_MISMATCH")
    if any(record.state != "CONSUMED_FAILED" or not record.consumed for record in records):
        raise GenerationAttemptError("PARENT_ATTEMPT_NOT_CONSUMED_FAILED")
    directory = batch / "authorizations"
    if directory.is_symlink():
        raise GenerationAttemptError("AUTHORIZATION_DIRECTORY_SYMLINK")
    grants = _read_authorizations(batch, scope(), records)
    if approved_attempt in grants:
        raise GenerationAttemptError("AUTHORIZATION_ALREADY_EXISTS")
    if set(grants) != set(range(2, approved_attempt)):
        raise GenerationAttemptError("HISTORICAL_AUTHORIZATION_MISMATCH")
    if any(auth.binding() != binding for auth, _ in grants.values()):
        raise GenerationAttemptError("HISTORICAL_PROVIDER_OR_CONTEXT_BINDING_MISMATCH")
    if approved_attempt >= 3:
        parent = records[1].state_data
        diagnostics = parent.get("failure_diagnostics")
        if (
            parent.get("dispatch_count") != 1
            or type(parent.get("physical_outbound_count")) is not int
            or parent.get("physical_outbound_count") != 1
            or parent.get("blocked_additional_outbound_count") != 0
            or not isinstance(diagnostics, dict)
            or any(diagnostics.get(k) != v for k, v in {
                "failure_class": "MODEL_GATEWAY_ERROR",
                "failure_category": "TIMEOUT", "gateway_error_code": "MODEL_TIMEOUT",
                "gateway_detail_code": "READ_TIMEOUT", "provider_http_status": 200,
                "response_received": True,
            }.items())
        ):
            raise GenerationAttemptError("ATTEMPT_002_TERMINAL_EVIDENCE_MISMATCH")
    if approved_attempt >= 4:
        parent = records[2].state_data
        diagnostics = parent.get("failure_diagnostics")
        if (
            parent.get("dispatch_count") != 1 or parent.get("physical_outbound_count") != 1
            or parent.get("blocked_additional_outbound_count") != 0
            or not isinstance(diagnostics, dict)
            or any(diagnostics.get(k) != v for k, v in {
                "failure_category": "NETWORK", "gateway_error_code": "MODEL_NETWORK_ERROR",
                "provider_http_status": 200, "response_received": True,
                "response_body_present": True, "raw_response_preserved": True,
                "raw_capture_policy": "EXACT_SUCCESS_RESPONSE",
                "response_body_sha256": structure.EXPECTED_SHA256,
                "response_body_size": structure.EXPECTED_SIZE,
            }.items())
        ):
            raise GenerationAttemptError("ATTEMPT_003_TERMINAL_EVIDENCE_MISMATCH")
        path = records[2].location / "provider_response.raw.json"
        if path.is_symlink():
            raise GenerationAttemptError("HISTORY_ARTIFACT_SYMLINK")
        raw = path.read_bytes()
        if digest(raw) != structure.EXPECTED_SHA256 or len(raw) != structure.EXPECTED_SIZE:
            raise GenerationAttemptError("ATTEMPT_003_RAW_IDENTITY_MISMATCH")
        facts = structure.structural_facts(raw)
        if any(facts.get(k) != v for k, v in {
            "valid_json": True, "top_level_type": "dict",
            "openai_compatible_envelope_valid": True, "completion_content_field_present": True,
            "completion_content_non_empty": True, "completion_content_parses_as_json": True,
            "parsed_content_top_level_type": "dict", "task_array_count": 40,
            "required_schema_keys_present": ["tasks"], "required_schema_keys_missing": [],
        }.items()):
            raise GenerationAttemptError("ATTEMPT_003_RAW_STRUCTURE_MISMATCH")
    return records


def history_file_hashes(batch: Path, records: tuple[AttemptRecord, ...]) -> dict[Path, str]:
    """Hash original history read-only; never rewrite or migrate consumed files."""
    paths = [p for p in batch.iterdir() if p.is_file() and not p.name.startswith(".publication-")]
    for record in records[1:]:
        paths.extend(p for p in record.location.rglob("*") if p.is_file())
    directory = batch / "authorizations"
    if directory.exists():
        paths.extend(directory.iterdir())
    if any(p.is_symlink() for p in paths):
        raise GenerationAttemptError("HISTORY_ARTIFACT_SYMLINK")
    return {p: digest(p.read_bytes()) for p in paths}


def freeze_authorization(
    batch: Path, binding: FrozenAttemptBinding, *, expected_parent_hash: str = PARENT_SHA256,
    approved_attempt: int = 2, checkout: Path | None = None,
    artifact_root: Path | None = None,
) -> Path:
    """Same implementation for host and fake directories; zero dispatch capability."""
    if HistoryScope(**{k: getattr(binding, k) for k in HistoryScope.model_fields}) != scope():
        raise GenerationAttemptError("AUTHORIZATION_SCOPE_MISMATCH")
    records = check_history(batch, expected_parent_hash, approved_attempt, binding)
    attestation_identity: tuple[str, str] | None = None
    if approved_attempt >= 5:
        if artifact_root is None or checkout is None:
            raise GenerationAttemptError("CANONICAL_ATTESTATION_REQUIRED")
        attestation_identity = verify_canonical_runtime_attestation(
            artifact_root, checkout, binding,
        )
    if approved_attempt >= 3:
        if checkout is None:
            raise GenerationAttemptError("TIMEOUT_POLICY_PROOF_REQUIRED")
        verify_timeout_policy(checkout)
        if approved_attempt >= 4:
            verify_response_and_request_integrity(checkout)
    previous = records[-1]
    historical_hashes = history_file_hashes(batch, records)
    identifier = uuid4()
    if identifier in {record.authorization_id for record in records}:
        raise GenerationAttemptError("DUPLICATE_ATTEMPT_AUTHORIZATION")
    auth = PathCAttemptAuthorization(
        **binding.model_dump(),
        authorization_id=identifier,
        attempt_number=approved_attempt,
        attempt_id=attempt_id_for(scope(), approved_attempt, identifier),
        previous_attempt_number=previous.number,
        previous_attempt_id=previous.attempt_id,
        previous_state_sha256=previous.state_sha256,
        canonical_runtime_attestation_run=(CANONICAL_ATTESTATION_RUN
                                           if approved_attempt == 6 else None),
        canonical_runtime_attestation_manifest_sha256=(
            attestation_identity[0] if approved_attempt == 6 and attestation_identity else None
        ),
        created_at_utc=datetime.now(UTC),
    ).with_integrity_hash()
    auth.validate_integrity()
    raw = canonical_bytes(auth.model_dump(mode="json", exclude_none=True)) + b"\n"
    PathCAttemptAuthorization.model_validate_json(raw).validate_integrity()
    directory = batch / "authorizations"
    directory.mkdir(exist_ok=True)
    temporary = batch / f".publication-{uuid4()}.tmp"
    target = directory / f"attempt_{approved_attempt:03d}.authorization.json"
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Repeat authoritative checks immediately before atomic no-replace publication.
        if check_history(batch, expected_parent_hash, approved_attempt, binding) != records:
            raise GenerationAttemptError("PARENT_CHANGED_DURING_FREEZE")
        if any(digest(path.read_bytes()) != h for path, h in historical_hashes.items()):
            raise GenerationAttemptError("HISTORY_CHANGED_DURING_FREEZE")
        if approved_attempt >= 4:
            assert checkout is not None
            verify_timeout_policy(checkout)
            verify_response_and_request_integrity(checkout)
        if approved_attempt >= 5:
            assert artifact_root is not None and checkout is not None
            latest_identity = verify_canonical_runtime_attestation(
                artifact_root, checkout, binding,
            )
            if approved_attempt == 6 and latest_identity != attestation_identity:
                raise GenerationAttemptError("CANONICAL_ATTESTATION_CHANGED_DURING_FREEZE")
        os.link(temporary, target)
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if any(digest(path.read_bytes()) != h for path, h in historical_hashes.items()):
        raise GenerationAttemptError("HISTORY_CHANGED_AFTER_FREEZE")
    return target


def governance_rationale_for_attempt(attempt_number: int) -> str:
    """Human-readable CLI description; not part of the authorization contract."""
    rationales = {
        2: (
            "Separate numbered authorization, not a retry/reset; Attempt 001 remains "
            "consumed with unrecoverable root cause."
        ),
        3: (
            "Separate numbered authorization, not a retry/reset of Attempt 002; both "
            "historical attempts remain consumed. Generic timeout wiring corrected and "
            "regression-tested; no Provider defect or guaranteed success claimed."
        ),
        4: (
            "Final normal generation attempt, separately authorized, not a retry/reset of "
            "Attempt 003. All historical attempts remain consumed. Attempt 003 returned a "
            "complete structurally valid 40-task response. Generic response reconstruction "
            "defect independently reproduced and fixed; timeout wiring and false NETWORK "
            "classification corrected; deterministic regressions passed. Historical gzip/br "
            "unconfirmed; no Provider defect or guaranteed success claimed."
        ),
        5: (
            "Final separately authorized Attempt 005; Attempts 001-004 remain consumed "
            "and unchanged. Canonical runtime attestation PASS and frozen identity matched; "
            "this authorization does not dispatch a Provider request."
        ),
        6: (
            "Final separately authorized Attempt 006; Attempts 001-005 remain consumed "
            "and unchanged. Canonical runtime attestation PASS and frozen identity matched; "
            "this authorization does not dispatch a Provider request."
        ),
    }
    try:
        return rationales[attempt_number]
    except KeyError as exc:
        raise ValueError("UNSUPPORTED_ATTEMPT_NUMBER") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorization only; no generation")
    parser.add_argument("--approved-attempt", type=int, choices=(2, 3, 4, 5, 6), default=2)
    arguments = parser.parse_args()
    try:
        root = CANONICAL_ARTIFACT_VOLUME_ROOT
        checkout = CANONICAL_CHECKOUT_ROOT
        binding = load_binding(root, checkout)
        batch = root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
        target = freeze_authorization(
            batch, binding, approved_attempt=arguments.approved_attempt, checkout=checkout,
            # Attempt history/context live in the named volume at /app/artifacts;
            # canonical runtime attestations are repository artifacts exposed
            # by the read-only checkout bind at /opt/artifacts.
            artifact_root=checkout / "artifacts",
        )
        auth = PathCAttemptAuthorization.model_validate_json(target.read_bytes())
        print(json.dumps({
            "authorization_path": str(target), "authorization_sha256": digest(target.read_bytes()),
            "attempt_id": auth.attempt_id, "authorization_id": str(auth.authorization_id),
            "attempt_number": auth.attempt_number,
            "parent_attempt_id": auth.previous_attempt_id,
            "parent_state_sha256": auth.previous_state_sha256,
            "state": auth.authorization_status, "provider_dispatch_count": 0,
            "governance_rationale": governance_rationale_for_attempt(auth.attempt_number),
        }))
        return 0
    except Exception as exc:
        code = (
            exc.code if isinstance(exc, GenerationAttemptError) else "AUTHORIZATION_FREEZE_FAILED"
        )
        print(json.dumps({"status": code, "provider_dispatch_count": 0}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
