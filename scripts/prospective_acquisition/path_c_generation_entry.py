"""Canonical Compose-runtime entry for the single PATH-C task generation."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
from app.core.config import Settings
from app.domain.providers import AdapterType
from app.infrastructure.db.models import ProviderProfileRow
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.llm.model_http_policy import model_generation_timeout
from app.security.secrets import SecretCipher
from app.services.prospective_acquisition_execution_identity import (
    FrozenProviderBinding,
    ProspectiveAcquisitionExecutionContext,
    assert_provider_binding_unchanged,
    resolve_provider_binding,
    validate_frozen_context,
)
from app.services.provider_profiles import ProviderProfileService
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from scripts.prospective_acquisition.capture_transport import RawResponseCaptureTransport
from scripts.prospective_acquisition.control import CallBudget, build_guarded_provider_client
from scripts.prospective_acquisition.path_c_attempts import (
    GenerationAttemptError,
    HistoryScope,
    resolve_operator_attempt,
    validate_authorized_attempt,
)
from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    FailureDiagnosticsRecorder,
    FailurePhase,
    ObservedPathCGateway,
)
from scripts.prospective_acquisition.path_c_generation import (
    BATCH_ID,
    EXECUTION_CONTEXT_ID,
    GENERATION_CONTRACT_VERSION,
    POPULATION_ID,
    PathCGenerationBlocked,
    execute_path_c_generation,
    frozen_attempt_binding,
    return_existing_frozen_success,
)


def _context_path(artifact_root: Path) -> Path:
    key = hashlib.sha256(
        json.dumps([POPULATION_ID, BATCH_ID], ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return artifact_root / "v2_1r_d4r_a6_closure" / "execution_contexts" / f"{key}.json"


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PathCGenerationBlocked("FROZEN_EXECUTION_CONTEXT_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise PathCGenerationBlocked("FROZEN_EXECUTION_CONTEXT_INVALID")
    return cast(dict[str, Any], value)


async def _read_owner_scope(session_factory: async_sessionmaker[Any], profile_id: UUID) -> str:
    async with session_factory() as session:
        read_only = await session.scalar(text("SHOW transaction_read_only"))
        if str(read_only).lower() not in {"on", "true", "1"}:
            raise PathCGenerationBlocked("DATABASE_READ_ONLY_NOT_CONFIRMED")
        owner_hash = await session.scalar(
            select(ProviderProfileRow.owner_hash).where(
                ProviderProfileRow.id == profile_id,
                ProviderProfileRow.status == "active",
                ProviderProfileRow.deleted_at.is_(None),
            )
        )
    if not isinstance(owner_hash, str) or len(owner_hash) != 64:
        raise PathCGenerationBlocked("FROZEN_PROVIDER_SCOPE_UNAVAILABLE")
    return owner_hash


def build_path_c_model_client(
    budget: CallBudget, transport: httpx.AsyncBaseTransport
) -> httpx.AsyncClient:
    """Reuse ordinary Worker generation I/O limits, retaining the one-call guard."""
    client = build_guarded_provider_client(budget, transport=transport)
    client.timeout = model_generation_timeout()
    return client


def _build_context(
    artifact: dict[str, Any], owner_hash: str
) -> ProspectiveAcquisitionExecutionContext:
    try:
        raw_binding = artifact["provider_binding"]
        binding = FrozenProviderBinding(
            profile_id=str(raw_binding["profile_id"]),
            adapter=str(raw_binding["adapter"]),
            model=str(raw_binding["model"]),
            endpoint_family=str(raw_binding["endpoint_family"]),
            profile_version=int(raw_binding["profile_version"]),
            credential_version_id=str(raw_binding["credential_version_id"]),
            credential_version=int(raw_binding["credential_version"]),
            credential_present=bool(raw_binding["credential_present"]),
            fingerprint_contract=str(raw_binding["fingerprint_contract"]),
            provider_safe_config_fingerprint=str(raw_binding["provider_safe_config_fingerprint"]),
        )
        context = ProspectiveAcquisitionExecutionContext(
            execution_context_id=str(artifact["execution_context_id"]),
            population_id=str(artifact["population_id"]),
            batch_id=str(artifact["batch_id"]),
            owner_hash=owner_hash,
            owner_scope_source=str(artifact["owner_scope_source"]),
            provider_resolution_policy=str(artifact["provider_resolution_policy"]),
            protocol_version=str(artifact["protocol_version"]),
            created_at=str(artifact["created_at"]),
            provider_binding=binding,
            context_hash=str(artifact["context_hash"]),
        )
        validate_frozen_context(context)
    except PathCGenerationBlocked:
        raise
    except Exception as exc:
        raise PathCGenerationBlocked("FROZEN_EXECUTION_CONTEXT_INVALID") from exc
    if (
        context.execution_context_id != EXECUTION_CONTEXT_ID
        or context.population_id != POPULATION_ID
        or context.batch_id != BATCH_ID
        or context.owner_scope_source != "AUTHENTICATED_CLIENT_SESSION"
        or hashlib.sha256(owner_hash.encode("ascii")).hexdigest()
        != artifact.get("owner_scope_fingerprint")
    ):
        raise PathCGenerationBlocked("FROZEN_EXECUTION_CONTEXT_IDENTITY_MISMATCH")
    if context.safe_artifact() != artifact:
        raise PathCGenerationBlocked("FROZEN_EXECUTION_CONTEXT_ARTIFACT_MISMATCH")
    return context


def _load_existing_master_key(settings: Settings) -> bytes:
    """Read an existing configured key without invoking the key-creation path."""
    try:
        if settings.secret_master_key_base64 is not None:
            encoded = settings.secret_master_key_base64.get_secret_value()
        else:
            if not settings.secret_master_key_file.is_file():
                raise PathCGenerationBlocked("PROVIDER_CREDENTIAL_KEY_UNAVAILABLE")
            encoded = settings.secret_master_key_file.read_text(encoding="ascii").strip()
        key = base64.b64decode(encoded, validate=True)
    except PathCGenerationBlocked:
        raise
    except Exception as exc:
        raise PathCGenerationBlocked("PROVIDER_CREDENTIAL_KEY_UNAVAILABLE") from exc
    if len(key) != 32:
        raise PathCGenerationBlocked("PROVIDER_CREDENTIAL_KEY_UNAVAILABLE")
    return key


async def _run(diagnostics: FailureDiagnosticsRecorder | None = None) -> dict[str, object]:
    settings = Settings()
    artifact_root = settings.artifact_root
    batch_root = artifact_root / "v2_1r_d4r_path_c_generation" / f"{POPULATION_ID}__{BATCH_ID}"
    selection = resolve_operator_attempt(
        batch_root,
        HistoryScope(
            population_id=POPULATION_ID,
            batch_id=BATCH_ID,
            execution_context_id=UUID(EXECUTION_CONTEXT_ID),
            generation_contract_version=GENERATION_CONTRACT_VERSION,
        ),
    )
    if selection.authorized is None:
        outcome = return_existing_frozen_success(selection.latest.location)
        return {
            "status": outcome.state,
            "manifest_path": outcome.manifest_path,
            "manifest_sha256": outcome.manifest_sha256,
            "provider_dispatch_count": 1,
            "reused_existing": True,
            "provider_http_count": 0,
            "database_writes": 0,
            "holdout_access": 0,
        }
    attempt = selection.authorized
    output_dir = attempt.location
    observations = diagnostics or FailureDiagnosticsRecorder()
    context_artifact = _safe_json(_context_path(artifact_root))
    profile_id = UUID(str(context_artifact["provider_binding"]["profile_id"]))
    engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"options": "-c default_transaction_read_only=on"},
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        owner_hash = await _read_owner_scope(session_factory, profile_id)
        context = _build_context(context_artifact, owner_hash)
        repository = ProviderProfileRepository(session_factory)
        observations.phase = FailurePhase.CREDENTIAL_RESOLUTION
        cipher = SecretCipher(_load_existing_master_key(settings))
        service = ProviderProfileService(
            repository,
            cipher,
            allow_insecure_endpoints=settings.allow_insecure_provider_endpoints,
        )
        observations.phase = FailurePhase.PROVIDER_RESOLUTION
        current_binding = await resolve_provider_binding(context, service, repository)
        if current_binding != context.provider_binding:
            raise PathCGenerationBlocked("EXECUTION_PROVIDER_BINDING_CHANGED")
        validate_authorized_attempt(attempt, frozen_attempt_binding(context, current_binding))
        observations.phase = FailurePhase.CREDENTIAL_RESOLUTION
        material = await service.get_test_material(owner_hash, UUID(current_binding.profile_id))
        budget = CallBudget(maximum=1)
        raw_path = output_dir / "provider_response.raw.json"
        provider_transport = httpx.AsyncHTTPTransport(
            proxy=settings.model_http_proxy or None,
            retries=0,
        )
        capture = RawResponseCaptureTransport(provider_transport, raw_path, observations)
        client = build_path_c_model_client(budget, capture)
        try:
            outcome = await execute_path_c_generation(
                # Contract sources are mounted individually read-only under
                # /opt/artifacts; API source remains under its canonical
                # read-only Compose bind at /app/apps/api; durable data stays
                # on /app/artifacts and is never shadowed.
                root=Path("/opt"),
                application_root=Path("/app"),
                output_dir=output_dir,
                context=context,
                provider_binding=current_binding,
                binding_recheck=lambda: assert_provider_binding_unchanged(
                    context, service, repository
                ),
                gateway=ObservedPathCGateway(client, observations),
                api_key=material.api_key,
                base_url=material.base_url,
                adapter_type=AdapterType(material.adapter_type),
                context_manifest_id=UUID(context.execution_context_id),
                physical_budget=budget,
                raw_response_path=raw_path,
                failure_diagnostics=observations,
                authorized_attempt=attempt,
            )
        finally:
            await client.aclose()
        return {
            "status": outcome.state,
            "manifest_path": outcome.manifest_path,
            "manifest_sha256": outcome.manifest_sha256,
            "provider_dispatch_count": outcome.provider_dispatch_count,
            "seed_capability": outcome.seed_capability,
            "seed_sent": outcome.seed_sent,
            "deterministic_generation_claim": outcome.deterministic_generation_claim,
            "reused_existing": outcome.reused_existing,
            "provider_http_count": budget.dispatched,
            "blocked_extra_http_attempts": budget.blocked_attempts,
            "search_count": 0,
            "reader_count": 0,
            "evidence_extractor_count": 0,
            "database_writes": 0,
            "holdout_access": 0,
            "attempt_number": attempt.authorization.attempt_number,
            "attempt_id": attempt.authorization.attempt_id,
            "authorization_id": str(attempt.authorization.authorization_id),
        }
    finally:
        await engine.dispose()


def _emit_safe_failure(exc: Exception, observations: FailureDiagnosticsRecorder) -> None:
    try:
        diagnostic = exc.diagnostics if isinstance(exc, PathCGenerationBlocked) else None
        if diagnostic is None:
            diagnostic = observations.snapshot(
                exc,
                physical_outbound_count=observations.physical_outbound_count,
                blocked_additional_outbound_count=observations.blocked_additional_outbound_count,
            )
        serialized = diagnostic.model_dump_json()
    except Exception:
        serialized = '{"failure_diagnostics_status":"UNAVAILABLE"}'
    sys.stderr.write("PATH_C_SAFE_FAILURE_DIAGNOSTICS " + serialized + "\n")


def main() -> int:
    observations = FailureDiagnosticsRecorder()
    try:
        result = asyncio.run(_run(observations))
    except GenerationAttemptError as exc:
        sys.stderr.write(f"PATH_C_GENERATION_BLOCKED {exc.code}\n")
        sys.stderr.write("PATH_C_PROVIDER_HTTP_COUNT 0\n")
        return 2
    except PathCGenerationBlocked as exc:
        sys.stderr.write(f"PATH_C_GENERATION_BLOCKED {exc.code}\n")
        if exc.code != "GENERATION_AUTHORIZATION_ALREADY_CONSUMED":
            _emit_safe_failure(exc, observations)
        return 2
    except Exception as exc:
        # Exception text may contain URL/request material; publish only type.
        sys.stderr.write(f"PATH_C_GENERATION_BLOCKED {type(exc).__name__.upper()}\n")
        _emit_safe_failure(exc, observations)
        return 3
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
