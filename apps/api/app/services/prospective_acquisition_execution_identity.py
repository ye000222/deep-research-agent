"""Immutable authenticated identity binding for prospective acquisition.

This development-only control-plane module never creates a real run or calls a
Provider by itself. Runtime callers must obtain ``ClientSession`` from the
normal authentication dependency and pass that object to the factory.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from app.domain.provider_profiles import ProviderProfileView
from app.security.client_sessions import ClientSession
from app.services.provider_fingerprint_contract import (
    FINGERPRINT_CONTRACT_VERSION,
    safe_provider_fingerprint,
)

OWNER_SCOPE_SOURCE = "AUTHENTICATED_CLIENT_SESSION"
PROVIDER_RESOLUTION_POLICY = "FROZEN_DEFAULT_PROFILE_V1"
_OWNER_HASH_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


class ExecutionIdentityError(RuntimeError):
    """Base class for fail-closed acquisition identity errors."""


class OwnerScopeRequired(ExecutionIdentityError):
    pass


class ProviderCardinalityError(ExecutionIdentityError):
    pass


class ExecutionProviderBindingChanged(ExecutionIdentityError):
    pass


class ProviderCredentialVersionInvalid(ExecutionIdentityError):
    """The active credential row and profile view do not share a valid version."""


class ExecutionContextMismatch(ExecutionIdentityError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenProviderBinding:
    profile_id: str
    adapter: str
    model: str
    endpoint_family: str
    profile_version: int
    credential_version_id: str
    credential_version: int
    credential_present: bool
    fingerprint_contract: str
    provider_safe_config_fingerprint: str

    def safe_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "adapter": self.adapter,
            "model": self.model,
            "endpoint_family": self.endpoint_family,
            "profile_version": self.profile_version,
            "credential_version_id": self.credential_version_id,
            "credential_version": self.credential_version,
            "credential_present": self.credential_present,
            "fingerprint_contract": self.fingerprint_contract,
            "provider_safe_config_fingerprint": self.provider_safe_config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveAcquisitionExecutionContext:
    execution_context_id: str
    population_id: str
    batch_id: str
    owner_hash: str
    owner_scope_source: str
    provider_resolution_policy: str
    protocol_version: str
    created_at: str
    provider_binding: FrozenProviderBinding | None = None
    context_hash: str = ""

    def __post_init__(self) -> None:
        if not self.execution_context_id or not self.population_id or not self.batch_id:
            raise ValueError("execution_context_identity_required")
        if not self.protocol_version or not self.created_at:
            raise ValueError("execution_context_metadata_required")
        if self.owner_scope_source != OWNER_SCOPE_SOURCE:
            raise ValueError("owner_scope_must_be_authenticated_client_session")
        if not _OWNER_HASH_PATTERN.fullmatch(self.owner_hash):
            raise OwnerScopeRequired("authenticated_owner_scope_missing_or_invalid")

    @property
    def owner_scope_fingerprint(self) -> str:
        return hashlib.sha256(self.owner_hash.encode("ascii")).hexdigest()

    def with_provider_binding(
        self, binding: FrozenProviderBinding
    ) -> ProspectiveAcquisitionExecutionContext:
        bound = replace(self, provider_binding=binding, context_hash="")
        return replace(bound, context_hash=_context_digest(bound))

    def safe_artifact(self) -> dict[str, object]:
        validate_frozen_context(self)
        binding = self.provider_binding
        if binding is None:
            raise ExecutionIdentityError("provider_binding_not_frozen")
        return {
            "contract_version": "prospective-acquisition-execution-context.v1",
            "execution_context_id": self.execution_context_id,
            "population_id": self.population_id,
            "batch_id": self.batch_id,
            "owner_scope_fingerprint": self.owner_scope_fingerprint,
            "owner_scope_source": self.owner_scope_source,
            "provider_resolution_policy": self.provider_resolution_policy,
            "provider_binding": binding.safe_dict(),
            "protocol_version": self.protocol_version,
            "created_at": self.created_at,
            "context_hash": self.context_hash,
        }


def _context_digest(context: ProspectiveAcquisitionExecutionContext) -> str:
    value = {
        "execution_context_id": context.execution_context_id,
        "population_id": context.population_id,
        "batch_id": context.batch_id,
        "owner_hash": context.owner_hash,
        "owner_scope_source": context.owner_scope_source,
        "provider_resolution_policy": context.provider_resolution_policy,
        "provider_binding": (
            context.provider_binding.safe_dict() if context.provider_binding else None
        ),
        "protocol_version": context.protocol_version,
        "created_at": context.created_at,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_context(context: ProspectiveAcquisitionExecutionContext) -> None:
    if context.provider_binding is None or not context.context_hash:
        raise ExecutionIdentityError("provider_binding_not_frozen")
    if context.context_hash != _context_digest(context):
        raise ExecutionIdentityError("execution_context_hash_mismatch")


def create_execution_context(
    authenticated_session: ClientSession | None,
    *,
    population_id: str,
    batch_id: str,
    protocol_version: str,
    created_at: datetime | None = None,
) -> ProspectiveAcquisitionExecutionContext:
    """Create context only from the authenticated application session object."""
    if not isinstance(authenticated_session, ClientSession):
        raise OwnerScopeRequired("authenticated_client_session_required")
    owner_hash = authenticated_session.owner_hash
    if not isinstance(owner_hash, str) or not _OWNER_HASH_PATTERN.fullmatch(owner_hash):
        raise OwnerScopeRequired("authenticated_owner_scope_missing_or_invalid")
    return ProspectiveAcquisitionExecutionContext(
        execution_context_id=str(uuid4()),
        population_id=population_id,
        batch_id=batch_id,
        owner_hash=owner_hash,
        owner_scope_source=OWNER_SCOPE_SOURCE,
        provider_resolution_policy=PROVIDER_RESOLUTION_POLICY,
        protocol_version=protocol_version,
        created_at=(created_at or datetime.now(UTC)).isoformat(),
    )


class ProviderProfileLister(Protocol):
    async def list_profiles(self, owner_hash: str) -> list[ProviderProfileView]: ...


class ActiveProfileRepository(Protocol):
    async def get_active(self, owner_hash: str, profile_id: Any) -> tuple[Any, Any]: ...


def _credential_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProviderCredentialVersionInvalid("PROVIDER_CREDENTIAL_VERSION_INVALID")
    return value


def _endpoint_family(url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ExecutionIdentityError("provider_endpoint_invalid")
    host = parsed.hostname.lower()
    port = parsed.port
    default_port = (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    )
    if port and not default_port:
        return f"{parsed.scheme.lower()}://{host}:{port}"
    return f"{parsed.scheme.lower()}://{host}"


async def resolve_provider_binding(
    context: ProspectiveAcquisitionExecutionContext,
    profile_service: ProviderProfileLister,
    profile_repository: ActiveProfileRepository,
) -> FrozenProviderBinding:
    """Select only the authenticated owner's default via the production service."""
    profiles = await profile_service.list_profiles(context.owner_hash)
    defaults = [
        profile
        for profile in profiles
        if getattr(profile, "status", None) == "active" and getattr(profile, "is_default", False)
    ]
    if not defaults:
        raise ProviderCardinalityError("NO_DEFAULT_PROFILE_FOR_ACQUISITION_OWNER")
    if len(defaults) != 1:
        raise ProviderCardinalityError("MULTIPLE_DEFAULT_PROFILES_FOR_ACQUISITION_OWNER")

    view = defaults[0]
    row, credential = await profile_repository.get_active(context.owner_hash, view.profile_id)
    if (
        getattr(row, "owner_hash", None) != context.owner_hash
        or getattr(row, "status", None) != "active"
        or getattr(row, "is_default", None) is not True
        or getattr(row, "deleted_at", None) is not None
        or getattr(credential, "revoked_at", None) is not None
        or getattr(credential, "deleted_at", None) is not None
    ):
        raise ExecutionIdentityError("provider_profile_changed_during_resolution")

    view_credential_id = str(view.credential_version_id)
    row_credential_version = _credential_version(
        getattr(credential, "credential_version", None)
    )
    view_credential_version = _credential_version(
        getattr(view, "credential_version", None)
    )
    if row_credential_version != view_credential_version:
        raise ProviderCredentialVersionInvalid("PROVIDER_CREDENTIAL_VERSION_INVALID")
    if (
        str(row.id) != str(view.profile_id)
        or str(credential.id) != view_credential_id
        or int(row.version) != int(view.version)
        or row.adapter_type != view.adapter_type.value
        or row.model != view.model
        or row.normalized_base_url != view.base_url
    ):
        raise ExecutionIdentityError("provider_profile_changed_during_resolution")

    safe_input = {
        "profile_id": str(view.profile_id),
        "adapter": view.adapter_type.value,
        "model": view.model,
        "base_url": view.base_url,
        "auth_type": row.auth_type,
        "profile_version": view.version,
        "credential": {
            "id": str(credential.id),
            "version": row_credential_version,
            "present": True,
        },
        "non_secret_settings": row.non_secret_settings,
    }
    safe_payload, fingerprint = safe_provider_fingerprint(safe_input)
    endpoint = safe_payload["endpoint"]
    if not isinstance(endpoint, dict):
        raise ExecutionIdentityError("provider_endpoint_fingerprint_invalid")
    endpoint_family = f"{endpoint['scheme']}://{endpoint['host']}"
    if endpoint.get("port") is not None:
        endpoint_family += f":{endpoint['port']}"
    return FrozenProviderBinding(
        profile_id=str(view.profile_id),
        adapter=view.adapter_type.value,
        model=view.model,
        endpoint_family=endpoint_family,
        profile_version=view.version,
        credential_version_id=str(credential.id),
        credential_version=row_credential_version,
        credential_present=True,
        fingerprint_contract=FINGERPRINT_CONTRACT_VERSION,
        provider_safe_config_fingerprint=fingerprint,
    )


async def assert_provider_binding_unchanged(
    context: ProspectiveAcquisitionExecutionContext,
    profile_service: ProviderProfileLister,
    profile_repository: ActiveProfileRepository,
) -> None:
    validate_frozen_context(context)
    expected = context.provider_binding
    try:
        current = await resolve_provider_binding(context, profile_service, profile_repository)
    except ExecutionIdentityError as exc:
        raise ExecutionProviderBindingChanged("EXECUTION_PROVIDER_BINDING_CHANGED") from exc
    if current != expected:
        raise ExecutionProviderBindingChanged("EXECUTION_PROVIDER_BINDING_CHANGED")


def bind_task_to_context(
    *, task_id: str, run_id: str, task_text: str, context: ProspectiveAcquisitionExecutionContext
) -> dict[str, str]:
    validate_frozen_context(context)
    return {
        "task_id": task_id,
        "run_id": run_id,
        "task_text": task_text,
        "execution_context_id": context.execution_context_id,
    }


async def create_bound_research_run(
    context: ProspectiveAcquisitionExecutionContext,
    run_service: Any,
    *,
    profile_service: ProviderProfileLister,
    profile_repository: ActiveProfileRepository,
    idempotency_key: str,
    query: str,
    saved_profile_version_id: Any,
    budget_tier: str,
) -> Any:
    validate_frozen_context(context)
    binding = context.provider_binding
    if binding is None:
        raise ExecutionIdentityError("provider_binding_not_frozen")
    await assert_provider_binding_unchanged(context, profile_service, profile_repository)
    if str(saved_profile_version_id) != binding.credential_version_id:
        raise ExecutionContextMismatch("run_provider_binding_mismatch")
    return await run_service.create_run(
        context.owner_hash,
        idempotency_key=idempotency_key,
        query=query,
        saved_profile_version_id=saved_profile_version_id,
        budget_tier=budget_tier,
    )


def operator_probe_reference(
    context: ProspectiveAcquisitionExecutionContext,
) -> dict[str, object]:
    """Safe contract reference; the fingerprint is not a substitute for owner_hash."""
    artifact = context.safe_artifact()
    return {
        "execution_context_id": artifact["execution_context_id"],
        "context_hash": artifact["context_hash"],
        "owner_scope_fingerprint": artifact["owner_scope_fingerprint"],
        "owner_scope_source": artifact["owner_scope_source"],
        "provider_binding": artifact["provider_binding"],
    }


def verify_operator_probe_reference(
    context: ProspectiveAcquisitionExecutionContext,
    reference: dict[str, object],
) -> None:
    """Verify an operator artifact reference against a trusted resolved context."""
    expected = operator_probe_reference(context)
    if reference != expected:
        raise ExecutionContextMismatch("operator_execution_context_reference_mismatch")
