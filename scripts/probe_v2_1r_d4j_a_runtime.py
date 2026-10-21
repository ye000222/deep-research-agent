"""Read-only canonical-runtime identity and v3.1 import probe for D4-J-A."""

# ruff: noqa: E501, I001

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, "/app/scripts")

from sqlalchemy import select
from uuid import UUID

from app.core.config import get_settings
from app.domain.provider_profiles import ProviderProfileView
from app.domain.source_context_extraction_v3_1 import (
    SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31,
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
    MeasurementSemanticsV31,
    SourceScopeV31,
    validate_source_context_v31,
)
from app.infrastructure.db.models import ProviderProfileRow
from app.infrastructure.runtime import ApplicationRuntime
from app.services.source_context_extractor_v3_1 import SourceContextExtractorV31
from app.llm.canary_strict import SingleOutboundClient

from provider_fingerprint_contract import safe_provider_fingerprint


def safe_profile(profile: ProviderProfileView, settings: dict[str, object]) -> dict[str, object]:
    return {
        "profile_id": str(profile.profile_id),
        "adapter": profile.adapter_type.value,
        "model": profile.model,
        "base_url": profile.base_url,
        "auth_type": "bearer",
        "profile_version": profile.version,
        "is_default": profile.is_default,
        "status": profile.status,
        "credential": {
            "id": str(profile.credential_version_id),
            "version": profile.credential_version,
            "present": True,
        },
        "non_secret_settings": settings,
    }


async def main() -> None:
    settings = get_settings()
    runtime = ApplicationRuntime.build(settings)
    try:
        if runtime.profile_service is None:
            raise RuntimeError("provider_profile_service_unavailable")
        async with runtime.business_db.session_factory() as session:
            locator = await session.scalar(
                select(ProviderProfileRow).where(
                    ProviderProfileRow.id == UUID("49a9ae5c-b062-4617-805f-dc587b706ce1"),
                    ProviderProfileRow.deleted_at.is_(None),
                )
            )
            if locator is None:
                raise RuntimeError("historical_profile_locator_not_found")
            owner_hash = locator.owner_hash
            current_non_secret_settings = dict(locator.non_secret_settings or {})
            current_auth_type = locator.auth_type
        views = await runtime.profile_service.list_profiles(owner_hash)
        matches = [view for view in views if str(view.profile_id) == str(locator.id)]
        if len(matches) != 1:
            raise RuntimeError(f"profile_service_resolution_count:{len(matches)}")
        profile = matches[0]
        if profile.status != "active" or not profile.is_default:
            raise RuntimeError("located_provider_profile_not_current_active_default")
        profile_safe = safe_profile(profile, current_non_secret_settings)
        profile_safe["auth_type"] = current_auth_type
        payload, fingerprint = safe_provider_fingerprint(profile_safe)
        endpoint = urlsplit(profile.base_url)
        from app.llm.adapters import LLMGateway

        output = {
            "runtime_kind": "ephemeral compose api, current source bind mount",
            "runtime_source_revision": settings.source_revision,
            "module_file": str(Path(__file__).resolve()),
            "v3_1": {
                "extractor_class": f"{SourceContextExtractorV31.__module__}.{SourceContextExtractorV31.__name__}",
                "extractor_module_file": sys.modules[SourceContextExtractorV31.__module__].__file__,
                "prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
                "schema": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
                "domain": SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31,
                "typed_measurement_model": MeasurementSemanticsV31.__module__ + "." + MeasurementSemanticsV31.__name__,
                "scope_node_model": SourceScopeV31.__module__ + "." + SourceScopeV31.__name__,
                "validator": validate_source_context_v31.__module__ + "." + validate_source_context_v31.__name__,
            },
            "guard": {
                "class": SingleOutboundClient.__module__ + "." + SingleOutboundClient.__name__,
                "gateway": LLMGateway.__module__ + "." + LLMGateway.__name__,
            },
            "provider": {
                "profile_id": str(profile.profile_id),
                "status": profile.status,
                "is_default": profile.is_default,
                "adapter": profile.adapter_type.value,
                "model": profile.model,
                "normalized_endpoint": profile.base_url,
                "endpoint_family": f"{endpoint.scheme}://{endpoint.hostname}{endpoint.path.rstrip('/')}" ,
                "profile_version": profile.version,
                "credential_present": True,
                "credential_version": profile.credential_version,
                "credential_id": str(profile.credential_version_id),
                "safe_payload": payload,
                "safe_fingerprint": fingerprint,
                "fingerprint_contract": "provider-safe-config-fingerprint.v1",
            },
            "activity": {"provider_generation_calls": 0, "external_provider_http": 0, "db_writes": 0, "holdout_access": 0},
        }
        print(json.dumps(output, sort_keys=True, ensure_ascii=False))
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
