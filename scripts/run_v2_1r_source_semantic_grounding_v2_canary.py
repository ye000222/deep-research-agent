"""Readiness and explicitly-invoked runner for the frozen source-grounding v2 cohort.

The readiness mode is offline except for a read-only lookup of the configured provider
profile. It never creates a database object or sends a provider request. Execution is a
separate explicit function and is not invoked by this task's readiness command.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.core.config import get_settings  # noqa: E402
from app.domain.providers import (  # noqa: E402
    AdapterType,
    CanonicalModelRequest,
    CanonicalModelResult,
)
from app.domain.source_context_grounding_v2 import (  # noqa: E402
    SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
    SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
    SourceExtractionResponseV2,
)
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow  # noqa: E402
from app.infrastructure.db.provider_profiles import ProviderProfileRepository  # noqa: E402
from app.llm.adapters import LLMGateway, ModelGatewayError  # noqa: E402
from app.llm.canary_strict import SingleOutboundClient  # noqa: E402
from app.security.secrets import SecretCipher  # noqa: E402
from app.services.provider_profiles import (  # noqa: E402
    ProviderProfileService,
    ProviderProfileTestMaterial,
)
from app.services.source_context_extractor_v2 import (  # noqa: E402
    SOURCE_CONTEXT_V2_INSTRUCTIONS,
    SourceContextExtractorV2,
    validate_source_grounding_v2,
)

SELECTION_MANIFEST = (
    ROOT
    / "artifacts"
    / (
        "v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_"
        "selection_manifest.json"
    )
)
SOURCE_DATASET = ROOT / "artifacts" / "v2_1_dev_extension_v2_candidates.json"
READINESS_MANIFEST = (
    ROOT / "artifacts" / ("v2_1r_source_semantic_grounding_evidence_v2_stage_b_readiness.json")
)
RESULT_ARTIFACT = (
    ROOT
    / "artifacts"
    / ("v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_result.json")
)
EXPECTED_SELECTION_SHA256 = "4aaa89330b5212537a4ac389f570038e8a3bd27bf9bc3fb6a58161d1c01c511a"
EXPECTED_PROFILE_ID = UUID("49a9ae5c-b062-4617-805f-dc587b706ce1")
EXPECTED_ADAPTER = "openai_compatible_chat"
EXPECTED_MODEL = "deepseek-flash"
EXPECTED_ENDPOINT_HOST = "api.deepseek.com"
EXPECTED_CREDENTIAL_VERSION = 4
EXPECTED_CLAIM_IDS = (
    "01a0c86c-7a7b-7815-94a5-996a3aea5ae6",
    "01a0c438-499d-7fbf-84b7-0cec4da63e25",
    "01a0c41c-281a-7f75-a720-474df7782723",
    "01a0c43b-3e7a-704f-b356-ad6f591a17ff",
    "01a0c82b-c337-7e71-8bdd-9e0954bd1aa0",
    "01a0c845-7eb2-7f79-b92d-5d6fcdd8f257",
    "01a0c41a-7636-721d-9b9d-7ee9ea95e573",
    "01a0c415-6a29-7cfa-b58e-4b14f51c469e",
)
EXECUTION_POLICY = {
    "version": "canary-single-outbound-per-claim-v1",
    "max_provider_attempts_per_claim": 1,
    "max_external_http_outbounds_per_claim": 1,
    "retry": 0,
    "regeneration": False,
    "fallback": "none; additional sends blocked before HTTP client",
    "repair": "none",
    "fusion_policy": "NONE",
    "database_writes": False,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected_json_object")
    return value


def verify_selection_and_resolve_claims() -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_raw = SELECTION_MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    declared_hash = manifest.get("manifest_sha256")
    normalized = dict(manifest)
    normalized.pop("manifest_sha256", None)
    if (
        stable_hash(normalized) != EXPECTED_SELECTION_SHA256
        or declared_hash != EXPECTED_SELECTION_SHA256
    ):
        raise ValueError("selection_manifest_hash_mismatch")
    if (
        manifest.get("development_only") is not True
        or manifest.get("holdout_content_read") is not False
    ):
        raise ValueError("selection_manifest_development_attestation_failed")
    selected = manifest.get("selection", {}).get("selected_claims", [])
    if tuple(row.get("claim_id") for row in selected) != EXPECTED_CLAIM_IDS:
        raise ValueError("selection_manifest_claim_ids_mismatch")

    dataset_bytes = SOURCE_DATASET.read_bytes()
    if sha256_bytes(dataset_bytes) != manifest.get("source_dataset", {}).get("sha256"):
        raise ValueError("development_dataset_hash_mismatch")
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    if dataset.get("purpose") != "development_only":
        raise ValueError("source_dataset_not_development_only")

    texts: dict[str, str] = {}
    for pair in dataset.get("pairs", []):
        for id_key, text_key in (("claim_a_id", "claim_a_text"), ("claim_b_id", "claim_b_text")):
            claim_id = pair.get(id_key)
            text = pair.get(text_key)
            if claim_id in EXPECTED_CLAIM_IDS and isinstance(text, str):
                existing = texts.get(claim_id)
                if existing is not None and existing != text:
                    raise ValueError("duplicate_claim_text_conflict")
                texts[claim_id] = text
    claims: list[dict[str, str]] = []
    for frozen in selected:
        claim_id = frozen["claim_id"]
        text = texts.get(claim_id)
        if not text or sha256_bytes(text.encode("utf-8")) != frozen.get("claim_sha256"):
            raise ValueError("frozen_claim_hash_mismatch")
        claims.append(
            {"claim_id": claim_id, "claim_text": text, "claim_sha256": frozen["claim_sha256"]}
        )
    if len(claims) != 8:
        raise ValueError("frozen_claim_count_mismatch")
    return manifest, claims


def compute_component_hashes() -> dict[str, str]:
    from app.domain import source_context_grounding_v2 as domain
    from app.domain import truth_condition_context as truth_context
    from app.services import source_context_extractor as legacy_extractor
    from app.services import source_context_extractor_v2 as extractor_v2

    return {
        "prompt_source_sha256": sha256_bytes(Path(extractor_v2.__file__).read_bytes()),
        "prompt_text_sha256": sha256_bytes(SOURCE_CONTEXT_V2_INSTRUCTIONS.encode("utf-8")),
        "schema_source_sha256": sha256_bytes(Path(domain.__file__).read_bytes()),
        "schema_sha256": stable_hash(SourceExtractionResponseV2.model_json_schema()),
        "domain_sha256": sha256_bytes(Path(domain.__file__).read_bytes()),
        "executor_parser_sha256": sha256_bytes(Path(extractor_v2.__file__).read_bytes()),
        "grounding_validator_sha256": sha256_bytes(
            inspect.getsource(validate_source_grounding_v2).encode("utf-8")
        ),
        "truth_condition_context_sha256": sha256_bytes(Path(truth_context.__file__).read_bytes()),
        # The frozen v2 manifest explicitly states that the v2 executor has no separate
        # normalizer. Its normalization reference is pinned through TruthConditionContext.
        "auxiliary_detector_reference_sha256": sha256_bytes(
            inspect.getsource(legacy_extractor._auxiliary_comparison).encode("utf-8")
        ),
    }


def verify_frozen_components(manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = manifest.get("frozen_components", {})
    expected = frozen.get("component_hashes", {})
    actual = compute_component_hashes()
    comparisons = {
        key: {"expected": expected.get(key), "actual": value, "match": expected.get(key) == value}
        for key, value in actual.items()
    }
    normalizer_reference = expected.get("normalizer_reference_sha256")
    normalizer_reference_matches = (
        isinstance(normalizer_reference, str)
        and len(normalizer_reference) == 64
        and comparisons["truth_condition_context_sha256"]["match"]
        and "no separate normalizer" in " ".join(frozen.get("normalizer_versions", [])).lower()
    )
    return {
        "prompt_version": SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
        "schema_version": SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
        "comparisons": comparisons,
        "normalizer_reference_sha256": normalizer_reference,
        "normalizer_reference_verified_via_frozen_truth_context": normalizer_reference_matches,
        "all_match": all(item["match"] for item in comparisons.values())
        and normalizer_reference_matches,
        "expected_versions_match": (
            frozen.get("prompt_version") == SOURCE_CONTEXT_GROUNDING_PROMPT_V2
            and frozen.get("schema_version") == SOURCE_CONTEXT_GROUNDING_SCHEMA_V2
            and frozen.get("fusion_policy") == "NONE"
        ),
    }


class RuntimeIdentityError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


async def resolve_provider_runtime_identity() -> tuple[dict[str, Any], ProviderProfileTestMaterial]:
    """Resolve and decrypt only the frozen profile; never return or log credential text."""
    settings = get_settings()
    key_file = settings.secret_master_key_file
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            row = (
                await session.execute(
                    select(ProviderProfileRow, CredentialVersionRow)
                    .join(
                        CredentialVersionRow,
                        CredentialVersionRow.profile_id == ProviderProfileRow.id,
                    )
                    .where(
                        ProviderProfileRow.id == EXPECTED_PROFILE_ID,
                        ProviderProfileRow.status == "active",
                        ProviderProfileRow.deleted_at.is_(None),
                        ProviderProfileRow.is_default.is_(True),
                        CredentialVersionRow.revoked_at.is_(None),
                        CredentialVersionRow.deleted_at.is_(None),
                    )
                )
            ).first()
            if row is None:
                raise RuntimeIdentityError("ACTIVE_PROFILE_NOT_FOUND")
            profile_row, credential_row = row
            safe = {
                "provider_type": profile_row.adapter_type,
                "profile_id": str(profile_row.id),
                "model_id": profile_row.model,
                "base_url": profile_row.normalized_base_url,
                "endpoint_host": profile_row.endpoint_host,
                "configuration_source": "current_workspace Settings + active PostgreSQL profile",
                "profile_version": profile_row.version,
                "credential_version": credential_row.credential_version,
                "credential_present": bool(credential_row.ciphertext and credential_row.nonce),
                "credential_source_category": "encrypted active DB credential version",
                "credential_decryption": "not_attempted",
            }
            owner_hash = profile_row.owner_hash
        if settings.secret_master_key_base64 is None and not key_file.is_file():
            raise RuntimeIdentityError("MASTER_KEY_NOT_CONFIGURED")
        configured_key = settings.secret_master_key_base64
        encoded_key = (
            configured_key.get_secret_value().encode("ascii")
            if configured_key is not None
            else key_file.read_bytes().strip()
        )
        master_key = base64.b64decode(encoded_key, validate=True)
        if len(master_key) != 32:
            raise RuntimeIdentityError("MASTER_KEY_INVALID")
        cipher = SecretCipher(master_key)
        service = ProviderProfileService(
            ProviderProfileRepository(sessions),
            cipher,
            allow_insecure_endpoints=settings.allow_insecure_provider_endpoints,
        )
        material = await service.get_test_material(owner_hash, EXPECTED_PROFILE_ID)
        safe["credential_decryption"] = "success"
        if (
            material.adapter_type.value != EXPECTED_ADAPTER
            or material.model != EXPECTED_MODEL
            or safe["endpoint_host"] != EXPECTED_ENDPOINT_HOST
            or safe["credential_version"] != EXPECTED_CREDENTIAL_VERSION
        ):
            raise RuntimeIdentityError("RUNTIME_IDENTITY_DIFFERS_FROM_AUTHORIZATION")
        safe["credential_version_id"] = str(material.credential_version_id)
        safe["identity_hash"] = stable_hash(
            {key: value for key, value in safe.items() if key != "credential_decryption"}
        )
        return safe, material
    except RuntimeIdentityError:
        raise
    except Exception as exc:
        # SQLAlchemy/driver exception strings can include connection details; retain type only.
        raise RuntimeIdentityError(
            f"RUNTIME_PROFILE_RESOLUTION_{type(exc).__name__.upper()}"
        ) from None
    finally:
        await engine.dispose()


def build_readiness_manifest() -> dict[str, Any]:
    manifest, claims = verify_selection_and_resolve_claims()
    component_check = verify_frozen_components(manifest)
    runner_hash = sha256_bytes(Path(__file__).read_bytes())
    execution_policy_hash = stable_hash(EXECUTION_POLICY)
    readiness: dict[str, Any] = {
        "artifact_version": "v2_1r_source_semantic_grounding_v2_stage_b_readiness_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_manifest_path": str(SELECTION_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "selection_manifest_normalized_sha256": EXPECTED_SELECTION_SHA256,
        "claim_ids": [claim["claim_id"] for claim in claims],
        "claim_hashes": {claim["claim_id"]: claim["claim_sha256"] for claim in claims},
        "runner_path": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "runner_sha256": runner_hash,
        "v2_component_verification": component_check,
        "provider_runtime_identity": {"status": "unresolved"},
        "execution_policy": EXECUTION_POLICY,
        "execution_policy_sha256": execution_policy_hash,
        "fusion_policy": "NONE",
        "provider_calls": 0,
        "database_writes": 0,
        "holdout_content_read": False,
        "readiness": "NOT_READY",
    }
    try:
        identity, _ = asyncio.run(resolve_provider_runtime_identity())
        readiness["provider_runtime_identity"] = {"status": "resolved", **identity}
    except RuntimeIdentityError as exc:
        readiness["provider_runtime_identity"] = {
            "status": "unresolved",
            "safe_reason": exc.safe_code,
        }
    components_pass = component_check["all_match"] and component_check["expected_versions_match"]
    identity = readiness["provider_runtime_identity"]
    expected_identity = (
        identity.get("status") == "resolved"
        and identity.get("profile_id") == str(EXPECTED_PROFILE_ID)
        and identity.get("provider_type") == EXPECTED_ADAPTER
        and identity.get("model_id") == EXPECTED_MODEL
        and identity.get("endpoint_host") == EXPECTED_ENDPOINT_HOST
        and identity.get("credential_version") == EXPECTED_CREDENTIAL_VERSION
    )
    if components_pass and expected_identity:
        readiness["readiness"] = "READY"
    readiness["readiness_reasons"] = [
        reason
        for passed, reason in (
            (components_pass, "FROZEN_V2_COMPONENT_MISMATCH"),
            (expected_identity, "PROVIDER_RUNTIME_IDENTITY_UNRESOLVED_OR_MISMATCH"),
        )
        if not passed
    ]
    return readiness


class CapturingGateway:
    """Observe the canonical model result without transforming it."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway
        self.provider_attempt_count = 0
        self.model_result: dict[str, Any] | None = None
        self.safe_error: dict[str, str] | None = None

    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> CanonicalModelResult:
        if allow_regeneration:
            raise RuntimeError("canary_regeneration_must_be_disabled")
        if request.metadata.get("prompt_version") != SOURCE_CONTEXT_GROUNDING_PROMPT_V2:
            raise RuntimeError("v2_prompt_route_mismatch")
        if request.metadata.get("schema_version") != SOURCE_CONTEXT_GROUNDING_SCHEMA_V2:
            raise RuntimeError("v2_schema_route_mismatch")
        if request.metadata.get("fusion_policy") != "NONE":
            raise RuntimeError("canary_fusion_must_be_none")
        if self.provider_attempt_count:
            raise RuntimeError("canary_provider_attempt_budget_exhausted")
        self.provider_attempt_count += 1
        try:
            result = await self._gateway.generate_structured(
                adapter_type=adapter_type,
                base_url=base_url,
                api_key=api_key,
                request=request,
                allow_regeneration=False,
            )
        except ModelGatewayError as exc:
            self.safe_error = {"code": exc.code, "detail_code": exc.detail_code or ""}
            raise
        except Exception as exc:
            self.safe_error = {"code": type(exc).__name__, "detail_code": "UNCLASSIFIED"}
            raise
        self.model_result = result.model_dump(mode="json")
        return result


async def execute_frozen_claims(
    claims: list[dict[str, str]], material: ProviderProfileTestMaterial
) -> dict[str, Any]:
    """Execute only an explicitly invoked frozen cohort.

    Readiness mode never calls this function.
    """
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as http_client:
        for claim in claims:
            results.append(await run_one_claim(claim, material, http_client))
    return {
        "selection_manifest_sha256": EXPECTED_SELECTION_SHA256,
        "prompt_version": SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
        "schema_version": SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
        "fusion_policy": "NONE",
        "claims": results,
        "provider_calls": sum(row["external_http_outbound_count"] for row in results),
        "database_writes": 0,
    }


async def run_one_claim(
    claim: dict[str, str],
    material: ProviderProfileTestMaterial,
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    strict_client = SingleOutboundClient(http_client)
    capture = CapturingGateway(LLMGateway(strict_client))  # type: ignore[arg-type]
    extractor = SourceContextExtractorV2(
        capture,
        adapter_type=material.adapter_type,
        base_url=material.base_url,
        api_key=material.api_key,
        model=material.model,
        max_calls=1,
    )
    result = await extractor.extract(claim["claim_text"], claim_ref=claim["claim_id"])
    if strict_client.external_http_outbound_count > 1:
        raise RuntimeError("strict_outbound_invariant_violated")
    return {
        "claim_id": claim["claim_id"],
        "claim_sha256": claim["claim_sha256"],
        "model_result": capture.model_result,
        "grounding_result": result.model_dump(mode="json"),
        "safe_gateway_error": capture.safe_error,
        "provider_attempt_count": capture.provider_attempt_count,
        "external_http_outbound_count": strict_client.external_http_outbound_count,
        "blocked_additional_attempt_count": strict_client.blocked_additional_attempt_count,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("readiness", "execute"), default="readiness")
    parser.add_argument("--execute-frozen-authorized-cohort", action="store_true")
    args = parser.parse_args()
    if args.mode == "readiness":
        readiness = build_readiness_manifest()
        write_json(READINESS_MANIFEST, readiness)
        print(
            json.dumps(
                {
                    "readiness": readiness["readiness"],
                    "reasons": readiness["readiness_reasons"],
                    "provider_calls": 0,
                    "artifact": str(READINESS_MANIFEST),
                },
                ensure_ascii=False,
            )
        )
        return 0 if readiness["readiness"] == "READY" else 2
    if not args.execute_frozen_authorized_cohort:
        print("execution requires --execute-frozen-authorized-cohort; provider_calls=0")
        return 2
    readiness = build_readiness_manifest()
    write_json(READINESS_MANIFEST, readiness)
    if readiness["readiness"] != "READY":
        print(
            json.dumps(
                {
                    "readiness": "NOT_READY",
                    "reasons": readiness["readiness_reasons"],
                    "provider_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return 2
    manifest, claims = verify_selection_and_resolve_claims()
    del manifest
    identity, material = asyncio.run(resolve_provider_runtime_identity())
    if identity.get("identity_hash") != readiness["provider_runtime_identity"].get("identity_hash"):
        print("provider identity changed after readiness; provider_calls=0")
        return 2
    result = asyncio.run(execute_frozen_claims(claims, material))
    write_json(RESULT_ARTIFACT, result)
    print(
        json.dumps(
            {"provider_calls": result["provider_calls"], "result_artifact": str(RESULT_ARTIFACT)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
