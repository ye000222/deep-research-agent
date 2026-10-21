"""Frozen six-Claim Source Context Extraction canary (no DB writes).

The script is copied into the running API container only to reuse its configured
database/profile decryption environment. It never stores credentials or submits
manual labels to the model. Run ``prepare`` before ``execute``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, os.environ.get("CANARY_APP_ROOT", "/app/apps/api"))

from app.core.config import get_settings
from app.domain.source_context_extraction import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_VERSION,
    SOURCE_CONTEXT_NORMALIZER_VERSION,
    SourceContextExtractionResult,
    SourceExtractionResponse,
)
from app.domain.truth_condition_context import ContextType
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.llm.adapters import LLMGateway, ModelGatewayError
from app.security.secrets import SecretCipher, load_or_create_master_key
from app.services.provider_profiles import ProviderProfileService
from app.services.source_context_extractor import (
    SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS,
    SourceContextExtractor,
    _auxiliary_comparison,
    _ground,
    _ground_response,
    _grounding_status,
)


CLAIM_IDS = (
    "01a0ae00-e95a-76bc-a1b2-35c919339bb6",
    "01a0c872-0ac8-7a2b-923a-637c1bb2a9aa",
    "01a0c418-10dc-7b31-ba54-fd6f66bcd240",
    "01a0c859-9dd1-770b-b136-ad2eccdf955e",
    "01a0c827-8c47-7964-b08b-36d2033c3aee",
    "01a0c82a-cdae-79f0-bcd3-83c1de09fd14",
)
EXPECTED_PROFILE_ID = UUID("49a9ae5c-b062-4617-805f-dc587b706ce1")
EXPECTED_AUDIT_MANIFEST_SHA256 = "35018bd6fbc128bc9c3c064e71ed12bf5fa3492755a02ebb337a214abe10b52d"
ARTIFACT_DIR = Path("/app/artifacts")
MANIFEST_PATH = ARTIFACT_DIR / "v2_1r_source_context_extraction_real_canary_manifest.json"
RESULT_PATH = ARTIFACT_DIR / "v2_1r_source_context_extraction_real_canary_result.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_hash(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def component_hashes(runner: Path, audit_selection: Path, manual_audit: Path) -> dict[str, str]:
    from app.domain import truth_condition_context as tcc
    from app.services import source_context_extractor as extractor

    schema = SourceExtractionResponse.model_json_schema()
    return {
        "prompt_source_sha256": sha256_file(Path(extractor.__file__)),
        "prompt_text_sha256": sha256_bytes(SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS.encode()),
        "schema_source_sha256": sha256_file(Path(__import__("app.domain.source_context_extraction", fromlist=["x"]).__file__)),
        "schema_sha256": stable_hash(schema),
        "service_parser_sha256": sha256_file(Path(extractor.__file__)),
        "grounding_validator_sha256": stable_hash(
            {name: inspect.getsource(fn) for name, fn in {
                "ground": _ground,
                "ground_response": _ground_response,
                "grounding_status": _grounding_status,
            }.items()}
        ),
        "truth_condition_context_sha256": sha256_file(Path(tcc.__file__)),
        "typed_normalizer_sha256": stable_hash({
            "normalize_source_semantic_value": inspect.getsource(extractor.normalize_source_semantic_value),
            "canonicalize_source_context": inspect.getsource(extractor.canonicalize_source_context),
        }),
        "auxiliary_detector_sha256": sha256_bytes(inspect.getsource(_auxiliary_comparison).encode()),
        "runner_sha256": sha256_file(runner),
        "audit_selection_sha256": sha256_file(audit_selection),
        "manual_audit_sha256": sha256_file(manual_audit),
    }


async def resolve_profile() -> tuple[dict[str, Any], Any]:
    settings = get_settings()
    if settings.secret_master_key_base64 is None and not settings.secret_master_key_file.is_file():
        raise RuntimeError("credential_master_key_unavailable")
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            row = (await session.execute(
                select(ProviderProfileRow, CredentialVersionRow)
                .join(CredentialVersionRow, CredentialVersionRow.profile_id == ProviderProfileRow.id)
                .where(
                    ProviderProfileRow.id == EXPECTED_PROFILE_ID,
                    ProviderProfileRow.status == "active",
                    ProviderProfileRow.deleted_at.is_(None),
                    ProviderProfileRow.is_default.is_(True),
                    ProviderProfileRow.endpoint_host == "api.deepseek.com",
                    CredentialVersionRow.revoked_at.is_(None),
                    CredentialVersionRow.deleted_at.is_(None),
                )
            )).first()
            if row is None:
                raise RuntimeError("configured_provider_profile_not_resolved")
            profile_row, credential_row = row
            owner_hash = profile_row.owner_hash
            profile_view = {
                "profile_id": str(profile_row.id),
                "adapter_type": profile_row.adapter_type,
                "endpoint_host": profile_row.endpoint_host,
                "model": profile_row.model,
                "profile_version": profile_row.version,
                "credential_version_id": str(credential_row.id),
                "credential_version": credential_row.credential_version,
            }
        cipher = SecretCipher(load_or_create_master_key(settings))
        profile_service = ProviderProfileService(
            ProviderProfileRepository(sessions), cipher,
            allow_insecure_endpoints=settings.allow_insecure_provider_endpoints,
        )
        material = await profile_service.get_test_material(owner_hash, EXPECTED_PROFILE_ID)
        if (
            material.profile_id != EXPECTED_PROFILE_ID
            or material.adapter_type.value != "openai_compatible_chat"
            or material.model != "deepseek-flash"
            or material.base_url.split("/")[2].lower() != "api.deepseek.com"
            or not material.api_key.get_secret_value()
            or profile_view["credential_version"] != 4
            or str(material.credential_version_id) != profile_view["credential_version_id"]
        ):
            raise RuntimeError("configured_provider_material_mismatch")
        profile_view.update({
            "base_url_path_present": bool("/" in material.base_url.removeprefix("https://")),
        })
        return profile_view, material
    finally:
        await engine.dispose()


def load_claims(audit_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("development_only") is not True or audit.get("holdout_content_read") is not False:
        raise RuntimeError("development_only_audit_attestation_failed")
    if audit.get("selection_manifest_sha256") != EXPECTED_AUDIT_MANIFEST_SHA256:
        raise RuntimeError("frozen_audit_manifest_hash_mismatch")
    rows = {item["claim_id"]: item for item in audit.get("claims", [])}
    if len(rows) != 40 or any(claim_id not in rows for claim_id in CLAIM_IDS):
        raise RuntimeError("frozen_40_claim_membership_failed")
    selected = [rows[claim_id] for claim_id in CLAIM_IDS]
    if any(not isinstance(item.get("claim_text"), str) or not item["claim_text"].strip() for item in selected):
        raise RuntimeError("selected_claim_text_missing")
    return audit, selected


async def prepare(args: argparse.Namespace) -> None:
    audit, selected = load_claims(args.manual_audit)
    profile, _material = await resolve_profile()
    hashes = component_hashes(args.runner, args.audit_selection, args.manual_audit)
    manifest = {
        "experiment_id": "v2_1r_source_context_extraction_real_development_canary_6",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_ids": list(CLAIM_IDS),
        "claims": [
            {
                "claim_id": item["claim_id"],
                "claim_sha256": sha256_bytes(item["claim_text"].encode("utf-8")),
                "cohort_recorded_in_development_audit": item.get("cohort"),
                "ordinal_in_frozen_audit": item.get("ordinal"),
            }
            for item in selected
        ],
        "selection_policy": {
            "version": "deterministic_generic_structural_stratification_v1",
            "policy": "six pre-frozen IDs from the 40-Claim development audit; strata cover shared/multiple context, epistemic attribution, condition/scope, method/measurement, temporal semantics, and multi-assertion structure; no selection based on entity, industry, source, benchmark, or expected outcome",
            "frozen_before_provider_calls": True,
        },
        "audit_set": {
            "identity": audit.get("selection_manifest"),
            "selection_manifest_sha256": audit.get("selection_manifest_sha256"),
            "manual_audit_sha256": hashes["manual_audit_sha256"],
            "selection_artifact_sha256": hashes["audit_selection_sha256"],
            "source_replay_sha256": audit.get("source_replay_sha256"),
            "count": len(audit["claims"]),
            "development_only": True,
            "holdout_content_used": False,
        },
        "contract": {
            "prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION,
            "prompt_hash": hashes["prompt_text_sha256"],
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_VERSION,
            "schema_hash": hashes["schema_sha256"],
            "service_parser_hash": hashes["service_parser_sha256"],
            "grounding_validator_hash": hashes["grounding_validator_sha256"],
            "truth_condition_context_version": "TruthConditionContext/current-frozen",
            "truth_condition_context_hash": hashes["truth_condition_context_sha256"],
            "typed_normalizer_versions": [SOURCE_CONTEXT_NORMALIZER_VERSION],
            "typed_normalizer_hash": hashes["typed_normalizer_sha256"],
            "auxiliary_detector_version": "project_source_context_graph/current-frozen",
            "auxiliary_detector_hash": hashes["auxiliary_detector_sha256"],
            "fusion_policy": "NONE",
        },
        "provider_profile": profile,
        "runner_hash": hashes["runner_sha256"],
        "component_hashes": hashes,
        "request_limits": {
            "max_provider_calls": 6,
            "calls_per_claim": 1,
            "retry": 0,
            "regeneration": False,
            "fallback": "none",
            "repair": "none",
            "provider_call_recorder": None,
            "database_writes": False,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "MANIFEST_FROZEN", "path": str(MANIFEST_PATH), "claim_count": len(selected), "runner_hash": hashes["runner_sha256"], "provider_profile": profile}, ensure_ascii=False))


class HardLimitTransport(httpx.AsyncBaseTransport):
    def __init__(self, limit: int = 6) -> None:
        self._inner = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        self.limit = limit
        self.total_sent = 0
        self.current_claim: str | None = None
        self.per_claim_sent: dict[str, int] = {}
        self.blocked_attempts: list[dict[str, str]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        claim_id = self.current_claim or "UNSCOPED"
        sent = self.per_claim_sent.get(claim_id, 0)
        if sent >= 1 or self.total_sent >= self.limit:
            self.blocked_attempts.append({"claim_id": claim_id, "reason": "outbound_limit"})
            raise httpx.RequestError("canary outbound hard limit", request=request)
        self.per_claim_sent[claim_id] = sent + 1
        self.total_sent += 1
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class ObservedGateway:
    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway
        self.last_request_id: str | None = None
        self.safe_error_code: str | None = None
        self.calls = 0

    async def generate_structured(self, **kwargs: Any) -> Any:
        self.calls += 1
        if kwargs.get("allow_regeneration") is not False:
            raise RuntimeError("regeneration_guard_failed")
        try:
            result = await self.gateway.generate_structured(**kwargs)
            self.last_request_id = result.provider_request_id
            return result
        except ModelGatewayError as exc:
            self.safe_error_code = exc.code
            request_id = exc.diagnostics.get("provider_request_id")
            self.last_request_id = request_id if isinstance(request_id, str) else None
            raise


def _family(value: object) -> str:
    return value.value if isinstance(value, ContextType) else str(value)


async def execute(args: argparse.Namespace) -> None:
    audit, selected = load_claims(args.manual_audit)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_profile, material = await resolve_profile()
    current_hashes = component_hashes(args.runner, args.audit_selection, args.manual_audit)
    if manifest["claim_ids"] != list(CLAIM_IDS):
        raise RuntimeError("claim_selection_changed")
    if manifest["provider_profile"] != current_profile:
        raise RuntimeError("provider_profile_changed_after_manifest")
    if manifest["component_hashes"] != current_hashes:
        raise RuntimeError("frozen_component_hash_changed_after_manifest")
    for manifest_row, claim in zip(manifest["claims"], selected, strict=True):
        if manifest_row["claim_id"] != claim["claim_id"] or manifest_row["claim_sha256"] != sha256_bytes(claim["claim_text"].encode("utf-8")):
            raise RuntimeError("claim_identity_hash_mismatch")

    settings = get_settings()
    transport = HardLimitTransport(limit=6)
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=15.0, read=90.0, write=30.0, pool=10.0),
        trust_env=False,
    )
    observed = ObservedGateway(LLMGateway(client, call_recorder=None))
    records: list[dict[str, Any]] = []
    try:
        for claim in selected:
            claim_id = claim["claim_id"]
            transport.current_claim = claim_id
            before_sent = transport.total_sent
            before_calls = observed.calls
            observed.last_request_id = None
            observed.safe_error_code = None
            extractor = SourceContextExtractor(
                observed,
                adapter_type=material.adapter_type,
                base_url=material.base_url,
                api_key=material.api_key,
                model=material.model,
                max_calls=1,
            )
            result = await extractor.extract(claim["claim_text"], claim_ref=claim_id)
            actual_sent = transport.total_sent - before_sent
            gateway_calls = observed.calls - before_calls
            records.append({
                "claim_id": claim_id,
                "claim_text": claim["claim_text"],
                "claim_sha256": result.claim_hash,
                "human_source_contexts_post_call_only": claim.get("manual_source_contexts", []),
                "human_detector_contexts_post_call_only": claim.get("detector_contexts", []),
                "result": result.model_dump(mode="json"),
                "actual_provider_outbound_count": actual_sent,
                "gateway_invocation_count": gateway_calls,
                "provider_request_id": observed.last_request_id,
                "safe_gateway_error_code": observed.safe_error_code,
            })
            if actual_sent > 1 or gateway_calls > 1 or transport.total_sent > 6:
                raise RuntimeError("hard_call_limit_invariant_failed")
    finally:
        await client.aclose()
    output = {
        "experiment_id": manifest["experiment_id"],
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "provider_profile": {k: v for k, v in current_profile.items() if k != "credential_version_id"},
        "frozen_hashes": manifest["component_hashes"],
        "claim_count": len(records),
        "provider_outbound_total": transport.total_sent,
        "gateway_internal_blocked_attempts": transport.blocked_attempts,
        "retry_count": 0,
        "regeneration": False,
        "fallback": "none; any additional send blocked before transport",
        "repair": "none",
        "fusion_policy": "NONE",
        "database_writes": False,
        "holdout_content_used": False,
        "records": records,
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CANARY_COMPLETE", "artifact": str(RESULT_PATH), "claim_count": len(records), "provider_outbound_total": transport.total_sent, "blocked_extra_attempts": len(transport.blocked_attempts)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--audit-selection", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path(__file__))
    args = parser.parse_args()
    try:
        asyncio.run(prepare(args) if args.mode == "prepare" else execute(args))
    except Exception as exc:
        print(json.dumps({"status": "STOPPED", "safe_reason": type(exc).__name__}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
