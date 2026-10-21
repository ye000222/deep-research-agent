"""D4-B isolated hybrid canary orchestration; no production Research wiring."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
from pydantic import SecretStr
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.domain.grounding_requirements import (  # noqa: E402
    GroundingBindingCandidate,
    GroundingNodeInput,
    GroundingNodeKind,
)
from app.domain.providers import AdapterType, CanonicalModelRequest  # noqa: E402
from app.domain.source_context_grounding_v2 import (  # noqa: E402
    SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
    SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
    EvidenceTargetKind,
    GroundingDimensionStatus,
    GroundingStatusV2,
    SemanticEvidenceBinding,
    SourceContextGroundingResultV2,
)
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow  # noqa: E402
from app.infrastructure.db.provider_profiles import ProviderProfileRepository  # noqa: E402
from app.llm.adapters import LLMGateway, ModelGatewayError  # noqa: E402
from app.llm.canary_semantic_pipeline_guard import (  # noqa: E402
    CanarySemanticPipelineGuard,
    SemanticCallPurpose,
)
from app.security.secrets import SecretCipher  # noqa: E402
from app.services.grounding_requirement_engine_v2 import (  # noqa: E402
    GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
    derive_grounding_obligations_v2,
    evaluate_grounding_node_v2,
)
from app.services.hybrid_semantic_support import evaluate_hybrid_semantic_support  # noqa: E402
from app.services.provider_profiles import ProviderProfileService  # noqa: E402
from app.services.source_context_extractor_v2 import (  # noqa: E402
    SourceContextExtractorV2,
)

from scripts.provider_fingerprint_contract import safe_provider_fingerprint  # noqa: E402

ATTEMPT_ID = "D4B_EXECUTION_ATTEMPT_2"
MANIFEST_PATH = ROOT / "artifacts/v2_1r_d4a_selection_manifest.json"
HASH_MANIFEST_PATH = ROOT / "artifacts/v2_1r_d4a_hash_manifest.json"
SEMANTIC_FREEZE_PATH = ROOT / "artifacts/v2_1r_d4a_semantic_freeze.json"
READINESS_PATH = ROOT / "artifacts/v2_1r_d4a1_provider_readiness.json"
POOL_PATH = ROOT / "artifacts/v2_1_dev_extension_v2_candidates.json"
OUTPUT_DIR = ROOT / "artifacts/v2_1r_d4b_execution_attempt_2"
PREFLIGHT_PATH = ROOT / "artifacts/v2_1r_d4b_execution_attempt_2_preflight_final.json"
EXPECTED_MANIFEST_HASH = "05a082ef9ae0483a6fc67e43fc2730dbd2280899ff2378fe5352f18db8ad188a"
EXPECTED_PROVIDER_FINGERPRINT = "32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2"
EXPECTED_CLAIMS = (
    "01a0c82b-3531-79b2-8160-a9e47e4c2bff",
    "01a0c86f-e837-79b4-ab15-62d6e4e84f74",
    "01a0c41c-280e-7fd4-b770-9741bac7b3de",
    "01a0c434-9753-7fff-be68-21d83da29bf2",
    "01a0c424-0fc5-7cea-ac2d-92d3274f5d85",
    "01a0c436-7ecc-7d21-b32d-c5308fa1acec",
    "01a0c853-cc4a-7648-b4e2-d175da55860d",
    "01a0c406-6984-7b9d-a66f-2a6ad7a3a805",
)
FROZEN_COMPONENTS = (
    "apps/api/app/domain/source_context_extraction.py",
    "apps/api/app/services/source_context_extractor_v2.py",
    "apps/api/app/domain/source_context_grounding_v2.py",
    "apps/api/app/domain/semantic_support_verification.py",
    "apps/api/app/services/hybrid_semantic_support.py",
    "apps/api/app/domain/grounding_requirements.py",
    "apps/api/app/services/grounding_requirement_engine.py",
    "apps/api/app/services/grounding_requirement_engine_v2.py",
    "apps/api/app/llm/canary_semantic_pipeline_guard.py",
    "apps/api/app/llm/adapters.py",
    "scripts/run_v2_1r_source_semantic_grounding_v2_canary.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _verify_selection() -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = _read_json(MANIFEST_PATH)
    normalized = dict(manifest)
    normalized.pop("normalized_manifest_sha256", None)
    digest = _sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    ids = tuple(row["claim_id"] for row in manifest.get("claims", []))
    if digest != EXPECTED_MANIFEST_HASH or ids != EXPECTED_CLAIMS:
        raise RuntimeError("FROZEN_SELECTION_MISMATCH")
    if manifest.get("development_only") is not True:
        raise RuntimeError("SELECTION_NOT_DEVELOPMENT_ONLY")
    if _sha256(POOL_PATH.read_bytes()) != manifest.get("source_pool_sha256"):
        raise RuntimeError("SELECTION_SOURCE_POOL_HASH_MISMATCH")
    pool = _read_json(POOL_PATH)
    rows = pool.get("pairs", [])
    by_claim: dict[str, tuple[dict[str, Any], str]] = {}
    for row in rows:
        by_claim[row["claim_a_id"]] = (row, row["claim_a_text"])
        by_claim[row["claim_b_id"]] = (row, row["claim_b_text"])
    claims: list[dict[str, str]] = []
    for frozen in manifest["claims"]:
        pair, text = by_claim.get(frozen["claim_id"], ({}, ""))
        if (
            not text
            or pair.get("pair_id") != frozen.get("source_pair_id")
            or _sha256(text.encode("utf-8")) != frozen.get("claim_sha256")
        ):
            raise RuntimeError("FROZEN_CLAIM_IDENTITY_MISMATCH")
        claims.append(
            {
                "claim_id": frozen["claim_id"],
                "claim_sha256": frozen["claim_sha256"],
                "claim_text": text,
            }
        )
    return manifest, claims


def _verify_frozen_components() -> dict[str, str]:
    manifest = _read_json(HASH_MANIFEST_PATH)
    freeze = _read_json(SEMANTIC_FREEZE_PATH)
    if freeze.get("version_markers", {}).get("grounding_requirement_contract") != (
        GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2
    ):
        raise RuntimeError("GROUNDING_REQUIREMENT_CONTRACT_NOT_V2")
    actual: dict[str, str] = {}
    for path in FROZEN_COMPONENTS:
        expected = freeze.get("components", {}).get(path) or manifest.get("components", {}).get(
            path
        )
        if not expected:
            raise RuntimeError("FROZEN_COMPONENT_HASH_MISSING")
        digest = _sha256((ROOT / path).read_bytes())
        if digest != expected:
            raise RuntimeError("FROZEN_COMPONENT_HASH_MISMATCH")
        actual[path] = digest
    return actual


def _verify_readiness_artifacts() -> dict[str, str]:
    contract = _read_json(ROOT / "artifacts/v2_1r_d4b1_hash_manifest.json")
    verified: dict[str, str] = {}
    for section in (
        "d4a_original_artifacts_preserved",
        "d4b1_artifacts",
        "readiness_code_and_tests",
    ):
        for relative, expected in contract.get(section, {}).items():
            actual = _sha256((ROOT / relative).read_bytes())
            if actual != expected:
                raise RuntimeError("READINESS_ARTIFACT_HASH_MISMATCH")
            verified[relative] = actual
    return verified


def _verify_runtime_import_paths() -> dict[str, str]:
    names = (
        "app.domain.source_context_extraction",
        "app.services.source_context_extractor_v2",
        "app.domain.source_context_grounding_v2",
        "app.domain.semantic_support_verification",
        "app.services.hybrid_semantic_support",
        "app.domain.grounding_requirements",
        "app.services.grounding_requirement_engine",
        "app.services.grounding_requirement_engine_v2",
        "app.llm.canary_semantic_pipeline_guard",
        "app.llm.adapters",
    )
    resolved: dict[str, str] = {}
    root = ROOT.resolve()
    for name in names:
        module = importlib.import_module(name)
        module_file = module.__file__
        if not isinstance(module_file, str):
            raise RuntimeError("RUNTIME_MODULE_PATH_UNAVAILABLE")
        location = Path(module_file).resolve()
        if not location.is_relative_to(root):
            raise RuntimeError("RUNTIME_IMPORTED_NON_WORKSPACE_MODULE")
        resolved[name] = location.relative_to(root).as_posix()
    return resolved


def _resolve_profile_and_material() -> tuple[dict[str, Any], Any]:
    """Use the existing ProviderProfileService path; expose only safe metadata."""
    import base64

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    async def resolve() -> tuple[dict[str, Any], Any]:
        settings = get_settings()
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with sessions() as session:
                found = (
                    await session.execute(
                        select(ProviderProfileRow, CredentialVersionRow)
                        .join(
                            CredentialVersionRow,
                            CredentialVersionRow.profile_id == ProviderProfileRow.id,
                        )
                        .where(
                            ProviderProfileRow.id == UUID("49a9ae5c-b062-4617-805f-dc587b706ce1"),
                            ProviderProfileRow.status == "active",
                            ProviderProfileRow.deleted_at.is_(None),
                            ProviderProfileRow.is_default.is_(True),
                            CredentialVersionRow.revoked_at.is_(None),
                            CredentialVersionRow.deleted_at.is_(None),
                        )
                    )
                ).first()
                if found is None:
                    raise RuntimeError("ACTIVE_PROFILE_NOT_FOUND")
                profile_row, credential_row = found
                safe_profile = {
                    "profile_id": str(profile_row.id),
                    "adapter": profile_row.adapter_type,
                    "model": profile_row.model,
                    "base_url": profile_row.normalized_base_url,
                    "auth_type": profile_row.auth_type,
                    "profile_version": profile_row.version,
                    "credential": {
                        "id": str(credential_row.id),
                        "version": credential_row.credential_version,
                        "present": bool(credential_row.ciphertext and credential_row.nonce),
                    },
                    "non_secret_settings": profile_row.non_secret_settings,
                }
                owner_hash = profile_row.owner_hash
            if (
                settings.secret_master_key_base64 is None
                and not settings.secret_master_key_file.is_file()
            ):
                raise RuntimeError("MASTER_KEY_NOT_CONFIGURED")
            encoded = (
                settings.secret_master_key_base64.get_secret_value().encode("ascii")
                if settings.secret_master_key_base64 is not None
                else settings.secret_master_key_file.read_bytes().strip()
            )
            key = base64.b64decode(encoded, validate=True)
            if len(key) != 32:
                raise RuntimeError("MASTER_KEY_INVALID")
            service = ProviderProfileService(
                ProviderProfileRepository(sessions),
                SecretCipher(key),
                allow_insecure_endpoints=settings.allow_insecure_provider_endpoints,
            )
            material = await service.get_test_material(owner_hash, UUID(safe_profile["profile_id"]))
            if (
                material.adapter_type.value != "openai_compatible_chat"
                or material.model != "deepseek-flash"
                or material.base_url.rstrip("/") != "https://api.deepseek.com/v1"
                or safe_profile["profile_version"] != 23
                or safe_profile["credential"]["version"] != 4
            ):
                raise RuntimeError("PROVIDER_IDENTITY_MISMATCH")
            payload, fingerprint = safe_provider_fingerprint(safe_profile)
            if fingerprint != EXPECTED_PROVIDER_FINGERPRINT:
                raise RuntimeError("PROVIDER_FINGERPRINT_MISMATCH")
            return {
                "profile_id": safe_profile["profile_id"],
                "adapter": safe_profile["adapter"],
                "model": safe_profile["model"],
                "base_url": safe_profile["base_url"],
                "endpoint_host": "api.deepseek.com",
                "profile_version": safe_profile["profile_version"],
                "credential_id": safe_profile["credential"]["id"],
                "credential_version": safe_profile["credential"]["version"],
                "credential_present": safe_profile["credential"]["present"],
                "credential_decryption": "success",
                "safe_config_fingerprint": fingerprint,
                "safe_config_payload": payload,
            }, material
        finally:
            await engine.dispose()

    return asyncio.run(resolve())


class _CountingPostClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.external_http_outbound_count = 0

    async def post(self, url: str, *, headers: Any, json: Any) -> httpx.Response:
        self.external_http_outbound_count += 1
        return await self.client.post(url, headers=headers, json=json)


class _CapturingGateway:
    def __init__(self, gateway: LLMGateway, purpose: SemanticCallPurpose) -> None:
        self.gateway = gateway
        self.purpose = purpose
        self.attempts = 0
        self.structured_output: dict[str, Any] | None = None
        self.usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None}
        self.latency_ms: int | None = None
        self.safe_error: dict[str, str] | None = None

    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> Any:
        if allow_regeneration:
            raise RuntimeError("REGENERATION_MUST_BE_DISABLED")
        if self.purpose is SemanticCallPurpose.EXTRACTOR:
            if (
                request.metadata.get("prompt_version") != SOURCE_CONTEXT_GROUNDING_PROMPT_V2
                or request.metadata.get("schema_version") != SOURCE_CONTEXT_GROUNDING_SCHEMA_V2
                or request.metadata.get("fusion_policy") != "NONE"
            ):
                raise RuntimeError("EXTRACTOR_CONTRACT_ROUTE_MISMATCH")
        elif (
            request.metadata.get("prompt_version") != "semantic-support-verifier.generic.v1"
            or request.metadata.get("schema_version") != "semantic-support-verifier-schema-v1"
            or request.metadata.get("retry") != "0"
            or request.metadata.get("regeneration") != "false"
            or request.metadata.get("fallback") != "none"
            or request.metadata.get("repair") != "none"
        ):
            raise RuntimeError("VERIFIER_CONTRACT_ROUTE_MISMATCH")
        if self.attempts:
            raise RuntimeError("PER_PURPOSE_MODEL_CALL_LIMIT")
        self.attempts += 1
        started = datetime.now(UTC)
        try:
            response = await self.gateway.generate_structured(
                adapter_type=adapter_type,
                base_url=base_url,
                api_key=api_key,
                request=request,
                allow_regeneration=False,
            )
            parsed = getattr(response, "parsed_object", None)
            if not isinstance(parsed, dict):
                text = getattr(response, "text", None)
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                    except (TypeError, ValueError):
                        parsed = None
            if isinstance(parsed, dict):
                self.structured_output = parsed
            self.usage = {
                "input_tokens": _usage_int(response, "input_tokens"),
                "output_tokens": _usage_int(response, "output_tokens"),
            }
            return response
        except ModelGatewayError as exc:
            self.safe_error = {"code": exc.code[:100], "detail_code": (exc.detail_code or "")[:100]}
            raise
        except Exception as exc:
            self.safe_error = {"code": type(exc).__name__[:100], "detail_code": "UNCLASSIFIED"}
            raise
        finally:
            self.latency_ms = max(0, int((datetime.now(UTC) - started).total_seconds() * 1000))


def _usage_int(response: Any, field: str) -> int | None:
    usage = getattr(response, "usage", None)
    value = getattr(usage, field, None) if usage is not None else None
    return value if isinstance(value, int) and value >= 0 else None


def _node_inputs(result: SourceContextGroundingResultV2) -> list[GroundingNodeInput]:
    span_map = {span.evidence_ref: span.exact_text for span in result.evidence_spans}
    nodes: list[GroundingNodeInput] = []
    for assertion in result.source_assertions:
        nodes.append(
            GroundingNodeInput(
                node_ref=assertion.source_assertion_ref,
                node_kind=GroundingNodeKind.ASSERTION,
                semantic_payload={
                    "assertion_content": [
                        span_map[x] for x in assertion.evidence_refs if x in span_map
                    ],
                    "scope_ref": assertion.source_scope_ref,
                },
                graph_role="SOURCE_ASSERTION",
            )
        )
    for context in result.source_contexts:
        nodes.append(
            GroundingNodeInput(
                node_ref=context.context_ref,
                node_kind=GroundingNodeKind.CONTEXT,
                semantic_payload={
                    "context_type": context.context_type.value,
                    "semantic_value": context.semantic_value,
                },
                graph_role="SOURCE_CONTEXT",
            )
        )
    for attachment in result.source_attachments:
        nodes.append(
            GroundingNodeInput(
                node_ref=attachment.attachment_ref,
                node_kind=GroundingNodeKind.ATTACHMENT,
                semantic_payload={
                    "context_ref": attachment.context_ref,
                    "relation": attachment.relation.value,
                    "target_refs": list(attachment.source_assertion_refs),
                    "source_scope_ref": attachment.source_scope_ref,
                },
                graph_role="SOURCE_ATTACHMENT",
            )
        )
    for scope in result.source_scopes:
        nodes.append(
            GroundingNodeInput(
                node_ref=scope.source_scope_ref,
                node_kind=GroundingNodeKind.SCOPE,
                semantic_payload={
                    "scope_value": scope.kind.value,
                    "member_refs": list(scope.source_assertion_refs),
                },
                graph_role="SOURCE_SCOPE",
            )
        )
    return nodes


def _target_payload(
    result: SourceContextGroundingResultV2, binding: SemanticEvidenceBinding
) -> dict[str, Any]:
    if binding.target_kind is EvidenceTargetKind.ASSERTION:
        assertion = next(
            (x for x in result.source_assertions if x.source_assertion_ref == binding.target_ref),
            None,
        )
        spans = {x.evidence_ref: x.exact_text for x in result.evidence_spans}
        return (
            {
                "assertion_content": [spans[x] for x in assertion.evidence_refs if x in spans],
                "scope_ref": assertion.source_scope_ref,
            }
            if assertion
            else {}
        )
    if binding.target_kind is EvidenceTargetKind.CONTEXT:
        context = next(
            (x for x in result.source_contexts if x.context_ref == binding.target_ref), None
        )
        return (
            {
                "context_type": context.context_type.value,
                "semantic_value": context.semantic_value,
                "lexical_value": context.lexical_value,
            }
            if context
            else {}
        )
    if binding.target_kind is EvidenceTargetKind.ATTACHMENT:
        edge = next(
            (x for x in result.source_attachments if x.attachment_ref == binding.target_ref), None
        )
        return (
            {
                "attachment_ref": edge.attachment_ref,
                "context_ref": edge.context_ref,
                "relation": edge.relation.value,
                "source_assertion_refs": list(edge.source_assertion_refs),
                "status": edge.status,
                "source_scope_ref": edge.source_scope_ref,
            }
            if edge
            else {}
        )
    scope = next(
        (x for x in result.source_scopes if x.source_scope_ref == binding.target_ref), None
    )
    return (
        {
            "scope_value": scope.kind.value,
            "member_refs": list(scope.source_assertion_refs),
        }
        if scope
        else {}
    )


def _source_dimension(
    result: SourceContextGroundingResultV2, kind: EvidenceTargetKind, ref: str, dimension: str
) -> GroundingDimensionStatus:
    assessment = next(
        (x for x in result.node_assessments if x.target_kind is kind and x.target_ref == ref), None
    )
    return (
        getattr(assessment.dimensions, dimension)
        if assessment
        else GroundingDimensionStatus.NOT_EVALUATED
    )


def _build_requirement_results(
    result: SourceContextGroundingResultV2,
    hybrid: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hybrid_by_ref = {x.binding_ref: x for x in hybrid.bindings}
    candidates: list[GroundingBindingCandidate] = []
    candidate_edges: dict[str, Any] = {}
    for binding in result.evidence_bindings:
        support = hybrid_by_ref.get(binding.binding_ref)
        edge = (
            next(
                (x for x in result.source_attachments if x.attachment_ref == binding.target_ref),
                None,
            )
            if binding.target_kind is EvidenceTargetKind.ATTACHMENT
            else None
        )
        relation = edge.relation.value if edge else None
        candidate = GroundingBindingCandidate(
            binding_ref=binding.binding_ref,
            evidence_ref=binding.evidence_ref,
            semantic_role=binding.semantic_role.value,
            target_kind=binding.target_kind.value,
            target_ref=binding.target_ref,
            canonical_payload=_target_payload(result, binding),
            relation=relation,
            scope_ref=edge.source_scope_ref if edge else None,
            textual_grounding=binding.dimensions.textual_grounding,
            structural_grounding=binding.dimensions.structural_grounding,
            semantic_support=(
                support.semantic_support if support else GroundingDimensionStatus.AMBIGUOUS
            ),
        )
        candidates.append(candidate)
        candidate_edges[binding.binding_ref] = edge

    node_results: list[dict[str, Any]] = []
    obligation_rows: list[dict[str, Any]] = []
    for node in _node_inputs(result):
        derivation = derive_grounding_obligations_v2(node)
        kind = {
            GroundingNodeKind.ASSERTION: EvidenceTargetKind.ASSERTION,
            GroundingNodeKind.CONTEXT: EvidenceTargetKind.CONTEXT,
            GroundingNodeKind.ATTACHMENT: EvidenceTargetKind.ATTACHMENT,
            GroundingNodeKind.SCOPE: EvidenceTargetKind.SCOPE,
        }[node.node_kind]
        textual = _source_dimension(result, kind, node.node_ref, "textual_grounding")
        structural = _source_dimension(result, kind, node.node_ref, "structural_grounding")
        target_overrides: dict[tuple[str, str], GroundingDimensionStatus] = {}
        multi_target_refs = {
            x.attachment_ref for x in result.source_attachments if len(x.source_assertion_refs) > 1
        }
        for candidate in candidates:
            edge = candidate_edges[candidate.binding_ref]
            if edge and edge.attachment_ref in multi_target_refs:
                for target in edge.source_assertion_refs:
                    target_overrides[(candidate.binding_ref, target)] = (
                        GroundingDimensionStatus.AMBIGUOUS
                    )
        assessment = evaluate_grounding_node_v2(
            derivation,
            tuple(candidates),
            node_textual_grounding=textual,
            node_structural_grounding=structural,
            target_support_overrides=target_overrides,
        )
        node_results.append(assessment.model_dump(mode="json"))
        for row in assessment.obligation_evaluations:
            obligation_rows.append(
                {
                    "node_ref": node.node_ref,
                    "obligation_ref": row.obligation.obligation_ref,
                    "obligation_type": row.obligation.obligation_type.value,
                    "requirement_level": row.obligation.requirement_level.value,
                    "status": row.status.value,
                    "matched_binding_refs": list(row.match.matched_binding_refs),
                    "excluded_candidates": [
                        x.model_dump(mode="json") for x in row.match.excluded_candidates
                    ],
                    "provenance": row.provenance,
                }
            )
    return node_results, obligation_rows


def _safety_metrics(
    result: SourceContextGroundingResultV2,
    node_results: list[dict[str, Any]],
    obligations: list[dict[str, Any]],
    hybrid: Any,
) -> dict[str, int]:
    invalid_spans = sum(
        x.textual_grounding_status is GroundingDimensionStatus.INVALID
        for x in result.evidence_spans
    )
    usable_nodes = [x for x in node_results if x["usability"] == "GROUNDED_USABLE"]
    invalid_binding_refs = {
        x.binding_ref
        for x in result.evidence_bindings
        if x.dimensions.textual_grounding is GroundingDimensionStatus.INVALID
        or x.dimensions.structural_grounding is GroundingDimensionStatus.INVALID
    }
    supported = {
        x.binding_ref
        for x in hybrid.bindings
        if x.semantic_support is GroundingDimensionStatus.VALID
    }
    bad_support_refs = {
        x.binding_ref
        for x in hybrid.bindings
        if x.semantic_support
        in {
            GroundingDimensionStatus.INVALID,
            GroundingDimensionStatus.MISSING_EVIDENCE,
            GroundingDimensionStatus.AMBIGUOUS,
        }
    }
    usable_by_node = {x["node_ref"] for x in usable_nodes}
    failed_required_rows = [
        row
        for row in obligations
        if row["requirement_level"] == "REQUIRED_FOR_NODE_SEMANTICS"
        and row["status"] != "SATISFIED"
    ]
    verifier_failure_refs = {
        item.binding_ref
        for item in hybrid.bindings
        if item.router_decision.route.value == "REQUIRES_SEMANTIC_INFERENCE"
        and hybrid.verifier.failure_status.value != "NONE"
    }
    return {
        "fabricated_source_span_count": invalid_spans,
        "textual_invalid_promoted_count": sum(
            x.evidence_ref
            in {
                s.evidence_ref
                for s in result.evidence_spans
                if s.textual_grounding_status is GroundingDimensionStatus.INVALID
            }
            and x.binding_ref in supported
            for x in result.evidence_bindings
        ),
        "structural_invalid_promoted_count": sum(
            x.dimensions.structural_grounding is GroundingDimensionStatus.INVALID
            and x.binding_ref in supported
            for x in result.evidence_bindings
        ),
        "unsupported_semantic_promoted_count": sum(
            row["node_ref"] in usable_by_node
            and row["requirement_level"] == "REQUIRED_FOR_NODE_SEMANTICS"
            and row["status"] == "SATISFIED"
            and bool(bad_support_refs.intersection(row["matched_binding_refs"]))
            for row in obligations
        ),
        "missing_required_binding_promoted_count": sum(
            row["node_ref"] in usable_by_node for row in failed_required_rows
        ),
        "unsafe_node_promotion_count": sum(
            bool(
                invalid_binding_refs.intersection(
                    {
                        ref
                        for row in obligations
                        if row["node_ref"] == n["node_ref"]
                        for ref in row["matched_binding_refs"]
                    }
                )
            )
            for n in usable_nodes
        ),
        "unsafe_failure_masking_count": sum(
            row["node_ref"] in usable_by_node
            and bool(verifier_failure_refs.intersection(row["matched_binding_refs"]))
            for row in obligations
        )
        + int(
            result.status in {GroundingStatusV2.INVALID, GroundingStatusV2.NOT_EVALUATED}
            and any(
                item["usability"] == "GROUNDED_USABLE"
                and any(
                    binding.target_ref == item["node_ref"]
                    and (
                        binding.dimensions.textual_grounding is GroundingDimensionStatus.INVALID
                        or binding.dimensions.structural_grounding
                        is GroundingDimensionStatus.INVALID
                    )
                    for binding in result.evidence_bindings
                )
                for item in usable_nodes
            )
        ),
        "unsafe_vacuous_promotion_count": sum(
            n["usability"] == "GROUNDED_USABLE"
            and not any(row["node_ref"] == n["node_ref"] for row in obligations)
            for n in node_results
        ),
        "undeclared_semantic_reuse_count": sum(
            1
            for row in obligations
            for item in row["excluded_candidates"]
            if item["reason"] == "BINDING_ALREADY_CONSUMED"
        ),
        "provenance_defect_count": sum(
            1
            for row in obligations
            if row["provenance"].get("contract_version")
            != GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2
            or not row["provenance"].get("matcher_version")
        ),
    }


async def _execute_claim(
    claim: dict[str, str],
    material: Any,
    http_client: httpx.AsyncClient,
    guard: CanarySemanticPipelineGuard,
) -> dict[str, Any]:
    claim_id = claim["claim_id"]
    extractor_client = _CountingPostClient(http_client)
    extractor_capture = _CapturingGateway(
        LLMGateway(
            guard.client_for(  # type: ignore[arg-type]
                claim_id, SemanticCallPurpose.EXTRACTOR, extractor_client
            )
        ),
        SemanticCallPurpose.EXTRACTOR,
    )
    extractor = SourceContextExtractorV2(
        extractor_capture,
        adapter_type=material.adapter_type,
        base_url=material.base_url,
        api_key=material.api_key,
        model=material.model,
        max_calls=1,
    )
    extraction = await extractor.extract(claim["claim_text"], claim_ref=claim_id)
    verifier_client = _CountingPostClient(http_client)
    verifier_capture = _CapturingGateway(
        LLMGateway(
            guard.client_for(  # type: ignore[arg-type]
                claim_id, SemanticCallPurpose.SEMANTIC_VERIFIER, verifier_client
            )
        ),
        SemanticCallPurpose.SEMANTIC_VERIFIER,
    )
    hybrid = await evaluate_hybrid_semantic_support(
        extraction,
        claim["claim_text"],
        gateway=verifier_capture,
        adapter_type=material.adapter_type,
        base_url=material.base_url,
        api_key=material.api_key,
        model=material.model,
        provider_identity="49a9ae5c-b062-4617-805f-dc587b706ce1",
        model_identity=material.model,
        materiality=(),
    )
    nodes, obligations = _build_requirement_results(extraction, hybrid)
    guard_counts = guard.snapshot(claim_id)
    outbound = (
        extractor_client.external_http_outbound_count + verifier_client.external_http_outbound_count
    )
    if outbound > 2 or outbound != guard_counts["total_call_count"]:
        raise RuntimeError("OUTBOUND_ACCOUNTING_MISMATCH")
    if guard_counts["extractor_call_count"] > 1 or guard_counts["verifier_call_count"] > 1:
        raise RuntimeError("PER_PURPOSE_OUTBOUND_LIMIT_VIOLATION")
    metrics = _safety_metrics(extraction, nodes, obligations, hybrid)
    total_tokens = (
        extraction.input_tokens + extraction.output_tokens + (hybrid.verifier.total_tokens or 0)
    )
    textual_distribution = Counter(
        binding.dimensions.textual_grounding.value for binding in extraction.evidence_bindings
    )
    structural_distribution = Counter(
        binding.dimensions.structural_grounding.value for binding in extraction.evidence_bindings
    )
    obligation_distribution = Counter(row["status"] for row in obligations)
    router_distribution = Counter(item.router_decision.route.value for item in hybrid.bindings)
    verdict_distribution = Counter(
        item.verifier_verdict.value for item in hybrid.bindings if item.verifier_verdict is not None
    )
    support_distribution = Counter(item.semantic_support.value for item in hybrid.bindings)
    missing_required_count = sum(
        row["status"] == "MISSING_REQUIRED_BINDING"
        and row["requirement_level"] == "REQUIRED_FOR_NODE_SEMANTICS"
        for row in obligations
    )
    raw_structured = {
        "extractor": extractor_capture.structured_output,
        "semantic_verifier": verifier_capture.structured_output,
    }
    return {
        "claim_id": claim_id,
        "claim_sha256": claim["claim_sha256"],
        "claim_text": claim["claim_text"],
        "raw_structured_model_outputs": raw_structured,
        "extractor": {
            "structured_status": extractor_capture.structured_output.get("status")
            if extractor_capture.structured_output
            else "NO_STRUCTURED_OUTPUT",
            "grounding_status": extraction.status.value,
            "diagnostics": list(extraction.diagnostics),
            "unresolved_items": list(extraction.unresolved_items),
            "source_assertion_count": len(extraction.source_assertions),
            "source_context_count": len(extraction.source_contexts),
            "source_attachment_count": len(extraction.source_attachments),
            "source_scope_count": len(extraction.source_scopes),
            "evidence_spans": [x.model_dump(mode="json") for x in extraction.evidence_spans],
            "semantic_evidence_bindings": [
                x.model_dump(mode="json") for x in extraction.evidence_bindings
            ],
            "grounding_result": extraction.model_dump(mode="json"),
            "safe_error": extractor_capture.safe_error,
        },
        "source_graph_counts": {
            "source_evidence_span_count": len(extraction.evidence_spans),
            "semantic_evidence_binding_count": len(extraction.evidence_bindings),
        },
        "grounding": {
            "textual_distribution": dict(textual_distribution),
            "structural_distribution": dict(structural_distribution),
        },
        "requirements": {
            "obligation_distribution": dict(obligation_distribution),
            "missing_required_binding_count": missing_required_count,
        },
        "hybrid_support": {
            **hybrid.model_dump(mode="json"),
            "router_distribution": dict(router_distribution),
            "verifier_requested_binding_refs": list(hybrid.verifier.input_binding_refs),
            "verifier_verdict_distribution": dict(verdict_distribution),
            "final_semantic_support_distribution": dict(support_distribution),
        },
        "grounding_requirement_contract": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
        "node_assessments": nodes,
        "obligation_evaluations": obligations,
        "node_usability": dict(Counter(item["usability"] for item in nodes)),
        "verification_assertion_readiness_count": sum(
            bool(x["verification_assertion_ready"]) for x in nodes
        ),
        "extraction_omissions": [
            *extraction.unresolved_items,
            *[
                f"{row['node_ref']}:{row['obligation_ref']}:{row['status']}"
                for row in obligations
                if row["status"] == "MISSING_REQUIRED_BINDING"
            ],
        ],
        "safety_metrics": metrics,
        "provider": {
            "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
            "adapter": material.adapter_type.value,
            "model": material.model,
            "endpoint_host": "api.deepseek.com",
        },
        "components": {
            "prompt_version": extraction.prompt_version,
            "schema_version": extraction.schema_version,
            "grounding_requirement_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
            "semantic_support_version": hybrid.verifier.contract_version,
            "component_sha256": _verify_frozen_components(),
        },
        "calls": {
            **guard_counts,
            "external_http_outbound_count": outbound,
            "blocked_call_count": guard_counts["blocked_call_count"],
            "input_tokens": extraction.input_tokens + (hybrid.verifier.input_tokens or 0),
            "output_tokens": extraction.output_tokens + (hybrid.verifier.output_tokens or 0),
            "total_tokens": total_tokens,
            "extractor_latency_ms": extraction.latency_ms,
            "verifier_latency_ms": hybrid.verifier.latency_ms,
            "extractor_safe_error": extractor_capture.safe_error,
            "verifier_safe_error": verifier_capture.safe_error,
        },
    }


def _aggregate(
    claim_results: list[dict[str, Any]], identity: dict[str, Any], hashes: dict[str, str]
) -> dict[str, Any]:
    metric_names = (
        "fabricated_source_span_count",
        "textual_invalid_promoted_count",
        "structural_invalid_promoted_count",
        "unsupported_semantic_promoted_count",
        "missing_required_binding_promoted_count",
        "unsafe_node_promotion_count",
        "unsafe_failure_masking_count",
        "unsafe_vacuous_promotion_count",
        "undeclared_semantic_reuse_count",
        "provenance_defect_count",
    )
    return {
        "artifact_version": "v2_1r_d4b_hybrid_canary_attempt.v1",
        "attempt_id": ATTEMPT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_manifest_normalized_sha256": EXPECTED_MANIFEST_HASH,
        "provider_identity": identity,
        "component_sha256": hashes,
        "grounding_requirement_contract": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
        "claims": claim_results,
        "totals": {
            "claim_count": len(claim_results),
            "extractor_call_count": sum(x["calls"]["extractor_call_count"] for x in claim_results),
            "verifier_call_count": sum(x["calls"]["verifier_call_count"] for x in claim_results),
            "total_call_count": sum(x["calls"]["total_call_count"] for x in claim_results),
            "blocked_call_count": sum(x["calls"]["blocked_call_count"] for x in claim_results),
            "external_http_outbound_count": sum(
                x["calls"]["external_http_outbound_count"] for x in claim_results
            ),
            "input_tokens": sum(x["calls"]["input_tokens"] for x in claim_results),
            "output_tokens": sum(x["calls"]["output_tokens"] for x in claim_results),
            "total_tokens": sum(x["calls"]["total_tokens"] for x in claim_results),
            "latency_ms_sum": sum(
                x["calls"]["extractor_latency_ms"] + (x["calls"]["verifier_latency_ms"] or 0)
                for x in claim_results
            ),
            "safety_metrics": {
                name: sum(x["safety_metrics"][name] for x in claim_results) for name in metric_names
            },
        },
        "holdout_accessed": False,
        "database_writes": 0,
    }


def run_preflight() -> dict[str, Any]:
    manifest, claims = _verify_selection()
    component_hashes = _verify_frozen_components()
    readiness_artifact_hashes = _verify_readiness_artifacts()
    runtime_imports = _verify_runtime_import_paths()
    readiness = _read_json(READINESS_PATH)
    if (
        readiness.get("fingerprint_contract_version") != "provider-safe-config-fingerprint.v1"
        or readiness.get("safe_config_fingerprint") != EXPECTED_PROVIDER_FINGERPRINT
        or readiness.get("authorization", {}).get("d4b_execution_authorized_by_this_artifact")
        is not False
    ):
        raise RuntimeError("D4A1_READINESS_ARTIFACT_MISMATCH")
    if GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2 != "grounding-requirements.generic.v2":
        raise RuntimeError("CURRENT_GROUNDING_REQUIREMENT_IMPORT_MISMATCH")
    return {
        "attempt_id": ATTEMPT_ID,
        "selection_manifest_normalized_sha256": EXPECTED_MANIFEST_HASH,
        "claim_ids": [x["claim_id"] for x in claims],
        "claim_hashes_verified": True,
        "selection_development_only": manifest["development_only"],
        "provider_safe_fingerprint_expected": EXPECTED_PROVIDER_FINGERPRINT,
        "provider_calls": 0,
        "external_http_outbound": 0,
        "database_writes": 0,
        "holdout_accessed": False,
        "grounding_requirement_contract": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
        "frozen_component_hashes": component_hashes,
        "readiness_artifact_hashes": readiness_artifact_hashes,
        "runtime_imports": runtime_imports,
        "strict_guard_class": "CanarySemanticPipelineGuard",
        "orchestrator_sha256": _sha256(Path(__file__).read_bytes()),
        "fake_integration_test_sha256": _sha256(
            (ROOT / "tests/unit/test_v2_1r_d4b_hybrid_canary_runner.py").read_bytes()
        ),
        "execution_ready_after_runtime_identity": True,
    }


async def execute_all(
    claims: list[dict[str, str]], material: Any, identity: dict[str, Any], hashes: dict[str, str]
) -> dict[str, Any]:
    guard = CanarySemanticPipelineGuard()
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        for claim in claims:
            row = await _execute_claim(claim, material, client, guard)
            rows.append(row)
            _write_json(OUTPUT_DIR / f"{claim['claim_id']}.json", row)
    return _aggregate(rows, identity, hashes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-frozen-authorized-cohort", action="store_true")
    args = parser.parse_args()
    try:
        preflight = run_preflight()
        if args.preflight:
            if PREFLIGHT_PATH.exists():
                raise RuntimeError("PREFLIGHT_ARTIFACT_ALREADY_EXISTS")
            _write_json(PREFLIGHT_PATH, preflight)
            print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
            return 0
        if not args.execute_frozen_authorized_cohort:
            print("EXECUTION_FLAG_REQUIRED provider_calls=0")
            return 2
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError("ATTEMPT_ARTIFACT_ALREADY_EXISTS")
        frozen_preflight = _read_json(PREFLIGHT_PATH)
        if frozen_preflight != preflight:
            raise RuntimeError("PREFLIGHT_CHANGED_BEFORE_EXECUTION")
        _, claims = _verify_selection()
        identity, material = _resolve_profile_and_material()
        hashes = _verify_frozen_components()
        if identity["safe_config_fingerprint"] != EXPECTED_PROVIDER_FINGERPRINT:
            raise RuntimeError("PROVIDER_FINGERPRINT_MISMATCH")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(OUTPUT_DIR / "preflight.json", {**preflight, "provider_identity": identity})
        result = asyncio.run(execute_all(claims, material, identity, hashes))
        _write_json(OUTPUT_DIR / "aggregate.json", result)
        print(
            json.dumps(
                {
                    "attempt_id": ATTEMPT_ID,
                    "claims": result["totals"]["claim_count"],
                    "provider_calls": result["totals"]["total_call_count"],
                    "external_http_outbound": result["totals"]["external_http_outbound_count"],
                    "artifact_dir": "artifacts/v2_1r_d4b_execution_attempt_2",
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_OR_FAILED_CLOSED",
                    "safe_error_code": type(exc).__name__,
                    "safe_detail_code": str(exc)[:120]
                    if str(exc).isupper() and len(str(exc)) < 121
                    else "UNCLASSIFIED",
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
