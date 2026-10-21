"""Execute the explicitly authorized D4-J-B v3.1 development-only Canary.

This orchestration script does not change extraction semantics. It verifies all
frozen identities before the first call, then invokes the frozen one-Claim
runner sequentially with a process-wide hard cap and persists each result before
any human review. Never rerun this script after any outbound may have occurred.
"""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select

ROOT = Path("/app")
FROZEN_ARTIFACTS = Path("/app/frozen-artifacts")
OUT = Path("/app/d4j-output")
sys.path.insert(0, "/app/apps/api")
sys.path.insert(0, "/app/scripts")

from app.core.config import get_settings  # noqa: E402
from app.domain.providers import AdapterType  # noqa: E402
from app.domain.source_context_extraction_v3_1 import (  # noqa: E402
    SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31,
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
)
from app.infrastructure.db.models import ProviderProfileRow  # noqa: E402
from app.infrastructure.runtime import ApplicationRuntime  # noqa: E402
from app.llm.adapters import LLMGateway  # noqa: E402
from app.llm.canary_strict import SingleOutboundClient  # noqa: E402
from app.services.source_context_extractor_v3_1 import SourceContextExtractorV31  # noqa: E402
from provider_fingerprint_contract import safe_provider_fingerprint  # noqa: E402
from v2_1r_d4j_a_canary_runner import execute_one_claim_v31  # noqa: E402

MANIFEST_SHA256 = "847f65be7dd2dddb6b4143c2f2d874779ff089ab62c5877b228b0a75ee6f1c74"
IMPLEMENTATION_FINGERPRINT = "a554d3d8264e41cb09376fb98b9a61ec86e1cadf21bef05ed032909ba7b516a2"
PROVIDER_FINGERPRINT = "32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2"
PROFILE_ID = UUID("49a9ae5c-b062-4617-805f-dc587b706ce1")
CLAIM_IDS = (
    "01a0c41c-a7d9-7ef5-bbde-b70d8d0787cd",
    "01a0c41c-2835-7208-bd74-5e9ddebd8963",
    "01a0c43b-3e5a-7801-8bda-b8ade1fd7eb8",
    "01a0c405-03ed-79fd-9758-15029694fc0c",
    "01a0c834-62f9-7c30-8813-20d24892e2f4",
    "01a0c836-0c24-7a71-a449-69b63110a68f",
    "01a0c82b-c360-7215-9e7e-f083d3fc91d5",
    "01a0c406-6970-7f63-be80-943c2a5822db",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    os.replace(temp, path)


def safe_profile_payload(profile: Any, non_secret_settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": str(profile.profile_id),
        "adapter": profile.adapter_type.value,
        "model": profile.model,
        "base_url": profile.base_url,
        "auth_type": "bearer",
        "profile_version": profile.version,
        "credential": {"id": str(profile.credential_version_id), "version": profile.credential_version, "present": True},
        "non_secret_settings": non_secret_settings,
    }


async def resolve_runtime_identity() -> tuple[dict[str, Any], Any, Any]:
    settings = get_settings()
    runtime = ApplicationRuntime.build(settings)
    try:
        if runtime.profile_service is None:
            raise RuntimeError("provider_profile_service_unavailable")
        async with runtime.business_db.session_factory() as session:
            locator = await session.scalar(
                select(ProviderProfileRow).where(
                    ProviderProfileRow.id == PROFILE_ID,
                    ProviderProfileRow.deleted_at.is_(None),
                )
            )
            if locator is None:
                raise RuntimeError("provider_profile_not_found")
            owner_hash = locator.owner_hash
            settings_payload = dict(locator.non_secret_settings or {})
        views = await runtime.profile_service.list_profiles(owner_hash)
        matches = [view for view in views if view.profile_id == PROFILE_ID]
        if len(matches) != 1:
            raise RuntimeError("provider_profile_resolution_count_mismatch")
        view = matches[0]
        if view.status != "active" or view.is_default is not True:
            raise RuntimeError("provider_profile_not_active_default")
        payload = safe_profile_payload(view, settings_payload)
        canonical_payload, fingerprint = safe_provider_fingerprint(payload)
        if fingerprint != PROVIDER_FINGERPRINT:
            raise RuntimeError("provider_fingerprint_mismatch")
        if (
            view.adapter_type is not AdapterType.OPENAI_COMPATIBLE_CHAT
            or view.model != "deepseek-flash"
            or view.base_url.rstrip("/") != "https://api.deepseek.com/v1"
            or view.version != 23
            or view.credential_version != 4
        ):
            raise RuntimeError("provider_identity_mismatch")
        material = await runtime.profile_service.get_test_material(owner_hash, PROFILE_ID)
        if (
            material.profile_id != PROFILE_ID
            or material.adapter_type is not AdapterType.OPENAI_COMPATIBLE_CHAT
            or material.model != "deepseek-flash"
            or material.base_url.rstrip("/") != "https://api.deepseek.com/v1"
            or material.credential_version_id != view.credential_version_id
        ):
            raise RuntimeError("decrypted_provider_material_identity_mismatch")
        identity = {
            "runtime_kind": "ephemeral Compose API service with current source bind mount",
            "source_revision": settings.source_revision,
            "profile_id": str(view.profile_id),
            "profile_status": view.status,
            "profile_is_default": view.is_default,
            "adapter": view.adapter_type.value,
            "model": view.model,
            "endpoint_family": "https://api.deepseek.com/v1",
            "profile_version": view.version,
            "credential_present": True,
            "credential_version": view.credential_version,
            "safe_fingerprint": fingerprint,
            "fingerprint_contract": "provider-safe-config-fingerprint.v1",
            "owner_hash_emitted": False,
            "canonical_safe_payload": canonical_payload,
        }
        return identity, material, runtime
    except BaseException:
        await runtime.close()
        raise


def verify_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    manifest_path = FROZEN_ARTIFACTS / "v2_1r_d4j_a_selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = dict(manifest)
    normalized.pop("normalized_manifest_sha256", None)
    manifest_hash = sha_bytes(canonical(normalized))
    if manifest_hash != MANIFEST_SHA256 or manifest.get("normalized_manifest_sha256") != MANIFEST_SHA256:
        raise RuntimeError("selection_manifest_hash_mismatch")
    if tuple(row["claim_id"] for row in manifest["claims"]) != CLAIM_IDS:
        raise RuntimeError("selection_order_mismatch")
    if len(manifest["claims"]) != 8 or any(row.get("development_only_forever") is not True for row in manifest["claims"]):
        raise RuntimeError("development_cohort_contract_mismatch")
    pool_path = FROZEN_ARTIFACTS / "v2_1_dev_extension_v2_candidates.json"
    if sha_file(pool_path) != manifest["candidate_source_sha256"]:
        raise RuntimeError("candidate_pool_hash_mismatch")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    pair_rows = pool.get("pairs", [])
    selected: list[dict[str, Any]] = []
    for row in manifest["claims"]:
        match = next(
            (
                (candidate[f"claim_{side}_text"], candidate["pair_id"])
                for candidate in pair_rows
                if candidate.get("pair_id") == row["source_pair_id"]
                for side in ("a", "b")
                if candidate.get(f"claim_{side}_id") == row["claim_id"]
            ),
            None,
        )
        if match is None:
            raise RuntimeError("authorized_claim_missing_from_frozen_development_pool")
        text, pair_id = match
        if not text.strip() or pair_id != row["source_pair_id"] or sha_bytes(text.encode("utf-8")) != row["claim_sha256"]:
            raise RuntimeError("authorized_claim_identity_mismatch")
        selected.append({**row, "claim_text": text})

    freeze_path = FROZEN_ARTIFACTS / "v2_1r_d4j_a" / "implementation_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    def frozen_component_path(relative: str) -> Path:
        if relative.startswith("artifacts/"):
            return FROZEN_ARTIFACTS / Path(*Path(relative).parts[1:])
        return ROOT / relative

    actual = {path: sha_file(frozen_component_path(path)) for path in freeze["component_hashes"]}
    implementation_fingerprint = sha_bytes(canonical(actual))
    if actual != freeze["component_hashes"] or implementation_fingerprint != IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError("implementation_fingerprint_mismatch")
    readiness = json.loads((FROZEN_ARTIFACTS / "v2_1r_d4j_a" / "readiness_status.json").read_text(encoding="utf-8"))
    if not all(value == "PASS" for value in readiness["gates"].values()):
        raise RuntimeError("readiness_gate_not_pass")
    plan = json.loads((FROZEN_ARTIFACTS / "v2_1r_d4j_a" / "execution_plan.json").read_text(encoding="utf-8"))
    if (
        plan["exact_claim_ids_in_order"] != list(CLAIM_IDS)
        or plan["selection_manifest_sha256"] != MANIFEST_SHA256
        or plan["implementation_fingerprint"] != IMPLEMENTATION_FINGERPRINT
        or plan["provider_fingerprint"] != PROVIDER_FINGERPRINT
        or plan["provider_calls_max_total"] != 8
        or plan["provider_generation_max_per_claim"] != 1
        or plan["semantic_verifier_calls"] != 0
        or plan["retry"] != 0
        or plan["regeneration"] is not False
        or plan["repair"] != "none"
        or plan["provider_fallback"] != "none"
        or plan["endpoint_fallback"] != "none"
        or plan["request_format_fallback"] != "none"
    ):
        raise RuntimeError("frozen_execution_plan_mismatch")
    runtime_module = Path(SourceContextExtractorV31.__module__.replace(".", "/") + ".py").resolve()
    actual_module = Path(sys.modules[SourceContextExtractorV31.__module__].__file__).resolve()
    if (
        actual_module != Path("/app/apps/api/app/services/source_context_extractor_v3_1.py")
        or SOURCE_CONTEXT_EXTRACTION_PROMPT_V31 != "source-context-extraction.generic.v3.1"
        or SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31 != "source-context-extraction-schema-v3.1"
        or SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31 != "source-context-extraction-domain-v3.1"
        or SingleOutboundClient.__module__ != "app.llm.canary_strict"
        or LLMGateway.__module__ != "app.llm.adapters"
        or runtime_module.name != actual_module.name
    ):
        raise RuntimeError("canonical_runtime_import_mismatch")
    return manifest, selected, actual


class HardLimitPostClient:
    """Process-wide max-eight transport boundary; request contents are never saved."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.total_outbound = 0
        self.per_claim_outbound: dict[str, int] = {}
        self.current_claim: str | None = None
        self.captures: list[dict[str, Any]] = []

    async def post(self, url: str, *, headers: Any, json: Any) -> httpx.Response:
        claim_id = self.current_claim
        if claim_id not in CLAIM_IDS:
            raise RuntimeError("outbound_without_authorized_claim_context")
        if self.total_outbound >= 8 or self.per_claim_outbound.get(claim_id, 0) >= 1:
            raise RuntimeError("global_or_per_claim_outbound_limit_exceeded")
        self.total_outbound += 1
        self.per_claim_outbound[claim_id] = self.per_claim_outbound.get(claim_id, 0) + 1
        started = datetime.now(UTC)
        try:
            response = await self.client.post(url, headers=headers, json=json)
        except Exception as exc:
            # Keep only exception type; never retain request headers/body or error text.
            self.captures.append({
                "claim_id": claim_id,
                "attempted_at_utc": started.isoformat(),
                "http_status": None,
                "exception_category": type(exc).__name__,
                "response_sha256": None,
                "provider_request_id": None,
                "structured_response": None,
            })
            raise
        await response.aread()
        body = response.content
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        structured_response: Any = None
        if response.is_success:
            try:
                structured_response = response.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                structured_response = None
        provider_id = structured_response.get("id") if isinstance(structured_response, dict) else None
        safe_request_id = provider_id if isinstance(provider_id, str) else request_id
        self.captures.append({
            "claim_id": claim_id,
            "attempted_at_utc": started.isoformat(),
            "http_status": response.status_code,
            "response_sha256": sha_bytes(body),
            "provider_request_id": safe_request_id if isinstance(safe_request_id, str) else None,
            "structured_response": structured_response,
        })
        return response


def extract_model_object(provider_response: Any) -> dict[str, Any] | None:
    if not isinstance(provider_response, dict):
        return None
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def run() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("output_directory_not_empty_do_not_overwrite_or_rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    _manifest, selected, component_hashes = verify_frozen_inputs()
    identity, material, runtime = await resolve_runtime_identity()
    if identity["safe_fingerprint"] != PROVIDER_FINGERPRINT:
        await runtime.close()
        raise RuntimeError("current_provider_fingerprint_mismatch")

    # This immutable preflight record is written before any outbound request.
    run_record: dict[str, Any] = {
        "artifact_version": "v2_1r_d4j_b_execution.v1",
        "phase": "D4-J-B",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "authorization": "explicit user authorization in current task",
        "selection_manifest_sha256": MANIFEST_SHA256,
        "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
        "provider_fingerprint": PROVIDER_FINGERPRINT,
        "provider_identity": identity,
        "execution_limits": {"claims": list(CLAIM_IDS), "provider_calls_max_total": 8, "provider_calls_max_per_claim": 1, "retry": 0, "regeneration": False, "repair": "none", "provider_fallback": "none", "endpoint_fallback": "none", "request_format_fallback": "none", "semantic_verifier_calls": 0, "holdout_access": 0, "database_writes": 0},
        "preflight": {"frozen_claim_order": True, "selection_manifest": True, "claim_hashes": True, "implementation_hashes": True, "provider_identity": True, "canonical_runtime": True, "v3_1_imports": True, "strict_guard": True, "holdout_access": 0},
        "component_hashes": component_hashes,
        "claim_states": [{"ordinal": i + 1, "claim_id": c["claim_id"], "claim_sha256": c["claim_sha256"], "source_pair_id": c["source_pair_id"], "state": "NOT_STARTED", "extractor_call_count": 0, "external_http_outbound_count": 0, "blocked_outbound_count": 0} for i, c in enumerate(selected)],
        "records": [],
        "provider_calls": 0,
        "external_http_outbound_count": 0,
        "semantic_verifier_calls": 0,
        "database_writes": 0,
        "holdout_access": 0,
    }
    write_json(OUT / "run_manifest.json", run_record)

    transport = httpx.AsyncHTTPTransport(retries=0)
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(connect=15, read=90, write=30, pool=10), trust_env=True)
    hard_client = HardLimitPostClient(client)
    try:
        for ordinal, claim in enumerate(selected, start=1):
            state = run_record["claim_states"][ordinal - 1]
            # Persist a one-way dispatch marker first. Never replay a DISPATCH_ARMED claim.
            state["state"] = "DISPATCH_ARMED"
            state["dispatch_armed_at_utc"] = datetime.now(UTC).isoformat()
            write_json(OUT / "run_manifest.json", run_record)
            hard_client.current_claim = claim["claim_id"]
            before_calls = hard_client.total_outbound
            post_start = len(hard_client.captures)
            started = datetime.now(UTC)
            try:
                result = await execute_one_claim_v31(
                    http_client=hard_client,
                    claim_text=claim["claim_text"],
                    claim_ref=claim["claim_id"],
                    adapter_type=material.adapter_type,
                    base_url=material.base_url,
                    api_key=material.api_key,
                    model=material.model,
                )
                extraction = result.extraction
                strict_outbound = result.external_http_outbound_count
                blocked = result.blocked_additional_attempt_count
            except Exception as exc:
                extraction = {"status": "UNRESOLVED", "diagnostics": ["orchestration_failure", type(exc).__name__], "model_call_count": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
                strict_outbound = max(0, hard_client.total_outbound - before_calls)
                blocked = 0
            elapsed = round((datetime.now(UTC) - started).total_seconds() * 1000)
            capture_slice = hard_client.captures[post_start:]
            raw_capture = capture_slice[-1] if capture_slice else None
            model_object = extract_model_object(raw_capture.get("structured_response")) if raw_capture else None
            raw_path = OUT / "raw" / f"claim_{ordinal:02d}.json"
            if raw_capture and raw_capture.get("structured_response") is not None:
                structured_bundle = {"provider_response": raw_capture["structured_response"], "model_structured_object": model_object}
                write_json(raw_path, structured_bundle)
                raw_hash = sha_file(raw_path)
            else:
                structured_bundle = {"http_status": raw_capture.get("http_status") if raw_capture else None, "exception_category": raw_capture.get("exception_category") if raw_capture else None, "response_sha256": raw_capture.get("response_sha256") if raw_capture else None}
                write_json(raw_path, structured_bundle)
                raw_hash = sha_file(raw_path)
            extraction_status = extraction.get("status", "UNRESOLVED")
            diagnostics = extraction.get("diagnostics", [])
            schema_validation = "FAIL" if any(str(x).startswith(("schema_validation_failed", "structured_output_invalid")) for x in diagnostics) else "PASS" if model_object is not None else "FAIL"
            spans = extraction.get("evidence_spans", [])
            bindings = extraction.get("evidence_bindings", [])
            textual_values = [item.get("textual_grounding") for item in bindings if isinstance(item, dict)]
            structural_values = [item.get("structural_grounding") for item in bindings if isinstance(item, dict)]
            normalized_extraction = {
                "schema_version": extraction.get("schema_version", SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31),
                "status": extraction_status,
                "assertions": extraction.get("source_assertions", []),
                "contexts": extraction.get("source_contexts", []),
                "attachments": extraction.get("source_attachments", []),
                "scopes": extraction.get("source_scopes", []),
                "relations": extraction.get("semantic_relations", []),
                "measurement_bases": extraction.get("measurement_bases", []),
                "evidence_spans": spans,
                "bindings": bindings,
            }
            result_record = {
                "claim": {"claim_id": claim["claim_id"], "claim_sha256": claim["claim_sha256"], "claim_text": claim["claim_text"]},
                "selection": {"manifest_sha256": MANIFEST_SHA256, "ordinal": ordinal, "source_pair_id": claim["source_pair_id"], "development_only": True},
                "provider": {"profile_id": identity["profile_id"], "adapter": identity["adapter"], "model": identity["model"], "endpoint_family": identity["endpoint_family"], "profile_version": identity["profile_version"], "credential_version": identity["credential_version"], "fingerprint": PROVIDER_FINGERPRINT, "fingerprint_contract": "provider-safe-config-fingerprint.v1"},
                "implementation": {"fingerprint": IMPLEMENTATION_FINGERPRINT, "component_hashes": component_hashes},
                "call_accounting": {"extractor_calls": extraction.get("model_call_count", 0), "external_http_outbound_count": strict_outbound, "blocked_additional_attempt_count": blocked, "semantic_verifier_calls": 0},
                "raw_output": {"artifact_path": raw_path.relative_to(OUT).as_posix(), "sha256": raw_hash, "provider_request_id": raw_capture.get("provider_request_id") if raw_capture else None, "structured_response": structured_bundle},
                "extraction": normalized_extraction,
                "validation": {"schema_validation": schema_validation, "diagnostics": list(diagnostics)},
                "grounding": {"textual_grounding": "INVALID" if "INVALID" in textual_values else "VALID" if textual_values else "NOT_APPLICABLE", "structural_grounding": "INVALID" if "INVALID" in structural_values else "VALID" if structural_values else "NOT_APPLICABLE"},
                "completeness": {"unresolved_items": model_object.get("unresolved_items", []) if model_object else [], "diagnostics": list(diagnostics)},
                "projection": {"status": "NOT_CONNECTED_NOT_EVALUATED"},
                "usage": {"input_tokens": extraction.get("input_tokens"), "output_tokens": extraction.get("output_tokens"), "total_tokens": (extraction.get("input_tokens", 0) or 0) + (extraction.get("output_tokens", 0) or 0), "latency_ms": extraction.get("latency_ms", elapsed)},
                "execution_diagnostics": {"execution_status": "SUCCESS" if strict_outbound == 1 and model_object is not None else "EXECUTION_OR_MODEL_FAILURE", "model_returned_status": model_object.get("status") if model_object else None, "provider_http_status": raw_capture.get("http_status") if raw_capture else None, "elapsed_ms_wall": elapsed, "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(), "semantic_support": "NOT_EVALUATED", "request_body_persisted": False},
            }
            per_claim_path = OUT / f"claim_{ordinal:02d}.json"
            write_json(per_claim_path, result_record)
            state.update({"state": "COMPLETED", "extractor_call_count": result_record["call_accounting"]["extractor_calls"], "external_http_outbound_count": strict_outbound, "blocked_outbound_count": blocked, "execution_status": result_record["execution_diagnostics"]["execution_status"], "result_path": per_claim_path.name, "raw_path": raw_path.relative_to(OUT).as_posix()})
            run_record["records"].append({"ordinal": ordinal, "result_path": per_claim_path.name, "result_sha256": sha_file(per_claim_path), "raw_path": raw_path.relative_to(OUT).as_posix(), "raw_sha256": raw_hash})
            run_record["provider_calls"] = sum(x.get("extractor_call_count", 0) for x in run_record["claim_states"])
            run_record["external_http_outbound_count"] = hard_client.total_outbound
            run_record["transport_captures"] = hard_client.captures
            write_json(OUT / "run_manifest.json", run_record)
            if strict_outbound > 1 or hard_client.total_outbound > 8 or result_record["call_accounting"]["extractor_calls"] > 1:
                raise RuntimeError("call_accounting_invariant_violated_stop_no_more_claims")
            print(json.dumps({"ordinal": ordinal, "claim_id": claim["claim_id"], "extractor_calls": result_record["call_accounting"]["extractor_calls"], "external_http": strict_outbound, "status": result_record["execution_diagnostics"]["execution_status"], "provider_request_id": result_record["raw_output"]["provider_request_id"], "latency_ms": result_record["usage"]["latency_ms"]}, sort_keys=True), flush=True)
    finally:
        await client.aclose()
        await runtime.close()
    run_record["completed_at_utc"] = datetime.now(UTC).isoformat()
    run_record["provider_calls"] = sum(x.get("extractor_call_count", 0) for x in run_record["claim_states"])
    run_record["external_http_outbound_count"] = hard_client.total_outbound
    run_record["transport_captures"] = hard_client.captures
    run_record["run_status"] = "ALL_EIGHT_ATTEMPTED" if len(run_record["records"]) == 8 else "INCOMPLETE_NO_RERUN"
    write_json(OUT / "run_manifest.json", run_record)
    print(json.dumps({"run_status": run_record["run_status"], "provider_calls": run_record["provider_calls"], "external_http_outbound_count": run_record["external_http_outbound_count"], "result_count": len(run_record["records"]), "output_dir": str(OUT)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
