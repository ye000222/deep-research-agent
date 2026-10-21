"""Execute only the 34 explicitly authorized pending Claims once each.

The six completed Canary outputs are reused; this runner never calls them.
It is an isolated evaluation harness and writes JSON under /tmp only (no DB
writes). Frozen extractor, schema, prompt, validators, and protocol are read-only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, os.environ.get("CANARY_APP_ROOT", "/app/apps/api"))
sys.path.insert(0, "/tmp/v21r-eval")

import run_v2_1r_source_context_extraction_canary as canary
from app.services.source_context_extractor import SourceContextExtractor


ROOT = Path("/tmp/v21r-eval")
AUDIT_PATH = ROOT / "manual_audit.json"
AUDIT_SELECTION_PATH = ROOT / "audit_selection.json"
CANARY_MANIFEST_PATH = ROOT / "canary_manifest.json"
CANARY_RESULT_PATH = ROOT / "canary_result.json"
EVAL_MANIFEST_PATH = ROOT / "eval_manifest.json"
PROTOCOL_PATH = ROOT / "matching_protocol.json"
RESULT_PATH = Path("/tmp/v2_1r_source_context_extraction_40_eval_result.json")
JOURNAL_PATH = Path("/tmp/v2_1r_source_context_extraction_40_eval_progress.json")

AUTHORIZED_PENDING_IDS = (
    "01a0c84e-9da9-7c02-b932-d5b8aca2bd1e",
    "01a0c3f5-e702-7de6-9b2d-95d70e3c98e0",
    "01a0c86a-a4b4-7df3-965e-49e36b5e2b81",
    "01a0c853-cc40-7fb3-a301-3ae526418013",
    "01a0c435-a745-7743-b8e9-8e2a019e0aaa",
    "01a0c824-555b-7726-8a33-2d98136b0339",
    "01a0a725-5062-7e1f-b7e6-9a4823e02d75",
    "01a0a280-af69-7a7f-9ce7-3ca2c833b16c",
    "01a0a56e-2c22-7cae-b549-a7dcfadd6b02",
    "01a0a5f0-e9cd-789a-9b17-afa3cf2f5bdb",
    "01a0907c-af4f-724d-8033-f15c534ff5a9",
    "01a0aff2-f4d1-7475-8384-f1fb4857f693",
    "01a0c83d-2b7c-794a-a8f9-63a6c482cd6b",
    "01a0c873-f06d-70d7-b1e7-03e3e28b9534",
    "01a0c895-475e-79cf-b39e-adb4a1deeaf6",
    "01a0c86f-4a85-78bd-b881-e3572c87241c",
    "01a0c841-bac9-7d88-a2d6-5ebbe038249b",
    "01a0a62b-2ba5-701f-9a67-a8717cebf17e",
    "01a0a0df-6389-760c-a247-68dd8ae23df2",
    "01a0af7f-f06b-7fa2-9c95-28ba267af9b8",
    "01a0b79d-3bbb-79d3-9b53-8bd2c7684c53",
    "01a09400-6005-7d39-a86f-93ddbe90dca7",
    "01a0c84e-9da1-7478-ac92-ed89de3957a5",
    "01a0c852-8c79-78ef-b48f-f5bc9f1b2a51",
    "01a0c844-074d-70ed-8137-4d0d56a254ee",
    "01a0c890-2762-7d97-9285-ab4d239b267c",
    "01a093ff-abfe-7077-a846-103a610c7a0e",
    "01a0af93-04c7-71cb-a6c5-328e68ca7f92",
    "01a0a4e7-1520-7ef4-8821-5aa3f1e56885",
    "01a0905d-88cd-7ecf-9434-ad09ab75006e",
    "01a0afec-4e7f-7eb4-922d-25d4f5f7d762",
    "01a0c835-1e99-7d3a-b0d3-41882d52cd21",
    "01a0c43c-8d2a-7f0f-a27f-aae8760568a3",
    "01a0c86c-b74f-77af-8259-624b11e64061",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_progress(records: list[dict[str, Any]], sent: int, manifest_hash: str) -> None:
    JOURNAL_PATH.write_text(json.dumps({
        "status": "IN_PROGRESS",
        "completed_claim_count": len(records),
        "provider_outbound_total": sent,
        "manifest_sha256": manifest_hash,
        "records": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run() -> None:
    if RESULT_PATH.exists() or JOURNAL_PATH.exists():
        raise RuntimeError("existing_execution_artifact_refuse_possible_duplicate_calls")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    canary_manifest = json.loads(CANARY_MANIFEST_PATH.read_text(encoding="utf-8"))
    canary_result = json.loads(CANARY_RESULT_PATH.read_text(encoding="utf-8"))
    manifest_bytes = EVAL_MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    if audit.get("claims_total") != 40 or len(audit.get("claims", [])) != 40:
        raise RuntimeError("frozen_audit_membership_invalid")
    if audit.get("development_only") is not True or audit.get("holdout_content_read") is not False:
        raise RuntimeError("development_only_attestation_invalid")
    if hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest() != manifest["source_audit"]["sha256"]:
        raise RuntimeError("manual_audit_hash_mismatch")
    if hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() != manifest["matching_protocol"]["sha256"]:
        raise RuntimeError("matching_protocol_hash_mismatch")
    manifest_copy = dict(manifest)
    declared_manifest_hash = manifest_copy.pop("manifest_sha256")
    if stable_hash(manifest_copy) != declared_manifest_hash:
        raise RuntimeError("evaluation_manifest_canonical_hash_mismatch")
    if manifest.get("pending_claim_ids") != list(AUTHORIZED_PENDING_IDS):
        raise RuntimeError("authorized_pending_claim_list_mismatch")
    pending_rows = [row for row in manifest["claims"] if row.get("origin") == "PROVIDER_PENDING"]
    reused_rows = [row for row in manifest["claims"] if row.get("origin") == "EXISTING_CANARY"]
    if [row["claim_id"] for row in pending_rows] != list(AUTHORIZED_PENDING_IDS) or len(reused_rows) != 6:
        raise RuntimeError("40_claim_manifest_origin_mismatch")
    if canary_result.get("claim_count") != 6 or canary_result.get("provider_outbound_total") != 6:
        raise RuntimeError("existing_canary_result_incomplete")
    if canary_result.get("database_writes") is not False or canary_result.get("holdout_content_used") is not False:
        raise RuntimeError("existing_canary_result_scope_mismatch")
    if canary_result.get("frozen_hashes") != canary_manifest.get("component_hashes"):
        raise RuntimeError("canary_manifest_result_hash_bundle_mismatch")

    audit_rows = {row["claim_id"]: row for row in audit["claims"]}
    for row in manifest["claims"]:
        original = audit_rows.get(row["claim_id"])
        if original is None:
            raise RuntimeError("manifest_claim_not_in_frozen_audit")
        actual_claim_hash = hashlib.sha256(original["claim_text"].encode("utf-8")).hexdigest()
        if actual_claim_hash != row["claim_sha256"]:
            raise RuntimeError("claim_hash_mismatch")

    selection_sha = canary_manifest["component_hashes"]["audit_selection_sha256"]
    manual_sha = canary_manifest["component_hashes"]["manual_audit_sha256"]
    if sha256_file(AUDIT_SELECTION_PATH) != selection_sha or sha256_file(AUDIT_PATH) != manual_sha:
        raise RuntimeError("frozen_audit_identity_changed")
    current_hashes = canary.component_hashes(Path(__file__), AUDIT_SELECTION_PATH, AUDIT_PATH)
    frozen_hashes = canary_manifest["component_hashes"]
    semantic_keys = (
        "prompt_source_sha256", "prompt_text_sha256", "schema_source_sha256", "schema_sha256",
        "service_parser_sha256", "grounding_validator_sha256", "truth_condition_context_sha256",
        "typed_normalizer_sha256", "auxiliary_detector_sha256",
    )
    for key in semantic_keys:
        if current_hashes.get(key) != frozen_hashes.get(key):
            raise RuntimeError(f"frozen_semantic_component_changed:{key}")

    # Read-only DB profile resolution and credential decryption; nothing is persisted.
    current_profile, material = await canary.resolve_profile()
    if current_profile != canary_manifest.get("provider_profile"):
        raise RuntimeError("provider_profile_or_credential_version_changed")
    if manifest["contract"]["provider_profile_id"] != current_profile["profile_id"]:
        raise RuntimeError("evaluation_provider_profile_mismatch")
    if manifest["contract"]["provider_model"] != current_profile["model"]:
        raise RuntimeError("evaluation_provider_model_mismatch")
    if manifest["contract"]["fusion_policy"] != "NONE" or manifest["request_limits_if_authorized"]["retry"] != 0:
        raise RuntimeError("frozen_execution_policy_invalid")

    print(json.dumps({
        "status": "PREFLIGHT_PASS_PROVIDER_EXECUTION_STARTING",
        "manifest_sha256_file": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_canonical_sha256": declared_manifest_hash,
        "claims_total": 40,
        "existing_canary_reused": 6,
        "new_claims_authorized": len(AUTHORIZED_PENDING_IDS),
        "current_provider_profile_id": current_profile["profile_id"],
        "adapter_type": current_profile["adapter_type"],
        "endpoint_host": current_profile["endpoint_host"],
        "model": current_profile["model"],
        "profile_version": current_profile["profile_version"],
        "credential_version": current_profile["credential_version"],
        "semantic_hash_match": True,
        "retry": 0,
        "regeneration": False,
        "fallback": "none",
        "repair": "none",
        "fusion_policy": "NONE",
        "database_writes": False,
        "holdout_content_used": False,
        "provider_calls_so_far": 0,
    }, ensure_ascii=False), flush=True)
    if os.environ.get("SOURCE_CONTEXT_EVAL_PREFLIGHT_ONLY") == "1":
        print(json.dumps({"status": "OFFLINE_PREFLIGHT_ONLY_PASS_NO_PROVIDER_CALLS"}), flush=True)
        return

    http_transport = canary.HardLimitTransport(limit=34)
    client = httpx.AsyncClient(
        transport=http_transport,
        timeout=httpx.Timeout(connect=15.0, read=90.0, write=30.0, pool=10.0),
        trust_env=False,
    )
    observed = canary.ObservedGateway(canary.LLMGateway(client, call_recorder=None))
    records: list[dict[str, Any]] = []
    manifest_file_hash = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        for index, claim_id in enumerate(AUTHORIZED_PENDING_IDS, start=1):
            claim = audit_rows[claim_id]
            claim_hash = hashlib.sha256(claim["claim_text"].encode("utf-8")).hexdigest()
            transport_before = http_transport.total_sent
            calls_before = observed.calls
            blocked_before = len(http_transport.blocked_attempts)
            http_transport.current_claim = claim_id
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
            # The extractor receives only raw Claim text and its opaque local ref.
            result = await extractor.extract(claim["claim_text"], claim_ref=claim_id)
            outbound = http_transport.total_sent - transport_before
            gateway_calls = observed.calls - calls_before
            if outbound > 1 or gateway_calls > 1:
                raise RuntimeError("per_claim_call_limit_invariant_failed")
            if len(http_transport.blocked_attempts) != blocked_before:
                raise RuntimeError("unexpected_extra_provider_attempt_blocked_stop")

            records.append({
                "claim_id": claim_id,
                "claim_text": claim["claim_text"],
                "claim_sha256": claim_hash,
                "origin": "PROVIDER_PENDING",
                "human_source_contexts_post_call_only": claim.get("manual_source_contexts", []),
                "human_detector_contexts_post_call_only": claim.get("detector_contexts", []),
                "result": result.model_dump(mode="json"),
                "actual_provider_outbound_count": outbound,
                "gateway_invocation_count": gateway_calls,
                "provider_request_id": observed.last_request_id,
                "safe_gateway_error_code": observed.safe_error_code,
            })
            write_progress(records, http_transport.total_sent, manifest_file_hash)
            print(json.dumps({
                "status": "CLAIM_COMPLETED",
                "ordinal": index,
                "claim_id": claim_id,
                "model_status": result.model_status,
                "grounding_status": result.grounding_status.value,
                "source_assertions": len(result.source_assertions),
                "source_contexts": len(result.source_contexts),
                "outbound_for_claim": outbound,
                "provider_outbound_total": http_transport.total_sent,
                "latency_ms": result.latency_ms,
            }, ensure_ascii=False), flush=True)
    finally:
        await client.aclose()

    output = {
        "experiment_id": manifest["experiment_id"],
        "manifest_path": "artifacts/v2_1r_source_context_extraction_40_eval_manifest.json",
        "manifest_file_sha256": manifest_file_hash,
        "manifest_canonical_sha256": declared_manifest_hash,
        "matching_protocol_sha256": manifest["matching_protocol"]["sha256"],
        "provider_profile": {key: value for key, value in current_profile.items() if key != "credential_version_id"},
        "frozen_semantic_component_hashes": {key: frozen_hashes[key] for key in semantic_keys},
        "claim_count_new": len(records),
        "provider_outbound_total_new": http_transport.total_sent,
        "retry_count": 0,
        "regeneration": False,
        "fallback": "none",
        "repair": "none",
        "fusion_policy": "NONE",
        "database_writes": False,
        "holdout_content_used": False,
        "records": records,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    JOURNAL_PATH.unlink(missing_ok=True)
    print(json.dumps({
        "status": "40_CLAIM_PENDING_EXECUTION_COMPLETE",
        "result_path": str(RESULT_PATH),
        "new_claims_completed": len(records),
        "provider_outbound_total_new": http_transport.total_sent,
        "blocked_extra_attempts": len(http_transport.blocked_attempts),
        "database_writes": False,
        "holdout_content_used": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as exc:  # Emit only a safe exception class, never message/secret.
        print(json.dumps({"status": "STOPPED", "safe_failure_class": type(exc).__name__}, ensure_ascii=False), flush=True)
        raise SystemExit(2)
