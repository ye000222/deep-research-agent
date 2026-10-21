"""Freeze an offline manifest for the V2.1R 40-Claim development evaluation.

This preparation tool makes no Provider calls and performs no database writes.
It reuses the previously completed six-Claim Canary outputs only after checking
the frozen semantic component hashes against the hashes independently measured
in the isolated API runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
AUDIT_PATH = ARTIFACTS / "v2_1r_truth_condition_context_c1_manual_audit.json"
CANARY_MANIFEST_PATH = ARTIFACTS / "v2_1r_source_context_extraction_real_canary_manifest.json"
CANARY_RESULT_PATH = ARTIFACTS / "v2_1r_source_context_extraction_real_canary_result.json"
OUT_PATH = ARTIFACTS / "v2_1r_source_context_extraction_40_eval_manifest.json"
PROTOCOL_PATH = ARTIFACTS / "v2_1r_source_context_matching_protocol_v1.json"

# Read-only measurements recomputed from the running isolated API container
# before any new Provider request. Keep the exact observed values in the
# manifest so this preparation is auditable without importing the live runtime.
RUNTIME_SEMANTIC_HASHES = {
    "prompt_source_sha256": "b894f2c543267b0a2ec83ada1662a0a11bc3cd8da37cfaf4bd0e7ce8c07c66e8",
    "prompt_text_sha256": "202d4802069df50612d4c6d06ecaebd68b105aea4dd962b234884a2897cd8ca4",
    "schema_source_sha256": "4f64212c3b4985fc0e7a2df7749fbb6d2ede95ff0a9c6ef25d6c3812a7814b81",
    "schema_sha256": "73bbda2541a86955510c649df0349591abd9825f7e5afcf81cb7cca67192e4f9",
    "service_parser_sha256": "b894f2c543267b0a2ec83ada1662a0a11bc3cd8da37cfaf4bd0e7ce8c07c66e8",
    "grounding_validator_sha256": "fa3e9853ab354442dae3be4a7c0d32aafae0a080d8680b651b3afa28b32bfeec",
    "truth_condition_context_sha256": "8b4b49d734cf4bdfbc5516f14eb71aba55e356d5802e1e6d400256dc5c568c5f",
    "typed_normalizer_sha256": "5f66193635d9524151053b015df6a79f2d0277f832e6e9361f13cf0e3fde0533",
    "auxiliary_detector_sha256": "68f205c4907db48b50a90b8359deac2c2c0b2705b9610b7adb68a2411414853d",
}

MATCHING_PROTOCOL: dict[str, Any] = {
    "protocol_version": "source-context-manual-matching-v1",
    "frozen_before_review_of_pending_provider_outputs": True,
    "unit": "human-annotated material context; preserve the existing 40-Claim membership and labels",
    "matching": {
        "many_to_many": "Allowed for semantic correspondence, but one model context cannot create more than one strict true positive for the same human context, and one human context cannot be counted more than once.",
        "EXACT": "Same material semantic content/family, grounded source span covering the human context, compatible semantic payload, correct target scope, and correct attachment; no added or lost material meaning.",
        "PARTIAL": "A grounded, semantically relevant context captures only part of the human context or has a limited payload/scope/attachment deficiency without being a wrong-type or unsupported claim. Report separately; never assign a fractional strict TP.",
        "MISSED": "A human material context has no corresponding model context after considering permitted many-to-many links.",
        "WRONG_TYPE": "Semantically corresponding content is represented under an incompatible context family/type. Count as a type mismatch, not exact; retain a separate semantic-presence note.",
        "WRONG_PAYLOAD": "The type is plausible but normalized meaning/payload materially differs from the human label.",
        "WRONG_SCOPE": "The context is attached to an incorrect or incomplete assertion/proposition target set, or has unsupported scope.",
        "WRONG_ATTACHMENT": "The context is linked to the wrong assertion(s), with attachment judged independently from family and payload.",
        "EXTRA_SUPPORTED": "A model context not mapped to a human label but whose complete meaning and scope are directly supported by the Claim; report as extra, not TP.",
        "EXTRA_UNSUPPORTED": "A model context not mapped to a human label and not fully grounded in the Claim; record as a fabrication/grounding concern.",
        "LABEL_AMBIGUOUS": "The frozen human label does not determine a unique type, span, scope, or attachment. Exclude it from strict scored denominators and report separately; do not relabel it."
    },
    "grounding": "A span is grounded only when its exact text and offsets resolve to the original Claim under the frozen grounding validator. Do not infer or fabricate missing spans.",
    "shared_contexts": "Evaluate each human shared context once against its full frozen target set; target-set coverage and attachment correctness are separate judgments.",
    "merge_split": "Review semantic unions/splits as a graph; map all justified edges, but count each human context and each model context at most once for strict TP accounting.",
    "semantic_vs_taxonomy": "A supported semantic context under OTHER may count as semantic presence in a secondary diagnostic, but is not an exact family match.",
    "metrics": {
        "strict_true_positive": "EXACT only",
        "partial": "Reported separately; no fractional weights",
        "material_context_coverage": "(EXACT + PARTIAL) / 123; label as coverage, not recall",
        "strict_precision_recall": "Use exact-only TP, and disclose how unsupported/ambiguous and unmatched model contexts affect denominators."
    }
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    audit = read_json(AUDIT_PATH)
    canary_manifest = read_json(CANARY_MANIFEST_PATH)
    canary_result = read_json(CANARY_RESULT_PATH)
    if audit.get("claims_total") != 40 or len(audit.get("claims", [])) != 40:
        raise SystemExit("STOP: frozen audit is not exactly 40 claims")
    if audit.get("development_only") is not True or audit.get("holdout_content_read") is not False:
        raise SystemExit("STOP: audit is not attested development-only")
    if audit.get("aggregate", {}).get("manual_contexts") != 123:
        raise SystemExit("STOP: expected frozen denominator 123 not found")
    if canary_result.get("claim_count") != 6 or canary_result.get("provider_outbound_total") != 6:
        raise SystemExit("STOP: existing Canary result is not a completed six-call result")
    if canary_result.get("database_writes") is not False or canary_result.get("holdout_content_used") is not False:
        raise SystemExit("STOP: Canary result violates the reuse constraints")

    frozen = canary_manifest.get("component_hashes", {})
    if any(frozen.get(key) != value for key, value in RUNTIME_SEMANTIC_HASHES.items()):
        raise SystemExit("STOP: semantic component hashes differ; do not mix Canary outputs")
    if canary_result.get("frozen_hashes", {}) != frozen:
        raise SystemExit("STOP: Canary result and frozen manifest hash bundles differ")
    if canary_manifest.get("fusion_policy") != "NONE" and canary_manifest.get("contract", {}).get("fusion_policy") != "NONE":
        raise SystemExit("STOP: prior Canary fusion policy is not NONE")

    audited = {claim["claim_id"]: claim for claim in audit["claims"]}
    canary_records = {record["claim_id"]: record for record in canary_result.get("records", [])}
    canary_ids = list(canary_manifest.get("claim_ids", []))
    if len(canary_ids) != 6 or set(canary_ids) != set(canary_records):
        raise SystemExit("STOP: six reusable Canary identities do not match result records")

    rows = []
    for ordinal, claim in enumerate(audit["claims"], start=1):
        claim_id = claim["claim_id"]
        claim_hash = sha256_bytes(claim["claim_text"].encode("utf-8"))
        canary_record = canary_records.get(claim_id)
        if canary_record and canary_record.get("claim_sha256") != claim_hash:
            raise SystemExit(f"STOP: Claim hash mismatch for existing Canary claim {claim_id}")
        if canary_record and (canary_record.get("result", {}).get("grounding_status") is None):
            raise SystemExit(f"STOP: no reusable extraction result for {claim_id}")
        rows.append({
            "ordinal": ordinal,
            "claim_id": claim_id,
            "claim_sha256": claim_hash,
            "manual_audit_identity_hash": claim.get("identity_hash"),
            "origin": "EXISTING_CANARY" if canary_record else "PROVIDER_PENDING",
            "cohort_label_from_frozen_audit": claim.get("cohort"),
            "holdout": False,
            "reused_result_ref": f"v2_1r_source_context_extraction_real_canary_result.json#records/{claim_id}" if canary_record else None,
        })

    pending = [row["claim_id"] for row in rows if row["origin"] == "PROVIDER_PENDING"]
    if len(pending) != 34 or len(rows) != 40:
        raise SystemExit("STOP: expected exactly 34 pending and 40 total")

    protocol_payload = dict(MATCHING_PROTOCOL)
    protocol_payload["manual_label_artifact_sha256"] = sha256_file(AUDIT_PATH)
    write_json(PROTOCOL_PATH, protocol_payload)
    protocol_sha = sha256_file(PROTOCOL_PATH)

    manifest = {
        "experiment_id": "v2_1r_source_context_extraction_development_eval_40",
        "status": "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION_FOR_34_PENDING_CLAIMS",
        "evaluation_scope": "development-only; no Holdout, fresh validation, 128 replay, or V2.2",
        "source_audit": {
            "path": str(AUDIT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(AUDIT_PATH),
            "selection_manifest_sha256": audit["selection_manifest_sha256"],
            "source_replay_sha256": audit["source_replay_sha256"],
            "claims_total": 40,
            "human_material_contexts": 123,
            "development_only": True,
            "holdout_content_read": False,
        },
        "reuse_canary": {
            "manifest_path": str(CANARY_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256_file(CANARY_MANIFEST_PATH),
            "result_path": str(CANARY_RESULT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": sha256_file(CANARY_RESULT_PATH),
            "reused_claim_ids": canary_ids,
            "reuse_authorized_by_hash_match": True,
            "provider_calls_already_spent": 6,
        },
        "contract": {
            "prompt_version": canary_manifest["contract"]["prompt_version"],
            "schema_version": canary_manifest["contract"]["schema_version"],
            "fusion_policy": "NONE",
            "provider_profile_id": canary_manifest["provider_profile"]["profile_id"],
            "adapter_type": canary_manifest["provider_profile"]["adapter_type"],
            "provider_model": canary_manifest["provider_profile"]["model"],
            "credential_version": canary_manifest["provider_profile"]["credential_version"],
            "semantic_component_hashes": RUNTIME_SEMANTIC_HASHES,
        },
        "matching_protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": protocol_sha,
            "protocol_version": protocol_payload["protocol_version"],
            "frozen_before_pending_output_review": True,
        },
        "authorization": {
            "prior_explicit_authorization_scope": "the six existing Canary Claims only",
            "remaining_34_provider_calls_authorized": False,
            "new_provider_calls_so_far": 0,
            "execution_blocked_until_explicit_user_authorization_for_exact_pending_ids": True,
        },
        "request_limits_if_authorized": {
            "max_new_provider_calls": 34,
            "calls_per_pending_claim": 1,
            "retry": 0,
            "regeneration": False,
            "fallback": "none",
            "repair": "none",
            "database_writes": False,
            "holdout_content_used": False,
        },
        "claims": rows,
        "pending_claim_ids": pending,
    }
    manifest["manifest_sha256"] = sha256_bytes(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    write_json(OUT_PATH, manifest)
    print(json.dumps({
        "manifest_path": str(OUT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256_file(OUT_PATH),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": protocol_sha,
        "total": len(rows),
        "reused": len(canary_ids),
        "pending": len(pending),
        "new_provider_calls": 0,
        "pending_claim_ids": pending,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
