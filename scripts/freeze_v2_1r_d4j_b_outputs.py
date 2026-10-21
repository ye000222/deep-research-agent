"""Freeze D4-J-B raw/normalized outputs and call accounting before review."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v2_1r_d4j_b"
EXPECTED_IDS = [
    "01a0c41c-a7d9-7ef5-bbde-b70d8d0787cd",
    "01a0c41c-2835-7208-bd74-5e9ddebd8963",
    "01a0c43b-3e5a-7801-8bda-b8ade1fd7eb8",
    "01a0c405-03ed-79fd-9758-15029694fc0c",
    "01a0c834-62f9-7c30-8813-20d24892e2f4",
    "01a0c836-0c24-7a71-a449-69b63110a68f",
    "01a0c82b-c360-7215-9e7e-f083d3fc91d5",
    "01a0c406-6970-7f63-be80-943c2a5822db",
]
MANIFEST_SHA256 = "847f65be7dd2dddb6b4143c2f2d874779ff089ab62c5877b228b0a75ee6f1c74"
IMPLEMENTATION_FINGERPRINT = "a554d3d8264e41cb09376fb98b9a61ec86e1cadf21bef05ed032909ba7b516a2"
PROVIDER_FINGERPRINT = "32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    temp.write_bytes(payload)
    temp.replace(path)


def main() -> None:
    run_path = OUT / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    results = [
        json.loads((OUT / f"claim_{i:02d}.json").read_text(encoding="utf-8"))
        for i in range(1, 9)
    ]
    if [item["claim"]["claim_id"] for item in results] != EXPECTED_IDS:
        raise SystemExit("claim_order_or_identity_mismatch")
    if run.get("run_status") != "ALL_EIGHT_ATTEMPTED" or len(run.get("records", [])) != 8:
        raise SystemExit("run_incomplete_or_accounting_records_missing")
    if (
        run.get("selection_manifest_sha256") != MANIFEST_SHA256
        or run.get("implementation_fingerprint") != IMPLEMENTATION_FINGERPRINT
        or run.get("provider_fingerprint") != PROVIDER_FINGERPRINT
    ):
        raise SystemExit("frozen_identity_mismatch")
    if (
        run.get("provider_calls") != 8
        or run.get("external_http_outbound_count") != 8
        or run.get("semantic_verifier_calls") != 0
        or run.get("database_writes") != 0
        or run.get("holdout_access") != 0
    ):
        raise SystemExit("aggregate_call_or_isolation_invariant_failed")
    raw_records = []
    for ordinal, result in enumerate(results, start=1):
        accounting = result["call_accounting"]
        expected_accounting = {
            "extractor_calls": 1,
            "external_http_outbound_count": 1,
            "blocked_additional_attempt_count": 0,
            "semantic_verifier_calls": 0,
        }
        if accounting != expected_accounting:
            raise SystemExit(f"per_claim_accounting_invariant_failed:{ordinal}")
        raw_rel = result["raw_output"]["artifact_path"]
        raw_path = OUT / raw_rel
        if not raw_path.is_file() or sha(raw_path) != result["raw_output"]["sha256"]:
            raise SystemExit(f"raw_output_hash_mismatch:{ordinal}")
        raw_records.append({
            "ordinal": ordinal,
            "claim_id": result["claim"]["claim_id"],
            "claim_sha256": result["claim"]["claim_sha256"],
            "provider_request_id": result["raw_output"]["provider_request_id"],
            "raw_output_path": raw_rel,
            "raw_output_sha256": sha(raw_path),
            "normalized_result_path": f"claim_{ordinal:02d}.json",
            "normalized_result_sha256": sha(OUT / f"claim_{ordinal:02d}.json"),
            "schema_status": result["validation"]["schema_validation"],
            "extraction_status": result["extraction"]["status"],
            "external_http_outbound_count": accounting["external_http_outbound_count"],
            "extractor_call_count": accounting["extractor_calls"],
            "blocked_additional_attempt_count": accounting["blocked_additional_attempt_count"],
        })
    manifest = {
        "artifact_version": "v2_1r_d4j_b_raw_freeze_manifest.v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "frozen_before_human_review": True,
        "selection_manifest_sha256": MANIFEST_SHA256,
        "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
        "provider_fingerprint": PROVIDER_FINGERPRINT,
        "provider_identity": run["provider_identity"],
        "call_limits": run["execution_limits"],
        "provider_calls": run["provider_calls"],
        "external_http_outbound_count": run["external_http_outbound_count"],
        "database_writes": run["database_writes"],
        "holdout_access": run["holdout_access"],
        "semantic_verifier_calls": run["semantic_verifier_calls"],
        "claim_results": raw_records,
        "run_manifest_sha256": sha(run_path),
        "hash_self_reference": "NOT_INCLUDED",
    }
    write_json(OUT / "raw_freeze_manifest.json", manifest)
    summary = {
        "status": "RAW_AND_NORMALIZED_OUTPUTS_FROZEN_BEFORE_REVIEW",
        "claims": len(raw_records),
        "provider_calls": run["provider_calls"],
        "external_http": run["external_http_outbound_count"],
        "freeze_manifest_sha256": sha(OUT / "raw_freeze_manifest.json"),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
