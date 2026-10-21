from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_d4j_selection_manifest_hash_and_freshness() -> None:
    manifest = read_json(ART / "v2_1r_d4j_a_selection_manifest.json")
    expected = manifest.pop("normalized_manifest_sha256")
    assert hashlib.sha256(canonical(manifest)).hexdigest() == expected
    criteria = read_json(ART / "v2_1r_d4j_a_selection_criteria.json")
    assert hashlib.sha256(canonical(criteria)).hexdigest() == manifest["selection_criteria_sha256"]
    pool_path = ART / "v2_1_dev_extension_v2_candidates.json"
    assert hashlib.sha256(pool_path.read_bytes()).hexdigest() == manifest["candidate_source_sha256"]
    pool = read_json(pool_path)
    pool_hashes = {
        (pair["pair_id"], pair[f"claim_{side}_id"]): hashlib.sha256(
            pair[f"claim_{side}_text"].encode("utf-8")
        ).hexdigest()
        for pair in pool["pairs"]
        for side in ("a", "b")
    }
    claims = manifest["claims"]
    assert len(claims) == 8
    assert len({item["claim_id"] for item in claims}) == 8
    assert len({item["source_pair_id"] for item in claims}) == 8
    assert all(item["development_only_forever"] is True for item in claims)
    assert all(
        pool_hashes[(item["source_pair_id"], item["claim_id"])] == item["claim_sha256"]
        for item in claims
    )

    audit = read_json(ART / "v2_1r_d4j_a_exclusion_audit.json")
    assert audit["selected_ids_overlap_prior_development"] == 0
    assert audit["selected_ids_overlap_d4b"] == 0
    assert audit["selected_ids_overlap_d4g_b"] == 0
    assert len(audit["candidate_identity_records"]) >= audit["candidate_unique_claim_count"]
    records = {item["claim_id"]: item for item in audit["candidate_identity_records"]}
    assert all(records[item["claim_id"]]["excluded"] is False for item in claims)


def test_d4j_frozen_components_and_provider_fingerprint() -> None:
    freeze = read_json(ART / "v2_1r_d4j_a/implementation_freeze.json")
    for relative, expected in freeze["component_hashes"].items():
        current = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if relative == "apps/api/app/domain/source_context_extraction_v3_1.py":
            # Historical D4-J provenance stays frozen; D4-L owns the parser patch identity.
            assert current != expected
            d4l = read_json(ART / "v2_1r_d4l/hash_manifest.json")
            assert d4l["parser_patch_version"] == "source-context-extraction-parser-v3.1.1"
            assert d4l["files"][relative] == current
            continue
        assert current == expected
    assert hashlib.sha256(canonical(freeze["component_hashes"])).hexdigest() == freeze["implementation_fingerprint"]

    provider = read_json(ART / "v2_1r_d4j_a/provider_readiness_snapshot.json")
    payload = provider["canonical_safe_payload"]
    assert hashlib.sha256(canonical(payload)).hexdigest() == provider["safe_fingerprint"]
    assert provider["secret_plaintext_read_or_emitted"] is False


def test_d4j_result_schema_contract_and_call_limits() -> None:
    schema = read_json(ART / "v2_1r_d4j_a/result_schema.json")
    required = set(schema["required"])
    assert {"claim", "selection", "provider", "implementation", "call_accounting", "raw_output", "extraction", "validation", "grounding", "completeness", "projection", "usage", "execution_diagnostics"} <= required
    assert schema["properties"]["call_accounting"]["properties"]["external_http_outbound_count"]["maximum"] == 1
    assert schema["properties"]["call_accounting"]["properties"]["semantic_verifier_calls"]["const"] == 0
    assert schema["properties"]["extraction"]["properties"]["schema_version"]["const"] == "source-context-extraction-schema-v3.1"


def test_d4j_historical_downstream_references_remain_unchanged() -> None:
    freeze = read_json(ART / "v2_1r_d4j_a/historical_downstream_reference_freeze.json")
    assert freeze["all_references_match"] is True
    assert freeze["downstream_modified"] is False
