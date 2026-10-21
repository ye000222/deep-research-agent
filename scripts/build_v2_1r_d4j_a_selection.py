"""Freeze a fresh development-only D4-J-A cohort using structural signals only."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
POOL = ART / "v2_1_dev_extension_v2_candidates.json"
CRITERIA = ART / "v2_1r_d4j_a_selection_criteria.json"
MANIFEST = ART / "v2_1r_d4j_a_selection_manifest.json"
AUDIT = ART / "v2_1r_d4j_a_exclusion_audit.json"
COHORT_SIZE = 8

MEMBERSHIP_SOURCES = (
    ("artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json", "claim_replays", "claim_id"),
    ("artifacts/v2_1r_source_context_extraction_40_eval_manifest.json", "claims", "claim_id"),
    ("artifacts/v2_1r_source_context_extraction_real_canary_manifest.json", "claim_ids", ""),
    ("artifacts/v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_selection_manifest.json", "selection.selected_claim_ids", ""),
    ("artifacts/v2_1r_bounded_shadow_100_selection_manifest.json", "selected_claims", "claim_id"),
    ("artifacts/v2_1r_qualifier_semantic_preservation_fresh_validation_12_selection_manifest.json", "selected_claims", "claim_id"),
    ("artifacts/v2_1r_final_fresh_qualifier_check_selection_manifest.json", "selected_claims", "claim_id"),
    ("artifacts/v2_1r_final_fresh_mini_shadow_3/pre_provider_manifest.json", "selected_claim_ids", ""),
    ("artifacts/v2_1r_final_fresh_mini_shadow_4/mini_shadow_4_execution_summary.json", "claims", "claim_id"),
    ("artifacts/v2_1r_fresh_mini_shadow_2_selection.json", "strata", "claim_id"),
    ("artifacts/v2_1r_mini_shadow_selection.json", "strata", "claim_id"),
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def at_path(value: Any, path: str) -> Any:
    for key in path.split("."):
        value = value[key]
    return value


def members(source: tuple[str, str, str]) -> set[str]:
    path, key_path, id_key = source
    values = at_path(read_json(ROOT / path), key_path)
    if not isinstance(values, list):
        raise ValueError(f"membership_not_list:{path}:{key_path}")
    result = {item if not id_key else item.get(id_key) for item in values}
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"invalid_membership_id:{path}:{key_path}")
    return result


def surface(text: str) -> dict[str, str | bool]:
    n = len(text)
    length = "XS" if n <= 80 else "S" if n <= 180 else "M" if n <= 360 else "L" if n <= 700 else "XL"
    terminals = len(re.findall(r"[.!?。！？]", text))
    separators = len(re.findall(r"[,，;；:：]", text))
    sep_bucket = "0" if separators == 0 else "1" if separators == 1 else "2-3" if separators <= 3 else "4+"
    return {
        "length_bucket": length,
        "sentence_count_bucket": "1" if terminals <= 1 else "2" if terminals == 2 else "3+",
        "clause_separator_count_bucket": sep_bucket,
        "numeric_presence": bool(re.search(r"\d", text)),
        "quotation_presence": bool(re.search(r"[\"'“”‘’「」『』《》〈〉]", text)),
        "parenthetical_presence": bool(re.search(r"[()\[\]{}（）【】［］｛｝]", text)),
        "newline_presence": "\n" in text or "\r" in text,
        "multi_clause_presence": terminals > 1 or separators > 0,
    }


def main() -> None:
    criteria = read_json(CRITERIA)
    criteria_hash = sha_bytes(canonical(criteria))
    pool = read_json(POOL)
    old_audit = read_json(ART / "v2_1r_d4a_exclusion_audit.json")
    d4g_a_audit = read_json(ART / "v2_1r_d4g_a_exclusion_audit.json")
    old_selected = {item["claim_id"] for item in read_json(ART / "v2_1r_d4a_selection_manifest.json")["claims"]}
    d4b_execution_ids = {
        path.stem for path in (ART / "v2_1r_d4b_execution_attempt_2").glob("*.json")
        if path.stem != "aggregate" and path.stem != "preflight"
    }
    if d4b_execution_ids != old_selected:
        raise ValueError("d4b_execution_membership_disagrees_with_d4a_selection")
    d4g_manifest = read_json(ART / "v2_1r_d4g_a_selection_manifest.json")
    d4g_b_execution = read_json(ART / "v2_1r_d4g_b_execution_attempt_1/execution_manifest.json")
    d4g_ids = set(d4g_b_execution["claims_in_order"])
    if d4g_ids != {item["claim_id"] for item in d4g_manifest["claims"]}:
        raise ValueError("d4g_b_membership_disagrees_with_frozen_manifest")

    source_members: dict[str, set[str]] = {path: members(source) for source in MEMBERSHIP_SOURCES for path in [source[0]]}
    prior_union = set().union(*source_members.values())
    if len(prior_union) != d4g_a_audit["prior_membership_identity_union_count"]:
        raise ValueError("authoritative_prior_union_does_not_reconcile_to_d4g_audit")
    if len(prior_union) != 168:
        raise ValueError("unexpected_authoritative_union_size")
    d4b = old_selected
    if len(d4b) != 8:
        raise ValueError("d4b_cohort_size_invalid")
    d4g = d4g_ids
    exclusions = prior_union | d4b | d4g

    records: list[dict[str, Any]] = []
    for pair in pool["pairs"]:
        for side in ("a", "b"):
            claim_id = pair[f"claim_{side}_id"]
            text = pair[f"claim_{side}_text"]
            if not isinstance(claim_id, str) or not isinstance(text, str):
                raise ValueError("candidate_record_invalid")
            records.append({
                "claim_id": claim_id,
                "source_pair_id": pair["pair_id"],
                "claim_sha256": sha_bytes(text.encode("utf-8")),
                "features": surface(text),
            })
    unique_ids = {record["claim_id"] for record in records}
    eligible = [record for record in records if record["claim_id"] not in exclusions]
    eligible_pairs = {record["source_pair_id"] for record in eligible}
    candidate_d4b = unique_ids & d4b
    candidate_d4g = unique_ids & d4g
    candidate_prior = unique_ids & prior_union
    if len(unique_ids) != old_audit["source_pool_unique_claims"]:
        raise ValueError("candidate_pool_unique_count_changed")
    if len(candidate_prior) != old_audit["excluded_from_source_pool"]:
        raise ValueError("candidate_prior_overlap_does_not_reconcile")
    if len(eligible) < COHORT_SIZE or len(eligible_pairs) < COHORT_SIZE:
        raise ValueError("insufficient_fresh_claims_or_pairs")

    chosen: list[dict[str, Any]] = []
    seen: dict[str, set[Any]] = {}
    used_ids: set[str] = set()
    used_pairs: set[str] = set()
    feature_names = tuple(eligible[0]["features"])
    while len(chosen) < COHORT_SIZE:
        options = []
        for item in eligible:
            if item["claim_id"] in used_ids or item["source_pair_id"] in used_pairs:
                continue
            score = sum(item["features"][name] not in seen.get(name, set()) for name in feature_names)
            rank = sha_bytes((f"generic-structural-diversity.v2|{item['source_pair_id']}|{item['claim_id']}|".encode()) + canonical(item["features"]))
            options.append((score, rank, item))
        _, rank, item = min(options, key=lambda option: (-option[0], option[1]))
        chosen.append({**item, "rank_sha256": rank})
        used_ids.add(item["claim_id"])
        used_pairs.add(item["source_pair_id"])
        for name, value in item["features"].items():
            seen.setdefault(name, set()).add(value)

    selected = [
        {"ordinal": i, "claim_id": row["claim_id"], "claim_sha256": row["claim_sha256"],
         "source_pair_id": row["source_pair_id"], "structural_signature": row["features"],
         "rank_sha256": row["rank_sha256"], "development_only": True, "development_only_forever": True}
        for i, row in enumerate(chosen, 1)
    ]
    exclusion_hash = sha_bytes(canonical(sorted(exclusions)))
    pool_hash = sha_bytes(POOL.read_bytes())
    candidate_set_hash = sha_bytes(canonical(sorted((r["source_pair_id"], r["claim_id"], r["claim_sha256"]) for r in records)))
    manifest: dict[str, Any] = {
        "artifact_version": "v2_1r_d4j_a_selection_manifest.v1",
        "canary_id": "v2_1r-d4j-a-fresh-real-development-canary",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "development_only": True, "development_only_forever": True,
        "claim_text_in_manifest": False, "semantic_expected_answers_in_manifest": False,
        "provider_outputs_used_for_selection": False, "human_labels_used_for_selection": False,
        "selection_criteria_path": "artifacts/v2_1r_d4j_a_selection_criteria.json",
        "selection_criteria_sha256": criteria_hash,
        "selection_algorithm_version": criteria["selection_algorithm"],
        "candidate_source": "artifacts/v2_1_dev_extension_v2_candidates.json",
        "candidate_source_sha256": pool_hash, "candidate_set_hash": candidate_set_hash,
        "exclusion_registry_path": "artifacts/v2_1r_d4j_a_exclusion_audit.json",
        "exclusion_registry_sha256": exclusion_hash,
        "cohort_size": COHORT_SIZE, "source_pair_limit": 1, "claims": selected,
        "manifest_hash_normalization": "UTF-8 JSON, recursively sorted keys, compact separators, ensure_ascii=false; normalized_manifest_sha256 omitted while hashing.",
    }
    manifest["normalized_manifest_sha256"] = sha_bytes(canonical(manifest))
    audit = {
        "artifact_version": "v2_1r_d4j_a_exclusion_audit.v1",
        "identity_only": True, "holdout_content_or_labels_read": False,
        "candidate_source": "artifacts/v2_1_dev_extension_v2_candidates.json",
        "candidate_source_sha256": pool_hash, "candidate_unique_claim_count": len(unique_ids),
        "authoritative_membership_sources": [
            {"path": path, "count": len(ids), "sha256": sha_bytes(canonical(sorted(ids)))}
            for path, ids in source_members.items()
        ],
        "prior_authoritative_development_union_count": len(prior_union),
        "prior_union_sha256": sha_bytes(canonical(sorted(prior_union))),
        "d4b_cohort_ids": sorted(d4b), "d4b_overlap_with_candidate_pool": len(candidate_d4b),
        "d4g_b_cohort_ids": d4g_b_execution["claims_in_order"],
        "d4g_manifest_sha256": d4g_manifest["normalized_manifest_sha256"],
        "d4g_b_execution_manifest_sha256": d4g_b_execution["execution_manifest_sha256"],
        "d4g_overlap_with_candidate_pool": len(candidate_d4g),
        "historical_quarantined_holdout_identity_exclusion_count": old_audit["source_pool_frozen_holdout_identity_exclusions"],
        "candidate_pool_overlap_with_frozen_holdout_ids": old_audit["source_pool_frozen_holdout_claim_id_overlap"],
        "historical_union_discrepancy_watch": {"current_authoritative_members": 168, "historical_reported_members": 169, "status": "BOOKKEEPING_WATCH_ONLY"},
        "candidate_pool_overlap_prior_development": len(candidate_prior),
        "eligible_candidate_identity_count": len({r["claim_id"] for r in eligible}),
        "eligible_source_pair_count": len(eligible_pairs),
        "selected_ids": [r["claim_id"] for r in selected],
        "selected_ids_overlap_prior_development": len(used_ids & prior_union),
        "selected_ids_overlap_d4b": len(used_ids & d4b),
        "selected_ids_overlap_d4g_b": len(used_ids & d4g),
        "selected_pair_count": len(used_pairs),
        "selection_used_semantics_provider_outputs_or_labels": False,
        "candidate_identity_records": [
            {
                "claim_id": record["claim_id"],
                "source_pair_id": record["source_pair_id"],
                "membership_evidence": sorted(
                    [path for path, ids in source_members.items() if record["claim_id"] in ids]
                    + (["artifacts/v2_1r_d4a_selection_manifest.json (D4-B selected cohort)"] if record["claim_id"] in d4b else [])
                    + (["artifacts/v2_1r_d4g_a_selection_manifest.json and D4-G-B execution manifest"] if record["claim_id"] in d4g else [])
                    + ["D4-A frozen Holdout identity audit: candidate-pool overlap is zero"]
                ),
                "excluded": record["claim_id"] in exclusions,
                "exclusion_cohorts": sorted(
                    [path for path, ids in source_members.items() if record["claim_id"] in ids]
                    + (["D4-B"] if record["claim_id"] in d4b else [])
                    + (["D4-G-B"] if record["claim_id"] in d4g else [])
                ),
                "selected_for_d4j_a": record["claim_id"] in used_ids,
            }
            for record in records
        ],
    }
    # The exclusion registry hash excludes its own field to avoid a self-reference.
    audit["exclusion_registry_sha256"] = exclusion_hash
    for path, obj in ((AUDIT, audit), (MANIFEST, manifest)):
        path.write_bytes(canonical(obj) + b"\n")
    print(json.dumps({
        "criteria_sha256": criteria_hash, "candidate_pool_unique": len(unique_ids),
        "prior_development_overlap": len(candidate_prior), "d4b_overlap": len(candidate_d4b),
        "d4g_overlap": len(candidate_d4g), "eligible": len({r['claim_id'] for r in eligible}),
        "eligible_pairs": len(eligible_pairs), "selected": [r["claim_id"] for r in selected],
        "manifest_sha256": manifest["normalized_manifest_sha256"], "exclusion_sha256": exclusion_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
