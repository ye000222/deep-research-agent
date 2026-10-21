"""Freeze a generic structural D4-G-A Claim cohort without semantic selection."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
POOL_PATH = ARTIFACTS / "v2_1_dev_extension_v2_candidates.json"
CRITERIA_PATH = ARTIFACTS / "v2_1r_d4g_a_selection_criteria.json"
MANIFEST_PATH = ARTIFACTS / "v2_1r_d4g_a_selection_manifest.json"
AUDIT_PATH = ARTIFACTS / "v2_1r_d4g_a_exclusion_audit.json"
COHORT_SIZE = 8

MEMBERSHIP_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json",
        "claim_replays",
        "claim_id",
    ),
    ("artifacts/v2_1r_source_context_extraction_40_eval_manifest.json", "claims", "claim_id"),
    ("artifacts/v2_1r_source_context_extraction_real_canary_manifest.json", "claim_ids", ""),
    (
        "artifacts/v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_selection_manifest.json",
        "selection.selected_claim_ids",
        "",
    ),
    ("artifacts/v2_1r_bounded_shadow_100_selection_manifest.json", "selected_claims", "claim_id"),
    (
        "artifacts/v2_1r_qualifier_semantic_preservation_fresh_validation_12_selection_manifest.json",
        "selected_claims",
        "claim_id",
    ),
    (
        "artifacts/v2_1r_final_fresh_qualifier_check_selection_manifest.json",
        "selected_claims",
        "claim_id",
    ),
    (
        "artifacts/v2_1r_final_fresh_mini_shadow_3/pre_provider_manifest.json",
        "selected_claim_ids",
        "",
    ),
    (
        "artifacts/v2_1r_final_fresh_mini_shadow_4/mini_shadow_4_execution_summary.json",
        "claims",
        "claim_id",
    ),
    ("artifacts/v2_1r_fresh_mini_shadow_2_selection.json", "strata", "claim_id"),
    ("artifacts/v2_1r_mini_shadow_selection.json", "strata", "claim_id"),
)
D4A_MANIFEST = "artifacts/v2_1r_d4a_selection_manifest.json"
D4A_AUDIT = "artifacts/v2_1r_d4a_exclusion_audit.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _at_path(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = value[part]
    return value


def _selected_membership(source: tuple[str, str, str]) -> set[str]:
    relative_path, array_path, id_key = source
    values = _at_path(_read_json(ROOT / relative_path), array_path)
    if not isinstance(values, list):
        raise ValueError(f"membership_source_not_list:{relative_path}:{array_path}")
    result: set[str] = set()
    for item in values:
        claim_id = item if not id_key else item.get(id_key)
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError(f"invalid_claim_identity:{relative_path}:{array_path}")
        result.add(claim_id)
    return result


def _surface(text: str) -> dict[str, str | bool]:
    length = len(text)
    length_bucket = (
        "XS"
        if length <= 80
        else "S"
        if length <= 180
        else "M"
        if length <= 360
        else "L"
        if length <= 700
        else "XL"
    )
    terminal_count = len(re.findall(r"[.!?。！？]", text))  # noqa: RUF001
    sentence_bucket = "1" if terminal_count <= 1 else "2" if terminal_count == 2 else "3+"
    separator_count = len(re.findall(r"[,，;；:：]", text))  # noqa: RUF001
    separator_bucket = (
        "0"
        if separator_count == 0
        else "1"
        if separator_count == 1
        else "2-3"
        if separator_count <= 3
        else "4+"
    )
    return {
        "length_bucket": length_bucket,
        "sentence_count_bucket": sentence_bucket,
        "clause_separator_count_bucket": separator_bucket,
        "numeric_presence": bool(re.search(r"\d", text)),
        "quotation_presence": bool(
            re.search(r"[\"'“”‘’「」『』《》〈〉]", text)  # noqa: RUF001
        ),
        "parenthetical_presence": bool(
            re.search(r"[()\[\]{}（）【】［］｛｝]", text)  # noqa: RUF001
        ),
        "newline_presence": "\n" in text or "\r" in text,
        "multi_clause_presence": terminal_count > 1 or separator_count > 0,
    }


def _rank(pair_id: str, claim_id: str, features: dict[str, Any]) -> str:
    payload = f"generic-structural-diversity.v2|{pair_id}|{claim_id}|".encode()
    return _sha256_bytes(payload + _canonical(features))


def _membership_report() -> tuple[set[str], dict[str, int]]:
    union: set[str] = set()
    counts: dict[str, int] = {}
    for source in MEMBERSHIP_SOURCES:
        ids = _selected_membership(source)
        name = source[0]
        counts[name] = len(ids)
        union.update(ids)
    return union, counts


def main() -> None:
    criteria = _read_json(CRITERIA_PATH)
    criteria_hash = _sha256_bytes(_canonical(criteria))
    pool = _read_json(POOL_PATH)
    d4a_manifest = _read_json(ROOT / D4A_MANIFEST)
    d4a_audit = _read_json(ROOT / D4A_AUDIT)
    prior_ids, source_counts = _membership_report()
    d4a_ids = {item["claim_id"] for item in d4a_manifest["claims"]}
    full_exclusion_ids = prior_ids | d4a_ids

    candidate_records: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pool["pairs"]:
        pair_id = pair["pair_id"]
        for side in ("a", "b"):
            claim_id = pair[f"claim_{side}_id"]
            claim_text = pair[f"claim_{side}_text"]
            if not isinstance(claim_id, str) or not isinstance(claim_text, str):
                raise ValueError("candidate_record_invalid")
            record = {
                "claim_id": claim_id,
                "source_pair_id": pair_id,
                "claim_sha256": _sha256_bytes(claim_text.encode("utf-8")),
                "features": _surface(claim_text),
            }
            candidate_records[(pair_id, claim_id)] = record

    unique_candidate_ids = {claim_id for _, claim_id in candidate_records}
    pre_d4a_overlap = unique_candidate_ids & prior_ids
    post_d4a_overlap = unique_candidate_ids & full_exclusion_ids
    eligible = [
        record
        for record in candidate_records.values()
        if record["claim_id"] not in full_exclusion_ids
    ]
    eligible_pairs = {record["source_pair_id"] for record in eligible}

    # Assert the already-frozen D4-A audit's source-pool accounting. The selected
    # D4-A cohort is then added as a distinct exclusion for this new D4-G cohort.
    if len(unique_candidate_ids) != d4a_audit["source_pool_unique_claims"]:
        raise ValueError("source_pool_identity_count_mismatch")
    if len(pre_d4a_overlap) != d4a_audit["excluded_from_source_pool"]:
        raise ValueError("frozen_exclusion_audit_candidate_overlap_mismatch")
    if len(unique_candidate_ids - prior_ids) != d4a_audit["eligible_after_exclusions"]:
        raise ValueError("frozen_exclusion_audit_eligible_count_mismatch")
    if len(eligible) < COHORT_SIZE or len(eligible_pairs) < COHORT_SIZE:
        raise ValueError("insufficient_fresh_claims_or_source_pairs")

    selected: list[dict[str, Any]] = []
    seen_values: dict[str, set[Any]] = {}
    used_claim_ids: set[str] = set()
    used_pairs: set[str] = set()
    feature_names = tuple(eligible[0]["features"].keys())
    while len(selected) < COHORT_SIZE:
        choices: list[tuple[int, str, dict[str, Any]]] = []
        for record in eligible:
            if record["claim_id"] in used_claim_ids or record["source_pair_id"] in used_pairs:
                continue
            score = sum(
                record["features"][name] not in seen_values.get(name, set())
                for name in feature_names
            )
            choices.append(
                (
                    score,
                    _rank(record["source_pair_id"], record["claim_id"], record["features"]),
                    record,
                )
            )
        if not choices:
            raise ValueError("greedy_selection_exhausted")
        _, rank, chosen = min(choices, key=lambda item: (-item[0], item[1]))
        selected.append({**chosen, "rank_sha256": rank})
        used_claim_ids.add(chosen["claim_id"])
        used_pairs.add(chosen["source_pair_id"])
        for name, value in chosen["features"].items():
            seen_values.setdefault(name, set()).add(value)

    candidate_set_payload = sorted(
        {
            (record["source_pair_id"], record["claim_id"], record["claim_sha256"])
            for record in candidate_records.values()
        }
    )
    candidate_set_hash = _sha256_bytes(_canonical(candidate_set_payload))
    exclusion_set_payload = sorted(full_exclusion_ids)
    exclusion_set_hash = _sha256_bytes(_canonical(exclusion_set_payload))
    selected_claims = [
        {
            "ordinal": ordinal,
            "claim_id": record["claim_id"],
            "claim_sha256": record["claim_sha256"],
            "source_pair_id": record["source_pair_id"],
            "structural_signature": record["features"],
            "rank_sha256": record["rank_sha256"],
            "development_only": True,
        }
        for ordinal, record in enumerate(selected, start=1)
    ]
    manifest: dict[str, Any] = {
        "artifact_version": "v2_1r_d4g_a_selection_manifest.v1",
        "canary_id": "v2_1r-d4g-a-fresh-real-development-canary",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "development_only_forever": True,
        "claim_text_in_manifest": False,
        "semantic_expected_answers_in_manifest": False,
        "provider_outputs_used_for_selection": False,
        "human_labels_used_for_selection": False,
        "selection_criteria_path": "artifacts/v2_1r_d4g_a_selection_criteria.json",
        "selection_criteria_sha256": criteria_hash,
        "selection_algorithm_version": criteria["selection_algorithm"],
        "candidate_source": "artifacts/v2_1_dev_extension_v2_candidates.json",
        "candidate_source_sha256": _sha256_bytes(POOL_PATH.read_bytes()),
        "candidate_set_hash": candidate_set_hash,
        "exclusion_set_hash": exclusion_set_hash,
        "exclusion_audit_path": "artifacts/v2_1r_d4g_a_exclusion_audit.json",
        "cohort_size": COHORT_SIZE,
        "source_pair_limit": 1,
        "claims": selected_claims,
        "manifest_hash_normalization": (
            "UTF-8 JSON; recursively sorted keys; compact separators; ensure_ascii=false; "
            "normalized_manifest_sha256 omitted during hashing."
        ),
    }
    manifest["normalized_manifest_sha256"] = _sha256_bytes(_canonical(manifest))

    old_audit = d4a_audit
    audit = {
        "artifact_version": "v2_1r_d4g_a_exclusion_audit.v1",
        "identity_only": True,
        "holdout_content_or_labels_read": False,
        "candidate_source": "artifacts/v2_1_dev_extension_v2_candidates.json",
        "candidate_source_sha256": _sha256_bytes(POOL_PATH.read_bytes()),
        "candidate_unique_claim_count": len(unique_candidate_ids),
        "prior_membership_sources": [
            {"path": source[0], "selected_membership_count": source_counts[source[0]]}
            for source in MEMBERSHIP_SOURCES
        ],
        "prior_membership_identity_union_count": len(prior_ids),
        "prior_membership_union_sha256": _sha256_bytes(_canonical(sorted(prior_ids))),
        "d4a_selected_claim_count": len(d4a_ids),
        "d4a_selected_claim_ids": sorted(d4a_ids),
        "d4a_audit_claimed_prior_exclusion_union_count": old_audit["excluded_identity_union_count"],
        "explicit_membership_union_count_delta_vs_d4a_audit": len(prior_ids)
        - old_audit["excluded_identity_union_count"],
        "prior_development_overlap_with_candidate_pool": len(pre_d4a_overlap),
        "d4a_audit_prior_pool_exclusions": old_audit["excluded_from_source_pool"],
        "eligible_after_prior_cohorts": len(unique_candidate_ids - prior_ids),
        "overlap_with_d4a_selected_claims": len(unique_candidate_ids & d4a_ids),
        "total_candidate_exclusions_including_d4a": len(post_d4a_overlap),
        "eligible_after_all_development_exclusions": len(
            {record["claim_id"] for record in eligible}
        ),
        "eligible_source_pair_count": len(eligible_pairs),
        "selected_ids_overlap_prior_membership": len(used_claim_ids & prior_ids),
        "selected_ids_overlap_d4a": len(used_claim_ids & d4a_ids),
        "selected_source_pair_count": len(used_pairs),
        "source_pool_frozen_holdout_exclusion_evidence": {
            "source_audit_path": "artifacts/v2_1r_d4a_exclusion_audit.json",
            "source_audit_sha256": _sha256_bytes(
                (ARTIFACTS / "v2_1r_d4a_exclusion_audit.json").read_bytes()
            ),
            "frozen_holdout_identity_exclusions_reported": old_audit[
                "source_pool_frozen_holdout_identity_exclusions"
            ],
            "candidate_pool_holdout_id_overlap_reported": old_audit[
                "source_pool_frozen_holdout_claim_id_overlap"
            ],
            "quarantined_or_frozen_holdout_content_opened": False,
        },
        "selection_used_provider_outputs_or_human_labels": False,
        "selection_claim_ids": [record["claim_id"] for record in selected_claims],
    }

    MANIFEST_PATH.write_bytes(_canonical(manifest) + b"\n")
    AUDIT_PATH.write_bytes(_canonical(audit) + b"\n")
    print(
        json.dumps(
            {
                "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
                "manifest_sha256": manifest["normalized_manifest_sha256"],
                "criteria_sha256": criteria_hash,
                "candidate_unique_claim_count": len(unique_candidate_ids),
                "prior_overlap": len(pre_d4a_overlap),
                "eligible_after_prior": len(unique_candidate_ids - prior_ids),
                "d4a_overlap": len(unique_candidate_ids & d4a_ids),
                "eligible_after_all": len({record["claim_id"] for record in eligible}),
                "eligible_source_pairs": len(eligible_pairs),
                "selected_count": len(selected_claims),
                "selected_pair_count": len(used_pairs),
                "holdout_content_or_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
