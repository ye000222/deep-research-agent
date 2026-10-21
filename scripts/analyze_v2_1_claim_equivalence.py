#!/usr/bin/env python3
"""Read-only, scoped V2.1 semantic Claim replay and artifact generator."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import psycopg
from app.domain.claim_architecture import CanonicalClaim, CanonicalClaimType, ClaimRelationType
from app.domain.semantic_claim_equivalence import MATCHER_VERSION, SemanticClaimMatcher

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BASELINE_PATH = ARTIFACTS / "v2_0_claim_baseline.json"
METRIC_VERSION = "claim-equivalence-v2-v1"
MAX_PAIRS_PER_SCOPE = 10_000
EVAL_DATASET_VERSION = "v2_1_claim_eval_holdout_v1"
DEVELOPMENT_SAMPLE_SIZE = 45
HOLDOUT_SAMPLE_SIZE = 25


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _claim_qualifiers(text: str) -> dict[str, str]:
    qualifiers: dict[str, str] = {}
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    versions = re.findall(r"\bv\s?\d+(?:\.\d+)*\b", text, flags=re.IGNORECASE)
    if years:
        qualifiers["time"] = years[-1]
    if versions:
        qualifiers["version"] = versions[-1].replace(" ", "")
    if re.search(r"\b(no|not|never|without|neither)\b", text, flags=re.IGNORECASE):
        qualifiers["_polarity_observed"] = "negative"
    return qualifiers


def _to_claim(row: dict[str, Any]) -> CanonicalClaim:
    claim_text = str(row["atomic_claim"])
    qualifiers = _claim_qualifiers(claim_text)
    polarity = None
    if "_polarity_observed" in qualifiers:
        qualifiers.pop("_polarity_observed")
        polarity = False
    claim_type_raw = str(row.get("claim_type") or "unknown").casefold()
    try:
        claim_type = CanonicalClaimType(claim_type_raw)
    except ValueError:
        claim_type = CanonicalClaimType.UNKNOWN
    return CanonicalClaim.from_text(
        claim_id=UUID(str(row["claim_id"])),
        run_id=UUID(str(row["run_id"])),
        question_id=str(row["question_id"]),
        dimension_key=str(row["dimension_key"]),
        original_text=claim_text,
        claim_type=claim_type,
        qualifiers=qualifiers,
        polarity=polarity,
    )


def load_historical_claims(database_url: str, run_ids: list[str]) -> list[dict[str, Any]]:
    """Read only the V2.0 baseline runs, with DB writes disabled at transaction level."""

    url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(url, options="-c default_transaction_read_only=on") as conn:
        rows = conn.execute(
            """
            SELECT c.id::text AS claim_id, c.run_id::text AS run_id,
                   c.question_id, c.dimension_key, c.atomic_claim, c.claim_type,
                   count(e.id) FILTER (
                     WHERE e.accepted AND e.relation = 'supports'
                   ) AS accepted_support_evidence_count,
                   coalesce(
                     array_agg(DISTINCT s.source_owner_key) FILTER (
                       WHERE e.accepted AND e.relation = 'supports'
                         AND s.source_owner_key IS NOT NULL
                     ), ARRAY[]::text[]
                   ) AS owner_keys
            FROM research_claims c
            LEFT JOIN research_evidence e
              ON e.claim_id = c.id AND e.run_id = c.run_id
            LEFT JOIN research_sources s ON s.id = e.source_id
            WHERE c.run_id = ANY(%s::uuid[])
            GROUP BY c.id, c.run_id, c.question_id, c.dimension_key,
                     c.atomic_claim, c.claim_type
            ORDER BY c.run_id, c.question_id, c.dimension_key, c.id
            """,
            (run_ids,),
        ).fetchall()
    names = (
        "claim_id", "run_id", "question_id", "dimension_key", "atomic_claim",
        "claim_type", "accepted_support_evidence_count", "owner_keys",
    )
    return [dict(zip(names, row, strict=True)) for row in rows]


def scoped_pairs(claim_rows: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], int, int, int]:
    scopes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(claim_rows):
        scope = (str(row["run_id"]), str(row["question_id"]), str(row["dimension_key"]))
        scopes[scope].append(index)
    global_pairs = len(claim_rows) * (len(claim_rows) - 1) // 2
    scoped_pair_count = sum(len(indices) * (len(indices) - 1) // 2 for indices in scopes.values())
    pairs: list[tuple[int, int]] = []
    for indices in scopes.values():
        group_pairs = len(indices) * (len(indices) - 1) // 2
        if group_pairs > MAX_PAIRS_PER_SCOPE:
            # Explicitly cap pathological individual scopes rather than creating an
            # unbounded Cartesian comparison workload.
            pairs.extend(itertools.islice(itertools.combinations(indices, 2), MAX_PAIRS_PER_SCOPE))
        else:
            pairs.extend(itertools.combinations(indices, 2))
    return (
        pairs,
        global_pairs - scoped_pair_count,
        max(0, scoped_pair_count - len(pairs)),
        scoped_pair_count,
    )


def compare_history(claim_rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [_to_claim(row) for row in claim_rows]
    pairs, cross_scope_filtered, cap_filtered, scoped_pair_count = scoped_pairs(claim_rows)
    matcher = SemanticClaimMatcher()
    results: list[dict[str, Any]] = []
    edge_by_ids: dict[tuple[str, str], tuple[str, float]] = {}
    relation_counts: Counter[str] = Counter()
    for left_index, right_index in pairs:
        comparison = matcher.compare(claims[left_index], claims[right_index])
        relation_counts[comparison.relation.value] += 1
        record = {
            "left_claim_id": comparison.left_claim_id,
            "right_claim_id": comparison.right_claim_id,
            "run_id": str(claims[left_index].run_id),
            "question_id": claims[left_index].question_id,
            "dimension_key": claims[left_index].dimension_key,
            "left_text": claims[left_index].original_text,
            "right_text": claims[right_index].original_text,
            "lexical_overlap_jaccard": _lexical_overlap(
                claims[left_index].original_text, claims[right_index].original_text
            ),
            **comparison.model_dump(mode="json", exclude={"left_claim_id", "right_claim_id"}),
        }
        results.append(record)
        key = (
            min(comparison.left_claim_id, comparison.right_claim_id),
            max(comparison.left_claim_id, comparison.right_claim_id),
        )
        edge_by_ids[key] = (comparison.relation.value, comparison.confidence)

    grouped_claim_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, claim in enumerate(claims):
        scope = (str(claim.run_id), claim.question_id, claim.dimension_key)
        grouped_claim_indices[scope].append(index)

    def high_confidence_same(left_index: int, right_index: int) -> bool:
        left_claim_id = str(claims[left_index].claim_id)
        right_claim_id = str(claims[right_index].claim_id)
        key = (min(left_claim_id, right_claim_id), max(left_claim_id, right_claim_id))
        edge = edge_by_ids.get(key)
        return edge is not None and edge[0] == ClaimRelationType.SAME.value and edge[1] >= 0.90

    clusters: list[dict[str, Any]] = []
    for scope, indices in sorted(grouped_claim_indices.items()):
        scope_clusters: list[list[int]] = []
        for index in indices:
            for cluster in scope_clusters:
                if all(high_confidence_same(index, member) for member in cluster):
                    cluster.append(index)
                    break
            else:
                scope_clusters.append([index])
        for cluster in scope_clusters:
            if len(cluster) < 2:
                continue
            rows = [claim_rows[index] for index in cluster]
            evidence_count = sum(
                int(row.get("accepted_support_evidence_count") or 0) for row in rows
            )
            owners = sorted({str(owner) for row in rows for owner in (row.get("owner_keys") or [])})
            clusters.append({
                "run_id": scope[0], "question_id": scope[1], "dimension_key": scope[2],
                "claim_ids": [str(claims[index].claim_id) for index in cluster],
                "claim_texts": [claims[index].original_text for index in cluster],
                "cluster_method": "complete_link_high_confidence_same",
                "accepted_support_evidence_count": evidence_count,
                "independent_owner_count": len(owners),
                "owner_keys": owners,
            })

    return {
        "claim_count": len(claims),
        "global_unscoped_pair_count_avoided": len(claims) * (len(claims) - 1) // 2,
        "candidate_pair_count": scoped_pair_count,
        "pairs_filtered_cross_run_question_dimension": cross_scope_filtered,
        "pairs_filtered_scope_cap": cap_filtered,
        "pairs_sent_to_deterministic_matcher": len(pairs),
        "pairs_skipped_by_pair_cap": cap_filtered,
        "semantic_model_calls": 0,
        "relation_counts": {
            relation.value: int(relation_counts[relation.value])
            for relation in ClaimRelationType
        },
        "clusters": clusters,
        "semantic_cluster_count": len(clusters),
        "cluster_size_distribution": dict(
            sorted(Counter(len(cluster["claim_ids"]) for cluster in clusters).items())
        ),
        "potential_multi_evidence_cluster_count": sum(
            1 for cluster in clusters if cluster["accepted_support_evidence_count"] >= 2
        ),
        "potential_multi_owner_cluster_count": sum(
            1 for cluster in clusters if cluster["independent_owner_count"] >= 2
        ),
        "claims_participating_in_multiple_clusters": 0,
        "pair_results": results,
        "unknown_reason_counts": unknown_reason_counts(results),
    }


def unknown_reason_counts(pair_results: list[dict[str, Any]]) -> dict[str, int]:
    """Return overlapping bounded diagnostics, not semantic labels."""

    counts: Counter[str] = Counter()
    for pair in pair_results:
        if pair.get("relation") != ClaimRelationType.UNKNOWN.value:
            continue
        signals = pair.get("signals") or {}
        reasons: set[str] = set()
        reason_codes = set(pair.get("reason_codes") or [])
        if "SEMANTIC_MODEL_UNCERTAIN" in reason_codes:
            reasons.add("SEMANTIC_MODEL_UNCERTAIN")
        if pair.get("qualifier_compatibility") == "incompatible":
            reasons.add("HARD_GUARD_INCOMPATIBLE")
        if pair.get("qualifier_compatibility") == "incomplete":
            reasons.add("QUALIFIER_AMBIGUOUS")
        if signals.get("subject_equal") is None:
            reasons.add("ENTITY_UNRESOLVED")
        if signals.get("predicate_equal") is None:
            reasons.add("PREDICATE_UNRESOLVED")
        if signals.get("numeric_values_equal") is None:
            reasons.add("VALUE_UNRESOLVED")
        if all(
            signals.get(key) is None
            for key in ("subject_equal", "predicate_equal", "numeric_values_equal")
        ):
            reasons.add("MISSING_STRUCTURED_FIELDS")
        if _lexical_overlap(str(pair.get("left_text", "")), str(pair.get("right_text", ""))) < 0.05:
            reasons.add("INSUFFICIENT_TEXT_SEMANTICS")
        if not reasons:
            reasons.add("OTHER")
        counts.update(reasons)
    return dict(sorted(counts.items()))


def make_manual_sample(pair_results: list[dict[str, Any]], limit: int = 40) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in pair_results:
        buckets[result["relation"]].append(result)
    rng = random.Random(2101)
    for items in buckets.values():
        rng.shuffle(items)
        items.sort(key=lambda item: float(item.get("lexical_overlap_jaccard") or 0), reverse=True)
    selected: list[dict[str, Any]] = []
    # Round-robin by relation prevents UNKNOWN from crowding out rarer outcomes.
    relations = ("same", "related", "contradicts", "unknown")
    while len(selected) < min(limit, len(pair_results)):
        progressed = False
        for relation in relations:
            if buckets[relation] and len(selected) < limit:
                item = buckets[relation].pop(0)
                selected.append(
                    {**item, "review_status": "pending_human_review", "review_findings": []}
                )
                progressed = True
        if not progressed:
            break
    return {
        "schema_version": "v2.1-semantic-audit-sample.v1",
        "matcher_version": MATCHER_VERSION,
        "sample_limit": limit,
        "sample_count": len(selected),
        "review_status": (
            "pending_independent_human_review"
            if selected
            else "not_available_no_historical_pairs"
        ),
        "selection_note": (
            "Stratified by relation and prioritized by lexical overlap to surface likely false "
            "splits. Overlap is a sampling aid, never an equivalence decision."
        ),
        "manual_audit_findings": {
            "false_merge": "NOT_REVIEWED",
            "false_split": "NOT_REVIEWED",
            "qualifier_loss": "NOT_REVIEWED",
            "entity_mismatch": "NOT_REVIEWED",
            "polarity_error": "NOT_REVIEWED",
        },
        "comparisons": selected,
    }


def _lexical_overlap(left: str, right: str) -> float:
    """Cheap review-sample prioritizer only; never used as a SAME decision."""

    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
        "的", "了", "在", "并", "以及", "该", "其", "一个", "通过", "表示", "指出",
    }

    def tokens(value: str) -> set[str]:
        normalized = value.casefold()
        word_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9._%-]*", normalized)
            if len(token) > 1 and token not in stopwords
        }
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        cjk_tokens = {cjk[index:index + 2] for index in range(len(cjk) - 1)}
        return word_tokens | cjk_tokens

    left_tokens = tokens(left)
    right_tokens = tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _eval_pair_id(pair: dict[str, Any]) -> str:
    material = "|".join(
        sorted((str(pair["left_claim_id"]), str(pair["right_claim_id"])))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _sampling_categories(pair: dict[str, Any]) -> set[str]:
    """Generic surface-form strata for review sampling only, never for matching."""

    left = str(pair.get("left_text") or "")
    right = str(pair.get("right_text") or "")
    categories: set[str] = set()
    overlap = float(pair.get("lexical_overlap_jaccard") or 0.0)
    if overlap >= 0.35:
        categories.add("lexical_near_match")
    left_numbers = re.findall(r"(?<![\w])\d+(?:[.,]\d+)?", left)
    right_numbers = re.findall(r"(?<![\w])\d+(?:[.,]\d+)?", right)
    if left_numbers and right_numbers:
        categories.add("numeric_claim_pair")
    left_years = set(re.findall(r"\b(?:19|20)\d{2}\b", left))
    right_years = set(re.findall(r"\b(?:19|20)\d{2}\b", right))
    if left_years and right_years and left_years != right_years:
        categories.add("different_time_tokens")
    left_versions = set(re.findall(r"\bv\s?\d+(?:\.\d+)*\b", left, flags=re.IGNORECASE))
    right_versions = set(re.findall(r"\bv\s?\d+(?:\.\d+)*\b", right, flags=re.IGNORECASE))
    if left_versions and right_versions and left_versions != right_versions:
        categories.add("different_version_tokens")
    negation = re.compile(r"\b(?:no|not|never|without|neither)\b|不|无|未|没有", re.I)
    if bool(negation.search(left)) != bool(negation.search(right)):
        categories.add("possible_polarity_pair")
    if overlap <= 0.10:
        categories.add("low_lexical_overlap_pair")
    if not categories:
        categories.add("remainder")
    return categories


def make_real_claim_eval_dataset(
    pair_results: list[dict[str, Any]],
    *,
    excluded_pair_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_ids = excluded_pair_ids or set()
    available_pair_ids = {_eval_pair_id(pair) for pair in pair_results}
    excluded_available_ids = excluded_ids & available_pair_ids
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_results:
        if _eval_pair_id(pair) in excluded_ids:
            continue
        for category in _sampling_categories(pair):
            queues[category].append(pair)
    categories = sorted(queues)
    for _category, items in queues.items():
        items.sort(key=lambda item: _eval_pair_id(item))

    selected: dict[str, dict[str, Any]] = {}
    target = DEVELOPMENT_SAMPLE_SIZE + HOLDOUT_SAMPLE_SIZE
    eligible_pair_count = len(available_pair_ids - excluded_available_ids)
    while len(selected) < min(target, eligible_pair_count):
        advanced = False
        for category in categories:
            while queues[category]:
                pair = queues[category].pop(0)
                pair_id = _eval_pair_id(pair)
                if pair_id in selected:
                    continue
                selected[pair_id] = pair
                advanced = True
                break
            if len(selected) >= target:
                break
        if not advanced:
            break

    selected_ids = sorted(
        selected,
        key=lambda pair_id: hashlib.sha256(
            f"{EVAL_DATASET_VERSION}:{pair_id}".encode()
        ).hexdigest(),
    )
    holdout_ids = selected_ids[: min(HOLDOUT_SAMPLE_SIZE, len(selected_ids))]
    holdout_set = set(holdout_ids)
    development_ids = [pair_id for pair_id in selected_ids if pair_id not in holdout_set]

    def review_item(pair_id: str) -> dict[str, Any]:
        pair = selected[pair_id]
        return {
            "pair_id": pair_id,
            "run_id": pair["run_id"],
            "question_id": pair["question_id"],
            "dimension_key": pair["dimension_key"],
            "claim_a_id": pair["left_claim_id"],
            "claim_a": pair["left_text"],
            "claim_b_id": pair["right_claim_id"],
            "claim_b": pair["right_text"],
            "surface_sampling_strata": sorted(_sampling_categories(pair)),
            "human_label": None,
            "review_rationale": "",
        }

    review_items = [review_item(pair_id) for pair_id in selected_ids]
    holdout_hash = hashlib.sha256(
        "\n".join(sorted(holdout_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "v2.1-real-claim-eval-candidates.v1",
        "dataset_version": EVAL_DATASET_VERSION,
        "source": "read-only historical V2.0 Claim pairs",
        "sampling_method": "deterministic round-robin across generic text-feature strata",
        "sampling_features_are_labels": False,
        "prior_prediction_exposure_excluded_from_both_splits": len(excluded_available_ids),
        "excluded_prior_prediction_exposure_pair_ids": sorted(excluded_available_ids),
        "sample_count": len(review_items),
        "target_development_count": DEVELOPMENT_SAMPLE_SIZE,
        "target_holdout_count": HOLDOUT_SAMPLE_SIZE,
        "development_ids": development_ids,
        "holdout_ids": holdout_ids,
        "holdout_membership_sha256": holdout_hash,
        "holdout_freeze_status": "FROZEN_MEMBERSHIP_AWAITING_HUMAN_LABELS",
        "label_status": "AWAITING_HUMAN_LABELS",
        "review_view": {
            "matcher_predictions_included": False,
            "items": review_items,
        },
        "matcher_predictions_separate": {
            item["pair_id"]: {
                "matcher_prediction": selected[item["pair_id"]]["relation"],
                "confidence": selected[item["pair_id"]]["confidence"],
                "reason_codes": selected[item["pair_id"]]["reason_codes"],
            }
            for item in review_items
        },
    }


def evaluation_waiting_artifact(
    dataset: dict[str, Any], split: str
) -> dict[str, Any]:
    pair_ids = dataset[f"{split}_ids"]
    return {
        "schema_version": "v2.1-labeled-evaluation.v1",
        "dataset_version": dataset["dataset_version"],
        "split": split,
        "candidate_count": len(pair_ids),
        "label_status": "AWAITING_HUMAN_LABELS",
        "metrics": {
            "same_precision": None,
            "same_recall": None,
            "false_same_count": None,
            "related_agreement": None,
            "contradicts_agreement": None,
            "unknown_rate": None,
        },
        "holdout_membership_sha256": (
            dataset["holdout_membership_sha256"] if split == "holdout" else None
        ),
        "holdout_used_for_tuning": False if split == "holdout" else None,
    }


def evaluation_waiting_markdown(evaluation: dict[str, Any]) -> str:
    lines = [
        f"# V2.1 {evaluation['split'].title()} Evaluation", "",
        f"- Dataset: `{evaluation['dataset_version']}`",
        f"- Candidates: {evaluation['candidate_count']}",
        f"- Label status: **{evaluation['label_status']}**", "",
        "No accuracy or agreement metrics are computed without independent human labels.",
        "",
    ]
    return "\n".join(lines)


def real_claim_eval_markdown(dataset: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Real Claim Evaluation Candidates", "",
        f"- Dataset version: `{dataset['dataset_version']}`",
        f"- Development pairs: {len(dataset['development_ids'])}",
        f"- Frozen holdout pairs: {len(dataset['holdout_ids'])}",
        f"- Holdout membership SHA-256: `{dataset['holdout_membership_sha256']}`",
        "- Previously prediction-exposed pairs excluded from both splits: "
        f"{dataset['prior_prediction_exposure_excluded_from_both_splits']}",
        f"- Label status: **{dataset['label_status']}**", "",
        "This is a blind labeling view: matcher predictions are intentionally omitted.",
        "Surface-feature strata are sampling aids only and are not labels.", "",
    ]
    for index, item in enumerate(dataset["review_view"]["items"], 1):
        lines.extend([
            f"## {index}. {item['pair_id']} — {item['question_id']} / {item['dimension_key']}",
            f"- Sampling strata: {', '.join(item['surface_sampling_strata'])}",
            f"- Claim A: {item['claim_a']}", f"- Claim B: {item['claim_b']}",
            "- Human label (SAME / RELATED / CONTRADICTS / UNKNOWN):",
            "- Review rationale:", "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def generalization_audit() -> dict[str, Any]:
    mechanisms = [
        {
            "mechanism": "Unicode, whitespace, case and punctuation normalization",
            "motivating_failure_category": "formatting_variance",
            "domain_independent_rationale": (
                "Equivalent presentation forms occur across languages and research domains."
            ),
            "affected_claim_types": "all",
            "false_positive_risk": (
                "Case or punctuation can be material in identifiers; "
                "entity/value guards run separately."
            ),
            "tests": [
                "unicode_and_punctuation_normalization",
                "negation_and_qualifier_preservation",
            ],
            "derived_from_development_data": False,
            "holdout_influenced": False,
        },
        {
            "mechanism": "Structured exact subject/predicate/value comparison",
            "motivating_failure_category": "semantic_identity_without_text_identity",
            "domain_independent_rationale": (
                "Canonical proposition fields have the same interpretation regardless of topic."
            ),
            "affected_claim_types": "all structured claims",
            "false_positive_risk": (
                "Incorrect upstream field extraction; conservative qualifier checks limit merging."
            ),
            "tests": ["structured_same", "entity_value_and_predicate_mismatch"],
            "derived_from_development_data": False,
            "holdout_influenced": False,
        },
        {
            "mechanism": "Qualifier compatibility guards",
            "motivating_failure_category": "material_scope_conflict",
            "domain_independent_rationale": (
                "Time, location, scope, condition, method and version "
                "qualify propositions across domains."
            ),
            "affected_claim_types": "all",
            "false_positive_risk": (
                "Qualifier key/value normalization may be incomplete; unknown stays non-SAME."
            ),
            "tests": [
                "different_year",
                "different_version",
                "scope_location_condition_mismatch",
            ],
            "derived_from_development_data": False,
            "holdout_influenced": False,
        },
        {
            "mechanism": "Exact numeric magnitude and safe unit-symbol aliases",
            "motivating_failure_category": "formatting_and_scale_variance",
            "domain_independent_rationale": (
                "Decimal magnitude equivalence is generic; no currency or contextual "
                "conversion is attempted."
            ),
            "affected_claim_types": "quantitative",
            "false_positive_risk": (
                "Ambiguous unit abbreviations; unit disagreement is not SAME."
            ),
            "tests": [
                "equivalent_numeric_magnitude",
                "incompatible_units",
                "approximate_value",
            ],
            "derived_from_development_data": False,
            "holdout_influenced": False,
        },
        {
            "mechanism": "Generic structured LLM fallback with strict schema and call cap",
            "motivating_failure_category": "unstructured_paraphrase",
            "domain_independent_rationale": (
                "It compares proposition semantics from only the pair and their qualifiers, "
                "without topic templates."
            ),
            "affected_claim_types": "all",
            "false_positive_risk": (
                "Model misclassification; known guard conflicts bypass it, invalid output "
                "becomes UNKNOWN, and it is not runtime-wired."
            ),
            "tests": [
                "paraphrase_stub_response",
                "known_conflict_blocks_fallback",
                "malformed_response_fails_unknown",
                "fallback_call_cap",
            ],
            "derived_from_development_data": False,
            "holdout_influenced": False,
            "prompt_version": "semantic-claim-equivalence.generic.v1",
            "actual_model_calls": 0,
        },
        {
            "mechanism": "Run/question/dimension pair blocking and complete-link clustering",
            "motivating_failure_category": "pair_explosion_and_transitive_merge_risk",
            "domain_independent_rationale": (
                "Research scope metadata is independent of benchmark/task identity; all "
                "cluster edges must be high-confidence SAME."
            ),
            "affected_claim_types": "all",
            "false_positive_risk": (
                "Upstream scope metadata may be wrong; cross-scope comparisons are not clustered."
            ),
            "tests": ["cross_scope_pair_blocking", "complete_link_cluster_safety"],
            "derived_from_development_data": False,
            "holdout_influenced": False,
            "threshold_governance": (
                "0.90 is a conservative clustering floor; no labeled-data calibration exists, "
                "so clusters remain diagnostic only."
            ),
        },
    ]
    source_files = [
        ROOT / "apps/api/app/domain/semantic_claim_equivalence.py",
        ROOT / "apps/api/app/services/semantic_claim_fallback.py",
    ]
    forbidden_literals = (
        "v1-dev-industrial-vision-defect", "low-altitude economy", "World Bank",
        "GPT-X", "01a0b",
    )
    matches = [
        {"file": path.relative_to(ROOT).as_posix(), "literal": literal}
        for path in source_files
        for literal in forbidden_literals
        if literal.casefold() in path.read_text(encoding="utf-8").casefold()
    ]
    return {
        "schema_version": "v2.1-generalization-audit.v1",
        "matcher_version": MATCHER_VERSION,
        "research_behavior_diff": "NONE",
        "production_sources_scanned": [path.relative_to(ROOT).as_posix() for path in source_files],
        "forbidden_special_case_matches": matches,
        "result": "NO_FORBIDDEN_SPECIAL_CASES" if not matches else "FAIL",
        "holdout_influenced": False,
        "holdout_labels_created": False,
        "mechanisms": mechanisms,
        "anti_overfitting_notes": [
            (
                "No benchmark, evaluation-ID, task-family, entity, publisher, or source-domain "
                "branches exist in production V2.1 code."
            ),
            (
                "The former founded/established predicate lookup was removed; raw paraphrases "
                "require the generic fallback or remain UNKNOWN."
            ),
            "Sampling strata use generic surface features and do not control matcher decisions.",
        ],
    }


def task_family_replay_summary(
    claim_rows: list[dict[str, Any]],
    pair_results: list[dict[str, Any]],
    run_metadata: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    run_family = {
        str(row["run_id"]): str(row.get("task_family") or row.get("benchmark_id") or "unknown")
        for row in run_metadata
    }
    grouped_claims: Counter[str] = Counter(
        run_family.get(str(row["run_id"]), "unknown") for row in claim_rows
    )
    grouped_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_results:
        grouped_pairs[run_family.get(str(pair["run_id"]), "unknown")].append(pair)
    summary: dict[str, dict[str, Any]] = {}
    for family in sorted(set(grouped_claims) | set(grouped_pairs)):
        pairs = grouped_pairs[family]
        summary[family] = {
            "claim_count": grouped_claims[family],
            "candidate_pair_count": len(pairs),
            "relation_counts": dict(Counter(str(pair["relation"]) for pair in pairs)),
            "unknown_reason_counts": unknown_reason_counts(pairs),
            "interpretation": "descriptive_unlabeled_not_accuracy_metrics",
        }
    return summary


def matching_audit() -> dict[str, Any]:
    return {
        "schema_version": "v2.1-existing-matching-audit.v1",
        "findings": {
            "normalization": (
                "V2.0 CanonicalClaim.from_text collapses whitespace; "
                "exact_proposition_key adds casefolding."
            ),
            "existing_semantic_matching": (
                "No Claim-specific semantic classifier or embedding matcher found."
            ),
            "existing_structured_matching": (
                "deterministic_relation_candidate compares subject/predicate, qualifiers, "
                "polarity and simple numeric values."
            ),
            "lexical_only": (
                "exact_proposition_key is lexical only; the V2.0 candidate helper is "
                "structured but is not a semantic paraphrase engine."
            ),
            "claim_deduplication": (
                "Persistence deduplicates by run/question claim_hash; it does not establish "
                "semantic equivalence."
            ),
            "qualifiers_in_equality": (
                "The V2.0 helper checks qualifier dictionaries after its exact-text early "
                "return, which can bypass conflicting structured qualifiers. V2.1 checks "
                "known conflicts before SAME."
            ),
            "llm_for_claim_comparison": (
                "No V2.0 Claim-comparison LLM call exists. V2.1 adds a generic, bounded, "
                "analyzer-only adapter on the existing LLMGateway; no historical model calls "
                "have been made."
            ),
            "embedding_infrastructure": (
                "No Claim embedding/similarity infrastructure found in application "
                "dependencies or code search."
            ),
            "safe_reuse": [
                "CanonicalClaim", "ClaimRelationType", "exact_proposition_key",
                "bounded numeric Decimal approach",
            ],
            "example_specific_predicate_synonyms": "REMOVED",
        },
        "research_behavior_diff": "NONE",
    }


def build_reports(
    database_url: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = load_json(BASELINE_PATH)
    run_ids = [str(row["run_id"]) for row in baseline.get("runs", [])]
    report: dict[str, Any] = {
        "schema_version": "v2.1-claim-equivalence.v1",
        "metric_definition_version": METRIC_VERSION,
        "matcher_version": MATCHER_VERSION,
        "source": (
            "read-only V2.0 baseline run Claims"
            if database_url
            else "unavailable: no reachable read-only historical DB or claim export supplied"
        ),
        "run_ids": run_ids,
        "research_behavior_diff": "NONE",
        "coverage_metric_definition": "coverage-v1 (unchanged)",
        "v2_0_metric_definition": "claim-v2-baseline-v1 (unchanged)",
        "semantic_model": {
            "used": False, "calls": 0, "tokens": 0, "latency_ms": 0, "cache_hits": 0
        },
        "semantic_fallback": {
            "available": True,
            "implementation": "bounded analyzer-only adapter on existing LLMGateway",
            "prompt_version": "semantic-claim-equivalence.generic.v1",
            "call_cap": 40,
            "calls_in_historical_replay": 0,
            "status": "AVAILABLE_NOT_INVOKED",
            "candidate_pairs_per_model_call": None,
        },
    }
    if not database_url:
        report.update({
            "historical_replay_status": "BLOCKED_NO_READ_ONLY_DATABASE_OR_CLAIM_EXPORT",
            "claim_count": None,
            "candidate_pair_count": None,
            "pairs_filtered_cross_run_question_dimension": None,
            "pairs_sent_to_deterministic_matcher": None,
            "relation_counts": {relation.value: None for relation in ClaimRelationType},
            "semantic_cluster_count": None,
            "cluster_size_distribution": {},
            "potential_multi_evidence_cluster_count": None,
            "potential_multi_owner_cluster_count": None,
            "limitation": (
                "The V2.0 baseline artifact contains only aggregate counts; it cannot "
                "reconstruct Claim text/IDs. No database connection or Claim export was "
                "available."
            ),
        })
        sample = make_manual_sample([])
        return report, sample, matching_audit()

    claim_rows = load_historical_claims(database_url, run_ids)
    replay = compare_history(claim_rows)
    report.update(replay)
    report["task_family_descriptive_behavior"] = task_family_replay_summary(
        claim_rows, replay["pair_results"], baseline.get("runs", [])
    )
    report["historical_replay_status"] = "COMPLETED_READ_ONLY"
    report["limitation"] = (
        "Historical rows lack V2 semantic annotations; replay counts are descriptive, not "
        "accuracy labels. Claims with only atomic text and no structured subject/predicate "
        "are conservatively UNKNOWN unless safely normalized exact matches. The generic "
        "LLMGateway fallback is available but was not invoked; independent labels and a "
        "bounded authorized model evaluation are still required to establish usefulness."
    )
    fixture_path = ROOT / "tests" / "fixtures" / "claim_semantics_v2_1.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    matcher = SemanticClaimMatcher()
    fixture_results: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures):
        left_data = fixture["left"]
        right_data = fixture["right"]
        run_id = UUID("00000000-0000-0000-0000-000000000001")
        def fixture_claim(
            data: dict[str, Any],
            offset: int,
            fixture_index: int = index,
            fixture_run_id: UUID = run_id,
        ) -> CanonicalClaim:
            return CanonicalClaim.from_text(
                claim_id=UUID(int=fixture_index * 2 + offset + 1), run_id=fixture_run_id,
                question_id="fixture-q", dimension_key="fixture-q:d1",
                original_text=data["text"], claim_type=CanonicalClaimType.QUANTITATIVE,
                subject=data.get("subject"), predicate=data.get("predicate"),
                object_or_value=data.get("value"), qualifiers=data.get("qualifiers", {}),
                polarity=data.get("polarity"),
            )
        comparison = matcher.compare(fixture_claim(left_data, 0), fixture_claim(right_data, 1))
        actual = comparison.relation.value
        fixture_results.append(
            {
                "case": fixture["case"], "expected": fixture["relation"],
                "actual": actual, "passed": actual == fixture["relation"],
            }
        )
    report["golden_fixture_results"] = fixture_results
    report["golden_fixture_counts"] = dict(Counter(item["actual"] for item in fixture_results))
    report["golden_fixture_failures"] = [item for item in fixture_results if not item["passed"]]
    expected_same = sum(
        item["expected"] == ClaimRelationType.SAME.value for item in fixture_results
    )
    predicted_same = sum(item["actual"] == ClaimRelationType.SAME.value for item in fixture_results)
    true_same = sum(
        item["expected"] == ClaimRelationType.SAME.value
        and item["actual"] == ClaimRelationType.SAME.value
        for item in fixture_results
    )
    false_same = sum(
        item["expected"] != ClaimRelationType.SAME.value
        and item["actual"] == ClaimRelationType.SAME.value
        for item in fixture_results
    )
    cluster_sizes = [len(cluster["claim_ids"]) for cluster in replay["clusters"]]
    relation_total = max(1, len(replay["pair_results"]))
    report["fixture_metrics"] = {
        "same_pair_precision": true_same / predicted_same if predicted_same else None,
        "same_pair_recall": true_same / expected_same if expected_same else None,
        "false_merge_count": false_same,
        "false_merge_rate": false_same / predicted_same if predicted_same else 0.0,
    }
    report["descriptive_metrics"] = {
        "unknown_rate": replay["relation_counts"][ClaimRelationType.UNKNOWN.value] / relation_total
        if replay["pair_results"] else None,
        "related_rate": replay["relation_counts"][ClaimRelationType.RELATED.value] / relation_total
        if replay["pair_results"] else None,
        "contradiction_candidate_rate": (
            replay["relation_counts"][ClaimRelationType.CONTRADICTS.value] / relation_total
            if replay["pair_results"]
            else None
        ),
        "mean_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else None,
        "multi_evidence_cluster_rate": (
            replay["potential_multi_evidence_cluster_count"] / len(replay["clusters"])
            if replay["clusters"]
            else None
        ),
        "multi_owner_cluster_rate": (
            replay["potential_multi_owner_cluster_count"] / len(replay["clusters"])
            if replay["clusters"]
            else None
        ),
    }
    return report, make_manual_sample(replay["pair_results"]), matching_audit()


def markdown_report(report: dict[str, Any]) -> str:
    relations = report.get("relation_counts", {})
    lines = [
        "# V2.1 Semantic Claim Equivalence",
        "",
        f"- Matcher: `{report['matcher_version']}`",
        f"- Metric: `{report['metric_definition_version']}`",
        f"- Historical replay: `{report['historical_replay_status']}`",
        f"- Source: {report['source']}",
        f"- Research behavior diff: **{report['research_behavior_diff']}**",
        "",
        "## Historical comparison",
        "",
        f"- Claims: {report.get('claim_count', 'NOT_RECONSTRUCTABLE')}",
        f"- Scoped candidate pairs: {report.get('candidate_pair_count', 'NOT_RECONSTRUCTABLE')}",
        f"- Cross-scope pairs blocked: "
        f"{report.get('pairs_filtered_cross_run_question_dimension', 'NOT_RECONSTRUCTABLE')}",
        f"- Pairs sent to deterministic matcher: "
        f"{report.get('pairs_sent_to_deterministic_matcher', 'NOT_RECONSTRUCTABLE')}",
        f"- Semantic model calls: {report['semantic_model']['calls']}",
        "- Semantic fallback: "
        f"{report.get('semantic_fallback', {}).get('status', 'NOT_AVAILABLE')} "
        f"(cap={report.get('semantic_fallback', {}).get('call_cap', 'n/a')})",
        f"- Relations: SAME={relations.get('same')}, RELATED={relations.get('related')}, "
        f"CONTRADICTS={relations.get('contradicts')}, UNKNOWN={relations.get('unknown')}",
        f"- Conservative clusters: {report.get('semantic_cluster_count', 'NOT_RECONSTRUCTABLE')}",
        f"- Potential multi-evidence clusters: "
        f"{report.get('potential_multi_evidence_cluster_count', 'NOT_RECONSTRUCTABLE')}",
        f"- Potential multi-owner clusters: "
        f"{report.get('potential_multi_owner_cluster_count', 'NOT_RECONSTRUCTABLE')}",
        f"- UNKNOWN diagnostic reasons (overlapping counts): "
        f"{report.get('unknown_reason_counts', 'NOT_RECONSTRUCTABLE')}",
        "",
        "## Task-family descriptive replay (unlabeled)",
        "",
    ]
    for family, summary in report.get("task_family_descriptive_behavior", {}).items():
        lines.append(
            f"- `{family}`: claims={summary['claim_count']}, "
            f"pairs={summary['candidate_pair_count']}, "
            f"relations={summary['relation_counts']}"
        )
    lines.extend([
        "", "These counts are descriptive, not accuracy or generalization labels.", "",
        "## Golden fixtures",
        "",
    ])
    for item in report.get("golden_fixture_results", []):
        outcome = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"- `{item['case']}`: expected `{item['expected']}`, "
            f"got `{item['actual']}` — {outcome}"
        )
    lines.extend(
        [
            "", "## Limitations", "",
            str(
                report.get(
                    "limitation",
                    "Historical replay completed; unlabeled history is descriptive only.",
                )
            ),
            "",
            "`SAME` clusters are potential semantic clusters only; they do not imply "
            "verification or independent corroboration.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# V2.1 Existing Claim Matching Audit", ""]
    for key, value in audit["findings"].items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(
        [
            "",
            "No matching path is connected to Research decisions; V2.1 remains offline/shadow.",
            "",
        ]
    )
    return "\n".join(lines)


def generalization_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Generalization Audit", "",
        f"- Result: **{audit['result']}**",
        f"- Research behavior diff: **{audit['research_behavior_diff']}**",
        f"- Production files scanned: {', '.join(audit['production_sources_scanned'])}",
        f"- Holdout influenced: `{str(audit['holdout_influenced']).lower()}`", "",
        "## Mechanisms", "",
    ]
    for mechanism in audit["mechanisms"]:
        lines.extend([
            f"### {mechanism['mechanism']}",
            f"- Generic rationale: {mechanism['domain_independent_rationale']}",
            f"- False-positive risk: {mechanism['false_positive_risk']}",
            f"- Holdout influenced: `{str(mechanism['holdout_influenced']).lower()}`",
            "",
        ])
    lines.extend(["## Anti-overfitting notes", ""])
    lines.extend(f"- {note}" for note in audit["anti_overfitting_notes"])
    lines.append("")
    return "\n".join(lines)


def sample_markdown(sample: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Historical Semantic Audit Sample", "",
        f"- Sample count: {sample['sample_count']}",
        f"- Review status: `{sample['review_status']}`", "",
    ]
    for index, item in enumerate(sample["comparisons"], 1):
        lines.extend([
            f"## {index}. {item['relation']} — {item['question_id']} / {item['dimension_key']}",
            f"- Left: {item['left_text']}", f"- Right: {item['right_text']}",
            f"- Reason: {item['rationale']}", f"- Human review: `{item['review_status']}`", "",
        ])
    if not sample["comparisons"]:
        lines.append(
            "No historical pairs were available: persisted Claim rows could not be read. "
            "The V2.0 aggregate artifact is not a substitute for raw Claim data."
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=os.environ.get("V2_CLAIM_BASELINE_DATABASE_URL"))
    parser.add_argument(
        "--project-env",
        action="store_true",
        help="resolve DATABASE_URL from the project's .env without printing it",
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    database_url = args.database_url
    if args.project_env:
        from app.core.config import Settings

        database_url = Settings(_env_file=ROOT / ".env").database_url
    try:
        report, sample, audit = build_reports(database_url)
    except psycopg.Error as exc:
        # Do not expose connection strings or driver diagnostics that could contain
        # deployment details; leave a transparent, explicitly blocked artifact.
        report = {
            "schema_version": "v2.1-claim-equivalence.v1",
            "metric_definition_version": METRIC_VERSION,
            "matcher_version": MATCHER_VERSION,
            "source": "read-only PostgreSQL requested but unreachable",
            "run_ids": [row["run_id"] for row in load_json(BASELINE_PATH).get("runs", [])],
            "research_behavior_diff": "NONE",
            "coverage_metric_definition": "coverage-v1 (unchanged)",
            "v2_0_metric_definition": "claim-v2-baseline-v1 (unchanged)",
            "historical_replay_status": "BLOCKED_DATABASE_UNAVAILABLE",
            "claim_count": None,
            "candidate_pair_count": None,
            "relation_counts": {relation.value: None for relation in ClaimRelationType},
            "semantic_cluster_count": None,
            "potential_multi_evidence_cluster_count": None,
            "potential_multi_owner_cluster_count": None,
            "semantic_model": {
                "used": False, "calls": 0, "tokens": 0, "latency_ms": 0, "cache_hits": 0
            },
            "limitation": (
                "Read-only historical DB connection unavailable "
                f"({type(exc).__name__}); no Claim-level replay was fabricated."
            ),
        }
        sample = make_manual_sample([])
        audit = matching_audit()
    prior_sample_path = args.output_dir / "v2_1_semantic_audit_sample.json"
    previously_exposed_pair_ids: set[str] = set()
    if prior_sample_path.exists():
        prior_sample = load_json(prior_sample_path)
        previously_exposed_pair_ids = {
            _eval_pair_id(item) for item in prior_sample.get("comparisons", [])
        }
    prior_dataset_path = args.output_dir / "v2_1_real_claim_eval_candidates.json"
    if prior_dataset_path.exists():
        prior_dataset = load_json(prior_dataset_path)
        previously_exposed_pair_ids.update(
            str(pair_id)
            for pair_id in (
                list(prior_dataset.get("development_ids", []))
                + list(prior_dataset.get("holdout_ids", []))
                + list(prior_dataset.get("excluded_prior_prediction_exposure_pair_ids", []))
            )
        )
        previous_review_items = prior_dataset.get("review_view", {}).get("items", [])
        previously_exposed_pair_ids.update(
            str(item["pair_id"])
            for item in previous_review_items
            if isinstance(item, dict) and item.get("pair_id")
        )
        previous_predictions = prior_dataset.get("matcher_predictions_separate", {})
        if isinstance(previous_predictions, dict):
            previously_exposed_pair_ids.update(str(pair_id) for pair_id in previous_predictions)
    dataset = make_real_claim_eval_dataset(
        report.get("pair_results", []),
        excluded_pair_ids=previously_exposed_pair_ids,
    )
    development_eval = evaluation_waiting_artifact(dataset, "development")
    holdout_eval = evaluation_waiting_artifact(dataset, "holdout")
    generalization = generalization_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "v2_1_existing_matching_audit.json": audit,
        "v2_1_existing_matching_audit.md": audit_markdown(audit),
        "v2_1_claim_equivalence.json": report,
        "v2_1_claim_equivalence.md": markdown_report(report),
        "v2_1_semantic_audit_sample.json": sample,
        "v2_1_semantic_audit_sample.md": sample_markdown(sample),
        "v2_1_real_claim_eval_candidates.json": dataset,
        "v2_1_real_claim_eval_candidates.md": real_claim_eval_markdown(dataset),
        "v2_1_development_eval.json": development_eval,
        "v2_1_development_eval.md": evaluation_waiting_markdown(development_eval),
        "v2_1_holdout_eval.json": holdout_eval,
        "v2_1_holdout_eval.md": evaluation_waiting_markdown(holdout_eval),
        "v2_1_generalization_audit.json": generalization,
        "v2_1_generalization_audit.md": generalization_audit_markdown(generalization),
    }
    for name, payload in outputs.items():
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        (args.output_dir / name).write_text(text, encoding="utf-8")
    summary = {
        "historical_replay_status": report["historical_replay_status"],
        "claim_count": report.get("claim_count"),
        "candidate_pair_count": report.get("candidate_pair_count"),
        "golden_fixture_failures": report.get("golden_fixture_failures", []),
        "evaluation_dataset_candidates": dataset["sample_count"],
        "development_count": len(dataset["development_ids"]),
        "holdout_count": len(dataset["holdout_ids"]),
        "generalization_audit": generalization["result"],
        "outputs": list(outputs),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
