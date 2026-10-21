"""Build a blind V2.1 development extension from historical Claim records.

This is dataset tooling only. It reads existing Claim/Evidence rows through a
read-only PostgreSQL transaction and never calls the semantic matcher or an LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATASET_PATH = ARTIFACTS / "v2_1_real_claim_eval_candidates.json"
CLAIM_REPORT_PATH = ARTIFACTS / "v2_1_claim_equivalence.json"
V1_JSON_PATH = ARTIFACTS / "v2_1_dev_extension_candidates.json"
V1_MD_PATH = ARTIFACTS / "v2_1_dev_extension_candidates.md"
DEV_EVAL_PATH = ARTIFACTS / "v2_1_development_eval.json"
AUDIT_SAMPLE_PATH = ARTIFACTS / "v2_1_semantic_audit_sample.json"
EXPECTED_HOLDOUT_HASH = "da07c803a64a978df146f6200144f5ae47715ddfeb19b584343d1d1a268ddc8b"
EXTENSION_VERSION = "v2_1_claim_eval_dev_extension_v2"
SELECTION_METHOD_VERSION = "claim-evidence-provenance-structure-v2"
TARGETS = {
    "proposition_identity_candidates": 18,
    "contradiction_scope_candidates": 15,
    "hard_negative_candidates": 10,
    "ambiguity_candidates": 8,
}
MINIMUM_EXTENSION_SIZE = 30
MAXIMUM_EXTENSION_SIZE = 50

NEGATION = re.compile(
    r"\b(?:no|not|never|without|neither|failed|declined|decreased)\b|"
    r"不|无|未|没有|并非|下降|减少|失败",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"(?<![\w])\d+(?:[.,]\d+)?\s*(?:%|percent|亿美元|亿元|million|"
    r"billion|trillion|km|kg|ms|s)?",
    re.IGNORECASE,
)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
VERSION = re.compile(r"\bv\s?\d+(?:\.\d+)*\b", re.IGNORECASE)
PROPER_ENTITY = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.-]{2,}|[A-Z]{3,})\b")
PRONOUN = re.compile(
    r"\b(?:it|they|them|their|this|that|these|those|its|he|she|former|latter)\b|"
    r"\b(?:respectively)\b|其|该|此|这些|那些|上述|前者|后者|该来源",
    re.IGNORECASE,
)
ENGLISH_PREDICATE_CUES = frozenset(
    (
        "is",
        "was",
        "are",
        "were",
        "has",
        "have",
        "had",
        "reached",
        "increased",
        "rose",
        "grew",
        "decreased",
        "declined",
        "fell",
        "includes",
        "comprises",
        "uses",
        "adopted",
        "contains",
        "covers",
        "defines",
        "requires",
        "supports",
        "operates",
        "announced",
        "reported",
        "estimates",
        "projects",
        "expects",
        "approved",
        "certified",
        "launched",
        "provides",
        "represents",
        "accounts",
        "caused",
        "led",
        "remains",
        "achieved",
        "stands",
    )
)
CHINESE_PREDICATE_CUES = (
    "达到",
    "增长",
    "提高",
    "增加",
    "下降",
    "减少",
    "包括",
    "采用",
    "使用",
    "定义",
    "要求",
    "支持",
    "运营",
    "宣布",
    "报告",
    "估计",
    "预计",
    "投产",
    "成立",
    "获批",
    "认证",
    "覆盖",
    "拥有",
    "提供",
    "导致",
    "保持",
    "实现",
    "代表",
    "构成",
)
PREDICATE_CUES = ENGLISH_PREDICATE_CUES | frozenset(CHINESE_PREDICATE_CUES)
GENERIC_STOPWORDS = frozenset(
    (
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "by",
        "from",
        "with",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "are",
        "was",
        "were",
        "is",
        "be",
        "been",
        "as",
        "at",
        "which",
        "while",
        "than",
        "then",
        "into",
        "over",
        "under",
        "about",
        "according",
        "的",
        "了",
        "在",
        "并",
        "以及",
        "该",
        "其",
        "一个",
        "通过",
        "表示",
        "指出",
        "可以",
        "这种",
        "具有",
        "对于",
        "与",
        "和",
        "将",
    )
)
GENERIC_SCOPE = frozenset(
    (
        "global",
        "globally",
        "worldwide",
        "national",
        "nationwide",
        "regional",
        "locally",
        "local",
        "domestic",
        "internationally",
        "international",
        "中国",
        "全球",
        "全国",
        "国内",
        "国际",
        "地区",
        "区域",
        "本地",
        "亚太",
    )
)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def pair_id(left_id: str, right_id: str) -> str:
    material = "|".join(sorted((str(left_id), str(right_id))))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def membership_hash(pair_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(pair_ids)).encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def _text_pair_key(left: str, right: str) -> tuple[str, str]:
    a, b = _normalize_text(left), _normalize_text(right)
    return (a, b) if a <= b else (b, a)


def _tokens(text: str) -> set[str]:
    folded = _normalize_text(text)
    latin = {
        token
        for token in re.findall(r"[a-z][a-z0-9._%-]*", folded)
        if len(token) > 1 and token not in GENERIC_STOPWORDS
    }
    cjk_text = "".join(re.findall(r"[\u3400-\u9fff]", folded))
    cjk = {
        cjk_text[index : index + width]
        for width in (2, 3)
        for index in range(max(0, len(cjk_text) - width + 1))
    }
    return latin | cjk


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _numbers(text: str) -> frozenset[str]:
    return frozenset(re.sub(r"\s+", "", value.casefold()) for value in NUMBER.findall(text))


def _years(text: str) -> frozenset[str]:
    return frozenset(YEAR.findall(text))


def _versions(text: str) -> frozenset[str]:
    return frozenset(re.sub(r"\s+", "", value.casefold()) for value in VERSION.findall(text))


def _scope_markers(text: str) -> frozenset[str]:
    folded = _normalize_text(text)
    return frozenset(token for token in GENERIC_SCOPE if token in folded)


def _entities(text: str) -> tuple[str, ...]:
    ignored = {"AI", "API", "URL", "PDF", "USD", "EUR", "GDP", "CEO", "R&D"}
    return tuple(
        token.casefold() for token in PROPER_ENTITY.findall(text) if token.upper() not in ignored
    )


def _predicate_cues(text: str) -> frozenset[str]:
    found: set[str] = set()
    folded = _normalize_text(text)
    for cue in ENGLISH_PREDICATE_CUES:
        if re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", folded):
            found.add(cue)
    for cue in CHINESE_PREDICATE_CUES:
        if cue in text:
            found.add(cue)
    return frozenset(found)


def _core_tokens(text: str) -> set[str]:
    tokens = _tokens(text)
    excluded = set(_numbers(text)) | set(_years(text)) | set(_versions(text))
    excluded.update(_entities(text))
    excluded.update(cue for cue in _predicate_cues(text) if cue.isascii())
    return tokens - excluded


def _has_polarity_conflict(left: str, right: str) -> bool:
    return bool(NEGATION.search(left)) != bool(NEGATION.search(right))


def _same_subject_anchor(left: str, right: str) -> bool:
    left_entities, right_entities = _entities(left), _entities(right)
    if left_entities and right_entities:
        # Requiring the first named anchor to agree avoids treating two different
        # named subjects as contradictory merely because they share a topic.
        return left_entities[0] == right_entities[0]
    left_head = _leading_anchor(left)
    right_head = _leading_anchor(right)
    return bool(left_head) and left_head == right_head


def _leading_anchor(text: str) -> str:
    first_clause = cast(
        str,
        re.split(r"[,，;；:：。.!?！？]", text, maxsplit=1)[0],  # noqa: RUF001
    ).strip()
    first_clause = re.sub(
        r"^(?:该来源|该公司|该系统|该模型|the source|the company)\s*",
        "",
        first_clause,
        flags=re.I,
    )
    latin = cast(list[str], re.findall(r"[A-Za-z][A-Za-z0-9&.-]{2,}", first_clause))
    if latin:
        return latin[0].casefold()
    cjk = "".join(cast(list[str], re.findall(r"[\u3400-\u9fff]", first_clause)))
    return cjk[:5] if len(cjk) >= 5 else ""


def _time_compatible(left: str, right: str) -> bool:
    left_years, right_years = _years(left), _years(right)
    left_versions, right_versions = _versions(left), _versions(right)
    return (not left_years or not right_years or left_years == right_years) and (
        not left_versions or not right_versions or left_versions == right_versions
    )


def _scope_compatible(left: str, right: str) -> bool:
    a, b = _scope_markers(left), _scope_markers(right)
    return not a or not b or a == b


def _masked_overlap(left: str, right: str) -> float:
    def masked_tokens(text: str) -> set[str]:
        value = NUMBER.sub(" VALUE ", _normalize_text(text))
        return _tokens(value)

    return _jaccard(masked_tokens(left), masked_tokens(right))


def _shared_source_ids(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return set(left.get("source_ids", [])) & set(right.get("source_ids", []))


def _shared_chunk_ids(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return set(left.get("chunk_ids", [])) & set(right.get("chunk_ids", []))


def pair_features(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_text, right_text = str(left["atomic_claim"]), str(right["atomic_claim"])
    left_tokens, right_tokens = _tokens(left_text), _tokens(right_text)
    shared_core = _core_tokens(left_text) & _core_tokens(right_text)
    shared_cues = _predicate_cues(left_text) & _predicate_cues(right_text)
    return {
        "text_overlap": _jaccard(left_tokens, right_tokens),
        "masked_overlap": _masked_overlap(left_text, right_text),
        "same_values": _numbers(left_text) == _numbers(right_text),
        "value_conflict": bool(_numbers(left_text) and _numbers(right_text))
        and _numbers(left_text) != _numbers(right_text),
        "time_compatible": _time_compatible(left_text, right_text),
        "same_explicit_time": bool(_years(left_text) or _versions(left_text))
        and _years(left_text) == _years(right_text)
        and _versions(left_text) == _versions(right_text),
        "same_time_qualifier": _years(left_text) == _years(right_text)
        and _versions(left_text) == _versions(right_text),
        "scope_compatible": _scope_compatible(left_text, right_text),
        "same_scope_qualifier": _scope_markers(left_text) == _scope_markers(right_text),
        "same_subject_anchor": _same_subject_anchor(left_text, right_text),
        "shared_predicate_cue": bool(shared_cues),
        "shared_core_token_count": len(shared_core),
        "shared_core_tokens": sorted(shared_core)[:12],
        "polarity_conflict": _has_polarity_conflict(left_text, right_text),
        "shared_source_count": len(_shared_source_ids(left, right)),
        "shared_chunk_count": len(_shared_chunk_ids(left, right)),
        "same_claim_hash": left.get("claim_hash") == right.get("claim_hash"),
    }


def _pair_record(
    left: dict[str, Any], right: dict[str, Any], stratum: str, features: dict[str, Any]
) -> dict[str, Any]:
    return {
        "pair_id": pair_id(str(left["id"]), str(right["id"])),
        "claim_a_id": str(left["id"]),
        "claim_b_id": str(right["id"]),
        "claim_a_text": str(left["atomic_claim"]),
        "claim_b_text": str(right["atomic_claim"]),
        "run_id": str(left["run_id"]),
        "question_id": str(left["question_id"]),
        "dimension_key": str(left["dimension_key"]),
        "sampling_stratum": stratum,
        "candidate_signals": {
            key: features[key]
            for key in (
                "same_subject_anchor",
                "shared_predicate_cue",
                "shared_core_token_count",
                "same_values",
                "value_conflict",
                "time_compatible",
                "same_explicit_time",
                "same_time_qualifier",
                "scope_compatible",
                "same_scope_qualifier",
                "shared_source_count",
                "shared_chunk_count",
                "masked_overlap",
            )
        },
        "human_label": None,
        "label_provenance": None,
        "review_rationale": None,
    }


def _claim_pair_rows(claims: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[
            (str(claim["run_id"]), str(claim["question_id"]), str(claim["dimension_key"]))
        ].append(claim)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group in grouped.values():
        group.sort(key=lambda item: str(item["id"]))
        for index, left in enumerate(group):
            pairs.extend((left, right) for right in group[index + 1 :])
    return pairs


def select_v2_candidates(
    claims: list[dict[str, Any]],
    *,
    excluded_pair_ids: set[str],
    excluded_text_pairs: set[tuple[str, str]] | None = None,
    target_sizes: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    targets = target_sizes or TARGETS
    excluded_text_keys = excluded_text_pairs or set()
    queues: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = {
        stratum: [] for stratum in targets
    }
    for left, right in _claim_pair_rows(claims):
        left_id, right_id = str(left["id"]), str(right["id"])
        current_pair_id = pair_id(left_id, right_id)
        if current_pair_id in excluded_pair_ids:
            continue
        left_text, right_text = str(left["atomic_claim"]), str(right["atomic_claim"])
        if not left_text.strip() or not right_text.strip():
            continue
        if _normalize_text(left_text) == _normalize_text(right_text):
            # Identical text records are not useful as a paraphrase/identity pair.
            continue
        if _text_pair_key(left_text, right_text) in excluded_text_keys:
            continue
        features = pair_features(left, right)
        shared_source = features["shared_source_count"] > 0
        shared_chunk = features["shared_chunk_count"] > 0
        strong_proposition_support = (
            features["same_subject_anchor"]
            and features["same_values"]
            and features["time_compatible"]
            and features["scope_compatible"]
            and not features["polarity_conflict"]
            and features["shared_core_token_count"] >= 2
            and features["text_overlap"] >= 0.08
            and (shared_chunk or shared_source)
        )
        if strong_proposition_support:
            queues["proposition_identity_candidates"].append((left, right, features))
            continue

        same_time_qualifier = features["same_time_qualifier"]
        incompatible_assertion = features["value_conflict"] or features["polarity_conflict"]
        contradiction_scope = (
            features["same_subject_anchor"]
            and same_time_qualifier
            and features["same_scope_qualifier"]
            and incompatible_assertion
            and features["masked_overlap"] >= 0.18
            and features["shared_core_token_count"] >= 2
            and (shared_chunk or shared_source)
        )
        if contradiction_scope:
            queues["contradiction_scope_candidates"].append((left, right, features))
            continue

        hard_negative = (
            features["same_subject_anchor"]
            and (shared_source or shared_chunk)
            and features["text_overlap"] >= 0.05
            and (
                not features["same_values"]
                or not features["time_compatible"]
                or not features["scope_compatible"]
                or features["shared_core_token_count"] < 2
            )
        )
        if hard_negative:
            queues["hard_negative_candidates"].append((left, right, features))
            continue

        unresolved_reference = bool(PRONOUN.search(left_text) or PRONOUN.search(right_text))
        missing_subject_with_provenance = (
            not features["same_subject_anchor"]
            and (shared_chunk or shared_source)
            and features["shared_core_token_count"] >= 2
        )
        if features["text_overlap"] >= 0.08 and (
            unresolved_reference or missing_subject_with_provenance
        ):
            queues["ambiguity_candidates"].append((left, right, features))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_text_pairs: set[tuple[str, str]] = set()
    available: dict[str, int] = {}
    for stratum, target in targets.items():
        pool = queues[stratum]
        available[stratum] = len(pool)
        if target <= 0:
            continue
        pool.sort(
            key=lambda item: hashlib.sha256(
                f"{EXTENSION_VERSION}:{stratum}:"
                f"{pair_id(str(item[0]['id']), str(item[1]['id']))}".encode()
            ).hexdigest()
        )
        accepted = 0
        for left, right, features in pool:
            current_id = pair_id(str(left["id"]), str(right["id"]))
            text_key = _text_pair_key(str(left["atomic_claim"]), str(right["atomic_claim"]))
            if current_id in selected_ids or text_key in selected_text_pairs:
                continue
            selected.append(_pair_record(left, right, stratum, features))
            selected_ids.add(current_id)
            selected_text_pairs.add(text_key)
            accepted += 1
            if accepted == target:
                break
        if accepted < target:
            raise ValueError(
                f"Insufficient safe {stratum}: selected {accepted}/{target}; "
                f"candidate pool={available[stratum]}. No output artifacts written."
            )
    return selected, available


def parse_confirmed_v1_labels(markdown: str) -> dict[str, tuple[str, str]]:
    blocks = re.split(r"(?m)^\d+\. ([0-9a-f]{24})\s*$", markdown)
    labels: dict[str, tuple[str, str]] = {}
    for index in range(1, len(blocks), 2):
        current_id, body = blocks[index], blocks[index + 1]
        label_match = re.search(
            r"(?m)^Human label .*?:\s*(RELATED|UNKNOWN|SAME|CONTRADICTS)\s*$",
            body,
        )
        if not label_match:
            continue
        rationale_match = re.search(r"(?m)^Review rationale:\s*(.*?)\s*$", body)
        labels[current_id] = (
            label_match.group(1),
            rationale_match.group(1) if rationale_match else "",
        )
    return labels


def _holdout_claim_ids(dataset: dict[str, Any], holdout_ids: set[str]) -> set[str]:
    result: set[str] = set()
    for row in dataset.get("review_view", {}).get("items", []):
        # Only membership and Claim ID fields are read for the frozen Holdout.
        if str(row.get("pair_id", "")) in holdout_ids:
            result.update((str(row["claim_a_id"]), str(row["claim_b_id"])))
    return result


def _historical_non_holdout_text_pairs(
    pair_rows: list[dict[str, Any]],
    holdout_ids: set[str],
    included_pair_ids: set[str],
) -> set[tuple[str, str]]:
    """Return selected historical text keys, never reading a Holdout row's text."""
    result: set[tuple[str, str]] = set()
    for row in pair_rows:
        current_id = pair_id(str(row["left_claim_id"]), str(row["right_claim_id"]))
        if current_id in holdout_ids or current_id not in included_pair_ids:
            continue
        left_text, right_text = row.get("left_text"), row.get("right_text")
        if isinstance(left_text, str) and isinstance(right_text, str):
            result.add(_text_pair_key(left_text, right_text))
    return result


def _read_database_claims(claim_ids: set[str], holdout_claim_ids: set[str]) -> list[dict[str, Any]]:
    if not claim_ids:
        raise ValueError("No non-holdout Claim IDs are available for sampling.")
    for value in claim_ids | holdout_claim_ids:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", value):
            raise ValueError("Unexpected Claim ID format; refusing to query the database.")
    requested = ",".join(f"'{value}'" for value in sorted(claim_ids))
    holdout = ",".join(f"'{value}'" for value in sorted(holdout_claim_ids)) or "NULL"
    sql = f"""
BEGIN TRANSACTION READ ONLY;
SELECT COALESCE(
  json_agg(row_data ORDER BY run_id, question_id, dimension_key, claim_id),
  '[]'::json
)::text
FROM (
  SELECT json_build_object(
    'claim_id', c.id::text,
    'run_id', c.run_id::text,
    'question_id', c.question_id,
    'dimension_key', c.dimension_key,
    'atomic_claim', c.atomic_claim,
    'claim_hash', c.claim_hash,
    'claim_type', c.claim_type,
    'status', c.status,
    'source_ids', COALESCE(ev.source_ids, ARRAY[]::text[]),
    'chunk_ids', COALESCE(ev.chunk_ids, ARRAY[]::text[])
  ) AS row_data,
  c.run_id::text AS run_id,
  c.question_id,
  c.dimension_key,
  c.id::text AS claim_id
  FROM research_claims c
  LEFT JOIN (
    SELECT claim_id,
      ARRAY_REMOVE(ARRAY_AGG(DISTINCT source_id::text), NULL) AS source_ids,
      ARRAY_REMOVE(ARRAY_AGG(DISTINCT chunk_id::text), NULL) AS chunk_ids
    FROM research_evidence
    GROUP BY claim_id
  ) ev ON ev.claim_id = c.id
  WHERE c.id IN ({requested})
    AND c.id NOT IN ({holdout})
) claim_rows;
COMMIT;
"""
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "deep_research",
            "-d",
            "deep_research",
            "-X",
            "-A",
            "-t",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
        ],
        input=sql,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=90,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Read-only Claim/Evidence query returned an unexpected result.")
    records = cast(list[dict[str, Any]], json.loads(lines[0]))
    normalized: list[dict[str, Any]] = []
    for row in records:
        normalized.append(
            {
                "id": row["claim_id"],
                "run_id": row["run_id"],
                "question_id": row["question_id"],
                "dimension_key": row["dimension_key"],
                "atomic_claim": row["atomic_claim"],
                "claim_hash": row["claim_hash"],
                "claim_type": row["claim_type"],
                "status": row["status"],
                "source_ids": list(row["source_ids"] or []),
                "chunk_ids": list(row["chunk_ids"] or []),
            }
        )
    return normalized


def _markdown(extension: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Development Extension V2 — Blind Human Review",
        "",
        f"- Dataset: `{extension['dataset_version']}`",
        f"- Pair count: {extension['pair_count']}",
        f"- Membership SHA-256: `{extension['membership_sha256']}`",
        f"- Selection method: `{extension['selection_method_version']}`",
        f"- Generated at: `{extension['generated_at']}`",
        "- Candidate strata are retrieval aids, not semantic labels.",
        "- Matcher predictions and LLM outputs are intentionally omitted.",
        "",
    ]
    for index, pair in enumerate(extension["pairs"], 1):
        lines.extend(
            [
                f"## {index}. {pair['pair_id']}",
                f"- Candidate stratum: {pair['sampling_stratum']}",
                f"- Scope: run={pair['run_id']}; question={pair['question_id']}; "
                f"dimension={pair['dimension_key']}",
                f"- Claim A (`{pair['claim_a_id']}`): {pair['claim_a_text']}",
                f"- Claim B (`{pair['claim_b_id']}`): {pair['claim_b_text']}",
                "- Human label (SAME / RELATED / CONTRADICTS / UNKNOWN):",
                "- Review rationale:",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _v1_sampling_audit(v1: dict[str, Any], labels: dict[str, tuple[str, str]]) -> dict[str, Any]:
    per_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in v1["pairs"]:
        label = labels[pair["pair_id"]][0]
        per_stratum[str(pair["primary_sampling_stratum"])][label] += 1
    distribution = {
        stratum: dict(sorted(counts.items())) for stratum, counts in sorted(per_stratum.items())
    }
    return {
        "schema_version": "v2.1-dev-sampling-audit.v1",
        "audited_dataset_version": v1["dataset_version"],
        "pair_count": len(v1["pairs"]),
        "human_label_counts": dict(sorted(Counter(label for label, _ in labels.values()).items())),
        "human_labels_by_original_stratum": distribution,
        "first_sampler_surface_rules": [
            "lexical overlap threshold",
            "numeric/time/version token compatibility or difference",
            "polarity-token difference",
            "pronoun/low-overlap ambiguity markers",
            "pairs were already restricted to a shared run/question/dimension scope",
        ],
        "diagnosis": {
            "result": "TOPICAL_RELATEDNESS_DOMINATED",
            "topical_relatedness": (
                "Pairs shared topic, question/dimension, or nearby surface terms, "
                "but those features did not establish the same "
                "subject-predicate-property fact slot."
            ),
            "proposition_identity": (
                "Requires stronger alignment of subject anchor, predicate/property, "
                "value and qualifiers; shared numbers or dimension alone are insufficient."
            ),
            "contradiction": (
                "Different numeric values or polarity alone do not establish contradiction; "
                "subject, predicate, time and scope must first align."
            ),
        },
        "matcher_predictions_used": False,
        "holdout_used": False,
        "production_behavior_changed": False,
    }


def _extension_diagnostics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"candidate_count": 0}
    signal = [pair["candidate_signals"] for pair in pairs]
    return {
        "candidate_count": len(pairs),
        "shared_subject_rate": sum(item["same_subject_anchor"] for item in signal) / len(signal),
        "shared_predicate_or_metric_cue_rate": sum(
            item["shared_predicate_cue"] or item["shared_core_token_count"] >= 2 for item in signal
        )
        / len(signal),
        "compatible_time_rate": sum(item["time_compatible"] for item in signal) / len(signal),
        "compatible_scope_rate": sum(item["scope_compatible"] for item in signal) / len(signal),
        "value_conflict_candidate_rate": sum(item["value_conflict"] for item in signal)
        / len(signal),
        "near_proposition_overlap_rate": sum(item["masked_overlap"] >= 0.18 for item in signal)
        / len(signal),
        "interpretation": "surface/provenance diagnostics only; not semantic accuracy or precision",
    }


def build_outputs() -> dict[str, str]:
    dataset = _read_json(DATASET_PATH)
    v1 = _read_json(V1_JSON_PATH)
    report = _read_json(CLAIM_REPORT_PATH)
    audit_sample = _read_json(AUDIT_SAMPLE_PATH)
    holdout_ids = [str(value) for value in dataset["holdout_ids"]]
    holdout_set = set(holdout_ids)
    holdout_hash_before = membership_hash(holdout_ids)
    if (
        dataset.get("dataset_version") != "v2_1_claim_eval_holdout_v1"
        or dataset.get("holdout_membership_sha256") != EXPECTED_HOLDOUT_HASH
        or holdout_hash_before != EXPECTED_HOLDOUT_HASH
    ):
        raise ValueError("Frozen Holdout version/hash mismatch; STOP before writing.")
    if len(dataset["development_ids"]) != 45 or len(holdout_ids) != 25:
        raise ValueError("Frozen Development/Holdout counts changed; STOP before writing.")
    if v1.get("dataset_version") != "v2_1_claim_eval_dev_extension_v1" or len(v1["pairs"]) != 36:
        raise ValueError("First extension membership/version changed; STOP before writing.")

    labels = parse_confirmed_v1_labels(V1_MD_PATH.read_text(encoding="utf-8"))
    v1_ids = [str(pair["pair_id"]) for pair in v1["pairs"]]
    if set(labels) != set(v1_ids) or len(labels) != 36:
        raise ValueError("The existing 36 human labels are incomplete or misaligned by pair ID.")
    label_counts = Counter(label for label, _ in labels.values())
    if label_counts != Counter({"RELATED": 35, "UNKNOWN": 1}):
        raise ValueError(f"Unexpected confirmed first-extension labels: {dict(label_counts)}")

    original_membership_hash = membership_hash(v1_ids)
    v1_labeled = json.loads(json.dumps(v1))
    for pair in v1_labeled["pairs"]:
        label, rationale = labels[pair["pair_id"]]
        pair["human_label"] = label
        pair["label_provenance"] = "human_confirmed_by_user"
        pair["review_rationale"] = rationale
    v1_labeled["label_status"] = "HUMAN_LABELED"
    v1_labeled["human_label_provenance"] = "human_confirmed_by_user"
    v1_labeled["label_counts"] = {
        "RELATED": label_counts["RELATED"],
        "UNKNOWN": label_counts["UNKNOWN"],
        "SAME": label_counts["SAME"],
        "CONTRADICTS": label_counts["CONTRADICTS"],
    }
    if (
        membership_hash([str(pair["pair_id"]) for pair in v1_labeled["pairs"]])
        != original_membership_hash
    ):
        raise RuntimeError("First extension membership/order changed while recording labels.")

    holdout_claim_ids = _holdout_claim_ids(dataset, holdout_set)
    excluded_pair_ids = (
        set(str(value) for value in dataset["development_ids"])
        | holdout_set
        | set(v1_ids)
        | set(str(value) for value in dataset["excluded_prior_prediction_exposure_pair_ids"])
        | {
            pair_id(str(row["left_claim_id"]), str(row["right_claim_id"]))
            for row in audit_sample["comparisons"]
        }
    )
    requested_claim_ids: set[str] = set()
    for row in report.get("pair_results", []):
        current_id = pair_id(str(row["left_claim_id"]), str(row["right_claim_id"]))
        # For Holdout rows, only IDs are used to skip. Text/prediction fields are not read.
        if current_id in holdout_set:
            continue
        requested_claim_ids.update((str(row["left_claim_id"]), str(row["right_claim_id"])))
    requested_claim_ids -= holdout_claim_ids
    claims = _read_database_claims(requested_claim_ids, holdout_claim_ids)
    if not claims:
        raise ValueError("Read-only DB query returned no eligible historical Claims.")

    exposed_ids = set(
        str(value) for value in dataset["excluded_prior_prediction_exposure_pair_ids"]
    )
    audit_pair_ids = {
        pair_id(str(row["left_claim_id"]), str(row["right_claim_id"]))
        for row in audit_sample["comparisons"]
    }
    excluded_text_pairs = _historical_non_holdout_text_pairs(
        report.get("pair_results", []),
        holdout_set,
        set(str(value) for value in dataset["development_ids"]) | exposed_ids | audit_pair_ids,
    )
    for row in dataset.get("review_view", {}).get("items", []):
        current_id = str(row.get("pair_id", ""))
        if current_id in holdout_set or current_id not in excluded_pair_ids:
            continue
        left_text, right_text = row.get("claim_a"), row.get("claim_b")
        if isinstance(left_text, str) and isinstance(right_text, str):
            excluded_text_pairs.add(_text_pair_key(left_text, right_text))
    for row in v1.get("pairs", []):
        left_text, right_text = row.get("claim_a_text"), row.get("claim_b_text")
        if isinstance(left_text, str) and isinstance(right_text, str):
            excluded_text_pairs.add(_text_pair_key(left_text, right_text))
    _, candidate_pool_counts = select_v2_candidates(
        claims,
        excluded_pair_ids=excluded_pair_ids,
        excluded_text_pairs=excluded_text_pairs,
        target_sizes={stratum: 0 for stratum in TARGETS},
    )
    selection_targets = {
        stratum: min(cap, candidate_pool_counts[stratum]) for stratum, cap in TARGETS.items()
    }
    if sum(selection_targets.values()) < MINIMUM_EXTENSION_SIZE:
        raise ValueError(
            "Safe candidate pools cannot reach the 30-pair minimum without relaxing "
            f"generic alignment guards: available targets={selection_targets}. "
            "No output artifacts written."
        )
    selected, _ = select_v2_candidates(
        claims,
        excluded_pair_ids=excluded_pair_ids,
        excluded_text_pairs=excluded_text_pairs,
        target_sizes=selection_targets,
    )
    selected_ids = [str(pair["pair_id"]) for pair in selected]
    if not MINIMUM_EXTENSION_SIZE <= len(selected) <= MAXIMUM_EXTENSION_SIZE:
        raise ValueError(
            "Deduplication left fewer than 30 or more than 50 safe candidates; "
            f"selected={len(selected)}. No output artifacts written."
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Extension V2 size or pair-ID uniqueness check failed.")
    if set(selected_ids) & excluded_pair_ids:
        raise ValueError("Extension V2 overlaps an existing/frozen/exposed pair ID.")
    selected_text_pairs = {
        _text_pair_key(pair["claim_a_text"], pair["claim_b_text"]) for pair in selected
    }
    if len(selected_text_pairs) != len(selected):
        raise ValueError("Extension V2 contains normalized-text duplicate pairs.")
    if any(str(pair["claim_a_id"]) == str(pair["claim_b_id"]) for pair in selected):
        raise ValueError("Extension V2 contains a self-pair.")
    selected_claim_ids = {
        str(pair[key]) for pair in selected for key in ("claim_a_id", "claim_b_id")
    }
    if selected_claim_ids & holdout_claim_ids:
        raise ValueError("Extension V2 references a Claim ID present in frozen Holdout.")

    distribution = dict(sorted(Counter(str(pair["sampling_stratum"]) for pair in selected).items()))
    timestamp = datetime.now(UTC).isoformat()
    extension = {
        "schema_version": "v2.1-dev-extension-candidates.v2",
        "dataset_version": EXTENSION_VERSION,
        "purpose": "development_only",
        "pair_count": len(selected),
        "membership_sha256": membership_hash(selected_ids),
        "generated_at": timestamp,
        "selection_method_version": SELECTION_METHOD_VERSION,
        "selection_method": (
            "Generic Claim/Evidence provenance and proposition-surface alignment; "
            "same-run/question/dimension pairs only; deterministic hash ordering."
        ),
        "sampling_distribution_by_stratum": distribution,
        "selected_target_by_stratum": selection_targets,
        "candidate_pool_counts_by_stratum": candidate_pool_counts,
        "candidate_pool_count_semantics": "eligible rows before selected-set de-duplication",
        "sampling_strata_are_labels": False,
        "matcher_predictions_included": False,
        "semantic_model_outputs_included": False,
        "excluded_existing_dev_count": len(dataset["development_ids"]),
        "excluded_first_extension_v1_count": len(v1_ids),
        "excluded_frozen_holdout_count": len(holdout_set),
        "excluded_frozen_holdout_claim_id_count": len(holdout_claim_ids),
        "excluded_previously_exposed_pair_count": len(
            dataset["excluded_prior_prediction_exposure_pair_ids"]
        ),
        "overlap_with_existing_dev": 0,
        "overlap_with_first_extension_v1": 0,
        "overlap_with_frozen_holdout": 0,
        "overlap_with_previously_exposed": 0,
        "normalized_text_overlap_with_existing_non_holdout_pairs": 0,
        "normalized_text_overlap_with_frozen_holdout": (
            "NOT_CHECKED_TO_PRESERVE_HOLDOUT_BLINDNESS"
        ),
        "holdout_pair_id_overlap": 0,
        "holdout_claim_id_overlap": 0,
        "generalization_audit": {
            "result": "NO_FORBIDDEN_SPECIAL_CASES",
            "benchmark_specific_sampling": False,
            "entity_alias_rules": False,
            "source_domain_rules": False,
            "question_specific_rules": False,
            "task_family_rules": False,
            "holdout_dependence": False,
            "matcher_prediction_dependence": False,
        },
        "surface_diagnostics": _extension_diagnostics(selected),
        "label_status": "AWAITING_HUMAN_LABELS",
        "pairs": selected,
    }

    dev_eval = _read_json(DEV_EVAL_PATH)
    base_dev_labels = [
        {
            "pair_id": str(pair_id_value),
            "human_label": "RELATED",
            "label_provenance": "human_confirmed_by_user",
            "label_source": "original_development",
        }
        for pair_id_value in dataset["development_ids"]
    ]
    v1_dev_labels = [
        {
            "pair_id": str(pair_id_value),
            "human_label": labels[str(pair_id_value)][0],
            "label_provenance": "human_confirmed_by_user",
            "label_source": "development_extension_v1",
            "review_rationale": labels[str(pair_id_value)][1],
        }
        for pair_id_value in v1_ids
    ]
    label_counts_all = Counter(item["human_label"] for item in base_dev_labels + v1_dev_labels)
    dev_eval.update(
        {
            "schema_version": "v2.1-labeled-evaluation.v2",
            "dataset_version": dataset["dataset_version"],
            "development_labeled_count": 81,
            "development_label_counts": {
                label: label_counts_all[label]
                for label in ("RELATED", "SAME", "CONTRADICTS", "UNKNOWN")
            },
            "development_labels_by_source": {
                "original_development": {"count": 45, "RELATED": 45},
                "development_extension_v1": {"count": 36, "RELATED": 35, "UNKNOWN": 1},
                "development_extension_v2": {
                    "count": len(selected),
                    "status": "AWAITING_HUMAN_LABELS",
                },
            },
            "labeled_pairs": base_dev_labels + v1_dev_labels,
            "matcher_predictions_separate_from_human_labels": True,
            "matcher_predictions_scope": (
                "existing_original_development_only; no new predictions run"
            ),
            "extension_v1_label_status": "HUMAN_LABELED",
            "extension_v2_dataset_version": EXTENSION_VERSION,
            "extension_v2_label_status": "AWAITING_HUMAN_LABELS",
            "holdout_membership_sha256": EXPECTED_HOLDOUT_HASH,
            "holdout_used_for_tuning": False,
        }
    )
    dev_eval["matcher_prediction_counts"] = dev_eval.get("matcher_prediction_counts", {})
    dev_eval["development_only_metrics"] = dev_eval.get("development_only_metrics", {})
    dev_eval["development_only_metrics"]["scope_note"] = (
        "Any pre-existing diagnostic metric remains scoped to the original "
        "45-pair development set; "
        "no new matcher prediction or metric was computed."
    )

    audit = _v1_sampling_audit(v1_labeled, labels)
    audit["v2_candidate_retrieval"] = {
        "source_claim_count": len(claims),
        "claims_with_evidence_source": sum(bool(claim["source_ids"]) for claim in claims),
        "claims_with_chunk_provenance": sum(bool(claim["chunk_ids"]) for claim in claims),
        "candidate_pool_counts_by_stratum": candidate_pool_counts,
        "selection_distribution": distribution,
        "method_uses_only_generic_fields": [
            "atomic_claim text",
            "question/dimension/run scope",
            "claim hash",
            "linked evidence source/chunk IDs",
            "generic numeric/time/polarity and proposition-token signals",
        ],
    }
    audit["holdout_membership_sha256_before"] = holdout_hash_before
    audit["holdout_membership_sha256_after"] = holdout_hash_before
    audit["holdout_contents_used"] = False

    extension_md = _markdown(extension)
    dev_md = [
        "# V2.1 Development Evaluation",
        "",
        "- Labeled Development pairs: 81",
        "- Labels: RELATED=80, UNKNOWN=1, SAME=0, CONTRADICTS=0",
        "- Provenance: `human_confirmed_by_user`",
        "- Extension V2: `v2_1_claim_eval_dev_extension_v2` — AWAITING_HUMAN_LABELS",
        "- Matcher predictions remain a separate historical artifact; no new predictions were run.",
        "- Frozen Holdout was not used for sampling, labeling, or tuning.",
        "",
    ]
    audit_md = [
        "# V2.1 Development Candidate Sampling Audit",
        "",
        f"- Audited extension: `{v1_labeled['dataset_version']}`",
        f"- Pair count: {len(v1_labeled['pairs'])}",
        f"- Human labels: {audit['human_label_counts']}",
        f"- Original-stratum label counts: {audit['human_labels_by_original_stratum']}",
        "- Conclusion: first sampler over-retrieved topical relatedness; "
        "strata were not semantic labels.",
        "- V1 labels were used only for this aggregate audit, never by the V2 sampler.",
        "- Holdout contents and labels were not used.",
        "",
        "## V2 surface diagnostics",
        "",
        f"- Candidate strata: {distribution}",
        f"- Candidate pool counts: {candidate_pool_counts}",
        f"- Diagnostics: {extension['surface_diagnostics']}",
        "- These are retrieval diagnostics, not semantic precision/accuracy.",
    ]
    output = {
        "artifacts/v2_1_dev_extension_candidates.json": json.dumps(
            v1_labeled, ensure_ascii=False, indent=2
        )
        + "\n",
        "artifacts/v2_1_development_eval.json": json.dumps(dev_eval, ensure_ascii=False, indent=2)
        + "\n",
        "artifacts/v2_1_development_eval.md": "\n".join(dev_md).rstrip() + "\n",
        "artifacts/v2_1_dev_candidate_sampling_audit.json": json.dumps(
            audit, ensure_ascii=False, indent=2
        )
        + "\n",
        "artifacts/v2_1_dev_candidate_sampling_audit.md": "\n".join(audit_md) + "\n",
        "artifacts/v2_1_dev_extension_v2_candidates.json": json.dumps(
            extension, ensure_ascii=False, indent=2
        )
        + "\n",
        "artifacts/v2_1_dev_extension_v2_candidates.md": extension_md,
    }
    return output


def main() -> None:
    outputs = build_outputs()
    for relative_path, content in outputs.items():
        (ROOT / relative_path).write_bytes(content.encode("utf-8"))
    extension = json.loads(outputs["artifacts/v2_1_dev_extension_v2_candidates.json"])
    print(
        json.dumps(
            {
                "dataset_version": extension["dataset_version"],
                "pair_count": extension["pair_count"],
                "membership_sha256": extension["membership_sha256"],
                "distribution": extension["sampling_distribution_by_stratum"],
                "candidate_pool_counts": extension["candidate_pool_counts_by_stratum"],
                "label_status": extension["label_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
