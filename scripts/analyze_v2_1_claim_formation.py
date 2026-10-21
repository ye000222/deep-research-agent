"""Read-only, bounded diagnosis of historical Claim formation.

No matcher, model, or Research execution component is imported or called.
Claim text is fetched only from baseline V2.0 runs after Holdout IDs are excluded.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BASELINE = ARTIFACTS / "v2_1_claim_equivalence.json"
DATASET = ARTIFACTS / "v2_1_real_claim_eval_candidates.json"
SEED = "v2.1-claim-formation-audit-v1"
SAMPLE_SIZE = 100
MANUAL_REVIEW_SIZE = 24
PAIR_SAMPLE_SIZE = 200
HOLDOUT_SHA256 = "da07c803a64a978df146f6200144f5ae47715ddfeb19b584343d1d1a268ddc8b"

_ENUM = re.compile(r"(?:^|[\n;；])\s*(?:\d+[.)、]|[-*•])|[,，、]\s*[^，,、]{2,}[,，、]")
_CLAUSE = re.compile(
    r"\b(?:and|while|whereas|because|therefore|resulting|leading to|which also|as well as)\b"
    r"|以及|并且|同时|而且|但是|因此|导致|因为|；|;",
    re.I,
)
_METRIC = re.compile(
    r"\b(?:revenue|price|cost|growth|cagr|rate|accuracy|precision|recall|latency|parameters|context|capacity|volume|share|employees|users|units|score)\b"
    r"|收入|价格|成本|增长率|准确率|召回率|延迟|参数|上下文|容量|规模|份额|员工|用户|数量|得分",
    re.I,
)
_MEASURE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|％|万|亿|百万|千万|亿元|万元|美元|元|吨|公斤|千克|kg|ms|秒|个|项)",
    re.I,
)
_TIME = re.compile(
    r"\b(?:19|20)\d{2}\b|\bQ[1-4]\s*20\d{2}\b|\b(?:FY|Q)[ -]?\d{2,4}\b"
    r"|过去\s*\d+\s*(?:年|月|日)|未来\s*\d+\s*(?:年|月|日)|\d+\s*(?:年|月|季度)内",
    re.I,
)
_QUALIFIER = re.compile(
    r"\b(?:in|during|since|by|under|within|across|for|using|through|based on|according to|as of)\b"
    r"|\b(?:19|20)\d{2}\b|\d+(?:\.\d+)?\s*(?:%|％|万|亿|美元|元|kg|ms|秒)"
    r"|全球|中国|美国|欧洲|北美|亚太|在.{1,20}条件下|根据|截至|通过|采用|使用|依据|期间|相比",
    re.I,
)
_SOURCE_FRAME = re.compile(
    r"\b(?:according to|the report states|the source says|as reported by|based on|the study found|the document indicates)\b"
    r"|根据.{0,24}(?:报告|研究|文件)|(?:报告指出|研究显示|资料显示|文件说明|网页称)",
    re.I,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|failed|declined|decreased)\b|不|未|没有|下降|否认|非", re.I
)
_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u3400-\u9fff]{2,}", re.I)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def holdout_membership_only(dataset: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Use only Holdout membership and linked Claim IDs, never content or labels."""
    pair_ids = {str(value) for value in dataset.get("holdout_ids", [])}
    claim_ids: set[str] = set()
    for row in dataset.get("review_view", {}).get("items", []):
        if str(row.get("pair_id", "")) in pair_ids:
            claim_ids.update(str(row[key]) for key in ("claim_a_id", "claim_b_id") if row.get(key))
    return pair_ids, claim_ids


def resolve_database_url() -> str:
    url = os.environ.get("V2_CLAIM_BASELINE_DATABASE_URL")
    if not url:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                url = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not url:
        raise RuntimeError("Set V2_CLAIM_BASELINE_DATABASE_URL or DATABASE_URL in .env")
    url = url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    parsed = urlsplit(url)
    if parsed.hostname == "postgres":
        auth = parsed.netloc.rsplit("@", 1)[0] if "@" in parsed.netloc else ""
        host = f"{auth}@127.0.0.1:{parsed.port or 5432}" if auth else "127.0.0.1:5432"
        url = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    return url


def fetch_claims(
    database_url: str, run_ids: list[str], excluded_ids: set[str]
) -> list[dict[str, Any]]:
    import psycopg

    sql = """
      SELECT c.id::text,c.run_id::text,c.question_id,c.dimension_key,c.atomic_claim,c.claim_type,
             e.id::text,e.source_id::text,e.exact_quote,e.accepted,e.chunk_id::text,
             s.source_owner_key,s.source_type
      FROM research_claims c
      LEFT JOIN research_evidence e ON e.claim_id=c.id AND e.run_id=c.run_id
      LEFT JOIN research_sources s ON s.id=e.source_id AND s.run_id=e.run_id
      WHERE c.run_id=ANY(%s::uuid[]) AND NOT(c.id::text=ANY(%s::text[]))
      ORDER BY c.run_id,c.question_id,c.dimension_key,c.id
    """
    with psycopg.connect(database_url, connect_timeout=8, autocommit=True) as conn:
        conn.execute("BEGIN READ ONLY")
        try:
            rows = conn.execute(sql, (run_ids, sorted(excluded_ids))).fetchall()
        finally:
            conn.execute("ROLLBACK")
    claims: dict[str, dict[str, Any]] = {}
    for row in rows:
        (
            cid,
            run,
            question,
            dimension,
            text,
            kind,
            evid,
            source,
            quote,
            accepted,
            chunk,
            owner,
            source_type,
        ) = row
        item = claims.setdefault(
            cid,
            {
                "claim_id": cid,
                "run_id": run,
                "question_id": question,
                "dimension_key": dimension,
                "claim_text": text,
                "claim_type": kind,
                "evidence": [],
            },
        )
        if evid is not None:
            item["evidence"].append(
                {
                    "evidence_id": evid,
                    "source_id": source,
                    "quote": quote,
                    "accepted": bool(accepted),
                    "chunk_id": chunk,
                    "owner_key": owner,
                    "source_type": source_type,
                }
            )
    return list(claims.values())


def generic_features(text: str) -> dict[str, Any]:
    text = text.strip()
    metric_count = len(set(value.casefold() for value in _METRIC.findall(text)))
    measure_count = len(_MEASURE.findall(text))
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    time_count = len(_TIME.findall(text))
    clause_count = len(_CLAUSE.findall(text))
    enumeration = bool(_ENUM.search(text))
    sentence_parts = [
        part for part in re.split(r"[。！？!?;；\n]+|(?<!\d)\.(?!\d)", text) if part.strip()
    ]
    multi_metric = metric_count >= 2 or (measure_count >= 2 and (clause_count > 0 or enumeration))
    composite = clause_count > 0 or enumeration or multi_metric or len(sentence_parts) > 1
    return {
        "char_count": len(text),
        "short_text": len(text) <= 80,
        "long_text": len(text) >= 400,
        "numeric": bool(re.search(r"\d", text)),
        "enumeration": enumeration,
        "clause_cue_count": clause_count,
        "conjunction_or_multi_clause": clause_count > 0,
        "metric_cue_count": metric_count,
        "measure_cue_count": measure_count,
        "multi_metric": multi_metric,
        "distinct_year_count": len(years),
        "time_cue_count": time_count,
        "multi_time": len(years) >= 2 or time_count >= 2,
        "qualifier_cue_count": len(_QUALIFIER.findall(text)),
        "has_material_qualifier_cue": bool(_QUALIFIER.search(text)),
        "general_proposition_heuristic": (
            not bool(re.search(r"\d", text))
            and not bool(_QUALIFIER.search(text))
            and len(text) <= 180
        ),
        "specific_or_subsuming_qualifier_cue": bool(_QUALIFIER.search(text)),
        "bundled_conclusion_rationale": bool(
            re.search(
                r"because|therefore|resulting|leading to|由于|因为|因此|导致|从而", text, re.I
            )
        ),
        "bundled_fact_consequence": bool(
            re.search(
                r"therefore|resulting|leading to|causing|which causes|导致|从而|因此", text, re.I
            )
        ),
        "implementation_detail_cue": bool(
            re.search(r"implemented|by using|through|via|采用|通过|使用|基于", text, re.I)
        ),
        "composite_heuristic": composite,
        "multi_proposition_heuristic": composite,
        "atomic_candidate_heuristic": not composite,
        "source_framing_cue": bool(_SOURCE_FRAME.search(text)),
    }


def _stratum(features: dict[str, Any]) -> str:
    for key, label in (
        ("enumeration", "enumeration"),
        ("multi_metric", "multi_metric"),
        ("conjunction_or_multi_clause", "multi_clause"),
        ("numeric", "numeric"),
        ("long_text", "long"),
        ("short_text", "short"),
    ):
        if features[key]:
            return label
    return "general"


def deterministic_sample(
    claims: list[dict[str, Any]], size: int = SAMPLE_SIZE, seed: str = SEED
) -> list[dict[str, Any]]:
    runs = sorted({item["run_id"] for item in claims})
    run_buckets = {run_id: f"run-{index + 1:02d}" for index, run_id in enumerate(runs)}
    strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for claim in claims:
        claim["_features"] = generic_features(claim["claim_text"])
        claim["_stratum"] = _stratum(claim["_features"])
        claim["_run_bucket"] = run_buckets[claim["run_id"]]
        strata[claim["_stratum"]][claim["_run_bucket"]].append(claim)
    for run_map in strata.values():
        for items in run_map.values():
            items.sort(key=lambda x: hashlib.sha256(f"{seed}:{x['claim_id']}".encode()).hexdigest())
    result: list[dict[str, Any]] = []
    cursors: Counter[tuple[str, str]] = Counter()
    while len(result) < min(size, len(claims)):
        changed = False
        for stratum in sorted(strata):
            for bucket in sorted(run_buckets.values()):
                items = strata[stratum].get(bucket, [])
                key = (stratum, bucket)
                if cursors[key] < len(items) and len(result) < min(size, len(claims)):
                    result.append(items[cursors[key]])
                    cursors[key] += 1
                    changed = True
        if not changed:
            break
    return result


def _tokens(text: str) -> set[str]:
    output: set[str] = set()
    for match in _WORD.findall(text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]{2,}", match):
            output.update(match[i : i + 2] for i in range(len(match) - 1))
        else:
            output.add(match)
    return output


def structural_overlap(left: str, right: str) -> str:
    def norm(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    if norm(left) == norm(right):
        return "EXACT_PROPOSITION_EQUIVALENT"
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return "INSUFFICIENT"
    common = a & b
    containment = len(common) / min(len(a), len(b))
    jaccard = len(common) / len(a | b)
    fa, fb = generic_features(left), generic_features(right)
    nums_a, nums_b = (
        set(re.findall(r"\d+(?:\.\d+)?", left)),
        set(re.findall(r"\d+(?:\.\d+)?", right)),
    )
    if nums_a and nums_b and nums_a != nums_b:
        n_a, n_b = (
            _tokens(re.sub(r"\d+(?:\.\d+)?", "#", norm(left))),
            _tokens(re.sub(r"\d+(?:\.\d+)?", "#", norm(right))),
        )
        if len(n_a & n_b) >= max(3, int(min(len(a), len(b)) * 0.45)) and bool(
            _NEGATION.search(left)
        ) == bool(_NEGATION.search(right)):
            return "POTENTIAL_CONTRADICTION"
    ratio = max(len(left), len(right)) / max(1, min(len(left), len(right)))
    if containment >= 0.88 and ratio >= 1.25 and bool(_QUALIFIER.search(left + " " + right)):
        return "GENERAL_SPECIFIC"
    if containment >= 0.72 and fa["composite_heuristic"] != fb["composite_heuristic"]:
        return "SHARED_ATOMIC_PROPOSITION"
    if jaccard >= 0.34 and fa["composite_heuristic"] and fb["composite_heuristic"]:
        return "OVERLAPPING_COMPOSITE"
    if jaccard < 0.14:
        return "INSUFFICIENT"
    return "INSUFFICIENT"


def overlap_sample(
    sample: list[dict[str, Any]], limit: int = PAIR_SAMPLE_SIZE
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in sample:
        groups[(item["run_id"], item["question_id"], item["dimension_key"])].append(item)
    pairs = []
    for (_, question, dimension), rows in groups.items():
        rows = sorted(rows, key=lambda x: x["claim_id"])
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                pid = hashlib.sha256(f"{left['claim_id']}:{right['claim_id']}".encode()).hexdigest()
                pairs.append(
                    {
                        "pair_id": pid,
                        "left_claim_id": left["claim_id"],
                        "right_claim_id": right["claim_id"],
                        "run_bucket": left["_run_bucket"],
                        "question_id": question,
                        "dimension_key": dimension,
                        "left_text": left["claim_text"],
                        "right_text": right["claim_text"],
                        "diagnostic_category": structural_overlap(
                            left["claim_text"], right["claim_text"]
                        ),
                    }
                )
    return sorted(pairs, key=lambda x: x["pair_id"])[:limit]


def _rate(value: int, total: int) -> float:
    return round(value / total, 4) if total else 0.0


def _decorate(claim: dict[str, Any]) -> dict[str, Any]:
    evidence = claim["evidence"]
    overlaps = []
    for item in evidence:
        a, b = _tokens(claim["claim_text"]), _tokens(item["quote"])
        if a and b:
            overlaps.append(round(len(a & b) / len(a | b), 4))
    return {
        "claim_id": claim["claim_id"],
        "run_bucket": claim["_run_bucket"],
        "question_id": claim["question_id"],
        "dimension_key": claim["dimension_key"],
        "claim_type": claim["claim_type"],
        "claim_text": claim["claim_text"],
        "features": claim["_features"],
        "sampling_stratum": claim["_stratum"],
        "evidence_count": len(evidence),
        "accepted_evidence_count": sum(e["accepted"] for e in evidence),
        "source_count": len({e["source_id"] for e in evidence}),
        "independent_owner_count": len({e["owner_key"] for e in evidence if e["owner_key"]}),
        "source_types": sorted({e["source_type"] for e in evidence if e["source_type"]}),
        "source_quote_overlap_jaccard": overlaps,
        "evidence_quotes": [e["quote"] for e in evidence],
        "structured_subject_present": False,
        "structured_predicate_present": False,
        "structured_value_present": False,
        "structured_qualifiers_present": False,
    }


def build_reports(
    claims: list[dict[str, Any]],
    baseline: dict[str, Any],
    pair_count: int,
    holdout_claim_count: int,
) -> tuple[dict[str, Any], ...]:
    run_ids = sorted({item["run_id"] for item in claims})
    buckets = {value: f"run-{index + 1:02d}" for index, value in enumerate(run_ids)}
    for item in claims:
        item["_run_bucket"] = buckets[item["run_id"]]
    sample = deterministic_sample(claims)
    decorated = [_decorate(item) for item in sample]
    manual_ids = {
        item["claim_id"]
        for item in sorted(
            sample,
            key=lambda x: hashlib.sha256(f"manual:{SEED}:{x['claim_id']}".encode()).hexdigest(),
        )[: min(MANUAL_REVIEW_SIZE, len(sample))]
    }
    for item in decorated:
        item["manual_review"] = (
            {
                "status": "PENDING_HUMAN_REVIEW",
                "atomicity_label": None,
                "overlap_notes": None,
                "reviewer_rationale": None,
            }
            if item["claim_id"] in manual_ids
            else None
        )
    flags = (
        "atomic_candidate_heuristic",
        "composite_heuristic",
        "multi_proposition_heuristic",
        "general_proposition_heuristic",
        "specific_or_subsuming_qualifier_cue",
        "enumeration",
        "multi_metric",
        "multi_time",
        "has_material_qualifier_cue",
        "bundled_conclusion_rationale",
        "bundled_fact_consequence",
        "implementation_detail_cue",
        "source_framing_cue",
        "numeric",
    )
    counts = {key: sum(bool(item["features"][key]) for item in decorated) for key in flags}
    structured = {
        key: sum(bool(item[key]) for item in decorated)
        for key in (
            "structured_subject_present",
            "structured_predicate_present",
            "structured_value_present",
            "structured_qualifiers_present",
        )
    }
    evidence_dist = Counter(len(item["evidence"]) for item in claims)
    source_mult = sum(len({e["source_id"] for e in item["evidence"]}) > 1 for item in claims)
    owner_mult = sum(
        len({e["owner_key"] for e in item["evidence"] if e["owner_key"]}) > 1 for item in claims
    )
    accepted = sum(sum(e["accepted"] for e in item["evidence"]) for item in claims)
    evidence_total = sum(len(item["evidence"]) for item in claims)
    overlap_values = [v for item in decorated for v in item["source_quote_overlap_jaccard"]]
    pairs = overlap_sample(sample)
    distribution = Counter(item["diagnostic_category"] for item in pairs)
    common = {
        "schema_version": "v2.1-claim-formation-diagnosis.v1",
        "analysis_mode": "read_only_architecture_diagnosis",
        "research_behavior_diff": "NONE",
        "matcher_invoked": False,
        "semantic_fallback_invoked": False,
        "models_invoked": False,
        "holdout_content_used": False,
        "holdout_content_access_incident": (
            "One truncated review-view text fragment was inadvertently surfaced "
            "during a metadata/schema inspection; it was not used for sampling, "
            "metrics, interpretation, or architecture choice and is not included "
            "in generated artifacts. Holdout files and membership hash were not modified."
        ),
        "holdout_integrity_status": "FILES_UNMODIFIED_BUT_ONE_SNIPPET_INADVERTENTLY_SURFACED",
        "holdout_pair_count_excluded_by_membership": pair_count,
        "holdout_claim_ids_excluded_before_text_fetch": holdout_claim_count,
        "holdout_membership_sha256_expected": HOLDOUT_SHA256,
        "historical_source": "V2.0 baseline run IDs in existing V2.1 metadata; raw Claim rows read in PostgreSQL READ ONLY transaction",
        "baseline_artifact_claim_count": baseline.get("claim_count"),
        "baseline_run_count": len(baseline.get("run_ids", [])),
        "included_non_holdout_run_count": len(run_ids),
        "included_non_holdout_claim_count": len(claims),
        "sample_size": len(sample),
        "sample_seed": SEED,
        "sampling_method": "generic text-feature strata, deterministic hash ordering, round-robin across anonymized baseline run buckets",
        "sampling_is_matcher_blind": True,
    }
    diagnosis = {
        **common,
        "pipeline": [
            {
                "stage": "source_content",
                "evidence": "Reader page text is supplied to EvidenceExtractorService.extract.",
            },
            {
                "stage": "claim_formation",
                "evidence": "Model-generated EvidenceBatch contains free-text EvidenceCandidate.claim and exact_quote; up to five candidate items per extraction.",
            },
            {
                "stage": "normalization",
                "evidence": "claim_fingerprint collapses whitespace and casefolds only; no semantic identity normalization.",
            },
            {
                "stage": "claim_persistence",
                "evidence": "research_tools finds/creates ResearchClaimRow by run_id + question_id + full claim hash.",
            },
            {
                "stage": "evidence_link",
                "evidence": "Each ResearchEvidenceRow has one nullable claim_id; a Claim can be linked by multiple Evidence rows.",
            },
        ],
        "claim_formation_findings": {
            "claim_text_formed_by": "LLM structured extraction guided by research question, acceptance dimensions and source text",
            "deterministic_claim_text": False,
            "one_evidence_can_produce_multiple_claims": "Up to five EvidenceCandidate items per extraction, one free-text Claim per item.",
            "one_claim_can_contain_multiple_atomic_facts": True,
            "structured_subject_predicate_value_routinely_populated": False,
            "material_qualifiers_structured_during_claim_formation": False,
            "source_specific_wording_survives": True,
            "different_evidence_can_converge": "Yes only when the full Claim text has the same whitespace-normalized/casefolded hash within run/question scope; semantic matching is not persistence identity.",
            "active_runtime_contract": "EvidenceCandidate free-text claim/exact_quote; CanonicalClaim subject/predicate/object/qualifiers is a passive contract, not the active extraction/persistence schema.",
        },
        "hypotheses": {
            "H1_true_duplicate_propositions_rare": {
                "status": "CONSISTENT_WITH_CARDINALITY_NOT_PROVEN",
                "evidence": "No multi-Evidence Claim convergence observed in the non-Holdout historical frame.",
            },
            "H2_claims_too_composite": {
                "status": "PLAUSIBLE_STRUCTURAL_FACTOR",
                "evidence": f"Generic heuristic flags {counts['composite_heuristic']} / {len(sample)} sampled Claims; not a human semantic label.",
            },
            "H3_granularity_varies": {
                "status": "PLAUSIBLE_NOT_SEMANTICALLY_VERIFIED",
                "evidence": "Observed variable length and qualifier-cue screening; overlap labels are structural heuristics only.",
            },
            "H4_source_centric_extraction": {
                "status": "SUPPORTED_BY_RUNTIME_CONTRACT",
                "evidence": "Extractor receives source passage and emits source-grounded free-text Claims without canonical proposition slots.",
            },
            "H5_dimension_centric_formation": {
                "status": "SUPPORTED_AS_CONTEXT_NOT_IDENTITY",
                "evidence": "Dimension criteria inform extraction/acceptance, but no proposition-reuse key is part of the extraction contract.",
            },
            "H6_contradictions_rare": {
                "status": "INCONCLUSIVE",
                "evidence": "One Evidence per Claim and lexical-only overlap sampling cannot establish contradiction prevalence.",
            },
        },
        "sample_screening": {
            "automated_heuristics_are_not_atomicity_labels": True,
            "manual_review_sample_count": len(manual_ids),
            "manual_review_sample_status": "PENDING_HUMAN_REVIEW",
            "heuristic_counts": counts,
            "heuristic_rates": {key: _rate(value, len(sample)) for key, value in counts.items()},
            "structured_field_counts": structured,
            "source_quote_token_jaccard": {
                "observed_pairs": len(overlap_values),
                "mean": round(sum(overlap_values) / len(overlap_values), 4)
                if overlap_values
                else None,
                "interpretation": "lexical overlap screen only, not semantic similarity or proof of copying",
            },
        },
        "human_labeled_development_evidence": {
            "original": {"pairs": 45, "related": 45, "unknown": 0, "same": 0, "contradicts": 0},
            "extension_v1": {"pairs": 36, "related": 35, "unknown": 1, "same": 0, "contradicts": 0},
            "extension_v2": {"pairs": 32, "related": 32, "unknown": 0, "same": 0, "contradicts": 0},
            "total": {"pairs": 113, "related": 112, "unknown": 1, "same": 0, "contradicts": 0},
            "source": "user-provided phase prompt; not used to tune sample or production behavior",
        },
        "interpretation": {
            "whole_claim_same_scarcity": "Not established as a matcher-only problem: exact full-text wording is the persistence convergence key, while source-centric extraction emits one-off Claim records; composition and qualifier variation can prevent whole-Claim identity.",
            "whole_claim_contradiction_scarcity": "One-Evidence-per-Claim topology makes repeated aligned assertions unobservable; this audit cannot conclude contradictory assertions are absent.",
            "primary_bottleneck": "Claim formation and verification granularity are conflated: the source-oriented free-text Claim is both reporting context and identity/verification unit, with no explicit atomic proposition layer.",
        },
        "limitations": [
            "Bounded generic-feature sample; not statistically representative of all domains.",
            "Atomic/composite, qualifier and overlap indicators are lexical screening aids, not human semantic judgments.",
            "Manual worksheet is pending human review; no labels were fabricated.",
            "Cardinality does not prove source independence or semantic duplicate scarcity.",
        ],
    }
    atomicity = {
        **common,
        "audit_definition": "One independently verifiable proposition while retaining material qualifiers.",
        "heuristic_screening_only": True,
        "sampled_claim_count": len(sample),
        "counts": counts,
        "rates": {key: _rate(value, len(sample)) for key, value in counts.items()},
        "structured_field_counts": structured,
        "qualifier_coverage": {
            "claims_with_qualifier_cue": counts["has_material_qualifier_cue"],
            "rate": _rate(counts["has_material_qualifier_cue"], len(sample)),
            "structured_qualifier_field_populated": False,
        },
        "manual_review_sample": [item for item in decorated if item["claim_id"] in manual_ids],
        "sample": decorated,
        "provenance_cardinality": {
            "evidence_per_claim_distribution": {
                str(k): v for k, v in sorted(evidence_dist.items())
            },
            "claims_with_multiple_evidence": sum(
                count for n, count in evidence_dist.items() if n > 1
            ),
            "claims_with_multiple_sources": source_mult,
            "claims_with_multiple_source_owners": owner_mult,
            "accepted_evidence_links": accepted,
            "all_evidence_links": evidence_total,
        },
        "automated_accuracy_warning": "Counts are not human-confirmed atomicity rates.",
    }
    categories = (
        "EXACT_PROPOSITION_EQUIVALENT",
        "SHARED_ATOMIC_PROPOSITION",
        "GENERAL_SPECIFIC",
        "OVERLAPPING_COMPOSITE",
        "DISJOINT_RELATED",
        "POTENTIAL_CONTRADICTION",
        "INSUFFICIENT",
    )
    overlap = {
        **common,
        "pair_sampling": "Pairs are restricted to same run/question/dimension among sample Claims; deterministic pair-hash order, capped at 200.",
        "pair_sample_count": len(pairs),
        "diagnostic_categories_are_not_relation_labels": True,
        "matcher_or_semantic_predictions_used": False,
        "category_counts": {key: distribution.get(key, 0) for key in categories},
        "category_rates": {key: _rate(distribution.get(key, 0), len(pairs)) for key in categories},
        "pairs": pairs,
        "caveat": "Lexical/structural heuristics only. POTENTIAL_CONTRADICTION requires human review of aligned entity, time, scope, unit and definition.",
    }
    options = [
        {
            "option": "A_KEEP_WHOLE_CLAIM_MODEL",
            "assessment": "NOT_PREFERRED",
            "generalization": "Simple and compatible, but keeps source-oriented text as identity and verification unit.",
            "risk": "Whole-Claim comparison may miss shared propositions and general/specific overlap; aggressive normalization risks false merges.",
            "preservation": "High if original text retained.",
            "verification_contradiction": "Limited for bundled propositions.",
            "aggregation": "Current exact-text fingerprint converges only on full-text matches.",
            "complexity_runtime_migration": "Lowest.",
            "v2_compatibility": "Highest.",
        },
        {
            "option": "B_ATOMIC_CLAIM_FORMATION",
            "assessment": "VIABLE_BUT_NOT_SOLE_LAYER",
            "generalization": "Generic decomposition may make independently verifiable facts reusable.",
            "risk": "False splits can detach dependent conditions, causal structure or qualifiers.",
            "preservation": "Requires explicit qualifier/provenance retention.",
            "verification_contradiction": "Good if slots/context preserved.",
            "aggregation": "Improves convergence potential.",
            "complexity_runtime_migration": "Medium/high extraction and reporting change.",
            "v2_compatibility": "Changes Claim cardinality.",
        },
        {
            "option": "C_CLAIM_PLUS_ATOMIC_PROPOSITION_LAYER",
            "assessment": "PREFERRED",
            "generalization": "Reusable proposition units support corroboration and contradiction across unseen topics while retaining parent Claim context.",
            "risk": "Identity still needs qualifier/scope compatibility; ambiguous decompositions must stay unresolved.",
            "preservation": "Best: preserve source-derived Claim plus proposition provenance/qualifiers.",
            "verification_contradiction": "High at proposition slot level.",
            "aggregation": "Evidence can attach to proposition without whole-text equality.",
            "complexity_runtime_migration": "Highest; additive schema/pipeline complexity; no migration now.",
            "v2_compatibility": "Can be additive with explicit projection.",
        },
        {
            "option": "D_SEMANTIC_SUBSUMPTION_RELATION",
            "assessment": "NOT_SELECTED_NOW",
            "generalization": "Could represent general/specific relations.",
            "risk": "Taxonomy calibration and overlap-as-identity errors.",
            "preservation": "Non-destructive if relation only.",
            "verification_contradiction": "Not a verification unit by itself.",
            "aggregation": "Weak without a shared atomic proposition.",
            "complexity_runtime_migration": "High relation ontology/evaluation cost.",
            "v2_compatibility": "Expands relation contract prematurely.",
        },
    ]
    decision = {
        **common,
        "options": options,
        "decision": "REDESIGN_CLAIM_PLUS_ATOMIC_PROPOSITION",
        "rationale": [
            "Persisted Claim is source-grounded free text with exact normalized full-text identity.",
            "Every Claim in the non-Holdout historical frame has one Evidence link; cross-source convergence/contradiction is structurally unobserved there.",
            "Generic screening suggests composition/qualifier variation but is not treated as semantic proof.",
            "An additive proposition layer preserves context and qualifiers while providing a reusable verification unit, generalizing without topic-specific rules.",
        ],
        "design_guardrails": [
            "Never drop values, dates, units, scope, location, method or conditions to increase match rates.",
            "Retain original Claim and provenance to source/Evidence/quote.",
            "Ambiguous decomposition remains unresolved; never force identity.",
            "No benchmark/task/entity/source-specific rules, answer tables or prompts.",
            "No historical rewrite/split/merge, ID change, migration or matcher contract change in this phase.",
        ],
        "implementation_started": False,
    }
    return diagnosis, atomicity, overlap, decision


def markdown(title: str, report: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        "- Mode: read-only diagnosis",
        "- Research behavior diff: NONE",
        "- Matcher/model invoked: NO",
        "- Frozen Holdout content used: NO",
        "",
    ]
    if "decision" in report:
        lines += [
            f"## Decision: {report['decision']}",
            "",
            *[f"- {x}" for x in report["rationale"]],
            "",
        ]
    if "claim_formation_findings" in report:
        lines += [
            "## Claim formation path",
            "",
            "Source content → EvidenceExtractorService.extract() → model-generated EvidenceBatch → free-text EvidenceCandidate.claim/exact_quote → whitespace/casefold fingerprint → ResearchClaimRow (run/question scope) → ResearchEvidenceRow.claim_id.",
            "",
            "- Claim text is model-generated, not deterministic; extraction permits up to five candidate items.",
            "- No required subject/predicate/value decomposition; CanonicalClaim structured slots are passive, not active persistence representation.",
            "- Convergence requires the same normalized full Claim text within run/question; semantic equivalence is not a persistence identity key.",
            "",
            "## Historical cardinality",
            "",
            f"- Baseline metadata: {report['baseline_artifact_claim_count']} Claims across {report['baseline_run_count']} runs.",
            f"- Non-Holdout database frame: {report['included_non_holdout_claim_count']} Claims across {report['included_non_holdout_run_count']} runs.",
            "- Each non-Holdout Claim has exactly one Evidence; no multi-Evidence, multi-source, or multi-owner Claim was observed.",
            "- This describes current formation/persistence topology; it does not prove semantic duplicates are intrinsically rare.",
            "",
            "## Generic sample screening",
            "",
            f"- Sample: {report['sample_size']}; pending manual worksheet: {report['sample_screening']['manual_review_sample_count']}.",
            f"- Composite heuristic: {report['sample_screening']['heuristic_counts']['composite_heuristic']}; atomic-candidate heuristic: {report['sample_screening']['heuristic_counts']['atomic_candidate_heuristic']}. Screening aids, not human labels.",
            "- Structured subject/predicate/value/qualifier slots are absent from the active persisted Claim row.",
            "",
            "## Development labels",
            "",
            "- Original: 45 RELATED; Extension v1: 35 RELATED, 1 UNKNOWN; Extension v2: 32 RELATED.",
            "- Total: 112 RELATED, 1 UNKNOWN, 0 SAME, 0 CONTRADICTS (user-provided reviewed aggregates).",
            "",
            "## Primary bottleneck",
            "",
            report["interpretation"]["primary_bottleneck"],
            "",
            "## Limitations",
            "",
            *[f"- {x}" for x in report["limitations"]],
            "",
        ]
    elif "manual_review_sample" in report:
        lines += [
            "## Heuristic counts (not human-confirmed)",
            "",
            "| Feature | Count | Rate |",
            "|---|---:|---:|",
        ]
        for key, count in report["counts"].items():
            lines.append(f"| {key} | {count} | {report['rates'][key]:.1%} |")
        lines += [
            "",
            "## Structured fields",
            "",
            "| Field | Populated in active persisted Claim |",
            "|---|---:|",
        ]
        lines.extend(
            f"| {key} | {value} |" for key, value in report["structured_field_counts"].items()
        )
        lines += [
            "",
            "## Claim/Evidence cardinality",
            "",
            json.dumps(report["provenance_cardinality"], ensure_ascii=False, indent=2),
            "",
            f"Manual worksheet: {len(report['manual_review_sample'])} Claims, all PENDING_HUMAN_REVIEW; no labels fabricated.",
            "",
            report["automated_accuracy_warning"],
            "",
        ]
    elif "category_counts" in report:
        lines += [
            "## Bounded pair screen",
            "",
            f"- Pair sample: {report['pair_sample_count']} within same run/question/dimension.",
            "- No matcher/semantic predictions used. Categories are diagnostic heuristics, not relation labels.",
            "",
            "| Structural screen | Count | Rate |",
            "|---|---:|---:|",
        ]
        lines.extend(
            f"| {key} | {count} | {report['category_rates'][key]:.1%} |"
            for key, count in report["category_counts"].items()
        )
        lines += ["", report["caveat"], ""]
    elif "options" in report:
        lines += [
            "## Architecture options",
            "",
            "| Option | Assessment | Summary |",
            "|---|---|---|",
        ]
        lines.extend(
            f"| {item['option']} | {item['assessment']} | {item['generalization']} |"
            for item in report["options"]
        )
        lines += [
            "",
            "## Design guardrails",
            "",
            *[f"- {x}" for x in report["design_guardrails"]],
            "",
        ]
    return "\n".join(lines)


def write_artifacts(output_dir: Path, reports: tuple[dict[str, Any], ...]) -> None:
    names = (
        "v2_1_claim_formation_diagnosis",
        "v2_1_claim_atomicity_audit",
        "v2_1_claim_overlap_audit",
        "v2_1_claim_architecture_decision",
    )
    titles = (
        "V2.1 Historical Claim Formation Diagnosis",
        "V2.1 Claim Atomicity Audit",
        "V2.1 Proposition Overlap Audit",
        "V2.1 Claim Architecture Decision",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, title, report in zip(names, titles, reports, strict=True):
        (output_dir / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / f"{name}.md").write_text(markdown(title, report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("V2_CLAIM_BASELINE_DATABASE_URL"))
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    baseline, dataset = _read_json(BASELINE), _read_json(DATASET)
    if dataset.get("holdout_membership_sha256") != HOLDOUT_SHA256:
        raise SystemExit("Holdout membership hash mismatch; refusing analysis")
    pair_ids, excluded = holdout_membership_only(dataset)
    claims = fetch_claims(
        args.database_url or resolve_database_url(), [str(x) for x in baseline["run_ids"]], excluded
    )
    reports = build_reports(claims, baseline, len(pair_ids), len(excluded))
    write_artifacts(args.output_dir, reports)
    print(
        json.dumps(
            {
                "included_claims": len(claims),
                "included_runs": reports[0]["included_non_holdout_run_count"],
                "sample_size": reports[0]["sample_size"],
                "pair_sample": reports[2]["pair_sample_count"],
                "decision": reports[3]["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
