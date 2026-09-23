"""Phase 14.4: read-only causal attribution of Phase 14.3 evidence-aware enrichment.

Phase 14.3 proved the Phase 14.2 pipeline (ResearchContext -> Query Enrichment ->
SearchTarget -> Provider) really fires, but the candidate arm showed no quality
gain (coverage 0.4524 -> 0.3929, accepted evidence 14.33 -> 10.67, open gap 3 -> 3)
while executing fewer searches (42 -> 34 per run).  This module answers *why* by
attributing the observed deltas to one of: 14.1/14.2 redundancy, coarse hint
mapping, execution gating, or a downstream acceptance bottleneck.

Strictly additive and read-only:

  * issues only ``SELECT`` statements against the persisted run/event/gap
    projections created by the finished Phase 14.3 experiment;
  * imports nothing from the production research path;
  * runs no benchmark and mutates no business data.

The pure analysis functions (no database access) are unit tested directly; the
database layer only shapes rows into the plain dicts those functions consume.

Analysis sections (1:1 with the Phase 14.4 spec):
  A  data set                -- run ids, benchmark/version, revisions
  B  14.1 vs 14.2 overlap    -- no-op / overlap / unique hint contribution
  C  research funnel         -- stage counts, conversion and drop-off rates
  D  search suppression      -- why search.query.started dropped 42 -> 34
  E  query specificity       -- did enrichment shrink result recall?
  F  evidence outcome        -- failure-bucket migration
  G  deterministic bottleneck-- why candidate coverage SD == 0
  H  context stickiness      -- stale ResearchContext usage
  I  cost / efficiency       -- tokens, provider requests, evidence yield
  J  root cause              -- Observation / Evidence / Inference / Confidence
  K  decision                -- one primary + optional secondary case (1-4)
  L  recommendation          -- next-phase suggestions only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psycopg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = REPOSITORY_ROOT / "artifacts" / "phase14_4_attribution.json"
DEFAULT_MARKDOWN_OUT = REPOSITORY_ROOT / "artifacts" / "phase14_4_attribution.md"

EXPERIMENT_MODE_BASELINE = "baseline"
EXPERIMENT_MODE_CANDIDATE = "evidence_aware"

#: Arm labels used across the report.  Kept separate from the experiment_mode
#: values ("baseline"/"evidence_aware") so every consumer reads one key space.
ARM_BASELINE = "baseline"
ARM_CANDIDATE = "candidate"

ENRICHED_EVENT = "research.query.context.enriched"
CONTEXT_RECORDED_EVENT = "research.context.recorded"
CLOSURE_FEEDBACK_EVENT = "closure.feedback.generated"
NEED_REFINED_EVENT = "research.need.refined"
QUERY_CANDIDATE_EVENT = "research.query_candidate.generated"
SEARCH_STARTED_EVENT = "search.query.started"
SEARCH_COMPLETED_EVENT = "search.completed"
SEARCH_REUSED_EVENT = "search.reused"
SOURCE_SPACE_EXHAUSTED_EVENT = "search.source_space_exhausted"
FEEDBACK_BLOCKED_EVENT = "feedback.query.execution.blocked"
PROVIDER_EXECUTION_EVENT = "provider.execution.started"
PROVIDER_FAILED_EVENT = "provider.attempt.failed"
SEARCH_TOOL_CALL_FAILED = "search.tool_call.failed"
CANDIDATE_CREATED_EVENT = "candidate.created"
CANDIDATE_SKIPPED_EVENT = "candidate.skipped"
CANDIDATE_DISPATCH_SKIPPED_EVENT = "candidate.dispatch_skipped"
READABLE_EVENT = "source.readable"
EVIDENCE_EXTRACTED_EVENT = "evidence.extracted"
EVIDENCE_SELECTION_STARTED_EVENT = "evidence.selection_started"
ALIGNMENT_COMPLETED_EVENT = "evidence.alignment.completed"
CLOSURE_COMPLETED_EVENT = "gap.closure.completed"
GAP_TRANSITION_EVENT = "gap.closure.transition"
GAP_OPENED_EVENT = "gap.opened"

#: Suppression buckets used to explain a missing search opportunity, in the
#: priority order the Phase 14.4 spec lists them.
SUPPRESSION_REASONS = (
    "query_already_executed",
    "reuse_only",
    "search_budget_exhausted",
    "logical_query_budget_exhausted",
    "source_space_exhausted",
    "provider_degraded",
    "feedback_execution_limit",
    "iteration_limit",
    "planner_no_target",
    "duplicate_query",
    "other",
)

#: Failure buckets for evidence outcome attribution (spec section 6).
FAILURE_BUCKETS = (
    "CLAIM_NOT_SUPPORTED",
    "NO_INDEPENDENT_VALIDATION",
    "NO_QUANTITATIVE_SUPPORT",
    "SOURCE_QUALITY_LOW",
    "DIMENSION_MISMATCH",
    "UNKNOWN",
)

#: rejection_reason -> failure bucket.  Values observed in the Phase 14.3 data:
#: source_role_mismatch, evidence_relevance_below_threshold,
#: claim_quote_entailment_failed, quote_not_located_for_graph,
#: prompt_injection_detected.
REJECTION_TO_BUCKET = {
    "claim_quote_entailment_failed": "CLAIM_NOT_SUPPORTED",
    "quote_not_located_for_graph": "CLAIM_NOT_SUPPORTED",
    "evidence_relevance_below_threshold": "DIMENSION_MISMATCH",
    "source_role_mismatch": "SOURCE_QUALITY_LOW",
    "prompt_injection_detected": "SOURCE_QUALITY_LOW",
}

#: feedback failure_reason -> failure bucket (closure.feedback.generated refs).
FEEDBACK_REASON_TO_BUCKET = {
    "claim_not_verified": "CLAIM_NOT_SUPPORTED",
    "claim_not_supported": "CLAIM_NOT_SUPPORTED",
    "insufficient_evidence": "NO_INDEPENDENT_VALIDATION",
    "independent_source_deficit": "NO_INDEPENDENT_VALIDATION",
    "no_valid_path": "SOURCE_QUALITY_LOW",
    "unknown": "UNKNOWN",
}

#: refined_need_type / missing_evidence_type -> failure bucket.
NEED_TYPE_TO_BUCKET = {
    "claim_verification": "CLAIM_NOT_SUPPORTED",
    "verification_support": "CLAIM_NOT_SUPPORTED",
    "dimension_targeted_evidence": "DIMENSION_MISMATCH",
    "dimension_aligned_evidence": "DIMENSION_MISMATCH",
    "independent_source": "NO_INDEPENDENT_VALIDATION",
}

#: run-level research_stop_reason -> suppression bucket.  Run termination is
#: not a per-search block, so these entries are recorded as run-level evidence
#: inside the matching budget bucket.
STOP_REASON_TO_BUCKET = {
    "deadline_exhausted": "search_budget_exhausted",
    "provider_request_budget_exhausted": "search_budget_exhausted",
    "search_budget_exhausted": "search_budget_exhausted",
    "logical_query_budget_exhausted": "logical_query_budget_exhausted",
    "iteration_limit": "iteration_limit",
}

CASE_SUMMARIES = {
    1: "Main-path enrichment mostly redundant with 14.1 candidate adaptation",
    2: "Enrichment intent correct, hint mapping too coarse",
    3: "Search adaptation works, bottleneck is downstream (acceptance/closure)",
    4: "Execution/budget gate masks enrichment (suppressed search opportunities)",
}


def case_summary(case_id: int) -> str:
    return CASE_SUMMARIES.get(case_id, "unknown")


# ---------------------------------------------------------------------------
# Pure text helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lower-case whitespace/punctuation tokenization; CJK runs stay intact."""

    normalised = text.lower()
    for separator in (",", ".", ";", ":", "(", ")", "[", "]", "{", "}", '"', "'", "/", "\\", "|"):
        normalised = normalised.replace(separator, " ")
    return [token for token in normalised.split() if token]


def hint_present(hint: str, text: str) -> bool:
    """True when *hint* already appears in *text* (token or substring form).

    English hints are compared as whole tokens so "paper" does not match
    "newspaper"; non-ASCII hints (CJK) fall back to substring containment so a
    translated work item such as "论文" is still detected inside a long CJK run.
    """

    cleaned = hint.strip().lower()
    if not cleaned:
        return False
    if cleaned.isascii() and cleaned.replace("-", "").isalnum():
        return cleaned in tokenize(text)
    return cleaned in text.lower()


def hints_present_in_text(hints: list[str], text: str) -> list[str]:
    return [hint for hint in hints if hint_present(hint, text)]


def ratio(numerator: float, denominator: float) -> float:
    """Zero-denominator-safe ratio rounded to 4 decimals."""

    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def mean_or_zero(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _counts_from(events: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(event.get("event_type", "")) for event in events)


def event_count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(1 for event in events if event.get("event_type") == event_type)


# ---------------------------------------------------------------------------
# Arm validation
# ---------------------------------------------------------------------------


def validate_arms(baseline_ids: list[str], candidate_ids: list[str]) -> list[str]:
    """Return a list of human-readable problems with the requested arm sizes."""

    problems: list[str] = []
    if not baseline_ids:
        problems.append("baseline arm has no run ids")
    if not candidate_ids:
        problems.append("candidate arm has no run ids")
    if baseline_ids and candidate_ids and len(baseline_ids) != len(candidate_ids):
        problems.append(
            f"arm size mismatch: baseline={len(baseline_ids)} candidate={len(candidate_ids)}"
        )
    overlap = set(baseline_ids) & set(candidate_ids)
    if overlap:
        problems.append(f"run ids present in both arms: {sorted(overlap)}")
    duplicates = [
        run_id for run_id, count in Counter(baseline_ids + candidate_ids).items() if count > 1
    ]
    if duplicates:
        problems.append(f"duplicate run ids: {sorted(duplicates)}")
    return problems


# ---------------------------------------------------------------------------
# B. 14.1 vs 14.2 overlap
# ---------------------------------------------------------------------------

OVERLAP_METRIC_KEYS = (
    "total_enrichment_attempts",
    "query_changed_count",
    "no_op_count",
    "hint_count_total",
    "added_hint_total",
    "avg_added_hint_count",
    "median_added_hint_count",
    "exact_no_op_rate",
    "partial_overlap_rate",
    "fully_novel_enrichment_rate",
    "full_overlap_rate",
    "mean_overlap_ratio",
)


def compute_enrichment_overlap(
    enrichments: list[dict[str, Any]],
    candidate_queries: list[dict[str, Any]],
    context_hints_by_need: dict[str, list[str]],
) -> dict[str, Any]:
    """Attribute 14.2 enrichment against what 14.1 already materialised.

    ``enrichments``: one dict per ``research.query.context.enriched`` event with
    keys ``question_id``, ``research_need_id``, ``original_query``,
    ``enriched_query``, ``applied_hints`` (list), ``hint_count``,
    ``applied_hint_count``, ``query_changed``.
    ``candidate_queries``: ``research.query_candidate.generated`` events with
    ``question_id``, ``query_text``, ``context_hints`` (list).
    ``context_hints_by_need``: need_id -> requested hint set.
    """

    total = len(enrichments)
    no_op = 0
    changed = 0
    added_counts: list[float] = []
    hint_total = 0
    added_total = 0

    # Hints the 14.1 QueryCandidate already materialised inside its query text.
    materialised_by_question: dict[str, set[str]] = defaultdict(set)
    for query in candidate_queries:
        question_id = str(query.get("question_id") or "")
        text = str(query.get("query_text") or "")
        for hint in query.get("context_hints") or []:
            hint_text = str(hint)
            if hint_present(hint_text, text):
                materialised_by_question[question_id].add(hint_text.lower())

    # Hints applied by 14.2 to the *main* search target, grouped by question.
    # Enriched events do not always carry ``applied_hints`` explicitly, so the
    # applied set is derived as "requested by the resolved context, absent from
    # the original query, present in the enriched query".
    applied_by_question: dict[str, set[str]] = defaultdict(set)
    enrichments_with_need = 0
    explicit_hint_events = 0
    derived_hint_events = 0
    context_missing_events = 0

    for enrichment in enrichments:
        question_id = str(enrichment.get("question_id") or "")
        need_id = str(enrichment.get("research_need_id") or "")
        original_query = str(enrichment.get("original_query") or "")
        enriched_query = str(enrichment.get("enriched_query") or "")
        explicit = [str(hint) for hint in enrichment.get("applied_hints") or []]
        if explicit:
            applied = explicit
            explicit_hint_events += 1
        else:
            requested = context_hints_by_need.get(need_id, [])
            applied = [
                hint
                for hint in requested
                if hint_present(hint, enriched_query) and not hint_present(hint, original_query)
            ]
            if requested:
                derived_hint_events += 1
            else:
                context_missing_events += 1
        hint_count = int(enrichment.get("hint_count") or 0)
        applied_count = int(enrichment.get("applied_hint_count") or 0)
        hint_total += hint_count
        added_total += applied_count
        added_counts.append(float(applied_count))
        if bool(enrichment.get("query_changed")):
            changed += 1
        if hint_count > 0 and applied_count == 0:
            no_op += 1
        if applied:
            applied_by_question[question_id].update(hint.lower() for hint in applied)
        if need_id:
            enrichments_with_need += 1

    # Per-question overlap between the 14.1 materialised hints and the hints
    # 14.2 kept appending on the main path.
    overlap_rows: list[dict[str, Any]] = []
    partial = 0
    novel = 0
    full = 0
    for question_id in sorted(set(applied_by_question) | set(materialised_by_question)):
        applied_set = applied_by_question.get(question_id, set())
        materialised_set = materialised_by_question.get(question_id, set())
        shared = sorted(applied_set & materialised_set)
        overlap_ratio = ratio(len(shared), len(applied_set))
        if applied_set:
            if overlap_ratio == 0.0:
                novel += 1
            elif overlap_ratio >= 1.0:
                full += 1
            else:
                partial += 1
        overlap_rows.append(
            {
                "question_id": question_id,
                "main_path_added_hints": sorted(applied_set),
                "candidate_materialised_hints": sorted(materialised_set),
                "shared_hints": shared,
                "unique_to_main_path": sorted(applied_set - materialised_set),
                "overlap_ratio": overlap_ratio,
            }
        )

    compared = sum(1 for row in overlap_rows if row["main_path_added_hints"])
    overlap_ratios = [
        float(row["overlap_ratio"]) for row in overlap_rows if row["main_path_added_hints"]
    ]
    vocabulary = sorted(
        {hint.lower() for hints in context_hints_by_need.values() for hint in hints}
    )
    unique_contribution = sorted(
        {hint for row in overlap_rows for hint in row["unique_to_main_path"]}
    )
    return {
        "total_enrichment_attempts": total,
        "query_changed_count": changed,
        "no_op_count": no_op,
        "hint_count_total": hint_total,
        "added_hint_total": added_total,
        "avg_added_hint_count": mean_or_zero(added_counts),
        "median_added_hint_count": median_or_zero(added_counts),
        "exact_no_op_rate": ratio(no_op, total),
        "partial_overlap_rate": ratio(partial, compared) if compared else 0.0,
        "fully_novel_enrichment_rate": ratio(novel, compared) if compared else 0.0,
        "full_overlap_rate": ratio(full, compared) if compared else 0.0,
        "mean_overlap_ratio": mean_or_zero(overlap_ratios),
        "enrichments_with_need_link": enrichments_with_need,
        "explicit_hint_events": explicit_hint_events,
        "derived_hint_events": derived_hint_events,
        "context_missing_events": context_missing_events,
        "questions_compared": compared,
        "resolved_hint_sets": context_hints_by_need,
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "per_question_overlap": overlap_rows,
        "unique_hint_contribution": unique_contribution,
        "context_recorded_hint_sets": sorted(
            {tuple(sorted(hints)) for hints in context_hints_by_need.values()}
        ),
    }


# ---------------------------------------------------------------------------
# C. Research funnel
# ---------------------------------------------------------------------------

#: (label, event type) pairs; tables supply the evidence/acceptance stages.
FUNNEL_STAGES: tuple[tuple[str, str], ...] = (
    ("closure_feedback", CLOSURE_FEEDBACK_EVENT),
    ("research_need_refined", NEED_REFINED_EVENT),
    ("research_context_recorded", CONTEXT_RECORDED_EVENT),
    ("query_enrichment_attempt", ENRICHED_EVENT),
    ("query_actually_changed", "enriched:changed"),
    ("search_query_started", SEARCH_STARTED_EVENT),
    ("provider_execution", PROVIDER_EXECUTION_EVENT),
    ("search_completed", SEARCH_COMPLETED_EVENT),
    ("candidate_urls", CANDIDATE_CREATED_EVENT),
    ("readable_source", READABLE_EVENT),
    ("evidence_extraction", EVIDENCE_EXTRACTED_EVENT),
    ("candidate_evidence", "table:candidate_evidence"),
    ("accepted_evidence", "table:accepted_evidence"),
    ("evidence_alignment", ALIGNMENT_COMPLETED_EVENT),
    ("closure_evaluation", CLOSURE_COMPLETED_EVENT),
    ("gap_transition", GAP_TRANSITION_EVENT),
)


def build_funnel(
    stage_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Build per-arm funnel rows with conversion/drop-off rates.

    ``stage_counts``: arm -> stage key -> count (already aggregated over runs).
    """

    out: dict[str, Any] = {}
    for arm, counts in stage_counts.items():
        rows: list[dict[str, Any]] = []
        previous: int | None = None
        for label, _source in FUNNEL_STAGES:
            count = int(counts.get(label, 0))
            conversion: float | None = None
            if previous is not None and previous != 0:
                conversion = ratio(count, previous)
            dropoff = None if conversion is None else round(1.0 - conversion, 4)
            rows.append(
                {
                    "stage": label,
                    "count": count,
                    "conversion_rate": conversion,
                    "dropoff_rate": dropoff,
                }
            )
            previous = count
        out[arm] = rows
    return out


def funnel_first_loss(funnel: dict[str, Any]) -> dict[str, Any]:
    """First stage where the candidate conversion drops clearly below baseline."""

    baseline_rows = funnel.get("baseline", [])
    candidate_rows = funnel.get("candidate", [])
    for index in range(1, min(len(baseline_rows), len(candidate_rows))):
        base_row = baseline_rows[index]
        cand_row = candidate_rows[index]
        base_conv = base_row.get("conversion_rate")
        cand_conv = cand_row.get("conversion_rate")
        if base_conv is None or cand_conv is None:
            continue
        if base_row["count"] == 0 or cand_row["count"] == 0:
            continue
        delta = round(float(cand_conv) - float(base_conv), 4)
        if delta <= -0.1:
            return {
                "stage": cand_row["stage"],
                "baseline_conversion": base_conv,
                "candidate_conversion": cand_conv,
                "conversion_delta": delta,
                "baseline_count": base_row["count"],
                "candidate_count": cand_row["count"],
            }
    return {
        "stage": "",
        "baseline_conversion": None,
        "candidate_conversion": None,
        "conversion_delta": 0.0,
        "baseline_count": 0,
        "candidate_count": 0,
    }


# ---------------------------------------------------------------------------
# D. Search suppression attribution
# ---------------------------------------------------------------------------


def classify_suppression(
    events: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    """Bucket each *missing* search opportunity using event-order evidence.

    ``events``: full event list of one run (``run_seq``, ``event_type``,
    ``refs``, ``metrics``); ``usage``: run usage snapshot scalars.
    """

    buckets: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    reuse_events = 0
    started_new = 0
    started_reused = 0
    exhausted_seq: list[int] = []

    ordered = sorted(events, key=lambda event: int(event.get("run_seq") or 0))
    for event in ordered:
        event_type = str(event.get("event_type") or "")
        raw_refs = event.get("refs")
        refs: dict[str, Any] = raw_refs if isinstance(raw_refs, dict) else {}
        raw_metrics = event.get("metrics")
        metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
        run_seq = int(event.get("run_seq") or 0)
        if event_type == SEARCH_STARTED_EVENT:
            reused = str(metrics.get("reused", "false")).lower() == "true"
            if reused:
                started_reused += 1
                buckets["reuse_only"] += 1
            else:
                started_new += 1
        elif event_type == SEARCH_REUSED_EVENT:
            reuse_events += 1
            buckets["query_already_executed"] += 1
            evidence["query_already_executed"].append(
                f"seq={run_seq} question={refs.get('question_id')} reused results"
            )
        elif event_type == SOURCE_SPACE_EXHAUSTED_EVENT:
            buckets["source_space_exhausted"] += 1
            exhausted_seq.append(run_seq)
            evidence["source_space_exhausted"].append(
                f"seq={run_seq} question={refs.get('question_id')} no new urls"
            )
        elif event_type == FEEDBACK_BLOCKED_EVENT:
            buckets["feedback_execution_limit"] += 1
            evidence["feedback_execution_limit"].append(
                f"seq={run_seq} question={refs.get('question_id')} "
                f"stop_reason={refs.get('stop_reason')} "
                f"feedback_execution_count={metrics.get('feedback_execution_count')}"
            )
        elif event_type == CANDIDATE_SKIPPED_EVENT and refs.get("reason") == "duplicate":
            buckets["duplicate_query"] += 1
            evidence["duplicate_query"].append(f"seq={run_seq} duplicate candidate url")
        elif event_type == PROVIDER_FAILED_EVENT:
            buckets["provider_degraded"] += 1
            evidence["provider_degraded"].append(
                f"seq={run_seq} error_code={metrics.get('error_code') or refs.get('error_code')}"
            )

    source_space_by_question = usage.get("source_space_exhausted_by_question") or {}
    strategy_exhausted_by_question = usage.get("query_strategy_exhausted_by_question") or {}
    if isinstance(source_space_by_question, dict):
        for question_id, flag in source_space_by_question.items():
            if flag:
                evidence["source_space_exhausted"].append(
                    f"question={question_id} flagged in usage"
                )
    if isinstance(strategy_exhausted_by_question, dict):
        for question_id, flag in strategy_exhausted_by_question.items():
            if flag:
                buckets["logical_query_budget_exhausted"] += 1
                evidence["logical_query_budget_exhausted"].append(
                    f"question={question_id} query strategy exhausted"
                )

    stop_reason = str(usage.get("research_stop_reason") or "")
    if stop_reason:
        stop_bucket = STOP_REASON_TO_BUCKET.get(stop_reason)
        if stop_bucket:
            buckets[stop_bucket] += 1
            evidence[stop_bucket].append(
                f"run-level stop_reason={stop_reason} (whole run terminated)"
            )
        else:
            evidence["other"].append(f"run stop_reason={stop_reason}")

    return {
        "buckets": {reason: buckets.get(reason, 0) for reason in SUPPRESSION_REASONS},
        "started_new_searches": started_new,
        "started_reused_searches": started_reused,
        "search_reused_events": reuse_events,
        "exhaustion_run_seqs": exhausted_seq,
        "stop_reason": stop_reason,
        "evidence": {key: values[:12] for key, values in sorted(evidence.items())},
    }


def summarize_suppression(per_run: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run suppression buckets into arm level totals."""

    totals: dict[str, dict[str, int]] = {}
    for _run_id, run in per_run.items():
        arm = str(run.get("arm") or "unknown")
        arm_totals = totals.setdefault(arm, {reason: 0 for reason in SUPPRESSION_REASONS})
        for reason in SUPPRESSION_REASONS:
            arm_totals[reason] += int(run["suppression"]["buckets"].get(reason, 0))
    return totals


# ---------------------------------------------------------------------------
# E. Query specificity / recall
# ---------------------------------------------------------------------------

PAIR_PREFIX_TOKENS = 5


def query_core(text: str, prefix_tokens: int = PAIR_PREFIX_TOKENS) -> str:
    """A stable pairing key: the leading tokens shared by a query family.

    Appended hint/adaptation suffixes live at the *end* of every observed query,
    so the leading tokens identify the underlying research question variant.
    """

    tokens = tokenize(text)
    return " ".join(tokens[:prefix_tokens])


def pair_query_outcomes(
    baseline_queries: list[dict[str, Any]],
    candidate_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pair baseline/candidate searches by (question, leading-token core).

    Each query dict carries ``question_id``, ``query``, ``query_family``,
    ``reused``, ``result_count``, ``unique_urls``, ``readable_sources``,
    ``accepted_evidence``, ``enriched`` (bool).
    """

    def bucketize(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (str(row.get("question_id") or ""), query_core(str(row.get("query") or "")))
            buckets[key].append(row)
        return buckets

    base_buckets = bucketize(baseline_queries)
    cand_buckets = bucketize(candidate_queries)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(rows),
            "reused": sum(1 for row in rows if row.get("reused")),
            "result_count": sum(int(row.get("result_count") or 0) for row in rows),
            "zero_result": sum(1 for row in rows if int(row.get("result_count") or 0) == 0),
            "unique_urls": sum(int(row.get("unique_urls") or 0) for row in rows),
            "readable_sources": sum(int(row.get("readable_sources") or 0) for row in rows),
            "accepted_evidence": sum(int(row.get("accepted_evidence") or 0) for row in rows),
            "avg_results": mean_or_zero(
                [float(row.get("result_count") or 0) for row in rows]
            ),
        }

    pairs: list[dict[str, Any]] = []
    for key in sorted(set(base_buckets) & set(cand_buckets)):
        base = aggregate(base_buckets[key])
        cand = aggregate(cand_buckets[key])
        pairs.append(
            {
                "question_id": key[0],
                "query_core": key[1],
                "baseline": base,
                "candidate": cand,
                "result_count_delta": cand["result_count"] - base["result_count"],
                "avg_results_delta": round(cand["avg_results"] - base["avg_results"], 4),
            }
        )

    def overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
        aggregated = aggregate(rows)
        aggregated["distinct_cores"] = len(
            {query_core(str(row.get("query") or "")) for row in rows}
        )
        aggregated["zero_result_rate"] = ratio(aggregated["zero_result"], aggregated["queries"])
        return aggregated

    baseline_overall = overall(baseline_queries)
    candidate_overall = overall(candidate_queries)
    paired_base = [pair["baseline"]["result_count"] for pair in pairs]
    paired_cand = [pair["candidate"]["result_count"] for pair in pairs]
    paired_base_total = sum(paired_base)
    paired_cand_total = sum(paired_cand)
    # Query-count-weighted per-query averages on the paired cores: the raw
    # result totals above are confounded by the differing number of searches
    # each arm ran on the same core, so the avg delta is the recall signal.
    paired_base_avg = ratio(
        paired_base_total, sum(pair["baseline"]["queries"] for pair in pairs)
    )
    paired_cand_avg = ratio(
        paired_cand_total, sum(pair["candidate"]["queries"] for pair in pairs)
    )
    return {
        "baseline_overall": baseline_overall,
        "candidate_overall": candidate_overall,
        "paired_core_count": len(pairs),
        "paired_baseline_results": paired_base_total,
        "paired_candidate_results": paired_cand_total,
        "paired_result_delta": paired_cand_total - paired_base_total,
        "paired_baseline_avg_results": paired_base_avg,
        "paired_candidate_avg_results": paired_cand_avg,
        "paired_avg_result_delta": round(paired_cand_avg - paired_base_avg, 4),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# F. Evidence failure migration
# ---------------------------------------------------------------------------


def compute_failure_migration(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_feedback: list[dict[str, Any]],
    candidate_feedback: list[dict[str, Any]],
    enriched_candidate_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare evidence rejection buckets and closure feedback buckets.

    ``*_rows``: research_evidence rows (``accepted``, ``rejection_reason``).
    ``*_feedback``: closure.feedback.generated events (``failure_reason``).
    ``enriched_candidate_rows``: candidate evidence rows discovered through a
    query that actually ran the enriched provider text; when provided, its
    distribution is reported separately (section F enrichment chain).
    """

    def evidence_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
        buckets: Counter[str] = Counter()
        accepted = 0
        for row in rows:
            if row.get("accepted"):
                accepted += 1
                continue
            reason = str(row.get("rejection_reason") or "")
            buckets[REJECTION_TO_BUCKET.get(reason, "UNKNOWN")] += 1
        total = len(rows)
        rejected = total - accepted
        distribution = {
            bucket: {
                "count": buckets.get(bucket, 0),
                "share_of_rows": ratio(buckets.get(bucket, 0), total),
                "share_of_rejections": ratio(buckets.get(bucket, 0), rejected),
            }
            for bucket in FAILURE_BUCKETS
        }
        return {
            "rows": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": ratio(accepted, total),
            "rejection_buckets": distribution,
        }

    def feedback_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
        buckets: Counter[str] = Counter()
        for row in rows:
            reason = str(row.get("failure_reason") or "")
            buckets[FEEDBACK_REASON_TO_BUCKET.get(reason, "UNKNOWN")] += 1
        total = len(rows)
        return {
            "events": total,
            "buckets": {
                bucket: {
                    "count": buckets.get(bucket, 0),
                    "share": ratio(buckets.get(bucket, 0), total),
                }
                for bucket in FAILURE_BUCKETS
            },
        }

    baseline_evidence = evidence_distribution(baseline_rows)
    candidate_evidence = evidence_distribution(candidate_rows)
    migrations: list[dict[str, Any]] = []
    for bucket in FAILURE_BUCKETS:
        base_share = float(baseline_evidence["rejection_buckets"][bucket]["share_of_rejections"])
        cand_share = float(candidate_evidence["rejection_buckets"][bucket]["share_of_rejections"])
        delta = round(cand_share - base_share, 4)
        if abs(delta) >= 0.02:
            migrations.append(
                {
                    "bucket": bucket,
                    "baseline_share": base_share,
                    "candidate_share": cand_share,
                    "share_delta": delta,
                    "direction": "increased" if delta > 0 else "decreased",
                }
            )
    migrations.sort(key=lambda row: abs(float(row["share_delta"])), reverse=True)
    result: dict[str, Any] = {
        "baseline_evidence": baseline_evidence,
        "candidate_evidence": candidate_evidence,
        "baseline_feedback": feedback_distribution(baseline_feedback),
        "candidate_feedback": feedback_distribution(candidate_feedback),
        "migrations": migrations,
        "migration_detected": bool(migrations),
    }
    if enriched_candidate_rows is not None:
        result["candidate_enriched_evidence"] = evidence_distribution(enriched_candidate_rows)
    return result


# ---------------------------------------------------------------------------
# G. Deterministic bottleneck
# ---------------------------------------------------------------------------

#: Usage/quality scalars compared across same-arm runs to find a pinned value.
BOTTLENECK_FIELDS: tuple[tuple[str, str], ...] = (
    ("usage", "searches"),
    ("usage", "search_reuses"),
    ("usage", "logical_queries"),
    ("usage", "iterations"),
    ("usage", "candidate_urls"),
    ("usage", "pages_fetched"),
    ("usage", "feedback_execution_count"),
    ("usage", "search_provider_requests"),
    ("usage", "source_space_exhausted_by_question"),
    ("usage", "query_strategy_exhausted_by_question"),
    ("usage", "research_stop_reason"),
    ("quality", "coverage"),
    ("quality", "accepted_evidence"),
    ("quality", "critical_gaps"),
)


def detect_deterministic_bottleneck(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Find scalars identical across all runs of one arm (SD == 0 signals).

    ``runs``: per-run dicts with ``usage`` and ``quality`` sub-dicts.
    """

    if len(runs) < 2:
        return {
            "run_count": len(runs),
            "pinned_fields": [],
            "identical_vectors": False,
            "coverage_sd_zero": False,
            "bottleneck_candidates": [],
        }
    pinned: list[dict[str, Any]] = []
    for section, field in BOTTLENECK_FIELDS:
        values = [json.dumps(run.get(section, {}).get(field), sort_keys=True) for run in runs]
        if len(set(values)) == 1:
            pinned.append({"section": section, "field": field, "value": json.loads(values[0])})
    coverages = [
        float(run.get("quality", {}).get("coverage") or 0.0) for run in runs
    ]
    coverage_sd_zero = len(set(coverages)) == 1
    pinned_names = {f"{row['section']}.{row['field']}" for row in pinned}
    bottleneck_candidates: list[str] = []
    if "usage.source_space_exhausted_by_question" in pinned_names:
        bottleneck_candidates.append("source_space_exhausted_by_question identical across runs")
    if "usage.query_strategy_exhausted_by_question" in pinned_names:
        bottleneck_candidates.append("query_strategy_exhausted_by_question identical across runs")
    if "usage.feedback_execution_count" in pinned_names:
        bottleneck_candidates.append("feedback_execution_count identical across runs")
    if "usage.searches" in pinned_names:
        bottleneck_candidates.append("searches identical across runs")
    if coverage_sd_zero:
        bottleneck_candidates.append("coverage identical across runs (SD == 0)")
    return {
        "run_count": len(runs),
        "pinned_fields": pinned,
        "identical_vectors": len(pinned) == len(BOTTLENECK_FIELDS),
        "coverage_sd_zero": coverage_sd_zero,
        "coverage_values": coverages,
        "bottleneck_candidates": bottleneck_candidates,
    }


# ---------------------------------------------------------------------------
# H. Context stickiness
# ---------------------------------------------------------------------------


def compute_context_stickiness(
    enrichments: list[dict[str, Any]],
    context_records: list[dict[str, Any]],
    need_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure how long a recorded ResearchContext keeps steering search.

    ``enrichments``: enriched events (``run_seq``, ``question_id``,
    ``research_need_id``).
    ``context_records``: research.context.recorded events (``run_seq``,
    ``question_id``, ``research_need_id``).
    ``need_events``: research.need.refined events (``run_seq``, ``need_id``,
    ``question_id``, ``refined_need_type``, ``missing_evidence_type``).
    """

    records_by_need: dict[str, list[int]] = defaultdict(list)
    for record in context_records:
        need_id = str(record.get("research_need_id") or "")
        if need_id:
            records_by_need[need_id].append(int(record.get("run_seq") or 0))

    usage_by_need: Counter[str] = Counter()
    for enrichment in enrichments:
        need_id = str(enrichment.get("research_need_id") or "")
        if need_id:
            usage_by_need[need_id] += 1

    # Old context used after a *newer* need for the same question appears.
    needs_by_question: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for need in need_events:
        question_id = str(need.get("question_id") or "")
        need_id = str(need.get("need_id") or "")
        if question_id and need_id:
            needs_by_question[question_id].append((int(need.get("run_seq") or 0), need_id))

    stale_count = 0
    stale_examples: list[dict[str, Any]] = []
    for enrichment in enrichments:
        question_id = str(enrichment.get("question_id") or "")
        need_id = str(enrichment.get("research_need_id") or "")
        run_seq = int(enrichment.get("run_seq") or 0)
        if not question_id or not need_id:
            continue
        newer = [
            seq
            for seq, other in needs_by_question.get(question_id, [])
            if other != need_id and seq < run_seq
        ]
        if newer:
            stale_count += 1
            if len(stale_examples) < 10:
                stale_examples.append(
                    {
                        "run_seq": run_seq,
                        "question_id": question_id,
                        "used_need_id": need_id,
                        "newer_need_run_seq": max(newer),
                    }
                )

    per_need_lifetime: list[dict[str, Any]] = []
    for need_id in sorted(set(records_by_need) | set(usage_by_need)):
        recorded_seqs = records_by_need.get(need_id, [])
        uses = usage_by_need.get(need_id, 0)
        per_need_lifetime.append(
            {
                "research_need_id": need_id,
                "recorded_count": len(recorded_seqs),
                "first_recorded_run_seq": min(recorded_seqs) if recorded_seqs else None,
                "usage_count": uses,
                "uses_per_record": ratio(uses, max(len(recorded_seqs), uses)),
            }
        )
    return {
        "context_records": len(context_records),
        "enrichment_uses": len(enrichments),
        "distinct_needs_used": len([need for need, count in usage_by_need.items() if count]),
        "avg_uses_per_need": mean_or_zero([float(count) for count in usage_by_need.values()]),
        "stale_context_usage_count": stale_count,
        "stale_examples": stale_examples,
        "per_need_lifetime": per_need_lifetime,
    }


# ---------------------------------------------------------------------------
# I. Incremental value (counterfactual approximation)
# ---------------------------------------------------------------------------


def compute_incremental_value(
    overlap: dict[str, Any],
    suppression: dict[str, dict[str, int]],
    specificity: dict[str, Any],
) -> dict[str, Any]:
    """Approximate what removing the 14.2 main-path enrichment would cost.

    This is a *counterfactual approximation from observed evidence only*: no
    benchmark is re-run and no simulation is attempted.
    """

    total = int(overlap.get("total_enrichment_attempts") or 0)
    no_op = int(overlap.get("no_op_count") or 0)
    unique_hints = overlap.get("unique_hint_contribution") or []
    if total == 0:
        return {
            "estimable": False,
            "reason": "no enrichment attempts observed in the candidate arm",
            "14_1_only_adaptation_coverage": None,
            "14_2_additional_adaptation_coverage": None,
            "14_2_no_op_rate": None,
            "14_2_unique_hint_contribution": [],
        }
    unique_contribution_rate = ratio(
        len(unique_hints), int(overlap.get("vocabulary_size") or 0) or len(unique_hints) or 1
    )
    # Counterfactual approximation only: the 14.1 candidate path materialises
    # the same vocabulary, so removing the 14.2 main-path application would
    # leave the hint vocabulary in circulation via 14.1.
    return {
        "estimable": True,
        "14_1_only_adaptation_coverage": round(1.0 - unique_contribution_rate, 4),
        "14_2_additional_adaptation_coverage": unique_contribution_rate,
        "14_2_no_op_rate": ratio(no_op, total),
        "14_2_unique_hint_contribution": unique_hints,
        "vocabulary_size": int(overlap.get("vocabulary_size") or 0),
        "suppression_context": suppression,
        "paired_recall_delta": specificity.get("paired_result_delta"),
        "caveat": (
            "Counterfactual computed from observed overlap/suppression/recall evidence; "
            "the 14.2-off scenario was not executed."
        ),
    }


# ---------------------------------------------------------------------------
# J. Cost / efficiency
# ---------------------------------------------------------------------------


def compute_cost_efficiency(
    baseline_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare cost and yield; distinguishes cheaper-but-worse from unnoticed value."""

    def arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
        totals = {
            "runtime_seconds": sum(
                float(row["usage"].get("runtime_seconds") or 0.0) for row in rows
            ),
            "tokens": sum(float(row["usage"].get("model_tokens") or 0.0) for row in rows),
            "provider_requests": sum(
                float(row["usage"].get("search_provider_requests") or 0.0) for row in rows
            ),
            "provider_failures": sum(
                float(row["usage"].get("search_provider_failures") or 0.0) for row in rows
            ),
            "search_started": sum(float(row["usage"].get("searches") or 0.0) for row in rows),
            "search_completed": sum(
                float(row.get("stage_counts", {}).get("search_completed") or 0.0)
                for row in rows
            ),
            "readable_sources": sum(
                float(row.get("stage_counts", {}).get("readable_source") or 0.0)
                for row in rows
            ),
            "evidence_extracted": sum(
                float(row.get("stage_counts", {}).get("evidence_extraction") or 0.0)
                for row in rows
            ),
            "accepted_evidence": sum(
                float(row["quality"].get("accepted_evidence") or 0.0) for row in rows
            ),
            "candidate_evidence": sum(
                float(row.get("stage_counts", {}).get("candidate_evidence") or 0.0)
                for row in rows
            ),
            "coverage_sum": sum(float(row["quality"].get("coverage") or 0.0) for row in rows),
        }
        return totals

    def per_unit(totals: dict[str, float]) -> dict[str, float]:
        return {
            "accepted_evidence_per_search": ratio(
                totals["accepted_evidence"], totals["search_started"]
            ),
            "accepted_evidence_per_provider_request": ratio(
                totals["accepted_evidence"], totals["provider_requests"]
            ),
            "coverage_per_search": ratio(totals["coverage_sum"], totals["search_started"]),
            "coverage_per_token": round(
                ratio(totals["coverage_sum"] * 1000.0, totals["tokens"]), 6
            ),
            "runtime_per_accepted_evidence": ratio(
                totals["runtime_seconds"], totals["accepted_evidence"]
            ),
            "tokens_per_accepted_evidence": ratio(totals["tokens"], totals["accepted_evidence"]),
            "readable_per_search": ratio(totals["readable_sources"], totals["search_started"]),
        }

    baseline_totals = arm(baseline_runs)
    candidate_totals = arm(candidate_runs)
    baseline_per_unit = per_unit(baseline_totals)
    candidate_per_unit = per_unit(candidate_totals)
    deltas = {
        key: round(candidate_per_unit[key] - baseline_per_unit[key], 6)
        for key in baseline_per_unit
    }
    cheaper = candidate_totals["tokens"] < baseline_totals["tokens"] or (
        candidate_totals["provider_requests"] < baseline_totals["provider_requests"]
    )
    quality_down = candidate_totals["accepted_evidence"] < baseline_totals["accepted_evidence"]
    efficiency_up = sum(1 for value in deltas.values() if value > 0) >= max(len(deltas) // 2, 1)
    if cheaper and quality_down:
        verdict = "cheaper but lower quality"
    elif not cheaper and not efficiency_up:
        verdict = "no cost advantage and no efficiency gain"
    elif efficiency_up and quality_down:
        verdict = "efficiency metrics mixed while quality declines (coverage insensitive?)"
    else:
        verdict = "efficiency comparable"
    return {
        "baseline_totals": baseline_totals,
        "candidate_totals": candidate_totals,
        "baseline_per_unit": baseline_per_unit,
        "candidate_per_unit": candidate_per_unit,
        "delta_per_unit": deltas,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# K. Decision (case selection)
# ---------------------------------------------------------------------------


def decide_cases(
    overlap: dict[str, Any],
    suppression: dict[str, dict[str, int]],
    failure_migration: dict[str, Any],
    deterministic: dict[str, Any],
    specificity: dict[str, Any],
    cost: dict[str, Any],
) -> dict[str, Any]:
    """Select one primary case (1-4) and at most one secondary case.

    Decisions are appended in evidence-strength order: redundancy, gating,
    coarse mapping, then the downstream fallback.
    """

    decisions: list[tuple[int, str]] = []

    exact_no_op = float(overlap.get("exact_no_op_rate") or 0.0)
    uniqueness = len(overlap.get("unique_hint_contribution") or [])
    mean_overlap = float(overlap.get("mean_overlap_ratio") or 0.0)
    overlap_high = exact_no_op >= 0.1 or (mean_overlap >= 0.9 and uniqueness == 0)
    hint_sets = overlap.get("context_recorded_hint_sets") or []
    candidate_suppression = suppression.get("candidate", {})
    baseline_suppression = suppression.get("baseline", {})
    candidate_sse = int(candidate_suppression.get("source_space_exhausted", 0))
    baseline_sse = int(baseline_suppression.get("source_space_exhausted", 0))
    suppressed_total = sum(int(value) for value in candidate_suppression.values())
    baseline_suppressed_total = sum(int(value) for value in baseline_suppression.values())
    gating_heavy = candidate_sse > baseline_sse and candidate_sse >= 3
    active_buckets = [
        bucket
        for bucket, row in (
            failure_migration.get("candidate_evidence", {}).get("rejection_buckets") or {}
        ).items()
        if int(row.get("count") or 0) > 0
    ]
    coarse_mapping = bool(hint_sets) and len(hint_sets) == 1 and len(active_buckets) >= 2
    quality_down = float(cost.get("candidate_totals", {}).get("accepted_evidence") or 0.0) < float(
        cost.get("baseline_totals", {}).get("accepted_evidence") or 0.0
    )
    migration = bool(failure_migration.get("migration_detected"))
    recall_down = int(specificity.get("paired_result_delta") or 0) < 0 or (
        float(specificity.get("paired_avg_result_delta") or 0.0) < 0
    )

    if overlap_high:
        decisions.append(
            (
                1,
                f"exact_no_op_rate={exact_no_op:.3f}, "
                f"mean_overlap_ratio={mean_overlap:.3f}, "
                f"unique_hint_contribution={uniqueness} "
                "(main-path hints repeat what 14.1 already materialised)",
            )
        )
    if gating_heavy:
        decisions.append(
            (
                4,
                f"candidate suppression signatures total {suppressed_total} "
                f"vs baseline {baseline_suppressed_total} with source-space exhaustion "
                f"{baseline_sse} -> {candidate_sse} "
                "(search opportunities intercepted by reuse/exhaustion/feedback gates)",
            )
        )
    if coarse_mapping and (recall_down or migration):
        decisions.append(
            (
                2,
                "single hint set {} reused across {} failure buckets {} while paired "
                "result delta={} (mapping is coarse)".format(
                    [sorted(hint_set) for hint_set in hint_sets],
                    len(active_buckets),
                    active_buckets,
                    specificity.get("paired_result_delta"),
                ),
            )
        )
    if not overlap_high and not gating_heavy and not coarse_mapping and quality_down:
        decisions.append(
            (
                3,
                "search-side adaptation present but downstream acceptance/closure unchanged",
            )
        )

    if not decisions:
        decisions.append((1, "no decisive attribution signal; default to redundancy review"))

    primary = decisions[0]
    secondary = decisions[1] if len(decisions) > 1 else None
    return {
        "primary_case": primary[0],
        "primary_case_summary": case_summary(primary[0]),
        "primary_rationale": primary[1],
        "secondary_case": secondary[0] if secondary else None,
        "secondary_case_summary": case_summary(secondary[0]) if secondary else None,
        "secondary_rationale": secondary[1] if secondary else None,
        "all_signals": [{"case": case, "rationale": why} for case, why in decisions],
        "deterministic_bottleneck_note": deterministic.get("bottleneck_candidates", []),
        "caveat": (
            "Case selection is an analytical judgement over observed Phase 14.3 evidence, "
            "not an executed counterfactual."
        ),
    }


# ---------------------------------------------------------------------------
# Database layer (read-only)
# ---------------------------------------------------------------------------


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _split_run_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _run_profile(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT status, termination_reason,
               usage_snapshot, budget_snapshot, quality_snapshot,
               normalized_goal
        FROM research_runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return {"run_id": run_id, "found": False, "usage": {}, "budget": {}, "quality": {}}
    status, termination_reason, usage, budget, quality, goal = row
    usage = usage if isinstance(usage, dict) else {}
    budget = budget if isinstance(budget, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    return {
        "run_id": run_id,
        "found": True,
        "status": status,
        "termination_reason": termination_reason,
        "normalized_goal": goal,
        "experiment_mode": usage.get("experiment_mode"),
        "budget_tier": budget.get("tier"),
        "source_revision": budget.get("source_revision"),
        "plan_template_run_id": budget.get("plan_template_run_id"),
        "usage": usage,
        "budget": budget,
        "quality": quality,
    }


def _fetch_events(
    connection: psycopg.Connection, run_id: str, event_types: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    if event_types:
        rows = connection.execute(
            """
            SELECT run_seq, event_type, refs, metrics
            FROM agent_events
            WHERE run_id = %s AND event_type = ANY(%s)
            ORDER BY run_seq
            """,
            (run_id, list(event_types)),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT run_seq, event_type, refs, metrics
            FROM agent_events
            WHERE run_id = %s
            ORDER BY run_seq
            """,
            (run_id,),
        ).fetchall()
    events: list[dict[str, Any]] = []
    for run_seq, event_type, refs, metrics in rows:
        events.append(
            {
                "run_seq": int(run_seq),
                "event_type": str(event_type),
                "refs": refs if isinstance(refs, dict) else {},
                "metrics": metrics if isinstance(metrics, dict) else {},
            }
        )
    return events


def _fetch_search_queries(connection: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT sq.id, sq.question_id, sq.query, sq.status, sq.result_count,
               count(DISTINCT lower(r.url)) AS unique_urls
        FROM research_search_queries sq
        LEFT JOIN research_search_results r ON r.search_query_id = sq.id
        WHERE sq.run_id = %s
        GROUP BY sq.id, sq.question_id, sq.query, sq.status, sq.result_count
        ORDER BY sq.created_at
        """,
        (run_id,),
    ).fetchall()
    queries: list[dict[str, Any]] = []
    for query_id, question_id, query, status, result_count, unique_urls in rows:
        queries.append(
            {
                "id": str(query_id),
                "question_id": str(question_id or ""),
                "query": str(query or ""),
                "status": str(status or ""),
                "result_count": int(result_count or 0),
                "unique_urls": int(unique_urls or 0),
                "readable_sources": 0,
                "accepted_evidence": 0,
                "reused": False,
                "query_family": "",
                "enriched": False,
            }
        )
    return queries


def _fetch_evidence_rows(connection: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT ev.question_id, ev.accepted, ev.rejection_reason, ev.source_id,
               rs.canonical_url
        FROM research_evidence ev
        LEFT JOIN research_sources rs ON rs.id = ev.source_id
        WHERE ev.run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "question_id": str(question_id or ""),
            "accepted": bool(accepted),
            "rejection_reason": str(rejection_reason or ""),
            "source_id": str(source_id or ""),
            "source_url": str(canonical_url or ""),
        }
        for question_id, accepted, rejection_reason, source_id, canonical_url in rows
    ]


def analyze_run(
    connection: psycopg.Connection,
    run_id: str,
) -> dict[str, Any]:
    """Collect every Phase 14.4 signal for one run (read-only)."""

    profile = _run_profile(connection, run_id)
    if not profile.get("found"):
        return {"profile": profile, "run_id": run_id}
    usage = profile["usage"]
    quality = profile["quality"]

    events = _fetch_events(connection, run_id)
    search_events = [
        event
        for event in events
        if event["event_type"]
        in (SEARCH_STARTED_EVENT, SEARCH_COMPLETED_EVENT, PROVIDER_EXECUTION_EVENT)
    ]
    enriched_events = [
        {
            "run_seq": event["run_seq"],
            "question_id": str(event["refs"].get("question_id") or ""),
            "research_need_id": str(event["refs"].get("research_need_id") or ""),
            "original_query": str(event["refs"].get("original_query") or ""),
            "enriched_query": str(event["refs"].get("enriched_query") or ""),
            "applied_hints": [
                str(hint) for hint in (event["refs"].get("applied_hints") or [])
            ]
            if isinstance(event["refs"].get("applied_hints"), list)
            else [],
            "hint_count": int(event["metrics"].get("hint_count") or 0),
            "applied_hint_count": int(event["metrics"].get("applied_hint_count") or 0),
            "query_changed": bool(event["metrics"].get("query_changed")),
        }
        for event in _fetch_events(connection, run_id, (ENRICHED_EVENT,))
    ]
    context_records = [
        {
            "run_seq": event["run_seq"],
            "question_id": str(event["refs"].get("question_id") or ""),
            "research_need_id": str(event["refs"].get("research_need_id") or ""),
            "hint_count": int(event["metrics"].get("hint_count") or 0),
        }
        for event in _fetch_events(connection, run_id, (CONTEXT_RECORDED_EVENT,))
    ]
    need_events = [
        {
            "run_seq": event["run_seq"],
            "question_id": str(event["refs"].get("question_id") or ""),
            "need_id": str(event["refs"].get("research_need_id") or ""),
            "refined_need_type": str(event["refs"].get("refined_need_type") or ""),
            "missing_evidence_type": str(event["refs"].get("missing_evidence_type") or ""),
        }
        for event in _fetch_events(connection, run_id, (NEED_REFINED_EVENT,))
    ]
    query_candidates: list[dict[str, Any]] = []
    for event in _fetch_events(connection, run_id, (QUERY_CANDIDATE_EVENT,)):
        refs = event["refs"]
        research_context = refs.get("research_context")
        context_hints: list[str] = []
        if isinstance(research_context, dict):
            hints = research_context.get("query_hints")
            if isinstance(hints, list):
                context_hints = [str(hint) for hint in hints]
        query_candidates.append(
            {
                "run_seq": event["run_seq"],
                "question_id": str(refs.get("question_id") or ""),
                "query_text": str(refs.get("query_text") or ""),
                "context_hints": context_hints,
            }
        )
    feedback_events = [
        {
            "run_seq": event["run_seq"],
            "question_id": str(event["refs"].get("question_id") or ""),
            "failure_reason": str(event["refs"].get("failure_reason") or ""),
            "requirement_type": str(event["refs"].get("requirement_type") or ""),
        }
        for event in _fetch_events(connection, run_id, (CLOSURE_FEEDBACK_EVENT,))
    ]

    queries = _fetch_search_queries(connection, run_id)
    evidence_rows = _fetch_evidence_rows(connection, run_id)

    # Link search.started -> persisted query rows (reuse flag + family +
    # whether the provider text was the post-enrichment query).
    started_by_tool_call: dict[str, dict[str, Any]] = {}
    for event in _fetch_events(connection, run_id, (SEARCH_STARTED_EVENT,)):
        tool_call_id = str(event["refs"].get("tool_call_id") or "")
        if tool_call_id:
            started_by_tool_call[tool_call_id] = event
    queries_by_id = {query["id"]: query for query in queries}
    for query_id, tool_call_id in connection.execute(
        "SELECT id, tool_call_id FROM research_search_queries WHERE run_id = %s",
        (run_id,),
    ).fetchall():
        query = queries_by_id.get(str(query_id))
        started_event = started_by_tool_call.get(str(tool_call_id))
        if query is None or started_event is None:
            continue
        query["reused"] = (
            str(started_event["metrics"].get("reused", "false")).lower() == "true"
        )
        query["query_family"] = str(started_event["refs"].get("query_family") or "")
        query["enriched"] = any(
            enrichment["enriched_query"] == str(started_event["refs"].get("query") or "")
            for enrichment in enriched_events
        )

    # Link readable sources and evidence rows back to the search query that
    # produced them.  candidate.created carries query_id + candidate_id; the
    # source.readable event carries candidate_id plus the page URLs (its
    # ``source_id`` field is persisted as null), and research_evidence joins to
    # research_sources.canonical_url, so the chain is
    # candidate_id -> query_id and evidence.source_url -> readable url -> query.
    # A URL is attributed to the first query that discovered it (setdefault) so
    # one evidence row is never double-counted across queries.
    candidate_to_query: dict[str, str] = {}
    for event in _fetch_events(connection, run_id, (CANDIDATE_CREATED_EVENT,)):
        candidate_id = str(event["refs"].get("candidate_id") or "")
        candidate_query_id = str(event["refs"].get("query_id") or "")
        if candidate_id and candidate_query_id:
            candidate_to_query.setdefault(candidate_id, candidate_query_id)
    url_to_query: dict[str, str] = {}
    readable_pairs: set[tuple[str, str]] = set()
    for event in _fetch_events(connection, run_id, (READABLE_EVENT,)):
        refs = event["refs"]
        linked_query_id = candidate_to_query.get(str(refs.get("candidate_id") or ""))
        url = str(
            refs.get("normalized_url") or refs.get("final_url") or refs.get("requested_url") or ""
        )
        if not url or not linked_query_id:
            continue
        readable_pairs.add((linked_query_id, url))
        for key in ("normalized_url", "final_url", "requested_url"):
            value = str(refs.get(key) or "")
            if value:
                url_to_query.setdefault(value, linked_query_id)
    readable_sources_by_query: Counter[str] = Counter(
        query_id for query_id, _url in readable_pairs
    )
    evidence_by_query: Counter[str] = Counter()
    accepted_by_query: Counter[str] = Counter()
    for row in evidence_rows:
        evidence_query_id = url_to_query.get(row["source_url"])
        if not evidence_query_id:
            continue
        evidence_by_query[evidence_query_id] += 1
        if row["accepted"]:
            accepted_by_query[evidence_query_id] += 1
    for query in queries:
        query["readable_sources"] = readable_sources_by_query.get(query["id"], 0)
        query["candidate_evidence"] = evidence_by_query.get(query["id"], 0)
        query["accepted_evidence"] = accepted_by_query.get(query["id"], 0)
    enriched_query_ids = {query["id"] for query in queries if query.get("enriched")}
    enriched_evidence_rows = [
        row for row in evidence_rows if url_to_query.get(row["source_url"]) in enriched_query_ids
    ]

    counters = _counts_from([{"event_type": event["event_type"]} for event in events])
    stage_counts = {
        "closure_feedback": event_count(events, CLOSURE_FEEDBACK_EVENT),
        "research_need_refined": event_count(events, NEED_REFINED_EVENT),
        "research_context_recorded": event_count(events, CONTEXT_RECORDED_EVENT),
        "query_enrichment_attempt": len(enriched_events),
        "query_actually_changed": sum(1 for item in enriched_events if item["query_changed"]),
        "search_query_started": event_count(events, SEARCH_STARTED_EVENT),
        "provider_execution": event_count(events, PROVIDER_EXECUTION_EVENT),
        "search_completed": event_count(events, SEARCH_COMPLETED_EVENT),
        "candidate_urls": event_count(events, CANDIDATE_CREATED_EVENT),
        "readable_source": event_count(events, READABLE_EVENT),
        "evidence_extraction": event_count(events, EVIDENCE_EXTRACTED_EVENT),
        "candidate_evidence": len(evidence_rows),
        "accepted_evidence": sum(1 for row in evidence_rows if row["accepted"]),
        "evidence_alignment": event_count(events, ALIGNMENT_COMPLETED_EVENT),
        "closure_evaluation": event_count(events, CLOSURE_COMPLETED_EVENT),
        "gap_transition": event_count(events, GAP_TRANSITION_EVENT),
    }
    suppression = classify_suppression(events, usage)
    context_hints_by_need: dict[str, list[str]] = {}
    context_by_question = usage.get("research_context_by_question")
    if isinstance(context_by_question, dict):
        for _question_id, entry in context_by_question.items():
            if not isinstance(entry, dict):
                continue
            hints = entry.get("query_hints")
            need_id = str(entry.get("research_need_id") or "")
            if need_id and isinstance(hints, list):
                context_hints_by_need[need_id] = [str(hint) for hint in hints]
    overlap = compute_enrichment_overlap(enriched_events, query_candidates, context_hints_by_need)
    stickiness = compute_context_stickiness(enriched_events, context_records, need_events)
    wall_clock_cap = profile.get("budget", {}).get("max_wall_clock_seconds")
    deadline_remaining = usage.get("deadline_remaining")
    runtime_seconds: float | None = None
    if isinstance(wall_clock_cap, (int, float)) and isinstance(deadline_remaining, (int, float)):
        runtime_seconds = round(float(wall_clock_cap) - float(deadline_remaining), 3)
    usage_scalars = {
        "runtime_seconds": runtime_seconds,
        "deadline_remaining": deadline_remaining,
        "searches": usage.get("searches"),
        "search_reuses": usage.get("search_reuses"),
        "logical_queries": usage.get("logical_queries"),
        "iterations": usage.get("iterations"),
        "candidate_urls": usage.get("candidate_urls"),
        "pages_fetched": usage.get("pages_fetched"),
        "feedback_execution_count": usage.get("feedback_execution_count"),
        "search_provider_requests": usage.get("search_provider_requests"),
        "search_provider_failures": usage.get("search_provider_failures"),
        "model_tokens": usage.get("model_tokens"),
        "source_space_exhausted_by_question": usage.get("source_space_exhausted_by_question"),
        "query_strategy_exhausted_by_question": usage.get("query_strategy_exhausted_by_question"),
        "research_stop_reason": usage.get("research_stop_reason"),
        "zero_yield_pages": usage.get("zero_yield_pages"),
        "duplicate_pages_skipped": usage.get("duplicate_pages_skipped"),
    }
    return {
        "run_id": run_id,
        "profile": profile,
        "usage": usage_scalars,
        "quality": {
            "coverage": quality.get("coverage"),
            "priority_one_coverage": quality.get("priority_one_coverage"),
            "accepted_evidence": quality.get("accepted_evidence"),
            "candidate_evidence": quality.get("candidate_evidence"),
            "critical_gaps": quality.get("critical_gaps"),
            "cross_validation": quality.get("cross_validation"),
            "open_gaps": quality.get("unresolved_gap_count"),
        },
        "counters": dict(counters),
        "stage_counts": stage_counts,
        "suppression": suppression,
        "overlap": overlap,
        "stickiness": stickiness,
        "queries": queries,
        "evidence_rows": evidence_rows,
        "enriched_evidence_rows": enriched_evidence_rows,
        "feedback_events": feedback_events,
        "enriched_events": enriched_events,
        "query_candidates": query_candidates,
        "context_records": context_records,
        "search_events": search_events,
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "-"
    return str(value)


def build_root_causes(report: dict[str, Any]) -> list[dict[str, str]]:
    """J. Root cause statements separated into observation/evidence/inference."""

    overlap = report["overlap_141_142"]
    suppression = report["suppression"]["totals"]
    specificity = report["specificity"]
    migration = report["failure_migration"]
    deterministic = report["deterministic_bottleneck"]
    stickiness = report["context_stickiness"]["candidate"]
    first_loss = report["funnel_first_loss"]

    candidate_pinned_text = (
        ", ".join(
            f"{row['section']}.{row['field']}"
            for row in deterministic["candidate"]["pinned_fields"]
        )
        or "(none)"
    )
    migration_text = (
        ", ".join(
            f"{row['bucket']} {row['share_delta']:+.3f}" for row in migration["migrations"]
        )
        or "none"
    )

    causes: list[dict[str, str]] = []
    causes.append(
        {
            "topic": "enrichment vocabulary repeats the 14.1 adaptation",
            "observation": (
                "Enrichment attempts mostly append the same four hints "
                "(benchmark/evaluation/paper/experiment) that the 14.1 QueryCandidate "
                "already wrote into its own query text."
            ),
            "evidence": (
                f"total attempts={overlap['total_enrichment_attempts']}, "
                f"no-op={overlap['no_op_count']} (rate={_fmt(overlap['exact_no_op_rate'])}), "
                f"unique hints contributed by the main path="
                f"{overlap['unique_hint_contribution']}, "
                f"full-overlap questions={_fmt(overlap['full_overlap_rate'])}"
            ),
            "inference": (
                "At the hint-set level the 14.2 main path adds little vocabulary that 14.1 "
                "did not already inject; the two consumers share one mapping."
            ),
            "confidence": "high (both consumers and their hint sets are recorded in events)",
        }
    )
    causes.append(
        {
            "topic": "search opportunities were intercepted before the provider",
            "observation": (
                "The candidate arm started fewer new searches while reuse events stayed flat."
            ),
            "evidence": (
                "candidate suppression buckets="
                f"{json.dumps(suppression.get('candidate', {}), ensure_ascii=False)}; "
                f"first funnel loss at stage '{first_loss['stage']}' "
                f"(conversion {_fmt(first_loss['baseline_conversion'])} -> "
                f"{_fmt(first_loss['candidate_conversion'])})"
            ),
            "inference": (
                "A share of the 42 -> 34 delta is produced by reuse/source-space exhaustion "
                "gating rather than by the enrichment text itself."
            ),
            "confidence": (
                "medium (event order recorded; causal share not isolated by an experiment)"
            ),
        }
    )
    causes.append(
        {
            "topic": "enriched queries return fewer and emptier result sets",
            "observation": (
                "Paired searches show lower result counts and a higher zero-result rate "
                "in the candidate arm."
            ),
            "evidence": (
                f"paired_result_delta={_fmt(specificity['paired_result_delta'])} over "
                f"{specificity['paired_core_count']} paired cores "
                f"(paired avg results {_fmt(specificity.get('paired_baseline_avg_results'))} "
                f"-> {_fmt(specificity.get('paired_candidate_avg_results'))}); "
                f"zero-result rate {_fmt(specificity['baseline_overall']['zero_result_rate'])} -> "
                f"{_fmt(specificity['candidate_overall']['zero_result_rate'])}; "
                f"avg results {_fmt(specificity['baseline_overall']['avg_results'])} -> "
                f"{_fmt(specificity['candidate_overall']['avg_results'])}"
            ),
            "inference": (
                "Longer hint-stuffed queries behave like stricter AND-queries at the provider; "
                "recall contracts while precision is not measurably improved downstream."
            ),
            "confidence": "medium (paired counts observed; no nDCG/precision labels available)",
        }
    )
    causes.append(
        {
            "topic": "candidate trajectories pinned by deterministic gates",
            "observation": (
                "All three candidate runs share identical search/reuse counts, exhausted "
                "question sets, provider observations and coverage."
            ),
            "evidence": (
                f"candidate pinned fields={candidate_pinned_text}; "
                f"coverage SD zero={deterministic['candidate']['coverage_sd_zero']}"
            ),
            "inference": (
                "The candidate arm's variance was suppressed by its own deterministic context "
                "and exhaustion pattern, so coverage SD == 0 is an execution artifact rather "
                "than evidence of stable quality."
            ),
            "confidence": "high (raw scalar vectors are identical)",
        }
    )
    causes.append(
        {
            "topic": "recorded context keeps steering after the need moves on",
            "observation": (
                "One recorded context per question kept being applied to later search targets "
                "even after newer refined needs appeared."
            ),
            "evidence": (
                f"stale_context_usage_count={stickiness.get('stale_context_usage_count')}, "
                f"avg_uses_per_need={_fmt(stickiness.get('avg_uses_per_need'))}"
            ),
            "inference": (
                "Context stickiness can keep appending an outdated hint family to searches "
                "after the underlying evidence need changed."
            ),
            "confidence": "medium (event-sequence based; need-question linkage incomplete for "
            "some events)",
        }
    )
    causes.append(
        {
            "topic": "downstream acceptance still governs the outcome",
            "observation": (
                "The candidate arm read more sources and produced more extraction events, yet "
                "fewer evidence rows and fewer acceptances; no gap transition happened in "
                "either arm."
            ),
            "evidence": (
                f"evidence rows {report['cost']['baseline_totals']['candidate_evidence']} -> "
                f"{report['cost']['candidate_totals']['candidate_evidence']}, accepted "
                f"{report['cost']['baseline_totals']['accepted_evidence']} -> "
                f"{report['cost']['candidate_totals']['accepted_evidence']}; "
                f"failure migration: {migration_text}"
            ),
            "inference": (
                "Search-side changes cannot move closure while acceptance and independence "
                "requirements reject the additional material."
            ),
            "confidence": "high (acceptance outcomes are persisted per evidence row)",
        }
    )
    return causes


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    data_set = report["data_set"]
    lines.append("# Phase 14.4 Result - Evidence-Aware Enrichment Attribution")
    lines.append("")
    lines.append("## A. Data Set")
    lines.append("")
    lines.append(f"- baseline runs: {', '.join(data_set['baseline_run_ids']) or '(none)'}")
    lines.append(f"- candidate runs: {', '.join(data_set['candidate_run_ids']) or '(none)'}")
    for arm in ("baseline", "candidate"):
        for run in data_set["arms"][arm]:
            lines.append(
                f"  - {arm}: `{run['run_id']}` status={run['status']} "
                f"mode={run['experiment_mode']} revision={run['source_revision']} "
                f"tier={run['budget_tier']} template={run['plan_template_run_id']}"
            )
    if data_set["run_count_validation_problems"]:
        lines.append(
            f"- validation problems: {'; '.join(data_set['run_count_validation_problems'])}"
        )
    else:
        lines.append("- validation: arm sizes match, no duplicate/cross-arm run ids")
    lines.append("")

    overlap = report["overlap_141_142"]
    lines.append("## B. 14.1 vs 14.2 Overlap")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in OVERLAP_METRIC_KEYS:
        lines.append(f"| {key} | {_fmt(overlap.get(key))} |")
    lines.append(f"| unique hint contribution | {overlap['unique_hint_contribution']} |")
    lines.append(f"| context vocabulary | {overlap.get('vocabulary')} |")
    lines.append(
        f"| hint derivation (explicit/derived/context-missing) | "
        f"{overlap.get('explicit_hint_events')}/{overlap.get('derived_hint_events')}/"
        f"{overlap.get('context_missing_events')} |"
    )
    lines.append("")
    for row in overlap.get("per_question_overlap", []):
        lines.append(
            f"- `{row['question_id']}` overlap_ratio={row['overlap_ratio']} "
            f"main-added={row['main_path_added_hints']} "
            f"14.1-materialised={row['candidate_materialised_hints']} "
            f"unique_to_main={row['unique_to_main_path']}"
        )
    lines.append("")

    lines.append("## C. Research Funnel Attribution")
    lines.append("")
    lines.append("| Stage | Baseline | Candidate | Base conv | Cand conv | Cand drop-off |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for base_row, cand_row in zip(
        report["funnel"]["baseline"], report["funnel"]["candidate"], strict=False
    ):
        lines.append(
            f"| {base_row['stage']} | {base_row['count']} | {cand_row['count']} | "
            f"{_fmt(base_row['conversion_rate'])} | {_fmt(cand_row['conversion_rate'])} | "
            f"{_fmt(cand_row['dropoff_rate'])} |"
        )
    first_loss = report["funnel_first_loss"]
    lines.append("")
    lines.append(
        f"- first material loss: stage `{first_loss['stage'] or '(none)'}` "
        f"conversion {_fmt(first_loss['baseline_conversion'])} -> "
        f"{_fmt(first_loss['candidate_conversion'])} "
        f"(delta {_fmt(first_loss['conversion_delta'])})"
    )
    lines.append("")

    lines.append("## D. Search Suppression Analysis")
    lines.append("")
    lines.append("| Reason | Baseline | Candidate |")
    lines.append("| --- | ---: | ---: |")
    base_supp = report["suppression"]["totals"].get("baseline", {})
    cand_supp = report["suppression"]["totals"].get("candidate", {})
    for reason in SUPPRESSION_REASONS:
        lines.append(
            f"| {reason} | {base_supp.get(reason, 0)} | {cand_supp.get(reason, 0)} |"
        )
    lines.append("")
    lines.append("Event-order evidence (candidate arm):")
    lines.append("")
    for run_id, buckets in report["suppression"]["per_run"].items():
        if buckets.get("arm") != "candidate":
            continue
        lines.append(
            f"- `{run_id[:13]}` new={buckets['started_new_searches']} "
            f"reused={buckets['started_reused_searches']} stop={buckets['stop_reason']}"
        )
        run_evidence = report["suppression"]["evidence"].get(run_id, {})
        for key, entries in run_evidence.items():
            if key in (
                "source_space_exhausted",
                "feedback_execution_limit",
                "query_already_executed",
                "search_budget_exhausted",
                "logical_query_budget_exhausted",
                "duplicate_query",
                "provider_degraded",
            ):
                for entry in entries[:4]:
                    lines.append(f"    - {key}: {entry}")
    lines.append("")

    specificity = report["specificity"]
    lines.append("## E. Query Specificity Analysis")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate |")
    lines.append("| --- | ---: | ---: |")
    base_overall = specificity["baseline_overall"]
    cand_overall = specificity["candidate_overall"]
    for key in (
        "queries",
        "distinct_cores",
        "result_count",
        "avg_results",
        "zero_result_rate",
        "reused",
    ):
        lines.append(f"| {key} | {_fmt(base_overall.get(key))} | {_fmt(cand_overall.get(key))} |")
    lines.append(
        f"| paired cores | {specificity['paired_core_count']} | "
        f"paired delta {_fmt(specificity['paired_result_delta'])} |"
    )
    lines.append(
        f"| paired avg results | {_fmt(specificity.get('paired_baseline_avg_results'))} | "
        f"{_fmt(specificity.get('paired_candidate_avg_results'))} "
        f"(delta {_fmt(specificity.get('paired_avg_result_delta'))}) |"
    )
    lines.append("")
    for pair in specificity.get("pairs", [])[:12]:
        lines.append(
            f"- `{pair['question_id']}` {pair['query_core']}: results "
            f"{pair['baseline']['result_count']} -> {pair['candidate']['result_count']} "
            f"(delta {pair['result_count_delta']}), avg {pair['baseline']['avg_results']} -> "
            f"{pair['candidate']['avg_results']}"
        )
    lines.append("")

    migration = report["failure_migration"]
    lines.append("## F. Evidence Outcome Analysis")
    lines.append("")
    lines.append(
        "| Bucket | Baseline share of rejections | Candidate share of rejections | Delta |"
    )
    lines.append("| --- | ---: | ---: | ---: |")
    for bucket in FAILURE_BUCKETS:
        base_share = migration["baseline_evidence"]["rejection_buckets"][bucket][
            "share_of_rejections"
        ]
        cand_share = migration["candidate_evidence"]["rejection_buckets"][bucket][
            "share_of_rejections"
        ]
        lines.append(
            f"| {bucket} | {_fmt(base_share)} | {_fmt(cand_share)} | "
            f"{_fmt(round(cand_share - base_share, 4))} |"
        )
    lines.append("")
    lines.append(
        f"- acceptance rate {_fmt(migration['baseline_evidence']['acceptance_rate'])} -> "
        f"{_fmt(migration['candidate_evidence']['acceptance_rate'])}"
    )
    for row in migration.get("migrations", []):
        lines.append(
            f"- migration: {row['bucket']} {row['direction']} "
            f"({row['baseline_share']} -> {row['candidate_share']})"
        )
    enriched_chain = migration.get("candidate_enriched_evidence")
    if isinstance(enriched_chain, dict):
        lines.append("")
        lines.append(
            "Enrichment chain (candidate evidence discovered through an actually "
            "enriched query):"
        )
        lines.append("")
        lines.append(
            f"- rows={enriched_chain['rows']}, accepted={enriched_chain['accepted']}, "
            f"acceptance rate={_fmt(enriched_chain['acceptance_rate'])}"
        )
        for bucket in FAILURE_BUCKETS:
            count = enriched_chain["rejection_buckets"][bucket]["count"]
            if int(count or 0) > 0:
                lines.append(
                    f"  - {bucket}: {count} "
                    f"(share of rejections "
                    f"{_fmt(enriched_chain['rejection_buckets'][bucket]['share_of_rejections'])})"
                )
    lines.append("")

    deterministic = report["deterministic_bottleneck"]
    lines.append("## G. Deterministic Bottleneck Analysis")
    lines.append("")
    lines.append(f"- candidate coverage values: {deterministic['candidate']['coverage_values']}")
    lines.append(f"- candidate coverage SD zero: {deterministic['candidate']['coverage_sd_zero']}")
    lines.append(
        f"- candidate bottleneck candidates: "
        f"{deterministic['candidate']['bottleneck_candidates']}"
    )
    pinned_candidate = (
        ", ".join(
            f"{row['section']}.{row['field']}"
            for row in deterministic["candidate"]["pinned_fields"]
        )
        or "(none)"
    )
    pinned_baseline = (
        ", ".join(
            f"{row['section']}.{row['field']}"
            for row in deterministic["baseline"]["pinned_fields"]
        )
        or "(none)"
    )
    lines.append(f"- pinned fields (candidate): {pinned_candidate}")
    lines.append(f"- pinned fields (baseline): {pinned_baseline}")
    lines.append("")

    stickiness = report["context_stickiness"]["candidate"]
    lines.append("## H. Context Stickiness Analysis")
    lines.append("")
    lines.append(f"- context records: {stickiness.get('context_records')}")
    lines.append(f"- enrichment uses: {stickiness.get('enrichment_uses')}")
    lines.append(f"- avg uses per need: {_fmt(stickiness.get('avg_uses_per_need'))}")
    lines.append(f"- stale context usage count: {stickiness.get('stale_context_usage_count')}")
    for example in stickiness.get("stale_examples", [])[:6]:
        lines.append(
            f"    - seq {example['run_seq']} `{example['question_id']}` still used need "
            f"`{example['used_need_id'][:13]}` after newer need at "
            f"seq {example['newer_need_run_seq']}"
        )
    lines.append("")

    cost = report["cost"]
    lines.append("## I. Cost / Efficiency Analysis")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate |")
    lines.append("| --- | ---: | ---: |")
    for key in sorted(cost["baseline_totals"]):
        lines.append(
            f"| total {key} | {_fmt(cost['baseline_totals'][key])} | "
            f"{_fmt(cost['candidate_totals'][key])} |"
        )
    for key in sorted(cost["baseline_per_unit"]):
        lines.append(
            f"| {key} | {_fmt(cost['baseline_per_unit'][key])} | "
            f"{_fmt(cost['candidate_per_unit'][key])} |"
        )
    lines.append(f"- verdict: {cost['verdict']}")
    lines.append("")

    lines.append("## J. Root Cause")
    lines.append("")
    for cause in build_root_causes(report):
        lines.append(f"### {cause['topic']}")
        lines.append(f"- Observation: {cause['observation']}")
        lines.append(f"- Evidence: {cause['evidence']}")
        lines.append(f"- Inference: {cause['inference']}")
        lines.append(f"- Confidence: {cause['confidence']}")
        lines.append("")

    decision = report["decision"]
    lines.append("## K. Decision")
    lines.append("")
    lines.append(
        f"- Primary: Case {decision['primary_case']} - {decision['primary_case_summary']}"
    )
    lines.append(f"  - rationale: {decision['primary_rationale']}")
    if decision.get("secondary_case"):
        lines.append(
            f"- Secondary: Case {decision['secondary_case']} - {decision['secondary_case_summary']}"
        )
        lines.append(f"  - rationale: {decision.get('secondary_rationale')}")
    lines.append(f"- all signals: {json.dumps(decision['all_signals'], ensure_ascii=False)}")
    lines.append(f"- caveat: {decision['caveat']}")
    lines.append("")

    lines.append("## L. Recommendation")
    lines.append("")
    lines.append(
        "- Analysis only: this phase deliberately proposes but does not implement changes."
    )
    for item in recommendation_for_case(decision["primary_case"]):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Wording discipline (spec section 16):")
    lines.append("")
    lines.append(
        "- This report does not claim \"14.2 is ineffective\". Under the fixed Phase 14.3 "
        "conditions the main-path enrichment produced no observed positive quality gain; "
        "Phase 14.4 attributes the absence to redundancy, query over-constraint, execution "
        "gating and downstream acceptance, per the evidence above."
    )
    lines.append(
        "- Candidate metrics declining is not by itself treated as enrichment causing the "
        "decline; causal claims are limited to event-chain evidence."
    )
    return "\n".join(lines) + "\n"


def recommendation_for_case(case_id: int) -> list[str]:
    if case_id == 1:
        return [
            "Keep the ResearchContext architecture and 14.1 candidate adaptation.",
            "Consider defaulting the 14.2 main-path enrichment to off (or removing it from "
            "the main search target) in a future phase - recommendation only, not implemented.",
        ]
    if case_id == 2:
        return [
            "Keep enrichment intent; refine the mapping from failure type to evidence "
            "requirement to context-specific hint vocabulary.",
            "Evaluate a per-failure-type hint table against paired recall before shipping.",
        ]
    if case_id == 3:
        return [
            "Focus next-phase work on evidence acceptance, claim verification and closure "
            "requirements rather than further search-text adaptation.",
        ]
    if case_id == 4:
        return [
            "Address research opportunity scheduling (reuse/budget/feedback gates) before "
            "tuning query content.",
        ]
    return ["No recommendation available for an unknown case."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-id", default=os.getenv("PHASE14_4_BASELINE_RUN_ID"))
    parser.add_argument("--candidate-run-id", default=os.getenv("PHASE14_4_CANDIDATE_RUN_ID"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")
    baseline_ids = _split_run_ids(args.baseline_run_id)
    candidate_ids = _split_run_ids(args.candidate_run_id)
    if not baseline_ids or not candidate_ids:
        parser.error("at least one --baseline-run-id and one --candidate-run-id are required")
    problems = validate_arms(baseline_ids, candidate_ids)
    if problems:
        parser.error("invalid arms: " + "; ".join(problems))

    with psycopg.connect(_database_uri(args.database_url)) as connection:
        report = build_report(connection, baseline_ids, candidate_ids)

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
    print(markdown)
    return 0


def build_report(
    connection: psycopg.Connection,
    baseline_ids: list[str],
    candidate_ids: list[str],
) -> dict[str, Any]:
    """Assemble the complete Phase 14.4 attribution report."""

    problems = validate_arms(baseline_ids, candidate_ids)
    baseline_runs = [analyze_run(connection, run_id) for run_id in baseline_ids]
    candidate_runs = [analyze_run(connection, run_id) for run_id in candidate_ids]

    def aggregated(runs: list[dict[str, Any]], key: str) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for run in runs:
            totals.update(run.get(key, {}))
        return {stage: totals.get(stage, 0) for stage, _source in FUNNEL_STAGES}

    baseline_stages = aggregated(baseline_runs, "stage_counts")
    candidate_stages = aggregated(candidate_runs, "stage_counts")
    funnel = build_funnel({"baseline": baseline_stages, "candidate": candidate_stages})

    suppression_per_run: dict[str, dict[str, Any]] = {}
    for run in baseline_runs + candidate_runs:
        arm = ARM_BASELINE
        mode = str(run.get("profile", {}).get("experiment_mode") or "")
        if mode == EXPERIMENT_MODE_CANDIDATE:
            arm = ARM_CANDIDATE
        suppression_per_run[str(run["run_id"])] = {
            "arm": arm,
            "suppression": run.get("suppression", {}),
        }
    suppression_totals = summarize_suppression(suppression_per_run)

    overlap = compute_enrichment_overlap(
        [item for run in candidate_runs for item in run.get("enriched_events", [])],
        [item for run in candidate_runs for item in run.get("query_candidates", [])],
        {
            need_id: hints
            for run in candidate_runs
            for need_id, hints in (
                run.get("overlap", {}).get("resolved_hint_sets") or {}
            ).items()
        },
    )
    specificity = pair_query_outcomes(
        [query for run in baseline_runs for query in run.get("queries", [])],
        [query for run in candidate_runs for query in run.get("queries", [])],
    )
    failure_migration = compute_failure_migration(
        [row for run in baseline_runs for row in run.get("evidence_rows", [])],
        [row for run in candidate_runs for row in run.get("evidence_rows", [])],
        [row for run in baseline_runs for row in run.get("feedback_events", [])],
        [row for run in candidate_runs for row in run.get("feedback_events", [])],
        [row for run in candidate_runs for row in run.get("enriched_evidence_rows", [])],
    )
    deterministic = {
        "baseline": detect_deterministic_bottleneck(baseline_runs),
        "candidate": detect_deterministic_bottleneck(candidate_runs),
    }
    stickiness = {
        "baseline": {
            "context_records": 0,
            "enrichment_uses": 0,
            "stale_context_usage_count": 0,
        },
        "candidate": {
            "context_records": sum(len(run.get("context_records", [])) for run in candidate_runs),
            "enrichment_uses": sum(len(run.get("enriched_events", [])) for run in candidate_runs),
            "distinct_needs_used": sum(
                int(run.get("stickiness", {}).get("distinct_needs_used") or 0)
                for run in candidate_runs
            ),
            "avg_uses_per_need": mean_or_zero(
                [
                    float(run.get("stickiness", {}).get("avg_uses_per_need") or 0.0)
                    for run in candidate_runs
                ]
            ),
            "stale_context_usage_count": sum(
                int(run.get("stickiness", {}).get("stale_context_usage_count") or 0)
                for run in candidate_runs
            ),
            "stale_examples": [
                example
                for run in candidate_runs
                for example in run.get("stickiness", {}).get("stale_examples", [])
            ][:10],
            "per_need_lifetime": [
                row
                for run in candidate_runs
                for row in run.get("stickiness", {}).get("per_need_lifetime", [])
            ],
        },
    }
    incremental = compute_incremental_value(overlap, suppression_totals, specificity)
    cost = compute_cost_efficiency(baseline_runs, candidate_runs)
    decision = decide_cases(
        overlap, suppression_totals, failure_migration, deterministic, specificity, cost
    )

    return {
        "experiment": "phase14.4_enrichment_attribution",
        "data_set": {
            "baseline_run_ids": baseline_ids,
            "candidate_run_ids": candidate_ids,
            "run_count_validation_problems": problems,
            "arms": {
                "baseline": [
                    {
                        "run_id": run["run_id"],
                        "status": run.get("profile", {}).get("status"),
                        "experiment_mode": run.get("profile", {}).get("experiment_mode"),
                        "source_revision": run.get("profile", {}).get("source_revision"),
                        "budget_tier": run.get("profile", {}).get("budget_tier"),
                        "plan_template_run_id": run.get("profile", {}).get("plan_template_run_id"),
                    }
                    for run in baseline_runs
                ],
                "candidate": [
                    {
                        "run_id": run["run_id"],
                        "status": run.get("profile", {}).get("status"),
                        "experiment_mode": run.get("profile", {}).get("experiment_mode"),
                        "source_revision": run.get("profile", {}).get("source_revision"),
                        "budget_tier": run.get("profile", {}).get("budget_tier"),
                        "plan_template_run_id": run.get("profile", {}).get("plan_template_run_id"),
                    }
                    for run in candidate_runs
                ],
            },
        },
        "overlap_141_142": overlap,
        "funnel": funnel,
        "funnel_first_loss": funnel_first_loss(funnel),
        "suppression": {
            "totals": suppression_totals,
            "per_run": {
                run_id: {
                    "arm": entry["arm"],
                    "buckets": entry["suppression"].get("buckets", {}),
                    "started_new_searches": entry["suppression"].get("started_new_searches"),
                    "started_reused_searches": entry["suppression"].get("started_reused_searches"),
                    "stop_reason": entry["suppression"].get("stop_reason"),
                }
                for run_id, entry in suppression_per_run.items()
            },
            "evidence": {
                run_id: entry["suppression"].get("evidence", {})
                for run_id, entry in suppression_per_run.items()
            },
        },
        "specificity": specificity,
        "failure_migration": failure_migration,
        "deterministic_bottleneck": deterministic,
        "context_stickiness": stickiness,
        "incremental_value": incremental,
        "cost": cost,
        "decision": decision,
        "runs": {
            "baseline": [
                {
                    "run_id": run["run_id"],
                    "usage": run.get("usage"),
                    "quality": run.get("quality"),
                    "stage_counts": run.get("stage_counts"),
                }
                for run in baseline_runs
            ],
            "candidate": [
                {
                    "run_id": run["run_id"],
                    "usage": run.get("usage"),
                    "quality": run.get("quality"),
                    "stage_counts": run.get("stage_counts"),
                }
                for run in candidate_runs
            ],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
