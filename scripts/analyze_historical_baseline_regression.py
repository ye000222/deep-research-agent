"""Phase 14.4 supplemental audit: historical baseline regression (read-only).

The Phase 14.3/14.4 experiments measured a current baseline coverage of
~0.4524 +/- 0.0545 on the English comparison benchmark, while historical
runs of this project scored ~0.85-0.87 (stored values).  This audit answers
*why* strictly from persisted data, without touching production research
behaviour:

  historical high coverage  ->  current baseline low coverage

is attributed to one (or a mix) of:

  Case A  metric / definition drift (the score semantics changed)
  Case B  true runtime regression   (same definition, worse behaviour)
  Case C  environment / provider drift
  Case D  mixed cause (each component quantified)

Method discipline (mirrors the audit spec):

  * the historical runs are **replayed** with the *current* coverage
    definition ported line-by-line from ``research_tools.py``; stored
    ``quality_snapshot.coverage`` is never taken at face value;
  * every statement is derived from persisted run / event / evidence / gap
    rows; where the historical code version cannot be recovered the report
    says ``UNKNOWN`` instead of guessing;
  * a unified funnel compares historical known-good runs against the
    current baseline and the first material divergence point is reported;
  * the auxiliary historical runs are auto-discovered (same goal, complete,
    data-complete, non-experiment) and selected **by recency, never by
    score**, so a lucky run cannot bias the baseline.

Strictly read-only: only SELECT statements, no production imports, no
benchmark is run, nothing is mutated.

Outputs:

  artifacts/historical_baseline_regression.json
  artifacts/historical_baseline_regression.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = REPOSITORY_ROOT / "artifacts" / "historical_baseline_regression.json"
DEFAULT_MARKDOWN_OUT = REPOSITORY_ROOT / "artifacts" / "historical_baseline_regression.md"

DEFAULT_ANCHOR_RUN_ID = "01a0bced-4618-76f5-876a-02c175d32cf1"
DEFAULT_CURRENT_BASELINE_IDS = (
    "01a0bf8b-fdc4-7008-a419-3edac7d3a9f6",
    "01a0bf8b-fdde-7813-a99d-8fb0ec22ab92",
    "01a0bfa7-3c70-712b-8109-064a8ba4b7f6",
)

#: Pool rule for auxiliary historical references (see the audit spec:
#: report the pool and the rule, never pick only the best-scoring run).
POOL_MIN_STORED_COVERAGE = 0.6
POOL_MAX_AUX = 2
POOL_MIN_ACCEPTED_EVIDENCE = 10
POOL_MIN_SOURCE_COUNT = 5
POOL_SIZE_HINT = 15

#: A stage count is considered materially divergent when |relative delta|
#: between the two arms exceeds this threshold.
DIVERGENCE_THRESHOLD = 0.15

#: Funnel stage order used for the unified comparison.  Every stage maps to
#: an integer count (or None when the historical data does not carry it).
FUNNEL_STAGES: tuple[tuple[str, str], ...] = (
    ("plan_items", "Planner targets (plan questions)"),
    ("plan_dimensions", "Plan requirement dimensions"),
    ("logical_queries", "Logical queries committed"),
    ("search_completed", "Search executions completed"),
    ("candidate_urls", "Candidate URLs discovered"),
    ("unique_urls", "Unique result URLs"),
    ("readable_attempts", "source.readable events"),
    ("unique_readable", "Unique readable sources"),
    ("reader_completed", "Reader completions (snapshots)"),
    ("chunked_snapshots", "Snapshots with chunks"),
    ("chunks_total", "Persisted content chunks"),
    ("extraction_calls", "Evidence extraction calls"),
    ("candidate_evidence", "Candidate evidence rows"),
    ("accepted_evidence", "Accepted evidence rows"),
    ("alignment_completed", "Evidence alignment completions"),
    ("gap_transitions", "Gap closure transitions"),
)

#: Behaviour-side stages used to distinguish "config-shaped" from
#: "behavioural" divergence (plan/query counts are task-shaped inputs).
BEHAVIOUR_STAGES = (
    "unique_readable",
    "reader_completed",
    "chunked_snapshots",
    "chunks_total",
    "candidate_evidence",
    "accepted_evidence",
)

#: Per-unit efficiency metrics: (key, numerator stage, denominator stage).
UNIT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("searches_per_dimension", "logical_queries", "plan_dimensions"),
    ("unique_urls_per_search", "unique_urls", "search_completed"),
    ("unique_sources_per_search", "unique_readable", "search_completed"),
    ("snapshots_per_search", "reader_completed", "search_completed"),
    ("chunks_per_snapshot", "chunks_total", "reader_completed"),
    ("evidence_rows_per_snapshot", "candidate_evidence", "reader_completed"),
    ("accepted_per_row", "accepted_evidence", "candidate_evidence"),
    ("accepted_per_snapshot", "accepted_evidence", "reader_completed"),
    ("accepted_per_search", "accepted_evidence", "search_completed"),
)

#: Markers copied verbatim from research_tools._requires_independent_sources.
INDEPENDENT_SOURCE_MARKERS = (
    "两个",
    "两条",
    "第二",
    "独立来源",
    "two ",
    "2 ",
    "independent",
    "市场规模",
    "市场份额",
    "市场预测",
    "增长率",
    "预测",
    "趋势",
    "第一",
    "领先",
    "比较",
    "market size",
    "market share",
    "forecast",
    "projection",
    "growth rate",
    "leading",
    "largest",
    "comparative",
)

#: Claim types that mark a dimension as high risk (research_tools rule).
HIGH_RISK_CLAIM_TYPES = {"numeric", "forecast", "comparative", "market"}
HIGH_RISK_IMPORTANCE = 0.75

CASE_LABELS = {
    "A": "Metric Definition Drift",
    "B": "True Runtime Regression",
    "C": "Environment / Provider Drift",
    "D": "Mixed Cause",
}

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Zero/None-safe ratio rounded to 4 decimals; None when undefined."""

    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 4)


def mean_or_none(values: list[float] | list[int]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round(statistics.fmean(cleaned), 4)


def mean_or_zero(values: list[float] | list[int]) -> float:
    result = mean_or_none(values)
    return 0.0 if result is None else result


def normalize_revision(revision: str | None) -> str:
    """Collapse a source_revision stamp into an auditable era bucket."""

    if not revision:
        return "none"
    if revision.startswith("c1c00954"):
        return "c1c009540b09-worktree-*"
    if revision.startswith("ea3f0635"):
        return "ea3f0635ce84"
    if revision.startswith("57a3b887"):
        return "57a3b8877f9c"
    if revision.startswith("v28-"):
        return "v28-search-fallback-proxy-and-bing-parser"
    if re.fullmatch(r"[0-9a-f]{8,40}(-.*)?", revision):
        return revision[:12]
    return revision[:24]


def requires_independent_sources(criterion: str) -> bool:
    """Port of research_tools._requires_independent_sources (marker match)."""

    normalized = criterion.casefold()
    return any(marker in normalized for marker in INDEPENDENT_SOURCE_MARKERS)


def claim_is_high_risk(claim: dict[str, Any]) -> bool:
    """Port of research_tools._claim_is_high_risk for a plain claim dict."""

    importance = float(claim.get("importance") or 0.0)
    claim_type = str(claim.get("claim_type") or "")
    return bool(
        str(claim.get("status") or "") == "disputed"
        or importance >= HIGH_RISK_IMPORTANCE
        or claim_type in HIGH_RISK_CLAIM_TYPES
    )


def replay_coverage(
    plan_items: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    plan_version: int | None,
) -> dict[str, Any]:
    """Recompute coverage with the *current* definition, from raw rows.

    Line-by-line port of the current production semantics in
    ``research_tools.py`` (covered-feedback quality snapshot):

      * plan items filtered by ``run.plan_version`` ordered by
        ``priority, question_id``;
      * per requirement ``index`` the dimension key is
        ``{question_id}:d{index}``;
      * dimension evidence counts join claims -> evidence -> sources with
        ``evidence.accepted`` and group by ``(question_id, dimension_key)``;
      * a dimension requires 2 independent owners when its criterion carries
        an independent-source marker OR any claim of that dimension is
        high risk, else 1;
      * score 1.0 when evidence > 0 and owners >= required, else 0.5 when
        evidence > 0, else 0.0;
      * question coverage = mean of requirement scores, weight = 4 - priority;
      * coverage = weighted mean, rounded to 4 decimals.

    Returns a dict with ``coverage``, ``coverage_map``, dimension details and
    aggregate counts (accepted evidence, owners, claims).
    """

    accepted_rows = [row for row in evidence_rows if row.get("accepted")]
    owner_by_source = {
        str(source["id"]): str(source.get("source_owner_key") or source["id"]) for source in sources
    }

    counts_by_question: dict[str, tuple[int, int]] = {}
    owners_by_question: dict[str, set[str]] = {}
    for row in accepted_rows:
        question_id = str(row.get("question_id") or "")
        evidence_count = counts_by_question.get(question_id, (0, 0))[0]
        owners = owners_by_question.setdefault(question_id, set())
        owner = owner_by_source.get(str(row.get("source_id")), str(row.get("source_id")))
        if owner not in owners and row.get("source_id") is not None:
            owners.add(owner)
        counts_by_question[question_id] = (evidence_count + 1, len(owners))

    counts_by_dimension: dict[tuple[str, str], tuple[int, int]] = {}
    owners_by_dimension: dict[tuple[str, str], set[str]] = {}
    claims_by_id = {str(claim["id"]): claim for claim in claims}
    for row in accepted_rows:
        claim_id = str(row.get("claim_id") or "")
        claim = claims_by_id.get(claim_id)
        if claim is None:
            continue
        key = (str(claim.get("question_id") or ""), str(claim.get("dimension_key") or ""))
        owners = owners_by_dimension.setdefault(key, set())
        owner = owner_by_source.get(str(row.get("source_id")), str(row.get("source_id")))
        if owner not in owners and row.get("source_id") is not None:
            owners.add(owner)
        evidence_count, _ = counts_by_dimension.get(key, (0, 0))
        counts_by_dimension[key] = (evidence_count + 1, len(owners))

    high_risk_dimension_keys = {
        str(claim.get("dimension_key") or "") for claim in claims if claim_is_high_risk(claim)
    }

    filtered_items = [
        item
        for item in plan_items
        if plan_version is None or int(item.get("plan_version", plan_version)) == plan_version
    ]

    coverage_map: list[dict[str, Any]] = []
    weighted_coverage = 0.0
    total_weight = 0.0
    for item in filtered_items:
        question_id = str(item.get("question_id") or "")
        requirements = [str(value) for value in (item.get("evidence_requirements") or [])]
        requirement_details: list[dict[str, Any]] = []
        requirement_scores: list[float] = []
        for index, criterion in enumerate(requirements, start=1):
            dimension_key = f"{question_id}:d{index}"
            dimension_evidence, dimension_owners = counts_by_dimension.get(
                (question_id, dimension_key), (0, 0)
            )
            marker_rule = requires_independent_sources(criterion)
            high_risk_rule = dimension_key in high_risk_dimension_keys
            required_sources = 2 if marker_rule or high_risk_rule else 1
            score = (
                1.0
                if dimension_evidence > 0 and dimension_owners >= required_sources
                else 0.5
                if dimension_evidence > 0
                else 0.0
            )
            requirement_scores.append(score)
            requirement_details.append(
                {
                    "dimension_key": dimension_key,
                    "criterion": criterion,
                    "coverage": score,
                    "accepted_evidence": dimension_evidence,
                    "independent_sources": dimension_owners,
                    "required_sources": required_sources,
                    "required_via_marker": marker_rule,
                    "required_via_high_risk_claim": high_risk_rule,
                }
            )
        dimension_coverage = (
            sum(requirement_scores) / len(requirement_scores) if requirement_scores else 0.0
        )
        priority = int(item.get("priority") or 1)
        weight = float(4 - priority)
        total_weight += weight
        weighted_coverage += dimension_coverage * weight
        coverage_map.append(
            {
                "question_id": question_id,
                "priority": priority,
                "coverage": round(dimension_coverage, 4),
                "requirements": requirement_details,
            }
        )

    coverage = round(weighted_coverage / total_weight, 4) if total_weight else 0.0
    return {
        "coverage": coverage,
        "coverage_map": coverage_map,
        "plan_items": len(filtered_items),
        "dimensions": sum(len(item["requirements"]) for item in coverage_map),
        "accepted_evidence": len(accepted_rows),
        "candidate_evidence": len(evidence_rows),
        "claim_count": len(claims),
        "claims_high_risk": sum(1 for claim in claims if claim_is_high_risk(claim)),
        "dimensions_requiring_two_sources": sum(
            1
            for item in coverage_map
            for requirement in item["requirements"]
            if requirement["required_sources"] == 2
        ),
        "total_weight": total_weight,
    }


def stage_counts(facts: dict[str, Any]) -> dict[str, int | None]:
    """Map one run's raw facts onto the unified funnel stages.

    Every stage is an absolute count; ``None`` means "this historical run
    does not carry the datum" and is never coerced to zero.
    """

    events: dict[str, int] = facts.get("event_counts") or {}
    usage: dict[str, Any] = facts.get("usage") or {}
    pools: dict[str, Any] = ((usage.get("resource_pools") or {}).get("logical_queries")) or {}
    replay: dict[str, Any] = facts.get("replay") or {}
    storage: dict[str, Any] = facts.get("storage") or {}

    def event(name: str) -> int | None:
        value = events.get(name)
        return int(value) if value is not None else None

    logical_committed = pools.get("committed") if isinstance(pools, dict) else None
    logical = int(logical_committed) if isinstance(logical_committed, (int, float)) else None
    if logical is None and event("search.query.started") is not None:
        logical = event("search.query.started")
    searches = usage.get("searches")
    searches = int(searches) if isinstance(searches, (int, float)) else None

    attempts = event("source.readable")
    duplicates = event("source.duplicate_skipped") or 0
    unique_readable = attempts - duplicates if attempts is not None else None
    chunked = storage.get("chunked_snapshots")
    snapshots = storage.get("snapshots")
    candidate_urls_value = usage.get("candidate_urls")
    candidate_urls = (
        int(candidate_urls_value)
        if isinstance(candidate_urls_value, (int, float))
        else event("candidate.created")
    )

    count_stage = {
        "plan_items": replay.get("plan_items"),
        "plan_dimensions": replay.get("dimensions"),
        "logical_queries": logical if logical is not None else searches,
        "search_completed": event("search.completed")
        if event("search.completed") is not None
        else searches,
        "candidate_urls": candidate_urls,
        "unique_urls": storage.get("unique_result_urls"),
        "readable_attempts": attempts,
        "unique_readable": unique_readable,
        "reader_completed": snapshots,
        "chunked_snapshots": chunked,
        "chunks_total": storage.get("chunks"),
        "extraction_calls": event("evidence.extracted"),
        "candidate_evidence": replay.get("candidate_evidence"),
        "accepted_evidence": replay.get("accepted_evidence"),
        "alignment_completed": event("evidence.alignment.completed"),
        "gap_transitions": event("gap.closure.transition"),
    }
    return {stage: count_stage.get(stage) for stage, _label in FUNNEL_STAGES}


def build_funnel_comparison(
    historical: dict[str, float | None],
    current: dict[str, float | None],
    threshold: float = DIVERGENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Row-wise arm comparison of stage means (historical vs current)."""

    rows: list[dict[str, Any]] = []
    for stage, label in FUNNEL_STAGES:
        hist_value = historical.get(stage)
        cur_value = current.get(stage)
        delta = (
            round(cur_value - hist_value, 4)
            if hist_value is not None and cur_value is not None
            else None
        )
        relative = ratio(delta, hist_value) if delta is not None else None
        comparable = hist_value is not None and cur_value is not None
        rows.append(
            {
                "stage": stage,
                "label": label,
                "historical": hist_value,
                "current": cur_value,
                "delta": delta,
                "relative_delta": relative,
                "comparable": comparable,
                "divergent": bool(
                    comparable and relative is not None and abs(relative) >= threshold
                ),
            }
        )
    return rows


def find_first_divergence(
    rows: list[dict[str, Any]], threshold: float = DIVERGENCE_THRESHOLD
) -> dict[str, Any] | None:
    """First funnel stage whose relative delta crosses the threshold."""

    for index, row in enumerate(rows):
        relative = row.get("relative_delta")
        if row.get("comparable") and relative is not None and abs(relative) >= threshold:
            return {
                "stage": row["stage"],
                "label": row["label"],
                "index": index,
                "relative_delta": relative,
                "kind": "behaviour" if row["stage"] in BEHAVIOUR_STAGES else "config",
            }
    return None


def build_unit_comparison(
    historical: dict[str, float | None],
    current: dict[str, float | None],
    threshold: float = DIVERGENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Per-unit efficiency comparison derived from arm-mean stage counts."""

    rows: list[dict[str, Any]] = []
    for key, numerator, denominator in UNIT_METRICS:
        hist_value = ratio(historical.get(numerator), historical.get(denominator))
        cur_value = ratio(current.get(numerator), current.get(denominator))
        delta = (
            round(cur_value - hist_value, 4)
            if hist_value is not None and cur_value is not None
            else None
        )
        relative = ratio(delta, hist_value) if delta is not None else None
        rows.append(
            {
                "metric": key,
                "numerator": numerator,
                "denominator": denominator,
                "historical": hist_value,
                "current": cur_value,
                "delta": delta,
                "relative_delta": relative,
                "divergent": bool(relative is not None and abs(relative) >= threshold),
            }
        )
    return rows


def classify_config_difference(
    field: str,
    historical: Any,
    current: Any,
    classification: str,
    note: str = "",
) -> dict[str, Any]:
    """One configuration comparison row with an explicit classification."""

    if classification not in {"research_semantic", "infrastructure", "unknown"}:
        raise ValueError(f"invalid classification: {classification}")
    equal = historical == current
    return {
        "field": field,
        "historical": historical,
        "current": current,
        "equal": equal,
        "classification": classification,
        "note": note,
    }


def decide_primary_case(inputs: dict[str, Any]) -> dict[str, Any]:
    """Pure decision function mapping measured inputs onto Case A/B/C/D.

    Input contract (all keys optional, missing data is handled):
      ``hist_replay_mean`` / ``cur_replay_mean``: replayed coverage means;
      ``formula_match_all``: whether every replay equals the stored value;
      ``formula_mismatch_runs``: run ids where stored != replay;
      ``first_divergence_kind``: "config" | "behaviour" | None;
      ``behaviour_divergence_count``: divergent behaviour stages;
      ``config_semantic_diff_count``: divergent research-semantic config fields;
      ``same_task``: {"hist_mean","cur_mean","hist_n","cur_n"} of the same
      task family cohort;
      ``provider_environment_divergence``: bool.
    """

    hist = inputs.get("hist_replay_mean")
    cur = inputs.get("cur_replay_mean")
    if hist is None or cur is None:
        return {
            "case": None,
            "status": "insufficient_data",
            "label": "Insufficient historical observations",
            "rationale": ["historical or current arm has no replayed coverage"],
            "components": {},
        }

    gap = round(hist - cur, 4)
    formula_match_all = bool(inputs.get("formula_match_all", True))
    mismatch_runs = list(inputs.get("formula_mismatch_runs") or [])
    behaviour_count = int(inputs.get("behaviour_divergence_count") or 0)
    config_count = int(inputs.get("config_semantic_diff_count") or 0)
    provider_env = bool(inputs.get("provider_environment_divergence"))
    same_task = inputs.get("same_task") or {}
    same_hist = same_task.get("hist_mean")
    same_cur = same_task.get("cur_mean")
    same_task_supported = (
        same_hist is not None and same_cur is not None and same_task.get("hist_n", 0) > 0
    )
    same_task_flat = bool(
        same_task_supported
        and isinstance(same_hist, (int, float))
        and isinstance(same_cur, (int, float))
        and round(float(same_hist) - float(same_cur), 4) <= 0.10
    )

    components = {
        "replayed_gap": gap,
        "formula_match_all": formula_match_all,
        "formula_mismatch_runs": mismatch_runs,
        "behaviour_divergence_count": behaviour_count,
        "config_semantic_diff_count": config_count,
        "provider_environment_divergence": provider_env,
        "same_task_hist_mean": same_hist,
        "same_task_cur_mean": same_cur,
        "same_task_flat_under_current_definition": same_task_flat,
    }
    rationale: list[str] = []

    if gap <= 0.10 and not mismatch_runs:
        rationale.append(
            "replayed historical coverage is within 0.10 of the current baseline; "
            "no material gap remains after unification"
        )
        return {
            "case": "A",
            "status": "decided",
            "label": CASE_LABELS["A"],
            "rationale": rationale,
            "components": components,
        }

    if mismatch_runs and gap <= 0.10:
        rationale.append("stored values drift from replayed values but no material gap remains")
        return {
            "case": "A",
            "status": "decided",
            "label": CASE_LABELS["A"],
            "rationale": rationale,
            "components": components,
        }

    if same_task_supported and not same_task_flat and behaviour_count >= 2:
        rationale.append(
            "same-task cohort replays clearly above the current baseline and behaviour-stage "
            "divergence is measured inside the same task -> genuine runtime regression"
        )
        case = "B"
    elif behaviour_count >= 2 and config_count >= 2:
        rationale.append(
            "behaviour-side funnel divergence exists AND research-semantic config differs"
        )
        case = "D"
    elif behaviour_count >= 2:
        rationale.append("behaviour-side divergence without a usable same-task control")
        case = "B"
    elif config_count >= 2 and same_task_flat:
        rationale.append(
            "research-semantic configuration differs while the same-task control replays flat"
        )
        case = "D"
    elif provider_env:
        rationale.append("provider/environment signals dominate the divergence")
        case = "C"
    else:
        rationale.append("mixed or not-fully-separable evidence across components")
        case = "D"

    return {
        "case": case,
        "status": "decided",
        "label": CASE_LABELS[case],
        "rationale": rationale,
        "components": components,
    }


def era_aggregate(rows: list[dict[str, Any]], key: str = "era") -> list[dict[str, Any]]:
    """Aggregate timeline rows per era bucket, flagging thin eras."""

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key)), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for era, members in buckets.items():
        aggregates.append(
            {
                "era": era,
                "runs": len(members),
                "first_seen": min(str(member["created_at"]) for member in members),
                "last_seen": max(str(member["created_at"]) for member in members),
                "mean_stored_coverage": mean_or_none(
                    [m["stored_coverage"] for m in members if m.get("stored_coverage") is not None]
                ),
                "mean_replayed_coverage": mean_or_none(
                    [
                        m["replayed_coverage"]
                        for m in members
                        if m.get("replayed_coverage") is not None
                    ]
                ),
                "mean_accepted_evidence": mean_or_none(
                    [
                        m["accepted_evidence"]
                        for m in members
                        if m.get("accepted_evidence") is not None
                    ]
                ),
                "insufficient_observations": len(members) < 2,
            }
        )
    return aggregates


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Database layer (read-only)
# ---------------------------------------------------------------------------


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _split_run_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def validate_run_sets(
    anchor_id: str, aux_ids: list[str], baseline_ids: list[str]
) -> list[str]:
    """Return human-readable problems with the requested run sets."""

    problems: list[str] = []
    if not anchor_id:
        problems.append("historical anchor run id is required")
    if not baseline_ids:
        problems.append("current baseline arm has no run ids")
    all_ids = [anchor_id, *aux_ids, *baseline_ids]
    duplicates = [run_id for run_id, count in Counter(all_ids).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate run ids: {sorted(duplicates)}")
    historical = {anchor_id, *aux_ids}
    overlap = historical & set(baseline_ids)
    if overlap:
        problems.append(f"run ids present in both arms: {sorted(overlap)}")
    return problems


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_run_card(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT id, status, termination_reason, normalized_goal, plan_version,
               scoring_rule_version, prompt_bundle_version, graph_schema_revision,
               created_at, finished_at, quality_snapshot, usage_snapshot, budget_snapshot
        FROM research_runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return {"run_id": run_id, "found": False}
    (
        rid,
        status,
        termination_reason,
        goal,
        plan_version,
        scoring_rule_version,
        prompt_bundle_version,
        graph_schema_revision,
        created_at,
        finished_at,
        quality,
        usage,
        budget,
    ) = row
    quality = quality if isinstance(quality, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    budget = budget if isinstance(budget, dict) else {}
    return {
        "run_id": str(rid),
        "found": True,
        "status": status,
        "termination_reason": termination_reason,
        "normalized_goal": goal,
        "plan_version": int(plan_version) if plan_version is not None else None,
        "scoring_rule_version": scoring_rule_version,
        "prompt_bundle_version": prompt_bundle_version,
        "graph_schema_revision": graph_schema_revision,
        "created_at": str(created_at)[:19],
        "finished_at": str(finished_at)[:19] if finished_at else None,
        "quality": quality,
        "usage": usage,
        "budget": budget,
        "source_revision": budget.get("source_revision"),
        "era": normalize_revision(budget.get("source_revision")),
        "plan_template_run_id": budget.get("plan_template_run_id"),
        "tier": budget.get("tier"),
        "experiment_mode": usage.get("experiment_mode"),
        "stored_coverage": _as_float(quality.get("coverage")),
        "stored_accepted_evidence": _as_int(quality.get("accepted_evidence")),
        "stored_candidate_evidence": _as_int(quality.get("candidate_evidence")),
        "stored_claim_count": _as_int(quality.get("claim_count")),
        "stored_source_count": _as_int(quality.get("source_count")),
        "stored_independent_source_count": _as_int(quality.get("independent_source_count")),
        "stored_critical_gaps": _as_int(quality.get("critical_gaps")),
        "stored_coverage_map": quality.get("coverage_map") or [],
    }


def _fetch_plan_items(
    connection: psycopg.Connection, run_id: str, plan_version: int | None
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT question_id, priority, evidence_requirements, plan_version
        FROM research_plan_items
        WHERE run_id = %s AND plan_version = %s
        ORDER BY priority, question_id
        """,
        (run_id, plan_version),
    ).fetchall()
    return [
        {
            "question_id": str(question_id),
            "priority": int(priority),
            "evidence_requirements": list(requirements or []),
            "plan_version": int(version),
        }
        for question_id, priority, requirements, version in rows
    ]


def _fetch_claims(connection: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, question_id, dimension_key, claim_type, importance, status
        FROM research_claims WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "id": str(claim_id),
            "question_id": str(question_id),
            "dimension_key": str(dimension_key),
            "claim_type": claim_type,
            "importance": float(importance or 0.0),
            "status": status,
        }
        for claim_id, question_id, dimension_key, claim_type, importance, status in rows
    ]


def _fetch_evidence(connection: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, claim_id, question_id, accepted, source_id, rejection_reason
        FROM research_evidence WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "id": str(evidence_id),
            "claim_id": str(claim_id) if claim_id is not None else None,
            "question_id": str(question_id) if question_id is not None else None,
            "accepted": bool(accepted),
            "source_id": str(source_id) if source_id is not None else None,
            "rejection_reason": rejection_reason,
        }
        for evidence_id, claim_id, question_id, accepted, source_id, rejection_reason in rows
    ]


def _fetch_sources(connection: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT id, source_owner_key FROM research_sources WHERE run_id = %s",
        (run_id,),
    ).fetchall()
    return [
        {"id": str(source_id), "source_owner_key": str(owner or source_id)}
        for source_id, owner in rows
    ]


def _fetch_event_counts(connection: psycopg.Connection, run_id: str) -> dict[str, int]:
    rows = connection.execute(
        "SELECT event_type, count(*) FROM agent_events WHERE run_id = %s GROUP BY event_type",
        (run_id,),
    ).fetchall()
    return {str(event_type): int(count) for event_type, count in rows}


def _fetch_storage_stats(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    snapshots = (
        connection.execute(
            """
        SELECT count(*), coalesce(avg(char_count), 0)
        FROM research_source_snapshots WHERE run_id = %s
        """,
            (run_id,),
        ).fetchone()
        or (0, 0)
    )
    chunked = (
        connection.execute(
            """
        SELECT count(*) FROM research_source_snapshots s
        WHERE s.run_id = %s
          AND EXISTS (SELECT 1 FROM research_source_chunks c WHERE c.snapshot_id = s.id)
        """,
            (run_id,),
        ).fetchone()
        or (0,)
    )
    chunks = (
        connection.execute(
            """
        SELECT count(*), coalesce(avg(token_count), 0)
        FROM research_source_chunks WHERE run_id = %s
        """,
            (run_id,),
        ).fetchone()
        or (0, 0)
    )
    unique_urls = (
        connection.execute(
            """
        SELECT count(DISTINCT lower(r.url))
        FROM research_search_results r
        JOIN research_search_queries q ON r.search_query_id = q.id
        WHERE q.run_id = %s
        """,
            (run_id,),
        ).fetchone()
        or (0,)
    )
    return {
        "snapshots": int(snapshots[0] or 0),
        "avg_snapshot_chars": round(float(snapshots[1] or 0)),
        "chunked_snapshots": int(chunked[0] or 0),
        "chunks": int(chunks[0] or 0),
        "avg_chunk_tokens": round(float(chunks[1] or 0)),
        "unique_result_urls": int(unique_urls[0] or 0),
    }


def _fetch_search_stats(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    queries = (
        connection.execute(
            """
        SELECT count(*), coalesce(sum(result_count), 0),
               count(*) FILTER (WHERE coalesce(result_count, 0) = 0)
        FROM research_search_queries WHERE run_id = %s
        """,
            (run_id,),
        ).fetchone()
        or (0, 0, 0)
    )
    domains = (
        connection.execute(
            """
        SELECT count(DISTINCT split_part(
                   regexp_replace(lower(r.url), '^https?://(www\\.)?', ''), '/', 1))
        FROM research_search_results r
        JOIN research_search_queries q ON r.search_query_id = q.id
        WHERE q.run_id = %s
        """,
            (run_id,),
        ).fetchone()
        or (0,)
    )
    top_domains = connection.execute(
        """
        SELECT split_part(
                   regexp_replace(lower(r.url), '^https?://(www\\.)?', ''), '/', 1) AS domain,
               count(*) AS hits
        FROM research_search_results r
        JOIN research_search_queries q ON r.search_query_id = q.id
        WHERE q.run_id = %s
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """,
        (run_id,),
    ).fetchall()
    total_queries = int(queries[0] or 0)
    zero_results = int(queries[2] or 0)
    return {
        "search_queries": total_queries,
        "result_count_sum": int(queries[1] or 0),
        "zero_result_queries": zero_results,
        "zero_result_rate": ratio(zero_results, total_queries),
        "distinct_result_domains": int(domains[0] or 0),
        "top_domains": [{"domain": str(d), "hits": int(h)} for d, h in top_domains],
    }


def _fetch_rejections(connection: psycopg.Connection, run_id: str) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT coalesce(rejection_reason, '(none)'), count(*)
        FROM research_evidence WHERE run_id = %s GROUP BY 1 ORDER BY 2 DESC
        """,
        (run_id,),
    ).fetchall()
    return {str(reason): int(count) for reason, count in rows}


def _fetch_provider_stats(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    pools = (
        connection.execute(
            """
        SELECT count(*) FILTER (WHERE event_type = 'provider.execution.started'),
               count(*) FILTER (WHERE event_type = 'provider.attempt.failed'),
               count(*) FILTER (WHERE event_type = 'search.query.started'),
               count(*) FILTER (WHERE event_type = 'search.completed'),
               count(*) FILTER (WHERE event_type = 'search.reused'),
               count(*) FILTER (WHERE event_type = 'search.source_space_exhausted'),
               count(*) FILTER (WHERE event_type = 'feedback.query.execution.blocked'),
               count(*) FILTER (WHERE event_type = 'closure.feedback.generated')
        FROM agent_events WHERE run_id = %s
        """,
            (run_id,),
        ).fetchone()
        or (0, 0, 0, 0, 0, 0, 0, 0)
    )
    started, failed = int(pools[0] or 0), int(pools[1] or 0)
    return {
        "provider_execution_started": started,
        "provider_attempt_failed": failed,
        "provider_failure_rate": ratio(failed, started),
        "search_query_started": int(pools[2] or 0),
        "search_completed_events": int(pools[3] or 0),
        "search_reused_events": int(pools[4] or 0),
        "source_space_exhausted_events": int(pools[5] or 0),
        "feedback_blocked_events": int(pools[6] or 0),
        "closure_feedback_events": int(pools[7] or 0),
    }


def _fetch_budget_summary(card: dict[str, Any]) -> dict[str, Any]:
    budget = card.get("budget") or {}
    usage = card.get("usage") or {}
    pools = usage.get("resource_pools") or {}
    summary: dict[str, Any] = {"limits": {}, "committed": {}, "remaining": {}}
    for pool_name, pool in sorted(pools.items()):
        if not isinstance(pool, dict):
            continue
        summary["limits"][pool_name] = pool.get("limit")
        summary["committed"][pool_name] = pool.get("committed")
        summary["remaining"][pool_name] = pool.get("remaining")
    summary["deadline_at"] = budget.get("deadline_at")
    summary["max_wall_clock_seconds"] = budget.get("max_wall_clock_seconds")
    summary["iterations"] = _as_int(usage.get("iterations"))
    summary["replans"] = _as_int(usage.get("replans"))
    summary["search_reuses"] = _as_int(usage.get("search_reuses"))
    return summary


def analyze_run(connection: psycopg.Connection, run_id: str, arm: str) -> dict[str, Any]:
    """Collect every audit signal for one run (read-only SELECTs only)."""

    card = _fetch_run_card(connection, run_id)
    if not card.get("found"):
        return {"run_id": run_id, "arm": arm, "found": False}
    plan_items = _fetch_plan_items(connection, run_id, card.get("plan_version"))
    claims = _fetch_claims(connection, run_id)
    evidence = _fetch_evidence(connection, run_id)
    sources = _fetch_sources(connection, run_id)
    replay = replay_coverage(plan_items, claims, evidence, sources, card.get("plan_version"))
    storage = _fetch_storage_stats(connection, run_id)
    search = _fetch_search_stats(connection, run_id)
    events = _fetch_event_counts(connection, run_id)
    rejections = _fetch_rejections(connection, run_id)
    provider = _fetch_provider_stats(connection, run_id)
    budget = _fetch_budget_summary(card)
    facts = {
        "usage": card.get("usage") or {},
        "event_counts": events,
        "replay": replay,
        "storage": storage,
    }
    stages = stage_counts(facts)
    accepted_rate = ratio(replay["accepted_evidence"], replay["candidate_evidence"])
    return {
        "run_id": run_id,
        "arm": arm,
        "found": True,
        "card": card,
        "replay": replay,
        "stages": stages,
        "storage": storage,
        "search": search,
        "events": events,
        "rejections": rejections,
        "provider": provider,
        "budget": budget,
        "accepted_rate": accepted_rate,
        "coverage_match": (
            card.get("stored_coverage") is not None
            and abs(card["stored_coverage"] - replay["coverage"]) < 1e-9
        ),
    }


def discover_pool(
    connection: psycopg.Connection,
    goal: str,
    anchor_id: str,
    before_iso: str,
) -> list[dict[str, Any]]:
    """Candidate historical references from the anchor's benchmark family."""

    rows = connection.execute(
        """
        SELECT r.id, r.status, r.created_at, r.quality_snapshot->>'coverage',
               r.usage_snapshot->>'experiment_mode',
               r.budget_snapshot->>'source_revision',
               r.budget_snapshot->>'plan_template_run_id',
               r.quality_snapshot->>'accepted_evidence',
               r.quality_snapshot->>'source_count'
        FROM research_runs r
        WHERE r.normalized_goal = %s
          AND r.id <> %s
          AND r.created_at < %s
          AND (r.usage_snapshot->>'experiment_mode') IS NULL
          AND r.status IN ('completed', 'completed_with_limitations')
          AND coalesce((r.quality_snapshot->>'coverage')::float, 0) >= %s
        ORDER BY r.created_at DESC
        LIMIT 60
        """,
        (goal, anchor_id, before_iso, POOL_MIN_STORED_COVERAGE),
    ).fetchall()
    pool: list[dict[str, Any]] = []
    for (rid, status, created, coverage, _mode, revision, template, accepted, sources) in rows:
        pool.append(
            {
                "run_id": str(rid),
                "status": status,
                "created_at": str(created)[:19],
                "stored_coverage": _as_float(coverage),
                "stored_accepted_evidence": _as_int(accepted),
                "stored_source_count": _as_int(sources),
                "source_revision": revision,
                "era": normalize_revision(revision),
                "plan_template_run_id": template,
            }
        )
    return pool


def select_aux_runs(
    connection: psycopg.Connection, pool: list[dict[str, Any]], max_aux: int = POOL_MAX_AUX
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick auxiliary runs by recency (never by score) with completeness checks."""

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in pool:
        accepted = candidate.get("stored_accepted_evidence") or 0
        sources = candidate.get("stored_source_count") or 0
        if accepted < POOL_MIN_ACCEPTED_EVIDENCE or sources < POOL_MIN_SOURCE_COUNT:
            skipped.append({**candidate, "skip_reason": "incomplete stored quality counts"})
            continue
        plan_items = connection.execute(
            "SELECT count(*) FROM research_plan_items WHERE run_id = %s",
            (candidate["run_id"],),
        ).fetchone()
        if not plan_items or int(plan_items[0] or 0) == 0:
            skipped.append({**candidate, "skip_reason": "no persisted plan items"})
            continue
        eligible.append(candidate)
        if len(eligible) >= max_aux:
            break
    return eligible, skipped


#: Same-task control cohort: real historical runs of the *baseline* goal family
#: (experiment runs excluded), used to separate task shape from behaviour.
SAME_TASK_MAX_RUNS = 40
SAME_TASK_MIN_RUNS = 3

#: Anchor-family timeline sampling (real runs only; era-stratified pick).
ANCHOR_TIMELINE_SINCE = "2026-09-11T00:00:00"
ANCHOR_TIMELINE_PER_ERA_CAP = 6
ANCHOR_TIMELINE_TOTAL_CAP = 36

#: usage_snapshot keys that may carry stop / exhaustion signals (defensive read).
STOP_FLAG_KEYS = (
    "stop_reason",
    "deadline_exhausted",
    "source_space_exhausted",
    "query_strategy_exhausted",
    "reuse_only",
    "query_already_executed",
    "budget_exhausted",
    "feedback_execution_limit_reached",
)

DEFAULT_DATABASE_URL = "postgresql://deep_research:deep_research@127.0.0.1:5432/deep_research"
BENCHMARK_SUITE_PATH = REPOSITORY_ROOT / "evals" / "benchmarks" / "v1_benchmark_suite.v1.json"

UNKNOWN_NOTE = "not persisted in run rows; UNKNOWN (no speculation)"


def stored_map_required_summary(stored_map: list[Any]) -> dict[str, Any]:
    """Summarize the required_sources / coverage values recorded by an era's map."""

    required_values: Counter[str] = Counter()
    coverage_values: Counter[str] = Counter()
    present = 0
    dimensions = 0
    for entry in stored_map:
        if not isinstance(entry, dict):
            continue
        for status in entry.get("requirement_statuses") or []:
            if not isinstance(status, dict):
                continue
            dimensions += 1
            value = status.get("required_sources")
            if value is not None:
                present += 1
                required_values[str(value)] += 1
            coverage = status.get("coverage")
            if coverage is not None:
                coverage_values[str(coverage)] += 1
    return {
        "dimensions": dimensions,
        "field_present": present,
        "required_value_counts": dict(required_values),
        "coverage_value_counts": dict(coverage_values),
    }


def benchmark_lookup() -> dict[str, dict[str, Any]]:
    """Map benchmark goal text -> {benchmark_id, benchmark_version, type}."""

    try:
        payload = json.loads(BENCHMARK_SUITE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for entry in payload.get("benchmarks") or []:
        goal = entry.get("goal")
        if not goal:
            continue
        lookup[str(goal)] = {
            "benchmark_id": entry.get("benchmark_id"),
            "benchmark_version": entry.get("benchmark_version"),
            "type": entry.get("type"),
        }
    return lookup


def _fetch_family_runs(
    connection: psycopg.Connection,
    goal: str,
    since_iso: str | None = None,
    before_iso: str | None = None,
    statuses: tuple[str, ...] = ("completed", "completed_with_limitations"),
    include_experiments: bool = False,
) -> list[dict[str, Any]]:
    """Real (non-experiment by default) runs of one goal family, chronological."""

    placeholders = ", ".join(["%s"] * len(statuses))
    sql = (
        "SELECT r.id, r.status, r.termination_reason, r.created_at, "
        "r.quality_snapshot->>'coverage', r.quality_snapshot->>'accepted_evidence', "
        "r.usage_snapshot->>'experiment_mode', r.budget_snapshot->>'source_revision' "
        f"FROM research_runs r WHERE r.normalized_goal = %s AND r.status IN ({placeholders})"
    )
    params: list[Any] = [goal, *statuses]
    if not include_experiments:
        sql += " AND (r.usage_snapshot->>'experiment_mode') IS NULL"
    if since_iso:
        sql += " AND r.created_at >= %s"
        params.append(since_iso)
    if before_iso:
        sql += " AND r.created_at < %s"
        params.append(before_iso)
    sql += " ORDER BY r.created_at"
    rows = connection.execute(sql, params).fetchall()
    return [
        {
            "run_id": str(rid),
            "status": status,
            "termination_reason": termination_reason,
            "created_at": str(created_at)[:19],
            "stored_coverage": _as_float(coverage),
            "stored_accepted_evidence": _as_int(accepted),
            "experiment_mode": mode,
            "source_revision": revision,
            "era": normalize_revision(revision),
        }
        for rid, status, termination_reason, created_at, coverage, accepted, mode, revision in rows
    ]


def replay_run_light(
    connection: psycopg.Connection, run_id: str, with_stages: bool = False
) -> dict[str, Any]:
    """Compact current-definition replay of one run (read-only)."""

    card = _fetch_run_card(connection, run_id)
    if not card.get("found"):
        return {"run_id": run_id, "found": False}
    plan_items = _fetch_plan_items(connection, run_id, card.get("plan_version"))
    claims = _fetch_claims(connection, run_id)
    evidence = _fetch_evidence(connection, run_id)
    sources = _fetch_sources(connection, run_id)
    replay = replay_coverage(plan_items, claims, evidence, sources, card.get("plan_version"))
    rejections = _fetch_rejections(connection, run_id)
    events = _fetch_event_counts(connection, run_id)
    usage = card.get("usage") or {}
    row: dict[str, Any] = {
        "run_id": run_id,
        "found": True,
        "status": card.get("status"),
        "termination_reason": card.get("termination_reason"),
        "created_at": card.get("created_at"),
        "source_revision": card.get("source_revision"),
        "era": card.get("era"),
        "tier": card.get("tier"),
        "stored_coverage": card.get("stored_coverage"),
        "replayed_coverage": replay["coverage"],
        "stored_accepted_evidence": card.get("stored_accepted_evidence"),
        "accepted_evidence": replay["accepted_evidence"],
        "replayed_accepted_evidence": replay["accepted_evidence"],
        "candidate_evidence": replay["candidate_evidence"],
        "dimensions": replay["dimensions"],
        "dimensions_requiring_two_sources": replay["dimensions_requiring_two_sources"],
        "searches": _as_int(usage.get("searches")),
        "critical_gaps": card.get("stored_critical_gaps"),
        "gap_transitions": int(events.get("gap.closure.transition") or 0),
        "rejections": rejections,
        "stored_map_summary": stored_map_required_summary(card.get("stored_coverage_map") or []),
        "coverage_match": (
            card.get("stored_coverage") is not None
            and abs(card["stored_coverage"] - replay["coverage"]) < 1e-9
        ),
    }
    if with_stages:
        storage = _fetch_storage_stats(connection, run_id)
        row["storage"] = storage
        row["stages"] = stage_counts(
            {
                "usage": usage,
                "event_counts": events,
                "replay": replay,
                "storage": storage,
            }
        )
    return row


def _stratify_by_era(
    rows: list[dict[str, Any]], per_era_cap: int, total_cap: int
) -> list[dict[str, Any]]:
    """Keep the most recent rows per era bucket (chronological output)."""

    picked: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for row in reversed(rows):
        era = str(row.get("era"))
        if seen[era] >= per_era_cap:
            continue
        picked.append(row)
        seen[era] += 1
        if len(picked) >= total_cap:
            break
    picked.reverse()
    return picked


def collect_same_task(
    connection: psycopg.Connection,
    baseline_cards: list[dict[str, Any]],
    max_runs: int = SAME_TASK_MAX_RUNS,
) -> dict[str, Any]:
    """Historical (pre-baseline) real runs of the *baseline* goal family.

    Replayed under the current definition this cohort is the same-task
    control: if it sits at the baseline level there is no demonstrable
    same-task behaviour regression; if it sits clearly above there is one.
    """

    goal = next(
        (card.get("normalized_goal") for card in baseline_cards if card.get("normalized_goal")),
        None,
    )
    if not goal:
        return {
            "goal": None,
            "rows": [],
            "hist_mean": None,
            "stored_mean": None,
            "accepted_mean": None,
            "hist_n": 0,
            "status": "insufficient_data",
        }
    created = [card["created_at"] for card in baseline_cards if card.get("created_at")]
    before_iso = min(created) if created else None
    family = _fetch_family_runs(connection, goal, before_iso=before_iso)
    family = family[-max_runs:]
    rows = [replay_run_light(connection, entry["run_id"], with_stages=True) for entry in family]
    played = [row for row in rows if row.get("found")]
    return {
        "goal": goal,
        "rule": (
            "same normalized_goal as the baseline arm, experiment_mode IS NULL, status "
            "completed/completed_with_limitations, created before the earliest baseline "
            f"run; most recent {max_runs} kept; coverage replayed with the current definition"
        ),
        "candidate_runs": len(family),
        "rows": rows,
        "hist_mean": mean_or_none([row["replayed_coverage"] for row in played]),
        "stored_mean": mean_or_none([row["stored_coverage"] for row in played]),
        "accepted_mean": mean_or_none([row["accepted_evidence"] for row in played]),
        "hist_n": len(played),
        "status": "ok" if played else "insufficient_data",
    }


def collect_anchor_timeline(
    connection: psycopg.Connection,
    anchor_goal: str | None,
    since_iso: str = ANCHOR_TIMELINE_SINCE,
) -> dict[str, Any]:
    """Era-stratified real-run timeline of the anchor's benchmark family."""

    if not anchor_goal:
        return {
            "since": since_iso,
            "candidate_runs": 0,
            "sampled_runs": 0,
            "rows": [],
            "era_aggregates": [],
            "status": "insufficient_data",
        }
    family = _fetch_family_runs(
        connection,
        anchor_goal,
        since_iso=since_iso,
        statuses=("completed", "completed_with_limitations", "failed"),
    )
    picked = _stratify_by_era(family, ANCHOR_TIMELINE_PER_ERA_CAP, ANCHOR_TIMELINE_TOTAL_CAP)
    rows = [replay_run_light(connection, entry["run_id"]) for entry in picked]
    return {
        "since": since_iso,
        "candidate_runs": len(family),
        "sampled_runs": len(rows),
        "rows": rows,
        "era_aggregates": era_aggregate(rows),
        "status": "ok" if rows else "insufficient_data",
    }


def arm_stage_summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-stage mean over an arm (None values are excluded, never zeroed)."""

    summary: dict[str, dict[str, Any]] = {}
    for stage, _label in FUNNEL_STAGES:
        values = [
            float(run["stages"][stage])
            for run in runs
            if run.get("stages") and run["stages"].get(stage) is not None
        ]
        summary[stage] = {
            "mean": mean_or_none(values),
            "n": len(values),
            "missing": len(runs) - len(values),
        }
    return summary


def arm_stage_means(summary: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    return {stage: payload["mean"] for stage, payload in summary.items()}


def build_funnel_section(
    historical_runs: list[dict[str, Any]],
    current_runs: list[dict[str, Any]],
    threshold: float = DIVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    """Funnel comparison of two arms with conversions and first divergence."""

    historical_summary = arm_stage_summary(historical_runs)
    current_summary = arm_stage_summary(current_runs)
    rows = build_funnel_comparison(
        arm_stage_means(historical_summary), arm_stage_means(current_summary), threshold
    )
    for index, row in enumerate(rows):
        row["historical_n"] = historical_summary[row["stage"]]["n"]
        row["current_n"] = current_summary[row["stage"]]["n"]
        if index == 0:
            row["conversion_historical"] = None
            row["conversion_current"] = None
            continue
        previous = rows[index - 1]
        row["conversion_historical"] = ratio(row["historical"], previous["historical"])
        row["conversion_current"] = ratio(row["current"], previous["current"])
    return {
        "stages": rows,
        "first_divergence": find_first_divergence(rows, threshold),
        "unit_rows": build_unit_comparison(
            arm_stage_means(historical_summary), arm_stage_means(current_summary), threshold
        ),
        "historical_summary": historical_summary,
        "current_summary": current_summary,
    }


def definition_diff_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Stored coverage_map vs current-definition dimensions, per requirement."""

    stored_map = (run.get("card") or {}).get("stored_coverage_map") or []
    current_lookup: dict[str, dict[str, Any]] = {}
    for question in (run.get("replay") or {}).get("coverage_map") or []:
        for requirement in question["requirements"]:
            current_lookup[str(requirement["dimension_key"])] = requirement
    rows: list[dict[str, Any]] = []
    for entry in stored_map:
        if not isinstance(entry, dict):
            continue
        question_id = str(entry.get("question_id") or "")
        for status in entry.get("requirement_statuses") or []:
            if not isinstance(status, dict):
                continue
            dimension_key = str(status.get("dimension_key") or "")
            current = current_lookup.get(dimension_key)
            stored_required = status.get("required_sources")
            current_required = current.get("required_sources") if current else None
            drift = (
                stored_required is not None
                and current_required is not None
                and int(stored_required) != int(current_required)
            )
            rows.append(
                {
                    "question_id": question_id or dimension_key.split(":")[0],
                    "dimension_key": dimension_key,
                    "stored_required_sources": stored_required,
                    "current_required_sources": current_required,
                    "stored_coverage": status.get("coverage"),
                    "current_coverage": current.get("coverage") if current else None,
                    "stored_field_present": stored_required is not None,
                    "required_sources_drift": drift,
                }
            )
    return rows


def summarize_definition_diff(run: dict[str, Any], max_examples: int = 4) -> dict[str, Any]:
    rows = definition_diff_rows(run)
    drift = [row for row in rows if row["required_sources_drift"]]
    absent = [row for row in rows if not row["stored_field_present"]]
    card = run.get("card") or {}
    replay = run.get("replay") or {}
    return {
        "run_id": run.get("run_id"),
        "stored_map_questions": len(card.get("stored_coverage_map") or []),
        "current_plan_items": replay.get("plan_items"),
        "stored_dimensions": len(rows),
        "stored_required_field_absent": len(absent),
        "required_sources_drift_dimensions": len(drift),
        "examples": drift[:max_examples],
        "stored_map_summary": stored_map_required_summary(
            card.get("stored_coverage_map") or []
        ),
        "current_dimensions_requiring_two_sources": replay.get(
            "dimensions_requiring_two_sources"
        ),
        "stored_coverage": card.get("stored_coverage"),
        "replayed_coverage": replay.get("coverage"),
        "coverage_match": run.get("coverage_match"),
    }


def build_definition_diff_section(historical_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage-definition component diff: measured rows plus explicit UNKNOWNs."""

    run_summaries = [summarize_definition_diff(run) for run in historical_runs]
    drift_dimensions = sum(s["required_sources_drift_dimensions"] for s in run_summaries)
    absent_fields = sum(s["stored_required_field_absent"] for s in run_summaries)
    stored_dimensions = sum(s["stored_dimensions"] for s in run_summaries)
    stored_questions = sum(s.get("stored_map_questions") or 0 for s in run_summaries)
    current_plan_items = sum(s.get("current_plan_items") or 0 for s in run_summaries)
    current_two = sum(s.get("current_dimensions_requiring_two_sources") or 0 for s in run_summaries)
    coverage_values: Counter[str] = Counter()
    for summary in run_summaries:
        coverage_values.update(summary["stored_map_summary"].get("coverage_value_counts") or {})
    mismatch_runs = [s["run_id"] for s in run_summaries if not s.get("coverage_match")]
    matched = len(run_summaries) - len(mismatch_runs)
    score_supported = set(coverage_values) <= {"0.0", "0", "0.5", "1.0", "1"}
    components = [
        {
            "component": "coverage numerator (evidence per dimension)",
            "historical_semantics": (
                "accepted evidence rows carrying dimension keys, as persisted in the era's "
                "coverage_map"
            ),
            "current_semantics": (
                "claims JOIN accepted evidence JOIN sources grouped by "
                "(question_id, dimension_key)"
            ),
            "could_affect_score": (
                f"replay reproduces stored coverage on {matched}/{len(run_summaries)} "
                "historical runs"
            ),
            "basis": "measured (replay vs stored)",
        },
        {
            "component": "coverage denominator / weighting",
            "historical_semantics": "run's own plan_version items; question weight = 4 - priority",
            "current_semantics": (
                "identical port: weighted mean over questions, rounded to 4 decimals"
            ),
            "could_affect_score": (
                f"stored map questions {stored_questions} vs current plan items "
                f"{current_plan_items}"
            ),
            "basis": "measured (count comparison)",
        },
        {
            "component": "independent-owner requirement (two-source rule)",
            "historical_semantics": (
                "eras whose stored maps record required_sources=1 on every dimension"
            ),
            "current_semantics": (
                "2 when the criterion matches an independent-source marker OR any claim of "
                "the dimension is high risk; else 1"
            ),
            "could_affect_score": (
                f"{drift_dimensions} stored dimensions record 1 while the current definition "
                f"requires 2; {absent_fields} stored dimensions carry no required_sources "
                f"field at all; current definition requires 2 on {current_two} dimensions"
            ),
            "basis": "measured (stored coverage_map fields)",
        },
        {
            "component": "score mapping per dimension",
            "historical_semantics": (
                f"stored dimension coverage values observed in {sorted(coverage_values) or 'n/a'}"
            ),
            "current_semantics": (
                "1.0 when evidence>0 and owners>=required; 0.5 when evidence>0; else 0.0"
            ),
            "could_affect_score": (
                "no drift detected" if score_supported else "stored values outside {0, 0.5, 1}"
            ),
            "basis": "measured (stored coverage values)",
        },
        {
            "component": "gap requirement projection / closure semantics",
            "historical_semantics": "UNKNOWN (not recoverable from persisted rows)",
            "current_semantics": (
                "gap closure computed in-loop; the coverage snapshot does not read gap tables"
            ),
            "could_affect_score": "unknown",
            "basis": "UNKNOWN (no speculation)",
        },
        {
            "component": "plan_version filtering",
            "historical_semantics": "stored map reflects the plan version at snapshot time",
            "current_semantics": "plan items filtered by research_runs.plan_version",
            "could_affect_score": (
                "no drift detected"
                if stored_questions == current_plan_items
                else "stored/current question counts differ"
            ),
            "basis": "measured (count comparison)",
        },
        {
            "component": "open/closed and current-state filtering",
            "historical_semantics": "UNKNOWN (no per-dimension closed/open flag in the stored map)",
            "current_semantics": "no current-state filter beyond evidence.accepted",
            "could_affect_score": "unknown",
            "basis": "UNKNOWN (no speculation)",
        },
    ]
    return {
        "components": components,
        "runs": run_summaries,
        "totals": {
            "stored_dimensions": stored_dimensions,
            "required_sources_drift_dimensions": drift_dimensions,
            "stored_required_field_absent": absent_fields,
            "stored_coverage_value_counts": dict(coverage_values),
            "stored_map_questions": stored_questions,
            "current_plan_items": current_plan_items,
            "current_dimensions_requiring_two_sources": current_two,
            "coverage_mismatch_runs": mismatch_runs,
        },
    }


def acceptance_policy_summary(runs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    per_run: list[dict[str, Any]] = []
    for run in runs:
        reasons = run.get("rejections") or {}
        totals.update(reasons)
        per_run.append({"run_id": run.get("run_id"), "reasons": dict(reasons)})
    return {
        "label": label,
        "reason_totals": dict(totals.most_common()),
        "reason_kinds": sorted(totals),
        "per_run": per_run,
    }


def _distinct_card_values(runs: list[dict[str, Any]], getter: Any) -> list[str]:
    values = set()
    for run in runs:
        card = run.get("card") or {}
        value = getter(card)
        if value is not None:
            values.add(str(value))
    return sorted(values)


def _distinct_nested_values(runs: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
    def getter(card: dict[str, Any]) -> Any:
        for container in (card.get("usage") or {}, card.get("budget") or {}):
            for key in keys:
                if container.get(key) is not None:
                    return container.get(key)
        return None

    return _distinct_card_values(runs, getter)


def build_config_diff(
    historical_runs: list[dict[str, Any]],
    current_runs: list[dict[str, Any]],
    benchmark_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Arm-level configuration comparison with an explicit classification."""

    rows: list[dict[str, Any]] = []

    def compare(field: str, getter: Any, classification: str, note: str = "") -> None:
        historical = _distinct_card_values(historical_runs, getter)
        current = _distinct_card_values(current_runs, getter)
        rows.append(classify_config_difference(field, historical, current, classification, note))

    compare(
        "normalized_goal",
        lambda card: card.get("normalized_goal"),
        "research_semantic",
        "benchmark task identity",
    )
    compare(
        "benchmark_id",
        lambda card: benchmark_map.get(card.get("normalized_goal") or "", {}).get("benchmark_id"),
        "research_semantic",
        "from evals/benchmarks/v1_benchmark_suite.v1.json",
    )
    compare(
        "benchmark_version",
        lambda card: benchmark_map.get(card.get("normalized_goal") or "", {}).get(
            "benchmark_version"
        ),
        "research_semantic",
        "suite definition version",
    )
    compare(
        "plan_template_run_id",
        lambda card: card.get("plan_template_run_id"),
        "research_semantic",
        "frozen/reused plan template drives plan shape",
    )
    compare(
        "scoring_rule_version",
        lambda card: card.get("scoring_rule_version"),
        "research_semantic",
        "scoring rule revision",
    )
    compare(
        "prompt_bundle_version",
        lambda card: card.get("prompt_bundle_version"),
        "research_semantic",
        "prompt bundle revision",
    )
    compare(
        "graph_schema_revision",
        lambda card: card.get("graph_schema_revision"),
        "research_semantic",
        "evidence graph schema revision",
    )
    compare(
        "plan_version",
        lambda card: card.get("plan_version"),
        "infrastructure",
        "internal replan counter; rows filtered per run",
    )
    compare(
        "source_revision",
        lambda card: card.get("source_revision"),
        "infrastructure",
        "code revision stamp",
    )
    compare("era", lambda card: card.get("era"), "infrastructure", "normalized revision era")
    compare(
        "tier",
        lambda card: card.get("tier"),
        "infrastructure",
        "budget scale; limits verified in the budget section",
    )
    compare(
        "experiment_mode",
        lambda card: card.get("experiment_mode"),
        "research_semantic",
        "experiment tagging of the run",
    )

    optional_fields = (
        ("profile", ("profile", "research_profile"), "unknown"),
        ("credential_class", ("credential_class", "credential_tier"), "unknown"),
        (
            "worker_concurrency",
            ("max_worker_concurrency", "worker_concurrency", "concurrency"),
            "infrastructure",
        ),
        ("alembic_revision", ("alembic_revision", "migration_version"), "unknown"),
    )
    for field, keys, classification in optional_fields:
        historical = _distinct_nested_values(historical_runs, keys)
        current = _distinct_nested_values(current_runs, keys)
        if not historical and not current:
            rows.append(classify_config_difference(field, None, None, "unknown", UNKNOWN_NOTE))
            continue
        note = "recovered from usage/budget snapshot"
        if classification == "unknown" and historical != current:
            note = "recovered value differs; semantic class uncertain"
        rows.append(classify_config_difference(field, historical, current, classification, note))

    semantic_differences = [
        row for row in rows if row["classification"] == "research_semantic" and not row["equal"]
    ]
    return {
        "rows": rows,
        "research_semantic_difference_count": len(semantic_differences),
        "research_semantic_differences": [row["field"] for row in semantic_differences],
    }


def _provider_run_row(run: dict[str, Any]) -> dict[str, Any]:
    provider = run.get("provider") or {}
    search = run.get("search") or {}
    stages = run.get("stages") or {}
    searches = stages.get("search_completed")
    return {
        "run_id": run.get("run_id"),
        "provider_execution_started": provider.get("provider_execution_started"),
        "provider_attempt_failed": provider.get("provider_attempt_failed"),
        "provider_failure_rate": provider.get("provider_failure_rate"),
        "search_queries": search.get("search_queries"),
        "result_count_sum": search.get("result_count_sum"),
        "results_per_query": ratio(search.get("result_count_sum"), search.get("search_queries")),
        "zero_result_rate": search.get("zero_result_rate"),
        "distinct_result_domains": search.get("distinct_result_domains"),
        "unique_urls": stages.get("unique_urls"),
        "unique_urls_per_search": ratio(stages.get("unique_urls"), searches),
        "unique_readable_per_search": ratio(stages.get("unique_readable"), searches),
        "top_domains": (search.get("top_domains") or [])[:5],
    }


def build_provider_comparison(
    historical_runs: list[dict[str, Any]], current_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    historical = [_provider_run_row(run) for run in historical_runs]
    current = [_provider_run_row(run) for run in current_runs]

    def arm_mean(rows: list[dict[str, Any]], key: str) -> float | None:
        return mean_or_none([row[key] for row in rows if row.get(key) is not None])

    mean_keys = (
        "provider_failure_rate",
        "results_per_query",
        "zero_result_rate",
        "distinct_result_domains",
        "unique_urls_per_search",
        "unique_readable_per_search",
    )
    return {
        "historical": historical,
        "current": current,
        "historical_mean": {key: arm_mean(historical, key) for key in mean_keys},
        "current_mean": {key: arm_mean(current, key) for key in mean_keys},
    }


def detect_provider_environment_divergence(comparison: dict[str, Any]) -> dict[str, Any]:
    """Flag only degradation-shaped provider signals (current worse than past)."""

    details: list[str] = []
    flagged = False
    historical = comparison.get("historical_mean") or {}
    current = comparison.get("current_mean") or {}
    history_failure = historical.get("provider_failure_rate")
    current_failure = current.get("provider_failure_rate")
    if history_failure is not None and current_failure is not None and history_failure > 0:
        relative = (current_failure - history_failure) / history_failure
        if relative >= 0.5 and (current_failure - history_failure) >= 0.05:
            flagged = True
            details.append(
                f"provider failure rate {history_failure} -> {current_failure} "
                f"(+{round(relative, 4)} relative)"
            )
    history_results = historical.get("results_per_query")
    current_results = current.get("results_per_query")
    if history_results is not None and current_results is not None and history_results > 0:
        relative = (history_results - current_results) / history_results
        if relative >= 0.5:
            flagged = True
            details.append(
                f"results per query {history_results} -> {current_results} "
                f"(-{round(relative, 4)} relative)"
            )
    history_zero = historical.get("zero_result_rate")
    current_zero = current.get("zero_result_rate")
    if history_zero is not None and current_zero is not None:
        increase = current_zero - history_zero
        if increase >= 0.1 and (history_zero == 0 or increase / history_zero >= 0.5):
            flagged = True
            details.append(f"zero-result rate {history_zero} -> {current_zero}")
    return {
        "flagged": flagged,
        "details": details,
        "note": (
            "provider statistics are measured on different query mixes across tasks; "
            "treat as a secondary signal only"
        ),
    }


def _budget_row(run: dict[str, Any]) -> dict[str, Any]:
    budget = run.get("budget") or {}
    provider = run.get("provider") or {}
    card = run.get("card") or {}
    usage = card.get("usage") or {}
    stop_flags = {key: usage.get(key) for key in STOP_FLAG_KEYS if key in usage}
    return {
        "run_id": run.get("run_id"),
        "status": card.get("status"),
        "termination_reason": card.get("termination_reason"),
        "iterations": budget.get("iterations"),
        "replans": budget.get("replans"),
        "search_reuses": budget.get("search_reuses"),
        "deadline_at": budget.get("deadline_at"),
        "max_wall_clock_seconds": budget.get("max_wall_clock_seconds"),
        "limits": budget.get("limits") or {},
        "committed": budget.get("committed") or {},
        "remaining": budget.get("remaining") or {},
        "stop_flags": stop_flags,
        "source_space_exhausted_events": provider.get("source_space_exhausted_events"),
        "feedback_blocked_events": provider.get("feedback_blocked_events"),
        "search_reused_events": provider.get("search_reused_events"),
    }


def build_budget_comparison(
    historical_runs: list[dict[str, Any]], current_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    historical = [_budget_row(run) for run in historical_runs]
    current = [_budget_row(run) for run in current_runs]
    pools = sorted({pool for row in historical + current for pool in row["limits"]})
    limit_rows: list[dict[str, Any]] = []
    for pool in pools:
        historical_limits = sorted(
            {str(row["limits"][pool]) for row in historical if pool in row["limits"]}
        )
        current_limits = sorted(
            {str(row["limits"][pool]) for row in current if pool in row["limits"]}
        )
        limit_rows.append(
            {
                "pool": pool,
                "historical_limits": historical_limits,
                "current_limits": current_limits,
                "equal": historical_limits == current_limits,
                "classification": "infrastructure",
            }
        )
    return {"historical": historical, "current": current, "limit_rows": limit_rows}


def _behaviour_divergence_counts(section: dict[str, Any] | None) -> tuple[int, int]:
    """(comparable behaviour stages, divergent behaviour stages) of a funnel."""

    if not section:
        return 0, 0
    comparable = 0
    divergent = 0
    for row in section.get("stages") or []:
        if row.get("stage") not in BEHAVIOUR_STAGES or not row.get("comparable"):
            continue
        comparable += 1
        if row.get("divergent"):
            divergent += 1
    return comparable, divergent


def build_findings(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Root-cause findings, each in Observation/Evidence/Inference/Confidence form."""

    findings: list[dict[str, Any]] = []
    historical = ctx["historical"]
    current = ctx["current"]
    same_task = ctx["same_task"]
    funnel = ctx["funnel"]
    same_task_funnel = ctx.get("same_task_funnel")
    config_diff = ctx["config_diff"]
    provider = ctx["provider"]
    provider_env = ctx["provider_env"]
    acceptance = ctx["acceptance_diff"]
    definition_diff = ctx["definition_diff"]
    budget = ctx["budget"]
    timeline = ctx["timeline"]

    goal_row = next((r for r in config_diff["rows"] if r["field"] == "normalized_goal"), None)
    if goal_row and not goal_row["equal"]:
        findings.append(
            {
                "id": "task-identity",
                "title": "Historical headline runs and the current baseline are different tasks",
                "observation": (
                    "The historical arm (anchor + auxiliary runs) and the current baseline do "
                    "not share normalized_goal, plan template or plan shape; the headline "
                    "coverage gap spans two different benchmark tasks."
                ),
                "evidence": [
                    f"historical goal: {goal_row['historical']}",
                    f"current goal: {goal_row['current']}",
                    f"historical plan shape: {historical['plan_shape']}",
                    f"current plan shape: {current['plan_shape']}",
                    f"benchmark ids: {historical['benchmark_id']} vs {current['benchmark_id']}",
                ],
                "inference": (
                    "The stored 0.8+ vs ~0.45 comparison is cross-task by construction and "
                    "cannot be read as a like-for-like behaviour regression."
                ),
                "confidence": "high",
            }
        )

    findings.append(
        {
            "id": "metric-replay",
            "title": "Stored coverage was re-derived with the current definition",
            "observation": (
                "quality_snapshot.coverage was never taken at face value: every run was "
                "replayed from plan items, claims, evidence and sources with the current formula."
            ),
            "evidence": [
                f"historical arm stored mean {historical['stored_mean']} vs replayed mean "
                f"{historical['replay_mean']}",
                f"historical replay mismatches: {historical['replay_mismatch_runs'] or 'none'}",
                f"current arm stored mean {current['stored_mean']} vs replayed mean "
                f"{current['replay_mean']}",
                f"current replay mismatches: {current['replay_mismatch_runs'] or 'none'}",
                f"same-task cohort stored mean {same_task.get('stored_mean')} vs replayed mean "
                f"{same_task.get('hist_mean')} (n={same_task.get('hist_n')})",
            ],
            "inference": (
                "Where stored and replayed values diverge the difference is definitional; "
                "where they match the stored score is reproducible under today's rules."
            ),
            "confidence": "high",
        }
    )

    drift_dimensions = definition_diff["totals"]["required_sources_drift_dimensions"]
    absent_fields = definition_diff["totals"]["stored_required_field_absent"]
    same_stored = same_task.get("stored_mean")
    same_replay = same_task.get("hist_mean")
    same_task_gap = (
        round(same_stored - same_replay, 4)
        if same_stored is not None and same_replay is not None
        else None
    )
    if drift_dimensions or absent_fields or (same_task_gap is not None and same_task_gap > 0.05):
        evidence = [
            f"{drift_dimensions} stored dimensions record required_sources=1 while the current "
            "definition requires 2",
            f"{absent_fields} stored dimensions carry no required_sources field at all",
            f"stored dimension coverage values observed: "
            f"{definition_diff['totals']['stored_coverage_value_counts']}",
        ]
        examples: list[dict[str, Any]] = next(
            (s["examples"] for s in definition_diff["runs"] if s["examples"]), []
        )
        for example in examples[:3]:
            evidence.append(
                f"example: {example['dimension_key']} stored required="
                f"{example['stored_required_sources']} current required="
                f"{example['current_required_sources']}"
            )
        if same_task_gap is not None:
            evidence.append(
                f"same-task cohort stored mean {same_stored} vs replayed mean {same_replay} "
                f"(gap {same_task_gap})"
            )
        findings.append(
            {
                "id": "definition-drift",
                "title": "Coverage / acceptance definition drifted across eras",
                "observation": (
                    "Older runs were scored with a looser independent-owner rule and a narrower "
                    "acceptance vocabulary; their stored scores are not comparable to today's "
                    "values without replay."
                ),
                "evidence": evidence,
                "inference": (
                    "A material part of the 'historical 0.8+' numbers, where they come from "
                    "older same-task runs, is definitional rather than behavioural."
                ),
                "confidence": "high" if drift_dimensions else "medium",
            }
        )

    current_kinds = set(acceptance["current"]["reason_kinds"])
    old_kinds = set(acceptance["historical"]["reason_kinds"])
    cohort_kinds = set((acceptance.get("same_task") or {}).get("reason_kinds") or [])
    historical_kinds = old_kinds | cohort_kinds
    if historical_kinds and historical_kinds != current_kinds:
        findings.append(
            {
                "id": "acceptance-policy-drift",
                "title": "Accepted-evidence statistics are not one vocabulary across eras",
                "observation": (
                    "Rejection reasons stored by old-era runs differ from the current rejection "
                    "vocabulary, so raw accepted_evidence counts are not comparable across eras "
                    "without a replay."
                ),
                "evidence": [
                    f"historical arm reasons: {sorted(old_kinds) or 'none'}",
                    f"same-task cohort reasons: {sorted(cohort_kinds) or 'none'}",
                    f"current reasons: {sorted(current_kinds) or 'none'}",
                ],
                "inference": (
                    "Acceptance strictness / reason vocabulary changed; the drop in stored "
                    "'accepted' counts is partly a change of statistic, not of search reach."
                ),
                "confidence": "medium",
            }
        )

    cross_first = funnel.get("first_divergence")
    same_comparable, same_divergent = _behaviour_divergence_counts(same_task_funnel)
    if cross_first:
        evidence = [
            f"first cross-task divergence: {cross_first['stage']} "
            f"(relative delta {cross_first['relative_delta']}, kind {cross_first['kind']})"
        ]
        if same_task_funnel:
            same_first = same_task_funnel.get("first_divergence")
            for row in same_task_funnel.get("stages") or []:
                if row.get("stage") in BEHAVIOUR_STAGES and row.get("divergent"):
                    evidence.append(
                        f"same-task behaviour divergence: {row['stage']} "
                        f"historical {row['historical']} -> current {row['current']} "
                        f"(relative {row['relative_delta']})"
                    )
            evidence.append(
                f"same-task first divergence: {same_first['stage'] if same_first else 'none'}"
            )
        findings.append(
            {
                "id": "funnel-divergence",
                "title": "Unified funnel first divergence",
                "observation": (
                    "Planning stages diverge first (task shape); behaviour stages are then "
                    "checked inside the same task to isolate true regression."
                ),
                "evidence": evidence,
                "inference": (
                    f"Same-task behaviour-stage divergences: {same_divergent} of "
                    f"{same_comparable} comparable stages."
                ),
                "confidence": "high" if same_task_funnel else "medium",
            }
        )

    limits_equal = bool(budget["limit_rows"]) and all(row["equal"] for row in budget["limit_rows"])
    findings.append(
        {
            "id": "budget-scheduling",
            "title": "Budget gate timeline",
            "observation": (
                "Budget limits and stop conditions were compared pool by pool between the arms."
            ),
            "evidence": [
                f"all pool limits equal across arms: {limits_equal}",
                f"historical termination reasons: "
                f"{sorted({str(row['termination_reason']) for row in budget['historical']})}",
                f"current termination reasons: "
                f"{sorted({str(row['termination_reason']) for row in budget['current']})}",
                f"historical source_space_exhausted events: "
                f"{[row['source_space_exhausted_events'] for row in budget['historical']]}",
                f"current source_space_exhausted events: "
                f"{[row['source_space_exhausted_events'] for row in budget['current']]}",
            ],
            "inference": (
                "Equal limits and comparable stop reasons indicate the current baseline was not "
                "gated out materially earlier than the historical runs."
                if limits_equal
                else "Limit differences exist; the current arm may stop earlier by configuration."
            ),
            "confidence": "high" if limits_equal else "medium",
        }
    )

    findings.append(
        {
            "id": "provider-environment",
            "title": "Provider / environment signals",
            "observation": (
                "Provider failure rate, results per query and zero-result rate were compared "
                "between the arms (cross-task query-mix caveat applies)."
            ),
            "evidence": [
                f"mean provider failure rate: historical "
                f"{provider['historical_mean']['provider_failure_rate']} -> current "
                f"{provider['current_mean']['provider_failure_rate']}",
                f"mean results per query: historical "
                f"{provider['historical_mean']['results_per_query']} -> current "
                f"{provider['current_mean']['results_per_query']}",
                f"mean zero-result rate: historical "
                f"{provider['historical_mean']['zero_result_rate']} -> current "
                f"{provider['current_mean']['zero_result_rate']}",
                f"flagged degradation signals: {provider_env['details'] or 'none'}",
            ],
            "inference": (
                "Provider/environment drift remains a secondary signal; no threshold-crossing "
                "degradation was measured."
                if not provider_env["flagged"]
                else "Degradation-shaped provider signals exist; quantify before attributing."
            ),
            "confidence": "medium",
        }
    )

    same_hist = same_task.get("hist_mean")
    same_cur = current.get("replay_mean")
    delta = (
        round(same_hist - same_cur, 4)
        if same_hist is not None and same_cur is not None
        else None
    )
    if same_task.get("hist_n"):
        findings.append(
            {
                "id": "same-task-control",
                "title": "Same-task control cohort (the decisive separator)",
                "observation": (
                    "Real historical runs of the baseline's own goal, replayed with the current "
                    "definition, form the same-task control for temporal regression."
                ),
                "evidence": [
                    f"cohort rule: {same_task.get('rule')}",
                    f"cohort size: {same_task.get('hist_n')} runs",
                    f"cohort stored mean {same_task.get('stored_mean')} vs replayed mean "
                    f"{same_hist}",
                    f"current baseline replay mean {same_cur}",
                    f"same-task temporal delta (replayed - baseline): {delta}",
                ],
                "inference": (
                    "No demonstrable same-task behaviour regression: the cohort replays at the "
                    "baseline level (|delta| <= 0.10)."
                    if delta is not None and abs(delta) <= 0.10
                    else "Same-task replay sits materially above the baseline; treat as a "
                    "genuine regression signal pending the funnel check."
                ),
                "confidence": "high" if same_task.get("hist_n", 0) >= 5 else "medium",
            }
        )

    anchor_timeline = timeline.get("anchor_family") or {}
    eras = anchor_timeline.get("era_aggregates") or []
    if anchor_timeline.get("candidate_runs"):
        evidence = [
            f"anchor-family real runs since {anchor_timeline.get('since')}: "
            f"{anchor_timeline.get('candidate_runs')} (sampled "
            f"{anchor_timeline.get('sampled_runs')})"
        ]
        for era in eras:
            evidence.append(
                f"era {era['era']}: runs={era['runs']} stored={era['mean_stored_coverage']} "
                f"replayed={era['mean_replayed_coverage']} "
                f"accepted={era['mean_accepted_evidence']}"
            )
        thin = [era for era in eras if era.get("insufficient_observations")]
        findings.append(
            {
                "id": "timeline",
                "title": "Era timeline of the anchor benchmark family",
                "observation": (
                    "Era-aggregated real runs of the anchor family locate (or fail to locate) "
                    "a step change in current-definition coverage."
                ),
                "evidence": evidence,
                "inference": (
                    "Insufficient historical observations: eras with <2 runs cannot locate "
                    "the step change." if thin else "Era aggregates above."
                ),
                "confidence": "low" if thin else "medium",
            }
        )
    else:
        findings.append(
            {
                "id": "timeline",
                "title": "Era timeline of the anchor benchmark family",
                "observation": "Insufficient historical observations",
                "evidence": ["no real anchor-family runs matched the timeline window"],
                "inference": "No timeline conclusion is drawn (no speculation).",
                "confidence": "low",
            }
        )

    return findings


def build_report(
    connection: psycopg.Connection,
    anchor_id: str,
    aux_override: list[str] | None,
    baseline_ids: list[str],
    pool_max_aux: int = POOL_MAX_AUX,
) -> dict[str, Any]:
    """Assemble the full read-only audit report."""

    anchor_run = analyze_run(connection, anchor_id, "historical_anchor")
    if not anchor_run.get("found"):
        raise SystemExit(f"historical anchor run not found: {anchor_id}")
    anchor_goal = anchor_run["card"].get("normalized_goal")

    baseline_runs = [analyze_run(connection, run_id, "current_baseline") for run_id in baseline_ids]
    missing_baseline = [run["run_id"] for run in baseline_runs if not run.get("found")]
    baseline_runs = [run for run in baseline_runs if run.get("found")]
    if not baseline_runs:
        raise SystemExit("no current baseline runs found")

    benchmark_map = benchmark_lookup()
    before_iso = min(run["card"]["created_at"] for run in baseline_runs)

    pool_records: list[dict[str, Any]] = []
    pool_skipped: list[dict[str, Any]] = []
    aux_runs: list[dict[str, Any]] = []
    if aux_override:
        selection_mode = "explicit --historical-aux-run-id override"
        aux_runs = [analyze_run(connection, run_id, "historical_aux") for run_id in aux_override]
        aux_runs = [run for run in aux_runs if run.get("found")]
    else:
        selection_mode = "auto-discovery by recency (never by score)"
        if anchor_goal:
            pool_records = discover_pool(connection, anchor_goal, anchor_id, before_iso)
            eligible, pool_skipped = select_aux_runs(connection, pool_records, max_aux=pool_max_aux)
            aux_runs = [
                analyze_run(connection, row["run_id"], "historical_aux") for row in eligible
            ]
            aux_runs = [run for run in aux_runs if run.get("found")]

    baseline_id_set = {run["run_id"] for run in baseline_runs}
    aux_runs = [
        run
        for run in aux_runs
        if run["run_id"] not in baseline_id_set and run["run_id"] != anchor_run["run_id"]
    ]
    historical_runs = [anchor_run, *aux_runs]

    funnel = build_funnel_section(historical_runs, baseline_runs)
    definition_diff = build_definition_diff_section(historical_runs)
    acceptance_diff = {
        "historical": acceptance_policy_summary(historical_runs, "historical_anchor+aux"),
        "current": acceptance_policy_summary(baseline_runs, "current_baseline"),
    }
    provider = build_provider_comparison(historical_runs, baseline_runs)
    provider_env = detect_provider_environment_divergence(provider)
    budget = build_budget_comparison(historical_runs, baseline_runs)
    config_diff = build_config_diff(historical_runs, baseline_runs, benchmark_map)

    same_task = collect_same_task(connection, [run["card"] for run in baseline_runs])
    same_task_runs = [row for row in same_task["rows"] if row.get("found") and row.get("stages")]
    same_task_funnel = (
        build_funnel_section(same_task_runs, baseline_runs)
        if len(same_task_runs) >= SAME_TASK_MIN_RUNS
        else None
    )
    same_task["funnel"] = same_task_funnel
    same_task["funnel_status"] = "ok" if same_task_funnel else "insufficient_data"
    acceptance_diff["same_task"] = acceptance_policy_summary(
        same_task["rows"], "same_task_cohort"
    )

    anchor_timeline = collect_anchor_timeline(connection, anchor_goal)
    timeline = {
        "same_task": sorted(
            (row for row in same_task["rows"] if row.get("found")),
            key=lambda row: str(row.get("created_at")),
        ),
        "anchor_family": anchor_timeline,
    }

    historical_stored_mean = mean_or_none(
        [
            run["card"]["stored_coverage"]
            for run in historical_runs
            if run["card"].get("stored_coverage") is not None
        ]
    )
    current_stored_mean = mean_or_none(
        [
            run["card"]["stored_coverage"]
            for run in baseline_runs
            if run["card"].get("stored_coverage") is not None
        ]
    )
    historical_replay_mean = mean_or_none([run["replay"]["coverage"] for run in historical_runs])
    current_replay_mean = mean_or_none([run["replay"]["coverage"] for run in baseline_runs])
    historical_mismatch = [
        run["run_id"] for run in historical_runs if not run.get("coverage_match")
    ]
    current_mismatch = [run["run_id"] for run in baseline_runs if not run.get("coverage_match")]

    def _delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return round(left - right, 4)

    cross_comparable, cross_behaviour = _behaviour_divergence_counts(funnel)
    if same_task_funnel:
        behaviour_count = _behaviour_divergence_counts(same_task_funnel)[1]
        behaviour_source = "same_task_funnel"
    else:
        behaviour_count = cross_behaviour
        behaviour_source = "cross_task_funnel"

    primary = decide_primary_case(
        {
            "hist_replay_mean": historical_replay_mean,
            "cur_replay_mean": current_replay_mean,
            "formula_match_all": not historical_mismatch,
            "formula_mismatch_runs": historical_mismatch,
            "first_divergence_kind": (funnel.get("first_divergence") or {}).get("kind"),
            "behaviour_divergence_count": behaviour_count,
            "config_semantic_diff_count": config_diff["research_semantic_difference_count"],
            "same_task": {
                "hist_mean": same_task.get("hist_mean"),
                "cur_mean": current_replay_mean,
                "hist_n": same_task.get("hist_n") or 0,
            },
            "provider_environment_divergence": provider_env["flagged"],
        }
    )
    primary["components"]["behaviour_divergence_source"] = behaviour_source
    primary["components"]["cross_task_behaviour_divergences"] = cross_behaviour
    primary["components"]["cross_task_comparable_behaviour_stages"] = cross_comparable

    metric_replay: list[dict[str, Any]] = []
    for run in historical_runs + baseline_runs:
        card = run["card"]
        stored = card.get("stored_coverage")
        metric_replay.append(
            {
                "run_id": run["run_id"],
                "arm": run["arm"],
                "status": card.get("status"),
                "created_at": card.get("created_at"),
                "source_revision": card.get("source_revision"),
                "era": card.get("era"),
                "historical_reported_coverage": stored,
                "recomputed_current_coverage": run["replay"]["coverage"],
                "delta": _delta(stored, run["replay"]["coverage"]),
                "coverage_match": run.get("coverage_match"),
                "stored_accepted_evidence": card.get("stored_accepted_evidence"),
                "current_definition_accepted_evidence": run["replay"]["accepted_evidence"],
                "candidate_evidence": run["replay"]["candidate_evidence"],
                "accepted_rate": run.get("accepted_rate"),
                "plan_items": run["replay"]["plan_items"],
                "dimensions": run["replay"]["dimensions"],
                "dimensions_requiring_two_sources": run["replay"][
                    "dimensions_requiring_two_sources"
                ],
                "stored_critical_gaps": card.get("stored_critical_gaps"),
            }
        )

    same_stored_mean = same_task.get("stored_mean")
    same_replay_mean = same_task.get("hist_mean")
    decomposition: dict[str, Any] = {
        "historical_arm": {
            "stored_mean": historical_stored_mean,
            "replayed_mean": historical_replay_mean,
            "definition_drift_points": _delta(historical_stored_mean, historical_replay_mean),
        },
        "current_baseline": {
            "stored_mean": current_stored_mean,
            "replayed_mean": current_replay_mean,
        },
        "headline_gap": {
            "stored_to_now_points": _delta(historical_stored_mean, current_replay_mean),
            "replayed_to_now_points": _delta(historical_replay_mean, current_replay_mean),
        },
        "same_task_cohort": {
            "n": same_task.get("hist_n"),
            "stored_mean": same_stored_mean,
            "replayed_mean": same_replay_mean,
            "definition_drift_points": _delta(same_stored_mean, same_replay_mean),
            "temporal_points_vs_baseline": _delta(same_replay_mean, current_replay_mean),
        },
        "task_configuration_points": _delta(historical_replay_mean, same_replay_mean),
        "attribution_labels": {
            "definition_drift": (
                "stored-to-replay collapse inside one era/task; score semantics, not behaviour"
            ),
            "task_configuration": (
                "historical-arm replay vs same-task replay; a different benchmark task and "
                "plan shape, not a behaviour change"
            ),
            "same_task_temporal": (
                "same-task replay vs current baseline; the only bucket that can signal a true "
                "regression"
            ),
        },
    }

    core_answer: list[str] = []
    headline = decomposition["headline_gap"]["stored_to_now_points"]
    if headline is not None:
        core_answer.append(f"Headline stored-to-now gap: {headline} coverage points.")
    if historical_stored_mean is not None and historical_replay_mean is not None:
        core_answer.append(
            f"Historical arm: stored {historical_stored_mean} vs current-definition replay "
            f"{historical_replay_mean} (definitional drift "
            f"{decomposition['historical_arm']['definition_drift_points']})."
        )
    if same_replay_mean is not None:
        core_answer.append(
            f"Same-task cohort (n={same_task.get('hist_n')}): stored {same_stored_mean} vs "
            f"replay {same_replay_mean}; temporal delta vs baseline "
            f"{decomposition['same_task_cohort']['temporal_points_vs_baseline']}."
        )
    core_answer.append(
        f"Decision inputs: replays {historical_replay_mean} (historical) vs "
        f"{current_replay_mean} (baseline); same-task flat under the current definition: "
        f"{primary['components'].get('same_task_flat_under_current_definition')}."
    )

    historical_meta = {
        "plan_shape": (
            f"{anchor_run['replay']['plan_items']} items / "
            f"{anchor_run['replay']['dimensions']} dims"
        ),
        "benchmark_id": benchmark_map.get(anchor_goal or "", {}).get("benchmark_id"),
        "stored_mean": historical_stored_mean,
        "replay_mean": historical_replay_mean,
        "replay_mismatch_runs": historical_mismatch,
    }
    current_meta = {
        "plan_shape": (
            f"{baseline_runs[0]['replay']['plan_items']} items / "
            f"{baseline_runs[0]['replay']['dimensions']} dims"
        ),
        "benchmark_id": benchmark_map.get(
            baseline_runs[0]["card"].get("normalized_goal") or "", {}
        ).get("benchmark_id"),
        "stored_mean": current_stored_mean,
        "replay_mean": current_replay_mean,
        "replay_mismatch_runs": current_mismatch,
    }
    findings = build_findings(
        {
            "historical": historical_meta,
            "current": current_meta,
            "same_task": same_task,
            "funnel": funnel,
            "same_task_funnel": same_task_funnel,
            "config_diff": config_diff,
            "provider": provider,
            "provider_env": provider_env,
            "acceptance_diff": acceptance_diff,
            "definition_diff": definition_diff,
            "budget": budget,
            "timeline": timeline,
        }
    )

    return {
        "schema_version": "historical-baseline-regression.v1",
        "generated_at": utc_now_iso(),
        "discipline": {
            "mode": "analysis-only / historical replay; strictly read-only SELECTs",
            "stored_values_policy": "quality_snapshot.coverage is never taken at face value",
            "benchmark_rerun": "none",
            "production_behavior_changed": False,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "pool_rule": {
                "min_stored_coverage": POOL_MIN_STORED_COVERAGE,
                "min_accepted_evidence": POOL_MIN_ACCEPTED_EVIDENCE,
                "min_source_count": POOL_MIN_SOURCE_COUNT,
                "max_aux": pool_max_aux,
                "ranking": "recency (never by score) to avoid lucky-run selection bias",
            },
            "unknown_marker": "UNKNOWN fields are marked, never speculated",
        },
        "dataset": {
            "anchor_run_id": anchor_id,
            "aux_run_ids": [run["run_id"] for run in aux_runs],
            "historical_arm_ids": [run["run_id"] for run in historical_runs],
            "current_baseline_ids": [run["run_id"] for run in baseline_runs],
            "missing_baseline_ids": missing_baseline,
            "selection": {
                "mode": selection_mode,
                "pool_size": len(pool_records),
                "pool_records": pool_records,
                "pool_skipped": pool_skipped,
                "rule": (
                    "same normalized_goal as the anchor, experiment_mode IS NULL, status "
                    "completed/completed_with_limitations, stored coverage >= "
                    f"{POOL_MIN_STORED_COVERAGE}, created before the baseline window; "
                    "ranked by recency"
                ),
            },
            "run_cards": [run["card"] for run in historical_runs + baseline_runs],
        },
        "metric_replay": metric_replay,
        "definition_diff": definition_diff,
        "acceptance_policy_diff": acceptance_diff,
        "config_diff": config_diff,
        "funnel": funnel,
        "provider_environment": {"comparison": provider, "divergence": provider_env},
        "budget_scheduling": budget,
        "same_task_control": same_task,
        "timeline": timeline,
        "decomposition": decomposition,
        "core_question_answer": core_answer,
        "findings": findings,
        "primary_diagnosis": primary,
        "next_actions": next_actions_for_case(primary.get("case")),
    }


def next_actions_for_case(case: str | None) -> list[str]:
    """Recommendations only; this audit never executes them."""

    if case == "A":
        return [
            "Unify the benchmark metric baseline: re-score stored runs with the current "
            "definition before any comparison.",
            "Do not change research logic; the apparent drop would be a scoring-semantics "
            "artifact.",
        ]
    if case == "B":
        return [
            "Enter a targeted regression bisect over the revisions between the last flat "
            "same-task run and the baseline window, guided by the first divergence stage.",
            "Do not change behaviour until the bisect identifies the introducing revision.",
        ]
    if case == "C":
        return [
            "Re-establish provider/source yield (provider pool health, search result yield) "
            "before re-evaluating coverage.",
            "Treat current coverage as environment-limited until the yield recovers.",
        ]
    if case == "D":
        return [
            "First calibrate metric definitions (replay-based comparison everywhere) so "
            "scores are comparable across eras.",
            "Then address the largest measured real regression component; quantify each piece "
            "before any behaviour change.",
        ]
    return [
        "Collect at least one replayable historical run per arm before drawing conclusions "
        "(insufficient data)."
    ]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if not value:
            return "(none)"
        return "; ".join(_fmt_cell(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "(none)"
        return ", ".join(f"{key}={_fmt_cell(item)}" for key, item in value.items())
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(cell) for cell in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render the audit report in the A-K structure required by the spec."""

    lines: list[str] = []
    lines.append("# Historical Baseline Regression Audit (read-only replay)")
    lines.append("")
    discipline = report.get("discipline") or {}
    lines.append(f"- generated_at: {report.get('generated_at')}")
    lines.append(f"- mode: {discipline.get('mode')}")
    lines.append(f"- stored values policy: {discipline.get('stored_values_policy')}")
    lines.append(f"- benchmark rerun: {discipline.get('benchmark_rerun')}")
    lines.append("")

    dataset = report["dataset"]
    lines.append("## A. Dataset")
    lines.append("")
    lines.append(f"- anchor: {dataset['anchor_run_id']}")
    lines.append(f"- aux runs: {_fmt_cell(dataset['aux_run_ids'])}")
    lines.append(f"- current baseline: {_fmt_cell(dataset['current_baseline_ids'])}")
    selection = dataset["selection"]
    lines.append(f"- selection mode: {selection['mode']}")
    lines.append(f"- pool rule: {selection['rule']}")
    if dataset["missing_baseline_ids"]:
        lines.append(f"- missing baseline runs: {_fmt_cell(dataset['missing_baseline_ids'])}")
    lines.append("")
    lines.append("Run cards:")
    lines.append("")
    lines.extend(
        _table(
            [
                "run_id",
                "status",
                "created_at",
                "era",
                "source_revision",
                "tier",
                "plan_version",
                "plan_template",
                "stored_cov",
                "goal(60)",
            ],
            [
                [
                    card.get("run_id"),
                    card.get("status"),
                    card.get("created_at"),
                    card.get("era"),
                    card.get("source_revision"),
                    card.get("tier"),
                    card.get("plan_version"),
                    card.get("plan_template_run_id"),
                    card.get("stored_coverage"),
                    (card.get("normalized_goal") or "")[:60],
                ]
                for card in dataset["run_cards"]
            ],
        )
    )
    lines.append("")
    pool = selection["pool_records"]
    if pool:
        lines.append(f"Historical candidate pool ({len(pool)} runs, first 20 shown):")
        lines.append("")
        lines.extend(
            _table(
                ["run_id", "created_at", "stored_cov", "accepted", "sources", "era"],
                [
                    [
                        row["run_id"],
                        row["created_at"],
                        row["stored_coverage"],
                        row["stored_accepted_evidence"],
                        row["stored_source_count"],
                        row["era"],
                    ]
                    for row in pool[:20]
                ],
            )
        )
        lines.append("")
    if selection["pool_skipped"]:
        lines.append("Skipped pool candidates:")
        lines.append("")
        lines.extend(
            _table(
                ["run_id", "created_at", "stored_cov", "reason"],
                [
                    [row["run_id"], row["created_at"], row["stored_coverage"], row["skip_reason"]]
                    for row in selection["pool_skipped"]
                ],
            )
        )
        lines.append("")

    lines.append("## B. Old Metric vs Current Metric Replay")
    lines.append("")
    lines.extend(
        _table(
            [
                "run_id",
                "arm",
                "era",
                "historical reported",
                "current-definition",
                "delta",
                "match",
                "stored accepted",
                "current-def accepted",
                "accepted rate",
            ],
            [
                [
                    row["run_id"],
                    row["arm"],
                    row["era"],
                    row["historical_reported_coverage"],
                    row["recomputed_current_coverage"],
                    row["delta"],
                    row["coverage_match"],
                    row["stored_accepted_evidence"],
                    row["current_definition_accepted_evidence"],
                    row["accepted_rate"],
                ]
                for row in report["metric_replay"]
            ],
        )
    )
    lines.append("")

    definition = report["definition_diff"]
    lines.append("## C. Coverage Definition Drift")
    lines.append("")
    lines.extend(
        _table(
            [
                "component",
                "historical semantics",
                "current semantics",
                "could affect score",
                "basis",
            ],
            [
                [
                    row["component"],
                    row["historical_semantics"],
                    row["current_semantics"],
                    row["could_affect_score"],
                    row["basis"],
                ]
                for row in definition["components"]
            ],
        )
    )
    lines.append("")
    lines.append("Per-run stored-map drift:")
    lines.append("")
    lines.extend(
        _table(
            [
                "run_id",
                "stored dims",
                "field absent",
                "drift dims",
                "stored required values",
                "current req=2",
                "stored cov",
                "replayed cov",
                "match",
            ],
            [
                [
                    row["run_id"],
                    row["stored_dimensions"],
                    row["stored_required_field_absent"],
                    row["required_sources_drift_dimensions"],
                    row["stored_map_summary"]["required_value_counts"],
                    row["current_dimensions_requiring_two_sources"],
                    row["stored_coverage"],
                    row["replayed_coverage"],
                    row["coverage_match"],
                ]
                for row in definition["runs"]
            ],
        )
    )
    lines.append("")

    acceptance = report["acceptance_policy_diff"]
    lines.append("Accepted-evidence policy diff (rejection reasons per arm):")
    lines.append("")
    lines.extend(
        _table(
            ["arm", "reason kinds", "reason totals"],
            [
                [
                    acceptance[key]["label"],
                    acceptance[key]["reason_kinds"],
                    acceptance[key]["reason_totals"],
                ]
                for key in ("historical", "same_task", "current")
                if key in acceptance
            ],
        )
    )
    lines.append("")

    funnel = report["funnel"]
    lines.append("## D. Unified Funnel Comparison (historical arm vs current baseline)")
    lines.append("")
    lines.extend(
        _table(
            [
                "stage",
                "historical",
                "n",
                "current",
                "n",
                "delta",
                "relative",
                "conv hist",
                "conv cur",
                "divergent",
            ],
            [
                [
                    row["label"],
                    row["historical"],
                    row["historical_n"],
                    row["current"],
                    row["current_n"],
                    row["delta"],
                    row["relative_delta"],
                    row["conversion_historical"],
                    row["conversion_current"],
                    row["divergent"],
                ]
                for row in funnel["stages"]
            ],
        )
    )
    lines.append("")
    lines.extend(
        _table(
            ["unit metric", "historical", "current", "delta", "relative", "divergent"],
            [
                [
                    row["metric"],
                    row["historical"],
                    row["current"],
                    row["delta"],
                    row["relative_delta"],
                    row["divergent"],
                ]
                for row in funnel["unit_rows"]
            ],
        )
    )
    lines.append("")
    same_task_control = report["same_task_control"]
    if same_task_control.get("funnel"):
        lines.append("Same-task funnel (cohort vs current baseline; identical task):")
        lines.append("")
        lines.extend(
            _table(
                ["stage", "historical", "current", "delta", "relative", "divergent"],
                [
                    [
                        row["label"],
                        row["historical"],
                        row["current"],
                        row["delta"],
                        row["relative_delta"],
                        row["divergent"],
                    ]
                    for row in same_task_control["funnel"]["stages"]
                ],
            )
        )
        lines.append("")

    lines.append("## E. First Divergence Point")
    lines.append("")
    first = funnel.get("first_divergence")
    if first:
        lines.append(
            f"- Cross-task funnel first material divergence: **{first['stage']}** "
            f"(kind: {first['kind']}, relative delta {first['relative_delta']})."
        )
    else:
        lines.append("- Cross-task funnel: no stage crossed the divergence threshold.")
    if same_task_control.get("funnel"):
        same_first = same_task_control["funnel"].get("first_divergence")
        if same_first:
            lines.append(
                f"- Same-task funnel first material divergence: **{same_first['stage']}** "
                f"(kind: {same_first['kind']}, relative delta {same_first['relative_delta']})."
            )
        else:
            lines.append(
                "- Same-task funnel: no stage crossed the divergence threshold (no "
                "same-task regression detected)."
            )
    else:
        lines.append("- Same-task funnel: insufficient_data (cohort below minimum size).")
    lines.append("")

    budget = report["budget_scheduling"]
    lines.append("## F. Budget / Scheduling Comparison")
    lines.append("")
    lines.extend(
        _table(
            ["pool", "historical limits", "current limits", "equal"],
            [
                [row["pool"], row["historical_limits"], row["current_limits"], row["equal"]]
                for row in budget["limit_rows"]
            ],
        )
    )
    lines.append("")
    lines.extend(
        _table(
            [
                "run_id",
                "status",
                "termination",
                "iterations",
                "reuses",
                "remaining",
                "stop flags",
                "source-space exhausted",
                "feedback blocked",
            ],
            [
                [
                    row["run_id"],
                    row["status"],
                    row["termination_reason"],
                    row["iterations"],
                    row["search_reuses"],
                    row["remaining"],
                    row["stop_flags"],
                    row["source_space_exhausted_events"],
                    row["feedback_blocked_events"],
                ]
                for row in budget["historical"] + budget["current"]
            ],
        )
    )
    lines.append("")

    provider_block = report["provider_environment"]
    comparison = provider_block["comparison"]
    lines.append("## G. Provider / Environment Comparison")
    lines.append("")
    lines.extend(
        _table(
            [
                "arm",
                "failure rate",
                "results/query",
                "zero-result rate",
                "domains",
                "unique urls/search",
                "unique readable/search",
            ],
            [
                [
                    "historical mean",
                    comparison["historical_mean"]["provider_failure_rate"],
                    comparison["historical_mean"]["results_per_query"],
                    comparison["historical_mean"]["zero_result_rate"],
                    comparison["historical_mean"]["distinct_result_domains"],
                    comparison["historical_mean"]["unique_urls_per_search"],
                    comparison["historical_mean"]["unique_readable_per_search"],
                ],
                [
                    "current mean",
                    comparison["current_mean"]["provider_failure_rate"],
                    comparison["current_mean"]["results_per_query"],
                    comparison["current_mean"]["zero_result_rate"],
                    comparison["current_mean"]["distinct_result_domains"],
                    comparison["current_mean"]["unique_urls_per_search"],
                    comparison["current_mean"]["unique_readable_per_search"],
                ],
            ],
        )
    )
    lines.append("")
    lines.extend(
        _table(
            [
                "run_id",
                "failure rate",
                "results/query",
                "zero-rate",
                "unique urls/search",
                "unique readable/search",
            ],
            [
                [
                    row["run_id"],
                    row["provider_failure_rate"],
                    row["results_per_query"],
                    row["zero_result_rate"],
                    row["unique_urls_per_search"],
                    row["unique_readable_per_search"],
                ]
                for row in comparison["historical"] + comparison["current"]
            ],
        )
    )
    lines.append("")
    divergence = provider_block["divergence"]
    lines.append(
        f"- degradation-shaped signal flagged: {divergence['flagged']} "
        f"({_fmt_cell(divergence['details'])})"
    )
    lines.append(f"- caveat: {divergence['note']}")
    lines.append("")

    lines.append("## H. Root Cause Findings")
    lines.append("")
    for finding in report["findings"]:
        lines.append(f"### {finding['id']}: {finding['title']}")
        lines.append("")
        lines.append(f"- Observation: {finding['observation']}")
        lines.append("- Evidence:")
        for item in finding["evidence"]:
            lines.append(f"  - {item}")
        lines.append(f"- Inference: {finding['inference']}")
        lines.append(f"- Confidence: {finding['confidence']}")
        lines.append("")

    diagnosis = report["primary_diagnosis"]
    lines.append("## I. Primary Diagnosis")
    lines.append("")
    lines.append(
        f"- Case: **{diagnosis.get('case')} - {diagnosis.get('label')}** "
        f"(status: {diagnosis.get('status')})"
    )
    for item in diagnosis.get("rationale") or []:
        lines.append(f"- rationale: {item}")
    lines.append("- components:")
    for key, value in (diagnosis.get("components") or {}).items():
        lines.append(f"  - {key} = {_fmt_cell(value)}")
    lines.append("")

    lines.append("## J. Core Question")
    lines.append("")
    lines.append(
        "Q: why did coverage reach ~0.8+ historically while the current baseline sits at ~0.45?"
    )
    lines.append("")
    for item in report["core_question_answer"]:
        lines.append(f"- {item}")
    lines.append("")
    decomposition = report["decomposition"]
    lines.extend(
        _table(
            ["bucket", "value"],
            [
                ["historical stored mean", decomposition["historical_arm"]["stored_mean"]],
                [
                    "historical replayed mean (current definition)",
                    decomposition["historical_arm"]["replayed_mean"],
                ],
                [
                    "historical arm definition drift",
                    decomposition["historical_arm"]["definition_drift_points"],
                ],
                [
                    "current baseline replayed mean",
                    decomposition["current_baseline"]["replayed_mean"],
                ],
                ["stored-to-now gap", decomposition["headline_gap"]["stored_to_now_points"]],
                ["replayed-to-now gap", decomposition["headline_gap"]["replayed_to_now_points"]],
                ["same-task cohort n", decomposition["same_task_cohort"]["n"]],
                ["same-task stored mean", decomposition["same_task_cohort"]["stored_mean"]],
                ["same-task replayed mean", decomposition["same_task_cohort"]["replayed_mean"]],
                [
                    "same-task definition drift",
                    decomposition["same_task_cohort"]["definition_drift_points"],
                ],
                [
                    "same-task temporal points vs baseline",
                    decomposition["same_task_cohort"]["temporal_points_vs_baseline"],
                ],
                [
                    "task configuration points (historical replay - same-task replay)",
                    decomposition.get("task_configuration_points"),
                ],
            ],
        )
    )
    lines.append("")

    lines.append("## K. Next Action (recommendations only, not executed)")
    lines.append("")
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 14.4 supplemental audit: replay historical runs with the current "
            "coverage definition and attribute the historical-vs-baseline gap (read-only)."
        )
    )
    parser.add_argument("--historical-run-id", default=DEFAULT_ANCHOR_RUN_ID)
    parser.add_argument(
        "--historical-aux-run-id",
        default=None,
        help="comma-separated aux override; default = auto-discovery by recency",
    )
    parser.add_argument(
        "--current-baseline-run-id", default=",".join(DEFAULT_CURRENT_BASELINE_IDS)
    )
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--pool-max-aux", type=int, default=POOL_MAX_AUX)
    args = parser.parse_args(argv)

    baseline_ids = _split_run_ids(args.current_baseline_run_id)
    aux_override = _split_run_ids(args.historical_aux_run_id)
    problems = validate_run_sets(args.historical_run_id, aux_override, baseline_ids)
    if problems:
        for problem in problems:
            print(f"[historical-baseline-regression] input problem: {problem}", file=sys.stderr)
        return 2

    with psycopg.connect(_database_uri(args.database_url)) as connection:
        report = build_report(
            connection,
            args.historical_run_id,
            aux_override or None,
            baseline_ids,
            pool_max_aux=args.pool_max_aux,
        )

    json_path = Path(args.json_out)
    markdown_path = Path(args.markdown_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    diagnosis = report["primary_diagnosis"]
    print(f"[historical-baseline-regression] case: {diagnosis.get('case')} "
          f"({diagnosis.get('label')})")
    print(f"[historical-baseline-regression] json: {json_path}")
    print(f"[historical-baseline-regression] markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
