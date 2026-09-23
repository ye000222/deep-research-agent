"""Phase 14.3 Task C: read-only A/B analysis of evidence-aware enrichment.

Compares one or more ``baseline`` runs (``EVIDENCE_AWARE_CONTEXT_ENABLED=false``)
against one or more ``evidence_aware`` candidate runs (switch on) that share the
same goal, budget tier, fixture, owner, provider profile and credential.  The
*only* intended difference is whether the Phase 14.2 ResearchContextEnricher
transformed the real SearchTarget query.

The tool is strictly additive: it issues ``SELECT`` statements against the
persisted run/gap/event projections, imports the pure gap classifier from the
Phase 13.0 bottleneck analyzer, and never imports or mutates runtime behavior.

Five metric groups (per the Phase 14.3 spec):

  1. Gap            total / closed / open / abandoned / closure_rate
  2. Failure bucket MISSING_INDEPENDENT_SOURCE / CLAIM_NOT_VERIFIED / ...
  3. Evidence       accepted evidence, extraction events, reports
  4. Search         search volume, enrichment count, average query-length delta,
                    and a direct proof that the provider query text changed
  5. Context effect failure_reason -> applied hints -> enriched query -> gap
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
DEFAULT_JSON_OUT = REPOSITORY_ROOT / "artifacts" / "phase14_3_ab_report.json"
DEFAULT_MARKDOWN_OUT = REPOSITORY_ROOT / "artifacts" / "phase14_3_ab_report.md"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# Reuse the pure, unit-tested bottleneck classifier instead of duplicating it.
from analyze_gap_closure_bottleneck import (  # noqa: E402
    BUCKET_ORDER,
    GapSignals,
    classify_gap,
)

ENRICHED_EVENT = "research.query.context.enriched"
RECORDED_EVENT = "research.context.recorded"
FEEDBACK_EVENT = "closure.feedback.generated"
SEARCH_STARTED_EVENT = "search.query.started"
EVIDENCE_EXTRACTED_EVENT = "evidence.extracted"
NEED_REFINED_EVENT = "research.need.refined"

CLOSED_GAP_STATUS = "resolved"
OPEN_GAP_STATUSES = ("open", "resolving")
ABANDONED_GAP_STATUS = "abandoned"


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _split_run_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _profile(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT status, termination_reason,
               usage_snapshot ->> 'experiment_mode',
               budget_snapshot ->> 'tier',
               budget_snapshot ->> 'source_revision',
               llm_config_snapshot ->> 'model',
               normalized_goal,
               COALESCE((quality_snapshot ->> 'coverage')::float, 0.0),
               COALESCE((quality_snapshot ->> 'priority_one_coverage')::float, 0.0),
               COALESCE((quality_snapshot ->> 'cross_validation')::float, 0.0),
               COALESCE((quality_snapshot ->> 'critical_gaps')::int, 0),
               COALESCE((quality_snapshot ->> 'accepted_evidence')::int, 0)
        FROM research_runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return {"run_id": run_id, "found": False}
    keys = (
        "status",
        "termination_reason",
        "experiment_mode",
        "budget_tier",
        "source_revision",
        "model",
        "normalized_goal",
        "coverage",
        "priority_one_coverage",
        "cross_validation",
        "critical_gaps",
        "accepted_evidence",
    )
    profile = dict(zip(keys, row, strict=True))
    profile["run_id"] = run_id
    profile["found"] = True
    return profile


def _gap_metrics(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT status, count(*)
        FROM research_gaps
        WHERE run_id = %s
          AND plan_version = (SELECT max(plan_version) FROM research_gaps WHERE run_id = %s)
        GROUP BY status
        """,
        (run_id, run_id),
    ).fetchall()
    counts = {str(status): int(count) for status, count in rows}
    closed = counts.get(CLOSED_GAP_STATUS, 0)
    open_gaps = sum(counts.get(status, 0) for status in OPEN_GAP_STATUSES)
    abandoned = counts.get(ABANDONED_GAP_STATUS, 0)
    total = closed + open_gaps + abandoned
    return {
        "total_gap": total,
        "closed_gap": closed,
        "open_gap": open_gaps,
        "abandoned_gap": abandoned,
        "closure_rate": round(closed / total, 4) if total else 0.0,
        "by_status": counts,
    }


def _failure_buckets(
    connection: psycopg.Connection, run_id: str
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT DISTINCT ON (question_id) question_id, status
        FROM research_gaps
        WHERE run_id = %s
        ORDER BY question_id, plan_version DESC
        """,
        (run_id,),
    ).fetchall()
    latest_status = {str(question): str(status) for question, status in rows}

    feedback = connection.execute(
        """
        SELECT refs
        FROM agent_events
        WHERE run_id = %s AND event_type = %s
        ORDER BY run_seq
        """,
        (run_id, FEEDBACK_EVENT),
    ).fetchall()
    # Keep the latest feedback per (question, dimension) to classify unclosed gaps.
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for (refs,) in feedback:
        if not isinstance(refs, dict):
            continue
        question_id = str(refs.get("question_id") or "")
        dimension_key = str(refs.get("dimension_key") or "")
        if not question_id or not dimension_key:
            continue
        latest[(question_id, dimension_key)] = {
            "requirement_type": str(refs.get("requirement_type") or ""),
            "failure_reason": str(refs.get("failure_reason") or ""),
        }

    gaps: list[GapSignals] = []
    for (question_id, dimension_key), fb in sorted(latest.items()):
        if latest_status.get(question_id, "open") == CLOSED_GAP_STATUS:
            continue  # already closed -> not a bottleneck
        gaps.append(
            GapSignals(
                question_id=question_id,
                dimension_key=dimension_key,
                priority=0,
                requirement_type=fb["requirement_type"],
                failure_reason=fb["failure_reason"],
                closure_result="",
                required_evidence_count=0,
                current_evidence_count=0,
                required_independent_sources=0,
                current_independent_sources=0,
                current_coverage=0.0,
                required_coverage=0.0,
                alignment_status="none",
            )
        )
    buckets = Counter(classify_gap(gap) for gap in gaps)
    total = sum(buckets.values()) or 1
    return {
        "unclosed_total": len(gaps),
        "distribution": {
            bucket: {
                "count": buckets.get(bucket, 0),
                "share": round(buckets.get(bucket, 0) / total, 4),
            }
            for bucket in BUCKET_ORDER
        },
    }


def _event_counts(connection: psycopg.Connection, run_id: str) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT event_type, count(*)
        FROM agent_events
        WHERE run_id = %s
          AND event_type = ANY(%s)
        GROUP BY event_type
        """,
        (run_id, [
            ENRICHED_EVENT,
            RECORDED_EVENT,
            SEARCH_STARTED_EVENT,
            EVIDENCE_EXTRACTED_EVENT,
            NEED_REFINED_EVENT,
        ]),
    ).fetchall()
    counts = {str(event_type): int(count) for event_type, count in rows}
    for event_type in (
        ENRICHED_EVENT,
        RECORDED_EVENT,
        SEARCH_STARTED_EVENT,
        EVIDENCE_EXTRACTED_EVENT,
        NEED_REFINED_EVENT,
    ):
        counts.setdefault(event_type, 0)
    return counts


def _search_behavior(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    counts = _event_counts(connection, run_id)
    enrich_rows = connection.execute(
        """
        SELECT run_seq, refs, metrics
        FROM agent_events
        WHERE run_id = %s AND event_type = %s
        ORDER BY run_seq
        """,
        (run_id, ENRICHED_EVENT),
    ).fetchall()
    provider_queries: list[str] = []
    query_rows = connection.execute(
        """
        SELECT refs ->> 'query'
        FROM agent_events
        WHERE run_id = %s AND event_type = %s
        """,
        (run_id, SEARCH_STARTED_EVENT),
    ).fetchall()
    for (query,) in query_rows:
        if isinstance(query, str) and query:
            provider_queries.append(query)

    char_deltas: list[int] = []
    word_deltas: list[int] = []
    enriched_texts: list[str] = []
    for _run_seq, refs, _metrics in enrich_rows:
        if not isinstance(refs, dict):
            continue
        original = str(refs.get("original_query") or "")
        enriched = str(refs.get("enriched_query") or "")
        if not original or not enriched or original == enriched:
            continue
        char_deltas.append(len(enriched) - len(original))
        word_deltas.append(len(enriched.split()) - len(original.split()))
        enriched_texts.append(enriched)

    provider_set = set(provider_queries)
    # A changed provider query is *proven* when an enriched text is actually the
    # text handed to the provider (search.query.started records the post-
    # enrichment query because record_search_query_started runs after enrichment).
    proven_hits = sum(1 for text in enriched_texts if text in provider_set)

    return {
        "search_query_started_count": counts[SEARCH_STARTED_EVENT],
        "context_recorded_count": counts[RECORDED_EVENT],
        "need_refined_count": counts[NEED_REFINED_EVENT],
        "enriched_count": counts[ENRICHED_EVENT],
        "evidence_extracted_count": counts[EVIDENCE_EXTRACTED_EVENT],
        "changed_enrichments": len(char_deltas),
        "average_char_length_change": (
            round(statistics.fmean(char_deltas), 2) if char_deltas else 0.0
        ),
        "average_word_length_change": (
            round(statistics.fmean(word_deltas), 2) if word_deltas else 0.0
        ),
        "provider_query_change_proven": proven_hits,
        "provider_query_change_confirmed": proven_hits > 0,
    }


def _context_effect(connection: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    gap_status_rows = connection.execute(
        """
        SELECT DISTINCT ON (question_id) question_id, status
        FROM research_gaps
        WHERE run_id = %s
        ORDER BY question_id, plan_version DESC
        """,
        (run_id,),
    ).fetchall()
    latest_status = {str(question): str(status) for question, status in gap_status_rows}

    failure_reasons: dict[str, str] = {}
    for (refs,) in connection.execute(
        """
        SELECT refs FROM agent_events
        WHERE run_id = %s AND event_type = %s ORDER BY run_seq
        """,
        (run_id, FEEDBACK_EVENT),
    ).fetchall():
        if isinstance(refs, dict):
            question_id = str(refs.get("question_id") or "")
            reason = str(refs.get("failure_reason") or "")
            if question_id and reason:
                failure_reasons[question_id] = reason

    effects: list[dict[str, Any]] = []
    for _run_seq, refs in connection.execute(
        """
        SELECT run_seq, refs FROM agent_events
        WHERE run_id = %s AND event_type = %s ORDER BY run_seq
        """,
        (run_id, ENRICHED_EVENT),
    ).fetchall():
        if not isinstance(refs, dict):
            continue
        question_id = str(refs.get("question_id") or "")
        original = str(refs.get("original_query") or "")
        enriched = str(refs.get("enriched_query") or "")
        added = enriched[len(original):].strip() if enriched.startswith(original) else ""
        effects.append(
            {
                "question_id": question_id,
                "failure_reason": failure_reasons.get(question_id, ""),
                "applied_hints_text": added,
                "original_query": original,
                "enriched_query": enriched,
                "gap_status_after": latest_status.get(question_id, "unknown"),
            }
        )
    return effects


def analyze_run(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    return {
        "profile": _profile(connection, run_id),
        "gap": _gap_metrics(connection, run_id),
        "failure_bucket": _failure_buckets(connection, run_id),
        "search": _search_behavior(connection, run_id),
        "context_effect": _context_effect(connection, run_id),
    }


def _metric_values(runs: list[dict[str, Any]], path: str, leaf: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        node = run[path]
        value = node[leaf]
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(runs: list[dict[str, Any]], path: str, leaf: str) -> float:
    values = _metric_values(runs, path, leaf)
    return round(statistics.fmean(values), 4) if values else 0.0


def _sd(runs: list[dict[str, Any]], path: str, leaf: str) -> float:
    """Sample standard deviation (ddof=1) as the error bar; 0.0 when n < 2."""

    values = _metric_values(runs, path, leaf)
    return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0


def _arm_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"run_count": 0}
    return {
        "run_count": len(runs),
        "mean_coverage": _mean(runs, "profile", "coverage"),
        "sd_coverage": _sd(runs, "profile", "coverage"),
        "mean_priority_one_coverage": _mean(runs, "profile", "priority_one_coverage"),
        "sd_priority_one_coverage": _sd(runs, "profile", "priority_one_coverage"),
        "mean_cross_validation": _mean(runs, "profile", "cross_validation"),
        "sd_cross_validation": _sd(runs, "profile", "cross_validation"),
        "mean_critical_gaps": _mean(runs, "profile", "critical_gaps"),
        "sd_critical_gaps": _sd(runs, "profile", "critical_gaps"),
        "mean_accepted_evidence": _mean(runs, "profile", "accepted_evidence"),
        "sd_accepted_evidence": _sd(runs, "profile", "accepted_evidence"),
        "mean_total_gap": _mean(runs, "gap", "total_gap"),
        "sd_total_gap": _sd(runs, "gap", "total_gap"),
        "mean_closed_gap": _mean(runs, "gap", "closed_gap"),
        "sd_closed_gap": _sd(runs, "gap", "closed_gap"),
        "mean_open_gap": _mean(runs, "gap", "open_gap"),
        "sd_open_gap": _sd(runs, "gap", "open_gap"),
        "mean_closure_rate": _mean(runs, "gap", "closure_rate"),
        "sd_closure_rate": _sd(runs, "gap", "closure_rate"),
        "mean_search_query_started": _mean(runs, "search", "search_query_started_count"),
        "sd_search_query_started": _sd(runs, "search", "search_query_started_count"),
        "mean_enriched_count": _mean(runs, "search", "enriched_count"),
        "sd_enriched_count": _sd(runs, "search", "enriched_count"),
        "mean_context_recorded_count": _mean(runs, "search", "context_recorded_count"),
        "mean_char_length_change": _mean(runs, "search", "average_char_length_change"),
        "provider_query_change_confirmed": any(
            run["search"]["provider_query_change_confirmed"] for run in runs
        ),
    }


def build_report(
    connection: psycopg.Connection,
    baseline_ids: list[str],
    candidate_ids: list[str],
) -> dict[str, Any]:
    baseline_runs = [analyze_run(connection, run_id) for run_id in baseline_ids]
    candidate_runs = [analyze_run(connection, run_id) for run_id in candidate_ids]
    return {
        "experiment": "phase14.3_evidence_aware_ab",
        "note": (
            "Small-n A/B: means carry sample-SD error bars (ddof=1). Treat a "
            "delta as directional only when it exceeds the pooled within-arm SD; "
            "this cannot establish statistical significance."
        ),
        "baseline": {
            "run_ids": baseline_ids,
            "runs": baseline_runs,
            "summary": _arm_summary(baseline_runs),
        },
        "candidate": {
            "run_ids": candidate_ids,
            "runs": candidate_runs,
            "summary": _arm_summary(candidate_runs),
        },
        "delta": {
            "closure_rate": round(
                _mean(candidate_runs, "gap", "closure_rate")
                - _mean(baseline_runs, "gap", "closure_rate"),
                4,
            ),
            "mean_open_gap": round(
                _mean(candidate_runs, "gap", "open_gap")
                - _mean(baseline_runs, "gap", "open_gap"),
                4,
            ),
            "mean_coverage": round(
                _mean(candidate_runs, "profile", "coverage")
                - _mean(baseline_runs, "profile", "coverage"),
                4,
            ),
            "mean_accepted_evidence": round(
                _mean(candidate_runs, "profile", "accepted_evidence")
                - _mean(baseline_runs, "profile", "accepted_evidence"),
                4,
            ),
        },
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    base = report["baseline"]["summary"]
    cand = report["candidate"]["summary"]
    delta = report["delta"]

    def cell(summary: dict[str, Any], key: str) -> str:
        mean_text = _fmt(summary.get(key))
        if summary.get("run_count", 0) < 2 or not key.startswith("mean_"):
            return mean_text
        sd = summary.get("sd_" + key[len("mean_") :])
        return f"{mean_text} ± {_fmt(sd)}" if sd is not None else mean_text

    def row(label: str, key: str, delta_key: str | None = None) -> str:
        d = _fmt(delta[delta_key]) if delta_key else "-"
        return f"| {label} | {cell(base, key)} | {cell(cand, key)} | {d} |"

    def count_row(label: str, key: str) -> str:
        return f"| {label} | {cell(base, key)} | {cell(cand, key)} | - |"

    def noise_note(metric: str, delta_key: str) -> str:
        sd_base = base.get(f"sd_{metric}", 0.0) or 0.0
        sd_cand = cand.get(f"sd_{metric}", 0.0) or 0.0
        pooled = (sd_base**2 + sd_cand**2) ** 0.5
        d = delta.get(delta_key, 0.0)
        verdict = "directional" if abs(d) > pooled else "within-noise"
        return (
            f"- `{delta_key}` delta={_fmt(d)} vs pooled within-arm SD={_fmt(round(pooled, 4))} "
            f"-> {verdict}"
        )

    confirmed = "provider_query_change_confirmed"
    lines = [
        "# Phase 14.3 Evidence-Aware A/B Report",
        "",
        f"- baseline runs: {', '.join(report['baseline']['run_ids']) or '(none)'}",
        f"- candidate runs: {', '.join(report['candidate']['run_ids']) or '(none)'}",
        f"- {report['note']}",
        "",
        "## C. Metric Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        count_row("run count", "run_count"),
        count_row("total gap (mean)", "mean_total_gap"),
        count_row("closed gap (mean)", "mean_closed_gap"),
        row("open gap (mean)", "mean_open_gap", "mean_open_gap"),
        row("closure_rate (mean)", "mean_closure_rate", "closure_rate"),
        row("coverage (mean)", "mean_coverage", "mean_coverage"),
        row("accepted evidence (mean)", "mean_accepted_evidence", "mean_accepted_evidence"),
        count_row("search.query.started (mean)", "mean_search_query_started"),
        count_row("query.context.enriched (mean)", "mean_enriched_count"),
        count_row("context.recorded (mean)", "mean_context_recorded_count"),
        count_row("avg query char delta", "mean_char_length_change"),
        (
            f"| provider query changed (proven) | "
            f"{'yes' if base.get(confirmed) else 'no'} | "
            f"{'yes' if cand.get(confirmed) else 'no'} | - |"
        ),
        "",
        "Signal vs noise (small-n caveat):",
        "",
        noise_note("coverage", "mean_coverage"),
        noise_note("closure_rate", "closure_rate"),
        noise_note("accepted_evidence", "mean_accepted_evidence"),
        noise_note("open_gap", "mean_open_gap"),
        "",
        "## D. Query Enrichment Analysis",
        "",
    ]
    candidate_runs = report["candidate"]["runs"]
    shown = 0
    for run in candidate_runs:
        for effect in run["context_effect"]:
            if shown >= 12:
                break
            lines.append(
                f"- `{effect['question_id']}` [{effect['failure_reason'] or 'n/a'}] "
                f"hints+`{effect['applied_hints_text'] or '(none)'}` -> "
                f"gap `{effect['gap_status_after']}`"
            )
            lines.append(f"    - original: `{effect['original_query']}`")
            lines.append(f"    - enriched: `{effect['enriched_query']}`")
            shown += 1
    if shown == 0:
        lines.append("- (no enrichment events observed in the candidate arm)")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-id", default=os.getenv("PHASE14_3_BASELINE_RUN_ID"))
    parser.add_argument("--candidate-run-id", default=os.getenv("PHASE14_3_CANDIDATE_RUN_ID"))
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
