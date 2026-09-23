"""Phase 13.0 read-only Gap Closure Bottleneck analyzer.

Analyzes *why* coverage / priority-one coverage did not stabilize, by classifying
the gaps the closure loop could not fully close for a research run.

This tool is strictly additive: it only issues ``SELECT`` statements against the
persisted event/gap projections and never imports or mutates runtime behavior.
The classifier is a pure function so it can be unit-tested without a database.

Classification buckets (mutually exclusive, deterministic priority order):
  MISSING_INDEPENDENT_SOURCE / CLAIM_NOT_VERIFIED / INSUFFICIENT_EVIDENCE /
  DIMENSION_MISMATCH / EVIDENCE_QUALITY_LOW
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

MISSING_INDEPENDENT_SOURCE = "MISSING_INDEPENDENT_SOURCE"
CLAIM_NOT_VERIFIED = "CLAIM_NOT_VERIFIED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
EVIDENCE_QUALITY_LOW = "EVIDENCE_QUALITY_LOW"

BUCKET_ORDER = (
    MISSING_INDEPENDENT_SOURCE,
    CLAIM_NOT_VERIFIED,
    INSUFFICIENT_EVIDENCE,
    DIMENSION_MISMATCH,
    EVIDENCE_QUALITY_LOW,
)

# Priority tiers follow the benchmark fixture: runtime priority 1 == P1 (top),
# 2 == P2, 3 == P3.
PRIORITY_TIER = {1: "P1", 2: "P2", 3: "P3"}


@dataclass(frozen=True, slots=True)
class GapSignals:
    """Persisted, read-only signals describing one unclosed dimension gap."""

    question_id: str
    dimension_key: str
    priority: int
    requirement_type: str
    failure_reason: str
    closure_result: str
    required_evidence_count: int
    current_evidence_count: int
    required_independent_sources: int
    current_independent_sources: int
    current_coverage: float
    required_coverage: float
    alignment_status: str


def classify_gap(signals: GapSignals) -> str:
    """Map one unclosed gap onto exactly one bottleneck bucket.

    The evaluator's own diagnosis (``failure_reason``) is authoritative and is
    checked first; ``requirement_type`` and the evidence-alignment verdict are
    used only to disambiguate coarse ``insufficient_evidence`` / ``unknown``
    feedback so every gap lands in exactly one bucket.
    """

    reason = signals.failure_reason
    req_type = signals.requirement_type

    if reason == "missing_independent_source" or req_type == "independent_source":
        return MISSING_INDEPENDENT_SOURCE
    if reason == "no_valid_path":
        return DIMENSION_MISMATCH
    if reason == "claim_not_verified":
        return CLAIM_NOT_VERIFIED
    if reason == "insufficient_evidence":
        if req_type == "evidence_quality":
            enough = (
                signals.required_evidence_count == 0
                or signals.current_evidence_count >= signals.required_evidence_count
            )
            return EVIDENCE_QUALITY_LOW if enough else INSUFFICIENT_EVIDENCE
        return INSUFFICIENT_EVIDENCE
    # reason is "unknown"/absent: infer from alignment, then requirement_type, then counts.
    if signals.alignment_status == "not_aligned":
        return DIMENSION_MISMATCH
    if req_type == "claim_verification":
        return CLAIM_NOT_VERIFIED
    if req_type == "evidence_quality":
        return EVIDENCE_QUALITY_LOW
    if signals.required_evidence_count > signals.current_evidence_count:
        return INSUFFICIENT_EVIDENCE
    if signals.current_coverage < signals.required_coverage:
        return EVIDENCE_QUALITY_LOW
    return INSUFFICIENT_EVIDENCE


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _fetch_rows(connection: psycopg.Connection, query: str, run_id: str) -> list[tuple[Any, ...]]:
    return connection.execute(query, (run_id,)).fetchall()


def _fetch_priorities(connection: psycopg.Connection, run_id: str) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT question_id, priority
        FROM research_plan_items
        WHERE run_id = %s
          AND plan_version = (
            SELECT max(plan_version) FROM research_plan_items WHERE run_id = %s
          )
        """,
        (run_id, run_id),
    ).fetchall()
    return {str(question): int(priority) for question, priority in rows}


def _latest_feedback(
    connection: psycopg.Connection, run_id: str
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _fetch_rows(
        connection,
        """
        SELECT run_seq, refs
        FROM agent_events
        WHERE run_id = %s AND event_type = 'closure.feedback.generated'
        ORDER BY run_seq
        """,
        run_id,
    )
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for run_seq, refs in rows:
        if not isinstance(refs, dict):
            continue
        question_id = str(refs.get("question_id") or "")
        dimension_key = str(refs.get("dimension_key") or "")
        if not question_id or not dimension_key:
            continue
        latest[(question_id, dimension_key)] = {
            "run_seq": int(run_seq),
            "gap_id": str(refs.get("gap_id") or ""),
            "requirement_type": str(refs.get("requirement_type") or ""),
            "failure_reason": str(refs.get("failure_reason") or ""),
            "closure_result": str(refs.get("closure_result") or refs.get("current_status") or ""),
        }
    return latest


def _latest_requirements(
    connection: psycopg.Connection, run_id: str
) -> dict[tuple[str, str], dict[str, float]]:
    rows = _fetch_rows(
        connection,
        """
        SELECT question_id, dimension_key, plan_version,
               required_evidence_count, current_evidence_count,
               required_independent_sources, current_independent_sources,
               current_coverage, required_coverage
        FROM gap_requirements
        WHERE run_id = %s
        ORDER BY plan_version
        """,
        run_id,
    )
    latest: dict[tuple[str, str], dict[str, float]] = {}
    for (
        question_id,
        dimension_key,
        _plan_version,
        req_ev,
        cur_ev,
        req_src,
        cur_src,
        cur_cov,
        req_cov,
    ) in rows:
        latest[(str(question_id), str(dimension_key))] = {
            "required_evidence_count": int(req_ev or 0),
            "current_evidence_count": int(cur_ev or 0),
            "required_independent_sources": int(req_src or 0),
            "current_independent_sources": int(cur_src or 0),
            "current_coverage": float(cur_cov or 0.0),
            "required_coverage": float(req_cov or 0.0),
        }
    return latest


def _latest_alignment(connection: psycopg.Connection, run_id: str) -> dict[str, str]:
    rows = _fetch_rows(
        connection,
        """
        SELECT run_seq, refs
        FROM agent_events
        WHERE run_id = %s AND event_type = 'evidence.alignment.completed'
        ORDER BY run_seq
        """,
        run_id,
    )
    result: dict[str, str] = {}
    for _run_seq, refs in rows:
        if not isinstance(refs, dict):
            continue
        gap_id = str(refs.get("gap_id") or "")
        status = str(refs.get("result") or refs.get("reason") or "")
        if gap_id and status:
            result[gap_id] = status  # ascending run_seq keeps the last write dominant
    return result


def collect_gaps(connection: psycopg.Connection, run_id: str) -> list[GapSignals]:
    feedback = _latest_feedback(connection, run_id)
    requirements = _latest_requirements(connection, run_id)
    alignment = _latest_alignment(connection, run_id)
    priorities = _fetch_priorities(connection, run_id)

    gaps: list[GapSignals] = []
    for (question_id, dimension_key), fb in sorted(feedback.items()):
        req = requirements.get((question_id, dimension_key), {})
        alignment_status = alignment.get(fb["gap_id"], "none")
        gaps.append(
            GapSignals(
                question_id=question_id,
                dimension_key=dimension_key,
                priority=priorities.get(question_id, 0),
                requirement_type=fb["requirement_type"],
                failure_reason=fb["failure_reason"],
                closure_result=fb["closure_result"],
                required_evidence_count=int(req.get("required_evidence_count", 0)),
                current_evidence_count=int(req.get("current_evidence_count", 0)),
                required_independent_sources=int(req.get("required_independent_sources", 0)),
                current_independent_sources=int(req.get("current_independent_sources", 0)),
                current_coverage=float(req.get("current_coverage", 0.0)),
                required_coverage=float(req.get("required_coverage", 0.0)),
                alignment_status=alignment_status,
            )
        )
    return gaps


def _bucket_profile(gaps: list[GapSignals]) -> dict[str, Any]:
    buckets = Counter(classify_gap(gap) for gap in gaps)
    total = sum(buckets.values()) or 1
    per_bucket: dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        count = buckets.get(bucket, 0)
        per_bucket[bucket] = {
            "count": count,
            "share": round(count / total, 4),
        }
    return {"total": len(gaps), "distribution": per_bucket}


def _gap_rows(gaps: list[GapSignals]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": gap.question_id,
            "dimension_key": gap.dimension_key,
            "priority": gap.priority,
            "tier": PRIORITY_TIER.get(gap.priority, "unknown"),
            "requirement_type": gap.requirement_type,
            "failure_reason": gap.failure_reason,
            "closure_result": gap.closure_result,
            "alignment_status": gap.alignment_status,
            "required_evidence_count": gap.required_evidence_count,
            "current_evidence_count": gap.current_evidence_count,
            "required_independent_sources": gap.required_independent_sources,
            "current_independent_sources": gap.current_independent_sources,
            "current_coverage": gap.current_coverage,
            "required_coverage": gap.required_coverage,
            "bucket": classify_gap(gap),
        }
        for gap in gaps
    ]


def analyze_run(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    gaps = collect_gaps(connection, run_id)
    p1_gaps = [gap for gap in gaps if gap.priority == 1]
    top = max(
        (_bucket_profile(gaps)["distribution"].items()),
        key=lambda item: item[1]["count"] if isinstance(item[1], dict) else 0,
        default=(None, {"count": 0, "share": 0.0}),
    )
    return {
        "run_id": run_id,
        "profile": _run_profile(connection, run_id),
        "unclosed_dimension_gaps": _gap_rows(gaps),
        "summary": _bucket_profile(gaps),
        "top_failure_pattern": top[0],
        "priority_one": {
            "count": len(p1_gaps),
            "distribution": _bucket_profile(p1_gaps)["distribution"] if p1_gaps else {},
            "gaps": _gap_rows(p1_gaps),
        },
    }


def _run_profile(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT status, termination_reason,
               (quality_snapshot->>'coverage')::text,
               (quality_snapshot->>'priority_one_coverage')::text,
               (quality_snapshot->>'priority_one_average_coverage')::text,
               (quality_snapshot->>'cross_validation')::text,
               (quality_snapshot->>'critical_gaps')::text,
               (quality_snapshot->>'accepted_evidence')::text
        FROM research_runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return {"run_id": run_id, "found": False}
    keys = (
        "status",
        "termination_reason",
        "coverage",
        "priority_one_coverage",
        "priority_one_average_coverage",
        "cross_validation",
        "critical_gaps",
        "accepted_evidence",
    )
    profile = dict(zip(keys, [None if value is None else value for value in row], strict=True))
    profile["run_id"] = run_id
    profile["found"] = True
    open_claim_gaps = connection.execute(
        "SELECT count(*) FROM research_gaps WHERE run_id = %s AND status = 'open'", (run_id,)
    ).fetchone()
    profile["open_claim_gaps"] = int(open_claim_gaps[0]) if open_claim_gaps else 0
    return profile


def build_report(
    connection: psycopg.Connection, run_id: str, baseline_run_id: str | None
) -> dict[str, Any]:
    report: dict[str, Any] = {"candidate": analyze_run(connection, run_id)}
    if baseline_run_id is not None:
        report["baseline"] = analyze_run(connection, baseline_run_id)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    cand = candidate["profile"]
    lines = [
        "# Phase 13.0 Gap Closure Bottleneck Analysis",
        "",
        f"- candidate run: `{candidate['run_id']}`",
        f"- status: {cand.get('status')} / {cand.get('termination_reason')}",
        (
            f"- coverage: {cand.get('coverage')}  "
            f"priority_one_coverage: {cand.get('priority_one_coverage')}  "
            f"priority_one_average_coverage: {cand.get('priority_one_average_coverage')}"
        ),
        f"- unclosed dimension gaps: {candidate['summary']['total']}  "
        f"open claim gaps: {cand.get('open_claim_gaps')}",
        f"- Top failure pattern: **{candidate['top_failure_pattern']}**",
        "",
        "## Bucket distribution (candidate)",
        "",
        "| Bucket | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for bucket, stat in candidate["summary"]["distribution"].items():
        lines.append(f"| {bucket} | {stat['count']} | {stat['share'] * 100:.1f}% |")
    p1 = candidate["priority_one"]
    lines += ["", f"## Priority-one (P1) gaps — {p1['count']} total", ""]
    if p1["distribution"]:
        lines += ["| Bucket | Count | Share |", "| --- | ---: | ---: |"]
        for bucket, stat in p1["distribution"].items():
            lines.append(f"| {bucket} | {stat['count']} | {stat['share'] * 100:.1f}% |")
    baseline = report.get("baseline")
    if baseline is not None:
        base = baseline["profile"]
        lines += ["", "## Candidate vs baseline", ""]
        lines += [
            "| Metric | Baseline | Candidate |",
            "| --- | --- | --- |",
            f"| run | `{baseline['run_id'][:8]}` | `{candidate['run_id'][:8]}` |",
            f"| coverage | {base.get('coverage')} | {cand.get('coverage')} |",
            (
                f"| priority_one_coverage | {base.get('priority_one_coverage')} | "
                f"{cand.get('priority_one_coverage')} |"
            ),
            (
                f"| unclosed dimension gaps | {baseline['summary']['total']} | "
                f"{candidate['summary']['total']} |"
            ),
            (
                f"| top failure pattern | {baseline['top_failure_pattern']} | "
                f"{candidate['top_failure_pattern']} |"
            ),
        ]
        lines += [
            "",
            "| Bucket | Baseline count | Candidate count |",
            "| --- | ---: | ---: |",
        ]
        for bucket in BUCKET_ORDER:
            b_count = baseline["summary"]["distribution"].get(bucket, {}).get("count", 0)
            c_count = candidate["summary"]["distribution"].get(bucket, {}).get("count", 0)
            lines.append(f"| {bucket} | {b_count} | {c_count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")

    with psycopg.connect(_database_uri(args.database_url)) as connection:
        report = build_report(connection, args.run_id, args.baseline_run_id)

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
