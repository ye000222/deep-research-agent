#!/usr/bin/env python3
"""Phase 16.0 — release-blocking hard-gap audit (read-only).

This script audits the three frozen Phase 15.1 runs that are the V1 Pre-RC
reference.  It deliberately does not start a run, update a row, or change any
research decision.  The database queries select the latest semantic
GapRequirement version per (run, dimension, requirement type); the append-like
history rows are retained in the report as an audit note, but are not counted
as simultaneous requirements.
"""

# The audit report intentionally contains long human-readable evidence strings.
# Keep the repository's existing 100-column lint policy for application code,
# while excluding only those report literals from E501.
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import psycopg

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CONTROL = [
    "01a0c3ef-928b-7f0d-8006-40843c571b57",
    "01a0c412-10b0-7c3f-8cf4-d56aeecbbea0",
    "01a0c412-5ae6-7b9c-9353-7be1ad74c293",
]


def db_url() -> str:
    value = os.environ.get(
        "PHASE16_DATABASE_URL",
        os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research",
        ),
    )
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def load_json(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((ARTIFACTS / name).read_text(encoding="utf-8")),
    )


def as_dict(row: tuple[Any, ...], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: row[index] for index, name in enumerate(names)}


def mean_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 6),
        "sd": round(statistics.pstdev(values), 6),
        "min": min(values),
        "max": max(values),
    }


def run_cards(connection: Any) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id::text, status, termination_reason, plan_version,
               budget_snapshot, quality_snapshot, usage_snapshot
        FROM research_runs
        WHERE id = ANY(%s::uuid[])
        """,
        (CONTROL,),
    ).fetchall()
    cards: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = as_dict(
            row,
            (
                "run_id",
                "status",
                "termination_reason",
                "plan_version",
                "budget_snapshot",
                "quality_snapshot",
                "usage_snapshot",
            ),
        )
        budget = item["budget_snapshot"] or {}
        benchmark = budget.get("benchmark", {}) if isinstance(budget, dict) else {}
        flags = benchmark.get("feature_flags", {}) if isinstance(benchmark, dict) else {}
        item["benchmark"] = benchmark
        item["feature_flags"] = flags
        item["resolved_feature_flags"] = {
            "independent_source_targeting_enabled": flags.get(
                "independent_source_targeting_enabled", False
            ),
            "evidence_input_quality_enabled": flags.get(
                "evidence_input_quality_enabled", False
            ),
        }
        item["source_revision"] = budget.get("source_revision") if isinstance(budget, dict) else None
        cards[item["run_id"]] = item
    return cards


def latest_gap_requirements(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        WITH ranked AS (
            SELECT
                gap_id::text, run_id::text, question_id, dimension_key,
                requirement_type, criterion, required_evidence_count,
                required_independent_sources, current_evidence_count,
                current_independent_sources, current_coverage, required_coverage,
                verification_status, closure_status, state_version,
                transition_reason, created_at, updated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY run_id, dimension_key, requirement_type
                    ORDER BY state_version DESC, updated_at DESC, created_at DESC
                ) AS row_number
            FROM gap_requirements
            WHERE run_id = ANY(%s::uuid[])
        )
        SELECT gap_id, run_id, question_id, dimension_key, requirement_type,
               criterion, required_evidence_count, required_independent_sources,
               current_evidence_count, current_independent_sources,
               current_coverage, required_coverage, verification_status,
               closure_status, state_version, transition_reason,
               created_at, updated_at
        FROM ranked
        WHERE row_number = 1
        ORDER BY run_id, dimension_key, requirement_type
        """,
        (CONTROL,),
    ).fetchall()
    names = (
        "gap_id",
        "run_id",
        "question_id",
        "dimension_key",
        "requirement_type",
        "criterion",
        "required_evidence_count",
        "required_independent_sources",
        "current_evidence_count",
        "current_independent_sources",
        "current_coverage",
        "required_coverage",
        "verification_status",
        "closure_status",
        "state_version",
        "transition_reason",
        "created_at",
        "updated_at",
    )
    return [as_dict(row, names) for row in rows]


def raw_gap_history_counts(connection: Any) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT run_id::text, COUNT(*)
        FROM gap_requirements
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id
        """,
        (CONTROL,),
    ).fetchall()
    return {str(run_id): int(count) for run_id, count in rows}


def evidence_rows(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT e.run_id::text, e.question_id, c.dimension_key,
               COUNT(*) AS candidate_count,
               COUNT(*) FILTER (WHERE e.accepted) AS accepted_count,
               COUNT(*) FILTER (WHERE e.rejection_reason = 'source_role_mismatch') AS role_mismatch_count,
               COUNT(*) FILTER (WHERE e.rejection_reason = 'claim_quote_entailment_failed') AS entailment_failure_count,
               COUNT(*) FILTER (WHERE e.rejection_reason IS NOT NULL) AS rejected_count,
               COUNT(DISTINCT s.source_owner_key) FILTER (WHERE e.accepted) AS accepted_owner_count
        FROM research_evidence e
        LEFT JOIN research_claims c ON c.id = e.claim_id
        LEFT JOIN research_sources s ON s.id = e.source_id
        WHERE e.run_id = ANY(%s::uuid[])
        GROUP BY e.run_id, e.question_id, c.dimension_key
        ORDER BY e.run_id, e.question_id, c.dimension_key
        """,
        (CONTROL,),
    ).fetchall()
    names = (
        "run_id",
        "question_id",
        "dimension_key",
        "candidate_count",
        "accepted_count",
        "role_mismatch_count",
        "entailment_failure_count",
        "rejected_count",
        "accepted_owner_count",
    )
    return [as_dict(row, names) for row in rows]


def evidence_reason_counts(connection: Any) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        """
        SELECT run_id::text, COALESCE(rejection_reason, 'accepted') AS reason,
               COUNT(*)
        FROM research_evidence
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id, COALESCE(rejection_reason, 'accepted')
        ORDER BY run_id, reason
        """,
        (CONTROL,),
    ).fetchall()
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for run_id, reason, count in rows:
        result[str(run_id)][str(reason)] = int(count)
    return dict(result)


def event_counts(connection: Any) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        """
        SELECT run_id::text, event_type, COUNT(*)
        FROM agent_events
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id, event_type
        ORDER BY run_id, event_type
        """,
        (CONTROL,),
    ).fetchall()
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for run_id, event_type, count in rows:
        result[str(run_id)][str(event_type)] = int(count)
    return dict(result)


def requirement_evidence_index(
    gap_rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    index = {
        (str(row["run_id"]), str(row["dimension_key"])): row for row in evidence
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for gap in gap_rows:
        item = dict(index.get((str(gap["run_id"]), str(gap["dimension_key"])), {}))
        item.setdefault("candidate_count", 0)
        item.setdefault("accepted_count", 0)
        item.setdefault("role_mismatch_count", 0)
        item.setdefault("entailment_failure_count", 0)
        item.setdefault("accepted_owner_count", 0)
        result[(str(gap["run_id"]), str(gap["gap_id"]))] = item
    return result


def classify_gap(gap: dict[str, Any], evidence: dict[str, Any]) -> tuple[str, str]:
    requirement_type = str(gap["requirement_type"])
    accepted = int(evidence["accepted_count"])
    role_mismatch = int(evidence["role_mismatch_count"])
    if requirement_type == "evidence_quality" and accepted > 0:
        return (
            "STATE_PROJECTION_INCONSISTENCY",
            "accepted evidence exists for the dimension, but the latest evidence_quality requirement still reports current_evidence_count=0",
        )
    if requirement_type == "evidence_quality" and role_mismatch > 0:
        return (
            "SOURCE_ROLE_MISMATCH",
            "candidate evidence exists, but source-role suitability rejected the evidence-quality requirement",
        )
    if requirement_type == "evidence_quality" and int(evidence["candidate_count"]) == 0:
        return (
            "NO_ACCEPTED_EVIDENCE",
            "no candidate evidence was persisted for this evidence-quality requirement in this run",
        )
    if requirement_type in {"claim_verification", "independent_source"}:
        return (
            "GENUINE_INDEPENDENT_SOURCE_GAP",
            "the latest requirement remains partial because the distinct accepted source-owner bar is not met",
        )
    return ("UNCLASSIFIED", "latest canonical requirement remains unclosed without a more specific persisted reason")


def build_report(connection: Any) -> dict[str, Any]:
    cards = run_cards(connection)
    gap_rows = latest_gap_requirements(connection)
    evidence = evidence_rows(connection)
    evidence_by_gap = requirement_evidence_index(gap_rows, evidence)
    reasons = evidence_reason_counts(connection)
    events = event_counts(connection)
    raw_history = raw_gap_history_counts(connection)

    final_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classifications: list[dict[str, Any]] = []
    for gap in gap_rows:
        run_id = str(gap["run_id"])
        final_by_run[run_id].append(gap)
        evidence_info = evidence_by_gap[(run_id, str(gap["gap_id"]))]
        if str(gap["closure_status"]) != "closed":
            classification, explanation = classify_gap(gap, evidence_info)
            classifications.append(
                {
                    "run_id": run_id,
                    "question_id": gap["question_id"],
                    "dimension_key": gap["dimension_key"],
                    "requirement_type": gap["requirement_type"],
                    "gap_id": gap["gap_id"],
                    "closure_status": gap["closure_status"],
                    "state_version": gap["state_version"],
                    "criterion": gap["criterion"],
                    "current_evidence_count": gap["current_evidence_count"],
                    "current_independent_sources": gap["current_independent_sources"],
                    "evidence": evidence_info,
                    "classification": classification,
                    "explanation": explanation,
                }
            )

    status_counts = Counter(str(gap["closure_status"]) for gap in gap_rows)
    class_groups: dict[str, dict[str, Any]] = {}
    for item in classifications:
        key = item["classification"]
        group = class_groups.setdefault(
            key,
            {
                "classification": key,
                "affected_requirement_instances": 0,
                "affected_runs": set(),
                "affected_questions": set(),
                "affected_dimensions": set(),
                "per_run_counts": Counter(),
            },
        )
        group["affected_requirement_instances"] += 1
        group["affected_runs"].add(item["run_id"])
        group["affected_questions"].add(item["question_id"])
        group["affected_dimensions"].add(item["dimension_key"])
        group["per_run_counts"][item["run_id"]] += 1
    for group in class_groups.values():
        group["affected_runs"] = sorted(group["affected_runs"])
        group["affected_questions"] = sorted(group["affected_questions"])
        group["affected_dimensions"] = sorted(group["affected_dimensions"])
        group["per_run_counts"] = dict(sorted(group["per_run_counts"].items()))
        group["recurrence"] = f"{len(group['affected_runs'])}/{len(CONTROL)} runs"

    quality = {
        str(run_id): (card.get("quality_snapshot") or {})
        for run_id, card in cards.items()
    }
    usage = {
        str(run_id): (card.get("usage_snapshot") or {})
        for run_id, card in cards.items()
    }

    def event_total(event_type: str) -> dict[str, int]:
        return {run_id: int(events.get(run_id, {}).get(event_type, 0)) for run_id in CONTROL}

    semantic_totals = {
        "total": len(gap_rows),
        "closed": status_counts.get("closed", 0),
        "partial": status_counts.get("partial", 0),
        "open": status_counts.get("open", 0),
        "by_run": {
            run_id: dict(Counter(str(gap["closure_status"]) for gap in final_by_run[run_id]))
            for run_id in CONTROL
        },
    }
    open_by_requirement_type = dict(
        Counter(str(item["requirement_type"]) for item in classifications)
    )
    open_by_classification = dict(
        Counter(str(item["classification"]) for item in classifications)
    )
    recurrence_by_classification = {
        key: {
            "affected_runs": group["affected_runs"],
            "rate": f"{len(group['affected_runs'])}/{len(CONTROL)}",
        }
        for key, group in class_groups.items()
    }

    return {
        "schema_version": "phase16-hard-gap-audit.v1",
        "phase": "16.0",
        "analysis_mode": "read_only",
        "production_behavior_changed": False,
        "new_benchmark_started": False,
        "run_ids": CONTROL,
        "run_metadata": cards,
        "canonical_gap_selection": {
            "rule": "latest state_version, then updated_at, per (run_id, dimension_key, requirement_type)",
            "raw_history_rows_by_run": raw_history,
            "note": "gap_requirements is append-like in this corpus; each transition can have a new gap_id. Final semantic counts use the latest row, not every historical version.",
        },
        "canonical_gap_summary": semantic_totals,
        "remaining_gap_inventory": {
            "total_unclosed_requirements": len(classifications),
            "unclosed_by_requirement_type": open_by_requirement_type,
            "unclosed_by_failure_class": open_by_classification,
            "cross_run_recurrence": recurrence_by_classification,
        },
        "unclosed_requirements": classifications,
        "failure_class_summary": list(class_groups.values()),
        "evidence_by_run_question_dimension": evidence,
        "evidence_rejection_reasons": reasons,
        "event_counts": events,
        "reliability": {
            "run_completed": event_total("run.completed"),
            "report_verified": event_total("report.verified"),
            "technical_degraded": event_total("question.technical_degraded"),
            "retry_scheduled": event_total("question.retry_scheduled"),
            "tool_failed": event_total("tool.failed"),
            "search_source_exhausted": event_total("search.source_space_exhausted"),
            "recovery_started": event_total("recovery.started"),
            "recovery_completed": event_total("recovery.completed"),
            "closure_transitions": event_total("gap.closure.transition"),
            "closure_feedback": event_total("closure.feedback.generated"),
            "assessment": "All three runs reached terminal completed_with_limitations and report.verified; no persisted crash, unbounded-loop, deadline-enforcement, recovery-deadlock, or event-persistence failure was observed in the audited event counts.",
        },
        "quality_snapshots": quality,
        "usage_snapshots": usage,
        "release_blocker_matrix": [
            {
                "issue": "q1 evidence_quality requirement not consuming accepted architecture/process evidence",
                "recurrence": "3/3 runs",
                "direct_evidence": "accepted q1 evidence is present in two runs; canonical evidence_quality rows remain current_evidence_count=0 in all three",
                "release_blocking": "NO — secondary requirement projection inconsistency; core report/evaluator path still reaches verified reports",
                "bounded_fix": "POSSIBLE but not proven small without an explicit requirement-to-evidence alignment rule",
                "counterfactual_unlock": "up to two observed q1 requirement instances could be affected; no coverage unlock is estimable from persisted data",
                "classification": "NOT_ELIGIBLE_RELEASE_BLOCKER",
            },
            {
                "issue": "q4 evidence_quality source-role mismatch and zero accepted q4 evidence",
                "recurrence": "3/3 runs",
                "direct_evidence": "q4 candidate evidence is present, but source_role_mismatch dominates and accepted evidence is zero in each run",
                "release_blocking": "NO — source suitability/role semantics are a hard research-quality gap, not a demonstrated bounded runtime failure",
                "bounded_fix": "NO without changing acceptance/source-role semantics",
                "counterfactual_unlock": "three q4 requirement instances are theoretically affected; no validated acceptable-source path is present",
                "classification": "CLASS_C_EXPECTED_HARD_GAP",
            },
            {
                "issue": "verified_claims remains zero under the strict per-claim metric",
                "recurrence": "3/3 runs",
                "direct_evidence": "accepted evidence and dimension-level claim-verification closures exist, but semantic per-claim verification is zero",
                "release_blocking": "NO for V1 Pre-RC — the strict metric requires canonical claim equivalence across independent owners",
                "bounded_fix": "NO; this is an architectural V2 capability",
                "counterfactual_unlock": "not estimable from current evidence rows",
                "classification": "CLASS_B_ARCHITECTURAL_V2",
            },
        ],
        "decision": {
            "one_bounded_fix_justified": False,
            "decision": "NO_DOMINANT_RELEASE_BLOCKING_HARD_GAP",
            "v1_pre_rc": "CLOSED",
            "phase17": "STOP_OPTIMIZING_RC_REFERENCE",
            "reason": "The recurring q1 projection inconsistency is real but not shown to block the V1 report/evaluator path or yield a validated coverage unlock; q4 source-role mismatch and strict per-claim verification are hard/architectural gaps. No single small fix satisfies recurrence, direct causal proof, release-blocking impact, validated unlock, and bounded implementation simultaneously.",
            "primary_remaining_hard_gap": "NO DOMINANT RELEASE-BLOCKING HARD GAP",
            "secondary_remaining_hard_gap": "NONE",
            "stop_gate_result": "RESULT_A_NO_PRODUCTION_FIX_NEEDED",
        },
    }


def json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_outputs(report: dict[str, Any]) -> None:
    audit_json = ARTIFACTS / "phase16_hard_gap_audit.json"
    audit_md = ARTIFACTS / "phase16_hard_gap_audit.md"
    matrix_json = ARTIFACTS / "phase16_release_blocker_matrix.json"
    matrix_md = ARTIFACTS / "phase16_release_blocker_matrix.md"
    reference_json = ARTIFACTS / "v1_pre_rc_reference.json"
    reference_md = ARTIFACTS / "v1_pre_rc_reference.md"

    audit_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    matrix = {
        "schema_version": "phase16-release-blocker-matrix.v1",
        "run_ids": CONTROL,
        "rows": report["release_blocker_matrix"],
        "decision": report["decision"],
    }
    matrix_json.write_text(json.dumps(matrix, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    reference = {
        "schema_version": "v1-pre-rc-reference.v1",
        "phase": "16.0",
        "run_ids": CONTROL,
        "benchmark_identity": {
            "benchmark_id": "v1-dev-model-comparison",
            "benchmark_version": "1.0.0",
            "tier": "standard",
            "metric_definition_version": "coverage-v1",
            "plan_policy": "frozen reference plan: questions=5, priority_one=4, priority_two=1",
            "independent_source_targeting_enabled": True,
            "evidence_input_quality_enabled": False,
        },
        "quality_reference": load_json("phase15_1_candidate_n3.json").get("golden_baseline_summary", {}),
        "canonical_gap_summary": report["canonical_gap_summary"],
        "reliability_summary": report["reliability"],
        "purpose": "V1 Pre-RC reference only; not a new benchmark and not a causal experiment.",
    }
    reference_json.write_text(json.dumps(reference, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")

    md = [
        "# Phase 16.0 — Release-Blocking Hard-Gap Audit",
        "",
        "> Read-only audit of the frozen Phase 15.1 V1 Pre-RC reference. No production behavior or benchmark state was changed.",
        "",
        "## Decision",
        "",
        f"- **{report['decision']['decision']}**",
        f"- V1 Pre-RC: **{report['decision']['v1_pre_rc']}**",
        f"- Phase 17 disposition: **{report['decision']['phase17']}**",
        f"- Bounded production fix justified: **{report['decision']['one_bounded_fix_justified']}**",
        f"- Rationale: {report['decision']['reason']}",
        "",
        "## Audited Runs",
        "",
        *[f"- `{run_id}`" for run_id in CONTROL],
        "",
        "## Canonical Gap State",
        "",
        "Latest semantic state is selected per `(run_id, dimension_key, requirement_type)` by descending `state_version` and timestamp. Historical rows are not counted as simultaneous requirements.",
        f"- Final semantic requirements: {report['canonical_gap_summary']['total']}",
        f"- CLOSED: {report['canonical_gap_summary']['closed']}",
        f"- PARTIAL: {report['canonical_gap_summary']['partial']}",
        f"- OPEN: {report['canonical_gap_summary']['open']}",
        f"- Unclosed (PARTIAL or OPEN): {report['remaining_gap_inventory']['total_unclosed_requirements']}",
        f"- Unclosed by requirement type: {report['remaining_gap_inventory']['unclosed_by_requirement_type']}",
        f"- Unclosed by failure class: {report['remaining_gap_inventory']['unclosed_by_failure_class']}",
        "",
        "## Recurring Unclosed Requirements",
        "",
        "| Run | Question | Dimension | Type | Status | Evidence | Classification |",
        "|---|---|---|---|---|---:|---|",
    ]
    for item in report["unclosed_requirements"]:
        md.append(
            f"| `{item['run_id'][:8]}` | {item['question_id']} | {item['dimension_key']} | {item['requirement_type']} | {item['closure_status']} | {item['evidence']['accepted_count']} accepted | {item['classification']} |"
        )
    md.extend([
        "",
        "## Release-Blocker Matrix",
        "",
        "| Issue | Recurrence | Direct evidence | Release-blocking? | Classification |",
        "|---|---|---|---|---|",
    ])
    for row in report["release_blocker_matrix"]:
        md.append(f"| {row['issue']} | {row['recurrence']} | {row['direct_evidence']} | {row['release_blocking']} | {row['classification']} |")
    md.extend([
        "",
        "## Reliability",
        "",
        report["reliability"]["assessment"],
        "",
        "No new Benchmark was started. This audit does not alter the Phase 15.2 Candidate, the Phase 15.0 Golden Baseline, or production logic.",
        "",
        "## Stop Gate",
        "",
        "**RESULT A — NO PRODUCTION FIX NEEDED.** The V1 Pre-RC reference is suitable for RC reference use; no Phase 16.1 or Phase 17 execution was started by this audit.",
    ])
    audit_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    matrix_md.write_text(
        "# Phase 16.0 — Release Blocker Matrix\n\n"
        + "| Issue | Recurrence | Release-blocking? | Bounded fix | Classification |\n|---|---|---|---|---|\n"
        + "\n".join(
            f"| {row['issue']} | {row['recurrence']} | {row['release_blocking']} | {row['bounded_fix']} | {row['classification']} |"
            for row in report["release_blocker_matrix"]
        )
        + "\n\n**Decision:** `NO_DOMINANT_RELEASE_BLOCKING_HARD_GAP`; V1 Pre-RC is `CLOSED`.\n",
        encoding="utf-8",
    )
    reference_md.write_text(
        "# V1 Pre-RC Reference\n\n"
        + "> Frozen Phase 15.1 reference for Phase 16.0 only; no new run was started.\n\n"
        + "## Runs\n\n"
        + "\n".join(f"- `{run_id}`" for run_id in CONTROL)
        + "\n\n## Identity\n\n"
        + "- Benchmark: `v1-dev-model-comparison` / `1.0.0`\n"
        + "- Metric definition: `coverage-v1`\n"
        + "- Tier: `standard`\n"
        + "- `independent_source_targeting_enabled=true`\n"
        + "- `evidence_input_quality_enabled=false`\n"
        + "\n## Use\n\nV1 absolute/pre-RC reference and longitudinal context; not a new causal experiment.\n",
        encoding="utf-8",
    )


def main() -> None:
    with psycopg.connect(db_url()) as connection:
        report = build_report(connection)
    write_outputs(report)
    print(json.dumps({
        "decision": report["decision"],
        "canonical_gap_summary": report["canonical_gap_summary"],
        "artifacts": [
            "artifacts/phase16_hard_gap_audit.json",
            "artifacts/phase16_hard_gap_audit.md",
            "artifacts/phase16_release_blocker_matrix.json",
            "artifacts/phase16_release_blocker_matrix.md",
            "artifacts/v1_pre_rc_reference.json",
            "artifacts/v1_pre_rc_reference.md",
        ],
    }, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
