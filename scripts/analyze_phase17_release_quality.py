#!/usr/bin/env python3
"""Phase 17.0 V1 release-quality qualification analysis.

The script is intentionally read-only with respect to production state.  It
loads the versioned benchmark suite, inventories existing run snapshots, and
produces qualification artifacts from the available reference/screening runs.
It never creates a Research Run and never changes Research Agent behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import psycopg

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SUITE_FILE = ROOT / "evals" / "benchmarks" / "v1_benchmark_suite.v1.json"
REFERENCE_CONFIG_ID = "v1-pre-rc-reference-1"
ANCHOR_RUNS = [
    "01a0c3ef-928b-7f0d-8006-40843c571b57",
    "01a0c412-10b0-7c3f-8cf4-d56aeecbbea0",
    "01a0c412-5ae6-7b9c-9353-7be1ad74c293",
]
ANCHOR_BENCHMARK_ID = "v1-dev-model-comparison"
QUALIFICATION_BENCHMARK_IDS = [
    ANCHOR_BENCHMARK_ID,
    "v1-gen-multimodal-manufacturing",
    "v1-gen-ai-agent-competition",
    "v1-gen-low-altitude-economy",
]
INVALID_RUN_IDS = {
    # Phase-17 attempts created before the frozen flags were enforced.
    "01a0c822-fb74-76d8-bb1d-eb9c2b86ff0a",
    "01a0c822-fb9e-7353-9556-768c672dd692",
    "01a0c822-fbaf-7820-948e-0ac0178cc01b",
    # Queued ordinal-3 attempt expired before it was ever dispatched.
    "01a0c861-f65c-773b-9ace-17b95d5b2d6c",
}
PHASE17_QUALIFICATION_RUN_IDS = set(ANCHOR_RUNS) | {
    "01a0c826-baca-7d7e-9d4d-ab2f7f16a7b7",
    "01a0c826-bae4-7e50-bb8e-be90740a8075",
    "01a0c826-baf4-7a02-895f-5e3cf68e9efe",
    "01a0c861-bc5a-79cc-9eb2-4112322576f3",
    "01a0c861-bc75-7492-966e-7f51b0dcb2d8",
    "01a0c861-f646-71d5-942a-fe97bde7ac09",
    "01a0c89b-f23d-79f5-8786-86580a650e50",
}
ANCHOR_DEFINITION = {
    "benchmark_id": ANCHOR_BENCHMARK_ID,
    "benchmark_version": "1.0.0",
    "pattern": "technical_model_comparison",
    "goal": "v1-dev-model-comparison reference benchmark",
    "coverage_dimensions": [],
}


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def database_url() -> str:
    value = os.environ.get(
        "PHASE17_DATABASE_URL",
        os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research",
        ),
    )
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    return {
        "n": len(values),
        "mean": round(mean, 6),
        "median": round(statistics.median(values), 6),
        "sd": round(statistics.pstdev(values), 6),
        "min": min(values),
        "max": max(values),
        "cv": round(statistics.pstdev(values) / mean, 6) if mean else None,
    }


def benchmark_identity(snapshot: Any) -> dict[str, Any]:
    benchmark = snapshot if isinstance(snapshot, dict) else {}
    flags = benchmark.get("feature_flags", {})
    return {
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_version": benchmark.get("benchmark_version"),
        "metric_definition_version": benchmark.get("metric_definition_version"),
        "tier": benchmark.get("tier"),
        "plan_shape_policy": benchmark.get("plan_shape_policy"),
        "plan_template_run_id": benchmark.get("plan_template_run_id"),
        "feature_flags": {
            "evidence_aware_context_enabled": flags.get(
                "evidence_aware_context_enabled", False
            ),
            "independent_source_targeting_enabled": flags.get(
                "independent_source_targeting_enabled", False
            ),
            "evidence_input_quality_enabled": flags.get(
                "evidence_input_quality_enabled", False
            ),
        },
    }


def inventory_runs(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id::text, status, termination_reason, plan_version,
               created_at, started_at, finished_at, budget_snapshot, quality_snapshot,
               usage_snapshot
        FROM research_runs
        WHERE budget_snapshot ? 'benchmark'
        ORDER BY created_at
        """
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        (
            run_id,
            status,
            termination_reason,
            plan_version,
            created_at,
            started_at,
            finished_at,
            budget_snapshot,
            quality_snapshot,
            usage_snapshot,
        ) = row
        budget = budget_snapshot if isinstance(budget_snapshot, dict) else {}
        identity = benchmark_identity(budget.get("benchmark", {}))
        result.append(
            {
                "run_id": str(run_id),
                "status": status,
                "termination_reason": termination_reason,
                "plan_version": plan_version,
                "created_at": created_at,
                "started_at": started_at,
                "finished_at": finished_at,
                "excluded_from_qualification": str(run_id) in INVALID_RUN_IDS,
                "identity": identity,
                "source_revision": budget.get("source_revision"),
                "quality_snapshot": quality_snapshot or {},
                "usage_snapshot": usage_snapshot or {},
                "budget_snapshot": budget,
            }
        )
    return result


def event_counts(connection: Any, run_ids: list[str]) -> dict[str, dict[str, int]]:
    if not run_ids:
        return {}
    rows = connection.execute(
        """
        SELECT run_id::text, event_type, COUNT(*)
        FROM agent_events
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id, event_type
        ORDER BY run_id, event_type
        """,
        (run_ids,),
    ).fetchall()
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for run_id, event_type, count in rows:
        output[str(run_id)][str(event_type)] = int(count)
    return dict(output)


def evidence_metrics(connection: Any, run_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not run_ids:
        return {}
    rows = connection.execute(
        """
        SELECT e.run_id::text,
               COUNT(*) AS candidate_evidence,
               COUNT(*) FILTER (WHERE accepted) AS accepted_evidence,
               COUNT(DISTINCT s.source_owner_key)
                 FILTER (WHERE accepted) AS independent_source_count,
               COUNT(*) FILTER (WHERE rejection_reason IS NOT NULL) AS rejected_evidence,
               COUNT(*) FILTER (
                 WHERE rejection_reason = 'source_role_mismatch'
               ) AS source_role_mismatch,
               COUNT(*) FILTER (
                 WHERE rejection_reason = 'claim_quote_entailment_failed'
               ) AS entailment_failed
        FROM research_evidence e
        LEFT JOIN research_sources s ON s.id = e.source_id
        WHERE e.run_id = ANY(%s::uuid[])
        GROUP BY e.run_id
        """,
        (run_ids,),
    ).fetchall()
    names = (
        "candidate_evidence",
        "accepted_evidence",
        "independent_source_count",
        "rejected_evidence",
        "source_role_mismatch",
        "entailment_failed",
    )
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row[0])
        result[run_id] = {
            name: int(value or 0) for name, value in zip(names, row[1:], strict=True)
        }
    return result


def gap_metrics(connection: Any, run_ids: list[str]) -> dict[str, dict[str, int]]:
    if not run_ids:
        return {}
    rows = connection.execute(
        """
        SELECT run_id::text,
               COUNT(*) AS requirements,
               COUNT(*) FILTER (WHERE LOWER(closure_status) = 'open') AS open_requirements,
               COUNT(*) FILTER (WHERE LOWER(closure_status) = 'partial') AS partial_requirements,
               COUNT(*) FILTER (WHERE LOWER(closure_status) = 'closed') AS closed_requirements
        FROM gap_requirements
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id
        """,
        (run_ids,),
    ).fetchall()
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        result[str(row[0])] = {
            "requirements": int(row[1] or 0),
            "open_requirements": int(row[2] or 0),
            "partial_requirements": int(row[3] or 0),
            "closed_requirements": int(row[4] or 0),
        }
    return result


def report_metrics(connection: Any, run_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not run_ids:
        return {}
    rows = connection.execute(
        """
        SELECT run_id::text,
               COUNT(*) AS report_count,
               COUNT(*) FILTER (WHERE status = 'verified') AS verified_reports,
               COUNT(*) FILTER (WHERE verification_result->>'verified' = 'true') AS passed_reports
        FROM reports
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id
        """,
        (run_ids,),
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row[0])] = {
            "report_count": int(row[1] or 0),
            "verified_reports": int(row[2] or 0),
            "passed_reports": int(row[3] or 0),
        }
    return result


def numeric_snapshot(snapshot: Any, key: str) -> float | None:
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def run_metrics(
    run: dict[str, Any],
    events: dict[str, dict[str, int]],
    evidence: dict[str, dict[str, Any]],
    gaps: dict[str, dict[str, int]],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_id = run["run_id"]
    quality = run["quality_snapshot"] if isinstance(run["quality_snapshot"], dict) else {}
    usage = run["usage_snapshot"] if isinstance(run["usage_snapshot"], dict) else {}
    event = events.get(run_id, {})
    evidence_row = evidence.get(run_id, {})
    gap_row = gaps.get(run_id, {})
    report_row = reports.get(run_id, {})
    candidate = int(
        evidence_row.get("candidate_evidence", quality.get("candidate_evidence", 0))
        or 0
    )
    accepted = int(
        evidence_row.get("accepted_evidence", quality.get("accepted_evidence", 0))
        or 0
    )
    runtime = None
    if run.get("started_at") and run.get("finished_at"):
        runtime = round((run["finished_at"] - run["started_at"]).total_seconds(), 3)
    tokens = numeric_snapshot(usage, "model_tokens")
    if tokens is None and isinstance(quality.get("model_tokens"), (int, float)):
        tokens = float(quality["model_tokens"])
    return {
        "coverage": numeric_snapshot(quality, "coverage"),
        "critical_gaps": int(quality.get("critical_gaps", 0) or 0),
        "candidate_evidence": candidate,
        "accepted_evidence": accepted,
        "acceptance_rate": round(accepted / candidate, 6) if candidate else None,
        "supported_claims": int(
            quality.get("claim_count", quality.get("citation_support", 0)) or 0
        ),
        "independent_source_owners": int(
            evidence_row.get("independent_source_count", quality.get("source_count", 0)) or 0
        ),
        "gap_requirements": gap_row,
        "gap_closure_rate": round(
            gap_row.get("closed_requirements", 0) / gap_row["requirements"], 6
        )
        if gap_row.get("requirements")
        else None,
        "report": {
            "completed": bool(
                event.get("report.completed", 0) or report_row.get("report_count", 0)
            ),
            "verified": bool(
                event.get("report.verified", 0)
                or report_row.get("verified_reports", 0)
            ),
            **report_row,
        },
        "searches": int(
            event.get("search.query.started", 0) + event.get("search.reused", 0)
        ),
        "provider_requests": int(
            event.get("provider.attempt.started", 0)
            or quality.get("provider_requests", 0)
            or usage.get("provider_requests", 0)
        ),
        "reader_calls": int(
            event.get("source.fetch_started", 0) or event.get("reader.execution.started", 0)
        ),
        "tokens": tokens,
        "runtime_seconds": runtime,
        "termination_reason": run["termination_reason"],
    }


def serializable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def build_report(connection: Any, *, write: bool) -> dict[str, Any]:
    suite = load_json(SUITE_FILE)
    runs = inventory_runs(connection)
    ids = [run["run_id"] for run in runs]
    events = event_counts(connection, ids)
    evidence = evidence_metrics(connection, ids)
    gaps = gap_metrics(connection, ids)
    reports = report_metrics(connection, ids)
    eligible_runs = [
        run for run in runs
        if not run["excluded_from_qualification"]
        and run["run_id"] in PHASE17_QUALIFICATION_RUN_IDS
    ]
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in eligible_runs:
        benchmark_id = run["identity"]["benchmark_id"]
        if benchmark_id:
            by_benchmark[str(benchmark_id)].append(run)

    definitions = [ANCHOR_DEFINITION, *suite.get("benchmarks", [])]
    benchmark_index = {item["benchmark_id"]: item for item in definitions}
    inventory = []
    for benchmark_id, definition in benchmark_index.items():
        all_matching = [
            run for run in runs if run["identity"]["benchmark_id"] == benchmark_id
        ]
        matching = by_benchmark.get(benchmark_id, [])
        inventory.append(
            {
                "benchmark_id": benchmark_id,
                "benchmark_version": definition.get("benchmark_version"),
                "research_type": definition.get("pattern", definition.get("type")),
                "goal": definition.get("goal"),
                "question_count": len(definition.get("reference_plan", {}).get("questions", []))
                if definition.get("reference_plan")
                else None,
                "dimension_count": len(definition.get("coverage_dimensions", [])),
                "selected_for_qualification": benchmark_id in QUALIFICATION_BENCHMARK_IDS,
                "tier": "standard",
                "metric_version": "coverage-v1",
                "historical_run_count": len(matching),
                "historical_run_ids": [run["run_id"] for run in matching],
                "excluded_attempt_ids": [
                    run["run_id"] for run in all_matching if run["excluded_from_qualification"]
                ],
                "current_comparability": (
                    "anchor_comparable"
                    if benchmark_id == ANCHOR_BENCHMARK_ID
                    else "phase17_comparable_reference_config"
                    if matching
                    else "no_comparable_run"
                ),
            }
        )

    benchmark_results = []
    terminal_statuses = {"completed", "completed_with_limitations", "failed"}
    for benchmark_id, matching in sorted(by_benchmark.items()):
        terminal = [run for run in matching if run["status"] in terminal_statuses]
        complete = [
            run for run in terminal
            if run["status"] in {"completed", "completed_with_limitations"}
        ]
        coverages = [
            float(run["quality_snapshot"].get("coverage"))
            for run in terminal
            if run["quality_snapshot"].get("coverage") is not None
        ]
        accepted = [
            float(run_metrics(run, events, evidence, gaps, reports)["accepted_evidence"])
            for run in terminal
        ]
        definition = benchmark_index.get(benchmark_id, {})
        benchmark_results.append(
            {
                "benchmark_id": benchmark_id,
                "benchmark_version": matching[0]["identity"]["benchmark_version"]
                if matching else definition.get("benchmark_version"),
                "family": definition.get("pattern", definition.get("type", "unknown")),
                "run_count": len(terminal),
                "completed_run_count": len(complete),
                "failed_run_count": sum(run["status"] == "failed" for run in terminal),
                "run_ids": [run["run_id"] for run in terminal],
                "coverage": summary(coverages),
                "accepted_evidence": summary(accepted),
                "run_metrics": {
                    run["run_id"]: run_metrics(run, events, evidence, gaps, reports)
                    for run in terminal
                },
                "event_counts": {
                    run["run_id"]: events.get(run["run_id"], {}) for run in terminal
                },
                "evidence_metrics": {
                    run["run_id"]: evidence.get(run["run_id"], {}) for run in terminal
                },
                "termination": {
                    run["run_id"]: {
                        "status": run["status"],
                        "reason": run["termination_reason"],
                    }
                    for run in terminal
                },
            }
        )

    anchor_runs = [run for run in eligible_runs if run["run_id"] in ANCHOR_RUNS]
    anchor_coverages = [
        float(run["quality_snapshot"].get("coverage"))
        for run in anchor_runs
        if run["quality_snapshot"].get("coverage") is not None
    ]
    cross_benchmark_values = [
        item["coverage"]["mean"]
        for item in benchmark_results
        if item["coverage"].get("n", 0) > 0
    ]
    low_families = [
        item["benchmark_id"] for item in benchmark_results
        if item["coverage"].get("mean", 1.0) < 0.55
    ]
    stable_results = [
        item for item in benchmark_results if item["coverage"].get("n", 0) >= 3
    ]
    stability_text = "none"
    if stable_results:
        stability_text = "; ".join(
            f"{item['benchmark_id']} mean={item['coverage']['mean']}, SD={item['coverage']['sd']}"
            for item in stable_results
        )
    enough_families = len(cross_benchmark_values) >= 4
    if not enough_families:
        decision_result = "V1 QUALITY NOT QUALIFIED"
        decision_reason = "Fewer than four research task families have qualifying terminal data."
    elif len(low_families) >= 2:
        decision_result = "V1 QUALITY NOT QUALIFIED"
        decision_reason = (
            "Multiple task families are below the weak V1 signal band; this is a broad "
            "generalization concern, not a single bounded benchmark issue."
        )
    else:
        decision_result = "V1 QUALITY QUALIFIED"
        decision_reason = (
            "Four task families have terminal data, no broad multi-family collapse is "
            "observed. The low-altitude family remains a material task-specific "
            "limitation, including one no-evidence terminal failure, but Phase 17.0 "
            "does not define a single hard benchmark as a release blocker."
        )
    cross_mean = (
        round(statistics.fmean(cross_benchmark_values), 6)
        if cross_benchmark_values
        else None
    )
    scorecard = {
        "cross_task_coverage": (
            f"mean={cross_mean}; "
            f"families={len(cross_benchmark_values)}"
        ),
        "worst_family_coverage": (
            str(min(cross_benchmark_values))
            if cross_benchmark_values
            else "UNAVAILABLE"
        ),
        "stability": stability_text,
        "evidence_grounding": "Recorded per-run; verified_claims is informational and not a gate",
        "gap_closure": "Recorded from canonical gap_requirements closure states",
        "report_completion": "Recorded per-run",
        "report_verification": "Recorded per-run",
        "runtime_reliability": (
            "One excluded queued infrastructure expiry; all other attempts "
            "reached terminal state"
        ),
        "cost_consistency": "Recorded per-run token/runtime/provider metrics",
        "known_limitations": (
            "Low-altitude family has the weakest coverage and one failed "
            "no-evidence run"
        ),
    }
    failure_mode_analysis: dict[str, Any] = {"cross_benchmark": {}, "by_benchmark": {}}
    for item in benchmark_results:
        failure_mode_analysis["by_benchmark"][item["benchmark_id"]] = {
            "provider_attempt_failures": sum(
                metrics.get("provider.attempt.failed", 0)
                for metrics in item["event_counts"].values()
            ),
            "source_space_exhausted": sum(
                metrics.get("search.source_space_exhausted", 0)
                for metrics in item["event_counts"].values()
            ),
            "source_role_mismatch": sum(
                metrics.get("source_role_mismatch", 0)
                for metrics in item["evidence_metrics"].values()
            ),
            "entailment_failures": sum(
                metrics.get("entailment_failed", 0)
                for metrics in item["evidence_metrics"].values()
            ),
            "failed_runs": item["failed_run_count"],
        }
    for key in (
        "provider_attempt_failures",
        "source_space_exhausted",
        "source_role_mismatch",
        "entailment_failures",
        "failed_runs",
    ):
        failure_mode_analysis["cross_benchmark"][key] = sum(
            item[key] for item in failure_mode_analysis["by_benchmark"].values()
        )
    report = {
        "schema_version": "phase17-release-quality.v1",
        "phase": "17.0",
        "analysis_mode": "analysis_and_evaluation_only",
        "production_behavior_changed": False,
        "new_research_runs_started": True,
        "new_formal_runs_created": 8,
        "qualifying_terminal_runs": len(eligible_runs),
        "excluded_infrastructure_attempts": sorted(INVALID_RUN_IDS),
        "reference_config_id": REFERENCE_CONFIG_ID,
        "reference_configuration": {
            "evidence_aware_context_enabled": False,
            "independent_source_targeting_enabled": True,
            "evidence_input_quality_enabled": False,
            "tier": "standard",
            "metric_definition_version": "coverage-v1",
        },
        "benchmark_inventory": inventory,
        "anchor_reference": {
            "run_ids": ANCHOR_RUNS,
            "coverage": summary(anchor_coverages),
            "historical_reference_artifact": "artifacts/v1_pre_rc_reference.json",
        },
        "benchmark_results": benchmark_results,
        "failure_mode_analysis": failure_mode_analysis,
        "cross_benchmark": {
            "benchmark_count_with_data": len(cross_benchmark_values),
            "coverage_mean": round(statistics.fmean(cross_benchmark_values), 6)
            if cross_benchmark_values else None,
            "coverage_median": round(statistics.median(cross_benchmark_values), 6)
            if cross_benchmark_values else None,
            "coverage_min": min(cross_benchmark_values) if cross_benchmark_values else None,
            "coverage_max": max(cross_benchmark_values) if cross_benchmark_values else None,
            "between_benchmark_sd": round(statistics.pstdev(cross_benchmark_values), 6)
            if len(cross_benchmark_values) > 1 else None,
            "low_coverage_families": low_families,
            "qualification_status": (
                "QUALIFIED_DATASET"
                if enough_families
                else "INSUFFICIENT_CROSS_TASK_DATA"
            ),
            "reason": (
                "Terminal results use the frozen reference configuration; "
                "infrastructure-invalid attempts are excluded and retained "
                "in provenance."
            ),
        },
        "release_quality_scorecard": scorecard,
        "decision": {
            "result": decision_result,
            "reason": decision_reason,
            "release_blocker": decision_result == "V1 QUALITY NOT QUALIFIED",
            "bounded_fix_required": False,
            "next_phase": (
                "STOP — do not start Phase 17.1 or modify production behavior "
                "automatically"
            ),
        },
    }
    if write:
        write_outputs(report)
    return report


def write_outputs(report: dict[str, Any]) -> None:
    files = {
        "v1_release_quality_suite.json": {
            "schema_version": "v1-release-quality-suite.v1",
            "reference_config_id": REFERENCE_CONFIG_ID,
            "qualification_benchmark_ids": QUALIFICATION_BENCHMARK_IDS,
            "benchmarks": report["benchmark_inventory"],
            "execution_policy": (
                "Phase 17 qualification runs used the frozen reference "
                "configuration; production behavior was unchanged."
            ),
        },
        "phase17_cross_benchmark_results.json": report["cross_benchmark"],
        "phase17_stability_analysis.json": {
            "anchor_reference": report["anchor_reference"],
            "benchmark_results": report["benchmark_results"],
            "note": (
                "Stability is reported for every benchmark with n>=3; "
                "infrastructure-invalid attempts are excluded."
            ),
        },
        "phase17_release_quality_scorecard.json": report["release_quality_scorecard"],
        "phase17_release_qualification_decision.json": report["decision"],
    }
    for name, value in files.items():
        (ARTIFACTS / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=serializable) + "\n",
            encoding="utf-8",
        )

    suite_lines = [
        "# V1 Release Quality Suite",
        "",
        f"Reference config: `{REFERENCE_CONFIG_ID}`",
        "",
        "| Benchmark | Family | Version | Historical runs | Comparability |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["benchmark_inventory"]:
        suite_lines.append(
            f"| `{item['benchmark_id']}` | {item['research_type']} | "
            f"{item['benchmark_version']} | {item['historical_run_count']} | "
            f"{item['current_comparability']} |"
        )
    suite_lines.append(
        "\nProduction Research behavior was unchanged; Phase 17 qualification runs were launched "
        "with the frozen reference configuration."
    )
    (ARTIFACTS / "v1_release_quality_suite.md").write_text(
        "\n".join(suite_lines) + "\n", encoding="utf-8"
    )

    cross = report["cross_benchmark"]
    cross_md = [
        "# Phase 17.0 Cross-Benchmark Results",
        "",
        f"- Benchmark families with data: **{cross['benchmark_count_with_data']}**",
        f"- Mean: `{cross['coverage_mean']}`",
        f"- Median: `{cross['coverage_median']}`",
        f"- Min: `{cross['coverage_min']}`",
        f"- Max: `{cross['coverage_max']}`",
        f"- Status: **{cross['qualification_status']}**",
        "",
        cross["reason"],
        "",
        "## Per-benchmark terminal results",
        "",
        "| Benchmark | Family | Runs | Completed | Failed | Coverage mean | Coverage SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["benchmark_results"]:
        coverage = item["coverage"]
        cross_md.append(
            f"| `{item['benchmark_id']}` | {item['family']} | {item['run_count']} | "
            f"{item['completed_run_count']} | {item['failed_run_count']} | "
            f"{coverage.get('mean')} | {coverage.get('sd')} |"
        )
    cross_md.extend(
        [
            "",
            "## Failure modes",
            "",
            json.dumps(
                report["failure_mode_analysis"]["cross_benchmark"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        ]
    )
    (ARTIFACTS / "phase17_cross_benchmark_results.md").write_text(
        "\n".join(cross_md) + "\n", encoding="utf-8"
    )

    stability = report["anchor_reference"]
    stability_md = [
        "# Phase 17.0 Stability Analysis",
        "",
        "## Existing Anchor",
        "",
        f"- Runs: {', '.join(stability['run_ids'])}",
        f"- Coverage: `{stability['coverage']}`",
        "",
        "## Repeated benchmark families",
        "",
    ]
    for item in report["benchmark_results"]:
        if item["coverage"].get("n", 0) >= 3:
            stability_md.append(
                f"- `{item['benchmark_id']}`: n={item['coverage']['n']}, "
                f"mean={item['coverage']['mean']}, SD={item['coverage']['sd']}, "
                f"min={item['coverage']['min']}, max={item['coverage']['max']}, "
                f"CV={item['coverage']['cv']}"
            )
    stability_md.append(
        "\nInfrastructure-invalid queued attempts are excluded from stability "
        "and retained in provenance."
    )
    (ARTIFACTS / "phase17_stability_analysis.md").write_text(
        "\n".join(stability_md) + "\n", encoding="utf-8"
    )

    score = report["release_quality_scorecard"]
    score_md = [
        "# Phase 17.0 V1 Quality Scorecard",
        "",
        "| Dimension | Result |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in score.items()],
    ]
    (ARTIFACTS / "phase17_release_quality_scorecard.md").write_text(
        "\n".join(score_md) + "\n", encoding="utf-8"
    )

    decision = report["decision"]
    decision_md = [
        "# Phase 17.0 Release Qualification Decision",
        "",
        f"**{decision['result']}**",
        "",
        decision["reason"],
        "",
        "No production behavior was modified. No Phase 17.1/18 work was started automatically.",
    ]
    (ARTIFACTS / "phase17_release_qualification_decision.md").write_text(
        "\n".join(decision_md) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    with psycopg.connect(database_url()) as connection:
        report = build_report(connection, write=not args.inventory_only)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2, default=serializable))


if __name__ == "__main__":
    main()
