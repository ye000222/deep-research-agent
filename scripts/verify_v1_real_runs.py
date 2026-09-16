"""Verify three consecutive real V1 runs against the hard quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg


@dataclass(frozen=True, slots=True)
class AcceptanceRun:
    run_id: str
    owner_hash: str
    normalized_goal: str
    status: str
    phase: str
    termination_reason: str
    saved_profile_id: str
    credential_version_id: str
    model: str
    budget_tier: str
    source_revision: str
    scoring_rule_version: str
    prompt_bundle_version: str
    graph_schema_revision: str
    coverage: float
    priority_one_coverage: float
    cross_validation: float
    critical_gaps: int
    report_count: int
    verified_report_count: int
    writing_started_count: int
    report_verified_count: int
    run_completed_count: int
    run_failed_count: int

    @property
    def configuration_fingerprint(self) -> tuple[str, ...]:
        return (
            self.owner_hash,
            self.normalized_goal,
            self.saved_profile_id,
            self.credential_version_id,
            self.model,
            self.budget_tier,
            self.source_revision,
            self.scoring_rule_version,
            self.prompt_bundle_version,
            self.graph_schema_revision,
        )


def evaluate_real_runs(
    runs: list[AcceptanceRun],
    *,
    required_count: int = 3,
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    """Return a machine-readable closeout verdict for newest-first runs."""

    failures: list[str] = []
    if len(runs) != required_count:
        failures.append(f"expected_{required_count}_consecutive_runs_found_{len(runs)}")

    if runs:
        fingerprint = runs[0].configuration_fingerprint
        if any(run.configuration_fingerprint != fingerprint for run in runs[1:]):
            failures.append("run_configuration_fingerprint_mismatch")

        source_revision = runs[0].source_revision.strip()
        if not source_revision or source_revision.casefold() in {"development", "unknown"}:
            failures.append("source_revision_not_release_identifiable")
        if expected_source_revision is not None and source_revision != expected_source_revision:
            failures.append("source_revision_mismatch")

    run_results: list[dict[str, Any]] = []
    for run in runs:
        checks = {
            "status_completed": run.status == "completed",
            "phase_terminal": run.phase == "terminal",
            "termination_quality_met": run.termination_reason == "quality_met",
            "coverage": run.coverage >= 0.85,
            "priority_one_coverage": run.priority_one_coverage >= 0.80,
            "cross_validation": run.cross_validation >= 0.70,
            "critical_gaps": run.critical_gaps == 0,
            "one_verified_report": run.report_count == 1
            and run.verified_report_count == 1,
            "report_stage_complete": run.writing_started_count >= 1
            and run.report_verified_count == 1
            and run.run_completed_count == 1
            and run.run_failed_count == 0,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            failures.append(f"run_{run.run_id}_failed:{','.join(failed_checks)}")
        run_results.append(
            {
                "run_id": run.run_id,
                "checks": checks,
                "metrics": {
                    "coverage": run.coverage,
                    "priority_one_coverage": run.priority_one_coverage,
                    "cross_validation": run.cross_validation,
                    "critical_gaps": run.critical_gaps,
                },
                "status": run.status,
                "termination_reason": run.termination_reason,
            }
        )

    return {
        "passed": not failures,
        "required_consecutive_runs": required_count,
        "evaluated_runs": len(runs),
        "source_revision": runs[0].source_revision if runs else None,
        "failures": failures,
        "runs": run_results,
    }


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def load_recent_runs(database_url: str, owner_hash: str, count: int) -> list[AcceptanceRun]:
    query = """
        SELECT
            r.id::text,
            r.owner_hash,
            r.normalized_goal,
            r.status,
            r.phase,
            COALESCE(r.termination_reason, ''),
            r.saved_profile_id::text,
            r.credential_version_id::text,
            COALESCE(r.llm_config_snapshot->>'model', ''),
            COALESCE(r.budget_snapshot->>'tier', ''),
            COALESCE(r.budget_snapshot->>'source_revision', ''),
            r.scoring_rule_version,
            r.prompt_bundle_version,
            r.graph_schema_revision,
            COALESCE((r.quality_snapshot->>'coverage')::float, 0.0),
            COALESCE((r.quality_snapshot->>'priority_one_coverage')::float, 0.0),
            COALESCE((r.quality_snapshot->>'cross_validation')::float, 0.0),
            COALESCE((r.quality_snapshot->>'critical_gaps')::int, 0),
            (SELECT COUNT(*) FROM reports p WHERE p.run_id = r.id),
            (SELECT COUNT(*) FROM reports p WHERE p.run_id = r.id AND p.status = 'verified'),
            (SELECT COUNT(*) FROM agent_events e
             WHERE e.run_id = r.id AND e.event_type = 'report.writing_started'),
            (SELECT COUNT(*) FROM agent_events e
             WHERE e.run_id = r.id AND e.event_type = 'report.verified'),
            (SELECT COUNT(*) FROM agent_events e
             WHERE e.run_id = r.id AND e.event_type = 'run.completed'),
            (SELECT COUNT(*) FROM agent_events e
             WHERE e.run_id = r.id AND e.event_type = 'run.failed')
        FROM research_runs r
        WHERE r.owner_hash = %s
        ORDER BY r.created_at DESC
        LIMIT %s
    """
    with psycopg.connect(_database_uri(database_url)) as connection:
        rows = connection.execute(query, (owner_hash, count)).fetchall()
    return [AcceptanceRun(*row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--owner-hash", default=os.getenv("V1_ACCEPTANCE_OWNER_HASH"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")
    if not args.owner_hash:
        parser.error("--owner-hash or V1_ACCEPTANCE_OWNER_HASH is required")
    if args.count < 3:
        parser.error("--count must be at least 3")

    runs = load_recent_runs(args.database_url, args.owner_hash, args.count)
    result = evaluate_real_runs(
        runs,
        required_count=args.count,
        expected_source_revision=args.expected_source_revision,
    )
    result["run_configuration"] = (
        {
            "normalized_goal_sha256": hashlib.sha256(
                runs[0].normalized_goal.encode("utf-8")
            ).hexdigest(),
            "saved_profile_id": runs[0].saved_profile_id,
            "credential_version_id": runs[0].credential_version_id,
            "model": runs[0].model,
            "budget_tier": runs[0].budget_tier,
            "source_revision": runs[0].source_revision,
            "scoring_rule_version": runs[0].scoring_rule_version,
            "prompt_bundle_version": runs[0].prompt_bundle_version,
            "graph_schema_revision": runs[0].graph_schema_revision,
        }
        if runs
        else None
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
