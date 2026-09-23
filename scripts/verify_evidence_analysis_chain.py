"""Phase 14.0 read-only runtime verification (no benchmark execution).

Proves that persisted run data can already flow through the new explanation
chain::

    Gap -> Evidence state -> Failure classification -> Research Need

For every unclosed (OPEN/PARTIAL) GapRequirement of one research run this
script rebuilds the generic domain requirement, runs the pure Phase 14.0
analyzers, and joins the result against the persisted feedback/need event
chain.  It only issues SELECTs and never mutates runtime state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.domain.evidence_failure_classification import (  # noqa: E402
    classify_evidence_failure,
)
from app.domain.gap_closure import (  # noqa: E402
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.research_need_refinement import refine_research_need  # noqa: E402


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _latest_run_id(connection: psycopg.Connection) -> str | None:
    row = connection.execute(
        """
        SELECT run_id::text
        FROM gap_requirements
        GROUP BY run_id
        ORDER BY max(updated_at) DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row is not None else None


def _requirement_rows(
    connection: psycopg.Connection, run_id: str
) -> list[dict[str, Any]]:
    cur = connection.execute(
        """
        SELECT gap_id, plan_version, question_id, dimension_key, requirement_type,
               criterion, required_evidence_count, required_independent_sources,
               current_evidence_count, current_independent_sources,
               verification_status, closure_status, state_version,
               created_at, updated_at
        FROM gap_requirements
        WHERE run_id = %s
          AND plan_version = (
            SELECT max(plan_version) FROM gap_requirements WHERE run_id = %s
          )
        ORDER BY question_id, dimension_key
        """,
        (run_id, run_id),
    )
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def _event_refs(
    connection: psycopg.Connection, run_id: str, event_type: str
) -> list[dict[str, Any]]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT refs FROM agent_events WHERE run_id = %s AND event_type = %s",
            (run_id, event_type),
        )
        if isinstance(row[0], dict)
    ]


def _to_requirement(row: dict[str, Any], run_id: str) -> GapRequirement:
    created_at = row["created_at"]
    updated_at = row["updated_at"]
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return GapRequirement(
        gap_id=UUID(str(row["gap_id"])),
        run_id=UUID(run_id),
        question_id=str(row["question_id"]),
        dimension_key=str(row["dimension_key"]),
        requirement_type=GapRequirementType(str(row["requirement_type"])),
        criterion=str(row["criterion"]),
        required_evidence_count=int(row["required_evidence_count"]),
        required_independent_sources=int(row["required_independent_sources"]),
        current_evidence_count=int(row["current_evidence_count"]),
        current_independent_sources=int(row["current_independent_sources"]),
        verification_status=VerificationStatus(str(row["verification_status"])),
        closure_status=GapClosureStatus(str(row["closure_status"])),
        created_at=created_at,
        updated_at=updated_at,
        state_version=int(row["state_version"]),
        plan_version=int(row["plan_version"]),
    )


def build_chain(connection: psycopg.Connection, run_id: str) -> dict[str, Any]:
    feedback_by_gap = {
        str(ref.get("gap_id")): str(ref.get("feedback_id"))
        for ref in _event_refs(connection, run_id, "closure.feedback.generated")
        if ref.get("gap_id") and ref.get("feedback_id")
    }
    need_by_feedback: dict[str, str] = {}
    for ref in _event_refs(connection, run_id, "research.need.generated"):
        if ref.get("feedback_id") and ref.get("need_id"):
            need_by_feedback.setdefault(str(ref["feedback_id"]), str(ref["need_id"]))
    dispatched_feedback = {
        str(ref.get("feedback_id"))
        for ref in _event_refs(connection, run_id, "research.feedback.dispatched")
        if ref.get("feedback_id")
    }

    entries: list[dict[str, Any]] = []
    unclosed = 0
    explained = 0
    for row in _requirement_rows(connection, run_id):
        requirement = _to_requirement(row, run_id)
        if requirement.closure_status is GapClosureStatus.CLOSED:
            continue
        unclosed += 1
        classification = classify_evidence_failure(requirement)
        refinement = refine_research_need(classification.primary_reason)
        gap_key = str(requirement.gap_id)
        feedback_id = feedback_by_gap.get(gap_key)
        need_id = need_by_feedback.get(feedback_id) if feedback_id else None
        has_explanation = feedback_id is not None or need_id is not None
        explained += int(has_explanation)
        entries.append(
            {
                "gap_id": gap_key,
                "closure_status": requirement.closure_status.value,
                "evidence_count": requirement.current_evidence_count,
                "independent_sources": requirement.current_independent_sources,
                "failure_reasons": [
                    reason.value for reason in classification.reasons
                ],
                "primary_failure_reason": classification.primary_reason.value,
                "refined_need_type": refinement.refined_need_type.value,
                "persisted_need_type": refinement.persisted_need_type.value,
                "query_hints": list(refinement.query_hints),
                "feedback_id": feedback_id,
                "research_need_id": need_id,
                "feedback_dispatched": feedback_id in dispatched_feedback
                if feedback_id
                else False,
            }
        )
    return {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "unclosed_gaps": unclosed,
        "gaps_with_persisted_feedback": explained,
        "chain_coverage": round(explained / unclosed, 4) if unclosed else 1.0,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")

    with psycopg.connect(_database_uri(args.database_url)) as connection:
        run_id = args.run_id or _latest_run_id(connection)
        if run_id is None:
            print("no gap_requirements found; nothing to verify")
            return 1
        report = build_chain(connection, run_id)

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered[:4000])
    print(
        f"\nunclosed={report['unclosed_gaps']} "
        f"with_feedback={report['gaps_with_persisted_feedback']} "
        f"coverage={report['chain_coverage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
