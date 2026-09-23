#!/usr/bin/env python3
"""Read-only Phase 15.2 Evidence Input Quality audit.

The audit intentionally reports evidence-level facts and does not infer that a
role label is an ownership identity.  It also never changes acceptance state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "apps" / "api"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(API))


def classify_role(*, source_type: str | None, domain: str | None) -> str:
    if not source_type or not domain:
        return "ROLE_METADATA_MISSING"
    if source_type.casefold() in {"webpage", "html", "pdf", "document"}:
        return "ROLE_CLASSIFICATION_TOO_COARSE"
    return "OBSERVED_ROLE"


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rejection_counts = Counter(
        row["rejection_reason"] for row in rows if row.get("rejection_reason")
    )
    role_rows = [
        row
        for row in rows
        if row.get("rejection_reason") == "source_role_mismatch"
    ]
    entailment_rows = [
        row
        for row in rows
        if row.get("rejection_reason") == "claim_quote_entailment_failed"
    ]
    return {
        "schema_version": "phase15-2-input-quality-audit.v1",
        "row_count": len(rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_role_mismatch": {
            "count": len(role_rows),
            "classification": dict(
                Counter(
                    classify_role(
                        source_type=row.get("source_type"),
                        domain=row.get("domain"),
                    )
                    for row in role_rows
                )
            ),
            "items": role_rows,
        },
        "claim_quote_entailment_failed": {
            "count": len(entailment_rows),
            "classification": {
                "AUDIT_FROM_PERSISTED_REASON": len(entailment_rows),
                "SOURCE_TEXT_REQUIRED_FOR_SUBCLASSIFICATION": len(entailment_rows),
            },
            "items": entailment_rows,
        },
        "relevance_threshold_audit": {
            "count": rejection_counts.get("evidence_relevance_below_threshold", 0),
            "classification": "REQUIRES_SCORE_AND_SOURCE_REVIEW",
        },
        "security_rejections": rejection_counts.get("prompt_injection_detected", 0),
        "acceptance_is_canonical": True,
    }


def _rows(connection: Any, run_ids: list[str]) -> list[dict[str, Any]]:
    records = connection.execute(
        """
        SELECT e.id::text, e.run_id::text, e.question_id, e.claim_id::text,
               e.source_id::text, e.claim, e.exact_quote, e.accepted,
               e.rejection_reason, s.source_type, s.source_owner_key, s.domain,
               s.canonical_url
        FROM research_evidence e
        LEFT JOIN research_sources s ON s.id = e.source_id
        WHERE e.run_id = ANY(%s::uuid[])
        ORDER BY e.created_at, e.id
        """,
        (run_ids,),
    ).fetchall()
    columns = (
        "evidence_id", "run_id", "question_id", "claim_id", "source_id",
        "claim", "exact_quote", "accepted", "rejection_reason", "source_type",
        "source_owner_key", "domain", "url",
    )
    return [dict(zip(columns, row, strict=True)) for row in records]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--json", default="artifacts/phase15_2_input_quality_audit.json")
    parser.add_argument("--markdown", default="artifacts/phase15_2_input_quality_audit.md")
    args = parser.parse_args()
    if psycopg is None or not args.database_url:
        raise SystemExit("psycopg and DATABASE_URL are required")
    uri = args.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(uri) as connection:
        report = audit_rows(_rows(connection, args.run_id))
    Path(args.json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    md = [
        "# Phase 15.2 Evidence Input Quality Audit",
        "",
        f"Rows: {report['row_count']}",
        "",
        "## Rejections",
        "",
    ]
    md.extend(f"- `{key}`: {value}" for key, value in report["rejection_counts"].items())
    md.extend(["", "## Role mismatch classification", ""])
    md.extend(
        f"- `{key}`: {value}"
        for key, value in report["source_role_mismatch"]["classification"].items()
    )
    md.extend(
        [
            "",
            "## Entailment audit",
            "",
            "Persisted reason is retained; source-text review is required to "
            "distinguish wrong quote from unsupported source.",
        ]
    )
    Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
