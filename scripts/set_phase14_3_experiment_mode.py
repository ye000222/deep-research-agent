"""Phase 14.3 Task B: tag one research run with its A/B ``experiment_mode``.

The benchmark validation only *observes* the Phase 14.2 capability; it must not
change research behavior.  ``experiment_mode`` is therefore written as an inert
metadata key on ``usage_snapshot`` that no runtime code reads (grep-verified)
and that only the read-only A/B analyzer (``analyze_phase14_3_ab.py``) consumes.

The write is scoped to a single jsonb path via ``jsonb_set`` so it can never
clobber the surrounding budget/usage/quality projections.  The research worker
rewrites the whole ``usage_snapshot`` dictionary under a row lock while a run is
active, so a mid-flight tag can be lost; the intended protocol is to apply the
tag right after the run is created (early availability) and re-apply it once the
run reaches a terminal state (authoritative).  This script is idempotent and can
be called any number of times.

Strictly additive tooling: it connects directly to PostgreSQL and imports no
runtime module.  It never touches the Planner, SearchTarget, Provider, Budget,
Gap Closure rules, or the Research Loop decision path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg

BASELINE = "baseline"
EVIDENCE_AWARE = "evidence_aware"
VALID_MODES = (BASELINE, EVIDENCE_AWARE)

#: The single usage_snapshot key owned by this tool.
EXPERIMENT_MODE_PATH = "experiment_mode"


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def tag_experiment_mode(
    connection: psycopg.Connection,
    run_id: str,
    experiment_mode: str,
) -> dict[str, object]:
    """Write ``experiment_mode`` onto one run and return the persisted view."""

    if experiment_mode not in VALID_MODES:
        raise ValueError(f"experiment_mode must be one of {VALID_MODES}")
    with connection, connection.cursor() as cursor:
        cursor.execute("SELECT status FROM research_runs WHERE id = %s", (run_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"research run {run_id} not found")
        cursor.execute(
            """
            UPDATE research_runs
            SET usage_snapshot = jsonb_set(
                    COALESCE(usage_snapshot, '{}'::jsonb),
                    %s,
                    to_jsonb(%s::text),
                    true
                )
            WHERE id = %s
            RETURNING usage_snapshot ->> 'experiment_mode'
            """,
            (f"{{{EXPERIMENT_MODE_PATH}}}", experiment_mode, run_id),
        )
        persisted = cursor.fetchone()
    return {
        "run_id": run_id,
        "run_status": str(row[0]),
        "experiment_mode": persisted[0] if persisted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-mode", required=True, choices=VALID_MODES)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")

    with psycopg.connect(_database_uri(args.database_url)) as connection:
        result = tag_experiment_mode(connection, args.run_id, args.experiment_mode)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # Keep the module importable from the analyzer without a repo-root shim.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
