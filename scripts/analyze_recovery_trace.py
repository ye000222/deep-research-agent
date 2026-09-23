"""Summarize a Recovery event trace JSON file for later Benchmark analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.domain.recovery_trace_collector import (  # noqa: E402
    RecoveryTraceCollector,
)


def summarize(events: list[dict[str, Any]]) -> dict[str, object]:
    """Return a compact, JSON-serializable Recovery trace summary."""

    trace = RecoveryTraceCollector.collect(events)
    return {
        "recovery_attempt_id": str(trace.recovery_attempt_id),
        "question_id": trace.question_id,
        "before_gap": list(trace.before_gap_state),
        "after_gap": list(trace.after_gap_state),
        "coverage_before": trace.before_coverage,
        "coverage_after": trace.after_coverage,
        "actions": {
            "query": len(trace.query_execution_ids),
            "reader": len(trace.reader_execution_ids),
            "evidence": len(trace.extraction_ids),
        },
        "alignment": len(trace.alignment_ids),
        "closure": len(trace.closure_evaluation_ids),
        "outcome": trace.outcome_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path, help="JSON array of persisted events")
    args = parser.parse_args()
    events = json.loads(args.events.read_text(encoding="utf-8"))
    print(json.dumps(summarize(events), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
