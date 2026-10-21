"""Freeze a deterministic development-only sample for Phase C.1 manual audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json"
OUTPUT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json"
SEED = "v2.1r-context-c1-structural-audit-v1"
TARGET = 40


def _bucket(count: int) -> str:
    return "1" if count == 1 else "2-3" if count <= 3 else "4+"


def _digest(claim_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{claim_id}".encode()).hexdigest()


def _stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    has_source_context = bool(row.get("source_context_graph", {}).get("contexts"))
    return (
        str(row["cohort"]),
        "detected" if has_source_context else "zero_detected",
        _bucket(len(row.get("propositions", []))),
    )


def main() -> None:
    replay = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = replay["claim_replays"]
    if len(rows) != 128 or any(
        row["cohort"] not in {"bounded_shadow", "fresh_validation", "final_unseen"} for row in rows
    ):
        raise SystemExit("Expected exactly 128 known development replay rows")
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[_stratum(row)].append(row)
    for bucket in strata.values():
        bucket.sort(key=lambda row: _digest(str(row["claim_id"])))

    # Round-robin over sorted structural strata guarantees representation from
    # each non-empty stratum before filling remaining slots by stable hash.
    selected: list[dict[str, Any]] = []
    depth = 0
    ordered_strata = sorted(strata)
    while len(selected) < TARGET:
        progressed = False
        for key in ordered_strata:
            bucket = strata[key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                progressed = True
                if len(selected) == TARGET:
                    break
        if not progressed:
            break
        depth += 1

    entries = []
    for row in selected:
        source_contexts = row.get("source_context_graph", {}).get("contexts", [])
        output_contexts = row.get("output_context_graph", {}).get("contexts", [])
        entries.append(
            {
                "claim_id": row["claim_id"],
                "identity_hash": row["identity_hash"],
                "cohort": row["cohort"],
                "stratum": list(_stratum(row)),
                "proposition_count": len(row.get("propositions", [])),
                "source_context_count_before_review": len(source_contexts),
                "source_context_families_before_review": sorted(
                    {item.get("context_type") for item in source_contexts}
                ),
                "output_context_count": len(output_contexts),
                "output_context_families": sorted(
                    {item.get("context_type") for item in output_contexts}
                ),
                "composition_group_count": len(
                    row.get("source_context_graph", {}).get("scope_expansions", {})
                ),
                "manual_review": "NOT_STARTED",
            }
        )
    payload = {
        "artifact": "V2.1R Truth-Condition Context Phase C.1 development audit selection",
        "selection_version": "v2_1r-context-c1-audit-selection-v1",
        "source_replay": SOURCE.relative_to(ROOT).as_posix(),
        "source_replay_id": replay["replay_id"],
        "source_replay_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "selection_seed": SEED,
        "selection_rule": (
            "round-robin over cohort x detection state x proposition-count bucket; "
            "stable SHA-256 order within each stratum"
        ),
        "target_count": TARGET,
        "selected_count": len(entries),
        "development_only": True,
        "holdout_content_read": False,
        "provider_calls": 0,
        "manual_review_started": False,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "stratum_counts": {
            "|".join(key): sum(1 for entry in entries if tuple(entry["stratum"]) == key)
            for key in ordered_strata
            if any(tuple(entry["stratum"]) == key for entry in entries)
        },
        "selected_claims": entries,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(entries)} frozen development claims")
    print(json.dumps(payload["stratum_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
