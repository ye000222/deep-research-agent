"""Re-evaluate only the 23 persisted shared-context candidates for Phase C.1."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/api"))

from app.domain.truth_condition_context import (  # noqa: E402
    ContextGraph,
    canonical_context_value,
    compare_context_graphs,
)

REPLAY = ROOT / "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json"
OUTPUT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json"

# Manual audit classification of the existing candidate set, based on source
# spans, proposition spans, typed outputs, and the bounded canonical comparison.
CLASSIFICATIONS = {
    ("01a0c86a-a4b4-7df3-965e-49e36b5e2b81", "sc0"): (
        "CANONICALIZATION_ERROR",
        "source method sub-operation and output method summary differ in granularity",
    ),
    ("01a0c85a-66b9-7e3e-b5ff-4787ab9887b9", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "source detector truncates a month-qualified publication date to year and attaches "
        "across clause",
    ),
    ("01a0c418-10aa-750b-b15c-944656609175", "sc0"): (
        "CORRECT_SHARED_CONTEXT",
        "explicit research attribution preserved after epistemic operator/relation "
        "canonicalization",
    ),
    ("01a0c3f5-e702-7de6-9b2d-95d70e3c98e0", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "complex-scenario phrase is duplicated as a generic scope candidate",
    ),
    ("01a0c3f5-e702-7de6-9b2d-95d70e3c98e0", "sc1"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "scenario belongs to the second proposition; source clause attachment over-expands",
    ),
    ("01a0c872-0ac8-7a2b-923a-637c1bb2a9aa", "sc0"): (
        "CORRECT_SHARED_CONTEXT",
        "review attribution and agent preserved",
    ),
    ("01a0c438-497d-76e7-948b-6cefaa29601a", "sc0"): (
        "CORRECT_SHARED_CONTEXT",
        "shared year target set preserved",
    ),
    ("01a0c86c-7a6a-7d0d-a006-9b383379f478", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "month-qualified date reduced to year and clause-wide target set over-expands",
    ),
    ("01a0c867-196b-7d29-8649-9a5b4bc08541", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "scenario/purpose phrase mis-typed as generic scope; payload is truncated",
    ),
    ("01a0c835-1e99-7d3a-b0d3-41882d52cd21", "sc0"): (
        "OUTPUT_PROJECTION_ERROR",
        "source obligation/attribution is absent from output epistemic fields",
    ),
    ("01a0c82f-63ee-779f-9516-008c4c1a126a", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "2024 applies to market-size proposition p0, not the separate "
        "source-attribution proposition",
    ),
    ("01a0c82f-63ee-779f-9516-008c4c1a126a", "sc1"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "2025 publication date belongs to p1; source clause attachment is too broad",
    ),
    ("01a0c82f-63ee-779f-9516-008c4c1a126a", "sc2"): (
        "CANONICALIZATION_ERROR",
        "report-horizon range is represented as a separate year occurrence and is not "
        "a supported source range form",
    ),
    ("01a0c82f-63ee-779f-9516-008c4c1a126a", "sc3"): (
        "OUTPUT_PROJECTION_ERROR",
        "2030 report-horizon endpoint is not represented in the output qualifiers",
    ),
    ("01a0c872-0ada-7891-b41e-32561698cc35", "sc0"): (
        "CORRECT_SHARED_CONTEXT",
        "review attribution and agent preserved",
    ),
    ("01a0c873-f06d-70d7-b1e7-03e3e28b9534", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "publication year belongs to p0; group/clause attachment reaches purpose propositions",
    ),
    ("01a0c844-5b88-7864-a812-82ea86a938ff", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "month-qualified release date reduced to year and attached beyond the release proposition",
    ),
    ("01a0c867-ec30-7349-a85b-31b1dc85c72a", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "event start year belongs to p0; clause-wide attachment over-expands to "
        "purpose proposition",
    ),
    ("01a0a725-5062-7e1f-b7e6-9a4823e02d75", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "X2000 product identifier is a temporal false positive",
    ),
    ("01a0a725-5062-7e1f-b7e6-9a4823e02d75", "sc1"): (
        "CANONICALIZATION_ERROR",
        "source combines an application set while output splits it into proposition-local values",
    ),
    ("01a0a62b-2ba5-701f-9a67-a8717cebf17e", "sc0"): (
        "CORRECT_SHARED_CONTEXT",
        "shared year target set preserved",
    ),
    ("01a093ff-abfe-7077-a846-103a610c7a0e", "sc0"): (
        "CORRECT_SHARED_CONTEXT",
        "shared year target set preserved",
    ),
    ("01a0c824-555b-7726-8a33-2d98136b0339", "sc0"): (
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "2025Q3 is reduced to year-only source context; output retains quarter granularity",
    ),
}


def main() -> None:
    source_bytes = REPLAY.read_bytes()
    replay = json.loads(source_bytes)
    results: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter(
        {
            name: 0
            for name in (
                "CORRECT_SHARED_CONTEXT",
                "TRUE_SCOPE_UNDEREXPANSION",
                "TRUE_SCOPE_OVEREXPANSION",
                "TRUE_ATTACHMENT_AMBIGUITY",
                "SOURCE_SCOPE_DETECTOR_ERROR",
                "OUTPUT_PROJECTION_ERROR",
                "CANONICALIZATION_ERROR",
                "DETECTOR_LIMIT",
            )
        }
    )
    for row in replay["claim_replays"]:
        for candidate in row.get("shared_contexts", []):
            key = (row["claim_id"], candidate["context_ref"])
            if key not in CLASSIFICATIONS:
                raise SystemExit(f"Missing manual classification for {key}")
            source_graph = ContextGraph.model_validate(row["source_context_graph"])
            output_graph = ContextGraph.model_validate(row["output_context_graph"])
            source_context = next(
                item
                for item in source_graph.contexts
                if item.context_ref == candidate["context_ref"]
            )
            result = compare_context_graphs(
                source_graph.model_copy(update={"contexts": (source_context,)}), output_graph
            )[0]
            classification, rationale = CLASSIFICATIONS[key]
            outcome_counts[result.outcome.value] += 1
            classification_counts[classification] += 1
            resolved_source_targets = sorted(
                member
                for target in source_context.scope.target_refs
                for member in source_graph.scope_expansions.get(target, (target,))
            )
            resolved_output_targets = sorted(
                {
                    member
                    for output in output_graph.contexts
                    if output.context_ref in result.output_context_refs
                    for edge in output.attachments
                    if edge.target_ref
                    for member in output_graph.scope_expansions.get(
                        edge.target_ref, (edge.target_ref,)
                    )
                }
            )
            results.append(
                {
                    "claim_id": row["claim_id"],
                    "context_ref": candidate["context_ref"],
                    "context_type": source_context.context_type.value,
                    "source_canonical_value": canonical_context_value(source_context),
                    "source_target_set": resolved_source_targets,
                    "matched_output_context_refs": list(result.output_context_refs),
                    "output_target_set": resolved_output_targets,
                    "comparator_outcome": result.outcome.value,
                    "comparator_reason_codes": list(result.reason_codes),
                    "classification": classification,
                    "manual_rationale": rationale,
                }
            )
    if len(results) != 23:
        raise SystemExit(f"Expected 23 existing shared-context candidates, got {len(results)}")
    payload = {
        "report_title": "Phase C.1 Shared Context Candidate Re-audit",
        "audit_version": "v2_1r-context-c1-shared-context-reaudit-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_replay": REPLAY.relative_to(ROOT).as_posix(),
        "source_replay_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "candidates_total": len(results),
        "replay_rerun": False,
        "holdout_content_read": False,
        "provider_calls": 0,
        "database_writes": 0,
        "comparator": (
            "bounded typed canonical comparator with explicit source/output target-set comparison"
        ),
        "comparator_outcome_counts": dict(sorted(outcome_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "candidates": results,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "written": OUTPUT.relative_to(ROOT).as_posix(),
                "outcomes": payload["comparator_outcome_counts"],
                "classifications": payload["classification_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
