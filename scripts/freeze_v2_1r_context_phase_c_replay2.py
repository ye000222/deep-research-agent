"""Freeze source and persisted development inputs for Truth-Context Replay #2."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import replay_v2_1r_truth_condition_context as phase_c  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_freeze.json"
EXTRA_INPUTS = {
    "c1_selection": ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json",
    "c1_manual_audit": ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json",
    "c1_shared_reaudit": ROOT
    / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json",
    "c1_closeout": ROOT / "artifacts/v2_1r_truth_condition_context_c1_report.json",
    "phase_c_manual_review": ROOT / "artifacts/v2_1r_truth_condition_context_128_manual_review.md",
}
RUNNER_FILES = (
    "scripts/replay_v2_1r_truth_condition_context_phase_c_replay2.py",
    "scripts/freeze_v2_1r_context_phase_c_replay2.py",
    "tests/unit/test_v2_1r_truth_condition_context_replay2.py",
)
FROZEN_SYMBOLS = (
    "ContextFamilyCoverage",
    "ContextPresenceConclusion",
    "VerificationAssertionView",
    "_SOURCE_DETECTION_COVERAGE",
    "canonical_context_value",
    "normalize_semantic_value",
    "project_source_context_graph",
    "project_output_context_graph",
    "compare_context_graphs",
    "build_verification_assertion_views",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol_hashes() -> dict[str, str]:
    import ast

    module = ROOT / "apps/api/app/domain/truth_condition_context.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = next(
                (
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name) and target.id in FROZEN_SYMBOLS
                ),
                name,
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if isinstance(name, str) and name in FROZEN_SYMBOLS:
            hashes[name] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode("utf-8")
            ).hexdigest()
    if set(hashes) != set(FROZEN_SYMBOLS):
        raise SystemExit(f"frozen symbols missing: {sorted(set(FROZEN_SYMBOLS) - set(hashes))}")
    return hashes


def main() -> None:
    if FREEZE.exists():
        raise SystemExit(f"refusing to overwrite freeze: {FREEZE}")
    input_paths = phase_c.input_artifact_paths()
    missing = [
        str(path) for path in [*input_paths.values(), *EXTRA_INPUTS.values()] if not path.is_file()
    ]
    if missing:
        raise SystemExit(f"required persisted inputs missing: {missing}")
    for output in (
        ROOT / "artifacts/v2_1r_truth_condition_context_128_replay2.json",
        ROOT / "artifacts/v2_1r_truth_condition_context_128_replay2.md",
        ROOT / "artifacts/v2_1r_truth_condition_context_128_replay2_manual_review.json",
    ):
        if output.exists():
            raise SystemExit(f"refusing to overwrite a pre-existing Replay #2 artifact: {output}")
    source_paths = {
        relative: ROOT / relative
        for relative in (
            *phase_c.HASHED_SOURCE_FILES,
            *RUNNER_FILES,
        )
    }
    payload: dict[str, Any] = {
        "freeze_version": "v2_1r-truth-condition-context-phase-c-replay2-freeze-v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "offline replay of the same persisted 128 development outputs; "
            "no model/provider/database/research execution"
        ),
        "source_file_sha256": {key: sha256(path) for key, path in sorted(source_paths.items())},
        "projection_symbol_ast_sha256": symbol_hashes(),
        "input_artifact_sha256": {key: sha256(path) for key, path in sorted(input_paths.items())},
        "review_protocol_artifact_sha256": {
            key: sha256(path) for key, path in sorted(EXTRA_INPUTS.items())
        },
        "cohort_counts_expected": {
            "bounded_shadow": 100,
            "fresh_validation": 12,
            "final_unseen": 16,
        },
        "provider_calls": 0,
        "database_reads_or_writes": False,
        "holdout_content_read": False,
        "outputs_preserved": [
            "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json",
            "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.md",
        ],
        "replay2_outputs_created_separately": [
            "artifacts/v2_1r_truth_condition_context_128_replay2.json",
            "artifacts/v2_1r_truth_condition_context_128_replay2_manual_review.json",
        ],
    }
    FREEZE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "freeze": FREEZE.relative_to(ROOT).as_posix(),
                "input_artifacts": len(input_paths),
                "review_artifacts": len(EXTRA_INPUTS),
                "source_files": len(source_paths),
                "frozen_symbols": len(FROZEN_SYMBOLS),
                "provider_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
