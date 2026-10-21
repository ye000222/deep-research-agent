"""Record the narrowly scoped analyzer compatibility amendment to the Replay #2 freeze."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import replay_v2_1r_truth_condition_context as phase_c

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_freeze.json"
OUTPUT = ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_compat_repair.json"
ALLOWED_FILES = (
    "scripts/replay_v2_1r_truth_condition_context_phase_c_replay2.py",
    "scripts/truth_condition_context_replay2_view_adapter.py",
    "tests/unit/test_v2_1r_truth_condition_context_replay2.py",
)
EXTRA_INPUTS = {
    "c1_selection": ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json",
    "c1_manual_audit": ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json",
    "c1_shared_reaudit": ROOT
    / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json",
    "c1_closeout": ROOT / "artifacts/v2_1r_truth_condition_context_c1_report.json",
    "phase_c_manual_review": ROOT / "artifacts/v2_1r_truth_condition_context_128_manual_review.md",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite compatibility attestation: {OUTPUT}")
    frozen = phase_c.load_json(FREEZE)
    repairable = {ALLOWED_FILES[0], ALLOWED_FILES[2]}
    current_source = {relative: sha(ROOT / relative) for relative in frozen["source_file_sha256"]}
    preserved = {
        relative: current_source[relative] == expected
        for relative, expected in frozen["source_file_sha256"].items()
        if relative not in repairable
    }
    current_inputs = {
        key: sha(path) for key, path in sorted(phase_c.input_artifact_paths().items())
    }
    current_reviews = {key: sha(path) for key, path in sorted(EXTRA_INPUTS.items())}
    if not all(preserved.values()):
        raise SystemExit(
            f"semantic/source freeze changed; no repair attestation written: {preserved}"
        )
    if current_inputs != frozen["input_artifact_sha256"]:
        raise SystemExit(
            "one or more frozen input artifacts changed; no repair attestation written"
        )
    if current_reviews != frozen["review_protocol_artifact_sha256"]:
        raise SystemExit(
            "one or more frozen review artifacts changed; no repair attestation written"
        )

    payload: dict[str, Any] = {
        "attestation_version": "replay2-verification-view-compat-repair-v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "base_freeze_sha256": sha(FREEZE),
        "base_freeze_path": FREEZE.relative_to(ROOT).as_posix(),
        "original_failed_attempt": {
            "classification": "SUMMARY_SCHEMA_COMPATIBILITY_FAILURE",
            "valid_for_semantic_decision": False,
            "result_artifact_persisted": False,
        },
        "field_semantics": {
            "classification": "ONE_FIELD_SPLIT_INTO_MULTIPLE_DIMENSIONS",
            "context_completeness": (
                "per-view preservation/unknown assessment derived from source graph, "
                "comparison outcomes and material unknowns"
            ),
            "source_context_completeness": (
                "source detector coverage/completeness contract from the current "
                "VerificationAssertionView"
            ),
            "equivalent": False,
        },
        "analyzer_only_files": list(ALLOWED_FILES),
        "repaired_file_sha256": {relative: sha(ROOT / relative) for relative in ALLOWED_FILES},
        "original_frozen_analyzer_sha256": {
            relative: frozen["source_file_sha256"].get(relative) for relative in repairable
        },
        "semantic_source_hashes_preserved": preserved,
        "input_artifacts_preserved": True,
        "review_protocol_artifacts_preserved": True,
        "semantic_behavior_change": False,
        "provider_calls": 0,
        "attestation_tool": Path(__file__).relative_to(ROOT).as_posix(),
        "attestation_tool_sha256": sha(Path(__file__).resolve()),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "compat_repair": OUTPUT.relative_to(ROOT).as_posix(),
                "semantic_files_unchanged": all(preserved.values()),
                "provider_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
