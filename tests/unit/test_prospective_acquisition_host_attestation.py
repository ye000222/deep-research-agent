from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from scripts.prospective_acquisition.execution_lane import (
    FrozenAcquisitionTask,
    VerifiedTaskBatch,
)
from scripts.prospective_acquisition.host_attestation import (
    _safe_manifest_summary,
    _verify_frozen_schema_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def test_manifest_summary_is_content_blind_and_counts_unique_identities() -> None:
    batch = VerifiedTaskBatch(
        manifest_sha256="a" * 64,
        tasks=(
            FrozenAcquisitionTask(
                task_id="task-001",
                task_text="must never appear",
                ordinal=1,
                task_execution_id="b" * 64,
                run_id=UUID(int=1),
            ),
            FrozenAcquisitionTask(
                task_id="task-002",
                task_text="also private",
                ordinal=2,
                task_execution_id="c" * 64,
                run_id=UUID(int=2),
            ),
        ),
        verification={},
        source_manifest_path=Path("manifest.json"),
    )

    summary = _safe_manifest_summary(batch)

    assert summary["task_count"] == 2
    assert summary["task_identity_count"] == 2
    assert summary["task_identity_unique"] is True
    assert summary["content_exposed"] is False
    serialized = json.dumps(summary)
    assert "must never appear" not in serialized
    assert "also private" not in serialized


def test_frozen_task_schema_uses_the_expected_canonical_sha() -> None:
    schema_path = ROOT / "artifacts/v2_1r_d4r_a51/task_manifest_schema.json"

    schema_sha256 = _verify_frozen_schema_identity(schema_path)

    assert schema_sha256 == (
        "f908b6397240f5216746edccd26825582223a3d0747b73c9e340c73a1b6d53d3"
    )


def test_frozen_task_schema_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_schema.write_text('{"type":"object"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="FROZEN_TASK_SCHEMA_SHA256_MISMATCH"):
        _verify_frozen_schema_identity(wrong_schema)


def test_missing_frozen_task_schema_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="FROZEN_TASK_SCHEMA_UNREADABLE"):
        _verify_frozen_schema_identity(tmp_path / "missing-schema.json")


def test_host_attestation_mounts_schema_separately_from_artifact_volume() -> None:
    script = ROOT / "scripts/operator_verify_prospective_acquisition_lane.ps1"
    source = script.read_text(encoding="utf-8")

    assert "${canonicalVolume}:/checkout/artifacts:ro" in source
    assert "${schemaRoot}:/opt/artifacts/v2_1r_d4r_a51:ro" in source
    assert "--artifact-root', '/checkout/artifacts'" in source
    assert (
        "--schema-path', '/opt/artifacts/v2_1r_d4r_a51/task_manifest_schema.json'"
        in source
    )


def test_host_operator_persists_current_runtime_fingerprint() -> None:
    script = ROOT / "scripts/operator_verify_prospective_acquisition_lane.ps1"
    source = script.read_text(encoding="utf-8")

    assert (
        "current_acquisition_runtime_fingerprint = "
        "$manifestAttestation.current_acquisition_runtime_fingerprint"
    ) in source
    assert "runtime_source_sha256 = $manifestAttestation.runtime_source_sha256" in source
    assert "host_attestation_$runId.json" in source
    assert (
        "fingerprint_contract_version = $manifestAttestation.fingerprint_contract_version"
        in source
    )
