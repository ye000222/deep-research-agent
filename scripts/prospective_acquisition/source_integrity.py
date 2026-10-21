"""Separate frozen generation provenance from current acquisition source identity."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

FINGERPRINT_CONTRACT_VERSION = "acquisition-runtime-source-fingerprint.v1"

# Explicit runtime allow-list derived from the acquisition lane's production
# component factory, persistence path, run-kind isolation, and hard-budget path.
# Tests, docs, and unrelated Research Agent modules are intentionally excluded.
ACQUISITION_RUNTIME_SOURCE_FILES = (
    "apps/api/alembic/versions/20261013_0026_research_run_kind.py",
    "apps/api/app/api/v1/research_runs.py",
    "apps/api/app/context/manager.py",
    "apps/api/app/domain/provider_adapter.py",
    "apps/api/app/domain/providers.py",
    "apps/api/app/domain/research_runs.py",
    "apps/api/app/domain/research_tools.py",
    "apps/api/app/infrastructure/artifacts.py",
    "apps/api/app/infrastructure/db/extraction_cache.py",
    "apps/api/app/infrastructure/db/outbox.py",
    "apps/api/app/infrastructure/db/provider_profiles.py",
    "apps/api/app/infrastructure/db/research_runs.py",
    "apps/api/app/infrastructure/db/research_tools.py",
    "apps/api/app/infrastructure/db/run_models.py",
    "apps/api/app/infrastructure/db/run_providers.py",
    "apps/api/app/llm/adapters.py",
    "apps/api/app/security/secrets.py",
    "apps/api/app/services/evidence_extractor.py",
    "apps/api/app/services/prospective_acquisition_execution_identity.py",
    "apps/api/app/services/provider_fingerprint_contract.py",
    "apps/api/app/services/provider_profiles.py",
    "apps/api/app/services/research_runs.py",
    "apps/api/app/tools/provider_adapters.py",
    "apps/api/app/tools/web_reader.py",
    "apps/api/app/tools/web_search.py",
    "apps/api/app/worker/dispatcher.py",
    "apps/api/app/worker/tasks.py",
    "scripts/operator_verify_prospective_acquisition_lane.ps1",
    "scripts/prospective_acquisition/control.py",
    "scripts/prospective_acquisition/execution_identity.py",
    "scripts/prospective_acquisition/execution_lane.py",
    "scripts/prospective_acquisition/execution_ledger.py",
    "scripts/prospective_acquisition/host_command_diagnostics.ps1",
    "scripts/prospective_acquisition/host_attestation.py",
    "scripts/prospective_acquisition/profile.py",
    "scripts/prospective_acquisition/production.py",
    "scripts/prospective_acquisition/schema_validation.py",
    "scripts/prospective_acquisition/source_integrity.py",
    "scripts/prospective_acquisition/verify_frozen_manifest.py",
)


class AcquisitionRuntimeFingerprintError(RuntimeError):
    """Safe error code for missing, malformed, or drifted runtime identity."""


def _aggregate_fingerprint(files: dict[str, str]) -> str:
    payload = {
        "contract_version": FINGERPRINT_CONTRACT_VERSION,
        "files": {key: files[key] for key in sorted(files)},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_acquisition_runtime(
    *, checkout_root: Path, api_root: Path
) -> dict[str, Any]:
    """Hash the bounded acquisition runtime set using canonical relative names."""
    hashes: dict[str, str] = {}
    for relative in ACQUISITION_RUNTIME_SOURCE_FILES:
        is_api_file = relative.startswith("apps/api/")
        source_root = api_root if is_api_file else checkout_root
        source_relative = relative.removeprefix("apps/api/") if is_api_file else relative
        try:
            content = (source_root / source_relative).read_bytes()
        except OSError as exc:
            raise AcquisitionRuntimeFingerprintError(
                "ACQUISITION_RUNTIME_SOURCE_UNAVAILABLE"
            ) from exc
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return {
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
        "runtime_source_file_count": len(hashes),
        "runtime_source_sha256": hashes,
        "current_acquisition_runtime_fingerprint": _aggregate_fingerprint(hashes),
    }


def validate_runtime_fingerprint_snapshot(
    snapshot: object, *, require_successful_host_attestation: bool = False
) -> dict[str, Any]:
    """Validate an attested snapshot's contract, allow-list, and aggregate."""
    if not isinstance(snapshot, dict):
        raise AcquisitionRuntimeFingerprintError("ACQUISITION_RUNTIME_ATTESTATION_INVALID")
    hashes = snapshot.get("runtime_source_sha256")
    fingerprint = snapshot.get("current_acquisition_runtime_fingerprint")
    if (
        (
            require_successful_host_attestation
            and snapshot.get("status")
            != "PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS"
        )
        or
        snapshot.get("fingerprint_contract_version") != FINGERPRINT_CONTRACT_VERSION
        or snapshot.get("runtime_source_file_count") != len(ACQUISITION_RUNTIME_SOURCE_FILES)
        or not isinstance(hashes, dict)
        or set(hashes) != set(ACQUISITION_RUNTIME_SOURCE_FILES)
        or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
               for value in hashes.values())
        or not isinstance(fingerprint, str)
        or _aggregate_fingerprint(hashes) != fingerprint
    ):
        raise AcquisitionRuntimeFingerprintError("ACQUISITION_RUNTIME_ATTESTATION_INVALID")
    return snapshot


def require_attested_runtime_match(
    attested_snapshot: object, current_snapshot: object
) -> None:
    """Fail closed before effects unless current sources equal the successful attestation."""
    attested = validate_runtime_fingerprint_snapshot(
        attested_snapshot, require_successful_host_attestation=True
    )
    current = validate_runtime_fingerprint_snapshot(current_snapshot)
    if (
        attested["current_acquisition_runtime_fingerprint"]
        != current["current_acquisition_runtime_fingerprint"]
        or attested["runtime_source_sha256"] != current["runtime_source_sha256"]
    ):
        raise AcquisitionRuntimeFingerprintError("ACQUISITION_RUNTIME_FINGERPRINT_MISMATCH")
