"""Content-blind canonical manifest and lane-preflight attestation helper."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from app.security.client_sessions import ClientSession
from app.services.prospective_acquisition_execution_identity import (
    FrozenProviderBinding,
    create_execution_context,
)

from scripts.prospective_acquisition.execution_lane import (
    ProspectiveAcquisitionExecutionLane,
    VerifiedTaskBatch,
    load_verified_task_batch,
)
from scripts.prospective_acquisition.execution_ledger import TaskExecutionLedger
from scripts.prospective_acquisition.path_c_generation import (
    BATCH_ID,
    POPULATION_ID,
    SCHEMA_SHA256,
)
from scripts.prospective_acquisition.source_integrity import (
    fingerprint_acquisition_runtime,
)
from scripts.prospective_acquisition.verify_frozen_manifest import (
    ATTEMPT_ID,
    AUTHORIZATION_ID,
    MANIFEST_SHA256,
)

_CONTEXT_ID = "8862f286-e707-461e-876f-ec57207802e8"


def _verify_frozen_schema_identity(schema_path: Path) -> str:
    try:
        schema_bytes = schema_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("FROZEN_TASK_SCHEMA_UNREADABLE") from exc
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    if schema_sha256 != SCHEMA_SHA256:
        raise RuntimeError("FROZEN_TASK_SCHEMA_SHA256_MISMATCH")
    return schema_sha256


def _safe_manifest_summary(batch: VerifiedTaskBatch) -> dict[str, object]:
    tasks = batch.tasks
    identities = [str(task.task_execution_id) for task in tasks]
    digest = hashlib.sha256("\n".join(sorted(identities)).encode("ascii")).hexdigest()
    return {
        "manifest_sha256": batch.manifest_sha256,
        "task_count": len(tasks),
        "task_identity_count": len(identities),
        "task_identity_unique": len(set(identities)) == len(identities),
        "task_identity_set_sha256": digest,
        "content_exposed": False,
    }


async def _attest(
    artifact_root: Path, checkout_root: Path, api_root: Path, schema_path: Path
) -> dict[str, object]:
    schema_sha256 = _verify_frozen_schema_identity(schema_path)
    batch = load_verified_task_batch(
        artifact_root,
        checkout_root=checkout_root,
        api_root=api_root,
        schema_path=schema_path,
    )
    context = create_execution_context(
        ClientSession(UUID("8862f286-e707-461e-876f-ec57207802e8"), "a" * 64, False),
        population_id=POPULATION_ID,
        batch_id=BATCH_ID,
        protocol_version="host-attestation-fake-only.v1",
    )
    context = replace(context, execution_context_id=_CONTEXT_ID, context_hash="")
    binding = FrozenProviderBinding(
        profile_id="00000000-0000-4000-8000-000000000001",
        adapter="openai_compatible_chat",
        model="host-attestation-fake-only",
        endpoint_family="https://provider.invalid",
        profile_version=1,
        credential_version_id="00000000-0000-4000-8000-000000000002",
        credential_version=1,
        credential_present=True,
        fingerprint_contract="provider-safe-config-fingerprint.v1",
        provider_safe_config_fingerprint="0" * 64,
    )
    context = context.with_provider_binding(binding)
    profile = SimpleNamespace(
        profile_version="prospective-acquisition-profile.v1",
        provider_outbound_max_per_task=1,
        search_outbound_max_per_task=1,
        search_results_per_task=1,
        pages_read_per_task=1,
        reader_outbound_max_per_task=4,
    )
    with tempfile.TemporaryDirectory(prefix="pa-lane-preflight-") as temp_dir:
        lane = ProspectiveAcquisitionExecutionLane(
            repository=SimpleNamespace(),
            ledger=TaskExecutionLedger(Path(temp_dir) / "ledger"),
            profile=profile,  # type: ignore[arg-type]
            profile_service=SimpleNamespace(),
            profile_repository=SimpleNamespace(),
            execution_context=context,
            components_ready=True,
        )
        # This attestation proves content-blind manifest loading and lane gates.
        # Provider identity is covered by disposable-PostgreSQL integration tests;
        # no Provider resolution or network client is constructed here.
        from unittest.mock import patch

        async def no_provider_access(*_args: object, **_kwargs: object) -> None:
            return None

        with patch(
            "scripts.prospective_acquisition.execution_lane.assert_provider_binding_unchanged",
            new=no_provider_access,
        ):
            await lane.preflight(batch)
    summary = _safe_manifest_summary(batch)
    if (
        summary["manifest_sha256"] != MANIFEST_SHA256
        or summary["task_count"] != 40
        or summary["task_identity_unique"] is not True
    ):
        raise RuntimeError("FROZEN_MANIFEST_LANE_PREFLIGHT_FAILED")
    return {
        "status": "PASS",
        "attempt_id": ATTEMPT_ID,
        "authorization_id": AUTHORIZATION_ID,
        "population_id": POPULATION_ID,
        "batch_id": BATCH_ID,
        **summary,
        "task_schema_sha256": schema_sha256,
        **fingerprint_acquisition_runtime(
            checkout_root=checkout_root, api_root=api_root
        ),
        "provider_calls": 0,
        "search_calls": 0,
        "reader_calls": 0,
        "external_http": 0,
        "database_writes": 0,
        "holdout_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(
            _attest(args.artifact_root, args.checkout_root, args.api_root, args.schema_path)
        )
    except Exception as exc:
        safe_code = str(exc) if str(exc).isupper() and " " not in str(exc) else "ATTESTATION_FAILED"
        result = {
            "status": "BLOCKED",
            "safe_error_code": safe_code,
            "provider_calls": 0,
            "search_calls": 0,
            "reader_calls": 0,
            "external_http": 0,
            "database_writes": 0,
            "holdout_access": 0,
        }
        print(json.dumps(result, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
