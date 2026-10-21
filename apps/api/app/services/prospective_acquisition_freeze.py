"""Immutable artifact persistence for the development-only acquisition freeze."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.prospective_acquisition_execution_identity import (
    ExecutionProviderBindingChanged,
    ProspectiveAcquisitionExecutionContext,
)


class FrozenContextConflict(RuntimeError):
    """An immutable population/batch context already exists with other identity."""


@dataclass(frozen=True, slots=True)
class FrozenContextArtifactResult:
    artifact: dict[str, Any]
    created: bool
    path: Path


def _artifact_path(
    artifact_root: Path, population_id: str, batch_id: str
) -> Path:
    key_bytes = json.dumps(
        [population_id, batch_id], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    key = hashlib.sha256(key_bytes).hexdigest()
    return (
        artifact_root
        / "v2_1r_d4r_a6_closure"
        / "execution_contexts"
        / f"{key}.json"
    )


def _reuse_or_reject(
    existing: object,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(existing, dict):
        raise FrozenContextConflict("frozen_execution_context_artifact_invalid")

    if existing.get("provider_binding") != candidate.get("provider_binding"):
        raise ExecutionProviderBindingChanged("EXECUTION_PROVIDER_BINDING_CHANGED")

    immutable_fields = (
        "contract_version",
        "execution_context_id",
        "population_id",
        "batch_id",
        "owner_scope_fingerprint",
        "owner_scope_source",
        "provider_resolution_policy",
        "protocol_version",
        "created_at",
        "context_hash",
    )
    if any(
        not isinstance(existing.get(field), str)
        or not existing[field]
        for field in immutable_fields
    ):
        raise FrozenContextConflict("frozen_execution_context_artifact_invalid")

    if any(
        existing.get(field) != candidate.get(field)
        for field in (
            "contract_version",
            "population_id",
            "batch_id",
            "owner_scope_fingerprint",
            "owner_scope_source",
            "provider_resolution_policy",
            "protocol_version",
        )
    ):
        raise FrozenContextConflict("EXECUTION_CONTEXT_ALREADY_FROZEN_DIFFERENT_IDENTITY")
    return existing


def freeze_context_artifact(
    context: ProspectiveAcquisitionExecutionContext,
    artifact_root: Path,
) -> FrozenContextArtifactResult:
    """Create one safe immutable artifact; identical submissions reuse it."""
    candidate = context.safe_artifact()
    path = _artifact_path(artifact_root, context.population_id, context.batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        candidate,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".freeze-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())

        try:
            os.link(temporary_path, path)
            return FrozenContextArtifactResult(candidate, True, path)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FrozenContextConflict(
                    "frozen_execution_context_artifact_unreadable"
                ) from exc
            return FrozenContextArtifactResult(
                _reuse_or_reject(existing, candidate), False, path
            )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
