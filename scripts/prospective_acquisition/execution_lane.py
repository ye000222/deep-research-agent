"""Bounded, sequential execution lane for frozen prospective-acquisition tasks.

This module is an execution primitive, not an operator. It is deliberately
dependency-injected so preflight and state-machine behavior can be exercised
without Docker, Provider, Search, Reader, or production database access.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid5

from app.domain.adaptive_scheduler import QueryFamily
from app.infrastructure.db.research_tools import ResearchTarget
from app.services.prospective_acquisition_execution_identity import (
    ActiveProfileRepository,
    ProspectiveAcquisitionExecutionContext,
    ProviderProfileLister,
    assert_provider_binding_unchanged,
    validate_frozen_context,
)

from scripts.prospective_acquisition.execution_ledger import (
    TaskExecutionLedger,
)
from scripts.prospective_acquisition.path_c_generation import BATCH_ID, POPULATION_ID
from scripts.prospective_acquisition.profile import (
    AcquisitionTask,
    BoundedProspectiveAcquisitionProfile,
)
from scripts.prospective_acquisition.verify_frozen_manifest import (
    ATTEMPT_ID,
    AUTHORIZATION_ID,
    MANIFEST_SHA256,
    verify_frozen_manifest,
)

_RUN_NAMESPACE = UUID("b781571e-20a4-4b7f-b87b-4f221e0c0378")
_MANIFEST_RELATIVE = Path(
    "v2_1r_d4r_path_c_generation"
) / f"{POPULATION_ID}__{BATCH_ID}" / "attempt_006" / "path_c_task_manifest.json"


def execution_ledger_directory(artifact_root: Path) -> Path:
    return artifact_root / POPULATION_ID / BATCH_ID / "execution"


class AcquisitionLaneBlocked(RuntimeError):
    """Safe, stable preflight or fail-closed execution error code."""


class AcquisitionRunRepository(Protocol):
    async def create_for_prospective_acquisition(
        self,
        execution_context: object,
        *,
        run_id: UUID,
        worker_task_id: str,
        idempotency_key: str,
        task_text: str,
        task_ordinal: int,
        task_execution_id: str,
        budget_snapshot: dict[str, object],
    ) -> tuple[object, bool]: ...

    async def renew_execution_lease(
        self, run_id: UUID, *, worker_task_id: str, lease_seconds: int
    ) -> bool: ...

    async def finalize_prospective_acquisition_task(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        success: bool,
        safe_failure_code: str | None = None,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class FrozenAcquisitionTask:
    task_id: str
    task_text: str
    ordinal: int
    task_execution_id: str
    run_id: UUID


@dataclass(frozen=True, slots=True)
class VerifiedTaskBatch:
    manifest_sha256: str
    tasks: tuple[FrozenAcquisitionTask, ...]
    verification: dict[str, object]
    source_manifest_path: Path


def task_execution_identity(
    *, task_id: str, ordinal: int, manifest_sha256: str = MANIFEST_SHA256
) -> str:
    if ordinal < 1 or not task_id:
        raise ValueError("task_identity_invalid")
    identity = {
        "population_id": POPULATION_ID,
        "batch_id": BATCH_ID,
        "attempt_id": ATTEMPT_ID,
        "authorization_id": AUTHORIZATION_ID,
        "manifest_sha256": manifest_sha256,
        "task_id": task_id,
        "task_ordinal": ordinal,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_verified_task_batch(
    artifact_root: Path,
    *,
    checkout_root: Path,
    api_root: Path,
    schema_path: Path,
) -> VerifiedTaskBatch:
    """Verify the complete frozen manifest before exposing its task texts."""
    result = verify_frozen_manifest(
        artifact_root,
        schema_path,
        expected_manifest_sha256=MANIFEST_SHA256,
        checkout_root=checkout_root,
        api_root=api_root,
    )
    required_true = (
        result.get("status") == "PATH_C_FROZEN_MANIFEST_VERIFIED",
        result.get("manifest_sha256") == MANIFEST_SHA256,
        result.get("task_count") == 40,
        result.get("schema_valid") is True,
        result.get("provenance_valid") is True,
        result.get("task_identity_unique") is True,
        result.get("content_exposed") is False,
        result.get("attempt_id") == ATTEMPT_ID,
        result.get("authorization_id") == AUTHORIZATION_ID,
    )
    if not all(required_true):
        raise AcquisitionLaneBlocked("FROZEN_MANIFEST_PREFLIGHT_FAILED")
    manifest_path = artifact_root / _MANIFEST_RELATIVE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionLaneBlocked("FROZEN_MANIFEST_UNREADABLE") from exc
    raw_tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 40:
        raise AcquisitionLaneBlocked("FROZEN_TASK_COUNT_NOT_EXACTLY_40")
    tasks: list[FrozenAcquisitionTask] = []
    for ordinal, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            raise AcquisitionLaneBlocked("FROZEN_TASK_ITEM_INVALID")
        task_id = item.get("task_id")
        task_text = item.get("task_text")
        expected_id = f"task-{ordinal:03d}"
        if task_id != expected_id or not isinstance(task_text, str) or not task_text.strip():
            raise AcquisitionLaneBlocked("FROZEN_TASK_IDENTITY_INVALID")
        execution_id = task_execution_identity(task_id=task_id, ordinal=ordinal)
        tasks.append(
            FrozenAcquisitionTask(
                task_id=task_id,
                task_text=task_text,
                ordinal=ordinal,
                task_execution_id=execution_id,
                run_id=uuid5(_RUN_NAMESPACE, execution_id),
            )
        )
    return VerifiedTaskBatch(MANIFEST_SHA256, tuple(tasks), result, manifest_path)


class ProspectiveAcquisitionExecutionLane:
    """Resume-safe sequencer around the unchanged bounded production profile."""

    def __init__(
        self,
        *,
        repository: AcquisitionRunRepository,
        ledger: TaskExecutionLedger,
        profile: BoundedProspectiveAcquisitionProfile,
        profile_service: ProviderProfileLister,
        profile_repository: ActiveProfileRepository,
        execution_context: ProspectiveAcquisitionExecutionContext,
        components_ready: bool,
    ) -> None:
        self._repository = repository
        self._ledger = ledger
        self._profile = profile
        self._profile_service = profile_service
        self._profile_repository = profile_repository
        self._context = execution_context
        self._components_ready = components_ready
        self._preflight_task_ids: frozenset[str] = frozenset()

    async def preflight(self, batch: VerifiedTaskBatch) -> None:
        validate_frozen_context(self._context)
        if (
            self._context.population_id != POPULATION_ID
            or self._context.batch_id != BATCH_ID
            or self._context.execution_context_id != "8862f286-e707-461e-876f-ec57207802e8"
        ):
            raise AcquisitionLaneBlocked("EXECUTION_CONTEXT_IDENTITY_MISMATCH")
        if batch.manifest_sha256 != MANIFEST_SHA256 or len(batch.tasks) != 40:
            raise AcquisitionLaneBlocked("FROZEN_MANIFEST_PREFLIGHT_FAILED")
        verification = batch.verification
        if not all(
            (
                verification.get("status") == "PATH_C_FROZEN_MANIFEST_VERIFIED",
                verification.get("attempt_id") == ATTEMPT_ID,
                verification.get("authorization_id") == AUTHORIZATION_ID,
                verification.get("manifest_sha256") == MANIFEST_SHA256,
                verification.get("task_count") == 40,
                verification.get("schema_valid") is True,
                verification.get("provenance_valid") is True,
                verification.get("task_identity_unique") is True,
                verification.get("generation_implementation_provenance_recorded") is True,
                verification.get("content_exposed") is False,
            )
        ):
            raise AcquisitionLaneBlocked("FROZEN_MANIFEST_VERIFICATION_INCOMPLETE")
        try:
            manifest_bytes = batch.source_manifest_path.read_bytes()
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = json.loads(manifest_bytes)
            raw_tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AcquisitionLaneBlocked("FROZEN_MANIFEST_UNREADABLE") from exc
        if (
            manifest_sha != MANIFEST_SHA256
            or not isinstance(raw_tasks, list)
            or len(raw_tasks) != 40
            or any(
                not isinstance(raw, dict)
                or raw.get("task_id") != task.task_id
                or raw.get("task_text") != task.task_text
                for raw, task in zip(raw_tasks, batch.tasks, strict=True)
            )
        ):
            raise AcquisitionLaneBlocked("FROZEN_MANIFEST_CONTENT_CHANGED")
        if not self._components_ready:
            raise AcquisitionLaneBlocked("PRODUCTION_COMPONENTS_NOT_READY")
        if self._profile.profile_version != "prospective-acquisition-profile.v1":
            raise AcquisitionLaneBlocked("BOUNDED_PROFILE_VERSION_MISMATCH")
        if (
            self._profile.provider_outbound_max_per_task != 1
            or self._profile.search_outbound_max_per_task != 1
            or self._profile.search_results_per_task != 1
            or self._profile.pages_read_per_task != 1
            or self._profile.reader_outbound_max_per_task != 4
        ):
            raise AcquisitionLaneBlocked("BOUNDED_PROFILE_CAP_MISMATCH")
        ids = [task.task_execution_id for task in batch.tasks]
        runs = [str(task.run_id) for task in batch.tasks]
        if (
            len(set(ids)) != 40
            or len(set(runs)) != 40
            or any(
                task.task_id != f"task-{index:03d}"
                or task.ordinal != index
                or task.task_execution_id
                != task_execution_identity(task_id=task.task_id, ordinal=index)
                or task.run_id != uuid5(_RUN_NAMESPACE, task.task_execution_id)
                for index, task in enumerate(batch.tasks, start=1)
            )
        ):
            raise AcquisitionLaneBlocked("TASK_EXECUTION_IDENTITY_NOT_UNIQUE")
        for task in batch.tasks:
            self._ledger.state(task.task_execution_id)
        await assert_provider_binding_unchanged(
            self._context, self._profile_service, self._profile_repository
        )
        self._preflight_task_ids = frozenset(task.task_execution_id for task in batch.tasks)

    async def execute_batch(self, batch: VerifiedTaskBatch) -> tuple[dict[str, object], ...]:
        await self.preflight(batch)
        results: list[dict[str, object]] = []
        for task in batch.tasks:
            results.append(await self.execute_task(task))
        return tuple(results)

    async def execute_task(self, task: FrozenAcquisitionTask) -> dict[str, object]:
        """Execute one preflighted task once; callers sequence the frozen batch."""
        if task.task_execution_id not in self._preflight_task_ids:
            raise AcquisitionLaneBlocked("TASK_NOT_PREFLIGHTED")
        state = self._ledger.prepare(
            task.task_execution_id, task_id=task.task_id, task_ordinal=task.ordinal
        )
        if state == "SUCCEEDED":
            return {"task_id": task.task_id, "state": "SUCCEEDED", "skipped": True}
        if state != "PREPARED":
            raise AcquisitionLaneBlocked("TASK_STATE_NOT_EXECUTABLE")
        try:
            await assert_provider_binding_unchanged(
                self._context, self._profile_service, self._profile_repository
            )
        except Exception:
            self._ledger.fail_pre_effect(
                task.task_execution_id,
                safe_failure_code="PROVIDER_BINDING_PREFLIGHT_FAILED",
                counters={
                    "search": 0,
                    "reader": 0,
                    "provider": 0,
                    "http": 0,
                    "production_db_writes": 0,
                },
            )
            raise

        started_at = datetime.now(UTC).isoformat()
        self._ledger.mark_external_effect_started(task.task_execution_id)
        worker_task_id = f"pa-{task.task_execution_id[:48]}"
        target = ResearchTarget(
            plan_version=1,
            question_id=task.task_id,
            question=task.task_text,
            query=task.task_text,
            gap_id=uuid5(_RUN_NAMESPACE, f"{task.task_execution_id}:gap"),
            tool_call_id=uuid5(_RUN_NAMESPACE, f"{task.task_execution_id}:tool"),
            source_id_seed=uuid5(_RUN_NAMESPACE, f"{task.task_execution_id}:source"),
            acceptance_dimensions=(),
            first_pass=True,
            query_family=QueryFamily.SCOPE.value,
            provider_request_allowance=1,
        )
        acquisition_task = AcquisitionTask(
            task_id=task.task_id,
            run_id=str(task.run_id),
            task_text=task.task_text,
            execution_context_id=self._context.execution_context_id,
            research_target=target,
            worker_task_id=worker_task_id,
        )
        budget = self._budget_snapshot()
        # The external-effect marker is durable before the first production DB write.
        await self._repository.create_for_prospective_acquisition(
            self._context,
            run_id=task.run_id,
            worker_task_id=worker_task_id,
            idempotency_key=f"prospective-acquisition:{task.task_execution_id}",
            task_text=task.task_text,
            task_ordinal=task.ordinal,
            task_execution_id=task.task_execution_id,
            budget_snapshot=budget,
        )
        execution = await self._execute_with_heartbeat(acquisition_task)
        success = not execution.status.startswith("FAILED:")
        safe_failure = None if success else "BOUNDED_TASK_EXECUTION_FAILED"
        finalized = await self._repository.finalize_prospective_acquisition_task(
            task.run_id,
            worker_task_id=worker_task_id,
            success=success,
            safe_failure_code=safe_failure,
        )
        if not finalized:
            raise AcquisitionLaneBlocked("ACQUISITION_RUN_FINALIZATION_REJECTED")
        persistence_refs: list[dict[str, str]] = []
        for claim in execution.result.claims:
            receipt_projection = {
                "claim_id": claim.claim_id,
                "evidence_ids": list(claim.evidence_ids),
                "source_ids": list(claim.source_ids),
                "snapshot_ids": list(claim.snapshot_ids),
                "source_diversity_key": claim.source_key,
                "source_owner_key": claim.owner_key,
                "content_hash": claim.content_hash,
            }
            reference_hash = hashlib.sha256(
                json.dumps(
                    receipt_projection, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            persistence_refs.append(
                {"claim_id": claim.claim_id, "persistence_receipt_sha256": reference_hash}
            )
        counters = {
            "provider": execution.provider_outbound_count,
            "search": execution.search_outbound_count,
            "reader": execution.reader_outbound_count,
            "http": execution.provider_outbound_count
            + execution.search_outbound_count
            + execution.reader_outbound_count,
            "run_create_and_terminalization_db_transactions": 2,
            "persistence_receipt_count": len(persistence_refs),
            "blocked_extra_outbound": execution.blocked_extra_outbound_count,
        }
        terminal = "SUCCEEDED" if success else "FAILED_AFTER_EFFECT"
        receipt_path = self._ledger.finish(
            task.task_execution_id,
            terminal_state=terminal,
            receipt={
                "population_id": POPULATION_ID,
                "batch_id": BATCH_ID,
                "attempt_id": ATTEMPT_ID,
                "authorization_id": AUTHORIZATION_ID,
                "manifest_sha256": MANIFEST_SHA256,
                "task_id": task.task_id,
                "task_ordinal": task.ordinal,
                "run_id": str(task.run_id),
                "worker_task_id": worker_task_id,
                "budget_counters": counters,
                "started_at": started_at,
                "terminal_outcome": execution.status,
                "safe_failure_code": safe_failure,
                "persistence_receipt_reference": persistence_refs,
            },
        )
        return {
            "task_id": task.task_id,
            "state": terminal,
            "run_id": str(task.run_id),
            "receipt_path": str(receipt_path),
            "budget_counters": counters,
        }

    async def _execute_with_heartbeat(self, task: AcquisitionTask) -> Any:
        context = self._context
        work = asyncio.create_task(
            self._profile.execute(
                (task,),
                execution_context=context,
                profile_service=self._profile_service,
                profile_repository=self._profile_repository,
            )
        )

        async def heartbeat() -> None:
            while not work.done():
                try:
                    await asyncio.wait_for(work, timeout=300)
                    return
                except TimeoutError:
                    renewed = await self._repository.renew_execution_lease(
                        UUID(task.run_id),
                        worker_task_id=task.worker_task_id or "",
                        lease_seconds=900,
                    )
                    if not renewed:
                        work.cancel()
                        raise AcquisitionLaneBlocked("ACQUISITION_LEASE_LOST") from None

        heart = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait({work, heart}, return_when=asyncio.FIRST_COMPLETED)
            if heart in done:
                error = heart.exception()
                if error is not None:
                    raise error
            result = await work
            if len(result.executions) != 1:
                raise AcquisitionLaneBlocked("BOUNDED_PROFILE_RESULT_CARDINALITY_INVALID")
            execution = result.executions[0]
            if (
                execution.provider_outbound_count > 1
                or execution.search_outbound_count > 1
                or execution.reader_outbound_count > 4
            ):
                raise AcquisitionLaneBlocked("BOUNDED_PROFILE_CAP_VIOLATION")
            return execution
        finally:
            if not heart.done():
                heart.cancel()
            await asyncio.gather(heart, return_exceptions=True)

    @staticmethod
    def _budget_snapshot() -> dict[str, object]:
        return {
            "tier": "prospective_acquisition_bounded.v1",
            "max_iterations": 1,
            "max_searches": 1,
            "max_logical_queries": 1,
            "max_provider_requests": 1,
            "max_pages": 1,
            "max_pages_fetched": 4,
            "max_page_fetch_attempts": 4,
            "max_pages_extracted": 1,
            "max_extraction_calls": 1,
            "max_verification_calls": 0,
            "max_scheduler_actions": 1,
            "max_tokens": 12_000,
            "max_evidence_call_tokens": 12_000,
            "allocation": {
                "planner_tokens": 0,
                "research_tokens": 12_000,
                "verification_tokens": 0,
                "writer_tokens_initial": 0,
                "safety_tokens": 0,
            },
            "max_wall_clock_seconds": 900,
        }
