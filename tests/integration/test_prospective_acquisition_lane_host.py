"""Disposable-PostgreSQL integration attestation for the acquisition lane.

All task execution uses an in-process fake profile. This test never constructs
an HTTP transport or accesses canonical task text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from app.core.config import Settings
from app.domain.research_runs import RunKind
from app.infrastructure.db.llm_capability_tests import LLMCapabilityTestRepository
from app.infrastructure.db.models import ProviderProfileRow
from app.infrastructure.db.outbox import TaskDispatchOutboxRepository
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.infrastructure.db.research_runs import (
    ResearchRunNotFoundError,
    ResearchRunRepository,
)
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow, TaskDispatchOutboxRow
from app.main import create_app
from app.security.secrets import SecretCipher, load_or_create_master_key
from app.services.prospective_acquisition_execution_identity import (
    create_execution_context,
    resolve_provider_binding,
)
from app.services.provider_profiles import ProviderProfileService
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text

from scripts.prospective_acquisition.control import ClaimRecord, TaskResult
from scripts.prospective_acquisition.execution_lane import (
    FrozenAcquisitionTask,
    ProspectiveAcquisitionExecutionLane,
)
from scripts.prospective_acquisition.execution_ledger import TaskExecutionLedger
from scripts.prospective_acquisition.profile import BoundedTaskExecution

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PROSPECTIVE_LANE_HOST_ATTESTATION") != "1",
    reason="operator host attestation must explicitly enable disposable PostgreSQL",
)

_OWNER = "d4r-lane-host-attestation"
_CONTEXT_ID = "8862f286-e707-461e-876f-ec57207802e8"
_POPULATION_ID = "v2_1r_d4r_prospective_population_001"
_BATCH_ID = "v2_1r_d4r_prospective_population_001_batch_001"


def _settings(tmp_path: Path) -> Settings:
    raw_url = os.environ.get("TEST_DATABASE_URL", "")
    parsed = urlsplit(raw_url.replace("postgresql+psycopg://", "postgresql://", 1))
    if (
        parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or parsed.port in {5432, 55432}
        or parsed.path.lstrip("/") != "d4r_lane_attestation"
        or os.getenv("PROSPECTIVE_LANE_DISPOSABLE_DB") != "1"
    ):
        raise RuntimeError("DISPOSABLE_DATABASE_GUARD_FAILED")
    return Settings(
        app_env="test",
        database_url=raw_url,
        checkpoint_database_uri=raw_url.replace(
            "postgresql+psycopg://", "postgresql://", 1
        ),
        persist_provider_credentials=True,
        secret_master_key_file=tmp_path / "test-master-key",
        external_probes_enabled=False,
        evidence_input_quality_enabled=False,
    )


def _create_authenticated_normal_run(settings: Settings) -> tuple[UUID, object, UUID]:
    app = create_app(settings)
    with TestClient(app) as client:
        profile_response = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "Disposable acquisition lane attestation",
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://provider.invalid/v1",
                "model": "fake-only-model",
                "api_key": "synthetic-not-a-real-credential",
                "is_default": True,
            },
        )
        assert profile_response.status_code == 201, profile_response.status_code
        profile_data = profile_response.json()
        run_response = client.post(
            "/api/v1/research-runs",
            json={
                "query": "Disposable lane integration fixture",
                "saved_profile_version_id": profile_data["credential_version_id"],
                "budget_tier": "quick",
            },
            headers={"Idempotency-Key": f"lane-host-{uuid4()}"},
        )
        assert run_response.status_code == 202, run_response.status_code
        cookie = client.cookies.get(settings.provider_session_cookie_name)
        assert cookie
        session = app.state.client_sessions.resolve(cookie)
        return UUID(run_response.json()["run_id"]), session, UUID(profile_data["profile_id"])


def test_postgres_lane_isolation_and_fake_execution(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    normal_run_id, authenticated_session, profile_id = _create_authenticated_normal_run(settings)
    database = PostgresRuntime(settings.database_url)
    repository = ResearchRunRepository(database.session_factory)
    outbox = TaskDispatchOutboxRepository(database.session_factory)
    profile_repository = ProviderProfileRepository(database.session_factory)
    profile_service = ProviderProfileService(
        profile_repository,
        SecretCipher(load_or_create_master_key(settings)),
        LLMCapabilityTestRepository(database.session_factory),
        allow_insecure_endpoints=False,
    )
    acquisition_run_ids: list[UUID] = []

    async def exercise() -> None:
        try:
            async with database.session_factory() as session:
                column = await session.scalar(
                    text(
                        "SELECT column_default FROM information_schema.columns "
                        "WHERE table_name='research_runs' AND column_name='run_kind'"
                    )
                )
                assert column and "research" in str(column)
                missing_kind = await session.scalar(
                    text("SELECT count(*) FROM research_runs WHERE run_kind IS NULL")
                )
                assert missing_kind == 0
                normal_row = await session.get(ResearchRunRow, normal_run_id)
                assert normal_row is not None
                assert normal_row.run_kind == RunKind.RESEARCH.value
                budget_snapshot = dict(normal_row.budget_snapshot)

            context = create_execution_context(
                authenticated_session,  # type: ignore[arg-type]
                population_id=_POPULATION_ID,
                batch_id=_BATCH_ID,
                protocol_version="host-attestation-test.v1",
            )
            binding = await resolve_provider_binding(context, profile_service, profile_repository)
            context = replace(context, execution_context_id=_CONTEXT_ID, context_hash="")
            context = context.with_provider_binding(binding)

            fake_claim = ClaimRecord(
                claim_id=str(uuid4()),
                created_at=datetime.now(UTC).isoformat(),
                source_refs=("claim-ref", "evidence-ref", "source-ref", "snapshot-ref"),
                source_key="source-diversity-hash",
                owner_key="source-owner-hash",
                content_hash="a" * 64,
                provenance_complete=True,
                identity_isolation_attested=True,
                evidence_ids=(str(uuid4()),),
                source_ids=(str(uuid4()),),
                snapshot_ids=(str(uuid4()),),
            )
            task = FrozenAcquisitionTask(
                task_id="task-001",
                task_text="Synthetic host attestation task; no research is performed.",
                ordinal=1,
                task_execution_id="b" * 64,
                run_id=uuid4(),
            )
            ledger = TaskExecutionLedger(tmp_path / "temporary-ledger")

            class FakeProfile:
                profile_version = "prospective-acquisition-profile.v1"
                provider_outbound_max_per_task = 1
                search_outbound_max_per_task = 1
                search_results_per_task = 1
                pages_read_per_task = 1
                reader_outbound_max_per_task = 4

                async def execute(self, tasks, **_kwargs):
                    assert ledger.state(task.task_execution_id) == "EXTERNAL_EFFECT_STARTED"
                    assert len(tasks) == 1
                    run_id = tasks[0].run_id
                    execution = BoundedTaskExecution(
                        task_id=task.task_id,
                        run_id=run_id,
                        status="CLAIM_RESULT",
                        result=TaskResult(task.task_id, run_id, (fake_claim,)),
                        provider_outbound_count=0,
                        search_outbound_count=0,
                        reader_outbound_count=0,
                        blocked_extra_outbound_count=0,
                    )
                    return type(
                        "FakeBatchResult",
                        (),
                        {"executions": (execution,)},
                    )()

            lane = ProspectiveAcquisitionExecutionLane(
                repository=repository,
                ledger=ledger,
                profile=FakeProfile(),  # type: ignore[arg-type]
                profile_service=profile_service,
                profile_repository=profile_repository,
                execution_context=context,
                components_ready=True,
            )
            lane._preflight_task_ids = frozenset({task.task_execution_id})
            acquisition_run_ids.append(task.run_id)
            result = await lane.execute_task(task)
            assert result["state"] == "SUCCEEDED"
            receipt_path = Path(str(result["receipt_path"]))
            receipt_text = receipt_path.read_text(encoding="utf-8")
            receipt = json.loads(receipt_text)
            assert fake_claim.claim_id in receipt_text
            assert fake_claim.evidence_ids[0] not in receipt_text
            projection = {
                "claim_id": fake_claim.claim_id,
                "evidence_ids": list(fake_claim.evidence_ids),
                "source_ids": list(fake_claim.source_ids),
                "snapshot_ids": list(fake_claim.snapshot_ids),
                "source_diversity_key": fake_claim.source_key,
                "source_owner_key": fake_claim.owner_key,
                "content_hash": fake_claim.content_hash,
            }
            receipt_reference = receipt["persistence_receipt_reference"][0]
            assert receipt_reference == {
                "claim_id": fake_claim.claim_id,
                "persistence_receipt_sha256": hashlib.sha256(
                    json.dumps(projection, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }

            async with database.session_factory() as session:
                acquisition_row = await session.get(ResearchRunRow, task.run_id)
                assert acquisition_row is not None
                assert acquisition_row.run_kind == RunKind.PROSPECTIVE_ACQUISITION_TASK.value
                assert acquisition_row.status == "completed"
                assert acquisition_row.worker_task_id == f"pa-{task.task_execution_id[:48]}"
                outbox_count = await session.scalar(
                    select(func.count()).select_from(TaskDispatchOutboxRow).where(
                        TaskDispatchOutboxRow.run_id == task.run_id
                    )
                )
                assert outbox_count == 0
                events = (
                    await session.scalars(
                        select(AgentEventRow).where(AgentEventRow.run_id == task.run_id)
                    )
                ).all()
                event_types = {event.event_type for event in events}
                assert "prospective_acquisition.task.started" in event_types
                assert "prospective_acquisition.task.completed" in event_types
                assert not any(value.startswith("report.") for value in event_types)

            normal_claims = await outbox.claim_batch(limit=10)
            assert any(item.run_id == normal_run_id for item in normal_claims)
            assert all(item.run_id != task.run_id for item in normal_claims)
            assert await repository.acquire_for_execution(
                normal_run_id, worker_task_id="normal-host-worker"
            ) is True
            assert await repository.acquire_for_execution(
                task.run_id, worker_task_id="ordinary-worker-must-not-acquire"
            ) is False
            ordinary_runs = await repository.list_recent(context.owner_hash, limit=100)
            assert normal_run_id in {item.run_id for item in ordinary_runs}
            assert task.run_id not in {item.run_id for item in ordinary_runs}
            with pytest.raises(ResearchRunNotFoundError):
                await repository.get(context.owner_hash, task.run_id)

            # Internal persistence lookup remains available by explicit ID.
            async with database.session_factory() as session:
                assert await session.get(ResearchRunRow, task.run_id) is not None

            expired_id = uuid4()
            expired_task_id = "c" * 64
            await repository.create_for_prospective_acquisition(
                context,
                run_id=expired_id,
                worker_task_id="pa-expired-lease-fixture",
                idempotency_key=f"pa-host-expired-{uuid4()}",
                task_text="Synthetic expired lease fixture.",
                task_ordinal=2,
                task_execution_id=expired_task_id,
                budget_snapshot=budget_snapshot,
            )
            acquisition_run_ids.append(expired_id)
            async with database.session_factory() as session, session.begin():
                row = await session.get(ResearchRunRow, expired_id)
                assert row is not None
                row.lease_until = datetime.now(UTC) - timedelta(seconds=5)
            reconciled = await repository.reconcile_expired_leases()
            assert expired_id not in reconciled
            async with database.session_factory() as session:
                row = await session.get(ResearchRunRow, expired_id)
                assert row is not None and row.status == "running"
                assert row.worker_task_id == "pa-expired-lease-fixture"
                expired_outbox_count = await session.scalar(
                    select(func.count()).select_from(TaskDispatchOutboxRow).where(
                        TaskDispatchOutboxRow.run_id == expired_id
                    )
                )
                assert expired_outbox_count == 0
            assert await repository.finalize_prospective_acquisition_task(
                expired_id,
                worker_task_id="pa-expired-lease-fixture",
                success=False,
                safe_failure_code="FAKE_TERMINAL_FAILURE",
            ) is True
            async with database.session_factory() as session:
                failed_row = await session.get(ResearchRunRow, expired_id)
                assert failed_row is not None and failed_row.status == "failed"
                failed_events = (
                    await session.scalars(
                        select(AgentEventRow).where(AgentEventRow.run_id == expired_id)
                    )
                ).all()
                failed_types = {event.event_type for event in failed_events}
                assert "prospective_acquisition.task.failed" in failed_types
                assert not any(value.startswith("report.") for value in failed_types)
        finally:
            async with database.session_factory() as session, session.begin():
                all_ids = [normal_run_id, *acquisition_run_ids]
                await session.execute(
                    delete(ResearchRunRow).where(ResearchRunRow.id.in_(all_ids))
                )
                await session.execute(
                    delete(ProviderProfileRow).where(ProviderProfileRow.id == profile_id)
                )

    try:
        asyncio.run(exercise())
    finally:
        asyncio.run(database.close())
