import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from app.core.config import Settings
from app.domain.identifiers import uuid7
from app.domain.planning import ResearchPlan
from app.domain.providers import TokenUsage, UsageAccuracy
from app.infrastructure.db.memory_models import MemoryItemRow
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow
from app.main import create_app
from app.memory.manager import ResearchMemoryManager
from app.retrieval.projections import rebuild_memory
from fastapi.testclient import TestClient
from sqlalchemy import select, update

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)


def _settings() -> Settings:
    database_url = os.environ["TEST_DATABASE_URL"]
    return Settings(
        app_env="test",
        database_url=database_url,
        checkpoint_database_uri=os.getenv(
            "CHECKPOINT_DATABASE_URI",
            database_url.replace("postgresql+psycopg://", "postgresql://"),
        ),
        persist_provider_credentials=True,
        secret_master_key_file=Path("artifacts/.secrets/provider_integration_master_key"),
        external_probes_enabled=False,
    )


async def _exercise_worker_lease(settings: Settings, run_id: UUID) -> None:
    database = PostgresRuntime(settings.database_url)
    repository = ResearchRunRepository(database.session_factory)
    try:
        assert await repository.acquire_for_execution(
            run_id,
            worker_task_id="integration-worker-1",
        )
        assert not await repository.acquire_for_execution(
            run_id,
            worker_task_id="integration-worker-duplicate",
        )
        await repository.interrupt_for_pending_planner(
            run_id,
            worker_task_id="integration-worker-1",
        )
    finally:
        await database.close()


async def _exercise_plan_persistence(settings: Settings, run_id: UUID) -> None:
    database = PostgresRuntime(settings.database_url)
    repository = ResearchRunRepository(database.session_factory)
    plan = ResearchPlan.model_validate(
        {
            "goal": "Validate persisted planning",
            "scope_summary": "Cover five independent validation dimensions.",
            "questions": [
                {
                    "id": f"q{index}",
                    "question": f"What evidence validates dimension {index}?",
                    "priority": 1,
                    "rationale": "Required for the integration acceptance criteria.",
                    "evidence_requirements": ["Two independent sources"],
                    "search_hints": [f"dimension {index}"],
                }
                for index in range(1, 6)
            ],
            "completion_criteria": ["All questions covered", "Sources cross-validated"],
        }
    )
    try:
        assert await repository.acquire_for_execution(
            run_id,
            worker_task_id="integration-planner-1",
        )
        assert await repository.save_generated_plan(
            run_id,
            worker_task_id="integration-planner-1",
            plan=plan,
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                accuracy=UsageAccuracy.EXACT,
            ),
        )
    finally:
        await database.close()


async def _exercise_memory_hybrid_retrieval(settings: Settings, run_id: UUID) -> None:
    database = PostgresRuntime(settings.database_url)
    manager = ResearchMemoryManager(database.session_factory)
    now = datetime.now(UTC)
    try:
        async with database.session_factory() as session, session.begin():
            run = await session.get(ResearchRunRow, run_id)
            assert run is not None
            session.add_all(
                [
                    MemoryItemRow(
                        id=uuid7(),
                        origin_run_id=run_id,
                        owner_hash=run.owner_hash,
                        scope_type="run",
                        scope_id=str(run_id),
                        memory_type="semantic",
                        content_summary="工业视觉缺陷检测正在验证多模态视觉语言模型.",
                        source_ref_ids=["evidence:integration-relevant"],
                        keywords=["工业视觉", "缺陷检测", "多模态"],
                        confidence=0.95,
                        importance=0.9,
                        fingerprint="a" * 64,
                        status="active",
                        access_count=0,
                        utility_count=0,
                        created_at=now,
                        updated_at=now,
                    ),
                    MemoryItemRow(
                        id=uuid7(),
                        origin_run_id=run_id,
                        owner_hash=run.owner_hash,
                        scope_type="run",
                        scope_id=str(run_id),
                        memory_type="semantic",
                        content_summary="农业供应链研究关注季节性库存.",
                        source_ref_ids=["evidence:integration-irrelevant"],
                        keywords=["农业", "供应链"],
                        confidence=0.95,
                        importance=0.9,
                        fingerprint="b" * 64,
                        status="active",
                        access_count=0,
                        utility_count=0,
                        created_at=now,
                        updated_at=now,
                    ),
                ]
            )
            await session.flush()
            assert await rebuild_memory(session, owner_hash=run.owner_hash) >= 2

        result = await manager.retrieve_for_run(
            run_id,
            query="工业视觉 缺陷检测 多模态",
            top_k=2,
        )
        assert result.result == "hit"
        assert result.items
        assert "工业视觉缺陷检测" in result.items[0].content_summary
    finally:
        await database.close()


async def _exercise_expired_worker_redelivery(settings: Settings, run_id: UUID) -> None:
    database = PostgresRuntime(settings.database_url)
    repository = ResearchRunRepository(database.session_factory)
    try:
        assert await repository.acquire_for_execution(
            run_id,
            worker_task_id="crashed-worker",
            lease_seconds=300,
        )
        async with database.session_factory() as session, session.begin():
            await session.execute(
                update(ResearchRunRow)
                .where(ResearchRunRow.id == run_id)
                .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
            )
        assert await repository.acquire_for_execution(
            run_id,
            worker_task_id="redelivered-worker",
            lease_seconds=300,
        )
        assert not await repository.acquire_for_execution(
            run_id,
            worker_task_id="duplicate-redelivery",
            lease_seconds=300,
        )
        await repository.fail_execution(
            run_id,
            worker_task_id="crashed-worker",
            error_code="STALE_WORKER_MUST_NOT_COMMIT",
        )
        async with database.session_factory() as session:
            row = await session.get(ResearchRunRow, run_id)
            assert row is not None
            assert row.status == "running"
            assert row.worker_task_id == "redelivered-worker"
            event_types = list(
                await session.scalars(
                    select(AgentEventRow.event_type)
                    .where(AgentEventRow.run_id == run_id)
                    .order_by(AgentEventRow.run_seq)
                )
            )
            assert event_types == ["run.created", "run.started", "run.recovered"]
    finally:
        await database.close()


def test_profile_and_research_run_survive_restart_with_replayable_events() -> None:
    settings = _settings()
    cookie_name = settings.provider_session_cookie_name
    profile_id: str
    signed_session: str

    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "Integration profile",
                "adapter_type": "openai_responses",
                "base_url": "https://api.openai.com/v1",
                "model": "integration-model-v1",
                "api_key": "synthetic-integration-key-1234",
                "is_default": True,
            },
        )
        assert created.status_code == 201
        payload = created.json()
        assert payload["model"] == "integration-model-v1"
        assert payload["credential_last_four"] == "1234"
        assert "api_key" not in payload
        profile_id = payload["profile_id"]
        signed_session = client.cookies[cookie_name]

    with TestClient(create_app(settings)) as restarted:
        restarted.cookies.set(cookie_name, signed_session)
        restored = restarted.get("/api/v1/llm/profiles")
        assert restored.status_code == 200
        profiles = restored.json()
        assert len(profiles) == 1
        assert profiles[0]["profile_id"] == profile_id
        assert profiles[0]["model"] == "integration-model-v1"
        assert profiles[0]["credential_last_four"] == "1234"
        assert "api_key" not in profiles[0]

        switched = restarted.patch(
            f"/api/v1/llm/profiles/{profile_id}",
            json={
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://api.openai.com/v1",
                "model": "integration-model-v1",
                "is_default": True,
            },
        )
        assert switched.status_code == 200
        switched_profile = switched.json()
        assert switched_profile["adapter_type"] == "openai_compatible_chat"
        assert switched_profile["credential_version"] == 2
        assert switched_profile["credential_last_four"] == "1234"
        assert "synthetic-integration-key" not in switched.text

        run_request = {
            "query": "  Verify   the empty research run lifecycle.  ",
            "saved_profile_version_id": switched_profile["credential_version_id"],
            "budget_tier": "quick",
        }
        run_headers = {"Idempotency-Key": "integration-empty-run-v1"}
        accepted = restarted.post(
            "/api/v1/research-runs",
            json=run_request,
            headers=run_headers,
        )
        assert accepted.status_code == 202
        run = accepted.json()
        assert run["normalized_goal"] == "Verify the empty research run lifecycle."
        assert run["status"] == "queued"
        assert run["next_event_seq"] == 2
        assert run["llm_config_snapshot"].get("base_url") == "https://api.openai.com/v1"
        assert run["llm_config_snapshot"].get("adapter_type") == "openai_compatible_chat"
        assert "api_key" not in str(run)
        run_id = run["run_id"]

        replayed_create = restarted.post(
            "/api/v1/research-runs",
            json=run_request,
            headers=run_headers,
        )
        assert replayed_create.status_code == 202
        assert replayed_create.json()["run_id"] == run_id

        first_replay = restarted.get(f"/api/v1/research-runs/{run_id}/events?follow=false")
        assert first_replay.status_code == 200
        assert "id: 1" in first_replay.text
        assert "event: run.created" in first_replay.text

        cancelled = restarted.post(f"/api/v1/research-runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert cancelled.json()["state_version"] == 2

        after_one = restarted.get(
            f"/api/v1/research-runs/{run_id}/events?follow=false",
            headers={"Last-Event-ID": "1"},
        )
        assert "id: 1" not in after_one.text
        assert "id: 2" in after_one.text
        assert "event: run.cancelled" in after_one.text

        resumed = restarted.post(f"/api/v1/research-runs/{run_id}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "queued"
        assert resumed.json()["state_version"] == 3

        after_two = restarted.get(
            f"/api/v1/research-runs/{run_id}/events?follow=false",
            headers={"Last-Event-ID": "2"},
        )
        assert "id: 2" not in after_two.text
        assert "id: 3" in after_two.text
        assert "event: run.status_changed" in after_two.text

        worker_run = restarted.post(
            "/api/v1/research-runs",
            json={**run_request, "query": "Exercise the Worker lease."},
            headers={"Idempotency-Key": "integration-worker-run-v1"},
        )
        assert worker_run.status_code == 202
        worker_run_id = worker_run.json()["run_id"]
        asyncio.run(_exercise_worker_lease(settings, UUID(worker_run_id)))

        worker_status = restarted.get(f"/api/v1/research-runs/{worker_run_id}")
        assert worker_status.status_code == 200
        assert worker_status.json()["status"] == "interrupted"
        assert worker_status.json()["termination_reason"] == "planner_not_implemented"

        worker_events = restarted.get(f"/api/v1/research-runs/{worker_run_id}/events?follow=false")
        assert "event: run.started" in worker_events.text
        assert "event: run.interrupted" in worker_events.text

        planner_run = restarted.post(
            "/api/v1/research-runs",
            json={**run_request, "query": "Persist a generated research plan."},
            headers={"Idempotency-Key": "integration-planner-run-v1"},
        )
        assert planner_run.status_code == 202
        planner_run_id = planner_run.json()["run_id"]
        asyncio.run(_exercise_plan_persistence(settings, UUID(planner_run_id)))

        planner_status = restarted.get(f"/api/v1/research-runs/{planner_run_id}")
        assert planner_status.status_code == 200
        assert planner_status.json()["phase"] == "researching"
        assert planner_status.json()["plan_version"] == 1
        assert planner_status.json()["termination_reason"] is None
        assert planner_status.json()["usage_snapshot"]["planner"]["total_tokens"] == 30

        persisted_plan = restarted.get(f"/api/v1/research-runs/{planner_run_id}/plan")
        assert persisted_plan.status_code == 200
        assert len(persisted_plan.json()["questions"]) == 5
        assert persisted_plan.json()["questions"][0]["id"] == "q1"

        asyncio.run(_exercise_memory_hybrid_retrieval(settings, UUID(planner_run_id)))

        planner_events = restarted.get(
            f"/api/v1/research-runs/{planner_run_id}/events?follow=false"
        )
        assert "event: plan.generated" in planner_events.text
        assert "event: run.interrupted" not in planner_events.text

        redelivery_run = restarted.post(
            "/api/v1/research-runs",
            json={**run_request, "query": "Recover from an expired Celery worker lease."},
            headers={"Idempotency-Key": "integration-redelivery-run-v1"},
        )
        assert redelivery_run.status_code == 202
        asyncio.run(
            _exercise_expired_worker_redelivery(
                settings,
                UUID(redelivery_run.json()["run_id"]),
            )
        )

        deleted = restarted.delete(f"/api/v1/llm/profiles/{profile_id}")
        assert deleted.status_code == 204
