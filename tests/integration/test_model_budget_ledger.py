"""PostgreSQL integration coverage for the per-attempt model token ledger.

Verifies atomic reservation, idempotent settlement, uncertain (unknown-usage)
conservative holding, explicit release, expired-lease return, and budget
refusal — all against a real transaction boundary.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.core.config import Settings
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import ResearchRunRow
from app.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import func, select

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


def _create_run(settings: Settings) -> tuple[UUID, str]:
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "Ledger integration profile",
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://api.openai.com/v1",
                "model": "ledger-model-v1",
                "api_key": "synthetic-ledger-key-1234",
                "is_default": True,
            },
        )
        assert created.status_code == 201
        profile_id = created.json()["profile_id"]
        switched = client.patch(
            f"/api/v1/llm/profiles/{profile_id}",
            json={
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://api.openai.com/v1",
                "model": "ledger-model-v1",
                "is_default": True,
            },
        )
        assert switched.status_code == 200
        credential_version_id = switched.json()["credential_version_id"]
        accepted = client.post(
            "/api/v1/research-runs",
            json={
                "query": "Verify the model token reservation ledger.",
                "saved_profile_version_id": credential_version_id,
                "budget_tier": "quick",
            },
            headers={"Idempotency-Key": f"ledger-run-{uuid4()}"},
        )
        assert accepted.status_code == 202
        return UUID(accepted.json()["run_id"]), "ledger-worker-1"


async def _set_budget(
    settings: Settings, run_id: UUID, worker: str, max_tokens: int
) -> None:
    database = PostgresRuntime(settings.database_url)
    try:
        run_repo = ResearchRunRepository(database.session_factory)
        assert await run_repo.acquire_for_execution(run_id, worker_task_id=worker)
        async with database.session_factory() as session, session.begin():
            row = await session.get(ResearchRunRow, run_id)
            assert row is not None
            row.budget_snapshot = {**row.budget_snapshot, "max_tokens": max_tokens}
            row.usage_snapshot = {"evidence_total_tokens": 0, "model_tokens": 0}
    finally:
        await database.close()


async def _ledger_exercises(settings: Settings, run_id: UUID, worker: str) -> None:
    database = PostgresRuntime(settings.database_url)
    repo = ResearchToolRepository(database.session_factory)
    try:
        attempt_a = uuid4()
        attempt_b = uuid4()
        attempt_c = uuid4()
        attempt_d = uuid4()
        attempt_e = uuid4()

        # 1. Atomic reservation plus idempotent re-reservation of the same attempt.
        first = await repo.reserve_model_tokens(
            run_id,
            worker_task_id=worker,
            question_id="q1",
            node="evidence_extractor",
            attempt_id=attempt_a,
            estimated_input=3000,
            max_output=1000,
        )
        assert first.granted is True
        assert first.reserved_total == 4000
        repeat = await repo.reserve_model_tokens(
            run_id,
            worker_task_id=worker,
            question_id="q1",
            node="evidence_extractor",
            attempt_id=attempt_a,
            estimated_input=3000,
            max_output=1000,
        )
        assert repeat.granted is True
        assert repeat.reserved_total == 4000

        # Outstanding reservation is reflected in the pre-call budget.
        budget = await repo.evidence_model_budget(
            run_id, worker_task_id=worker, question_id="q1"
        )
        assert budget.remaining_tokens == 20_000 - 4000

        # 2. Idempotent settlement frees the outstanding reservation.
        assert await repo.settle_model_reservation(
            run_id, worker_task_id=worker, attempt_id=attempt_a, actual_total=2500
        ) is True
        assert await repo.settle_model_reservation(
            run_id, worker_task_id=worker, attempt_id=attempt_a, actual_total=2500
        ) is False
        budget = await repo.evidence_model_budget(
            run_id, worker_task_id=worker, question_id="q1"
        )
        assert budget.remaining_tokens == 20_000

        # 3. Unknown-usage failure stays as a conservative uncertain holding.
        assert (
            await repo.reserve_model_tokens(
                run_id,
                worker_task_id=worker,
                question_id="q1",
                node="evidence_extractor",
                attempt_id=attempt_b,
                estimated_input=3000,
                max_output=1000,
            )
        ).granted is True
        assert await repo.mark_model_reservation_uncertain(
            run_id, worker_task_id=worker, attempt_id=attempt_b
        ) is True
        assert await repo.mark_model_reservation_uncertain(
            run_id, worker_task_id=worker, attempt_id=attempt_b
        ) is False
        budget = await repo.evidence_model_budget(
            run_id, worker_task_id=worker, question_id="q1"
        )
        assert budget.remaining_tokens == 20_000 - 4000

        # 4. Budget refusal when the request cannot be afforded.
        denied = await repo.reserve_model_tokens(
            run_id,
            worker_task_id=worker,
            question_id="q1",
            node="evidence_extractor",
            attempt_id=attempt_c,
            estimated_input=18_000,
            max_output=5000,
        )
        assert denied.granted is False
        assert denied.status == "insufficient_budget"

        # 5. Explicit release returns only what never became billable.
        assert (
            await repo.reserve_model_tokens(
                run_id,
                worker_task_id=worker,
                question_id="q1",
                node="evidence_extractor",
                attempt_id=attempt_d,
                estimated_input=1000,
                max_output=500,
            )
        ).granted is True
        assert await repo.release_model_reservation(
            run_id, worker_task_id=worker, attempt_id=attempt_d
        ) is True
        assert await repo.release_model_reservation(
            run_id, worker_task_id=worker, attempt_id=attempt_d
        ) is False
        budget = await repo.evidence_model_budget(
            run_id, worker_task_id=worker, question_id="q1"
        )
        assert budget.remaining_tokens == 20_000 - 4000

        # 6. An expired lease returns its holding until reconciliation.
        expired = await repo.reserve_model_tokens(
            run_id,
            worker_task_id=worker,
            question_id="q1",
            node="evidence_extractor",
            attempt_id=attempt_e,
            estimated_input=1000,
            max_output=500,
            lease_until=datetime.now(UTC) - timedelta(minutes=1),
        )
        assert expired.granted is True
        budget = await repo.evidence_model_budget(
            run_id, worker_task_id=worker, question_id="q1"
        )
        assert budget.remaining_tokens == 20_000 - 4000

        # Persisted row statuses match the ledger transitions.
        async with database.session_factory() as session:
            rows = (
                await session.execute(
                    select(ModelBudgetReservationRow).where(
                        ModelBudgetReservationRow.run_id == run_id
                    )
                )
            ).scalars().all()
        by_attempt = {row.attempt_id: row for row in rows}
        assert by_attempt[attempt_a].status == "settled"
        assert by_attempt[attempt_a].actual_total == 2500
        assert by_attempt[attempt_b].status == "uncertain"
        assert attempt_c not in by_attempt  # refused before any row was created
        assert by_attempt[attempt_d].status == "released"
        assert by_attempt[attempt_e].status == "reserved"  # expired lease only returned funds
    finally:
        await database.close()


def test_model_token_reservation_ledger() -> None:
    settings = _settings()
    run_id, worker = _create_run(settings)
    asyncio.run(_set_budget(settings, run_id, worker, max_tokens=20_000))
    asyncio.run(_ledger_exercises(settings, run_id, worker))


EXTRACTOR_VERSION = "evidence_extractor.v1"


async def _cache_exercises(settings: Settings, run_id: UUID) -> None:
    from app.infrastructure.db.extraction_cache import ExtractionCacheRepository
    from app.infrastructure.db.extraction_cache_models import ExtractionCacheRow

    database = PostgresRuntime(settings.database_url)
    try:
        repo = ExtractionCacheRepository(database.session_factory)
        key = {
            "question_hash": "q" * 64,
            "snapshot_hash": "s" * 64,
            "adapter": "openai_compatible_chat",
            "model": "ledger-model-v1",
            "extractor_version": EXTRACTOR_VERSION,
        }
        evidence = [{"claim": "Cached claim", "exact_quote": "Quoted."}]

        assert await repo.put(run_id, **key, evidence=evidence) is True
        assert await repo.get(run_id, **key) == evidence
        # Idempotent re-put keeps a single row.
        assert await repo.put(run_id, **key, evidence=evidence) is True
        assert await repo.get(run_id, **key) == evidence

        # An empty result is never cached as a durable success.
        assert await repo.put(run_id, **key, evidence=[]) is False
        async with database.session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(ExtractionCacheRow).where(
                    ExtractionCacheRow.run_id == run_id,
                    ExtractionCacheRow.question_hash == key["question_hash"],
                )
            )
        assert int(count or 0) == 1
    finally:
        await database.close()


def test_extraction_cache_persistence() -> None:
    settings = _settings()
    run_id, _ = _create_run(settings)
    asyncio.run(_cache_exercises(settings, run_id))


WRITER_VERSION = "report_writer.v1"


async def _report_draft_cache_exercises(settings: Settings, run_id: UUID) -> None:
    from app.infrastructure.db.report_draft_cache import ReportDraftCacheRepository
    from app.infrastructure.db.report_draft_cache_models import ReportDraftCacheRow

    database = PostgresRuntime(settings.database_url)
    try:
        repo = ReportDraftCacheRepository(database.session_factory)
        draft = {"title": "T", "sections": [{"question_id": "q1"}]}

        assert (
            await repo.put(
                run_id,
                context_hash="c" * 64,
                writer_version=WRITER_VERSION,
                draft=draft,
            )
            is True
        )
        assert (
            await repo.get(run_id, context_hash="c" * 64, writer_version=WRITER_VERSION) == draft
        )
        # Different context hash misses (evidence/config change invalidates).
        assert (
            await repo.get(run_id, context_hash="d" * 64, writer_version=WRITER_VERSION) is None
        )
        # Idempotent put keeps one row per (run, context).
        assert (
            await repo.put(
                run_id,
                context_hash="c" * 64,
                writer_version=WRITER_VERSION,
                draft=draft,
            )
            is True
        )
        assert await repo.put(
            run_id, context_hash="e" * 64, writer_version=WRITER_VERSION, draft={}
        ) is False  # empty draft never cached
        async with database.session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(ReportDraftCacheRow).where(
                    ReportDraftCacheRow.run_id == run_id,
                    ReportDraftCacheRow.context_hash == "c" * 64,
                )
            )
        assert int(count or 0) == 1
    finally:
        await database.close()


def test_report_draft_cache_persistence() -> None:
    settings = _settings()
    run_id, _ = _create_run(settings)
    asyncio.run(_report_draft_cache_exercises(settings, run_id))
