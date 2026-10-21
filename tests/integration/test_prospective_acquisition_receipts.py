"""PostgreSQL integration proof for bounded prospective acquisition receipts.

These tests are deliberately restricted to the dedicated local test database
configured by ``TEST_DATABASE_URL``. All Search, Reader, and Provider HTTP is
served by ``httpx.MockTransport``; no public network or live Provider is used.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from app.core.config import Settings
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import EvidenceCandidate, ReadPage, ScoredEvidence
from app.domain.research_work import (
    LegacyResearchObjective,
    ResearchWorkKind,
    WorkAttemptState,
    legacy_objective_hash,
    legacy_work_identity,
)
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.llm_capability_tests import LLMCapabilityTestRepository
from app.infrastructure.db.models import ProviderProfileRow
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import (
    IterationEvaluation,
    ResearchTarget,
    ResearchToolRepository,
)
from app.infrastructure.db.research_work import ResearchWorkBindingError, ResearchWorkRepository
from app.infrastructure.db.run_models import ResearchRunRow, ResearchWorkRow, WorkAttemptRow
from app.infrastructure.db.run_providers import RunProviderBindingRepository
from app.main import create_app
from app.security.secrets import SecretCipher, load_or_create_master_key
from app.services.provider_profiles import ProviderProfileService
from app.services.research_loop import ResearchAttemptResult
from app.services.research_orchestrator import ResearchOrchestrator
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scripts.prospective_acquisition.execution_identity import (
    ProspectiveAcquisitionExecutionContext,
    create_execution_context,
    resolve_provider_binding,
)
from scripts.prospective_acquisition.production import ProductionAcquisitionComponentFactory
from scripts.prospective_acquisition.profile import (
    AcquisitionTask,
    BoundedProspectiveAcquisitionProfile,
)

_INTEGRATION_ENABLED = os.getenv("RUN_POSTGRES_INTEGRATION") == "1"
_TEST_URL_PRESENT = bool(os.getenv("TEST_DATABASE_URL"))
pytestmark = pytest.mark.skipif(
    not (_INTEGRATION_ENABLED and _TEST_URL_PRESENT),
    reason=(
        "requires RUN_POSTGRES_INTEGRATION=1 and TEST_DATABASE_URL; "
        "never falls back to DATABASE_URL"
    ),
)

_QUESTION = "What does the source report about the labeled test set?"
_CLAIM = "The test set contains 2,400 labeled images."
_QUOTE = _CLAIM
_PAGE_TEXT = (
    "Published 2026-09-01. Author: Deterministic Test Laboratory. "
    "Methods and dataset details are described in this report. "
    + (_CLAIM + " ") * 20
)
_PAGE_URL = "https://www.nature.com/articles/deterministic-test-fixture"


def _test_settings(tmp_path: Path) -> Settings:
    raw_url = os.environ["TEST_DATABASE_URL"]
    parsed = urlsplit(raw_url.replace("postgresql+psycopg://", "postgresql://", 1))
    # A hard local/database-name check prevents accidentally pointing these
    # write tests at the app's compose or production database.
    if (
        parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 55432
        or parsed.path.lstrip("/") != "d4ra6_test"
    ):
        raise RuntimeError(
            "TEST_DATABASE_URL must target localhost:55432/d4ra6_test"
        )
    checkpoint_url = raw_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return Settings(
        app_env="test",
        database_url=raw_url,
        checkpoint_database_uri=checkpoint_url,
        persist_provider_credentials=True,
        secret_master_key_file=tmp_path / "provider-test-master-key",
        external_probes_enabled=False,
        evidence_input_quality_enabled=False,
    )


def _create_run(settings: Settings) -> tuple[UUID, str, UUID, object]:
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "D4-R-A.6 isolated integration profile",
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://api.openai.com/v1",
                "model": "d4ra6-fake-model",
                "api_key": "synthetic-test-only-not-a-real-key",
                "is_default": True,
            },
        )
        assert created.status_code == 201, created.text
        profile = created.json()
        accepted = client.post(
            "/api/v1/research-runs",
            json={
                "query": _QUESTION,
                "saved_profile_version_id": profile["credential_version_id"],
                "budget_tier": "quick",
            },
            headers={"Idempotency-Key": f"d4ra6-final-{uuid4()}"},
        )
        assert accepted.status_code == 202, accepted.text
        cookie = client.cookies.get(settings.provider_session_cookie_name)
        assert cookie
        session = app.state.client_sessions.resolve(cookie)
        return (
            UUID(accepted.json()["run_id"]),
            "d4ra6-integration-worker",
            UUID(profile["profile_id"]),
            session,
        )


async def _acquire_run(
    settings: Settings,
    run_id: UUID,
    worker_task_id: str,
) -> None:
    database = PostgresRuntime(settings.database_url)
    try:
        acquired = await ResearchRunRepository(database.session_factory).acquire_for_execution(
            run_id, worker_task_id=worker_task_id
        )
        assert acquired is True
    finally:
        await database.close()


async def _cleanup_owned_rows(settings: Settings, run_id: UUID, profile_id: UUID) -> None:
    """Delete only the unique run/profile created by the current test."""
    database = PostgresRuntime(settings.database_url)
    try:
        async with database.session_factory() as session, session.begin():
            await session.execute(delete(ResearchRunRow).where(ResearchRunRow.id == run_id))
            await session.execute(
                delete(ProviderProfileRow).where(ProviderProfileRow.id == profile_id)
            )
    finally:
        await database.close()


def _run_postgres_concurrent_reservation_and_execution_ownership_fencing(
    tmp_path: Path,
) -> None:
    """Real PostgreSQL row locks permit one executor and one recoverable owner."""

    settings = _test_settings(tmp_path)
    run_id, worker, profile_id, _session = _create_run(settings)
    state_ref = f"runs/{run_id}/state/concurrency-test.json"
    state_hash = "a" * 64
    work_objective = LegacyResearchObjective(
        question_id="q1",
        question="A PostgreSQL-only deterministic work reservation fixture.",
        rationale="Exercise the persisted legacy scheduler without external calls.",
        requirements=("fixture requirement",),
        priority=1,
        plan_version=1,
    )
    objective_hash = legacy_objective_hash(work_objective)
    work_ref = legacy_work_identity(
        run_ref=run_id,
        plan_version=1,
        question_id="q1",
        objective_hash=objective_hash,
    )
    second_objective = work_objective.model_copy(
        update={"question_id": "q2", "question": "A second isolated executor fixture."}
    )
    second_objective_hash = legacy_objective_hash(second_objective)
    second_work_ref = legacy_work_identity(
        run_ref=run_id,
        plan_version=1,
        question_id="q2",
        objective_hash=second_objective_hash,
    )
    run_state = SimpleNamespace(
        run_ref=run_id,
        state_ref=state_ref,
        state_hash=state_hash,
        state_version=1,
    )
    database = PostgresRuntime(settings.database_url)
    first_lease: str | None = None
    runner = asyncio.Runner()
    try:
        async def prepare() -> str:
            async with database.session_factory() as session, session.begin():
                run = await session.get(ResearchRunRow, run_id)
                assert run is not None
                run.orchestrator_version = "V3"
                run.plan_version = 1
                run.latest_run_state_ref = state_ref
                run.latest_run_state_hash = state_hash
                run.latest_run_state_version = 1
                session.add(
                    ResearchWorkRow(
                        work_ref=work_ref,
                        run_id=run_id,
                        producer_kind="LEGACY_PLANNER",
                        producer_ref="plan:1:question:q1",
                        work_kind=ResearchWorkKind.LEGACY_RESEARCH.value,
                        objective=work_objective.model_dump(mode="json"),
                        objective_hash=objective_hash,
                        priority=1,
                        constraints={},
                        execution_bounds={"max_attempts": 3},
                        budget_attribution={"scope": "RUN_LEVEL"},
                        source_run_state_ref=state_ref,
                        source_run_state_hash=state_hash,
                        source_run_state_version=1,
                        status="ELIGIBLE",
                        schema_version="research-work-v1",
                    )
                )
                session.add(
                    ResearchWorkRow(
                        work_ref=second_work_ref,
                        run_id=run_id,
                        producer_kind="LEGACY_PLANNER",
                        producer_ref="plan:1:question:q2",
                        work_kind=ResearchWorkKind.LEGACY_RESEARCH.value,
                        objective=second_objective.model_dump(mode="json"),
                        objective_hash=second_objective_hash,
                        priority=1,
                        constraints={},
                        execution_bounds={"max_attempts": 3},
                        budget_attribution={"scope": "RUN_LEVEL"},
                        source_run_state_ref=state_ref,
                        source_run_state_hash=state_hash,
                        source_run_state_version=1,
                        status="ELIGIBLE",
                        schema_version="research-work-v1",
                    )
                )
            acquired = await ResearchRunRepository(
                database.session_factory
            ).acquire_for_execution(run_id, worker_task_id=worker)
            assert acquired is True
            async with database.session_factory() as session:
                run = await session.get(ResearchRunRow, run_id)
                assert run is not None and run.lease_owner
                return run.lease_owner

        first_lease = runner.run(prepare())
        repository = ResearchWorkRepository(database.session_factory)

        async def reserve_both() -> tuple[object, object]:
            return await asyncio.gather(
                repository.reserve_next(
                    run_id,
                    run_state=run_state,  # type: ignore[arg-type]
                    reservation_identity=f"{run_id}:caller-a",
                ),
                repository.reserve_next(
                    run_id,
                    run_state=run_state,  # type: ignore[arg-type]
                    reservation_identity=f"{run_id}:caller-b",
                ),
            )

        first, second = runner.run(reserve_both())
        assert first is not None and second is not None
        attempt_ref = first[1].attempt_ref
        assert second[1].attempt_ref == attempt_ref
        assert first[1].attempt_index == second[1].attempt_index == 1

        class _RollbackTransaction:
            def __init__(self, transaction: object) -> None:
                self.transaction = transaction

            async def __aenter__(self) -> object:
                return await self.transaction.__aenter__()  # type: ignore[attr-defined]

            async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
                if exc_type is not None:
                    return await self.transaction.__aexit__(exc_type, exc, tb)  # type: ignore[attr-defined]
                failure = RuntimeError("simulated_precommit_failure")
                await self.transaction.__aexit__(RuntimeError, failure, None)  # type: ignore[attr-defined]
                raise failure

        class _RollbackSession:
            def __init__(self, session: object) -> None:
                self.session = session

            async def __aenter__(self) -> _RollbackSession:
                await self.session.__aenter__()  # type: ignore[attr-defined]
                return self

            async def __aexit__(self, *args: object) -> object:
                return await self.session.__aexit__(*args)  # type: ignore[attr-defined]

            def begin(self) -> _RollbackTransaction:
                return _RollbackTransaction(self.session.begin())  # type: ignore[attr-defined]

            def __getattr__(self, name: str) -> object:
                return getattr(self.session, name)

        class _RollbackSessionFactory:
            def __call__(self) -> _RollbackSession:
                return _RollbackSession(database.session_factory())

        rollback_repository = ResearchWorkRepository(  # type: ignore[arg-type]
            _RollbackSessionFactory()
        )
        with pytest.raises(RuntimeError, match="simulated_precommit_failure"):
            runner.run(
                rollback_repository.claim_attempt_execution_ownership(
                    run_id,
                    attempt_ref,
                    worker_task_id=worker,
                    run_lease_token=first_lease,
                    execution_owner_token="rolled-back-owner",
                )
            )

        async def claim_both() -> tuple[bool, bool]:
            return await asyncio.gather(
                repository.claim_attempt_execution_ownership(
                    run_id,
                    attempt_ref,
                    worker_task_id=worker,
                    run_lease_token=first_lease,
                    execution_owner_token="caller-a-token",
                ),
                repository.claim_attempt_execution_ownership(
                    run_id,
                    attempt_ref,
                    worker_task_id=worker,
                    run_lease_token=first_lease,
                    execution_owner_token="caller-b-token",
                ),
            )

        claims = runner.run(claim_both())
        assert sum(claims) == 1
        winner = "caller-a-token" if claims[0] else "caller-b-token"
        blocked = "caller-b-token" if claims[0] else "caller-a-token"
        runner.run(
            repository.mark_attempt_executing(
                attempt_ref,
                run_ref=run_id,
                worker_task_id=worker,
                run_lease_token=first_lease,
                execution_owner_token=winner,
            )
        )
        assert runner.run(
            repository.assert_attempt_execution_ownership(
                run_id,
                attempt_ref,
                worker_task_id=worker,
                run_lease_token=first_lease,
                execution_owner_token=winner,
            )
        )
        assert not runner.run(
            repository.assert_attempt_execution_ownership(
                run_id,
                attempt_ref,
                worker_task_id=worker,
                run_lease_token=first_lease,
                execution_owner_token=blocked,
            )
        )

        async def expire_and_reacquire() -> str:
            async with database.session_factory() as session, session.begin():
                run = await session.get(ResearchRunRow, run_id)
                assert run is not None
                run.lease_until = datetime.now(UTC) - timedelta(seconds=1)
            acquired = await ResearchRunRepository(
                database.session_factory
            ).acquire_for_execution(run_id, worker_task_id=worker)
            assert acquired is True
            async with database.session_factory() as session:
                run = await session.get(ResearchRunRow, run_id)
                assert run is not None and run.lease_owner
                return run.lease_owner

        new_lease = runner.run(expire_and_reacquire())
        assert new_lease != first_lease
        async def claim_stale_both() -> tuple[bool, bool]:
            return await asyncio.gather(
                repository.claim_attempt_execution_ownership(
                    run_id,
                    attempt_ref,
                    worker_task_id=worker,
                    run_lease_token=new_lease,
                    execution_owner_token="recovered-b",
                ),
                repository.claim_attempt_execution_ownership(
                    run_id,
                    attempt_ref,
                    worker_task_id=worker,
                    run_lease_token=new_lease,
                    execution_owner_token="recovered-c",
                ),
            )

        takeover = runner.run(claim_stale_both())
        assert sum(takeover) == 1
        recovered_owner = "recovered-b" if takeover[0] else "recovered-c"
        assert not runner.run(
            repository.assert_attempt_execution_ownership(
                run_id,
                attempt_ref,
                worker_task_id=worker,
                run_lease_token=first_lease,
                execution_owner_token=winner,
            )
        )
        with pytest.raises(ResearchWorkBindingError, match="lease_token_mismatch"):
            runner.run(
                repository.settle_attempt(
                    attempt_ref,
                    run_ref=run_id,
                    worker_task_id=worker,
                    run_lease_token=first_lease,
                    execution_owner_token=winner,
                    state=WorkAttemptState.COMPLETED,
                )
            )
        runner.run(
            repository.settle_attempt(
                attempt_ref,
                run_ref=run_id,
                worker_task_id=worker,
                run_lease_token=new_lease,
                execution_owner_token=recovered_owner,
                state=WorkAttemptState.COMPLETED,
            )
        )

        fake_executor_invocations = 0

        class _Runs:
            async def get_run_state_inputs(self, _run_id: UUID) -> dict[str, object]:
                return {"orchestrator_version": "V3", "lease_owner": new_lease}

            async def get_plan_for_execution(self, _run_id: UUID) -> object:
                return object()

        class _States:
            async def load_current(self, _run_id: UUID) -> object:
                return run_state

        class _Work:
            async def materialize_legacy_plan(self, *_args: object) -> tuple[()]:
                return ()

            async def reserve_next(self, *args: object, **kwargs: object) -> object:
                return await repository.reserve_next(*args, **kwargs)  # type: ignore[arg-type]

            async def claim_attempt_execution_ownership(
                self, *args: object, **kwargs: object
            ) -> bool:
                return await repository.claim_attempt_execution_ownership(
                    *args, **kwargs  # type: ignore[arg-type]
                )

            async def assert_attempt_execution_ownership(
                self, *args: object, **kwargs: object
            ) -> bool:
                return await repository.assert_attempt_execution_ownership(
                    *args, **kwargs  # type: ignore[arg-type]
                )

            async def mark_attempt_executing(
                self, *args: object, **kwargs: object
            ) -> object:
                return await repository.mark_attempt_executing(
                    *args, **kwargs  # type: ignore[arg-type]
                )

            async def settle_attempt(
                self, *args: object, **kwargs: object
            ) -> object:
                return await repository.settle_attempt(
                    *args, **kwargs  # type: ignore[arg-type]
                )

        class _Tools:
            async def prepare_target_for_work(
                self, _run_id: UUID, **kwargs: object
            ) -> ResearchTarget:
                item = kwargs["work_item"]
                assert item.objective.question_id == "q2"  # type: ignore[attr-defined]
                return ResearchTarget(
                    plan_version=1,
                    question_id="q2",
                    question="A second isolated executor fixture.",
                    query="offline fake target",
                    gap_id=uuid4(),
                    tool_call_id=uuid4(),
                    source_id_seed=uuid4(),
                )

            async def replan_requested(self, *_args: object, **_kwargs: object) -> bool:
                return False

        class _Loop:
            async def execute_target(
                self, *_args: object, **_kwargs: object
            ) -> ResearchAttemptResult:
                nonlocal fake_executor_invocations
                fake_executor_invocations += 1
                await asyncio.sleep(0)
                return ResearchAttemptResult(
                    pages_read=0,
                    accepted_evidence=0,
                    outcome="no_evidence",
                    evidence_refs=(),
                )

            async def project_work_iteration(
                self, *_args: object, **_kwargs: object
            ) -> IterationEvaluation:
                return IterationEvaluation(
                    continue_research=True,
                    decision="continue_plan",
                    stop_reason=None,
                    question_status="active",
                    coverage=0.0,
                    information_gain=0.0,
                )

        orchestrator = ResearchOrchestrator(
            runs=_Runs(),  # type: ignore[arg-type]
            work=_Work(),  # type: ignore[arg-type]
            run_states=_States(),  # type: ignore[arg-type]
            loop=_Loop(),  # type: ignore[arg-type]
            tools=_Tools(),  # type: ignore[arg-type]
        )

        async def execute_two_callers() -> tuple[object, object]:
            return await asyncio.gather(
                orchestrator.run_next_work(run_id, worker_task_id=worker),
                orchestrator.run_next_work(run_id, worker_task_id=worker),
            )

        results = runner.run(execute_two_callers())
        assert sum(result.decision != "work_already_owned" for result in results) == 1
        assert fake_executor_invocations == 1
        async def final_state() -> tuple[int, str, str]:
            async with database.session_factory() as session:
                count = await session.scalar(
                    select(func.count()).select_from(WorkAttemptRow).where(
                        WorkAttemptRow.work_ref == work_ref
                    )
                )
                work = await session.get(ResearchWorkRow, work_ref)
                attempt = await session.get(WorkAttemptRow, attempt_ref)
                assert work is not None and attempt is not None
                executor_work = await session.get(ResearchWorkRow, second_work_ref)
                executor_attempt_count = await session.scalar(
                    select(func.count()).select_from(WorkAttemptRow).where(
                        WorkAttemptRow.work_ref == second_work_ref
                    )
                )
                assert executor_work is not None
                return (
                    int(count or 0),
                    work.status,
                    attempt.state,
                    int(executor_attempt_count or 0),
                    executor_work.status,
                )

        assert runner.run(final_state()) == (
            1,
            "COMPLETED",
            "COMPLETED",
            1,
            "COMPLETED",
        )
    finally:
        try:
            runner.run(_cleanup_owned_rows(settings, run_id, profile_id))
            runner.run(database.close())
        finally:
            runner.close()


async def _freeze_context(
    settings: Settings,
    session: object,
) -> tuple[
    ProspectiveAcquisitionExecutionContext,
    ProviderProfileService,
    ProviderProfileRepository,
    PostgresRuntime,
]:
    database = PostgresRuntime(settings.database_url)
    profiles = ProviderProfileRepository(database.session_factory)
    service = ProviderProfileService(
        profiles,
        SecretCipher(load_or_create_master_key(settings)),
        LLMCapabilityTestRepository(database.session_factory),
        allow_insecure_endpoints=False,
    )
    context = create_execution_context(
        session,  # type: ignore[arg-type]
        population_id="d4ra6-final-test-population",
        batch_id=f"d4ra6-final-test-batch-{uuid4()}",
        protocol_version="prospective-development-population.v1",
    )
    binding = await resolve_provider_binding(context, service, profiles)
    return context.with_provider_binding(binding), service, profiles, database


def _target(*, source_id: UUID | None = None) -> ResearchTarget:
    return ResearchTarget(
        plan_version=1,
        question_id="q1",
        question=_QUESTION,
        query=_QUESTION,
        gap_id=uuid4(),
        tool_call_id=uuid4(),
        source_id_seed=source_id or uuid4(),
        provider_request_allowance=1,
    )


def _page() -> ReadPage:
    return ReadPage(
        final_url=_PAGE_URL,
        title="Deterministic dataset report",
        clean_text=_PAGE_TEXT,
        content_hash=hashlib.sha256(_PAGE_TEXT.encode()).hexdigest(),
        fetched_at=datetime.now(UTC),
    )


def _accepted_evidence() -> ScoredEvidence:
    return ScoredEvidence(
        candidate=EvidenceCandidate(
            claim=_CLAIM,
            exact_quote=_QUOTE,
            relevance=0.99,
            confidence=0.99,
        ),
        source_reliability=0.9,
        evidence_score=0.95,
        accepted=True,
    )


def _rejected_evidence() -> ScoredEvidence:
    return ScoredEvidence(
        candidate=EvidenceCandidate(
            claim="The report provides additional validation details.",
            exact_quote="This quote is intentionally absent from the persisted page.",
            relevance=0.2,
            confidence=0.2,
        ),
        source_reliability=0.9,
        evidence_score=0.2,
        accepted=False,
        rejection_reason="fixture_rejected",
    )


def _context_manifest() -> dict[str, int | bool]:
    return {
        "source_chars": len(_PAGE_TEXT),
        "selected_chars": len(_PAGE_TEXT),
        "truncated": False,
        "compact_fallback": False,
        "empty_result_rescue": False,
        "transient_retry": False,
        "cache_hit": False,
    }


async def _count_rows(
    sessions: async_sessionmaker[AsyncSession], model: type[object], run_id: UUID
) -> int:
    async with sessions() as session:
        value = await session.scalar(
            select(func.count()).select_from(model).where(model.run_id == run_id)  # type: ignore[attr-defined]
        )
    return int(value or 0)


def test_record_page_receipt_is_committed_and_relations_are_independently_queryable(
    tmp_path: Path,
) -> None:
    settings = _test_settings(tmp_path)
    run_id, worker, profile_id, _ = _create_run(settings)

    async def exercise() -> None:
        database = PostgresRuntime(settings.database_url)
        try:
            target = _target()
            result = await ResearchToolRepository(
                database.session_factory
            ).record_page_with_receipts(
                run_id,
                worker_task_id=worker,
                target=target,
                source_id=target.source_id_seed,
                page=_page(),
                artifact_uri="runs/test/sources/fixture.txt",
                evidence=[_accepted_evidence()],
                usage=TokenUsage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    accuracy=UsageAccuracy.UNAVAILABLE,
                ),
                context_manifest=_context_manifest(),
            )
            assert result.accepted_count == 1
            assert len(result.claim_receipts) == 1
            receipt = result.claim_receipts[0]
            assert receipt.accepted_and_persisted is True
            assert receipt.evidence_ids and receipt.source_ids and receipt.snapshot_ids
            assert receipt.source_provenance_refs
            assert receipt.source_diversity_key
            assert receipt.source_owner_key == "nature.com"

            # Use a fresh session after record_page_with_receipts returned: this
            # proves commit visibility, not merely in-session object creation.
            async with database.session_factory() as session:
                claim = await session.get(ResearchClaimRow, receipt.claim_id)
                evidence = [
                    await session.get(ResearchEvidenceRow, item_id)
                    for item_id in receipt.evidence_ids
                ]
                sources = [
                    await session.get(ResearchSourceRow, item_id)
                    for item_id in receipt.source_ids
                ]
                snapshots = [
                    await session.get(ResearchSourceSnapshotRow, item_id)
                    for item_id in receipt.snapshot_ids
                ]
                chunks = (
                    await session.scalars(
                        select(ResearchSourceChunkRow).where(
                            ResearchSourceChunkRow.snapshot_id.in_(receipt.snapshot_ids)
                        )
                    )
                ).all()
            assert claim is not None and claim.run_id == run_id
            assert all(row is not None for row in evidence + sources + snapshots)
            assert all(row is not None and row.claim_id == claim.id for row in evidence)
            assert all(row is not None and row.source_id in receipt.source_ids for row in evidence)
            assert all(row is not None and row.source_id in receipt.source_ids for row in snapshots)
            assert chunks
        finally:
            await database.close()

    try:
        asyncio.run(_acquire_run(settings, run_id, worker))
        asyncio.run(exercise())
    finally:
        asyncio.run(_cleanup_owned_rows(settings, run_id, profile_id))


def test_record_page_receipt_rollback_emits_no_receipt_and_leaves_no_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _test_settings(tmp_path)
    run_id, worker, profile_id, _ = _create_run(settings)

    async def fail_after_persistence_writes(*args: object, **kwargs: object) -> object:
        raise RuntimeError("deterministic failure before transaction commit")

    monkeypatch.setattr(
        "app.infrastructure.db.research_tools.refresh_question_relations",
        fail_after_persistence_writes,
    )

    async def exercise() -> None:
        database = PostgresRuntime(settings.database_url)
        try:
            target = _target()
            with pytest.raises(RuntimeError, match="deterministic failure"):
                await ResearchToolRepository(database.session_factory).record_page_with_receipts(
                    run_id,
                    worker_task_id=worker,
                    target=target,
                    source_id=target.source_id_seed,
                    page=_page(),
                    artifact_uri="runs/test/sources/rollback-fixture.txt",
                    evidence=[_accepted_evidence()],
                    usage=TokenUsage(
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        accuracy=UsageAccuracy.UNAVAILABLE,
                    ),
                    context_manifest=_context_manifest(),
                )
            # Independent session queries after the failed transaction prove
            # that its source/snapshot/claim/evidence writes did not escape.
            assert await _count_rows(database.session_factory, ResearchSourceRow, run_id) == 0
            assert await _count_rows(
                database.session_factory, ResearchSourceSnapshotRow, run_id
            ) == 0
            assert await _count_rows(database.session_factory, ResearchClaimRow, run_id) == 0
            assert await _count_rows(database.session_factory, ResearchEvidenceRow, run_id) == 0
        finally:
            await database.close()

    try:
        asyncio.run(_acquire_run(settings, run_id, worker))
        asyncio.run(exercise())
    finally:
        asyncio.run(_cleanup_owned_rows(settings, run_id, profile_id))


def test_partial_persistence_receipt_attests_only_committed_accepted_lineage(
    tmp_path: Path,
) -> None:
    settings = _test_settings(tmp_path)
    run_id, worker, profile_id, _ = _create_run(settings)

    async def exercise() -> None:
        database = PostgresRuntime(settings.database_url)
        try:
            await _acquire_run(settings, run_id, worker)
            target = _target()
            result = await ResearchToolRepository(
                database.session_factory
            ).record_page_with_receipts(
                run_id,
                worker_task_id=worker,
                target=target,
                source_id=target.source_id_seed,
                page=_page(),
                artifact_uri="runs/test/sources/partial-fixture.txt",
                evidence=[_accepted_evidence(), _rejected_evidence()],
                usage=TokenUsage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    accuracy=UsageAccuracy.UNAVAILABLE,
                ),
                context_manifest=_context_manifest(),
            )
            assert result.inserted_count == 2
            assert result.accepted_count == 1
            assert len(result.claim_receipts) == 1
            receipt = result.claim_receipts[0]
            async with database.session_factory() as session:
                accepted_rows = (
                    await session.scalars(
                        select(ResearchEvidenceRow).where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.accepted.is_(True),
                        )
                    )
                ).all()
                rejected_rows = (
                    await session.scalars(
                        select(ResearchEvidenceRow).where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.accepted.is_(False),
                        )
                    )
                ).all()
            assert len(accepted_rows) == 1
            assert len(rejected_rows) == 1
            assert set(receipt.evidence_ids) == {accepted_rows[0].id}
            assert rejected_rows[0].id not in receipt.evidence_ids
            assert receipt.claim_id == accepted_rows[0].claim_id
            assert receipt.accepted_and_persisted is True
        finally:
            await database.close()

    try:
        asyncio.run(exercise())
    finally:
        asyncio.run(_cleanup_owned_rows(settings, run_id, profile_id))


def test_bounded_production_path_persists_claim_and_authoritative_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _test_settings(tmp_path)
    run_id, worker, profile_id, authenticated_session = _create_run(settings)

    async def exercise() -> None:
        context, profile_service, profile_repository, database = await _freeze_context(
            settings, authenticated_session
        )
        try:
            target = _target()
            task = AcquisitionTask(
                task_id=f"d4ra6-task-{uuid4()}",
                run_id=str(run_id),
                task_text=target.query,
                execution_context_id=context.execution_context_id,
                research_target=target,
                worker_task_id=worker,
            )
            provider_http: list[str] = []
            search_http: list[str] = []
            reader_http: list[str] = []

            provider_content = json.dumps(
                {
                    "items": [
                        {
                            "claim": _CLAIM,
                            "exact_quote": _QUOTE,
                            "relevance": 0.99,
                            "confidence": 0.99,
                        }
                    ]
                }
            )

            def provider_transport() -> httpx.AsyncBaseTransport:
                def respond(request: httpx.Request) -> httpx.Response:
                    provider_http.append(str(request.url))
                    return httpx.Response(
                        200,
                        json={
                            "id": "fake-provider-response",
                            "choices": [
                                {
                                    "message": {"role": "assistant", "content": provider_content},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 40,
                                "completion_tokens": 30,
                                "total_tokens": 70,
                            },
                        },
                    )

                return httpx.MockTransport(respond)

            def search_transport() -> httpx.AsyncBaseTransport:
                def respond(request: httpx.Request) -> httpx.Response:
                    search_http.append(str(request.url))
                    return httpx.Response(
                        200,
                        json={
                            "results": [
                                {
                                    "url": _PAGE_URL,
                                    "title": "Deterministic dataset report",
                                    "content": _CLAIM,
                                }
                            ],
                            "unresponsive_engines": [],
                        },
                    )

                return httpx.MockTransport(respond)

            def reader_transport() -> httpx.AsyncBaseTransport:
                def respond(request: httpx.Request) -> httpx.Response:
                    reader_http.append(str(request.url))
                    html = (
                        "<html><head><title>Deterministic dataset report</title></head>"
                        "<body><article><h1>Dataset report</h1><p>"
                        + _PAGE_TEXT
                        + "</p></article></body></html>"
                    )
                    return httpx.Response(
                        200,
                        headers={"content-type": "text/html; charset=utf-8"},
                        text=html,
                    )

                return httpx.MockTransport(respond)

            async def bypass_dns_only(_: str) -> None:
                # The Reader's HTTP remains on MockTransport; skip only DNS
                # resolution in the SSRF guard because this fixture host is not
                # intended to resolve or receive real traffic.
                return None

            monkeypatch.setattr("app.tools.web_reader._require_public_destination", bypass_dns_only)
            monkeypatch.setattr(
                "app.services.evidence_extractor.get_settings", lambda: settings
            )

            bindings = RunProviderBindingRepository(database.session_factory)
            repository = ResearchToolRepository(database.session_factory)
            cipher = SecretCipher(load_or_create_master_key(settings))
            factory = ProductionAcquisitionComponentFactory(
                search_base_url="https://search.fixture.invalid",
                bindings=bindings,
                cipher=cipher,
                repository=repository,
                artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
            )
            profile = BoundedProspectiveAcquisitionProfile(
                factory,
                expected_task_count=1,
                transport_factory=provider_transport,
                search_transport_factory=search_transport,
                reader_transport_factory=reader_transport,
            )
            result = await profile.execute(
                [task],
                execution_context=context,
                profile_service=profile_service,
                profile_repository=profile_repository,
            )

            execution = result.executions[0]
            assert execution.status == "CLAIM_RESULT"
            assert execution.result.claims
            assert execution.provider_outbound_count == 1
            assert execution.search_outbound_count == 1
            assert execution.reader_outbound_count == 1
            assert execution.blocked_extra_outbound_count == 0
            assert result.batch_provider_outbound_count == 1
            assert result.batch_search_outbound_count == 1
            assert result.batch_reader_outbound_count == 1
            assert len(provider_http) == len(search_http) == len(reader_http) == 1

            claim_record = execution.result.claims[0]
            assert claim_record.provenance_complete is True
            assert claim_record.identity_isolation_attested is True
            refs = claim_record.source_refs
            assert any(ref.startswith("claim:") for ref in refs)
            assert any(ref.startswith("evidence:") for ref in refs)
            assert any(ref.startswith("source:") for ref in refs)
            assert any(ref.startswith("snapshot:") for ref in refs)

            # Resolve the Claim row, then independently verify its accepted
            # Evidence and persisted Source/Snapshot relationships.
            claim_id = UUID(claim_record.claim_id)
            async with database.session_factory() as session:
                claim = await session.get(ResearchClaimRow, claim_id)
                evidence_rows = (
                    await session.scalars(
                        select(ResearchEvidenceRow).where(
                            ResearchEvidenceRow.claim_id == claim_id,
                            ResearchEvidenceRow.accepted.is_(True),
                        )
                    )
                ).all()
                assert claim is not None
                assert evidence_rows
                for evidence in evidence_rows:
                    source = await session.get(ResearchSourceRow, evidence.source_id)
                    snapshot = await session.get(ResearchSourceSnapshotRow, evidence.snapshot_id)
                    assert source is not None and source.run_id == run_id
                    assert snapshot is not None and snapshot.source_id == source.id
                    assert evidence.claim_id == claim.id
                    assert evidence.exact_quote == _QUOTE
            assert claim_record.source_key
            assert claim_record.owner_key == "nature.com"
        finally:
            await database.close()

    try:
        asyncio.run(_acquire_run(settings, run_id, worker))
        asyncio.run(exercise())
    finally:
        asyncio.run(_cleanup_owned_rows(settings, run_id, profile_id))
