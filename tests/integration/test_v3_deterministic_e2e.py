"""Deterministic V3 orchestration E2E over real services and repositories.

The test database is an in-memory SQLite database driven through a tiny async
session facade. Production repositories, the V3 orchestrator, ResearchLoop,
evidence persistence, semantic shadow pipeline and RunState publisher remain
the exercised implementations; only external boundaries are deterministic fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import traceback
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import app.infrastructure.db.memory_models  # noqa: F401
import pytest
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_runs import RunPhase, RunStatus
from app.domain.research_tools import ReadPage, SearchResult
from app.domain.research_work import ResearchWorkStatus, WorkAttemptState
from app.domain.semantic_runtime_budget import SemanticRuntimeBudget
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.base import Base
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.research_work import (
    ResearchWorkBindingError,
    ResearchWorkRepository,
)
from app.infrastructure.db.run_models import (
    AgentEventRow,
    ResearchPlanItemRow,
    ResearchRunRow,
    ResearchWorkRow,
    WorkAttemptRow,
)
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.research_loop import ResearchLoopService
from app.services.research_orchestrator import ResearchOrchestrator
from app.services.run_state_publisher import RunStatePublisher
from app.services.run_state_reducer import RunStateReducer
from app.services.semantic_engine import V23DeterministicSemanticEngine
from app.services.semantic_research_shadow import (
    SemanticResearchShadowPipeline,
    SemanticShadowMode,
)
from pydantic import SecretStr
from sqlalchemy import DateTime, create_engine
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import Executable
from sqlalchemy.types import TypeDecorator


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type: JSONB, _compiler: Any, **_kw: Any) -> str:
    return "JSON"


@compiles(TSVECTOR, "sqlite")
def _compile_tsvector_for_sqlite(_type: TSVECTOR, _compiler: Any, **_kw: Any) -> str:
    return "TEXT"


class _Transaction:
    def __init__(self, session: _AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> _AsyncSession:
        self._session._sync.begin()
        return self._session

    async def __aexit__(self, exc_type: object, *_args: object) -> None:
        if exc_type is None:
            self._session._sync.commit()
        else:
            self._session._sync.rollback()


class _AsyncSession:
    """Synchronous SQLite operations exposed with AsyncSession's used surface."""

    def __init__(self, sync_session: Session) -> None:
        self._sync = sync_session

    async def __aenter__(self) -> _AsyncSession:
        return self

    async def __aexit__(self, exc_type: object, *_args: object) -> None:
        if exc_type is not None:
            self._sync.rollback()
        self._sync.close()

    def begin(self) -> _Transaction:
        return _Transaction(self)

    async def scalar(self, statement: Executable) -> Any:
        return self._sync.scalar(statement)

    async def scalars(self, statement: Executable) -> Any:
        return self._sync.scalars(statement)

    async def execute(self, statement: Executable) -> Any:
        return self._sync.execute(statement)

    async def get(self, model: type[Any], key: object) -> Any:
        return self._sync.get(model, key)

    def add(self, row: object) -> None:
        self._sync.add(row)

    async def flush(self) -> None:
        self._sync.flush()


class _AsyncSessionFactory:
    def __init__(self, sync_factory: sessionmaker[Session]) -> None:
        self._sync_factory = sync_factory

    def __call__(self) -> _AsyncSession:
        return _AsyncSession(self._sync_factory())


class _SQLiteE2ESession(Session):
    """Supply PostgreSQL's generated AgentEvent identity only in this fixture."""

    def __init__(
        self,
        *args: Any,
        identity_sequence: Iterator[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._agent_event_identity_sequence = identity_sequence

    def flush(self, objects: Any = None) -> None:
        for row in self.new:
            if isinstance(row, AgentEventRow) and row.global_id is None:
                row.global_id = next(self._agent_event_identity_sequence)
        super().flush(objects)


class _SQLiteUTCDateTime(TypeDecorator[datetime]):
    """Preserve PostgreSQL timestamptz semantics in this SQLite-only test."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: datetime | None, _dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, _dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


@pytest.fixture(scope="module", autouse=True)
def _sqlite_utc_datetime_compatibility() -> Iterator[None]:
    """Limit timezone emulation to this E2E module's SQLite metadata lifetime."""

    columns = (
        ModelBudgetReservationRow.__table__.c.lease_until,
        ResearchRunRow.__table__.c.lease_until,
    )
    original_types = tuple(column.type for column in columns)
    for column in columns:
        column.type = _SQLiteUTCDateTime()
    try:
        yield
    finally:
        for column, original_type in zip(columns, original_types, strict=True):
            column.type = original_type


def _drive(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _plan(goal: str) -> ResearchPlan:
    return ResearchPlan(
        goal=goal,
        scope_summary="Find directly quoted evidence for a deterministic research question.",
        questions=[
            ResearchQuestion(
                id=f"q{index}",
                question=(
                    "What statement does the source make about the example program? "
                    f"Question {index}."
                ),
                priority=1,
                rationale="Record the source-backed statement for this question.",
                evidence_requirements=["A directly quoted statement from the source"],
                search_hints=["example program source statement"],
            )
            for index in range(1, 6)
        ],
        completion_criteria=["A direct source statement is recorded", "Evidence is persisted"],
    )


class _Search:
    provider_name = "V3DeterministicFake"
    last_request_count = 1
    last_timeout_count = 0
    last_fallback_count = 0
    last_healthy_response_count = 1
    last_unresponsive_response_count = 0
    last_productive_response_count = 1

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, *, limit: int = 8, **_kwargs: Any) -> list[SearchResult]:
        self.calls.append(query)
        return [
            SearchResult(
                title="Deterministic source",
                url=(
                    "https://example.org/v3-deterministic-source/"
                    f"{hashlib.sha256(query.encode()).hexdigest()[:16]}"
                ),
                snippet="The example program produced a documented result.",
                rank=1,
            )
        ][:limit]


class _Reader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def read(self, url: str, **_kwargs: Any) -> ReadPage:
        self.calls.append(url)
        content = (
            "The example program produced a documented result. "
            "This source statement is provided as deterministic test evidence."
        )
        return ReadPage(
            final_url=url,
            title="Deterministic source",
            clean_text=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


class _Gateway:
    """Deterministic model boundary; calls still pass production prompt/parser code."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_structured(self, *, request: Any, **_kwargs: Any) -> CanonicalModelResult:
        if request.task_kind == "atomic_proposition_shadow_decomposition":
            self.calls.append(request.task_kind)
            return CanonicalModelResult(
                parsed_object={
                    "schema_version": "atomic-proposition-schema-v3",
                    "decomposition_status": "single_proposition",
                    "propositions": [
                        {
                            "local_ref": "p0",
                            "proposition_text": "The example program produced a documented result.",
                            "subject": "example program",
                            "predicate": "produced",
                            "typed_value": {
                                "kind": "entity",
                                "text": "a documented result",
                            },
                            "polarity": "asserted",
                            "source_span": "The example program produced a documented result.",
                        }
                    ],
                },
                usage=TokenUsage(
                    input_tokens=12, output_tokens=8, total_tokens=20, accuracy=UsageAccuracy.EXACT
                ),
                provider_request_id="v3-deterministic-fake",
            )
        if request.task_kind in {"evidence_extraction", "evidence_extraction_compact"}:
            self.calls.append(request.task_kind)
            content = "\n".join(
                part.value
                if isinstance(part.value, str)
                else json.dumps(part.value, sort_keys=True)
                for part in request.content_parts
            )
            quote = "The example program produced a documented result."
            payload: dict[str, Any] = {
                "items": [
                    {
                        "claim": quote,
                        "exact_quote": quote,
                        "dimension_key": None,
                        "relation": "supports",
                        "relevance": 0.99,
                        "confidence": 0.99,
                    }
                ]
                if quote in content
                else {"items": []}
            }
            return CanonicalModelResult(
                parsed_object=payload,
                usage=TokenUsage(
                    input_tokens=12, output_tokens=8, total_tokens=20, accuracy=UsageAccuracy.EXACT
                ),
                provider_request_id="v3-deterministic-fake",
            )
        raise AssertionError(f"unexpected deterministic model task: {request.task_kind}")


class _Bindings:
    def __init__(self, run_id: UUID) -> None:
        self.binding = type(
            "Binding",
            (),
            {
                "run_id": run_id,
                "encrypted_secret": object(),
                "credential_id": uuid4(),
                "credential_version": 1,
                "adapter_type": AdapterType.OPENAI_COMPATIBLE_CHAT,
                "base_url": "https://example.invalid/v1",
                "model": "deterministic-fake",
                "extraction_cache_enabled": False,
                "relevant_chunks_enabled": False,
                "max_output_tokens": None,
                "context_window": None,
            },
        )()

    async def get(self, run_id: UUID) -> Any:
        assert run_id == self.binding.run_id
        return self.binding


class _Cipher:
    def decrypt(self, *_args: Any, **_kwargs: Any) -> SecretStr:
        return SecretStr("test-only-never-used-for-network")


def _semantic_budget() -> SemanticRuntimeBudget:
    return SemanticRuntimeBudget(
        max_ap_decomposition_calls=1,
        max_source_semantic_extraction_calls=1,
        max_identity_resolution_calls=1,
        max_evidence_inputs=2,
        max_propositions_per_evidence=4,
        max_source_assertions_per_evidence=4,
        max_candidate_pairs=0,
        max_direction_assessments=0,
    )


def _new_store(tmp_path: Path) -> tuple[_AsyncSessionFactory, Any]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = _AsyncSessionFactory(
        sessionmaker(
            engine,
            class_=_SQLiteE2ESession,
            expire_on_commit=False,
            identity_sequence=count(1),
        )
    )
    artifacts = LocalArtifactStore(tmp_path)
    return factory, artifacts


def _seed_run(factory: _AsyncSessionFactory, run_id: UUID, goal: str) -> None:
    profile_id, credential_id = uuid4(), uuid4()
    row = ResearchRunRow(
        id=run_id,
        owner_hash="a" * 64,
        idempotency_key=f"e2e-{run_id}",
        original_query=goal,
        normalized_goal=goal,
        constraints={},
        status=RunStatus.QUEUED.value,
        run_kind="research",
        orchestrator_version="V3",
        phase=RunPhase.INITIALIZING.value,
        state_version=1,
        plan_version=0,
        credential_status="available",
        saved_profile_id=profile_id,
        credential_version_id=credential_id,
        llm_config_snapshot={"provider": "fake"},
        budget_snapshot={
            "max_tokens": 50000,
            "max_iterations": 5,
            "max_provider_requests": 20,
            "max_logical_queries": 20,
            "max_pages": 20,
            "max_pages_fetched": 20,
            "max_pages_extracted": 20,
            "max_page_fetch_attempts": 40,
            "max_extraction_calls": 20,
            "max_verification_calls": 0,
            "max_scheduler_actions": 50,
            "allocation": {"writer": 5000, "research": 40000},
        },
        usage_snapshot={},
        quality_snapshot={"coverage_map": []},
    )
    with factory._sync_factory() as session:
        session.add(row)
        session.commit()
        # Keep the synthetic FK targets absent; SQLite foreign keys are disabled
        # by default for this test-only in-memory backend.


def _components(
    factory: _AsyncSessionFactory,
    artifacts: Any,
    run_id: UUID,
    search: _Search,
    reader: _Reader,
    gateway: _Gateway,
) -> tuple[
    ResearchOrchestrator,
    ResearchRunRepository,
    ResearchToolRepository,
    ResearchWorkRepository,
    RunStatePublisher,
    SemanticResearchShadowPipeline,
]:
    runs = ResearchRunRepository(factory)  # type: ignore[arg-type]
    tools = ResearchToolRepository(factory)  # type: ignore[arg-type]
    work = ResearchWorkRepository(factory)  # type: ignore[arg-type]
    reducer = RunStateReducer()
    publisher = RunStatePublisher(artifacts=artifacts, runs=runs)
    shadow = SemanticResearchShadowPipeline(
        repository=tools,
        bindings=cast(Any, _Bindings(run_id)),
        cipher=cast(Any, _Cipher()),
        gateway=cast(Any, gateway),
        artifacts=artifacts,
        budget=_semantic_budget(),
        run_state_runs=runs,
        run_state_reducer=reducer,
        run_state_publisher=publisher,
        semantic_engine=V23DeterministicSemanticEngine(),
    )
    extractor = EvidenceExtractorService(
        cast(Any, _Bindings(run_id)),
        cast(Any, _Cipher()),
        cast(Any, gateway),
        None,
        artifacts,
    )
    loop = ResearchLoopService(
        tools,
        search,  # type: ignore[arg-type]
        reader,  # type: ignore[arg-type]
        extractor,
        artifacts,
        semantic_shadow_mode=SemanticShadowMode.SHADOW,
        semantic_shadow=shadow,
    )
    execution_errors: list[str] = []
    execute_target = loop.execute_target

    async def _capture_test_execution_error(*args: Any, **kwargs: Any) -> Any:
        try:
            return await execute_target(*args, **kwargs)
        except Exception:
            execution_errors.append(traceback.format_exc())
            raise

    loop.execute_target = _capture_test_execution_error  # type: ignore[method-assign]
    orchestrator = ResearchOrchestrator(
        runs=runs,
        work=work,
        run_states=publisher,
        loop=loop,
        tools=tools,
    )
    orchestrator._test_execution_errors = execution_errors  # type: ignore[attr-defined]
    return orchestrator, runs, tools, work, publisher, shadow


def _start_scenario(tmp_path: Path, goal: str) -> dict[str, Any]:
    factory, artifacts = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, goal)
    search, reader, gateway = _Search(), _Reader(), _Gateway()
    orchestrator, runs, tools, work, publisher, shadow = _components(
        factory, artifacts, run_id, search, reader, gateway
    )
    acquired = _drive(
        runs.acquire_for_execution(
            run_id,
            worker_task_id="v3-e2e-worker",
        )
    )
    assert acquired is True
    lease_owner = _assert_execution_lease(
        factory, run_id, worker_task_id="v3-e2e-worker"
    )
    _drive(publisher.initialize_genesis(run_ref=run_id, reducer=RunStateReducer()))
    plan = _plan(goal)
    saved = _drive(
        runs.save_generated_plan(
            run_id,
            worker_task_id="v3-e2e-worker",
            plan=plan,
            usage=TokenUsage(
                input_tokens=0, output_tokens=0, total_tokens=0, accuracy=UsageAccuracy.EXACT
            ),
        )
    )
    assert saved is True
    return {
        "factory": factory,
        "artifacts": artifacts,
        "run_id": run_id,
        "orchestrator": orchestrator,
        "runs": runs,
        "tools": tools,
        "work": work,
        "publisher": publisher,
        "shadow": shadow,
        "plan": plan,
        "search": search,
        "reader": reader,
        "gateway": gateway,
        "lease_owner": lease_owner,
    }


def _assert_execution_lease(
    factory: _AsyncSessionFactory,
    run_id: UUID,
    *,
    worker_task_id: str,
) -> str:
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        assert run is not None
        assert run.status == RunStatus.RUNNING.value
        assert run.worker_task_id == worker_task_id
        assert run.lease_owner
        lease_until = run.lease_until
        assert lease_until is not None
        if lease_until.tzinfo is None:
            lease_until = lease_until.replace(tzinfo=UTC)
        assert lease_until > datetime.now(UTC)
        return run.lease_owner


def _expire_execution_lease(factory: _AsyncSessionFactory, run_id: UUID) -> None:
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        assert run is not None
        run.lease_until = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()


@pytest.mark.integration
def test_v3_full_deterministic_research_e2e(tmp_path: Path) -> None:
    """Fresh Run → persisted Plan → production V3 orchestration/executor path."""
    scenario = _start_scenario(tmp_path, "Find deterministic evidence for the example program.")
    run_id = scenario["run_id"]
    assert _assert_execution_lease(
        scenario["factory"], run_id, worker_task_id="v3-e2e-worker"
    ) == scenario["lease_owner"]
    state_before = _drive(scenario["publisher"].load_current(run_id))
    assert state_before is not None and state_before.state_version == 1
    plan_rows = _drive(_all_rows(scenario["factory"], ResearchPlanItemRow))
    assert len(plan_rows) == 5

    iterations = []
    for _ in range(5):
        iterations.append(
            _drive(scenario["orchestrator"].run_next_work(run_id, worker_task_id="v3-e2e-worker"))
        )
    assert any(result.pages_read > 0 and result.accepted_evidence > 0 for result in iterations), {
        "iterations": iterations,
        "execution_errors": scenario["orchestrator"]._test_execution_errors,
    }
    evidence = _drive(_all_rows(scenario["factory"], ResearchEvidenceRow))
    accepted = [row for row in evidence if row.accepted]
    assert accepted
    assert all(row.question_id and row.source_id and row.snapshot_id for row in accepted)
    assert len({row.evidence_hash for row in accepted}) == len(accepted)
    work_rows = _drive(_all_rows(scenario["factory"], ResearchWorkRow))
    attempt_rows = _drive(_all_rows(scenario["factory"], WorkAttemptRow))
    assert len(work_rows) == 5 and all(row.status == "COMPLETED" for row in work_rows)
    assert len(attempt_rows) == 5 and all(row.state == "COMPLETED" for row in attempt_rows)
    # V3 execution progress is authoritative in Work/Attempt; the durable
    # planner rows remain the immutable source plan rather than a status mirror.
    assert all(row.status == "pending" for row in plan_rows)
    current = _drive(scenario["publisher"].load_current(run_id))
    assert current is not None and current.state_version > 1
    snapshot = _drive(scenario["publisher"].load_semantic_snapshot(current))
    assert snapshot is not None
    assert current.evidence_universe_hash == snapshot.evidence_universe_hash
    assert current.semantic_evidence_hash == snapshot.semantic_evidence_hash
    assert scenario["shadow"].budget_state.provider_ap_decomposition_calls > 0
    assert scenario["shadow"].budget_state.provider_source_semantic_extraction_calls > 0, {
        "budget_state": scenario["shadow"].budget_state,
        "diagnostics": scenario["shadow"]._diagnostics,
    }


@pytest.mark.integration
def test_v3_completed_run_new_instance_resume_is_idempotent(tmp_path: Path) -> None:
    """A cold instance re-enters V3 scheduling from durable state without replay."""
    scenario = _start_scenario(tmp_path, "Find deterministic evidence for the example program.")
    run_id = scenario["run_id"]
    for _ in range(5):
        _drive(scenario["orchestrator"].run_next_work(run_id, worker_task_id="v3-e2e-worker"))
    before = _drive(_counts(scenario["factory"]))
    factory, artifacts, run_id = scenario["factory"], scenario["artifacts"], scenario["run_id"]
    old_lease_owner = scenario["lease_owner"]
    old_publisher = scenario["publisher"]
    old_state = _drive(old_publisher.load_current(run_id))
    _expire_execution_lease(factory, run_id)
    search_b, reader_b, gateway_b = _Search(), _Reader(), _Gateway()
    scenario.clear()
    new_orchestrator, _runs, _tools, _work, publisher_b, _shadow = _components(
        factory, artifacts, run_id, search_b, reader_b, gateway_b
    )
    resumed_worker_task_id = "v3-e2e-worker-resumed"
    assert _drive(
        _runs.acquire_for_execution(
            run_id,
            worker_task_id=resumed_worker_task_id,
        )
    ) is True
    resumed_lease_owner = _assert_execution_lease(
        factory, run_id, worker_task_id=resumed_worker_task_id
    )
    assert resumed_lease_owner != old_lease_owner
    result = _drive(
        new_orchestrator.run_next_work(
            run_id, worker_task_id=resumed_worker_task_id
        )
    )
    after = _drive(_counts(factory))
    assert result.decision == "no_pending_question"
    assert after == before
    assert search_b.calls == [] and reader_b.calls == [] and gateway_b.calls == []
    new_state = _drive(publisher_b.load_current(run_id))
    assert new_state.state_version == old_state.state_version


@pytest.mark.integration
def test_v3_fresh_runs_are_deterministic(tmp_path: Path) -> None:
    """Fresh independent Runs have equal normalized deterministic projections."""
    left = _start_scenario(
        tmp_path / "left", "Find deterministic evidence for the example program."
    )
    right = _start_scenario(
        tmp_path / "right", "Find deterministic evidence for the example program."
    )
    for scenario in (left, right):
        for _ in range(5):
            _drive(
                scenario["orchestrator"].run_next_work(
                    scenario["run_id"], worker_task_id="v3-e2e-worker"
                )
            )
    assert _projection(left) == _projection(right)


@pytest.mark.integration
def test_v3_run_next_work_fails_closed_without_lease(tmp_path: Path) -> None:
    scenario = _start_scenario(tmp_path, "Find deterministic evidence for the example program.")
    run_id = scenario["run_id"]
    with scenario["factory"]._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        assert run is not None
        run.lease_owner = None
        run.lease_until = None
        session.commit()

    with pytest.raises(RuntimeError, match="research_execution_lease_identity_unavailable"):
        _drive(
            scenario["orchestrator"].run_next_work(
                run_id, worker_task_id="v3-e2e-worker"
            )
        )
    assert scenario["search"].calls == []
    assert scenario["reader"].calls == []
    assert scenario["gateway"].calls == []


def test_run_status_canonical_value_passes_execution_ownership_guard(
    tmp_path: Path,
) -> None:
    scenario = _start_scenario(tmp_path, "Run status ownership contract regression.")
    run_id = scenario["run_id"]
    work_ref, attempt_ref = uuid4(), uuid4()
    with scenario["factory"]._sync_factory() as session:
        session.add(
            ResearchWorkRow(
                work_ref=work_ref,
                run_id=run_id,
                producer_kind="TEST",
                producer_ref="status-contract",
                work_kind="LEGACY_RESEARCH",
                objective={},
                objective_hash="a" * 64,
                priority=1,
                constraints={},
                execution_bounds={},
                budget_attribution={},
                source_run_state_ref="test/run-state.json",
                source_run_state_hash="b" * 64,
                source_run_state_version=1,
                status=ResearchWorkStatus.IN_PROGRESS.value,
                schema_version="test-v1",
            )
        )
        session.add(
            WorkAttemptRow(
                attempt_ref=attempt_ref,
                work_ref=work_ref,
                attempt_index=1,
                state=WorkAttemptState.RESERVED.value,
                reservation_identity=f"status-contract:{attempt_ref}",
                source_run_state_ref="test/run-state.json",
                source_run_state_version=1,
                schema_version="work-attempt-v1",
            )
        )
        session.commit()

    lease_owner = scenario["lease_owner"]
    assert isinstance(lease_owner, str)
    claimed = _drive(
        scenario["work"].claim_attempt_execution_ownership(
            run_id,
            attempt_ref,
            worker_task_id="v3-e2e-worker",
            run_lease_token=lease_owner,
            execution_owner_token=uuid4().hex,
        )
    )
    assert claimed is True


def test_noncanonical_run_status_fails_execution_ownership_closed(
    tmp_path: Path,
) -> None:
    scenario = _start_scenario(tmp_path, "Run status negative contract test.")
    run_id = scenario["run_id"]
    with scenario["factory"]._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        assert run is not None
        run.status = "RUNNING"
        session.commit()

    with pytest.raises(ResearchWorkBindingError, match="execution_run_lease_not_owned"):
        _drive(
            _lock_current_execution_run(
                scenario["factory"],
                scenario["work"],
                run_id,
                worker_task_id="v3-e2e-worker",
            )
        )


async def _lock_current_execution_run(
    factory: _AsyncSessionFactory,
    repository: ResearchWorkRepository,
    run_id: UUID,
    *,
    worker_task_id: str,
) -> ResearchRunRow:
    async with factory() as session:
        return await repository._lock_current_execution_run(
            session, run_id, worker_task_id=worker_task_id  # type: ignore[arg-type]
        )


async def _all_rows(factory: _AsyncSessionFactory, model: type[Any]) -> list[Any]:
    async with factory() as session:
        return list(session._sync.scalars(__import__("sqlalchemy").select(model)).all())


async def _counts(factory: _AsyncSessionFactory) -> tuple[int, int, int, int, int]:
    from app.infrastructure.db.run_models import ResearchWorkRow, WorkAttemptRow

    return (
        len(await _all_rows(factory, ResearchWorkRow)),
        len(await _all_rows(factory, WorkAttemptRow)),
        len(await _all_rows(factory, ResearchPlanItemRow)),
        len(await _all_rows(factory, ResearchEvidenceRow)),
        len(await _all_rows(factory, ResearchRunRow)),
    )


def _projection(scenario: dict[str, Any]) -> tuple[Any, ...]:
    evidence = _drive(_all_rows(scenario["factory"], ResearchEvidenceRow))
    works = _drive(_all_rows(scenario["factory"], ResearchWorkRow))
    attempts = _drive(_all_rows(scenario["factory"], WorkAttemptRow))
    state = _drive(scenario["publisher"].load_current(scenario["run_id"]))
    snapshot = _drive(scenario["publisher"].load_semantic_snapshot(state))
    return (
        tuple(
            sorted(
                (
                    row.question_id,
                    row.claim,
                    row.exact_quote,
                    row.relation,
                    row.accepted,
                )
                for row in evidence
            )
        ),
        tuple(sorted((row.producer_ref, row.objective_hash, row.status) for row in works)),
        tuple(sorted((row.attempt_index, row.state) for row in attempts)),
        (
            snapshot.execution_status,
            snapshot.semantic_engine_version,
            snapshot.schema_version,
            snapshot.required_product_versions,
            snapshot.semantic_statuses,
            snapshot.known_limitations,
            tuple(sorted(item.product_kind for item in snapshot.output_product_bindings)),
        ),
        (
            state.lifecycle_status,
            state.phase,
            state.semantic_execution_status,
            state.semantic_engine_version,
            state.semantic_schema_version,
            state.known_limitations,
            state.capability_states,
            state.deferred_states,
        ),
    )
