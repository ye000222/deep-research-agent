from __future__ import annotations

from collections.abc import Coroutine
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.domain.run_state import (
    EvidenceUniverse,
    EvidenceUniverseEntry,
    SemanticExecutionStatus,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudget
from app.services.durable_semantic_input import DurableSemanticInputLoader
from app.services.run_state_publisher import RunStatePublisher
from app.services.run_state_reducer import RunStateReducer
from app.services.semantic_engine import V23DeterministicSemanticEngine
from app.services.semantic_feedback_recovery import (
    SemanticFeedbackRecoveryCoordinator,
    SemanticRecoveryError,
)
from app.services.semantic_research_shadow import SemanticResearchShadowPipeline

from tests.unit.test_durable_semantic_input_loader import (
    _Artifacts as DurableTextArtifacts,
)
from tests.unit.test_durable_semantic_input_loader import (
    _dataset,
)
from tests.unit.test_durable_semantic_input_loader import (
    _run_immediate as _run_loader_immediate,
)
from tests.unit.test_durable_semantic_input_loader import (
    _SessionFactory as DurableSessionFactory,
)
from tests.unit.test_semantic_research_shadow import (
    FakeBindings,
    FakeCipher,
    FakeGateway,
    MemoryArtifactStore,
    _budget,
)

_RUN = UUID("11111111-1111-4111-8111-111111111111")
_EVIDENCE = UUID("22222222-2222-4222-8222-222222222222")


def _run_immediate(coroutine: Coroutine[object, object, object]) -> object:
    try:
        return coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    coroutine.close()
    raise AssertionError("fake recovery coroutine unexpectedly suspended")


def _universe() -> EvidenceUniverse:
    return EvidenceUniverse.build(
        _RUN,
        (
            EvidenceUniverseEntry(
                evidence_ref=_EVIDENCE,
                source_ref=UUID("33333333-3333-4333-8333-333333333333"),
                accepted=True,
                semantic_eligible=True,
            ),
        ),
    )


class _Rows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _Session:
    def __init__(self, attempts: list[tuple[list[str]]], events: list[object]) -> None:
        self.attempts = attempts
        self.events = events

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def execute(self, _statement: object) -> _Rows:
        return _Rows(self.attempts)

    async def scalars(self, _statement: object) -> _Rows:
        return _Rows(self.events)


class _SessionFactory:
    def __init__(self, attempts: list[tuple[list[str]]], events: list[object]) -> None:
        self.attempts = attempts
        self.events = events

    def __call__(self) -> _Session:
        return _Session(self.attempts, self.events)


class _Publisher:
    def __init__(self, states: list[object], universe: EvidenceUniverse) -> None:
        self.states = states
        self.universe = universe
        self.orphans: list[str] = []

    async def load_current(self, _run_ref: UUID) -> object:
        return self.states.pop(0)

    async def load_semantic_snapshot(self, _state: object) -> object:
        return SimpleNamespace(
            semantic_evidence_refs=self.universe.semantic_evidence_refs,
            evidence_universe_hash=self.universe.evidence_universe_hash,
            semantic_evidence_hash=self.universe.semantic_evidence_hash,
        )

    async def recover_orphan_candidate(self, artifact_ref: str) -> bool:
        self.orphans.append(artifact_ref)
        return True


class _Runs:
    def __init__(self, universes: list[EvidenceUniverse]) -> None:
        self.universes = universes

    async def load_evidence_universe(self, _run_ref: UUID) -> EvidenceUniverse:
        return self.universes.pop(0)


class _Loader:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str, tuple[UUID, ...] | None]] = []

    async def load_page(
        self, run_ref: UUID, page_ref: str, evidence_refs: tuple[UUID, ...] | None = None
    ) -> object:
        self.calls.append((run_ref, page_ref, evidence_refs))
        return SimpleNamespace(receipt_status="AVAILABLE", claim_receipts=())


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, object]] = []

    async def process_persisted_page(self, run_ref: UUID, persisted: object) -> None:
        self.calls.append((run_ref, persisted))


def _state(universe: EvidenceUniverse, *, complete: bool) -> object:
    return SimpleNamespace(
        run_ref=_RUN,
        evidence_universe_hash=universe.evidence_universe_hash if complete else "0" * 64,
        semantic_evidence_hash=universe.semantic_evidence_hash if complete else "1" * 64,
        semantic_execution_status=(
            SemanticExecutionStatus.COMPLETE
            if complete
            else SemanticExecutionStatus.INCOMPLETE
        ),
        semantic_engine_version="v2.3-deterministic-rebuild-v1",
        semantic_schema_version="semantic-state-snapshot-v1",
        semantic_state_ref="runs/run/semantic.json" if complete else None,
    )


def _coordinator(
    *, attempts: list[tuple[list[str]]], events: list[object], states: list[object],
    universes: list[EvidenceUniverse],
) -> tuple[SemanticFeedbackRecoveryCoordinator, _Loader, _Pipeline, _Publisher]:
    universe = universes[0]
    publisher = _Publisher(states, universe)
    loader = _Loader()
    pipeline = _Pipeline()
    coordinator = SemanticFeedbackRecoveryCoordinator(
        sessions=_SessionFactory(attempts, events),  # type: ignore[arg-type]
        runs=_Runs(universes),  # type: ignore[arg-type]
        publisher=publisher,  # type: ignore[arg-type]
        loader=loader,  # type: ignore[arg-type]
        semantic_pipeline=pipeline,  # type: ignore[arg-type]
    )
    return coordinator, loader, pipeline, publisher


def _event() -> object:
    return SimpleNamespace(
        refs={
            "diagnostic_version": "evidence-formation-diagnostic.v1",
            "stage": "PERSISTENCE",
            "outcome": "SUCCEEDED",
            "page_ref": "page-1",
            "evidence_ref": str(_EVIDENCE),
        }
    )


def test_synced_state_is_cw6_noop() -> None:
    universe = _universe()
    coordinator, loader, pipeline, _publisher = _coordinator(
        attempts=[], events=[], states=[_state(universe, complete=True)], universes=[universe]
    )
    assert _run_immediate(coordinator.recover_semantic_feedback_for_run(_RUN)) == 0
    assert loader.calls == []
    assert pipeline.calls == []


def test_lag_discovers_page_and_reuses_durable_loader() -> None:
    universe = _universe()
    coordinator, loader, pipeline, _publisher = _coordinator(
        attempts=[([str(_EVIDENCE)],)],
        events=[_event()],
        states=[_state(universe, complete=False), _state(universe, complete=True)],
        universes=[universe, universe],
    )
    assert _run_immediate(coordinator.recover_semantic_feedback_for_run(_RUN)) == 1
    assert loader.calls == [(_RUN, "page-1", (_EVIDENCE,))]
    assert len(pipeline.calls) == 1


def test_lag_without_attempt_evidence_binding_fails_closed() -> None:
    universe = _universe()
    coordinator, loader, pipeline, _publisher = _coordinator(
        attempts=[],
        events=[_event()],
        states=[_state(universe, complete=False)],
        universes=[universe],
    )
    with pytest.raises(SemanticRecoveryError, match="ATTEMPT_PAGE_EVIDENCE_BINDING_INCOMPLETE"):
        _run_immediate(coordinator.recover_semantic_feedback_for_run(_RUN))
    assert loader.calls == []
    assert pipeline.calls == []


def test_known_orphan_uses_existing_publisher_seam() -> None:
    universe = _universe()
    coordinator, _loader, _pipeline, publisher = _coordinator(
        attempts=[], events=[], states=[], universes=[universe]
    )
    assert _run_immediate(
        coordinator.recover_known_orphan_candidate("runs/r/state/candidate.json")
    ) is True
    assert publisher.orphans == ["runs/r/state/candidate.json"]


def test_cw2_rehydrates_durable_evidence_and_publishes_authoritative_state() -> None:
    claim_text = "设备成功率为12.4%。"
    dataset, source_text, run_ref, page_ref = _dataset(text=claim_text)
    evidence = dataset["evidence"][0]
    claim = dataset["claims"][evidence.claim_id]
    source = dataset["sources"][evidence.source_id]
    evidence_ref = evidence.id
    claim_ref = claim.id
    source_ref = source.id
    event_rows = dataset["events"]

    class _AttemptSessionFactory(DurableSessionFactory):
        def __call__(self) -> object:
            session = super().__call__()

            async def execute(_statement: object) -> object:
                return _Rows([([str(evidence_ref)],)])

            session.execute = execute  # type: ignore[attr-defined]
            return session

    class _RunStore:
        def __init__(self) -> None:
            self.pointer: dict[str, object] = {
                "orchestrator_version": "V3",
                "status": "RUNNING",
                "phase": "RESEARCHING",
                "budget": {},
                "usage": {},
                "termination_reason": None,
                "latest_run_state_ref": None,
                "latest_run_state_hash": None,
                "latest_run_state_version": None,
            }

        async def get_run_state_inputs(self, _run_ref: UUID) -> dict[str, object]:
            return dict(self.pointer)

        async def load_evidence_universe(self, _run_ref: UUID) -> EvidenceUniverse:
            return EvidenceUniverse.build(
                run_ref,
                (
                    EvidenceUniverseEntry(
                        evidence_ref=evidence_ref,
                        claim_ref=claim_ref,
                        source_ref=source_ref,
                        accepted=True,
                        semantic_eligible=True,
                    ),
                ),
            )

        async def advance_run_state_pointer(self, state: object) -> None:
            self.pointer.update(
                {
                    "latest_run_state_ref": state.state_ref,  # type: ignore[attr-defined]
                    "latest_run_state_hash": state.state_hash,  # type: ignore[attr-defined]
                    "latest_run_state_version": state.state_version,  # type: ignore[attr-defined]
                }
            )

    class _ArtifactStore(MemoryArtifactStore):
        async def save_immutable_json(self, artifact_uri: str, value: object) -> str:
            return await self.save_json(artifact_uri, value)

    run_store = _RunStore()
    artifacts = _ArtifactStore()
    reducer = RunStateReducer()
    publisher = RunStatePublisher(artifacts=artifacts, runs=run_store)  # type: ignore[arg-type]
    _run_loader_immediate(publisher.initialize_genesis(run_ref=run_ref, reducer=reducer))

    gateway = FakeGateway()
    budget: SemanticRuntimeBudget = _budget()
    pipeline = SemanticResearchShadowPipeline(
        repository=object(),  # type: ignore[arg-type]
        bindings=FakeBindings(run_ref),  # type: ignore[arg-type]
        cipher=FakeCipher(),  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
        artifacts=artifacts,  # type: ignore[arg-type]
        budget=budget,
        run_state_runs=run_store,
        run_state_reducer=reducer,
        run_state_publisher=publisher,
        semantic_engine=V23DeterministicSemanticEngine(),
    )
    pipeline._run_orchestrator_versions[run_ref] = "V3"

    loader = DurableSemanticInputLoader(
        _AttemptSessionFactory(dataset),  # type: ignore[arg-type]
        DurableTextArtifacts(source_text),  # type: ignore[arg-type]
    )
    coordinator = SemanticFeedbackRecoveryCoordinator(
        sessions=_AttemptSessionFactory(dataset),  # type: ignore[arg-type]
        runs=run_store,  # type: ignore[arg-type]
        publisher=publisher,
        loader=loader,
        semantic_pipeline=pipeline,
    )

    # Simulate a new process: only durable rows/artifacts, not a prior receipt,
    # are available to the recovery coordinator.
    assert _run_immediate(coordinator.recover_semantic_feedback_for_run(run_ref)) == 1
    current = _run_immediate(publisher.load_current(run_ref))
    assert current is not None
    expected_universe = _run_loader_immediate(run_store.load_evidence_universe(run_ref))
    assert current.evidence_universe_hash == expected_universe.evidence_universe_hash
    assert current.semantic_state_ref is not None
    assert current.semantic_execution_status is SemanticExecutionStatus.COMPLETE
    assert any("/work-products/" in uri for uri in artifacts.values)
    assert gateway.calls
    assert all(call["request"].task_kind != "research" for call in gateway.calls)
    assert len(event_rows) == 2
    assert claim.atomic_claim == claim_text
    assert page_ref == "candidate-page-1"


@pytest.mark.parametrize("defect", ["missing_bundle", "invalid_product_hash"])
def test_cw3_completed_state_with_missing_or_invalid_product_fails_closed(
    defect: str,
) -> None:
    from tests.unit.test_v3_semantic_engine_integration import (
        _RUN as V3_RUN,
    )
    from tests.unit.test_v3_semantic_engine_integration import (
        _ArtifactStore as V3Artifacts,
    )
    from tests.unit.test_v3_semantic_engine_integration import (
        _bundle as valid_bundle,
    )
    from tests.unit.test_v3_semantic_engine_integration import (
        _RunRepository as V3Runs,
    )
    from tests.unit.test_v3_semantic_engine_integration import (
        _universe as v3_universe,
    )

    class _LoaderMustNotRun:
        async def load_page(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("invalid completed product must stop before replay")

    class _PipelineMustNotRun:
        async def process_persisted_page(self, *_args: object) -> None:
            raise AssertionError("invalid completed product must not form products")

    async def scenario() -> None:
        artifacts = V3Artifacts()
        runs = V3Runs()
        reducer = RunStateReducer()
        publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        genesis = await publisher.initialize_genesis(run_ref=V3_RUN, reducer=reducer)
        universe = v3_universe()
        bundle_ref = f"runs/{V3_RUN}/semantic-shadow/bundle.json"
        bundle = valid_bundle()
        await artifacts.save_immutable_json(bundle_ref, bundle)
        snapshot = await V23DeterministicSemanticEngine().evaluate(
            run_ref=V3_RUN,
            evidence_universe=universe,
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": bundle_ref,
            },
        )
        state = reducer.reduce(
            run_ref=V3_RUN,
            run_control={
                "status": "RUNNING",
                "phase": "RESEARCHING",
                "budget": {},
                "usage": {},
                "termination_reason": None,
            },
            evidence_universe=universe,
            semantic_snapshot=snapshot,
            previous=genesis,
        )
        await publisher.publish(state=state, semantic_snapshot=snapshot)
        if defect == "missing_bundle":
            artifacts.values.pop(bundle_ref)
        else:
            bundle_value = artifacts.values[bundle_ref]
            assert isinstance(bundle_value, dict)
            bundle_value["verification_assessments"] = []

        coordinator = SemanticFeedbackRecoveryCoordinator(
            sessions=object(),  # type: ignore[arg-type]
            runs=runs,  # type: ignore[arg-type]
            publisher=publisher,
            loader=_LoaderMustNotRun(),  # type: ignore[arg-type]
            semantic_pipeline=_PipelineMustNotRun(),
        )
        if defect == "missing_bundle":
            with pytest.raises(KeyError):
                await coordinator.recover_semantic_feedback_for_run(V3_RUN)
        else:
            with pytest.raises(ValueError, match="semantic_product_content_hash_mismatch"):
                await coordinator.recover_semantic_feedback_for_run(V3_RUN)

    _run_loader_immediate(scenario())
