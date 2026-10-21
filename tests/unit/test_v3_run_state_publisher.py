from __future__ import annotations

from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from app.domain.run_state import (
    EvidenceUniverse,
    EvidenceUniverseEntry,
    GapProjectionStatus,
    ResearchRunState,
    SemanticExecutionStatus,
    build_research_run_state,
    canonical_hash,
)
from app.infrastructure.db.research_runs import (
    ResearchRunRepository,
    RunStatePointerConflictError,
)
from app.services.run_state_publisher import RunStatePublisher
from app.services.run_state_reducer import RunStateReducer

_RUN = UUID("00000000-0000-0000-0000-000000000101")


def _run_immediate(coroutine: Coroutine[Any, Any, object]) -> object:
    try:
        return_value = coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError(f"fake state publisher unexpectedly suspended: {return_value!r}")


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def save_immutable_json(self, ref: str, value: object) -> str:
        previous = self.values.get(ref)
        if previous is not None and previous != value:
            raise ValueError("immutable_artifact_content_conflict")
        self.values[ref] = value
        return ref

    async def read_json(self, ref: str) -> object:
        return self.values[ref]


class _FakeRunRepository:
    def __init__(self, artifacts: _FakeArtifactStore, *, fail_first: bool = False) -> None:
        self.artifacts = artifacts
        self.fail_first = fail_first
        self.calls = 0
        self.pointer: tuple[str, str, int] | None = None

    async def advance_run_state_pointer(self, state: ResearchRunState) -> None:
        self.calls += 1
        state_ref = str(state.state_ref)
        assert state_ref in self.artifacts.values
        if self.fail_first and self.calls == 1:
            raise RuntimeError("simulated_cas_conflict")
        self.pointer = (
            state_ref,
            state.state_hash,
            state.state_version,
        )


class _FakeDBSession:
    def __init__(self, row: SimpleNamespace) -> None:
        self.row = row
        self.locked = False

    async def __aenter__(self) -> _FakeDBSession:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> _FakeDBSession:
        return self

    async def scalar(self, statement: object) -> SimpleNamespace:
        self.locked = getattr(statement, "_for_update_arg", None) is not None
        return self.row

    async def flush(self) -> None:
        return None


def _next_state(previous: ResearchRunState, *, phase: str) -> ResearchRunState:
    return build_research_run_state(
        run_ref=_RUN,
        state_version=previous.state_version + 1,
        parent_state_ref=previous.state_ref,
        parent_state_hash=previous.state_hash,
        expected_parent_version=previous.state_version,
        lifecycle_status="RUNNING",
        phase=phase,
        evidence_universe_hash=previous.evidence_universe_hash,
        semantic_evidence_hash=previous.semantic_evidence_hash,
        semantic_state_ref=None,
        semantic_state_hash=None,
        semantic_execution_status=SemanticExecutionStatus.INCOMPLETE,
        gap_projection_status=GapProjectionStatus.NOT_AVAILABLE,
        known_limitations=("SEMANTIC_STATE_NOT_AVAILABLE",),
        capability_states=(),
        deferred_states=(),
        budget_projection=(),
        budget_state_hash=canonical_hash({}),
        usage_state_hash=canonical_hash({}),
        termination_reason=None,
        execution_refs=(),
    )


def _state():
    universe = EvidenceUniverse.build(_RUN, ())
    return build_research_run_state(
        run_ref=_RUN,
        state_version=1,
        parent_state_ref=None,
        parent_state_hash=None,
        expected_parent_version=None,
        lifecycle_status="RUNNING",
        phase="INITIALIZING",
        evidence_universe_hash=universe.evidence_universe_hash,
        semantic_evidence_hash=universe.semantic_evidence_hash,
        semantic_execution_status=SemanticExecutionStatus.INCOMPLETE,
        gap_projection_status=GapProjectionStatus.NOT_AVAILABLE,
        known_limitations=("SEMANTIC_STATE_NOT_AVAILABLE",),
        capability_states=(),
        deferred_states=(),
        budget_projection=(),
        budget_state_hash=canonical_hash({}),
        usage_state_hash=canonical_hash({}),
        termination_reason=None,
        execution_refs=(),
    )


def _publisher(*, fail_first: bool = False):
    artifacts = _FakeArtifactStore()
    repository = _FakeRunRepository(artifacts, fail_first=fail_first)
    return RunStatePublisher(
        artifacts=artifacts, runs=repository  # type: ignore[arg-type]
    ), artifacts, repository


def test_artifact_is_durable_before_pointer_and_pointer_replay_is_safe() -> None:
    publisher, artifacts, repository = _publisher()
    state = _state()

    _run_immediate(publisher.publish(state=state, semantic_snapshot=None))
    _run_immediate(publisher.publish(state=state, semantic_snapshot=None))

    assert state.state_ref in artifacts.values
    assert repository.calls == 2
    assert repository.pointer == (state.state_ref, state.state_hash, 1)


def test_cas_failure_leaves_durable_orphan_for_recovery() -> None:
    publisher, artifacts, repository = _publisher(fail_first=True)
    initial = _state()
    universe = EvidenceUniverse.build(
        _RUN,
        (
            EvidenceUniverseEntry(
                evidence_ref=UUID("00000000-0000-0000-0000-000000000211"),
                claim_ref=UUID("00000000-0000-0000-0000-000000000212"),
                source_ref=UUID("00000000-0000-0000-0000-000000000213"),
                source_snapshot_ref=UUID("00000000-0000-0000-0000-000000000214"),
                source_content_hash="a" * 64,
                accepted=True,
                semantic_eligible=True,
            ),
        ),
    )
    state = RunStateReducer().reduce(
        run_ref=_RUN,
        run_control={
            "status": "RUNNING",
            "phase": "RESEARCHING",
            "budget": {},
            "usage": {},
            "termination_reason": None,
        },
        evidence_universe=universe,
        semantic_snapshot=None,
        previous=initial,
    )
    assert state.state_version == initial.state_version + 1
    assert state.evidence_universe_hash == universe.evidence_universe_hash

    with pytest.raises(RuntimeError, match="simulated_cas_conflict"):
        _run_immediate(publisher.publish(state=state, semantic_snapshot=None))
    assert state.state_ref in artifacts.values

    # Recreate the publisher after the process-local instance is discarded;
    # only the artifact store and durable pointer repository survive.
    del publisher
    recovered_publisher = RunStatePublisher(
        artifacts=artifacts, runs=repository  # type: ignore[arg-type]
    )
    _run_immediate(recovered_publisher.recover_orphan_candidate(state.state_ref))
    assert repository.pointer == (state.state_ref, state.state_hash, 2)


def test_immutable_artifact_conflict_does_not_overwrite() -> None:
    store = _FakeArtifactStore()

    _run_immediate(store.save_immutable_json("runs/test/state.json", {"value": 1}))
    with pytest.raises(ValueError, match="immutable_artifact_content_conflict"):
        _run_immediate(store.save_immutable_json("runs/test/state.json", {"value": 2}))
    assert store.values["runs/test/state.json"] == {"value": 1}


def test_repository_pointer_uses_row_lock_and_rejects_out_of_order_state() -> None:
    row = SimpleNamespace(
        orchestrator_version="V3",
        latest_run_state_ref=None,
        latest_run_state_hash=None,
        latest_run_state_version=None,
    )
    session = _FakeDBSession(row)
    repository = ResearchRunRepository(lambda: session)  # type: ignore[arg-type]
    initial = _state()
    _run_immediate(repository.advance_run_state_pointer(initial))
    assert session.locked
    advanced = _next_state(initial, phase="RESEARCHING")
    _run_immediate(repository.advance_run_state_pointer(advanced))
    stale = _next_state(initial, phase="STALE_RESEARCHING")
    with pytest.raises(RunStatePointerConflictError, match="run_state_version_conflict"):
        _run_immediate(repository.advance_run_state_pointer(stale))
    assert row.latest_run_state_ref == advanced.state_ref
    assert row.latest_run_state_version == 2


def test_repository_refuses_to_add_v3_pointer_to_legacy_run() -> None:
    row = SimpleNamespace(
        orchestrator_version="V2_LEGACY",
        latest_run_state_ref=None,
        latest_run_state_hash=None,
        latest_run_state_version=None,
    )
    repository = ResearchRunRepository(lambda: _FakeDBSession(row))  # type: ignore[arg-type]
    with pytest.raises(RunStatePointerConflictError, match="legacy_run_cannot_receive_v3_pointer"):
        _run_immediate(repository.advance_run_state_pointer(_state()))
