from __future__ import annotations

import json
from collections.abc import Coroutine
from typing import Any
from uuid import UUID

import pytest
from app.domain.run_state import (
    EvidenceUniverse,
    EvidenceUniverseEntry,
    ResearchRunState,
    SemanticExecutionStatus,
    canonical_hash,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudget
from app.services.run_state_publisher import RunStatePublisher
from app.services.run_state_reducer import RunStateReducer
from app.services.semantic_engine import V23DeterministicSemanticEngine
from app.services.semantic_research_shadow import SemanticResearchShadowPipeline

_RUN = UUID("00000000-0000-0000-0000-000000000101")
_CLAIM = UUID("00000000-0000-0000-0000-000000000102")
_EVIDENCE = UUID("00000000-0000-0000-0000-000000000103")
_SOURCE = UUID("00000000-0000-0000-0000-000000000104")
def _universe(*, evidence_ref: UUID = _EVIDENCE) -> EvidenceUniverse:
    return EvidenceUniverse.build(
        _RUN,
        (
            EvidenceUniverseEntry(
                evidence_ref=evidence_ref,
                claim_ref=_CLAIM,
                source_ref=_SOURCE,
                accepted=True,
                semantic_eligible=True,
            ),
        ),
    )


def _bundle() -> dict[str, object]:
    return {
        "schema_version": "semantic-shadow-verification-snapshot-v2",
        "run_ref": str(_RUN),
        "claim_refs": [str(_CLAIM)],
        "evidence_refs": [str(_EVIDENCE)],
        "proposition_refs": ["proposition-1"],
        "formation_assessments": [{"status": "UNKNOWN"}],
        "proposition_group_refs": [],
        "proposition_groups": [],
        "pair_assessments": [],
        "proposition_evidence_links": [],
        "corroboration_assessments": [],
        "conflicts": {"schema_version": "proposition-conflict-assessment-v1"},
        "verification_assessments": [{"status": "UNKNOWN"}],
        "conflict_scope_completeness": {
            "schema_version": "conflict-scope-completeness-v1",
            "completeness": "UNKNOWN",
        },
        "schema_versions": {
            "semantic_snapshot": "semantic-shadow-verification-snapshot-v2",
            "atomic_proposition": "atomic-proposition-schema-v3",
            "target_formation_schema": "target-proposition-formation-response-v1",
            "proposition_groups": "proposition-support-group-v1",
            "proposition_pair_identity": "proposition-pair-identity-runtime-v1",
            "direction": "proposition-evidence-direction-v1",
            "corroboration": "independent-corroboration-v1",
            "conflicts": "proposition-conflict-assessment-v1",
            "verification": "proposition-verification-assessment-v1",
            "conflict_scope_completeness": "conflict-scope-completeness-v1",
        },
    }


class _RunRepository:
    def __init__(self) -> None:
        self.state: dict[str, object] = {
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

    async def get_run_state_inputs(self, run_id: UUID) -> dict[str, object]:
        assert run_id == _RUN
        return dict(self.state)

    async def load_evidence_universe(self, run_id: UUID) -> EvidenceUniverse:
        assert run_id == _RUN
        return _universe()

    async def advance_run_state_pointer(self, state: ResearchRunState) -> None:
        version = state.state_version
        current = self.state["latest_run_state_version"]
        assert version == (1 if current is None else current + 1)
        self.state["latest_run_state_ref"] = state.state_ref
        self.state["latest_run_state_hash"] = state.state_hash
        self.state["latest_run_state_version"] = version


class _EmptyRunRepository(_RunRepository):
    async def load_evidence_universe(self, run_id: UUID) -> EvidenceUniverse:
        assert run_id == _RUN
        return EvidenceUniverse.build(_RUN, ())


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drive only the immediate fake-boundary coroutines without a host loop."""
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("test coroutine unexpectedly requires an event loop")


class _ArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def save_immutable_json(self, ref: str, value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        existing = self.values.get(ref)
        if existing is not None and json.dumps(
            existing, sort_keys=True, separators=(",", ":")
        ) != encoded:
            raise ValueError("immutable_artifact_content_conflict")
        self.values[ref] = value
        return ref

    async def save_json(self, ref: str, value: object) -> str:
        self.values[ref] = value
        return ref

    async def read_json(self, ref: str) -> object:
        return self.values[ref]


def test_v23_rebuild_produces_bound_complete_snapshot_and_preserves_unknown() -> None:
    universe = _universe()
    bundle = _bundle()
    snapshot = _run(
        V23DeterministicSemanticEngine().evaluate(
            run_ref=_RUN,
            evidence_universe=universe,
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": f"runs/{_RUN}/semantic-shadow/bundle.json",
            },
        )
    )
    assert snapshot.execution_status is SemanticExecutionStatus.COMPLETE
    assert snapshot.state_scope == "RUN_CURRENT_EVIDENCE_UNIVERSE"
    assert snapshot.semantic_input_hash == canonical_hash(
        {
            "run_ref": str(_RUN),
            "evidence_universe_hash": universe.evidence_universe_hash,
            "semantic_evidence_hash": universe.semantic_evidence_hash,
            "semantic_evidence_refs": [str(_EVIDENCE)],
            "evidence_entries": [
                item.model_dump(mode="json")
                for item in universe.entries
                if item.semantic_eligible
            ],
            "required_product_versions": snapshot.required_product_versions,
        }
    )
    assert snapshot.semantic_statuses
    assert bundle["verification_assessments"] == [{"status": "UNKNOWN"}]
    assert snapshot.evidence_universe_hash == universe.evidence_universe_hash


def test_v23_engine_is_deterministic_and_fails_closed_on_missing_universe_member() -> None:
    engine = V23DeterministicSemanticEngine()
    bundle = _bundle()
    context = {
        "semantic_products": bundle,
        "product_bundle_ref": f"runs/{_RUN}/semantic-shadow/bundle.json",
    }
    first = _run(
        engine.evaluate(run_ref=_RUN, evidence_universe=_universe(), execution_context=context)
    )
    second = _run(
        engine.evaluate(run_ref=_RUN, evidence_universe=_universe(), execution_context=context)
    )
    assert first.snapshot_hash == second.snapshot_hash
    changed_universe = _universe(
        evidence_ref=UUID("00000000-0000-0000-0000-000000000199")
    )
    incomplete = _run(
        engine.evaluate(
            run_ref=_RUN,
            evidence_universe=changed_universe,
            execution_context=context,
        )
    )
    assert incomplete.execution_status is SemanticExecutionStatus.INCOMPLETE
    assert "MISSING_OR_UNBOUND:evidence_universe_coverage" in incomplete.known_limitations


def test_snapshot_publication_cold_recovery_and_product_hash_validation() -> None:
    async def scenario() -> None:
        universe = _universe()
        artifacts = _ArtifactStore()
        bundle_ref = f"runs/{_RUN}/semantic-shadow/bundle.json"
        bundle = _bundle()
        await artifacts.save_immutable_json(bundle_ref, bundle)
        snapshot = await V23DeterministicSemanticEngine().evaluate(
            run_ref=_RUN,
            evidence_universe=universe,
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": bundle_ref,
            },
        )
        runs = _RunRepository()
        reducer = RunStateReducer()
        state = reducer.reduce(
            run_ref=_RUN,
            run_control={
                "status": "RUNNING",
                "phase": "RESEARCHING",
                "budget": {},
                "usage": {},
                "termination_reason": None,
            },
            evidence_universe=universe,
            semantic_snapshot=snapshot,
        )
        publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        await publisher.publish(state=state, semantic_snapshot=snapshot)
        fresh_publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        restored = await fresh_publisher.load_current(_RUN)
        assert restored == state
        restored_snapshot = await fresh_publisher.load_semantic_snapshot(restored)
        assert restored_snapshot == snapshot

        # The run pointer is authoritative; changed product bytes are detected on cold load.
        await artifacts.save_json(bundle_ref, {**bundle, "verification_assessments": []})
        with pytest.raises(ValueError, match="semantic_product_content_hash_mismatch"):
            await fresh_publisher.load_current(_RUN)

    _run(scenario())


def test_budget_not_processed_is_incomplete_and_diagnostic_is_retained() -> None:
    bundle = _bundle()
    bundle["diagnostics"] = [
        {"status": "NOT_PROCESSED_BUDGET", "reason_code": "PAIR_BUDGET_EXHAUSTED"}
    ]
    bundle["budget_state"] = {"candidate_pairs": 2}
    snapshot = _run(
        V23DeterministicSemanticEngine().evaluate(
            run_ref=_RUN,
            evidence_universe=_universe(),
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": f"runs/{_RUN}/semantic-shadow/bundle.json",
            },
        )
    )
    assert snapshot.execution_status is SemanticExecutionStatus.INCOMPLETE
    assert "NOT_PROCESSED_BUDGET" in snapshot.deferred_states
    assert "PAIR_BUDGET_EXHAUSTED" in snapshot.diagnostic_codes
    assert snapshot.budget_state_hash == canonical_hash({"candidate_pairs": 2})


def test_semantic_rebuild_failure_is_durable_failed_state_not_incomplete() -> None:
    universe = _universe()
    engine = V23DeterministicSemanticEngine()
    snapshot = engine.failed_snapshot(run_ref=_RUN, evidence_universe=universe)

    assert snapshot.execution_status is SemanticExecutionStatus.FAILED
    assert snapshot.failure_codes == ("SEMANTIC_REBUILD_FAILED",)
    assert snapshot.output_product_bindings == ()
    assert snapshot.evidence_universe_hash == universe.evidence_universe_hash
    assert snapshot.snapshot_hash == engine.failed_snapshot(
        run_ref=_RUN, evidence_universe=universe
    ).snapshot_hash

    async def publish_and_recover() -> None:
        artifacts = _ArtifactStore()
        runs = _RunRepository()
        reducer = RunStateReducer()
        state = reducer.reduce(
            run_ref=_RUN,
            run_control={
                "status": "RUNNING",
                "phase": "RESEARCHING",
                "budget": {},
                "usage": {},
                "termination_reason": None,
            },
            evidence_universe=universe,
            semantic_snapshot=snapshot,
            execution_refs=(
                ("semantic_snapshot_candidate", snapshot.artifact_ref),
                ("semantic_snapshot_hash", snapshot.snapshot_hash),
            ),
        )
        publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        await publisher.publish(state=state, semantic_snapshot=snapshot)
        restored = await publisher.load_current(_RUN)
        assert restored is not None
        assert restored.semantic_execution_status is SemanticExecutionStatus.FAILED
        assert await publisher.load_semantic_snapshot(restored) == snapshot

    _run(publish_and_recover())


def test_cross_version_bundle_fails_closed_without_binding_mismatched_product() -> None:
    bundle = _bundle()
    bundle["schema_versions"] = {
        **bundle["schema_versions"],  # type: ignore[typeddict-item]
        "atomic_proposition": "atomic-proposition-schema-v2",
    }
    snapshot = _run(
        V23DeterministicSemanticEngine().evaluate(
            run_ref=_RUN,
            evidence_universe=_universe(),
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": f"runs/{_RUN}/semantic-shadow/bundle.json",
            },
        )
    )
    assert snapshot.execution_status is SemanticExecutionStatus.INCOMPLETE
    assert "propositions" not in {item.product_kind for item in snapshot.output_product_bindings}


def test_v3_genesis_is_published_and_cold_loaded_from_pointer() -> None:
    async def scenario() -> None:
        artifacts = _ArtifactStore()
        runs = _RunRepository()
        reducer = RunStateReducer()
        publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        genesis = await publisher.initialize_genesis(run_ref=_RUN, reducer=reducer)
        assert genesis.state_version == 1
        assert genesis.semantic_state_ref is None
        assert genesis.semantic_execution_status is SemanticExecutionStatus.INCOMPLETE

        fresh_publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        restored = await fresh_publisher.initialize_genesis(run_ref=_RUN, reducer=reducer)
        assert restored == genesis
        assert runs.state["latest_run_state_ref"] == genesis.state_ref

    _run(scenario())


def test_real_v23_rebuild_boundary_publishes_snapshot_and_run_state() -> None:
    async def scenario() -> None:
        artifacts = _ArtifactStore()
        runs = _EmptyRunRepository()
        reducer = RunStateReducer()
        publisher = RunStatePublisher(artifacts=artifacts, runs=runs)  # type: ignore[arg-type]
        await publisher.initialize_genesis(run_ref=_RUN, reducer=reducer)
        budget = SemanticRuntimeBudget(
            max_ap_decomposition_calls=0,
            max_source_semantic_extraction_calls=0,
            max_identity_resolution_calls=0,
            max_formation_validation_calls=0,
            max_evidence_inputs=0,
            max_propositions_per_evidence=0,
            max_source_assertions_per_evidence=0,
            max_candidate_pairs=0,
            max_direction_assessments=0,
        )
        pipeline = SemanticResearchShadowPipeline(
            repository=object(),  # type: ignore[arg-type]
            bindings=object(),  # type: ignore[arg-type]
            cipher=object(),  # type: ignore[arg-type]
            gateway=object(),  # type: ignore[arg-type]
            artifacts=artifacts,  # type: ignore[arg-type]
            budget=budget,
            run_state_runs=runs,
            run_state_reducer=reducer,
            run_state_publisher=publisher,
            semantic_engine=V23DeterministicSemanticEngine(),
        )
        pipeline._run_orchestrator_versions[_RUN] = "V3"
        await pipeline._rebuild_deterministic_assessments(_RUN)
        bundle_ref, bundle = await pipeline._persist_snapshot_document(_RUN)
        await pipeline._publish_v3_run_state(_RUN, bundle_ref, bundle)

        restored = await RunStatePublisher(artifacts=artifacts, runs=runs).load_current(_RUN)  # type: ignore[arg-type]
        assert restored is not None
        assert restored.state_version == 2
        assert restored.semantic_execution_status is SemanticExecutionStatus.COMPLETE
        assert restored.gap_projection_status.value == "DERIVED"
        assert restored.gap_refs == ()
        assert restored.gap_projection_hash == canonical_hash([])
        assert restored.intent_refs == ()
        control_refs = dict(restored.execution_refs)
        assert control_refs["gap_projection_artifact_ref"] in artifacts.values
        assert control_refs["intent_projection_artifact_ref"] in artifacts.values
        assert restored.semantic_state_ref is not None
        semantic = await RunStatePublisher(
            artifacts=artifacts, runs=runs  # type: ignore[arg-type]
        ).load_semantic_snapshot(restored)
        assert semantic is not None
        empty_universe_hash = EvidenceUniverse.build(_RUN, ()).evidence_universe_hash
        assert semantic.evidence_universe_hash == empty_universe_hash

    _run(scenario())
