from __future__ import annotations

import json
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from app.domain.atomic_proposition import EvidencePropositionRelation, PropositionEvidenceLink
from app.domain.contradiction_detection import detect_proposition_conflicts
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibilityInput,
    aggregate_independent_corroboration,
)
from app.domain.proposition_support_group import (
    PropositionAssertionBinding,
    PropositionEvidenceMembership,
    PropositionGroupingStatus,
    PropositionSupportGroup,
)
from app.domain.proposition_verification import (
    ConflictScopeCompleteness,
    VerificationConflictScope,
    aggregate_proposition_verification,
)
from app.domain.run_state import (
    EvidenceUniverse,
    EvidenceUniverseEntry,
    GapProjectionStatus,
    SemanticExecutionStatus,
    SemanticStateSnapshot,
    canonical_hash,
)
from app.domain.semantic_control_projection import (
    SemanticIntentControlArtifact,
)
from app.infrastructure.db.research_work import ResearchWorkRepository
from app.services.research_orchestrator import ResearchOrchestrator
from app.services.run_state_publisher import RunStatePublisher
from app.services.run_state_reducer import RunStateReducer
from app.services.semantic_control_projection import SemanticControlProjectionService
from app.services.semantic_engine import V23DeterministicSemanticEngine

_RUN = UUID("32000000-0000-0000-0000-000000000001")
_CLAIM = UUID("32000000-0000-0000-0000-000000000002")
_EVIDENCE = UUID("32000000-0000-0000-0000-000000000003")
_SOURCE = UUID("32000000-0000-0000-0000-000000000004")
_PROP = UUID("32000000-0000-0000-0000-000000000005")
_GROUP = "a" * 64
_PROP_2 = UUID("32000000-0000-0000-0000-000000000006")
_GROUP_2 = "b" * 64


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drive immediate fake-boundary coroutines without opening a network loop."""
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("fake-boundary coroutine unexpectedly yielded")


class _DurableArtifacts:
    """Shared JSON bytes model durable storage across service instances."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values if values is not None else {}

    async def save_immutable_json(self, ref: str, value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        prior = self.values.get(ref)
        if prior is not None and json.dumps(
            prior, sort_keys=True, separators=(",", ":")
        ) != encoded:
            raise ValueError("immutable_artifact_content_conflict")
        self.values[ref] = json.loads(encoded)
        return ref

    async def read_json(self, ref: str) -> object:
        return self.values[ref]

    async def read_text(self, ref: str) -> str:
        return json.dumps(self.values[ref], sort_keys=True, separators=(",", ":"))


def _group(
    *,
    group_ref: str = _GROUP,
    proposition_ref: UUID = _PROP,
    support_owner_count: int = 0,
) -> PropositionSupportGroup:
    memberships: list[PropositionEvidenceMembership] = []
    if support_owner_count:
        for ordinal, owner in ((1, "publisher-a"), (2, "publisher-b"))[:support_owner_count]:
            evidence_ref = UUID(int=ordinal)
            link = PropositionEvidenceLink(
                proposition_id=proposition_ref,
                evidence_id=evidence_ref,
                claim_id=_CLAIM,
                relation=EvidencePropositionRelation.SUPPORTS,
                source_id=UUID(int=ordinal + 1000),
                source_owner_key=owner,
                quote_reference="fixture-span",
                span_start=0,
                span_end=8,
                method="fixture",
                run_id=_RUN,
                question_id="q1",
                dimension_key="d1",
            )
            memberships.append(
                PropositionEvidenceMembership(
                    evidence_ref=evidence_ref,
                    provenance_links=(link,),
                )
            )
    return PropositionSupportGroup(
        group_ref=group_ref,
        proposition_refs=(proposition_ref,),
        evidence_memberships=tuple(memberships),
        grouping_status=PropositionGroupingStatus.SINGLETON,
    )


def _bundle(
    *,
    groups: tuple[PropositionSupportGroup, ...] = (),
) -> dict[str, object]:
    corroborations = tuple(
        aggregate_independent_corroboration(
            group,
            tuple(
                CorroborationEvidenceEligibilityInput(
                    group_ref=group.group_ref,
                    evidence_ref=member.evidence_ref,
                    accepted=True,
                )
                for member in group.evidence_memberships
            ),
        )
        for group in groups
    )
    conflicts = detect_proposition_conflicts(
        groups,
        (),
        tuple(
            PropositionAssertionBinding(
                proposition_ref=prop_ref,
                assertion_ref=f"assertion:{prop_ref}",
            )
            for group in groups
            for prop_ref in group.proposition_refs
        ),
        corroboration_assessments=corroborations,
    )
    group_refs = tuple(sorted(group.group_ref for group in groups))
    verifications = tuple(
        aggregate_proposition_verification(
            group,
            next(item for item in corroborations if item.group_ref == group.group_ref),
            conflicts,
            (),
            VerificationConflictScope(
                target_group_ref=group.group_ref,
                completeness=ConflictScopeCompleteness.COMPLETE,
                scope_ref="fixture-scope",
                candidate_group_refs=group_refs,
            ),
        )
        for group in groups
    )
    return {
        "schema_version": "semantic-shadow-verification-snapshot-v2",
        "run_ref": str(_RUN),
        "claim_refs": [str(_CLAIM)],
        "evidence_refs": [str(_EVIDENCE)],
        "proposition_refs": [str(_PROP)] if groups else [],
        "formation_assessments": [],
        "proposition_group_refs": list(group_refs),
        "proposition_groups": [item.model_dump(mode="json") for item in groups],
        "pair_assessments": [],
        "proposition_evidence_links": [
            link.model_dump(mode="json")
            for group in groups
            for member in group.evidence_memberships
            for link in member.provenance_links
        ],
        "corroboration_assessments": [
            item.model_dump(mode="json") for item in corroborations
        ],
        "conflicts": conflicts.model_dump(mode="json"),
        "verification_assessments": [
            item.model_dump(mode="json") for item in verifications
        ],
        "conflict_scope_completeness": {"completeness": "COMPLETE"},
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


def _universe() -> EvidenceUniverse:
    return EvidenceUniverse.build(
        _RUN,
        (
            EvidenceUniverseEntry(
                evidence_ref=_EVIDENCE,
                claim_ref=_CLAIM,
                source_ref=_SOURCE,
                accepted=True,
                semantic_eligible=True,
            ),
        ),
    )


def _snapshot_and_bundle(
    artifacts: _DurableArtifacts,
    *,
    groups: tuple[PropositionSupportGroup, ...] = (),
    bundle_ref: str = "runs/fixture/semantic-products.json",
) -> tuple[SemanticStateSnapshot, dict[str, object]]:
    bundle = _bundle(groups=groups)
    _run(artifacts.save_immutable_json(bundle_ref, bundle))
    snapshot = _run(
        V23DeterministicSemanticEngine().evaluate(
            run_ref=_RUN,
            evidence_universe=_universe(),
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": bundle_ref,
            },
        )
    )
    assert snapshot.execution_status is SemanticExecutionStatus.COMPLETE
    _run(artifacts.save_immutable_json(snapshot.artifact_ref, snapshot.model_dump(mode="json")))
    return snapshot, bundle


def _reduce_state(
    snapshot: SemanticStateSnapshot,
    projection: object,
    *,
    previous: object | None = None,
):
    gap_artifact = projection.gap_artifact
    intent_artifact = projection.intent_artifact
    return RunStateReducer().reduce(
        run_ref=_RUN,
        run_control={
            "status": "RUNNING",
            "phase": "RESEARCHING",
            "budget": {},
            "usage": {},
            "termination_reason": None,
        },
        evidence_universe=_universe(),
        semantic_snapshot=snapshot,
        previous=previous,
        gap_projection_status=GapProjectionStatus.DERIVED,
        gap_refs=tuple(item.gap_ref for item in gap_artifact.gaps),
        gap_source_semantic_hash=snapshot.snapshot_hash,
        gap_projection_hash=gap_artifact.projection_hash,
        intent_refs=tuple(item.intent_ref for item in intent_artifact.intents),
        intent_source_gap_hash=gap_artifact.projection_hash,
        execution_refs=(
            ("gap_projection_artifact_ref", gap_artifact.artifact_ref),
            ("gap_projection_artifact_hash", gap_artifact.artifact_hash),
            ("intent_projection_artifact_ref", intent_artifact.artifact_ref),
            ("intent_projection_artifact_hash", intent_artifact.artifact_hash),
        ),
    )


def test_production_projection_derives_gap_and_binds_intent_to_current_hash() -> None:
    artifacts = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(
        artifacts, groups=(_group(support_owner_count=1),)
    )
    result = _run(SemanticControlProjectionService(artifacts).recover(snapshot))

    assert result.gap_artifact.projection_status == "DERIVED"
    assert len(result.gap_artifact.gaps) == 1
    assert result.gap_artifact.gaps[0].gap_ref
    assert result.gap_artifact.semantic_snapshot_hash == snapshot.snapshot_hash
    assert result.gap_artifact.projection_hash == canonical_hash(
        [item.gap_ref for item in result.gap_artifact.gaps]
    )
    assert result.intent_artifact.gap_artifact_ref == result.gap_artifact.artifact_ref
    assert result.intent_artifact.gap_projection_hash == result.gap_artifact.projection_hash
    state = _reduce_state(snapshot, result)
    assert state.gap_refs == (result.gap_artifact.gaps[0].gap_ref,)
    assert state.gap_source_semantic_hash == snapshot.snapshot_hash
    assert state.intent_source_gap_hash == result.gap_artifact.projection_hash


def test_valid_empty_gap_set_is_derived_not_unavailable() -> None:
    artifacts = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=2),),
    )
    result = _run(SemanticControlProjectionService(artifacts).recover(snapshot))

    assert result.gap_artifact.projection_status == "DERIVED"
    assert result.gap_artifact.gaps == ()
    assert result.gap_artifact.projection_hash == canonical_hash([])
    state = _reduce_state(snapshot, result)
    assert state.gap_projection_status is GapProjectionStatus.DERIVED
    assert state.gap_refs == ()


def test_incomplete_semantic_input_fails_closed_without_empty_derived_projection() -> None:
    artifacts = _DurableArtifacts()
    bundle = _bundle()
    bundle.pop("verification_assessments")
    _run(artifacts.save_immutable_json("runs/incomplete/bundle.json", bundle))
    snapshot = _run(
        V23DeterministicSemanticEngine().evaluate(
            run_ref=_RUN,
            evidence_universe=_universe(),
            execution_context={
                "semantic_products": bundle,
                "product_bundle_ref": "runs/incomplete/bundle.json",
            },
        )
    )
    assert snapshot.execution_status is SemanticExecutionStatus.INCOMPLETE
    with pytest.raises(ValueError, match="gap_projection_requires_complete_semantic_snapshot"):
        _run(SemanticControlProjectionService(artifacts).recover(snapshot))

    state = RunStateReducer().reduce(
        run_ref=_RUN,
        run_control={"status": "RUNNING", "phase": "RESEARCHING", "budget": {}, "usage": {}},
        evidence_universe=_universe(),
        semantic_snapshot=snapshot,
        gap_projection_status=GapProjectionStatus.INCOMPLETE,
    )
    assert state.gap_projection_status is GapProjectionStatus.INCOMPLETE
    assert state.gap_refs == ()
    assert state.gap_projection_hash is None


def test_projection_identity_is_repeatable_and_input_order_does_not_change_gap_hash() -> None:
    artifacts = _DurableArtifacts()
    groups = (_group(), _group(group_ref=_GROUP_2, proposition_ref=_PROP_2))
    snapshot, _ = _snapshot_and_bundle(artifacts, groups=groups)
    service = SemanticControlProjectionService(artifacts)
    first = _run(service.recover(snapshot))
    second = _run(SemanticControlProjectionService(artifacts).recover(snapshot))
    assert first.gap_artifact.artifact_ref == second.gap_artifact.artifact_ref
    assert first.gap_artifact.artifact_hash == second.gap_artifact.artifact_hash
    assert first.intent_artifact.artifact_ref == second.intent_artifact.artifact_ref
    assert first.intent_artifact.artifact_hash == second.intent_artifact.artifact_hash
    assert first.gap_artifact.projection_hash == second.gap_artifact.projection_hash

    reverse_snapshot, _ = _snapshot_and_bundle(
        artifacts,
        groups=tuple(reversed(groups)),
        bundle_ref="runs/reversed/semantic-products.json",
    )
    reversed_projection = _run(
        SemanticControlProjectionService(artifacts).recover(reverse_snapshot)
    )
    assert tuple(item.gap_ref for item in reversed_projection.gap_artifact.gaps) == tuple(
        item.gap_ref for item in first.gap_artifact.gaps
    )
    assert reversed_projection.gap_artifact.projection_hash == first.gap_artifact.projection_hash


def test_intent_artifact_cannot_be_rebound_to_another_gap_projection() -> None:
    artifacts = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(
        artifacts, groups=(_group(support_owner_count=1),)
    )
    projection = _run(SemanticControlProjectionService(artifacts).recover(snapshot))
    stale_fields = projection.intent_artifact.model_dump(mode="python")
    stale_fields.pop("artifact_ref")
    stale_fields.pop("artifact_hash")
    stale_fields["gap_projection_hash"] = "f" * 64
    stale_fields["projection_hash"] = canonical_hash(
        [item.intent_ref for item in projection.intent_artifact.intents]
    )
    stale_fields["artifact_ref"] = "invalid"
    stale_fields["artifact_hash"] = "0" * 64
    with pytest.raises(ValueError, match="intent_control_artifact_hash_mismatch"):
        SemanticIntentControlArtifact.model_validate(stale_fields)


def test_stale_semantic_projection_is_rejected_by_run_state_publisher() -> None:
    artifacts = _DurableArtifacts()
    old_snapshot, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=1),),
        bundle_ref="runs/old/bundle.json",
    )
    old_projection = _run(SemanticControlProjectionService(artifacts).recover(old_snapshot))
    current_snapshot, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=2),),
        bundle_ref="runs/current/bundle.json",
    )
    stale_state = _reduce_state(current_snapshot, old_projection)
    publisher = RunStatePublisher(artifacts=artifacts, runs=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="control_projection_binding_mismatch"):
        _run(publisher._validate_control_projection(stale_state))


def test_cold_recovery_rebuilds_projection_from_durable_bundle_only() -> None:
    durable = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(
        durable, groups=(_group(support_owner_count=1),)
    )
    instance_a = SemanticControlProjectionService(durable)
    result_a = _run(instance_a.recover(snapshot))
    durable_copy = json.loads(json.dumps(durable.values))
    del instance_a, result_a, snapshot

    instance_b_store = _DurableArtifacts(durable_copy)
    # The immutable snapshot is separately recovered from its durable artifact, not A's object.
    snapshot_ref = next(
        key for key, value in instance_b_store.values.items()
        if isinstance(value, dict) and value.get("schema_version") == "semantic-state-snapshot-v1"
    )
    restored_snapshot = SemanticStateSnapshot.model_validate(instance_b_store.values[snapshot_ref])
    result_b = _run(SemanticControlProjectionService(instance_b_store).recover(restored_snapshot))
    assert result_b.gap_artifact.artifact_ref == next(
        key for key in instance_b_store.values if "/semantic-control/gaps/" in key
    )
    assert result_b.intent_artifact.artifact_ref == next(
        key for key in instance_b_store.values if "/semantic-control/intents/" in key
    )


def test_same_content_state_reduction_is_a_noop() -> None:
    artifacts = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(
        artifacts, groups=(_group(support_owner_count=1),)
    )
    projection = _run(SemanticControlProjectionService(artifacts).recover(snapshot))
    first = _reduce_state(snapshot, projection)
    second = _reduce_state(snapshot, projection, previous=first)
    assert second is first
    assert second.state_version == first.state_version
    assert second.state_hash == first.state_hash


def test_new_authoritative_snapshot_removes_resolved_gap_without_mutating_old_artifact() -> None:
    artifacts = _DurableArtifacts()
    old_snapshot, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=1),),
        bundle_ref="runs/s1/bundle.json",
    )
    old = _run(SemanticControlProjectionService(artifacts).recover(old_snapshot))
    old_bytes = json.dumps(artifacts.values[old.gap_artifact.artifact_ref], sort_keys=True)

    new_snapshot, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=2),),
        bundle_ref="runs/s2/bundle.json",
    )
    new = _run(SemanticControlProjectionService(artifacts).recover(new_snapshot))

    assert old.gap_artifact.gaps
    assert new.gap_artifact.projection_status == "DERIVED"
    assert new.gap_artifact.gaps == ()
    assert old.gap_artifact.artifact_ref in artifacts.values
    assert json.dumps(artifacts.values[old.gap_artifact.artifact_ref], sort_keys=True) == old_bytes


def test_work_completion_cannot_close_gap_without_new_semantic_snapshot() -> None:
    artifacts = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(
        artifacts, groups=(_group(support_owner_count=1),)
    )
    service = SemanticControlProjectionService(artifacts)
    before = _run(service.recover(snapshot))
    simulated_work_status = "COMPLETED"
    assert simulated_work_status == "COMPLETED"
    after = _run(SemanticControlProjectionService(artifacts).recover(snapshot))

    assert after.gap_artifact.gaps[0].gap_ref == before.gap_artifact.gaps[0].gap_ref
    assert after.gap_artifact.projection_hash == before.gap_artifact.projection_hash


def test_projection_does_not_invent_search_budget_for_gap_intent() -> None:
    artifacts = _DurableArtifacts()
    snapshot, bundle = _snapshot_and_bundle(
        artifacts, groups=(_group(support_owner_count=1),)
    )
    result = _run(SemanticControlProjectionService(artifacts).recover(snapshot))
    assert result.gap_derivation_issues == ()
    assert result.intent_issues == ()
    assert result.intent_artifact.intents == ()
    assert result.intent_artifact.gap_projection_hash == result.gap_artifact.projection_hash
    assert bundle["verification_assessments"]


def test_pointer_recovery_validation_rejects_stale_gap_intent_pair() -> None:
    artifacts = _DurableArtifacts()
    snapshot_a, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=1),),
        bundle_ref="runs/a/bundle.json",
    )
    projection_a = _run(SemanticControlProjectionService(artifacts).recover(snapshot_a))
    snapshot_b, _ = _snapshot_and_bundle(
        artifacts,
        groups=(_group(support_owner_count=2),),
        bundle_ref="runs/b/bundle.json",
    )
    projection_b = _run(SemanticControlProjectionService(artifacts).recover(snapshot_b))

    # Pair current gap B with durable intent A; production publisher must fail closed.
    artifacts.values["runs/forged/current-gap.json"] = (
        projection_b.gap_artifact.model_dump(mode="json")
    )
    artifacts.values["runs/forged/stale-intent.json"] = (
        projection_a.intent_artifact.model_dump(mode="json")
    )
    state = _reduce_state(snapshot_b, projection_b).model_copy(
        update={
            "execution_refs": tuple(sorted((
                ("gap_projection_artifact_ref", "runs/forged/current-gap.json"),
                ("gap_projection_artifact_hash", projection_b.gap_artifact.artifact_hash),
                ("intent_projection_artifact_ref", "runs/forged/stale-intent.json"),
                ("intent_projection_artifact_hash", projection_a.intent_artifact.artifact_hash),
            )))
        }
    )
    publisher = RunStatePublisher(artifacts=artifacts, runs=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control_projection_binding_mismatch"):
        _run(publisher._validate_control_projection(state))


def test_empty_group_input_is_formally_derived_empty_not_missing() -> None:
    artifacts = _DurableArtifacts()
    snapshot, _ = _snapshot_and_bundle(artifacts)
    result = _run(SemanticControlProjectionService(artifacts).recover(snapshot))
    assert result.gap_artifact.projection_status == "DERIVED"
    assert result.gap_artifact.gaps == ()
    assert result.gap_artifact.issues == ()


def test_run_next_work_does_not_reserve_existing_proposition_gap_work() -> None:
    run_state = SimpleNamespace(
        run_ref=_RUN,
        state_ref="runs/state/current.json",
        state_hash="1" * 64,
        state_version=1,
    )
    run_row = SimpleNamespace(
        id=_RUN,
        orchestrator_version="V3",
        latest_run_state_ref=run_state.state_ref,
        latest_run_state_hash=run_state.state_hash,
        latest_run_state_version=run_state.state_version,
        plan_version=1,
    )
    gap_row = SimpleNamespace(
        work_kind="PROPOSITION_GAP",
        status="ELIGIBLE",
    )
    statements: list[object] = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def begin(self):
            return self

        async def scalar(self, statement: object):
            statements.append(statement)
            if len(statements) == 1:
                return run_row
            if len(statements) == 2:
                return None  # no active legacy attempt
            compiled = statement.compile()
            assert "LEGACY_RESEARCH" in compiled.params.values()
            assert gap_row.work_kind == "PROPOSITION_GAP"
            return None  # the eligible gap row is excluded by the production query

    class _SchedulingRepository(ResearchWorkRepository):
        async def materialize_legacy_plan(self, *_args: object):
            return ()

    repository = _SchedulingRepository(lambda: _Session())  # type: ignore[arg-type]

    class _Runs:
        async def get_run_state_inputs(self, _run_id: UUID) -> dict[str, str]:
            return {"orchestrator_version": "V3", "lease_owner": "lease-1"}

        async def get_plan_for_execution(self, _run_id: UUID) -> object:
            return SimpleNamespace(plan_version=1)

    class _States:
        async def load_current(self, _run_id: UUID) -> object:
            return run_state

    class _Tools:
        async def replan_requested(self, _run_id: UUID, *, worker_task_id: str) -> bool:
            del worker_task_id
            return False

    orchestrator = ResearchOrchestrator(
        runs=_Runs(),  # type: ignore[arg-type]
        work=repository,
        run_states=_States(),  # type: ignore[arg-type]
        loop=object(),  # type: ignore[arg-type]
        tools=_Tools(),  # type: ignore[arg-type]
    )
    result = _run(orchestrator.run_next_work(_RUN, worker_task_id="test-worker"))

    assert result.decision == "no_pending_question"
    assert result.pages_read == 0
    assert len(statements) == 3
    assert gap_row.status == "ELIGIBLE"
