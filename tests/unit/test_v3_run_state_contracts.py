from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from app.domain.run_state import (
    EvidenceUniverse,
    EvidenceUniverseEntry,
    GapProjectionStatus,
    SemanticExecutionStatus,
    SemanticProductBinding,
    SemanticStateSnapshot,
    build_research_run_state,
    canonical_hash,
)
from app.services.run_state_reducer import RunStateConsistencyError, RunStateReducer

_RUN = UUID("00000000-0000-0000-0000-000000000001")
_EVIDENCE_A = UUID("00000000-0000-0000-0000-000000000002")
_EVIDENCE_B = UUID("00000000-0000-0000-0000-000000000003")
_SOURCE = UUID("00000000-0000-0000-0000-000000000004")
_CLAIM = UUID("00000000-0000-0000-0000-000000000005")
_HASH_A = "a" * 64


def _universe() -> EvidenceUniverse:
    return EvidenceUniverse.build(
        _RUN,
        (
            EvidenceUniverseEntry(
                evidence_ref=_EVIDENCE_B,
                claim_ref=_CLAIM,
                source_ref=_SOURCE,
                source_snapshot_ref=UUID("00000000-0000-0000-0000-000000000006"),
                source_content_hash=_HASH_A,
                accepted=False,
                semantic_eligible=False,
            ),
            EvidenceUniverseEntry(
                evidence_ref=_EVIDENCE_A,
                claim_ref=_CLAIM,
                source_ref=_SOURCE,
                source_snapshot_ref=UUID("00000000-0000-0000-0000-000000000006"),
                source_content_hash=_HASH_A,
                accepted=True,
                semantic_eligible=True,
            ),
        ),
    )


def _semantic(universe: EvidenceUniverse) -> SemanticStateSnapshot:
    versions = (("verification", "v1"),)
    execution_ref = SemanticStateSnapshot.deterministic_execution_ref(
        run_ref=_RUN,
        semantic_engine_version="v2.3-rebuild-v1",
        evidence_universe_hash=universe.evidence_universe_hash,
        semantic_evidence_hash=universe.semantic_evidence_hash,
        required_product_versions=versions,
    )
    product = SemanticProductBinding(
        product_kind="verification",
        product_ref="ref-v",
        content_hash=_HASH_A,
        product_version="v1",
        run_ref=_RUN,
        evidence_universe_hash=universe.evidence_universe_hash,
        semantic_execution_ref=execution_ref,
        semantic_engine_version="v2.3-rebuild-v1",
    )
    return SemanticStateSnapshot.build(
        run_ref=_RUN,
        semantic_execution_ref=execution_ref,
        semantic_engine_version="v2.3-rebuild-v1",
        evidence_universe_hash=universe.evidence_universe_hash,
        semantic_evidence_hash=universe.semantic_evidence_hash,
        semantic_input_hash=_HASH_A,
        semantic_output_hash=canonical_hash([product.model_dump(mode="json")]),
        semantic_evidence_refs=universe.semantic_evidence_refs,
        required_product_versions=versions,
        required_product_refs=(("verification", "ref-v", _HASH_A),),
        output_product_refs=(("verification", "ref-v", _HASH_A),),
        output_product_bindings=(product,),
        semantic_statuses=(("proposition-1", "UNKNOWN"),),
        known_limitations=(),
        capability_states=(),
        deferred_states=(),
        execution_status=SemanticExecutionStatus.COMPLETE,
        failure_codes=(),
    )


def test_evidence_universe_is_order_independent_and_distinguishes_semantic_subset() -> None:
    first = _universe()
    entries = tuple(reversed(first.entries))
    second = EvidenceUniverse.build(_RUN, entries)
    assert first.evidence_universe_hash == second.evidence_universe_hash
    assert first.semantic_evidence_hash == second.semantic_evidence_hash
    assert first.semantic_evidence_refs == (_EVIDENCE_A,)
    assert len(first.entries) == 2


def test_evidence_universe_rejects_wrong_hash() -> None:
    universe = _universe()
    with pytest.raises(ValueError, match="evidence_universe_hash_mismatch"):
        EvidenceUniverse.model_validate(
            {
                **universe.model_dump(mode="python"),
                "evidence_universe_hash": "0" * 64,
            }
        )


def test_duplicate_evidence_observation_does_not_change_universe_identity() -> None:
    entry = _universe().entries[0]
    once = EvidenceUniverse.build(_RUN, (entry,))
    repeated = EvidenceUniverse.build(_RUN, (entry, entry))
    assert once.evidence_universe_hash == repeated.evidence_universe_hash
    assert once.semantic_evidence_hash == repeated.semantic_evidence_hash

    conflicting = entry.model_copy(update={"accepted": not entry.accepted})
    with pytest.raises(ValueError, match="duplicate_evidence_identity_conflict"):
        EvidenceUniverse.build(_RUN, (entry, conflicting))


def test_semantic_snapshot_hash_is_canonical_and_unknown_can_be_complete() -> None:
    semantic = _semantic(_universe())
    cold_rebuild = _semantic(_universe())
    assert semantic.execution_status is SemanticExecutionStatus.COMPLETE
    assert semantic.semantic_statuses == (("proposition-1", "UNKNOWN"),)
    assert semantic.semantic_execution_ref == cold_rebuild.semantic_execution_ref
    assert semantic.snapshot_hash == cold_rebuild.snapshot_hash
    assert semantic.snapshot_hash == SemanticStateSnapshot.compute_hash(
        semantic.model_dump(mode="json")
    )


def test_complete_snapshot_rejects_missing_required_product() -> None:
    execution_ref = SemanticStateSnapshot.deterministic_execution_ref(
        run_ref=_RUN,
        semantic_engine_version="v1",
        evidence_universe_hash=_HASH_A,
        semantic_evidence_hash=_HASH_A,
        required_product_versions=(("verification", "v1"),),
    )
    with pytest.raises(ValueError, match="required_semantic_product_missing"):
        SemanticStateSnapshot.build(
            run_ref=_RUN,
            semantic_execution_ref=execution_ref,
            semantic_engine_version="v1",
            evidence_universe_hash=_HASH_A,
            semantic_evidence_hash=_HASH_A,
            semantic_input_hash=_HASH_A,
            semantic_output_hash=canonical_hash([]),
            semantic_evidence_refs=(),
            required_product_versions=(("verification", "v1"),),
            required_product_refs=(("verification", "ref-v", _HASH_A),),
            output_product_refs=(),
            execution_status=SemanticExecutionStatus.COMPLETE,
        )


def test_semantic_snapshot_rejects_cross_universe_product_binding() -> None:
    universe = _universe()
    execution_ref = SemanticStateSnapshot.deterministic_execution_ref(
        run_ref=_RUN,
        semantic_engine_version="v2.3-rebuild-v1",
        evidence_universe_hash=universe.evidence_universe_hash,
        semantic_evidence_hash=universe.semantic_evidence_hash,
        required_product_versions=(("verification", "v1"),),
    )
    product = SemanticProductBinding(
        product_kind="verification",
        product_ref="ref-v",
        content_hash=_HASH_A,
        product_version="v1",
        run_ref=_RUN,
        evidence_universe_hash="b" * 64,
        semantic_execution_ref=execution_ref,
        semantic_engine_version="v2.3-rebuild-v1",
    )
    with pytest.raises(ValueError, match="semantic_product_binding_mismatch"):
        SemanticStateSnapshot.build(
            run_ref=_RUN,
            semantic_execution_ref=execution_ref,
            semantic_engine_version="v2.3-rebuild-v1",
            evidence_universe_hash=universe.evidence_universe_hash,
            semantic_evidence_hash=universe.semantic_evidence_hash,
            semantic_input_hash=_HASH_A,
            semantic_output_hash=canonical_hash([product.model_dump(mode="json")]),
            semantic_evidence_refs=universe.semantic_evidence_refs,
            required_product_versions=(("verification", "v1"),),
            required_product_refs=(("verification", "ref-v", _HASH_A),),
            output_product_refs=(("verification", "ref-v", _HASH_A),),
            output_product_bindings=(product,),
            execution_status=SemanticExecutionStatus.COMPLETE,
        )


def test_run_state_hash_excludes_created_at_and_has_artifact_identity() -> None:
    values = {
        "run_ref": _RUN,
        "state_version": 1,
        "parent_state_ref": None,
        "parent_state_hash": None,
        "expected_parent_version": None,
        "lifecycle_status": "RUNNING",
        "phase": "RESEARCHING",
        "evidence_universe_hash": _HASH_A,
        "semantic_evidence_hash": _HASH_A,
        "semantic_state_ref": None,
        "semantic_state_hash": None,
        "semantic_execution_status": SemanticExecutionStatus.INCOMPLETE,
        "gap_projection_status": GapProjectionStatus.NOT_AVAILABLE,
        "known_limitations": ("SEMANTIC_STATE_NOT_AVAILABLE",),
        "capability_states": (),
        "deferred_states": (),
        "budget_projection": (),
        "budget_state_hash": canonical_hash({}),
        "usage_state_hash": canonical_hash({}),
        "termination_reason": None,
        "execution_refs": (),
    }
    first = build_research_run_state(
        **values, created_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    second = build_research_run_state(
        **values, created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=1)
    )
    assert first.state_hash == second.state_hash
    assert first.state_ref == second.state_ref


def test_reducer_preserves_unknown_and_advances_lineage() -> None:
    universe = _universe()
    semantic = _semantic(universe)
    reducer = RunStateReducer()
    first = reducer.reduce(
        run_ref=_RUN,
        run_control={"status": "RUNNING", "phase": "RESEARCHING", "budget": {}},
        evidence_universe=universe,
        semantic_snapshot=semantic,
    )
    second = reducer.reduce(
        run_ref=_RUN,
        run_control={"status": "RUNNING", "phase": "VERIFYING", "budget": {}},
        evidence_universe=universe,
        semantic_snapshot=semantic,
        previous=first,
    )
    assert first.semantic_execution_status is SemanticExecutionStatus.COMPLETE
    assert second.semantic_execution_status is SemanticExecutionStatus.COMPLETE
    assert second.semantic_state_hash == semantic.snapshot_hash
    assert second.state_version == 2
    assert second.parent_state_ref == first.state_ref
    assert second.state_hash != first.state_hash


def test_same_content_replay_is_noop() -> None:
    universe = _universe()
    semantic = _semantic(universe)
    reducer = RunStateReducer()
    first = reducer.reduce(
        run_ref=_RUN,
        run_control={"status": "RUNNING", "phase": "RESEARCHING", "budget": {}},
        evidence_universe=universe,
        semantic_snapshot=semantic,
    )
    replay = reducer.reduce(
        run_ref=_RUN,
        run_control={"status": "RUNNING", "phase": "RESEARCHING", "budget": {}},
        evidence_universe=universe,
        semantic_snapshot=semantic,
        previous=first,
    )
    assert replay is first


def test_reducer_rejects_semantic_state_from_other_universe() -> None:
    universe = _universe()
    semantic = _semantic(universe).model_copy(update={"evidence_universe_hash": "b" * 64})
    with pytest.raises(RunStateConsistencyError, match="evidence_universe_mismatch"):
        RunStateReducer().reduce(
            run_ref=_RUN,
            run_control={"status": "RUNNING", "phase": "RESEARCHING"},
            evidence_universe=universe,
            semantic_snapshot=semantic,
        )


def test_gap_projection_requires_same_current_semantic_hash() -> None:
    universe = _universe()
    semantic = _semantic(universe)
    with pytest.raises(RunStateConsistencyError, match="gap_projection_semantic_state_mismatch"):
        RunStateReducer().reduce(
            run_ref=_RUN,
            run_control={"status": "RUNNING", "phase": "RESEARCHING"},
            evidence_universe=universe,
            semantic_snapshot=semantic,
            gap_projection_status=GapProjectionStatus.DERIVED,
            gap_refs=(),
            gap_source_semantic_hash="0" * 64,
        )


def test_canonical_hash_does_not_depend_on_mapping_insertion_order() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
