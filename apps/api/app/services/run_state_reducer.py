"""Deterministic projection of durable control and formal semantic state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from app.domain.run_state import (
    EvidenceUniverse,
    GapProjectionStatus,
    ResearchRunState,
    SemanticExecutionStatus,
    SemanticStateSnapshot,
    build_research_run_state,
    canonical_hash,
)


class RunStateConsistencyError(ValueError):
    """Raised when the inputs cannot safely produce a canonical run state."""


class RunStateReducer:
    """Project facts and references; never computes semantic truth."""

    def reduce(
        self,
        *,
        run_ref: UUID,
        run_control: Mapping[str, Any],
        evidence_universe: EvidenceUniverse,
        semantic_snapshot: SemanticStateSnapshot | None,
        previous: ResearchRunState | None = None,
        gap_projection_status: GapProjectionStatus = GapProjectionStatus.NOT_AVAILABLE,
        gap_refs: tuple[str, ...] = (),
        gap_source_semantic_hash: str | None = None,
        gap_projection_hash: str | None = None,
        intent_refs: tuple[str, ...] = (),
        intent_source_gap_hash: str | None = None,
        known_limitations: tuple[str, ...] = (),
        capability_states: tuple[str, ...] = (),
        deferred_states: tuple[str, ...] = (),
        execution_refs: tuple[tuple[str, str], ...] = (),
    ) -> ResearchRunState:
        if evidence_universe.run_ref != run_ref:
            raise RunStateConsistencyError("evidence_universe_run_mismatch")
        if previous is not None and previous.run_ref != run_ref:
            raise RunStateConsistencyError("parent_state_run_mismatch")

        semantic_status = SemanticExecutionStatus.INCOMPLETE
        semantic_ref: str | None = None
        semantic_hash: str | None = None
        semantic_engine_version = previous.semantic_engine_version if previous else None
        semantic_schema_version = previous.semantic_schema_version if previous else None
        semantic_limitations = set(known_limitations)
        semantic_capabilities = set(capability_states)
        semantic_deferred = set(deferred_states)
        if semantic_snapshot is None:
            semantic_limitations.add("SEMANTIC_STATE_NOT_AVAILABLE")
        else:
            if semantic_snapshot.run_ref != run_ref:
                raise RunStateConsistencyError("semantic_snapshot_run_mismatch")
            if previous is not None and previous.semantic_engine_version is not None and (
                semantic_snapshot.semantic_engine_version != previous.semantic_engine_version
                or semantic_snapshot.schema_version != previous.semantic_schema_version
            ):
                raise RunStateConsistencyError("semantic_snapshot_engine_version_mismatch")
            if (
                semantic_snapshot.evidence_universe_hash
                != evidence_universe.evidence_universe_hash
                or semantic_snapshot.semantic_evidence_hash
                != evidence_universe.semantic_evidence_hash
                or semantic_snapshot.semantic_evidence_refs
                != evidence_universe.semantic_evidence_refs
            ):
                raise RunStateConsistencyError("semantic_snapshot_evidence_universe_mismatch")
            semantic_status = semantic_snapshot.execution_status
            semantic_engine_version = semantic_snapshot.semantic_engine_version
            semantic_schema_version = semantic_snapshot.schema_version
            semantic_limitations.update(semantic_snapshot.known_limitations)
            semantic_capabilities.update(semantic_snapshot.capability_states)
            semantic_deferred.update(semantic_snapshot.deferred_states)
            if semantic_status is SemanticExecutionStatus.COMPLETE:
                semantic_ref = semantic_snapshot.artifact_ref
                semantic_hash = semantic_snapshot.snapshot_hash
            else:
                semantic_limitations.add(f"SEMANTIC_EXECUTION_{semantic_status.value}")

        if gap_projection_status is GapProjectionStatus.DERIVED:
            if semantic_snapshot is None or semantic_status is not SemanticExecutionStatus.COMPLETE:
                raise RunStateConsistencyError("gap_projection_requires_complete_semantic_state")
            if gap_source_semantic_hash != semantic_snapshot.snapshot_hash:
                raise RunStateConsistencyError("gap_projection_semantic_state_mismatch")
            canonical_gap_refs = tuple(sorted(set(gap_refs)))
            if gap_projection_hash != canonical_hash(list(canonical_gap_refs)):
                raise RunStateConsistencyError("gap_projection_hash_mismatch")
            if intent_refs and intent_source_gap_hash != gap_projection_hash:
                raise RunStateConsistencyError("intent_projection_gap_binding_mismatch")
        elif gap_refs or intent_refs:
            raise RunStateConsistencyError("unavailable_gap_projection_has_refs")
        elif gap_projection_hash is not None or intent_source_gap_hash is not None:
            raise RunStateConsistencyError("unavailable_gap_projection_has_hash")

        canonical_gaps = (
            tuple(sorted(set(gap_refs)))
            if gap_projection_status is GapProjectionStatus.DERIVED
            else ()
        )
        canonical_intents = (
            tuple(sorted(set(intent_refs)))
            if gap_projection_status is GapProjectionStatus.DERIVED
            else ()
        )
        budget_state_hash = canonical_hash(run_control.get("budget", {}))
        usage_state_hash = canonical_hash(run_control.get("usage", {}))
        desired = {
            "lifecycle_status": str(run_control["status"]),
            "phase": str(run_control["phase"]),
            "evidence_universe_hash": evidence_universe.evidence_universe_hash,
            "semantic_evidence_hash": evidence_universe.semantic_evidence_hash,
            "semantic_state_ref": semantic_ref,
            "semantic_state_hash": semantic_hash,
            "semantic_engine_version": semantic_engine_version,
            "semantic_schema_version": semantic_schema_version,
            "semantic_execution_status": semantic_status,
            "gap_projection_status": gap_projection_status,
            "gap_refs": canonical_gaps,
            "gap_source_semantic_hash": (
                gap_source_semantic_hash
                if gap_projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            "gap_projection_hash": (
                gap_projection_hash
                if gap_projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            "intent_refs": canonical_intents,
            "intent_source_gap_hash": (
                intent_source_gap_hash
                if gap_projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            "budget_state_hash": budget_state_hash,
            "usage_state_hash": usage_state_hash,
            "known_limitations": tuple(sorted(semantic_limitations)),
            "capability_states": tuple(sorted(semantic_capabilities)),
            "deferred_states": tuple(sorted(semantic_deferred)),
            "budget_projection": _canonical_budget(run_control.get("budget", {})),
            "termination_reason": (
                str(run_control["termination_reason"])
                if run_control.get("termination_reason") is not None
                else None
            ),
            "execution_refs": tuple(sorted(set(execution_refs))),
        }
        if previous is not None and all(
            getattr(previous, field) == value for field, value in desired.items()
        ):
            return previous

        state_version = 1 if previous is None else previous.state_version + 1
        state = build_research_run_state(
            run_ref=run_ref,
            state_version=state_version,
            parent_state_ref=previous.state_ref if previous else None,
            parent_state_hash=previous.state_hash if previous else None,
            expected_parent_version=previous.state_version if previous else None,
            lifecycle_status=str(run_control["status"]),
            phase=str(run_control["phase"]),
            evidence_universe_hash=evidence_universe.evidence_universe_hash,
            semantic_evidence_hash=evidence_universe.semantic_evidence_hash,
            semantic_state_ref=semantic_ref,
            semantic_state_hash=semantic_hash,
            semantic_engine_version=semantic_engine_version,
            semantic_schema_version=semantic_schema_version,
            semantic_execution_status=semantic_status,
            gap_projection_status=gap_projection_status,
            gap_refs=canonical_gaps,
            gap_source_semantic_hash=(
                gap_source_semantic_hash
                if gap_projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            gap_projection_hash=(
                gap_projection_hash
                if gap_projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            intent_refs=canonical_intents,
            intent_source_gap_hash=(
                intent_source_gap_hash
                if gap_projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            budget_state_hash=budget_state_hash,
            usage_state_hash=usage_state_hash,
            known_limitations=tuple(sorted(semantic_limitations)),
            capability_states=tuple(sorted(semantic_capabilities)),
            deferred_states=tuple(sorted(semantic_deferred)),
            budget_projection=_canonical_budget(run_control.get("budget", {})),
            termination_reason=(
                str(run_control["termination_reason"])
                if run_control.get("termination_reason") is not None
                else None
            ),
            execution_refs=tuple(sorted(set(execution_refs))),
        )
        return state


def _canonical_budget(value: object) -> tuple[tuple[str, int | str | bool | None], ...]:
    if not isinstance(value, Mapping):
        return ()
    projected: list[tuple[str, int | str | bool | None]] = []
    for key, item in value.items():
        if isinstance(item, (int, str, bool)) or item is None:
            projected.append((str(key), item))
    return tuple(sorted(projected))
