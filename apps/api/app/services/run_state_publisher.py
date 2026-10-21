"""Artifact-first publication and recovery for canonical run state."""

from __future__ import annotations

from uuid import UUID

from app.domain.run_state import (
    GapProjectionStatus,
    ResearchRunState,
    SemanticExecutionStatus,
    SemanticStateSnapshot,
    canonical_hash,
)
from app.domain.semantic_control_projection import (
    SemanticGapControlArtifact,
    SemanticIntentControlArtifact,
)
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.services.run_state_reducer import RunStateReducer

_PRODUCT_FIELDS = {
    "claims": "claim_refs",
    "propositions": "proposition_refs",
    "formation_assessments": "formation_assessments",
    "proposition_groups": "proposition_groups",
    "pair_assessments": "pair_assessments",
    "proposition_evidence_links": "proposition_evidence_links",
    "corroboration_assessments": "corroboration_assessments",
    "conflicts": "conflicts",
    "verification_assessments": "verification_assessments",
    "conflict_scope_completeness": "conflict_scope_completeness",
}
_PRODUCT_VERSION_FIELDS = {
    "claims": "semantic_snapshot",
    "propositions": "atomic_proposition",
    "formation_assessments": "target_formation_schema",
    "proposition_groups": "proposition_groups",
    "pair_assessments": "proposition_pair_identity",
    "proposition_evidence_links": "direction",
    "corroboration_assessments": "corroboration",
    "conflicts": "conflicts",
    "verification_assessments": "verification",
    "conflict_scope_completeness": "conflict_scope_completeness",
}


class RunStatePublisher:
    """Persist immutable outputs before attempting the DB latest-pointer CAS."""

    def __init__(
        self, *, artifacts: LocalArtifactStore, runs: ResearchRunRepository
    ) -> None:
        self._artifacts = artifacts
        self._runs = runs

    async def load_current(self, run_ref: UUID) -> ResearchRunState | None:
        """Restore authority exclusively from the DB pointer and immutable artifact."""

        inputs = await self._runs.get_run_state_inputs(run_ref)
        artifact_ref = inputs["latest_run_state_ref"]
        state_hash = inputs["latest_run_state_hash"]
        state_version = inputs["latest_run_state_version"]
        if artifact_ref is None and state_hash is None and state_version is None:
            return None
        if not isinstance(artifact_ref, str) or not isinstance(state_hash, str):
            raise ValueError("run_state_pointer_incomplete")
        raw = await self._artifacts.read_json(artifact_ref)
        if not isinstance(raw, dict):
            raise ValueError("run_state_artifact_invalid")
        state = ResearchRunState.model_validate(raw)
        if (
            state.run_ref != run_ref
            or state.state_ref != artifact_ref
            or state.state_hash != state_hash
            or state.state_version != state_version
        ):
            raise ValueError("run_state_pointer_artifact_mismatch")
        if state.semantic_state_ref is not None or dict(state.execution_refs).get(
            "semantic_snapshot_candidate"
        ) is not None:
            await self.load_semantic_snapshot(state)
        await self._validate_control_projection(state)
        return state

    async def initialize_genesis(
        self, *, run_ref: UUID, reducer: RunStateReducer
    ) -> ResearchRunState | None:
        """Durably create the first V3 state, or return the cold-loaded authority."""
        inputs = await self._runs.get_run_state_inputs(run_ref)
        if inputs["orchestrator_version"] != "V3":
            return None
        current = await self.load_current(run_ref)
        if current is not None:
            return current
        universe = await self._runs.load_evidence_universe(run_ref)
        state = reducer.reduce(
            run_ref=run_ref,
            run_control={
                "status": inputs["status"],
                "phase": inputs["phase"],
                "budget": inputs["budget"],
                "usage": inputs["usage"],
                "termination_reason": inputs["termination_reason"],
            },
            evidence_universe=universe,
            semantic_snapshot=None,
        )
        await self.publish(state=state, semantic_snapshot=None)
        return state

    async def load_semantic_snapshot(
        self, state: ResearchRunState
    ) -> SemanticStateSnapshot | None:
        """Cold-load and verify the snapshot and every referenced formal product."""

        complete = state.semantic_execution_status is SemanticExecutionStatus.COMPLETE
        artifact_ref = (
            state.semantic_state_ref
            if complete
            else dict(state.execution_refs).get("semantic_snapshot_candidate")
        )
        expected_hash = (
            state.semantic_state_hash
            if complete
            else dict(state.execution_refs).get("semantic_snapshot_hash")
        )
        if not isinstance(artifact_ref, str) or not isinstance(expected_hash, str):
            if complete:
                raise ValueError("run_state_semantic_pointer_missing")
            return None
        raw = await self._artifacts.read_json(artifact_ref)
        if not isinstance(raw, dict):
            raise ValueError("semantic_state_artifact_invalid")
        snapshot = SemanticStateSnapshot.model_validate(raw)
        if (
            snapshot.run_ref != state.run_ref
            or snapshot.artifact_ref != artifact_ref
            or snapshot.snapshot_hash != expected_hash
            or snapshot.evidence_universe_hash != state.evidence_universe_hash
            or snapshot.semantic_evidence_hash != state.semantic_evidence_hash
            or snapshot.execution_status is not state.semantic_execution_status
            or snapshot.semantic_engine_version != state.semantic_engine_version
            or snapshot.schema_version != state.semantic_schema_version
        ):
            raise ValueError("run_state_semantic_snapshot_identity_mismatch")
        await self._validate_semantic_products(snapshot)
        return snapshot

    async def load_control_projection(
        self, state: ResearchRunState
    ) -> tuple[SemanticGapControlArtifact | None, SemanticIntentControlArtifact | None]:
        """Cold-load only the Gap/Intent artifacts referenced by this RunState."""

        await self._validate_control_projection(state)
        refs = dict(state.execution_refs)
        gap_ref = refs.get("gap_projection_artifact_ref")
        intent_ref = refs.get("intent_projection_artifact_ref")
        if gap_ref is None and intent_ref is None:
            return None, None
        if not isinstance(gap_ref, str) or not isinstance(intent_ref, str):
            raise ValueError("control_projection_artifact_refs_incomplete")
        gap_raw = await self._artifacts.read_json(gap_ref)
        intent_raw = await self._artifacts.read_json(intent_ref)
        if not isinstance(gap_raw, dict) or not isinstance(intent_raw, dict):
            raise ValueError("control_projection_artifact_invalid")
        return (
            SemanticGapControlArtifact.model_validate(gap_raw),
            SemanticIntentControlArtifact.model_validate(intent_raw),
        )

    async def _validate_semantic_products(self, snapshot: SemanticStateSnapshot) -> None:
        bundles: dict[str, dict[str, object]] = {}
        for binding in snapshot.output_product_bindings:
            if "#" not in binding.product_ref:
                raise ValueError("semantic_product_reference_invalid")
            bundle_ref, product_kind = binding.product_ref.rsplit("#", maxsplit=1)
            if product_kind != binding.product_kind or product_kind not in _PRODUCT_FIELDS:
                raise ValueError("semantic_product_reference_kind_mismatch")
            bundle = bundles.get(bundle_ref)
            if bundle is None:
                raw = await self._artifacts.read_json(bundle_ref)
                if not isinstance(raw, dict):
                    raise ValueError("semantic_product_bundle_invalid")
                if (
                    raw.get("run_ref") != str(snapshot.run_ref)
                    or bundle_ref not in snapshot.provenance_refs
                    or raw.get("schema_version") != "semantic-shadow-verification-snapshot-v2"
                ):
                    raise ValueError("semantic_product_bundle_identity_mismatch")
                bundle = raw
                bundles[bundle_ref] = bundle
            product = bundle.get(_PRODUCT_FIELDS[product_kind])
            if product is None or canonical_hash(product) != binding.content_hash:
                raise ValueError("semantic_product_content_hash_mismatch")
            versions = bundle.get("schema_versions")
            if not isinstance(versions, dict) or versions.get(
                _PRODUCT_VERSION_FIELDS[product_kind]
            ) != binding.product_version:
                raise ValueError("semantic_product_version_mismatch")

    async def _validate_control_projection(self, state: ResearchRunState) -> None:
        refs = dict(state.execution_refs)
        gap_ref = refs.get("gap_projection_artifact_ref")
        intent_ref = refs.get("intent_projection_artifact_ref")
        if gap_ref is None and intent_ref is None:
            if state.gap_projection_status is GapProjectionStatus.DERIVED:
                raise ValueError("derived_gap_projection_artifact_missing")
            return
        if not isinstance(gap_ref, str) or not isinstance(intent_ref, str):
            raise ValueError("control_projection_artifact_refs_incomplete")
        gap_raw = await self._artifacts.read_json(gap_ref)
        intent_raw = await self._artifacts.read_json(intent_ref)
        if not isinstance(gap_raw, dict) or not isinstance(intent_raw, dict):
            raise ValueError("control_projection_artifact_invalid")
        gap = SemanticGapControlArtifact.model_validate(gap_raw)
        intents = SemanticIntentControlArtifact.model_validate(intent_raw)
        if (
            gap.run_ref != state.run_ref
            or intents.run_ref != state.run_ref
            or gap.artifact_ref != gap_ref
            or intents.artifact_ref != intent_ref
            or refs.get("gap_projection_artifact_hash") != gap.artifact_hash
            or refs.get("intent_projection_artifact_hash") != intents.artifact_hash
            or gap.semantic_snapshot_ref != state.semantic_state_ref
            or gap.semantic_snapshot_hash != state.semantic_state_hash
            or intents.semantic_snapshot_ref != gap.semantic_snapshot_ref
            or intents.semantic_snapshot_hash != gap.semantic_snapshot_hash
            or intents.gap_artifact_ref != gap_ref
            or intents.gap_projection_hash != gap.projection_hash
        ):
            raise ValueError("control_projection_binding_mismatch")
        if state.gap_projection_status is GapProjectionStatus.DERIVED:
            if gap.projection_status != "DERIVED":
                raise ValueError("run_state_gap_projection_status_mismatch")
            if (
                state.gap_refs != tuple(sorted(item.gap_ref for item in gap.gaps))
                or state.gap_projection_hash != gap.projection_hash
                or state.gap_source_semantic_hash != gap.semantic_snapshot_hash
                or state.intent_refs != tuple(sorted(item.intent_ref for item in intents.intents))
                or state.intent_source_gap_hash != gap.projection_hash
            ):
                raise ValueError("run_state_control_projection_content_mismatch")
            if intents.issues and (
                state.intent_refs
                or "INTENT_PROJECTION_INCOMPLETE" not in state.known_limitations
            ):
                raise ValueError("run_state_hides_intent_projection_issue")
        elif gap.projection_status == "DERIVED":
            raise ValueError("run_state_hides_derived_gap_projection")

    async def publish(
        self,
        *,
        state: ResearchRunState,
        semantic_snapshot: SemanticStateSnapshot | None,
    ) -> None:
        if semantic_snapshot is None and state.semantic_state_ref is not None:
            raise ValueError("run_state_semantic_artifact_missing")
        if semantic_snapshot is not None:
            if semantic_snapshot.run_ref != state.run_ref:
                raise ValueError("semantic_snapshot_run_mismatch")
            if state.semantic_execution_status is not semantic_snapshot.execution_status:
                raise ValueError("run_state_semantic_status_mismatch")
            if (
                state.semantic_engine_version != semantic_snapshot.semantic_engine_version
                or state.semantic_schema_version != semantic_snapshot.schema_version
            ):
                raise ValueError("run_state_semantic_contract_mismatch")
            if state.semantic_state_hash is not None and (
                state.semantic_state_hash != semantic_snapshot.snapshot_hash
                or state.semantic_state_ref != semantic_snapshot.artifact_ref
            ):
                raise ValueError("run_state_semantic_snapshot_identity_mismatch")
            if (
                state.evidence_universe_hash != semantic_snapshot.evidence_universe_hash
                or state.semantic_evidence_hash != semantic_snapshot.semantic_evidence_hash
            ):
                raise ValueError("run_state_semantic_universe_mismatch")
            if semantic_snapshot.execution_status is SemanticExecutionStatus.COMPLETE:
                await self._validate_semantic_products(semantic_snapshot)
            await self._artifacts.save_immutable_json(
                semantic_snapshot.artifact_ref, semantic_snapshot.model_dump(mode="json")
            )

        await self._validate_control_projection(state)
        await self._artifacts.save_immutable_json(
            state.state_ref, state.model_dump(mode="json")
        )
        # A retry after the pointer commit is idempotent; a stale candidate is
        # rejected by repository CAS and remains an inert orphan artifact.
        await self._runs.advance_run_state_pointer(state)

    async def recover_orphan_candidate(self, artifact_ref: str) -> bool:
        """Replay the pointer CAS for a previously durable RunState artifact."""

        raw = await self._artifacts.read_json(artifact_ref)
        if not isinstance(raw, dict):
            raise ValueError("run_state_artifact_invalid")
        state = ResearchRunState.model_validate(raw)
        if state.state_ref != artifact_ref:
            raise ValueError("run_state_artifact_ref_mismatch")
        if state.semantic_state_ref is not None or dict(state.execution_refs).get(
            "semantic_snapshot_candidate"
        ) is not None:
            await self.load_semantic_snapshot(state)
        await self._validate_control_projection(state)
        await self._runs.advance_run_state_pointer(state)
        return True
