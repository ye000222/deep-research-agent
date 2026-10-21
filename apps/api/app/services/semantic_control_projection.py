"""Deterministic, cold-recoverable V3.2 gap and intent projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.domain.atomic_proposition import PropositionEvidenceLink
from app.domain.contradiction_detection import PropositionConflictAssessment
from app.domain.corroboration_aggregation import IndependentCorroborationAssessment
from app.domain.gap_research_intent import GapResearchIntentIssue
from app.domain.proposition_research_gap import (
    PropositionGapDerivationIssue,
    PropositionGapReasonCode,
    PropositionResearchGap,
)
from app.domain.proposition_support_group import PropositionSupportGroup
from app.domain.proposition_verification import PropositionVerificationAssessment
from app.domain.run_state import SemanticExecutionStatus, SemanticStateSnapshot, canonical_hash
from app.domain.semantic_control_projection import (
    SemanticGapControlArtifact,
    SemanticIntentControlArtifact,
    semantic_gap_artifact_ref,
    semantic_intent_artifact_ref,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudgetState
from app.infrastructure.artifacts import LocalArtifactStore
from app.services.gap_research_intent import GapResearchIntentService
from app.services.proposition_gap_derivation import (
    PropositionGapDerivationService,
    PropositionGapDerivationTarget,
)


@dataclass(frozen=True, slots=True)
class SemanticControlProjectionResult:
    gap_artifact: SemanticGapControlArtifact
    intent_artifact: SemanticIntentControlArtifact
    gap_derivation_issues: tuple[PropositionGapDerivationIssue, ...]
    intent_issues: tuple[GapResearchIntentIssue, ...]


class SemanticControlProjectionService:
    """Consume only products already bound by a complete semantic snapshot."""

    def __init__(self, artifacts: LocalArtifactStore) -> None:
        self._artifacts = artifacts

    async def recover(
        self, snapshot: SemanticStateSnapshot
    ) -> SemanticControlProjectionResult:
        """Rebuild from the snapshot's immutable product bundle after a cold start."""

        if len(snapshot.provenance_refs) != 1:
            raise ValueError("semantic_product_bundle_provenance_ambiguous")
        raw = await self._artifacts.read_json(snapshot.provenance_refs[0])
        if not isinstance(raw, Mapping):
            raise ValueError("semantic_product_bundle_unavailable")
        return await self.derive_and_persist(snapshot=snapshot, bundle=raw)

    async def derive_and_persist(
        self,
        *,
        snapshot: SemanticStateSnapshot,
        bundle: Mapping[str, Any],
    ) -> SemanticControlProjectionResult:
        if snapshot.execution_status is not SemanticExecutionStatus.COMPLETE:
            raise ValueError("gap_projection_requires_complete_semantic_snapshot")
        self._validate_bound_bundle(snapshot, bundle)
        groups = tuple(
            PropositionSupportGroup.model_validate(item)
            for item in bundle["proposition_groups"]
        )
        corroborations = tuple(
            IndependentCorroborationAssessment.model_validate(item)
            for item in bundle["corroboration_assessments"]
        )
        verifications = tuple(
            PropositionVerificationAssessment.model_validate(item)
            for item in bundle["verification_assessments"]
        )
        conflicts = PropositionConflictAssessment.model_validate(bundle["conflicts"])
        budget_raw = bundle.get("budget_state")
        budget = (
            SemanticRuntimeBudgetState.model_validate(budget_raw)
            if isinstance(budget_raw, Mapping)
            else None
        )
        corr_by_group = {item.group_ref: item for item in corroborations}
        verification_by_group = {item.group_ref: item for item in verifications}
        target_groups = {item.group_ref: item for item in groups}
        targets: list[PropositionGapDerivationTarget] = []
        for group_ref in sorted(set(corr_by_group) | set(verification_by_group)):
            group = target_groups.get(group_ref)
            verification = verification_by_group.get(group_ref)
            corroboration = corr_by_group.get(group_ref)
            if group is None or verification is None or corroboration is None:
                continue
            targets.append(
                PropositionGapDerivationTarget(
                    run_ref=snapshot.run_ref,
                    objective_scope_ref=f"proposition-group:{group.group_ref}",
                    target_group=group,
                    verification_assessment=verification,
                    corroboration_assessments=corroborations,
                    conflict_assessment=conflicts,
                    support_groups=groups,
                    proposition_evidence_links=tuple(
                        # Exact persisted product records are parsed below and
                        # supplied without re-evaluating their direction.
                        _parse_link(item)
                        for item in bundle["proposition_evidence_links"]
                        if str(item.get("proposition_id"))
                        in {str(ref) for ref in group.proposition_refs}
                    ),
                    budget_state=budget,
                    attempt_count=0,
                )
            )
        if len(targets) != len(groups):
            target_group_refs = {item.group_ref for item in groups}
            represented_refs = {item.target_group.group_ref for item in targets}
            missing_refs = sorted(target_group_refs - represented_refs)
            extra_refs = sorted(represented_refs - target_group_refs)
            derivation_issues = tuple(
                PropositionGapDerivationIssue(
                    run_ref=snapshot.run_ref,
                    target_group_ref=group_ref,
                    target_proposition_refs=(
                        target_groups[group_ref].proposition_refs
                        if group_ref in target_groups
                        else ()
                    ),
                    reason_code=PropositionGapReasonCode.INPUT_REFERENCE_MISMATCH,
                    unresolved_reason="FORMAL_GAP_INPUT_MISSING",
                )
                for group_ref in missing_refs
            ) + tuple(
                PropositionGapDerivationIssue(
                    run_ref=snapshot.run_ref,
                    target_group_ref=group_ref,
                    reason_code=PropositionGapReasonCode.INPUT_REFERENCE_MISMATCH,
                    unresolved_reason="FORMAL_GAP_INPUT_ORPHANED",
                )
                for group_ref in extra_refs
            )
            gaps: tuple[PropositionResearchGap, ...] = ()
            gap_status = "INCOMPLETE"
        else:
            result = PropositionGapDerivationService().derive(tuple(targets))
            derivation_issues = result.issues
            gaps = result.gaps
            gap_status = "INCOMPLETE" if derivation_issues else "DERIVED"
        gap_hash = canonical_hash([item.gap_ref for item in gaps])
        gap_fields: dict[str, Any] = dict(
            schema_version="semantic-gap-control-projection-v1",
            run_ref=snapshot.run_ref,
            semantic_snapshot_ref=snapshot.artifact_ref,
            semantic_snapshot_hash=snapshot.snapshot_hash,
            projection_status=gap_status,
            gaps=gaps,
            issues=derivation_issues,
            projection_hash=gap_hash,
        )
        gap_artifact_hash = canonical_hash(
            {
                **gap_fields,
                "run_ref": str(snapshot.run_ref),
                "gaps": [item.model_dump(mode="json") for item in gaps],
                "issues": [item.model_dump(mode="json") for item in derivation_issues],
            }
        )
        gap_ref = semantic_gap_artifact_ref(snapshot.run_ref, gap_artifact_hash)
        gap_artifact = SemanticGapControlArtifact(
            **gap_fields,
            artifact_hash=gap_artifact_hash,
            artifact_ref=gap_ref,
        )
        intent_result = GapResearchIntentService().derive(gaps) if gap_status == "DERIVED" else None
        intent_issues = intent_result.issues if intent_result is not None else ()
        intents = (
            intent_result.intents
            if intent_result is not None and not intent_issues
            else ()
        )
        intent_hash = canonical_hash([item.intent_ref for item in intents])
        intent_fields: dict[str, Any] = dict(
            schema_version="semantic-intent-control-projection-v1",
            run_ref=snapshot.run_ref,
            semantic_snapshot_ref=snapshot.artifact_ref,
            semantic_snapshot_hash=snapshot.snapshot_hash,
            gap_artifact_ref=gap_ref,
            gap_projection_hash=gap_hash,
            intents=intents,
            issues=tuple(item.model_dump(mode="json") for item in intent_issues),
            projection_hash=intent_hash,
        )
        intent_artifact_hash = canonical_hash(
            {
                **intent_fields,
                "run_ref": str(snapshot.run_ref),
                "intents": [item.model_dump(mode="json") for item in intents],
            }
        )
        intent_ref = semantic_intent_artifact_ref(snapshot.run_ref, intent_artifact_hash)
        intent_artifact = SemanticIntentControlArtifact(
            **intent_fields,
            artifact_hash=intent_artifact_hash,
            artifact_ref=intent_ref,
        )
        await self._artifacts.save_immutable_json(gap_ref, gap_artifact.model_dump(mode="json"))
        await self._artifacts.save_immutable_json(
            intent_ref, intent_artifact.model_dump(mode="json")
        )
        return SemanticControlProjectionResult(
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
            gap_derivation_issues=derivation_issues,
            intent_issues=intent_issues,
        )

    @staticmethod
    def _validate_bound_bundle(
        snapshot: SemanticStateSnapshot, bundle: Mapping[str, Any]
    ) -> None:
        if bundle.get("run_ref") != str(snapshot.run_ref):
            raise ValueError("semantic_bundle_run_mismatch")
        bindings = {item.product_kind: item for item in snapshot.output_product_bindings}
        fields = {
            "proposition_groups": "proposition_groups",
            "corroboration_assessments": "corroboration_assessments",
            "conflicts": "conflicts",
            "verification_assessments": "verification_assessments",
            "proposition_evidence_links": "proposition_evidence_links",
        }
        for product_kind, field_name in fields.items():
            binding = bindings.get(product_kind)
            value = bundle.get(field_name)
            if binding is None or value is None or canonical_hash(value) != binding.content_hash:
                raise ValueError("semantic_product_binding_mismatch")


def _parse_link(value: Mapping[str, Any]) -> PropositionEvidenceLink:
    return PropositionEvidenceLink.model_validate(value)
