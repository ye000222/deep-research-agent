"""OFF/SHADOW-only semantic verification orchestration over committed receipts."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.assertion_composition import AssertionComposition
from app.domain.atomic_proposition import (
    AtomicProposition,
    ClaimPropositionLink,
    PropositionEvidenceLink,
)
from app.domain.atomic_proposition_contracts import (
    ATOMIC_DECOMPOSITION_PROMPT_V3,
    ATOMIC_PROPOSITION_SCHEMA_V3,
    AtomicPropositionExecutionMode,
)
from app.domain.claim_architecture import (
    CanonicalClaim,
    CanonicalClaimType,
    ClaimEvidenceLink,
    ClaimEvidenceRelation,
)
from app.domain.comparability_evaluator import ComparabilityEvaluator
from app.domain.conflict_scope_completeness import (
    ConflictScopeCompletenessEvaluator,
    PlannedPropositionPair,
)
from app.domain.contradiction_detection import detect_proposition_conflicts
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibilityInput,
    aggregate_independent_corroboration,
)
from app.domain.evidence_graph import claim_fingerprint
from app.domain.proposition_evidence_direction import (
    EvidenceDirectionScope,
    MeasurementAdmissibilityBinding,
    PropositionDirectionTarget,
    SourceAssertionEvidenceBinding,
    evaluate_proposition_evidence_directions,
)
from app.domain.proposition_pair_identity import (
    PAIR_IDENTITY_RUNTIME,
    PropositionPairInput,
    PropositionPairTarget,
    content_hash,
    project_pair_views,
)
from app.domain.proposition_support_group import (
    EvidenceSourceIdentity,
    PropositionAssertionBinding,
    assess_owner_independence,
    group_propositions,
)
from app.domain.proposition_verification import (
    aggregate_proposition_verification,
)
from app.domain.relation_evaluator import RelationEvaluator
from app.domain.run_state import (
    GapProjectionStatus,
    SemanticExecutionStatus,
)
from app.domain.semantic_identity_alignment import SemanticAlignmentStatus
from app.domain.semantic_runtime_budget import (
    SemanticRuntimeBudget,
    SemanticRuntimeBudgetState,
)
from app.domain.source_context_extraction_v3_4 import SOURCE_SEMANTIC_RUNTIME_PROMPT_V2
from app.domain.truth_condition_context import (
    ContextGraph,
    ContextOrigin,
    ContextPresenceConclusion,
)
from app.domain.verification_assertion import (
    AssertionGroundingUsability,
    ComparisonEligibility,
    PropositionPairAssessment,
    VerificationAssertion,
    project_verification_assertion,
)
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_tools import (
    ClaimPersistenceReceipt,
    PagePersistenceResult,
    ResearchToolRepository,
)
from app.infrastructure.db.run_providers import RunProviderBindingRepository
from app.llm.adapters import LLMGateway
from app.security.secrets import SecretCipher
from app.services.atomic_proposition_decomposer import (
    AtomicPropositionDecomposer,
    ShadowDecompositionResult,
    ShadowDecompositionStatus,
)
from app.services.proposition_pair_identity_runtime import (
    PairAlignmentRuntimeResult,
    PropositionPairSemanticAlignmentRuntime,
    plan_proposition_pairs,
    request_payload,
    validate_pair_product,
)
from app.services.run_state_publisher import RunStatePublisher
from app.services.run_state_reducer import RunStateReducer
from app.services.semantic_control_projection import SemanticControlProjectionService
from app.services.semantic_engine import SemanticEngine
from app.services.semantic_identity_alignment_runtime import (
    AlignmentRuntimeStatus,
    IdentityAlignmentPair,
    IdentityAlignmentRuntimeResult,
    SemanticIdentityAlignmentRuntime,
    alignment_cache_ref,
    deterministic_pair_ref,
    parse_alignment_response,
)
from app.services.source_semantic_runtime_v3_3 import (
    SourceSemanticRuntimeInput,
    SourceSemanticRuntimeStatus,
)
from app.services.source_semantic_runtime_v3_4 import (
    SourceSemanticExtractionV34Runtime,
    SourceSemanticRuntimeResultV34,
    rehydrate_typed_value_grounding,
)
from app.services.target_proposition_formation import (
    TARGET_FORMATION_PROMPT_VERSION,
    TARGET_FORMATION_SCHEMA_VERSION,
    TARGET_FORMATION_VALIDATOR_VERSION,
    FormationExecutionStatus,
    FormationReasonCode,
    FormationStatus,
    TargetFormationRuntimeResult,
    TargetPropositionFormationAssessment,
    TargetPropositionFormationValidator,
    unknown_formation_result,
)


class SemanticShadowMode(StrEnum):
    OFF = "OFF"
    SHADOW = "SHADOW"


class SemanticShadowStage(StrEnum):
    AP_FORMATION = "AP_FORMATION"
    TARGET_FORMATION_VALIDATION = "TARGET_FORMATION_VALIDATION"
    SOURCE_ASSERTION_FORMATION = "SOURCE_ASSERTION_FORMATION"
    GROUNDING = "GROUNDING"
    DIRECTION = "DIRECTION"
    GROUPING = "GROUPING"
    PROPOSITION_PAIR_IDENTITY = "PROPOSITION_PAIR_IDENTITY"
    CORROBORATION = "CORROBORATION"
    CONTRADICTION = "CONTRADICTION"
    VERIFICATION = "VERIFICATION"
    PERSISTENCE = "PERSISTENCE"
    BUDGET = "BUDGET"


class SemanticShadowDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: UUID
    claim_ref: UUID | None = None
    evidence_ref: UUID | None = None
    stage: SemanticShadowStage
    status: str = Field(min_length=1, max_length=80)
    reason_code: str = Field(min_length=1, max_length=120)
    schema_version: str = "semantic-shadow-diagnostic-v1"
    atomic_proposition_schema: str = ATOMIC_PROPOSITION_SCHEMA_V3
    atomic_proposition_prompt: str = ATOMIC_DECOMPOSITION_PROMPT_V3
    source_runtime_version: str = SOURCE_SEMANTIC_RUNTIME_PROMPT_V2


class SemanticShadowVerificationSnapshot(BaseModel):
    """V2.4-read-only derived artifact; intentionally excludes source/reader text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "semantic-shadow-verification-snapshot-v2"
    run_ref: UUID
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_refs: tuple[UUID, ...]
    evidence_refs: tuple[UUID, ...]
    proposition_refs: tuple[UUID, ...]
    formation_assessments: tuple[dict[str, Any], ...] = ()
    proposition_group_refs: tuple[str, ...]
    proposition_groups: tuple[dict[str, Any], ...] = ()
    pair_assessments: tuple[dict[str, Any], ...] = ()
    pair_candidate_plan: dict[str, Any] = Field(default_factory=dict)
    pair_execution_states: dict[str, str] = Field(default_factory=dict)
    proposition_evidence_links: tuple[dict[str, Any], ...]
    corroboration_assessments: tuple[dict[str, Any], ...]
    conflicts: dict[str, Any]
    verification_assessments: tuple[dict[str, Any], ...]
    conflict_scope_completeness: dict[str, Any] | None = None
    budget_state: SemanticRuntimeBudgetState
    diagnostics: tuple[SemanticShadowDiagnostic, ...]
    schema_versions: dict[str, str]
    artifact_ref: str


class ShadowTargetState(BaseModel):
    """Derived claim state; the index stores only references to these products."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "semantic-shadow-target-state-v1"
    run_ref: UUID
    claim_ref: UUID
    ap_product_ref: str
    formation_product_ref: str
    formation_assessments: tuple[TargetPropositionFormationAssessment, ...]
    composition: AssertionComposition | None = None
    assertions: tuple[VerificationAssertion, ...]
    claim_links: tuple[ClaimPropositionLink, ...]
    evidence_links: tuple[PropositionEvidenceLink, ...]
    acceptance: dict[UUID, bool]
    sources: dict[UUID, tuple[UUID, str | None, str | None]]
    processed_evidence: tuple[UUID, ...]


class SemanticResearchShadowPipeline:
    """Run-local bounded side path; legacy research decisions are never inputs/outputs."""

    def __init__(
        self,
        *,
        repository: ResearchToolRepository,
        bindings: RunProviderBindingRepository,
        cipher: SecretCipher,
        gateway: LLMGateway,
        artifacts: LocalArtifactStore,
        budget: SemanticRuntimeBudget | None,
        run_state_runs: Any | None = None,
        run_state_reducer: RunStateReducer | None = None,
        run_state_publisher: RunStatePublisher | None = None,
        semantic_engine: SemanticEngine | None = None,
    ) -> None:
        self._repository = repository
        self._bindings = bindings
        self._cipher = cipher
        self._gateway = gateway
        self._artifacts = artifacts
        self._budget = budget
        self._run_state_runs = run_state_runs
        self._run_state_reducer = run_state_reducer
        self._run_state_publisher = run_state_publisher
        self._semantic_engine = semantic_engine
        self._semantic_control_projection = SemanticControlProjectionService(artifacts)
        self._state = SemanticRuntimeBudgetState()
        self._ap_cache: dict[tuple[UUID, str, str], ShadowDecompositionResult] = {}
        self._formation_cache: dict[str, TargetFormationRuntimeResult] = {}
        self._source_cache: dict[tuple[UUID, str, str], SourceSemanticRuntimeResultV34] = {}
        self._processed_receipts: set[tuple[UUID, UUID]] = set()
        self._claims: dict[
            UUID,
            tuple[CanonicalClaim, str, tuple[AtomicProposition, ...], AssertionComposition | None],
        ] = {}
        self._claim_links: dict[tuple[UUID, UUID], ClaimPropositionLink] = {}
        self._target_assertions: dict[UUID, Any] = {}
        self._formation_assessments: dict[UUID, TargetPropositionFormationAssessment] = {}
        self._evidence_links: dict[tuple[UUID, UUID, str], Any] = {}
        self._acceptance_by_evidence: dict[UUID, bool] = {}
        self._source_by_evidence: dict[UUID, tuple[UUID, str | None, str | None]] = {}
        self._diagnostics: list[SemanticShadowDiagnostic] = []
        self._binding: Any | None = None
        self._api_key: Any | None = None
        self._receipt_snapshot_uri: str | None = None
        self._restored_products: set[str] = set()
        self._known_propositions: dict[UUID, AtomicProposition] = {}
        self._claim_state_refs: dict[str, str] = {}
        self._pair_batch_refs: set[str] = set()
        self._pair_results: dict[str, PairAlignmentRuntimeResult] = {}
        self._pair_attempted_refs: set[str] = set()
        self._pair_execution_states: dict[str, str] = {}
        self._index_run: UUID | None = None
        self._pair_plan_summary: dict[str, Any] = {}
        self._pair_plan_pairs: tuple[PropositionPairInput, ...] = ()
        self._last_pair_assessments: tuple[PropositionPairAssessment, ...] = ()
        self._pair_assessment_provenance: list[dict[str, Any]] = []
        self._target_compositions: dict[UUID, AssertionComposition | None] = {}
        self._run_orchestrator_versions: dict[UUID, str] = {}

    @property
    def budget_state(self) -> SemanticRuntimeBudgetState:
        return self._state

    async def process_persisted_page(self, run_id: UUID, persisted: PagePersistenceResult) -> None:
        """Consume only committed, accepted receipts; never raise into ResearchLoop."""

        if self._run_state_runs is not None and run_id not in self._run_orchestrator_versions:
            try:
                run_inputs = await self._run_state_runs.get_run_state_inputs(run_id)
                version = run_inputs.get("orchestrator_version")
                if version not in {"V2_LEGACY", "V3"}:
                    raise ValueError("unsupported_run_orchestrator_version")
                self._run_orchestrator_versions[run_id] = str(version)
            except Exception as exc:
                self._diagnostics.append(
                    SemanticShadowDiagnostic(
                        run_ref=run_id,
                        stage=SemanticShadowStage.PERSISTENCE,
                        status="FAILED_ISOLATED",
                        reason_code=_safe_reason(exc),
                    )
                )
                return

        if self._budget is None:
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    stage=SemanticShadowStage.BUDGET,
                    status="NOT_PROCESSED",
                    reason_code="SHADOW_NOT_CONFIGURED",
                )
            )
            await self._persist_snapshot(run_id)
            return
        if persisted.receipt_status != "AVAILABLE":
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    stage=SemanticShadowStage.PERSISTENCE,
                    status="NOT_PROCESSED",
                    reason_code=persisted.receipt_reason_code or "PERSISTENCE_RECEIPT_UNAVAILABLE",
                )
            )
            await self._persist_snapshot(run_id)
            return
        try:
            await self._restore_derived_index(run_id)
        except (ValueError, KeyError, TypeError):
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    stage=SemanticShadowStage.PERSISTENCE,
                    status="INVALID_INPUT",
                    reason_code="DERIVED_INDEX_INVALID",
                )
            )
            return
        page_marker = _page_receipt_marker(run_id, persisted)
        receipt_hash = page_marker.rsplit("/", maxsplit=1)[-1].removesuffix(".json")
        self._receipt_snapshot_uri = (
            f"runs/{run_id}/semantic-shadow/receipts/{receipt_hash}/snapshot.json"
        )
        # A receipt completion pointer is not a target-state catalog. Rehydrate
        # durable targets before replaying any new cross-receipt grouping work.
        if not await self._artifacts.create_json_once(
            page_marker,
            {
                "run_ref": str(run_id),
                "receipt_ref": page_marker.rsplit("/", maxsplit=1)[-1].removesuffix(".json"),
                "state": "RESERVED",
            },
        ):
            # RESERVED only records an attempt. A durable work product, not the
            # reservation, decides whether provider work is replay-suppressed.
            pass
        for receipt in sorted(persisted.claim_receipts, key=lambda item: str(item.claim_id)):
            try:
                await self._process_claim_receipt(run_id, receipt)
                await self._persist_claim_target_state(run_id, receipt)
            except Exception as exc:
                # Stable, typed diagnostics only; never serialize exception text or payloads.
                self._diagnostics.append(
                    SemanticShadowDiagnostic(
                        run_ref=run_id,
                        claim_ref=receipt.claim_id,
                        stage=SemanticShadowStage.PERSISTENCE,
                        status="FAILED_ISOLATED",
                        reason_code=_safe_reason(exc),
                    )
                )
        try:
            await self._save_derived_index(run_id)
            await self._rebuild_deterministic_assessments(run_id)
            artifact_ref, snapshot_payload = await self._persist_snapshot_document(run_id)
            await self._save_derived_index(run_id)
            await self._publish_v3_run_state(run_id, artifact_ref, snapshot_payload)
            if self._receipt_snapshot_uri is not None:
                await self._artifacts.save_json(
                    self._receipt_snapshot_uri,
                    {"artifact_ref": artifact_ref, "state": "COMPLETED"},
                )
        except Exception as exc:
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    stage=SemanticShadowStage.PERSISTENCE,
                    status="FAILED_ISOLATED",
                    reason_code=_safe_reason(exc),
                )
            )

    async def _provider_binding(self, run_id: UUID) -> tuple[Any, Any]:
        if self._binding is None or self._binding.run_id != run_id:
            binding = await self._bindings.get(run_id)
            api_key = self._cipher.decrypt(
                binding.encrypted_secret,
                credential_id=binding.credential_id,
                adapter_type=binding.adapter_type.value,
                credential_version=binding.credential_version,
            )
            self._binding, self._api_key = binding, api_key
        return self._binding, self._api_key

    async def _load_product(self, artifact_uri: str, model: Any) -> Any | None:
        try:
            payload = await self._artifacts.read_json(artifact_uri)
        except (FileNotFoundError, KeyError, AttributeError):
            return None
        if not isinstance(payload, dict):
            raise ValueError("SEMANTIC_WORK_PRODUCT_INVALID")
        return model.model_validate(_restore_persisted_decimal_values(payload))

    async def _completed_marker_without_product(self, marker_uri: str) -> bool:
        try:
            payload = await self._artifacts.read_json(marker_uri)
        except (FileNotFoundError, KeyError, AttributeError):
            return False
        return isinstance(payload, dict) and payload.get("state") == "COMPLETED"

    def _restore_provider_counter(self, product_ref: str, counter: str, amount: int) -> None:
        if product_ref in self._restored_products:
            return
        self._restored_products.add(product_ref)
        if amount:
            self._state = self._state.model_copy(
                update={counter: getattr(self._state, counter) + amount}
            )

    async def _process_claim_receipt(self, run_id: UUID, receipt: ClaimPersistenceReceipt) -> None:
        if not receipt.accepted_and_persisted:
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    claim_ref=receipt.claim_id,
                    stage=SemanticShadowStage.PERSISTENCE,
                    status="NOT_PROCESSED",
                    reason_code="RECEIPT_NOT_ACCEPTED_AND_PERSISTED",
                )
            )
            return
        if claim_fingerprint(receipt.claim_text) != receipt.claim_content_hash:
            raise ValueError("claim_receipt_invalid")
        if set(receipt.evidence_ids) != {
            item.evidence_id for item in receipt.evidence_receipts
        } or any(
            item.claim_id != receipt.claim_id
            or item.source_id not in receipt.source_ids
            or item.snapshot_id not in receipt.snapshot_ids
            for item in receipt.evidence_receipts
        ):
            raise ValueError("claim_receipt_evidence_binding_mismatch")
        budget = self._budget
        if budget is None:
            return
        claim_key = (receipt.claim_id, ATOMIC_PROPOSITION_SCHEMA_V3, ATOMIC_DECOMPOSITION_PROMPT_V3)
        if claim_key not in self._ap_cache:
            product_uri = _ap_product_uri(run_id, receipt)
            restored = await self._load_product(product_uri, ShadowDecompositionResult)
            if restored is not None:
                self._ap_cache[claim_key] = restored
                self._restore_provider_counter(
                    product_uri, "provider_ap_decomposition_calls", restored.model_call_count
                )
            else:
                if await self._completed_marker_without_product(
                    _attempt_marker(run_id, "claim", receipt.claim_id, claim_key[1:])
                ):
                    self._diagnostics.append(
                        SemanticShadowDiagnostic(
                            run_ref=run_id,
                            claim_ref=receipt.claim_id,
                            stage=SemanticShadowStage.AP_FORMATION,
                            status="INVALID_INPUT",
                            reason_code="COMPLETED_AP_PRODUCT_MISSING",
                        )
                    )
                    return
            if (
                claim_key not in self._ap_cache
                and self._state.provider_ap_decomposition_calls >= budget.max_ap_decomposition_calls
            ):
                self._diagnostics.append(
                    SemanticShadowDiagnostic(
                        run_ref=run_id,
                        claim_ref=receipt.claim_id,
                        stage=SemanticShadowStage.BUDGET,
                        status="NOT_PROCESSED_BUDGET",
                        reason_code="AP_DECOMPOSITION_PROVIDER_BUDGET_EXHAUSTED",
                    )
                )
            elif claim_key not in self._ap_cache:
                binding, api_key = await self._provider_binding(run_id)
                marker = _attempt_marker(run_id, "claim", receipt.claim_id, claim_key[1:])
                await self._artifacts.create_json_once(
                    marker,
                    {
                        "run_ref": str(run_id),
                        "claim_ref": str(receipt.claim_id),
                        "state": "RESERVED",
                    },
                )
                self._state, allowed = self._state.consume(
                    budget, "provider_ap_decomposition_calls"
                )
                if allowed:
                    decomposer = AtomicPropositionDecomposer(
                        _RunScopedGateway(self._gateway, run_id),
                        adapter_type=binding.adapter_type,
                        base_url=binding.base_url,
                        api_key=api_key,
                        model=binding.model,
                        provider_profile_id=None,
                        max_calls=1,
                    )
                    result = await decomposer.decompose_claim(
                        receipt.claim_text,
                        {"claim_id": str(receipt.claim_id), "run_id": str(run_id)},
                        claim_id=str(receipt.claim_id),
                        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
                        prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V3,
                        execution_mode=AtomicPropositionExecutionMode.SHADOW,
                    )
                    self._ap_cache[claim_key] = result
                    await self._artifacts.save_json(product_uri, _ap_product_payload(result))
                    self._restored_products.add(product_uri)
                    await self._artifacts.save_json(
                        marker,
                        {
                            "run_ref": str(run_id),
                            "claim_ref": str(receipt.claim_id),
                            "state": "COMPLETED",
                            "result_artifact_ref": product_uri,
                        },
                    )
                    self._diagnostics.append(
                        SemanticShadowDiagnostic(
                            run_ref=run_id,
                            claim_ref=receipt.claim_id,
                            stage=SemanticShadowStage.AP_FORMATION,
                            status=result.status.value,
                            reason_code=(
                                "AP_DECOMPOSITION_RESOLVED"
                                if result.status is ShadowDecompositionStatus.RESOLVED
                                else "AP_DECOMPOSITION_UNRESOLVED_OR_FAILED"
                            ),
                        )
                    )
        decomposition = self._ap_cache.get(claim_key)
        if decomposition is None or decomposition.status is not ShadowDecompositionStatus.RESOLVED:
            return
        canonical_claim = CanonicalClaim.from_text(
            claim_id=receipt.claim_id,
            original_text=receipt.claim_text,
            run_id=run_id,
            question_id=_receipt_question(receipt),
            dimension_key=_receipt_dimension(receipt),
            claim_type=CanonicalClaimType.UNKNOWN,
        )
        propositions = tuple(
            sorted(decomposition.propositions, key=lambda item: str(item.proposition_id))
        )
        links = tuple(
            ClaimPropositionLink(
                claim_id=receipt.claim_id,
                proposition_id=proposition.proposition_id,
                proposition_index=(
                    proposition.decomposition.proposition_index
                    if proposition.decomposition.proposition_index is not None
                    else index
                ),
                claim_span=proposition.decomposition.source_span,
                decomposition_status=proposition.decomposition.status,
            )
            for index, proposition in enumerate(propositions)
        )
        proposition_ids = {item.proposition_id for item in propositions}
        output_graph = ContextGraph(
            origin=ContextOrigin.OUTPUT,
            claim_ref=str(receipt.claim_id),
            contexts=(),
            context_presence=ContextPresenceConclusion.DETECTOR_CANNOT_DECIDE,
            source_detection_complete=False,
        )
        provisional_assertions = tuple(
            _project_target_assertion(
                proposition,
                decomposition=decomposition,
                claim_id=receipt.claim_id,
                claim_text=receipt.claim_text,
                proposition_ids=proposition_ids,
                output_graph=output_graph,
            )
            for proposition in propositions
        )
        eligibility = {
            proposition.proposition_id: assertion.composition_eligibility.value
            for proposition, assertion in zip(propositions, provisional_assertions, strict=True)
        }
        formation = await self._formation_for_claim(
            run_id,
            receipt,
            decomposition,
            propositions,
            links,
            eligibility,
        )
        assessment_by_prop = {item.proposition_ref: item for item in formation.assessments}
        self._formation_assessments.update(assessment_by_prop)
        self._claims[receipt.claim_id] = (
            canonical_claim,
            receipt.claim_text,
            propositions,
            decomposition.assertion_composition,
        )
        self._known_propositions.update({item.proposition_id: item for item in propositions})
        self._target_compositions.update(
            {item.proposition_id: decomposition.assertion_composition for item in propositions}
        )
        for proposition, membership, provisional in zip(
            propositions, links, provisional_assertions, strict=True
        ):
            self._claim_links[(receipt.claim_id, proposition.proposition_id)] = membership
            assessment = assessment_by_prop.get(proposition.proposition_id)
            grounding = _target_grounding_from_formation(assessment)
            self._target_assertions[proposition.proposition_id] = provisional.model_copy(
                update={"grounding_usability": grounding}
            )
        for evidence in sorted(receipt.evidence_receipts, key=lambda item: str(item.evidence_id)):
            receipt_key = (receipt.claim_id, evidence.evidence_id)
            if receipt_key in self._processed_receipts:
                continue
            self._processed_receipts.add(receipt_key)
            try:
                await self._process_evidence_receipt(run_id, receipt, evidence, decomposition)
            except Exception as exc:
                # One malformed/unavailable Evidence must not discard sibling results.
                self._diagnostic_evidence(
                    run_id,
                    receipt.claim_id,
                    evidence.evidence_id,
                    SemanticShadowStage.SOURCE_ASSERTION_FORMATION,
                    "FAILED_ISOLATED",
                    _safe_reason(exc),
                )

    async def _formation_for_claim(
        self,
        run_id: UUID,
        receipt: ClaimPersistenceReceipt,
        decomposition: ShadowDecompositionResult,
        propositions: tuple[AtomicProposition, ...],
        links: tuple[ClaimPropositionLink, ...],
        composition_eligibility: dict[UUID, str],
    ) -> TargetFormationRuntimeResult:
        budget = self._budget
        if budget is None:
            return unknown_formation_result(
                receipt.claim_id,
                propositions,
                links,
                FormationExecutionStatus.NOT_PROCESSED_BUDGET,
                FormationReasonCode.BUDGET_EXHAUSTED,
            )
        product_uri = _formation_product_uri(run_id, receipt, decomposition)
        cached = self._formation_cache.get(product_uri)
        if cached is not None:
            return cached
        restored = await self._load_product(product_uri, TargetFormationRuntimeResult)
        if restored is not None:
            self._formation_cache[product_uri] = restored
            self._restore_provider_counter(
                product_uri,
                "provider_formation_validation_calls",
                restored.model_call_count,
            )
            return cast(TargetFormationRuntimeResult, restored)
        marker = _attempt_marker(
            run_id,
            "formation",
            receipt.claim_id,
            [TARGET_FORMATION_PRODUCT_VERSION, product_uri],
        )
        if await self._completed_marker_without_product(marker):
            result = unknown_formation_result(
                receipt.claim_id,
                propositions,
                links,
                FormationExecutionStatus.INVALID,
                FormationReasonCode.INVALID_REFERENCE,
            )
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    claim_ref=receipt.claim_id,
                    stage=SemanticShadowStage.TARGET_FORMATION_VALIDATION,
                    status="INVALID_INPUT",
                    reason_code="COMPLETED_FORMATION_PRODUCT_MISSING",
                )
            )
            return result
        if self._state.provider_formation_validation_calls >= budget.max_formation_validation_calls:
            result = unknown_formation_result(
                receipt.claim_id,
                propositions,
                links,
                FormationExecutionStatus.NOT_PROCESSED_BUDGET,
                FormationReasonCode.BUDGET_EXHAUSTED,
            )
            self._formation_cache[product_uri] = result
            self._diagnostics.append(
                SemanticShadowDiagnostic(
                    run_ref=run_id,
                    claim_ref=receipt.claim_id,
                    stage=SemanticShadowStage.TARGET_FORMATION_VALIDATION,
                    status=result.execution_status.value,
                    reason_code=FormationReasonCode.BUDGET_EXHAUSTED.value,
                )
            )
            return result
        binding, api_key = await self._provider_binding(run_id)
        await self._artifacts.create_json_once(
            marker,
            {"run_ref": str(run_id), "claim_ref": str(receipt.claim_id), "state": "RESERVED"},
        )
        self._state, allowed = self._state.consume(budget, "provider_formation_validation_calls")
        if not allowed:
            result = unknown_formation_result(
                receipt.claim_id,
                propositions,
                links,
                FormationExecutionStatus.NOT_PROCESSED_BUDGET,
                FormationReasonCode.BUDGET_EXHAUSTED,
            )
            return result
        validator = TargetPropositionFormationValidator(
            _RunScopedGateway(self._gateway, run_id),
            adapter_type=binding.adapter_type,
            base_url=binding.base_url,
            api_key=api_key,
            model=binding.model,
        )
        result = await validator.validate_claim(
            claim_id=receipt.claim_id,
            claim_text=receipt.claim_text,
            propositions=propositions,
            links=links,
            composition=decomposition.assertion_composition,
            composition_eligibility=composition_eligibility,
        )
        self._formation_cache[product_uri] = result
        await self._artifacts.save_json(product_uri, result.model_dump(mode="json"))
        self._restored_products.add(product_uri)
        await self._artifacts.save_json(
            marker,
            {
                "run_ref": str(run_id),
                "claim_ref": str(receipt.claim_id),
                "state": "COMPLETED",
                "result_artifact_ref": product_uri,
            },
        )
        self._diagnostics.append(
            SemanticShadowDiagnostic(
                run_ref=run_id,
                claim_ref=receipt.claim_id,
                stage=SemanticShadowStage.TARGET_FORMATION_VALIDATION,
                status=result.execution_status.value,
                reason_code=(
                    result.reason_codes[0].value
                    if result.reason_codes
                    else result.execution_status.value
                ),
            )
        )
        return result

    async def _process_evidence_receipt(
        self,
        run_id: UUID,
        claim_receipt: ClaimPersistenceReceipt,
        evidence: Any,
        decomposition: ShadowDecompositionResult,
    ) -> None:
        if not evidence.accepted_and_persisted:
            self._diagnostic_evidence(
                run_id,
                claim_receipt.claim_id,
                evidence.evidence_id,
                SemanticShadowStage.PERSISTENCE,
                "NOT_PROCESSED",
                "EVIDENCE_NOT_ACCEPTED_AND_PERSISTED",
            )
            return
        span = evidence.canonical_span
        if span is None or not _canonical_receipt_binding_is_valid(evidence):
            self._diagnostic_evidence(
                run_id,
                claim_receipt.claim_id,
                evidence.evidence_id,
                SemanticShadowStage.GROUNDING,
                "INVALID_INPUT",
                "CANONICAL_SPAN_PROVENANCE_UNAVAILABLE_OR_MISMATCHED",
            )
            return
        if (claim_receipt.claim_id, evidence.evidence_id) not in self._processed_receipts:
            return
        if self._budget is None:
            return
        budget = self._budget
        propositions = self._claims[claim_receipt.claim_id][2]
        if (
            self._state.evidence_inputs >= budget.max_evidence_inputs
            or len(propositions) > budget.max_propositions_per_evidence
            or budget.max_source_assertions_per_evidence == 0
        ):
            self._diagnostic_evidence(
                run_id,
                claim_receipt.claim_id,
                evidence.evidence_id,
                SemanticShadowStage.BUDGET,
                "NOT_PROCESSED_BUDGET",
                "EVIDENCE_SCOPE_OR_ASSERTION_BUDGET_EXHAUSTED",
            )
            return
        cache_key = (evidence.evidence_id, span.span_ref, SOURCE_SEMANTIC_RUNTIME_PROMPT_V2)
        source_result = self._source_cache.get(cache_key)
        if source_result is None:
            product_uri = _source_product_uri(run_id, claim_receipt.claim_id, evidence, span)
            restored = await self._load_product(product_uri, SourceSemanticRuntimeResultV34)
            if restored is not None:
                try:
                    source_result = rehydrate_typed_value_grounding(restored, span)
                except (ValueError, KeyError, TypeError):
                    self._diagnostic_evidence(
                        run_id,
                        claim_receipt.claim_id,
                        evidence.evidence_id,
                        SemanticShadowStage.SOURCE_ASSERTION_FORMATION,
                        "INVALID_INPUT",
                        "TYPED_VALUE_RESUME_VALIDATION_FAILED",
                    )
                    return
                self._source_cache[cache_key] = source_result
                self._restore_provider_counter(
                    product_uri,
                    "provider_source_semantic_extraction_calls",
                    source_result.model_call_count,
                )
        if source_result is None:
            product_uri = _source_product_uri(run_id, claim_receipt.claim_id, evidence, span)
            marker = _attempt_marker(
                run_id,
                "evidence",
                evidence.evidence_id,
                (span.span_ref, SOURCE_SEMANTIC_RUNTIME_PROMPT_V2),
            )
            if await self._completed_marker_without_product(marker):
                self._diagnostic_evidence(
                    run_id,
                    claim_receipt.claim_id,
                    evidence.evidence_id,
                    SemanticShadowStage.SOURCE_ASSERTION_FORMATION,
                    "INVALID_INPUT",
                    "COMPLETED_SOURCE_PRODUCT_MISSING",
                )
                return
            if (
                self._state.provider_source_semantic_extraction_calls
                >= budget.max_source_semantic_extraction_calls
            ):
                self._diagnostic_evidence(
                    run_id,
                    claim_receipt.claim_id,
                    evidence.evidence_id,
                    SemanticShadowStage.BUDGET,
                    "NOT_PROCESSED_BUDGET",
                    "SOURCE_SEMANTIC_PROVIDER_BUDGET_EXHAUSTED",
                )
                return
            binding, api_key = await self._provider_binding(run_id)
            await self._artifacts.create_json_once(
                marker,
                {
                    "run_ref": str(run_id),
                    "claim_ref": str(claim_receipt.claim_id),
                    "evidence_ref": str(evidence.evidence_id),
                    "span_ref": span.span_ref,
                    "state": "RESERVED",
                },
            )
            self._state, allowed = self._state.consume(
                budget, "provider_source_semantic_extraction_calls"
            )
            if not allowed:
                self._diagnostic_evidence(
                    run_id,
                    claim_receipt.claim_id,
                    evidence.evidence_id,
                    SemanticShadowStage.BUDGET,
                    "NOT_PROCESSED_BUDGET",
                    "SOURCE_SEMANTIC_PROVIDER_BUDGET_EXHAUSTED",
                )
                return
            runtime = SourceSemanticExtractionV34Runtime(
                _RunScopedGateway(self._gateway, run_id),
                adapter_type=binding.adapter_type,
                base_url=binding.base_url,
                api_key=api_key,
                model=binding.model,
                max_calls=1,
            )
            source_result = await runtime.extract(
                SourceSemanticRuntimeInput(
                    source_ref=span.source_ref,
                    snapshot_ref=span.snapshot_ref,
                    canonical_evidence_span=span,
                    context_window=None,
                    max_assertions=min(64, budget.max_source_assertions_per_evidence),
                )
            )
            self._source_cache[cache_key] = source_result
            await self._artifacts.save_json(product_uri, _source_product_payload(source_result))
            self._restored_products.add(product_uri)
            await self._artifacts.save_json(
                marker,
                {
                    "run_ref": str(run_id),
                    "claim_ref": str(claim_receipt.claim_id),
                    "evidence_ref": str(evidence.evidence_id),
                    "state": "COMPLETED",
                    "result_artifact_ref": product_uri,
                },
            )
        if source_result.runtime_status not in {
            SourceSemanticRuntimeStatus.COMPLETED,
            SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED,
        }:
            self._diagnostic_evidence(
                run_id,
                claim_receipt.claim_id,
                evidence.evidence_id,
                SemanticShadowStage.SOURCE_ASSERTION_FORMATION,
                source_result.runtime_status.value,
                _first_reason(source_result.reason_codes, "SOURCE_RUNTIME_NOT_COMPLETED"),
            )
            return
        canonical_claim = self._claims[claim_receipt.claim_id][0]
        claim_evidence = ClaimEvidenceLink(
            claim_id=claim_receipt.claim_id,
            evidence_id=evidence.evidence_id,
            relation=ClaimEvidenceRelation.INSUFFICIENT,
            source_id=evidence.source_id,
            source_owner_key=evidence.source_owner_key,
            quote_reference=span.span_ref,
            created_by="semantic-shadow-persisted-receipt",
        )
        targets = tuple(
            PropositionDirectionTarget(
                membership=self._claim_links[(claim_receipt.claim_id, proposition.proposition_id)],
                proposition=proposition,
                assertion=self._target_assertions[proposition.proposition_id],
            )
            for proposition in self._claims[claim_receipt.claim_id][2]
        )
        source_bindings = tuple(
            SourceAssertionEvidenceBinding(
                source_assertion_ref=assertion.assertion_ref,
                evidence_id=evidence.evidence_id,
                source_id=evidence.source_id,
                source_ref=assertion.source_ref,
                snapshot_ref=assertion.snapshot_ref,
                canonical_evidence_span_ref=assertion.canonical_evidence_span_ref,
                source_owner_key=evidence.source_owner_key,
            )
            for assertion in source_result.assertions
        )
        measurement_bindings = tuple(
            MeasurementAdmissibilityBinding(context_ref=ref, admissibility=value)
            for ref, value in sorted(source_result.measurement_admissibility.items())
        )
        alignment_pairs = tuple(
            IdentityAlignmentPair(
                target=target.proposition,
                source=assertion,
                target_proposition_ref=str(target.proposition.proposition_id),
                source_grounding_refs=tuple(
                    sorted(
                        {
                            ref
                            for grounded in assertion.field_grounding
                            for ref in grounded.evidence_refs
                        }
                    )
                ),
                context_refs=assertion.truth_condition_context_refs,
                target_context_payload=tuple(
                    {"context_ref": str(index), "qualifier": qualifier.model_dump(mode="json")}
                    for index, qualifier in enumerate(target.proposition.qualifiers)
                ),
            )
            for target in targets
            for assertion in source_result.assertions
        )
        alignment_uri = _alignment_product_uri(
            run_id, claim_receipt.claim_id, evidence, span, alignment_pairs
        )
        alignment_result = await self._load_product(alignment_uri, IdentityAlignmentRuntimeResult)
        if alignment_result is not None:
            self._restore_provider_counter(
                alignment_uri,
                "provider_identity_resolution_calls",
                alignment_result.model_call_count,
            )
        else:
            marker = _attempt_marker(
                run_id, "alignment", evidence.evidence_id, alignment_cache_ref(alignment_pairs)
            )
            if await self._completed_marker_without_product(marker):
                self._diagnostic_evidence(
                    run_id,
                    claim_receipt.claim_id,
                    evidence.evidence_id,
                    SemanticShadowStage.DIRECTION,
                    "INVALID_INPUT",
                    "COMPLETED_ALIGNMENT_PRODUCT_MISSING",
                )
                return
            eligible_grounding = any(
                _alignment_pair_has_grounded_field(pair) for pair in alignment_pairs
            )
            if eligible_grounding:
                self._state, allowed = self._state.consume(
                    budget, "provider_identity_resolution_calls"
                )
                if not allowed:
                    self._diagnostic_evidence(
                        run_id,
                        claim_receipt.claim_id,
                        evidence.evidence_id,
                        SemanticShadowStage.BUDGET,
                        "NOT_PROCESSED_BUDGET",
                        "IDENTITY_ALIGNMENT_PROVIDER_BUDGET_EXHAUSTED",
                    )
                    return
                binding, api_key = await self._provider_binding(run_id)
                alignment_runtime = SemanticIdentityAlignmentRuntime(
                    _RunScopedGateway(self._gateway, run_id),
                    adapter_type=binding.adapter_type,
                    base_url=binding.base_url,
                    api_key=api_key,
                    model=binding.model,
                )
                alignment_result = await alignment_runtime.align(
                    alignment_pairs,
                    {item.context_ref: item for item in source_result.truth_condition_contexts},
                )
            else:
                alignment_result = _unknown_alignment_result(alignment_pairs)
            await self._artifacts.save_json(alignment_uri, alignment_result.model_dump(mode="json"))
            self._restored_products.add(alignment_uri)
            await self._artifacts.save_json(
                marker,
                {
                    "run_ref": str(run_id),
                    "claim_ref": str(claim_receipt.claim_id),
                    "evidence_ref": str(evidence.evidence_id),
                    "state": "COMPLETED",
                    "result_artifact_ref": alignment_uri,
                },
            )
        if alignment_result.status in {
            AlignmentRuntimeStatus.FAILED,
            AlignmentRuntimeStatus.INVALID,
        }:
            self._diagnostic_evidence(
                run_id,
                claim_receipt.claim_id,
                evidence.evidence_id,
                SemanticShadowStage.DIRECTION,
                alignment_result.status.value,
                _first_reason(alignment_result.reason_codes, "IDENTITY_ALIGNMENT_FAILED"),
            )
            return
        direction_scope = EvidenceDirectionScope(
            claim=canonical_claim,
            claim_evidence=claim_evidence,
            targets=targets,
            source_assertions=source_result.assertions,
            source_bindings=source_bindings,
            truth_condition_contexts=source_result.truth_condition_contexts,
            source_contexts=(
                source_result.source_extraction.source_contexts
                if source_result.source_extraction is not None
                else ()
            ),
            measurement_admissibility=measurement_bindings,
        )
        direction = evaluate_proposition_evidence_directions(
            (direction_scope,),
            budget=budget,
            initial_state=self._state,
            identity_alignments=alignment_result.facts,
        )
        self._state = direction.budget_state
        for link in direction.links:
            link_key = (link.claim_id, link.evidence_id, link.relation.value)
            self._evidence_links[link_key] = link
        self._acceptance_by_evidence[evidence.evidence_id] = True
        self._source_by_evidence[evidence.evidence_id] = (
            evidence.source_id,
            evidence.source_owner_key,
            claim_receipt.source_diversity_key,
        )
        self._diagnostic_evidence(
            run_id,
            claim_receipt.claim_id,
            evidence.evidence_id,
            SemanticShadowStage.DIRECTION,
            "COMPLETED",
            "DIRECTION_ASSESSMENTS_CREATED",
        )

    async def _persist_claim_target_state(
        self, run_id: UUID, receipt: ClaimPersistenceReceipt
    ) -> None:
        decomposition = self._ap_cache.get(
            (
                receipt.claim_id,
                ATOMIC_PROPOSITION_SCHEMA_V3,
                ATOMIC_DECOMPOSITION_PROMPT_V3,
            )
        )
        memberships = tuple(
            item
            for (claim_id, _), item in self._claim_links.items()
            if claim_id == receipt.claim_id
        )
        if decomposition is None or not memberships:
            return
        evidence_ids = {
            ref for claim_id, ref in self._processed_receipts if claim_id == receipt.claim_id
        }
        state = ShadowTargetState(
            run_ref=run_id,
            claim_ref=receipt.claim_id,
            ap_product_ref=_ap_product_uri(run_id, receipt),
            formation_product_ref=_formation_product_uri(run_id, receipt, decomposition),
            formation_assessments=tuple(
                self._formation_assessments[item.proposition_id]
                for item in memberships
                if item.proposition_id in self._formation_assessments
            ),
            composition=decomposition.assertion_composition,
            assertions=tuple(self._target_assertions[item.proposition_id] for item in memberships),
            claim_links=memberships,
            evidence_links=tuple(
                item for item in self._evidence_links.values() if item.claim_id == receipt.claim_id
            ),
            acceptance={
                ref: self._acceptance_by_evidence[ref]
                for ref in evidence_ids
                if ref in self._acceptance_by_evidence
            },
            sources={
                ref: self._source_by_evidence[ref]
                for ref in evidence_ids
                if ref in self._source_by_evidence
            },
            processed_evidence=tuple(sorted(evidence_ids, key=str)),
        )
        payload = _target_state_payload(state)
        uri = _product_ref("target-state", run_id, content_hash(payload))
        await self._artifacts.save_json(uri, payload)
        self._claim_state_refs[str(receipt.claim_id)] = uri

    async def _save_derived_index(self, run_id: UUID) -> None:
        await self._artifacts.save_json(
            _derived_index_uri(run_id),
            {
                "schema_version": "semantic-shadow-derived-index-v1",
                "run_ref": str(run_id),
                "claim_state_refs": self._claim_state_refs,
                "pair_batch_refs": sorted(self._pair_batch_refs),
                "pair_attempted_refs": sorted(self._pair_attempted_refs),
                "pair_execution_states": self._pair_execution_states,
                "accounted_product_refs": sorted(self._restored_products),
                "budget_state": self._state.model_dump(mode="json"),
            },
        )

    async def _restore_derived_index(self, run_id: UUID) -> None:
        if self._index_run == run_id:
            return
        if self._index_run is not None:
            raise ValueError("DERIVED_INDEX_RUN_MISMATCH")
        try:
            raw = await self._artifacts.read_json(_derived_index_uri(run_id))
        except (FileNotFoundError, KeyError):
            self._index_run = run_id
            return
        if not isinstance(raw, dict) or raw.get("run_ref") != str(run_id):
            raise ValueError("DERIVED_INDEX_RUN_MISMATCH")
        if raw.get("schema_version") != "semantic-shadow-derived-index-v1":
            raise ValueError("DERIVED_INDEX_VERSION_MISMATCH")
        self._state = SemanticRuntimeBudgetState.model_validate(raw["budget_state"])
        self._restored_products.update(raw["accounted_product_refs"])
        self._claim_state_refs = dict(raw["claim_state_refs"])
        for claim_ref, uri in sorted(self._claim_state_refs.items()):
            state = await self._load_product(uri, ShadowTargetState)
            if state is None or state.run_ref != run_id or str(state.claim_ref) != claim_ref:
                raise ValueError("TARGET_STATE_BINDING_MISMATCH")
            if uri != _product_ref(
                "target-state", run_id, content_hash(_target_state_payload(state))
            ):
                raise ValueError("TARGET_STATE_HASH_MISMATCH")
            decomposition = await self._load_product(
                state.ap_product_ref, ShadowDecompositionResult
            )
            formation = await self._load_product(
                state.formation_product_ref, TargetFormationRuntimeResult
            )
            if decomposition is None:
                raise ValueError("TARGET_STATE_INPUT_PRODUCT_MISSING")
            props = {item.proposition_id: item for item in decomposition.propositions}
            assessments = {
                item.proposition_ref: item
                for item in (
                    formation.assessments if formation is not None else state.formation_assessments
                )
            }
            for membership, assertion in zip(state.claim_links, state.assertions, strict=True):
                prop_ref = membership.proposition_id
                assessment = assessments.get(prop_ref)
                if (
                    membership.claim_id != state.claim_ref
                    or prop_ref not in props
                    or assertion.assertion_ref != str(prop_ref)
                    or assertion.source_proposition_id != prop_ref
                    or assertion.grounding_usability != _target_grounding_from_formation(assessment)
                    or (assessment is not None and assessment.claim_ref != state.claim_ref)
                ):
                    raise ValueError("TARGET_STATE_PROVENANCE_MISMATCH")
                self._known_propositions[prop_ref] = props[prop_ref]
                self._target_compositions[prop_ref] = state.composition
                self._target_assertions[prop_ref] = assertion
                self._claim_links[(state.claim_ref, prop_ref)] = membership
                if assessment is not None:
                    self._formation_assessments[prop_ref] = assessment
            for link in state.evidence_links:
                if link.run_id != run_id or link.claim_id != state.claim_ref:
                    raise ValueError("TARGET_STATE_LINK_MISMATCH")
                self._evidence_links[(link.claim_id, link.evidence_id, link.relation.value)] = link
            self._acceptance_by_evidence.update(state.acceptance)
            self._source_by_evidence.update(state.sources)
            self._processed_receipts.update(
                (state.claim_ref, ref) for ref in state.processed_evidence
            )
        self._pair_batch_refs.update(raw["pair_batch_refs"])
        self._pair_attempted_refs.update(raw.get("pair_attempted_refs", ()))
        self._pair_execution_states.update(raw.get("pair_execution_states", {}))
        for uri in sorted(self._pair_batch_refs):
            product = await self._load_product(uri, PairAlignmentRuntimeResult)
            if product is not None:
                if any(binding.run_ref != run_id for binding in product.bindings):
                    raise ValueError("PAIR_PRODUCT_RUN_MISMATCH")
                for binding in product.bindings:
                    self._pair_results[binding.pair_ref] = product
        self._index_run = run_id

    def _pair_targets(self) -> tuple[PropositionPairTarget, ...]:
        targets = []
        for ref, proposition in sorted(
            self._known_propositions.items(), key=lambda item: str(item[0])
        ):
            formation = self._formation_assessments.get(ref)
            targets.append(
                PropositionPairTarget(
                    proposition=proposition,
                    assertion=self._target_assertions[ref],
                    claim_refs=tuple(
                        sorted({claim for claim, prop in self._claim_links if prop == ref}, key=str)
                    ),
                    formation_ref=formation.assessment_ref if formation is not None else "",
                    formation_status=formation.formation_status.value
                    if formation
                    else "FORMATION_UNKNOWN",
                    composition=self._target_compositions.get(ref),
                )
            )
        return tuple(targets)

    async def _align_proposition_pairs(self, run_id: UUID) -> list[PropositionPairAssessment]:
        budget = self._budget
        if budget is None:
            return []
        targets = self._pair_targets()
        plan = plan_proposition_pairs(
            run_id,
            targets,
            max_pairs=budget.max_candidate_pairs,
            batch_size=budget.proposition_pair_batch_size,
            max_request_bytes=budget.proposition_pair_request_bytes,
        )
        self._pair_plan_pairs = plan.pairs
        self._pair_plan_summary = {
            "limits": {
                "max_pair_count": budget.max_candidate_pairs,
                "batch_size": budget.proposition_pair_batch_size,
                "max_request_bytes": budget.proposition_pair_request_bytes,
            },
            "enumerated_pair_count": plan.enumerated_pair_count,
            "pair_refs": [pair.binding.pair_ref for pair in plan.pairs],
            "batch_refs": [content_hash(request_payload(batch)) for batch in plan.batches],
            "oversized_pair_refs": list(plan.oversized_pair_refs),
        }
        for ref in plan.oversized_pair_refs:
            self._pair_execution_states[ref] = "NOT_PROCESSED_REQUEST_SIZE"
        plan_uri = _product_ref("pair-plan", run_id, content_hash(self._pair_plan_summary))
        await self._artifacts.save_json(plan_uri, self._pair_plan_summary)
        # Decide all admissible batches before observing any semantic output.
        scheduled: list[tuple[tuple[PropositionPairInput, ...], str]] = []
        for batch in plan.batches:
            pending = tuple(
                pair for pair in batch if pair.binding.pair_ref not in self._pair_results
            )
            if any(pair.binding.pair_ref in self._pair_attempted_refs for pair in pending):
                self._pair_diagnostic(run_id, "FAILED_ISOLATED", "PAIR_ATTEMPT_WITHOUT_PRODUCT")
                for pair in pending:
                    if pair.binding.pair_ref in self._pair_attempted_refs:
                        self._pair_execution_states[pair.binding.pair_ref] = "FAILED_ISOLATED"
            pending = tuple(
                pair for pair in pending if pair.binding.pair_ref not in self._pair_attempted_refs
            )
            if not pending:
                continue
            uri = _product_ref(
                "proposition-pair-identity", run_id, content_hash(request_payload(pending))
            )
            restored = await self._load_product(uri, PairAlignmentRuntimeResult)
            if restored is not None:
                validate_pair_product(restored, pending)
                self._restore_provider_counter(
                    uri, "provider_identity_resolution_calls", restored.model_call_count
                )
                self._pair_batch_refs.add(uri)
                for pair in pending:
                    self._pair_results[pair.binding.pair_ref] = restored
                    self._pair_execution_states[pair.binding.pair_ref] = restored.status
                continue
            marker = f"{uri}.attempt.json"
            try:
                await self._artifacts.read_json(marker)
            except (FileNotFoundError, KeyError):
                pass
            else:
                self._pair_diagnostic(run_id, "FAILED_ISOLATED", "PAIR_ATTEMPT_WITHOUT_PRODUCT")
                for pair in pending:
                    self._pair_execution_states[pair.binding.pair_ref] = "FAILED_ISOLATED"
                continue
            if (
                self._state.provider_identity_resolution_calls
                >= budget.max_identity_resolution_calls
                or not self._state.can_consume_direction(budget)
            ):
                self._pair_diagnostic(
                    run_id, "NOT_PROCESSED_BUDGET", "PAIR_IDENTITY_BUDGET_EXHAUSTED"
                )
                for pair in pending:
                    self._pair_execution_states[pair.binding.pair_ref] = "NOT_PROCESSED_BUDGET"
                continue
            admitted = []
            for pair in pending:
                self._state, allowed = self._state.consume_direction_pair(budget)
                if not allowed:
                    self._pair_diagnostic(
                        run_id, "NOT_PROCESSED_BUDGET", "PAIR_EVALUATION_BUDGET_EXHAUSTED"
                    )
                    break
                admitted.append(pair)
            admitted_batch = tuple(admitted)
            for pair in pending[len(admitted_batch) :]:
                self._pair_execution_states[pair.binding.pair_ref] = "NOT_PROCESSED_BUDGET"
            uri = _product_ref(
                "proposition-pair-identity", run_id, content_hash(request_payload(admitted_batch))
            )
            marker = f"{uri}.attempt.json"
            if not await self._artifacts.create_json_once(
                marker,
                {
                    "run_ref": str(run_id),
                    "state": "RESERVED",
                    "product_ref": uri,
                },
            ):
                continue
            self._state, allowed = self._state.consume(budget, "provider_identity_resolution_calls")
            if not allowed:
                raise ValueError("PAIR_BUDGET_RESERVATION_MISMATCH")
            self._pair_batch_refs.add(uri)
            self._pair_attempted_refs.update(pair.binding.pair_ref for pair in admitted_batch)
            for pair in admitted_batch:
                self._pair_execution_states[pair.binding.pair_ref] = "RESERVED"
            self._restored_products.add(uri)
            scheduled.append((admitted_batch, uri))
        await self._save_derived_index(run_id)
        for batch, uri in scheduled:
            binding, api_key = await self._provider_binding(run_id)
            runtime = PropositionPairSemanticAlignmentRuntime(
                _RunScopedGateway(self._gateway, run_id),
                adapter_type=binding.adapter_type,
                base_url=binding.base_url,
                api_key=api_key,
                model=binding.model,
                max_request_bytes=budget.proposition_pair_request_bytes,
            )
            result = await runtime.align(batch)
            await self._artifacts.save_json(uri, result.model_dump(mode="json"))
            await self._artifacts.save_json(
                f"{uri}.attempt.json",
                {
                    "run_ref": str(run_id),
                    "state": "COMPLETED",
                    "product_ref": uri,
                },
            )
            for pair in batch:
                self._pair_results[pair.binding.pair_ref] = result
                self._pair_execution_states[pair.binding.pair_ref] = result.status
            self._pair_diagnostic(run_id, result.status, "PAIR_IDENTITY_PRODUCT_CREATED")
        await self._save_derived_index(run_id)
        by_ref = {target.proposition.proposition_id: target for target in targets}
        comparisons = []
        self._pair_assessment_provenance = []
        seen: set[str] = set()
        for product in self._pair_results.values():
            if product.status != "COMPLETED":
                continue
            product_pairs = tuple(
                PropositionPairInput(
                    binding=binding,
                    left=by_ref[binding.left_proposition_ref],
                    right=by_ref[binding.right_proposition_ref],
                )
                for binding in product.bindings
            )
            validate_pair_product(product, product_pairs)
            for binding in product.bindings:
                if binding.pair_ref in seen:
                    continue
                seen.add(binding.pair_ref)
                left, right = (
                    by_ref[binding.left_proposition_ref],
                    by_ref[binding.right_proposition_ref],
                )
                pair = PropositionPairInput(binding=binding, left=left, right=right)
                views = project_pair_views(
                    pair, tuple(fact for fact in product.facts if fact.binding == binding)
                )
                a2 = ComparabilityEvaluator.evaluate(*views)
                a3 = RelationEvaluator.evaluate(*views, a2)
                comparisons.append(a3)
                self._pair_assessment_provenance.append(
                    {
                        "pair_ref": binding.pair_ref,
                        "alignment_product_ref": _product_ref(
                            "proposition-pair-identity",
                            run_id,
                            content_hash(request_payload(product_pairs)),
                        ),
                        "alignment_refs": [
                            fact.alignment_ref for fact in product.facts if fact.binding == binding
                        ],
                        "assessment": a3.model_dump(mode="json"),
                    }
                )
        self._last_pair_assessments = tuple(comparisons)
        return comparisons

    def _pair_diagnostic(self, run_id: UUID, status: str, reason: str) -> None:
        self._diagnostics.append(
            SemanticShadowDiagnostic(
                run_ref=run_id,
                stage=SemanticShadowStage.PROPOSITION_PAIR_IDENTITY,
                status=status,
                reason_code=reason,
            )
        )

    def _diagnostic_evidence(
        self,
        run_id: UUID,
        claim_id: UUID,
        evidence_id: UUID,
        stage: SemanticShadowStage,
        status: str,
        reason: str,
    ) -> None:
        self._diagnostics.append(
            SemanticShadowDiagnostic(
                run_ref=run_id,
                claim_ref=claim_id,
                evidence_ref=evidence_id,
                stage=stage,
                status=status,
                reason_code=reason,
            )
        )

    async def _rebuild_deterministic_assessments(self, run_id: UUID) -> None:
        budget = self._budget
        if budget is None:
            return
        claim_links = tuple(
            self._claim_links[key]
            for key in sorted(self._claim_links, key=lambda item: (str(item[0]), str(item[1])))
        )
        assertion_bindings = tuple(
            PropositionAssertionBinding(
                proposition_ref=proposition_id,
                assertion_ref=assertion.assertion_ref,
            )
            for proposition_id, assertion in sorted(
                self._target_assertions.items(), key=lambda x: str(x[0])
            )
        )
        pair_assessments = await self._align_proposition_pairs(run_id)
        links = tuple(
            self._evidence_links[key]
            for key in sorted(
                self._evidence_links, key=lambda item: (str(item[0]), str(item[1]), item[2])
            )
        )
        grouped = group_propositions(claim_links, links, assertion_bindings, pair_assessments)
        corroborations = []
        for group in grouped.groups:
            refs = sorted(
                {membership.evidence_ref for membership in group.evidence_memberships}, key=str
            )
            identities = {
                ref: EvidenceSourceIdentity(
                    evidence_ref=ref,
                    source_ref=self._source_by_evidence[ref][0],
                    source_owner_key=self._source_by_evidence[ref][1],
                    source_diversity_key=self._source_by_evidence[ref][2],
                )
                for ref in refs
                if ref in self._source_by_evidence
            }
            owner_assessments = tuple(
                assess_owner_independence(identities[left], identities[right])
                for index, left in enumerate(sorted(identities, key=str))
                for right in sorted(identities, key=str)[index + 1 :]
            )
            corroborations.append(
                aggregate_independent_corroboration(
                    group,
                    tuple(
                        CorroborationEvidenceEligibilityInput(
                            group_ref=group.group_ref,
                            evidence_ref=ref,
                            accepted=self._acceptance_by_evidence.get(ref),
                        )
                        for ref in refs
                    ),
                    owner_assessments,
                )
            )
        conflicts = detect_proposition_conflicts(
            grouped.groups,
            tuple(pair_assessments),
            assertion_bindings,
            corroboration_assessments=tuple(corroborations),
        )
        completeness = ConflictScopeCompletenessEvaluator().evaluate(
            run_ref=run_id,
            groups=grouped.groups,
            assertion_bindings=assertion_bindings,
            coverage=conflicts,
            planned_pairs=tuple(
                PlannedPropositionPair(
                    pair_ref=item.binding.pair_ref,
                    left_proposition_ref=item.binding.left_proposition_ref,
                    right_proposition_ref=item.binding.right_proposition_ref,
                )
                for item in self._pair_plan_pairs
            ),
            pair_execution_states=self._pair_execution_states,
            pair_assessments=tuple(pair_assessments),
        )
        verification = tuple(
            aggregate_proposition_verification(
                group,
                next((item for item in corroborations if item.group_ref == group.group_ref), None),
                conflicts,
                conflict_scope=completeness.for_target(group.group_ref),
            )
            for group in grouped.groups
        )
        self._last_deterministic = {
            "groups": grouped.groups,
            "corroborations": tuple(corroborations),
            "conflicts": conflicts,
            "verification": verification,
            "conflict_scope_completeness": completeness,
        }

    async def _persist_snapshot(self, run_id: UUID) -> str:
        artifact_ref, _ = await self._persist_snapshot_document(run_id)
        return artifact_ref

    async def _persist_snapshot_document(
        self, run_id: UUID
    ) -> tuple[str, dict[str, Any]]:
        deterministic = getattr(self, "_last_deterministic", {})
        groups = tuple(deterministic.get("groups", ()))
        corroborations = tuple(deterministic.get("corroborations", ()))
        conflicts = deterministic.get("conflicts")
        verification = tuple(deterministic.get("verification", ()))
        links = tuple(
            self._evidence_links[key]
            for key in sorted(
                self._evidence_links, key=lambda item: (str(item[0]), str(item[1]), item[2])
            )
        )
        payload: dict[str, Any] = {
            "run_ref": str(run_id),
            "claim_refs": sorted({str(claim) for claim, _ in self._claim_links}),
            "evidence_refs": sorted(str(item) for item in self._acceptance_by_evidence),
            "proposition_refs": sorted(str(item) for item in self._target_assertions),
            "formation_assessments": [
                item.model_dump(mode="json")
                for _, item in sorted(
                    self._formation_assessments.items(), key=lambda pair: str(pair[0])
                )
            ],
            "proposition_group_refs": [item.group_ref for item in groups],
            "proposition_groups": [item.model_dump(mode="json") for item in groups],
            "pair_assessments": self._pair_assessment_provenance,
            "pair_candidate_plan": self._pair_plan_summary,
            "pair_execution_states": self._pair_execution_states,
            "proposition_evidence_links": [item.model_dump(mode="json") for item in links],
            "corroboration_assessments": [item.model_dump(mode="json") for item in corroborations],
            "conflicts": conflicts.model_dump(mode="json") if conflicts is not None else {},
            "verification_assessments": [item.model_dump(mode="json") for item in verification],
            "conflict_scope_completeness": (
                deterministic["conflict_scope_completeness"].model_dump(mode="json")
                if deterministic.get("conflict_scope_completeness") is not None
                else None
            ),
            "budget_state": self._state.model_dump(mode="json"),
            "diagnostics": [item.model_dump(mode="json") for item in self._diagnostics],
            "schema_versions": {
                "semantic_snapshot": "semantic-shadow-verification-snapshot-v2",
                "atomic_proposition": ATOMIC_PROPOSITION_SCHEMA_V3,
                "atomic_proposition_prompt": ATOMIC_DECOMPOSITION_PROMPT_V3,
                "source_runtime": SOURCE_SEMANTIC_RUNTIME_PROMPT_V2,
                "target_formation_schema": TARGET_FORMATION_SCHEMA_VERSION,
                "target_formation_prompt": TARGET_FORMATION_PROMPT_VERSION,
                "target_formation_validator": TARGET_FORMATION_VALIDATOR_VERSION,
                "direction": "proposition-evidence-direction-v1",
                "proposition_groups": "proposition-support-group-v1",
                "corroboration": "independent-corroboration-v1",
                "conflicts": "proposition-conflict-assessment-v1",
                "conflict_scope_completeness": "conflict-scope-completeness-v1",
                "verification": "proposition-verification-assessment-v1",
                "proposition_pair_identity": PAIR_IDENTITY_RUNTIME,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        snapshot_ref = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        artifact_ref = f"runs/{run_id}/semantic-shadow/{snapshot_ref}.json"
        payload["snapshot_ref"] = snapshot_ref
        payload["artifact_ref"] = artifact_ref
        snapshot = SemanticShadowVerificationSnapshot.model_validate(payload)
        serialized = snapshot.model_dump(mode="json")
        if self._run_orchestrator_versions.get(run_id) == "V3":
            await self._artifacts.save_immutable_json(artifact_ref, serialized)
        else:
            # Preserve the existing V2.3 shadow artifact behavior for legacy runs.
            await self._artifacts.save_json(artifact_ref, serialized)
        return artifact_ref, serialized

    async def _publish_v3_run_state(
        self,
        run_id: UUID,
        product_bundle_ref: str,
        product_bundle: dict[str, Any],
    ) -> None:
        if (
            self._run_state_runs is None
            or self._run_state_reducer is None
            or self._run_state_publisher is None
            or self._semantic_engine is None
        ):
            return
        inputs = await self._run_state_runs.get_run_state_inputs(run_id)
        if inputs.get("orchestrator_version") != "V3":
            return
        universe = await self._run_state_runs.load_evidence_universe(run_id)
        previous = await self._run_state_publisher.load_current(run_id)
        try:
            semantic_snapshot = await self._semantic_engine.evaluate(
                run_ref=run_id,
                evidence_universe=universe,
                execution_context={
                    "semantic_products": product_bundle,
                    "product_bundle_ref": product_bundle_ref,
                },
                previous_semantic_state_ref=(
                    previous.semantic_state_ref if previous is not None else None
                ),
            )
        except Exception:
            # Do not leak exception text or pretend a failed integrity/rebuild
            # check is merely incomplete. The fixed code is non-sensitive and
            # the snapshot remains bound to this exact durable universe.
            failed_snapshot = getattr(self._semantic_engine, "failed_snapshot", None)
            if failed_snapshot is None:
                raise
            semantic_snapshot = failed_snapshot(
                run_ref=run_id,
                evidence_universe=universe,
            )
        control = {
            "status": inputs["status"],
            "phase": inputs["phase"],
            "budget": inputs["budget"],
            "usage": inputs["usage"],
            "termination_reason": inputs["termination_reason"],
        }
        projection = None
        if semantic_snapshot.execution_status is SemanticExecutionStatus.COMPLETE:
            projection = await self._semantic_control_projection.recover(semantic_snapshot)
        projection_status = (
            GapProjectionStatus.DERIVED
            if projection is not None
            and projection.gap_artifact.projection_status == "DERIVED"
            else GapProjectionStatus.INCOMPLETE
        )
        artifact_refs = [
            ("semantic_snapshot_candidate", semantic_snapshot.artifact_ref),
            ("semantic_snapshot_hash", semantic_snapshot.snapshot_hash),
        ]
        if projection is not None:
            artifact_refs.extend(
                (
                    ("gap_projection_artifact_ref", projection.gap_artifact.artifact_ref),
                    ("gap_projection_artifact_hash", projection.gap_artifact.artifact_hash),
                    ("intent_projection_artifact_ref", projection.intent_artifact.artifact_ref),
                    ("intent_projection_artifact_hash", projection.intent_artifact.artifact_hash),
                )
            )
        state = self._run_state_reducer.reduce(
            run_ref=run_id,
            run_control=control,
            evidence_universe=universe,
            semantic_snapshot=semantic_snapshot,
            previous=previous,
            gap_projection_status=projection_status,
            gap_refs=(
                tuple(sorted(gap.gap_ref for gap in projection.gap_artifact.gaps))
                if projection_status is GapProjectionStatus.DERIVED and projection is not None
                else ()
            ),
            gap_source_semantic_hash=(
                semantic_snapshot.snapshot_hash
                if projection_status is GapProjectionStatus.DERIVED
                else None
            ),
            gap_projection_hash=(
                projection.gap_artifact.projection_hash
                if projection_status is GapProjectionStatus.DERIVED and projection is not None
                else None
            ),
            intent_refs=(
                tuple(sorted(intent.intent_ref for intent in projection.intent_artifact.intents))
                if projection_status is GapProjectionStatus.DERIVED and projection is not None
                else ()
            ),
            intent_source_gap_hash=(
                projection.gap_artifact.projection_hash
                if projection_status is GapProjectionStatus.DERIVED and projection is not None
                else None
            ),
            known_limitations=(
                tuple(
                    sorted(
                        {
                            *(
                                {"SEMANTIC_PRODUCTS_INCOMPLETE"}
                                if semantic_snapshot.execution_status
                                is SemanticExecutionStatus.INCOMPLETE
                                else set()
                            ),
                            *(
                                {"GAP_PROJECTION_INCOMPLETE"}
                                if projection_status is GapProjectionStatus.INCOMPLETE
                                else set()
                            ),
                            *(
                                {"INTENT_PROJECTION_INCOMPLETE"}
                                if projection is not None and projection.intent_issues
                                else set()
                            ),
                        }
                    )
                )
            ),
            execution_refs=tuple(artifact_refs),
        )
        await self._run_state_publisher.publish(
            state=state,
            semantic_snapshot=semantic_snapshot,
        )


def _receipt_question(receipt: ClaimPersistenceReceipt) -> str:
    questions = {item.question_id for item in receipt.evidence_receipts}
    if len(questions) != 1:
        raise ValueError("claim_receipt_question_scope_mismatch")
    return next(iter(questions))


def _receipt_dimension(receipt: ClaimPersistenceReceipt) -> str:
    dimensions = {item.dimension_key for item in receipt.evidence_receipts}
    if len(dimensions) != 1:
        raise ValueError("claim_receipt_dimension_scope_mismatch")
    return next(iter(dimensions))


def _first_reason(reasons: tuple[str, ...], fallback: str) -> str:
    return reasons[0] if reasons else fallback


def _safe_reason(exc: Exception) -> str:
    safe_codes = {
        "invalid_provenance": "INVALID_PROVENANCE",
        "provider_unavailable": "PROVIDER_BINDING_UNAVAILABLE",
        "budget_exhausted": "BUDGET_EXHAUSTED",
    }
    name = type(exc).__name__.casefold()
    for needle, code in safe_codes.items():
        if needle in name or needle in str(exc).casefold():
            return code
    return "SEMANTIC_SHADOW_STAGE_FAILED"


def _attempt_marker(run_id: UUID, kind: str, identity: UUID, version: object) -> str:
    key = json.dumps(version, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"runs/{run_id}/semantic-shadow/attempts/{kind}-{identity}-{fingerprint}.json"


def _product_ref(kind: str, run_id: UUID, identity: object) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"runs/{run_id}/semantic-shadow/work-products/{kind}-{digest}.json"


def _ap_product_uri(run_id: UUID, receipt: ClaimPersistenceReceipt) -> str:
    return _product_ref(
        "ap",
        run_id,
        [
            str(receipt.claim_id),
            receipt.claim_content_hash,
            ATOMIC_PROPOSITION_SCHEMA_V3,
            ATOMIC_DECOMPOSITION_PROMPT_V3,
        ],
    )


TARGET_FORMATION_PRODUCT_VERSION = (
    TARGET_FORMATION_SCHEMA_VERSION,
    TARGET_FORMATION_PROMPT_VERSION,
    TARGET_FORMATION_VALIDATOR_VERSION,
)


def _formation_product_uri(
    run_id: UUID,
    receipt: ClaimPersistenceReceipt,
    decomposition: ShadowDecompositionResult,
) -> str:
    ap_identity = [
        {
            "proposition_id": str(item.proposition_id),
            "payload": item.model_dump(mode="json"),
        }
        for item in sorted(decomposition.propositions, key=lambda value: str(value.proposition_id))
    ]
    return _product_ref(
        "target-formation",
        run_id,
        [
            str(receipt.claim_id),
            receipt.claim_content_hash,
            decomposition.schema_version,
            decomposition.prompt_version,
            ap_identity,
            decomposition.assertion_composition.model_dump(mode="json")
            if decomposition.assertion_composition is not None
            else None,
            TARGET_FORMATION_PRODUCT_VERSION,
        ],
    )


def _target_grounding_from_formation(
    assessment: Any,
) -> AssertionGroundingUsability:
    if assessment is None:
        return AssertionGroundingUsability.UNKNOWN
    if assessment.formation_status is FormationStatus.FORMATION_SUFFICIENT:
        return AssertionGroundingUsability.SUFFICIENT
    if assessment.formation_status is FormationStatus.FORMATION_NOT_PRESERVED:
        return AssertionGroundingUsability.INSUFFICIENT
    return AssertionGroundingUsability.UNKNOWN


def _project_target_assertion(
    proposition: AtomicProposition,
    *,
    decomposition: ShadowDecompositionResult,
    claim_id: UUID,
    claim_text: str,
    proposition_ids: set[UUID],
    output_graph: ContextGraph,
) -> Any:
    assertion = project_verification_assertion(
        proposition,
        proposition_ref=str(proposition.proposition_id),
        output_context_graph=output_graph,
        composition=decomposition.assertion_composition,
        claim_id=claim_id,
        claim_text=claim_text,
        proposition_ids=proposition_ids,
    )
    if decomposition.assertion_composition is not None or len(proposition_ids) != 1:
        return assertion
    # The frozen v3 composition validator represents a valid single-proposition
    # decomposition as one direct root and derives its eligibility. Project that
    # already-validated root instead of treating absent group metadata as ambiguity.
    local_ref = next(
        (
            ref
            for ref, runtime_ref in decomposition.model_local_references.items()
            if runtime_ref == str(proposition.proposition_id)
        ),
        None,
    )
    direct_roots = tuple(
        item for item in decomposition.root_assertions if item.get("member_type") == "proposition"
    )
    if (
        local_ref is not None
        and len(decomposition.root_assertions) == 1
        and len(direct_roots) == 1
        and direct_roots[0].get("ref") == local_ref
        and decomposition.derived_member_eligibility.get(local_ref) == "INDEPENDENTLY_ASSERTED"
    ):
        return assertion.model_copy(
            update={"composition_eligibility": ComparisonEligibility.ELIGIBLE}
        )
    return assertion


def _source_product_uri(run_id: UUID, claim_id: UUID, evidence: Any, span: Any) -> str:
    return _product_ref(
        "source",
        run_id,
        [
            str(claim_id),
            str(evidence.evidence_id),
            span.span_ref,
            SOURCE_SEMANTIC_RUNTIME_PROMPT_V2,
            "source-context-extraction-schema-v3.4",
        ],
    )


def _alignment_product_uri(
    run_id: UUID,
    claim_id: UUID,
    evidence: Any,
    span: Any,
    pairs: tuple[IdentityAlignmentPair, ...],
) -> str:
    return _product_ref(
        "identity-alignment",
        run_id,
        [str(claim_id), str(evidence.evidence_id), span.span_ref, alignment_cache_ref(pairs)],
    )


def _ap_product_payload(result: ShadowDecompositionResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json", exclude_none=True)
    for item, proposition in zip(payload["propositions"], result.propositions, strict=True):
        item["value"] = proposition.value.model_dump(mode="json", exclude_unset=True)
    # Do not retain raw provider envelope/diagnostic parser copies; the typed,
    # validated propositions and closed validation outcomes are the work product.
    payload["model_structured_result"] = None
    payload["parser_result"] = {}
    return payload


def _source_product_payload(result: SourceSemanticRuntimeResultV34) -> dict[str, Any]:
    payload = result.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    for item, assertion in zip(payload["assertions"], result.assertions, strict=True):
        item["value"] = assertion.value.model_dump(mode="json", exclude_unset=True)
    payload["anchors"] = []
    payload["source_evidence_spans"] = []
    extraction = payload.get("source_extraction")
    if isinstance(extraction, dict):
        extraction["evidence_spans"] = []
    return payload


def _restore_persisted_decimal_values(value: Any) -> Any:
    """Restore Decimal strings emitted by Pydantic JSON-mode work products.

    This is a typed-artifact serialization adapter, not source normalization:
    it only converts numeric fields inside the existing NumericValue shape.
    """

    if isinstance(value, list):
        return [_restore_persisted_decimal_values(item) for item in value]
    if not isinstance(value, dict):
        return value
    restored = {key: _restore_persisted_decimal_values(item) for key, item in value.items()}
    numeric_fields = {"value", "lower_bound", "upper_bound", "tolerance"}
    if "operator" in restored and numeric_fields.intersection(restored):
        for key in numeric_fields.intersection(restored):
            field_value = restored[key]
            if isinstance(field_value, str):
                try:
                    restored[key] = Decimal(field_value)
                except InvalidOperation:
                    continue
    return restored


def _alignment_pair_has_grounded_field(pair: IdentityAlignmentPair) -> bool:
    from app.domain.source_atomic_assertion import SourceAssertionField, SourceGroundingState

    by_field = {item.field: item for item in pair.source.field_grounding}
    return any(
        (item := by_field.get(field)) is not None
        and all(
            state is SourceGroundingState.VALID
            for state in (item.textual_grounding, item.semantic_support, item.structural_grounding)
        )
        for field in (SourceAssertionField.SUBJECT, SourceAssertionField.PREDICATE)
    )


def _unknown_alignment_result(
    pairs: tuple[IdentityAlignmentPair, ...],
) -> IdentityAlignmentRuntimeResult:
    payload = {
        "schema_version": "semantic-identity-alignment-response-v1",
        "status": "RESOLVED",
        "alignments": [
            {
                "pair_ref": deterministic_pair_ref(
                    pair.target_proposition_ref,
                    pair.source.assertion_ref,
                    "semantic-identity-alignment-response-v1",
                ),
                "subject_relation": SemanticAlignmentStatus.UNKNOWN.value,
                "predicate_relation": SemanticAlignmentStatus.UNKNOWN.value,
                "reason_code": "SOURCE_GROUNDING_UNAVAILABLE",
            }
            for pair in pairs
        ],
    }
    return IdentityAlignmentRuntimeResult(
        status=AlignmentRuntimeStatus.COMPLETED_WITH_UNKNOWN,
        facts=parse_alignment_response(payload, pairs),
        reason_codes=("SOURCE_GROUNDING_UNAVAILABLE",),
    )


def _page_receipt_marker(run_id: UUID, persisted: PagePersistenceResult) -> str:
    identity = [
        {
            "claim_ref": str(claim.claim_id),
            "claim_hash": claim.claim_content_hash,
            "evidence": [
                {
                    "evidence_ref": str(item.evidence_id),
                    "source_ref": str(item.source_id),
                    "snapshot_ref": str(item.snapshot_id),
                    "span_ref": item.canonical_span.span_ref if item.canonical_span else None,
                }
                for item in sorted(claim.evidence_receipts, key=lambda ref: str(ref.evidence_id))
            ],
        }
        for claim in sorted(persisted.claim_receipts, key=lambda ref: str(ref.claim_id))
    ]
    encoded = json.dumps(
        ["semantic-shadow-pair-integration-v1", identity],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"runs/{run_id}/semantic-shadow/receipts/{digest}.json"


def _derived_index_uri(run_id: UUID) -> str:
    return f"runs/{run_id}/semantic-shadow/work-products/index/derived-state-v1.json"


def _target_state_payload(state: ShadowTargetState) -> dict[str, Any]:
    payload = state.model_dump(mode="json")
    for item, assertion in zip(payload["assertions"], state.assertions, strict=True):
        item["typed_value"] = assertion.typed_value.model_dump(mode="json", exclude_unset=True)
    return payload


def _canonical_receipt_binding_is_valid(evidence: Any) -> bool:
    span = evidence.canonical_span
    return bool(
        span is not None
        and span.source_ref == str(evidence.source_id)
        and span.snapshot_ref
        == hashlib.sha256(evidence.snapshot_content_hash.encode("utf-8")).hexdigest()
    )


class _RunScopedGateway:
    """Add the ResearchRun reference so normal redacted call logging is retained."""

    def __init__(self, gateway: LLMGateway, run_id: UUID) -> None:
        self._gateway = gateway
        self._run_id = str(run_id)

    async def generate_structured(self, **kwargs: Any) -> Any:
        request = kwargs["request"]
        kwargs["request"] = request.model_copy(
            update={"metadata": {**request.metadata, "run_id": self._run_id}}
        )
        return await self._gateway.generate_structured(**kwargs)
