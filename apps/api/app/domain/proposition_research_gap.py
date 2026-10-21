"""Versioned, proposition-scoped research-gap contracts for V2.4 Step 1."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION = "proposition-research-gap-v1"
GAP_DERIVATION_DIAGNOSTIC_VERSION = "proposition-gap-diagnostic-v1"
GAP_SCOPE_POLICY_VERSION = "proposition-gap-scope-policy-v1"


class PropositionResearchGapType(StrEnum):
    MISSING_SUPPORT = "MISSING_SUPPORT"
    MISSING_INDEPENDENT_CORROBORATION = "MISSING_INDEPENDENT_CORROBORATION"
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
    SCOPE_COMPLETION = "SCOPE_COMPLETION"
    CONTEXT_DISAMBIGUATION = "CONTEXT_DISAMBIGUATION"
    SEMANTIC_CAPABILITY_BLOCKED = "SEMANTIC_CAPABILITY_BLOCKED"
    SEMANTIC_WORK_NOT_PROCESSED = "SEMANTIC_WORK_NOT_PROCESSED"


class GapSearchability(StrEnum):
    SEARCHABLE = "SEARCHABLE"
    NON_SEARCHABLE = "NON_SEARCHABLE"
    UNKNOWN_SEARCHABILITY = "UNKNOWN_SEARCHABILITY"


class PropositionGapLifecycle(StrEnum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"
    EXHAUSTED = "EXHAUSTED"
    SUPERSEDED = "SUPERSEDED"


class DesiredEvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DISAMBIGUATES = "DISAMBIGUATES"


class ScopeExpansionPolicy(StrEnum):
    ALLOWED = "ALLOWED"
    DISALLOWED = "DISALLOWED"
    UNKNOWN = "UNKNOWN"


class GapDiagnosticCode(StrEnum):
    UNSUPPORTED_VALUE_PROFILE = "UNSUPPORTED_VALUE_PROFILE"
    UNSUPPORTED_SEMANTIC_RELATION = "UNSUPPORTED_SEMANTIC_RELATION"
    FORMATION_CAPABILITY_LIMITATION = "FORMATION_CAPABILITY_LIMITATION"
    GROUNDING_CAPABILITY_LIMITATION = "GROUNDING_CAPABILITY_LIMITATION"
    IDENTITY_CAPABILITY_LIMITATION = "IDENTITY_CAPABILITY_LIMITATION"
    FORMATION_UNRESOLVED = "FORMATION_UNRESOLVED"
    GROUNDING_UNRESOLVED = "GROUNDING_UNRESOLVED"
    IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
    CONTEXT_NOT_COMPARABLE = "CONTEXT_NOT_COMPARABLE"
    NOT_PROCESSED_BUDGET = "NOT_PROCESSED_BUDGET"


class PropositionGapReasonCode(StrEnum):
    NO_ELIGIBLE_SUPPORT = "NO_ELIGIBLE_SUPPORT"
    OWNER_CORROBORATION_INSUFFICIENT = "OWNER_CORROBORATION_INSUFFICIENT"
    CONFLICT_SCOPE_NOT_COMPLETE = "CONFLICT_SCOPE_NOT_COMPLETE"
    CONFLICT_SCOPE_MISMATCH = "CONFLICT_SCOPE_MISMATCH"
    ACTIVE_CONFLICT_PRESENT = "ACTIVE_CONFLICT_PRESENT"
    NOT_COMPARABLE_CONTEXT = "NOT_COMPARABLE_CONTEXT"
    SEMANTIC_CAPABILITY_LIMITATION = "SEMANTIC_CAPABILITY_LIMITATION"
    SEMANTIC_WORK_BUDGET_NOT_PROCESSED = "SEMANTIC_WORK_BUDGET_NOT_PROCESSED"
    VERIFICATION_ASSESSMENT_UNKNOWN = "VERIFICATION_ASSESSMENT_UNKNOWN"
    CORROBORATION_ASSESSMENT_MISSING = "CORROBORATION_ASSESSMENT_MISSING"
    CONFLICT_ASSESSMENT_MISSING = "CONFLICT_ASSESSMENT_MISSING"
    UNKNOWN_REASON_CODE = "UNKNOWN_REASON_CODE"
    UNKNOWN_CONTRACT_VERSION = "UNKNOWN_CONTRACT_VERSION"
    INPUT_REFERENCE_MISMATCH = "INPUT_REFERENCE_MISMATCH"
    SCOPE_EXPANSION_DISALLOWED = "SCOPE_EXPANSION_DISALLOWED"
    SCOPE_EXPANSION_POLICY_UNKNOWN = "SCOPE_EXPANSION_POLICY_UNKNOWN"
    RESEARCH_BUDGET_EXHAUSTED = "RESEARCH_BUDGET_EXHAUSTED"
    RESEARCH_BUDGET_STATE_UNKNOWN = "RESEARCH_BUDGET_STATE_UNKNOWN"


class GapPriorityClass(StrEnum):
    ACTIVE_CONFLICT = "ACTIVE_CONFLICT"
    MISSING_SUPPORT = "MISSING_SUPPORT"
    INDEPENDENT_CORROBORATION = "INDEPENDENT_CORROBORATION"
    SCOPE_OR_CONTEXT = "SCOPE_OR_CONTEXT"
    BLOCKED_WORK = "BLOCKED_WORK"


class SourceCharacteristic(StrEnum):
    DIRECT_BASIS = "DIRECT_BASIS"
    ORIGINALITY_BASIS = "ORIGINALITY_BASIS"
    AUTHORITY_FIT_BASIS = "AUTHORITY_FIT_BASIS"


class GapDiagnostic(BaseModel):
    """Versioned normalized diagnostic; unknown codes are retained and fail closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GAP_DERIVATION_DIAGNOSTIC_VERSION
    code: str = Field(min_length=1, max_length=128)
    target_proposition_refs: tuple[UUID, ...] = ()
    target_group_refs: tuple[str, ...] = ()
    basis_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def canonical_refs(self) -> GapDiagnostic:
        if self.target_proposition_refs != tuple(
            sorted(set(self.target_proposition_refs), key=str)
        ):
            raise ValueError("gap_diagnostic_proposition_refs_must_be_canonical")
        if self.target_group_refs != tuple(sorted(set(self.target_group_refs))):
            raise ValueError("gap_diagnostic_group_refs_must_be_canonical")
        if self.basis_refs != tuple(sorted(set(self.basis_refs))):
            raise ValueError("gap_diagnostic_basis_refs_must_be_canonical")
        return self


class GapScopePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GAP_SCOPE_POLICY_VERSION
    expansion: ScopeExpansionPolicy = ScopeExpansionPolicy.UNKNOWN
    policy_ref: str | None = Field(default=None, min_length=1, max_length=256)


class GapPriorityInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    priority_class: GapPriorityClass
    parent_research_priority: int | None = Field(default=None, ge=1, le=3)
    attempt_count: int = Field(default=0, ge=0)

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        parent_priority = self.parent_research_priority or 4
        class_order = {
            GapPriorityClass.ACTIVE_CONFLICT: 0,
            GapPriorityClass.MISSING_SUPPORT: 1,
            GapPriorityClass.INDEPENDENT_CORROBORATION: 2,
            GapPriorityClass.SCOPE_OR_CONTEXT: 3,
            GapPriorityClass.BLOCKED_WORK: 4,
        }[self.priority_class]
        return class_order, parent_priority, self.attempt_count


class RequiredEvidenceCharacteristics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_refs: tuple[UUID, ...] = ()
    desired_relations: tuple[DesiredEvidenceRelation, ...] = ()
    required_owner_independence: bool = False
    excluded_owner_keys: tuple[str, ...] = ()
    required_context_refs: tuple[str, ...] = ()
    required_time_version_scope_condition_refs: tuple[str, ...] = ()
    a2_conflict_pair_refs: tuple[str, ...] = ()
    support_side_evidence_refs: tuple[str, ...] = ()
    contradict_side_evidence_refs: tuple[str, ...] = ()
    conflict_group_refs: tuple[str, ...] = ()
    source_characteristics: tuple[SourceCharacteristic, ...] = ()

    @model_validator(mode="after")
    def canonical_characteristics(self) -> RequiredEvidenceCharacteristics:
        for name in (
            "target_proposition_refs",
            "desired_relations",
            "excluded_owner_keys",
            "required_context_refs",
            "required_time_version_scope_condition_refs",
            "a2_conflict_pair_refs",
            "support_side_evidence_refs",
            "contradict_side_evidence_refs",
            "conflict_group_refs",
            "source_characteristics",
        ):
            value = getattr(self, name)
            key = (
                str
                if name == "target_proposition_refs"
                else (lambda item: item.value if isinstance(item, StrEnum) else item)
            )
            if value != tuple(sorted(set(value), key=key)):
                raise ValueError(f"{name}_must_be_unique_and_canonical")
        if self.required_owner_independence and not self.excluded_owner_keys:
            raise ValueError("independent_evidence_requires_known_excluded_owner_keys")
        return self


class PropositionResearchGap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_ref: UUID
    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_refs: tuple[UUID, ...] = Field(min_length=1)
    parent_claim_refs: tuple[UUID, ...] = ()
    objective_scope_ref: str = Field(min_length=1, max_length=256)
    verification_assessment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gap_type: PropositionResearchGapType
    searchability: GapSearchability
    reason_codes: tuple[PropositionGapReasonCode, ...] = Field(min_length=1)
    required_evidence: RequiredEvidenceCharacteristics
    existing_evidence_refs: tuple[str, ...] = ()
    existing_owner_refs: tuple[str, ...] = ()
    existing_conflict_refs: tuple[str, ...] = ()
    source_reliability_refs: tuple[str, ...] = ()
    source_reliability_reason_codes: tuple[str, ...] = ()
    budget_state_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    budget_state: dict[str, int] | None = None
    lifecycle_state: PropositionGapLifecycle
    attempt_count: int = Field(default=0, ge=0)
    priority_inputs: GapPriorityInputs
    schema_version: str = PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION

    @model_validator(mode="after")
    def canonical_gap(self) -> PropositionResearchGap:
        if self.target_proposition_refs != tuple(
            sorted(set(self.target_proposition_refs), key=str)
        ):
            raise ValueError("gap_target_propositions_must_be_canonical")
        if self.parent_claim_refs != tuple(sorted(set(self.parent_claim_refs), key=str)):
            raise ValueError("gap_parent_claims_must_be_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("gap_reason_codes_must_be_canonical")
        for name in (
            "existing_evidence_refs",
            "existing_owner_refs",
            "existing_conflict_refs",
            "source_reliability_refs",
            "source_reliability_reason_codes",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name}_must_be_unique_and_canonical")
        if self.required_evidence.target_group_ref != self.target_group_ref:
            raise ValueError("gap_required_evidence_group_mismatch")
        if self.required_evidence.target_proposition_refs != self.target_proposition_refs:
            raise ValueError("gap_required_evidence_propositions_mismatch")
        expected_lifecycle = (
            PropositionGapLifecycle.OPEN
            if self.searchability is GapSearchability.SEARCHABLE
            else PropositionGapLifecycle.BLOCKED
        )
        if self.lifecycle_state is not expected_lifecycle:
            raise ValueError("gap_lifecycle_must_be_deterministic_projection")
        if (
            self.gap_type is PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED
            and self.searchability is not GapSearchability.NON_SEARCHABLE
        ):
            raise ValueError("semantic_capability_gap_must_be_non_searchable")
        if (
            self.gap_type is PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED
            and self.searchability is not GapSearchability.NON_SEARCHABLE
        ):
            raise ValueError("unprocessed_work_gap_must_be_non_searchable")
        if self.searchability is GapSearchability.UNKNOWN_SEARCHABILITY and (
            self.lifecycle_state is not PropositionGapLifecycle.BLOCKED
        ):
            raise ValueError("unknown_searchability_must_fail_closed")
        if self.attempt_count != self.priority_inputs.attempt_count:
            raise ValueError("gap_attempt_count_priority_mismatch")
        return self

    @property
    def priority_ordering_key(self) -> tuple[int, int, int, int, str]:
        searchable_rank = 0 if self.searchability is GapSearchability.SEARCHABLE else 1
        return (searchable_rank, *self.priority_inputs.ordering_key, self.gap_ref)


class PropositionGapDerivationIssue(BaseModel):
    """Per-target fail-closed diagnostic that must not block other targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: UUID
    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_refs: tuple[UUID, ...] = ()
    assessment_ref: str | None = None
    reason_code: PropositionGapReasonCode
    unresolved_reason: str = Field(min_length=1, max_length=256)
    searchability: GapSearchability = GapSearchability.UNKNOWN_SEARCHABILITY
    lifecycle_state: PropositionGapLifecycle = PropositionGapLifecycle.BLOCKED
    schema_version: str = PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION

    @model_validator(mode="after")
    def must_fail_closed(self) -> PropositionGapDerivationIssue:
        if self.searchability is not GapSearchability.UNKNOWN_SEARCHABILITY:
            raise ValueError("derivation_issue_must_have_unknown_searchability")
        if self.lifecycle_state is not PropositionGapLifecycle.BLOCKED:
            raise ValueError("derivation_issue_must_be_blocked")
        return self
