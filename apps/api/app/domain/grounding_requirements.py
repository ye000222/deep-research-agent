"""Ephemeral contracts for shadow grounding obligations and node usability."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.source_context_grounding_v2 import GroundingDimensionStatus

GROUNDING_REQUIREMENT_CONTRACT_VERSION = "grounding-requirements.generic.v1"


class GroundingNodeKind(StrEnum):
    ASSERTION = "ASSERTION"
    CONTEXT = "CONTEXT"
    ATTACHMENT = "ATTACHMENT"
    SCOPE = "SCOPE"
    GROUP = "GROUP"
    METADATA = "METADATA"


class GroundingRequirementLevel(StrEnum):
    REQUIRED_FOR_NODE_SEMANTICS = "REQUIRED_FOR_NODE_SEMANTICS"
    OPTIONAL_CORROBORATION = "OPTIONAL_CORROBORATION"
    STRUCTURAL_SUPPORT_ONLY = "STRUCTURAL_SUPPORT_ONLY"


class GroundingCardinalityPolicy(StrEnum):
    EXACTLY_ONE = "EXACTLY_ONE"
    AT_LEAST_ONE = "AT_LEAST_ONE"
    ALL_EXPECTED_MEMBERS = "ALL_EXPECTED_MEMBERS"
    PER_TARGET = "PER_TARGET"
    PER_VALUE = "PER_VALUE"


class GroundingObligationType(StrEnum):
    ASSERTION_TEXT_EVIDENCE = "ASSERTION_TEXT_EVIDENCE"
    ASSERTION_CONTENT_SUPPORT = "ASSERTION_CONTENT_SUPPORT"
    ASSERTION_GRAPH_MEMBERSHIP = "ASSERTION_GRAPH_MEMBERSHIP"
    CONTEXT_VALUE = "CONTEXT_VALUE"
    CONDITION_SEMANTICS = "CONDITION_SEMANTICS"
    SCOPE_VALUE = "SCOPE_VALUE"
    SCOPE_MEMBERSHIP = "SCOPE_MEMBERSHIP"
    TARGET_APPLICABILITY = "TARGET_APPLICABILITY"
    CONFIGURATION_VALUE = "CONFIGURATION_VALUE"
    MEASUREMENT_VALUE = "MEASUREMENT_VALUE"
    METRIC_IDENTITY = "METRIC_IDENTITY"
    CATEGORY_IDENTITY = "CATEGORY_IDENTITY"
    METHOD_VALUE = "METHOD_VALUE"
    METHOD_RELATION = "METHOD_RELATION"
    EPISTEMIC_OPERATOR = "EPISTEMIC_OPERATOR"
    EPISTEMIC_AGENT = "EPISTEMIC_AGENT"
    EPISTEMIC_TARGET_RELATIONSHIP = "EPISTEMIC_TARGET_RELATIONSHIP"
    MODALITY = "MODALITY"
    NUMERIC_ASSOCIATION = "NUMERIC_ASSOCIATION"
    ENUMERATION_COUNT = "ENUMERATION_COUNT"
    ENUMERATION_MEMBER = "ENUMERATION_MEMBER"
    STRUCTURAL_REFERENCE = "STRUCTURAL_REFERENCE"
    SHARED_CONTEXT_MEMBERSHIP = "SHARED_CONTEXT_MEMBERSHIP"


class GroundingDerivationStatus(StrEnum):
    RESOLVED = "RESOLVED"
    DERIVATION_UNRESOLVED = "DERIVATION_UNRESOLVED"
    UNSUPPORTED_NODE_SHAPE = "UNSUPPORTED_NODE_SHAPE"
    CONTRACT_ERROR = "CONTRACT_ERROR"


class ZeroObligationReason(StrEnum):
    STRUCTURAL_ONLY_NODE = "STRUCTURAL_ONLY_NODE"
    NO_SEMANTIC_GROUNDING_REQUIRED = "NO_SEMANTIC_GROUNDING_REQUIRED"
    DERIVATION_UNRESOLVED = "DERIVATION_UNRESOLVED"


class GroundingObligationStatus(StrEnum):
    SATISFIED = "SATISFIED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MISSING_REQUIRED_BINDING = "MISSING_REQUIRED_BINDING"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"
    TEXTUALLY_INVALID = "TEXTUALLY_INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DERIVATION_UNRESOLVED = "DERIVATION_UNRESOLVED"


class GroundingNodeUsability(StrEnum):
    GROUNDED_USABLE = "GROUNDED_USABLE"
    PARTIALLY_GROUNDED_UNRESOLVED = "PARTIALLY_GROUNDED_UNRESOLVED"
    UNGROUNDED_UNUSABLE = "UNGROUNDED_UNUSABLE"
    SEMANTIC_SUPPORT_AMBIGUOUS = "SEMANTIC_SUPPORT_AMBIGUOUS"
    NOT_EVALUATED = "NOT_EVALUATED"
    STRUCTURALLY_GROUNDED = "STRUCTURALLY_GROUNDED"
    NO_SEMANTIC_GROUNDING_REQUIRED = "NO_SEMANTIC_GROUNDING_REQUIRED"


class GroundingCandidateExclusionReason(StrEnum):
    ROLE_MISMATCH = "ROLE_MISMATCH"
    TARGET_KIND_MISMATCH = "TARGET_KIND_MISMATCH"
    TARGET_REF_MISMATCH = "TARGET_REF_MISMATCH"
    PAYLOAD_MISMATCH = "PAYLOAD_MISMATCH"
    RELATION_MISMATCH = "RELATION_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    MEMBER_MISMATCH = "MEMBER_MISMATCH"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    BINDING_ALREADY_CONSUMED = "BINDING_ALREADY_CONSUMED"


class GroundingRequirementContract(BaseModel):
    """Versioned descriptor for a pure, deterministic contract registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str = GROUNDING_REQUIREMENT_CONTRACT_VERSION
    supported_node_kinds: tuple[GroundingNodeKind, ...] = tuple(GroundingNodeKind)
    derivation_is_deterministic: bool = True
    uses_human_labels: bool = False
    uses_support_outcomes: bool = False
    uses_provider_identity: bool = False


class GroundingNodeInput(BaseModel):
    """Semantic-only input to obligation derivation; contains no verdicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_ref: str = Field(min_length=1, max_length=128)
    node_kind: GroundingNodeKind
    semantic_payload: dict[str, Any] = Field(default_factory=dict)
    graph_role: str = Field(min_length=1, max_length=100)


class GroundingTargetConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_kind: str | None = None
    target_ref: str | None = None
    relation: str | None = None
    scope_ref: str | None = None
    semantic_key: str | None = None
    expected_value: Any | None = None


class GroundingObligation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_ref: str = Field(min_length=1, max_length=80)
    node_ref: str = Field(min_length=1, max_length=128)
    node_kind: GroundingNodeKind
    obligation_type: GroundingObligationType
    expected_semantic_role_family: tuple[str, ...] = Field(min_length=1)
    requirement_level: GroundingRequirementLevel
    target_constraint: GroundingTargetConstraint
    cardinality_policy: GroundingCardinalityPolicy
    expected_member_key: str | None = None
    expected_target_key: str | None = None
    expected_value_key: str | None = None
    contract_version: str = GROUNDING_REQUIREMENT_CONTRACT_VERSION
    derivation_rule_id: str = Field(min_length=1, max_length=128)
    diagnostics: tuple[str, ...] = ()
    allow_binding_reuse: bool = False
    binding_reuse_group: str | None = None

    @model_validator(mode="after")
    def reuse_is_explicit(self) -> GroundingObligation:
        if self.allow_binding_reuse and not self.binding_reuse_group:
            raise ValueError("binding_reuse_requires_explicit_group")
        if (
            self.cardinality_policy is GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS
            and self.expected_member_key is None
        ):
            raise ValueError("all_expected_members_requires_member_key")
        if (
            self.cardinality_policy is GroundingCardinalityPolicy.PER_TARGET
            and self.expected_target_key is None
        ):
            raise ValueError("per_target_requires_target_key")
        if (
            self.cardinality_policy is GroundingCardinalityPolicy.PER_VALUE
            and self.expected_value_key is None
        ):
            raise ValueError("per_value_requires_value_key")
        return self


class GroundingObligationDerivationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_ref: str
    node_kind: GroundingNodeKind
    status: GroundingDerivationStatus
    obligations: tuple[GroundingObligation, ...] = ()
    zero_obligation_reason: ZeroObligationReason | None = None
    contract_version: str = GROUNDING_REQUIREMENT_CONTRACT_VERSION
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def state_is_explicit(self) -> GroundingObligationDerivationResult:
        if (
            self.status is GroundingDerivationStatus.RESOLVED
            and not self.obligations
            and self.zero_obligation_reason is None
        ):
            raise ValueError("resolved_zero_obligations_requires_explicit_reason")
        if self.obligations and self.zero_obligation_reason is not None:
            raise ValueError("obligations_and_zero_reason_are_mutually_exclusive")
        if self.status is not GroundingDerivationStatus.RESOLVED and self.obligations:
            raise ValueError("failed_derivation_must_not_emit_obligations")
        return self


class GroundingBindingCandidate(BaseModel):
    """Ephemeral projection of one existing binding and its independent results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str = Field(min_length=1, max_length=128)
    evidence_ref: str = Field(min_length=1, max_length=128)
    semantic_role: str = Field(min_length=1, max_length=100)
    target_kind: str = Field(min_length=1, max_length=100)
    target_ref: str = Field(min_length=1, max_length=128)
    canonical_payload: dict[str, Any] = Field(default_factory=dict)
    relation: str | None = None
    scope_ref: str | None = None
    member_key: str | None = None
    value_key: str | None = None
    textual_grounding: GroundingDimensionStatus
    structural_grounding: GroundingDimensionStatus
    semantic_support: GroundingDimensionStatus


class GroundingCandidateExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str
    reason: GroundingCandidateExclusionReason


class GroundingObligationMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_ref: str
    candidate_binding_refs: tuple[str, ...] = ()
    matched_binding_refs: tuple[str, ...] = ()
    excluded_candidates: tuple[GroundingCandidateExclusion, ...] = ()
    duplicate_evidence_refs: tuple[str, ...] = ()
    cardinality_satisfied: bool = False
    diagnostics: tuple[str, ...] = ()


class GroundingRequirementDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    node_ref: str
    obligation_ref: str
    expected_role_family: tuple[str, ...]
    target_constraint: GroundingTargetConstraint
    contract_version: str = GROUNDING_REQUIREMENT_CONTRACT_VERSION
    reason: str


class GroundingObligationEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation: GroundingObligation
    match: GroundingObligationMatch
    status: GroundingObligationStatus
    binding_statuses: tuple[tuple[str, str, str, str], ...] = ()
    diagnostics: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)


class NodeGroundingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_ref: str
    node_kind: GroundingNodeKind
    derivation: GroundingObligationDerivationResult
    obligation_evaluations: tuple[GroundingObligationEvaluation, ...] = ()
    usability: GroundingNodeUsability
    verification_assertion_ready: bool = False
    aggregation_precedence: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    extraction_omissions: tuple[GroundingRequirementDiagnostics, ...] = ()
