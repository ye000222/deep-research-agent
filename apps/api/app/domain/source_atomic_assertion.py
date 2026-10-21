"""Versioned, non-persistent source-side assertion bridge contracts.

These types are deliberately distinct from target AtomicProposition and the
frozen v3.3 source-grounding response.  They retain only typed semantics and
provenance references, never document or quote payloads.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.assertion_composition import AssertionGroupOperator
from app.domain.atomic_proposition import PropositionPolarity, PropositionValue
from app.domain.source_context_extraction_v3_3 import SourceExtractionResponseV33
from app.domain.truth_condition_context import TruthConditionContext
from app.domain.verification_assertion import SemanticIdentity

SOURCE_ATOMIC_ASSERTION_SCHEMA_VERSION: Literal["source-atomic-assertion-v1"] = (
    "source-atomic-assertion-v1"
)
SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION: Literal[
    "source-semantic-runtime-response-v1"
] = "source-semantic-runtime-response-v1"


class SourceAssertionEligibility(StrEnum):
    INDEPENDENTLY_ASSERTED = "INDEPENDENTLY_ASSERTED"
    NOT_INDEPENDENTLY_ASSERTED = "NOT_INDEPENDENTLY_ASSERTED"
    UNKNOWN = "UNKNOWN"


class SourceAssertionField(StrEnum):
    SUBJECT = "SUBJECT"
    PREDICATE = "PREDICATE"
    VALUE = "VALUE"
    POLARITY = "POLARITY"
    CONTEXT = "CONTEXT"
    ATTRIBUTION = "ATTRIBUTION"
    MODALITY = "MODALITY"


class SourceGroundingState(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class SourceFieldGrounding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: SourceAssertionField
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    textual_grounding: SourceGroundingState
    semantic_support: SourceGroundingState
    structural_grounding: SourceGroundingState
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def canonical_evidence_refs(self) -> SourceFieldGrounding:
        if tuple(sorted(set(self.evidence_refs))) != self.evidence_refs:
            raise ValueError("field_grounding_evidence_refs_must_be_unique_and_sorted")
        return self


class SourceSemanticTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_representation: str = Field(min_length=1, max_length=1_000)
    semantic_identity: SemanticIdentity = Field(default_factory=SemanticIdentity.unresolved)


class SourceAssertionCompositionMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    position: int = Field(ge=0)
    assertion_ref: str | None = Field(default=None, min_length=1, max_length=64)
    group_ref: str | None = Field(default=None, min_length=1, max_length=64)
    eligibility: SourceAssertionEligibility | None = None

    @model_validator(mode="after")
    def reference_shape(self) -> SourceAssertionCompositionMember:
        if (self.assertion_ref is None) == (self.group_ref is None):
            raise ValueError("composition_member_requires_exactly_one_reference")
        if self.assertion_ref is not None and self.eligibility is None:
            raise ValueError("assertion_leaf_requires_eligibility")
        if self.group_ref is not None and self.eligibility is not None:
            raise ValueError("composition_group_member_has_no_leaf_eligibility")
        return self


class SourceAssertionCompositionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(min_length=1, max_length=64)
    operator: AssertionGroupOperator
    members: tuple[SourceAssertionCompositionMember, ...] = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def canonical_members(self) -> SourceAssertionCompositionGroup:
        if tuple(item.position for item in self.members) != tuple(range(len(self.members))):
            raise ValueError("composition_positions_must_be_contiguous")
        return self


class SourceAssertionComposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source-assertion-composition-v1"] = (
        "source-assertion-composition-v1"
    )
    root_members: tuple[SourceAssertionCompositionMember, ...] = Field(min_length=1, max_length=32)
    groups: tuple[SourceAssertionCompositionGroup, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def canonical_roots(self) -> SourceAssertionComposition:
        if tuple(item.position for item in self.root_members) != tuple(
            range(len(self.root_members))
        ):
            raise ValueError("composition_root_positions_must_be_contiguous")
        refs = {group.group_ref for group in self.groups}
        if len(refs) != len(self.groups):
            raise ValueError("composition_group_refs_must_be_unique")
        for group in self.groups:
            for member in group.members:
                if member.group_ref is not None and member.group_ref not in refs:
                    raise ValueError("composition_group_reference_missing")
        for member in self.root_members:
            if member.group_ref is not None and member.group_ref not in refs:
                raise ValueError("composition_root_group_reference_missing")
        return self


class SourceAtomicAssertion(BaseModel):
    """A source-authored typed assertion, not a target proposition or DB row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assertion_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: Literal["source-atomic-assertion-v1"] = (
        SOURCE_ATOMIC_ASSERTION_SCHEMA_VERSION
    )
    source_ref: str = Field(min_length=1, max_length=2_000)
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_evidence_span_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_assertion_anchor_refs: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_span_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    source_evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    subject: SourceSemanticTerm
    predicate: SourceSemanticTerm
    value: PropositionValue
    polarity: PropositionPolarity
    truth_condition_context_refs: tuple[str, ...] = Field(default=(), max_length=64)
    attribution_context_refs: tuple[str, ...] = Field(default=(), max_length=16)
    modality_context_refs: tuple[str, ...] = Field(default=(), max_length=16)
    composition_ref: str | None = Field(default=None, min_length=1, max_length=64)
    verification_eligibility: SourceAssertionEligibility = SourceAssertionEligibility.UNKNOWN
    field_grounding: tuple[SourceFieldGrounding, ...] = Field(min_length=4, max_length=7)

    @model_validator(mode="after")
    def validate_refs_and_grounding(self) -> SourceAtomicAssertion:
        if tuple(sorted(set(self.evidence_span_refs))) != self.evidence_span_refs:
            raise ValueError("evidence_span_refs_must_be_unique_and_sorted")
        if tuple(sorted(set(self.source_evidence_refs))) != self.source_evidence_refs:
            raise ValueError("source_evidence_refs_must_be_unique_and_sorted")
        required_fields = {
            SourceAssertionField.SUBJECT,
            SourceAssertionField.PREDICATE,
            SourceAssertionField.VALUE,
            SourceAssertionField.POLARITY,
        }
        fields = tuple(item.field for item in self.field_grounding)
        if len(set(fields)) != len(fields):
            raise ValueError("field_grounding_fields_must_be_unique")
        if tuple(sorted(fields, key=lambda item: item.value)) != fields:
            raise ValueError("field_grounding_must_be_canonically_ordered")
        if not required_fields.issubset(fields):
            raise ValueError("source_assertion_requires_core_field_grounding_refs")
        source_refs = set(self.source_evidence_refs)
        if any(not set(item.evidence_refs).issubset(source_refs) for item in self.field_grounding):
            raise ValueError("field_grounding_refs_must_belong_to_source_assertion")
        return self


class SourceAtomicAssertionCandidateV1(BaseModel):
    """Provider-facing candidate parsed before deterministic ref construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    local_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    subject: SourceSemanticTerm
    predicate: SourceSemanticTerm
    value: PropositionValue
    polarity: PropositionPolarity
    truth_condition_context_refs: tuple[str, ...] = Field(default=(), max_length=64)
    field_evidence_refs: dict[SourceAssertionField, tuple[str, ...]]
    verification_eligibility: SourceAssertionEligibility = SourceAssertionEligibility.UNKNOWN
    composition_ref: str | None = Field(default=None, max_length=64)


class SourceSemanticRuntimeResponseV1(BaseModel):
    """Versioned envelope; the nested v3.3 object is validated as v3.3 unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source-semantic-runtime-response-v1"] = (
        SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION
    )
    status: Literal["RESOLVED", "UNRESOLVED"]
    source_extraction: SourceExtractionResponseV33
    assertion_candidates: tuple[SourceAtomicAssertionCandidateV1, ...] = Field(
        default=(), max_length=64
    )
    truth_condition_contexts: tuple[TruthConditionContext, ...] = Field(default=(), max_length=128)
    composition: SourceAssertionComposition | None = None
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceSemanticRuntimeResponseV1:
        if self.status == "RESOLVED" and not self.assertion_candidates:
            raise ValueError("resolved_runtime_response_requires_assertion_candidate")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_runtime_response_requires_reason")
        return self
