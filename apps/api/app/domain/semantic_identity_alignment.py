"""Comparison-local semantic identity alignment facts for the V2.3-D shadow bridge."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.verification_assertion import (
    SemanticIdentity,
    SemanticIdentityBasis,
    SemanticIdentityStatus,
)

SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA_VERSION = "semantic-identity-alignment-v1"
SEMANTIC_IDENTITY_ALIGNMENT_RUNTIME_VERSION = "semantic-identity-alignment-runtime-v1"


class SemanticIdentityFieldRole(StrEnum):
    SUBJECT = "SUBJECT"
    PREDICATE = "PREDICATE"


class SemanticAlignmentStatus(StrEnum):
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"


class SemanticIdentityAlignmentReasonCode(StrEnum):
    IDENTITY_ALIGNED = "IDENTITY_ALIGNED"
    IDENTITY_DISTINCT = "IDENTITY_DISTINCT"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    SOURCE_GROUNDING_UNAVAILABLE = "SOURCE_GROUNDING_UNAVAILABLE"


class SemanticIdentityAlignmentFact(BaseModel):
    """Auditable pair-level relation; it does not assert a global ontology identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alignment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_ref: str = Field(min_length=1, max_length=128)
    source_assertion_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    field_role: SemanticIdentityFieldRole
    alignment_status: SemanticAlignmentStatus
    reason_code: SemanticIdentityAlignmentReasonCode
    target_semantic_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_semantic_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_grounding_refs: tuple[str, ...] = ()
    basis_refs: tuple[str, ...] = ()
    schema_version: str = SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA_VERSION
    runtime_version: str = SEMANTIC_IDENTITY_ALIGNMENT_RUNTIME_VERSION

    @model_validator(mode="after")
    def validate_reason_and_grounding(self) -> SemanticIdentityAlignmentFact:
        expected_reason = {
            SemanticAlignmentStatus.SAME: SemanticIdentityAlignmentReasonCode.IDENTITY_ALIGNED,
            SemanticAlignmentStatus.DIFFERENT: (
                SemanticIdentityAlignmentReasonCode.IDENTITY_DISTINCT
            ),
        }.get(self.alignment_status)
        if expected_reason is not None:
            if self.reason_code is not expected_reason or not self.source_grounding_refs:
                raise ValueError("affirmative_alignment_requires_grounding_and_matching_reason")
        elif self.reason_code in {
            SemanticIdentityAlignmentReasonCode.IDENTITY_ALIGNED,
            SemanticIdentityAlignmentReasonCode.IDENTITY_DISTINCT,
        }:
            raise ValueError("unknown_alignment_cannot_claim_identity_result")
        return self


def project_pair_identities(
    fact: SemanticIdentityAlignmentFact,
) -> tuple[SemanticIdentity, SemanticIdentity]:
    """Create deterministic identities scoped to this one validated comparison."""

    if fact.alignment_status is SemanticAlignmentStatus.UNKNOWN:
        return SemanticIdentity.unresolved(), SemanticIdentity.unresolved()
    namespace = f"comparison:{fact.alignment_ref}:{fact.field_role.value.lower()}"
    common = {
        "status": SemanticIdentityStatus.RESOLVED,
        "namespace": namespace,
        "normalization_version": fact.schema_version,
        "basis": SemanticIdentityBasis.UPSTREAM_RESOLVER,
        "provenance_ref": fact.alignment_ref,
    }
    target_key = f"{fact.alignment_ref}:{fact.field_role.value}:TARGET"
    if fact.alignment_status is SemanticAlignmentStatus.SAME:
        shared = {**common, "key": f"{fact.alignment_ref}:{fact.field_role.value}:SAME"}
        identity = SemanticIdentity.model_validate(shared)
        return identity, identity
    source_key = f"{fact.alignment_ref}:{fact.field_role.value}:SOURCE"
    return (
        SemanticIdentity.model_validate({**common, "key": target_key}),
        SemanticIdentity.model_validate({**common, "key": source_key}),
    )
