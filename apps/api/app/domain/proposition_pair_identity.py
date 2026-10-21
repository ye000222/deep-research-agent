"""Versioned, pair-local target identity facts; never proposition equivalence."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.assertion_composition import AssertionComposition
from app.domain.atomic_proposition import AtomicProposition
from app.domain.verification_assertion import (
    AssertionGroundingUsability,
    ComparisonEligibility,
    SemanticIdentity,
    SemanticIdentityBasis,
    VerificationAssertion,
)

PAIR_IDENTITY_SCHEMA: Literal["proposition-pair-identity-v1"] = "proposition-pair-identity-v1"
PAIR_IDENTITY_RUNTIME: Literal["proposition-pair-identity-runtime-v1"] = (
    "proposition-pair-identity-runtime-v1"
)
PAIR_IDENTITY_PROMPT: Literal["proposition-pair-identity.generic.v1"] = (
    "proposition-pair-identity.generic.v1"
)
PAIR_KIND: Literal["PROPOSITION_PROPOSITION"] = "PROPOSITION_PROPOSITION"


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class PairFieldRole(StrEnum):
    SUBJECT = "SUBJECT"
    PREDICATE = "PREDICATE"


class PairIdentityStatus(StrEnum):
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"


class PairIdentityReason(StrEnum):
    IDENTITY_ALIGNED = "IDENTITY_ALIGNED"
    IDENTITY_DISTINCT = "IDENTITY_DISTINCT"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class PropositionPairTarget(BaseModel):
    """Existing target and formation projection, with no external evidence input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition: AtomicProposition
    assertion: VerificationAssertion
    claim_refs: tuple[UUID, ...] = Field(min_length=1)
    formation_ref: str
    formation_status: str
    composition: AssertionComposition | None = None

    @model_validator(mode="after")
    def consistent_refs(self) -> PropositionPairTarget:
        if self.proposition.proposition_id != self.assertion.source_proposition_id:
            raise ValueError("PAIR_TARGET_REF_MISMATCH")
        if self.assertion.assertion_ref != str(self.proposition.proposition_id):
            raise ValueError("PAIR_TARGET_ASSERTION_REF_MISMATCH")
        if (
            self.assertion.subject != self.proposition.subject
            or self.assertion.predicate != self.proposition.predicate
            or self.assertion.typed_value != self.proposition.value
            or self.assertion.polarity != self.proposition.polarity
            or self.assertion.material_qualifiers != self.proposition.qualifiers
            or self.assertion.modality != self.proposition.modality
            or self.assertion.attribution != self.proposition.attribution
        ):
            raise ValueError("PAIR_TARGET_PROJECTION_MISMATCH")
        return self

    @property
    def eligible(self) -> bool:
        return (
            self.formation_status == "FORMATION_SUFFICIENT"
            and bool(self.formation_ref)
            and self.assertion.grounding_usability is AssertionGroundingUsability.SUFFICIENT
            and self.assertion.composition_eligibility is ComparisonEligibility.ELIGIBLE
        )

    def provider_payload(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition.model_dump(mode="json"),
            "formation_ref": self.formation_ref,
            "formation_status": self.formation_status,
            "material_contexts": [
                item.model_dump(mode="json") for item in self.assertion.material_contexts
            ],
            "composition_eligibility": self.assertion.composition_eligibility.value,
            "composition": self.composition.model_dump(mode="json") if self.composition else None,
        }


class PropositionPairBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: UUID
    pair_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_kind: Literal["PROPOSITION_PROPOSITION"] = PAIR_KIND
    left_proposition_ref: UUID
    right_proposition_ref: UUID
    left_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_refs: tuple[str, ...]
    schema_version: Literal["proposition-pair-identity-v1"] = PAIR_IDENTITY_SCHEMA
    runtime_version: Literal["proposition-pair-identity-runtime-v1"] = PAIR_IDENTITY_RUNTIME
    prompt_version: Literal["proposition-pair-identity.generic.v1"] = PAIR_IDENTITY_PROMPT


class PropositionPairInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding: PropositionPairBinding
    left: PropositionPairTarget
    right: PropositionPairTarget

    @model_validator(mode="after")
    def validate_binding(self) -> PropositionPairInput:
        if self.binding != bind_pair(self.binding.run_ref, self.left, self.right):
            raise ValueError("PAIR_INPUT_BINDING_MISMATCH")
        return self

    def provider_payload(self) -> dict[str, Any]:
        return {
            **self.binding.model_dump(mode="json"),
            "left": self.left.provider_payload(),
            "right": self.right.provider_payload(),
        }


def bind_pair(
    run_ref: UUID, left: PropositionPairTarget, right: PropositionPairTarget
) -> PropositionPairBinding:
    left_ref, right_ref = left.proposition.proposition_id, right.proposition.proposition_id
    if str(left_ref) >= str(right_ref):
        raise ValueError("PAIR_REFS_MUST_BE_DISTINCT_AND_CANONICAL")
    if not left.eligible or not right.eligible:
        raise ValueError("PAIR_FORMATION_OR_ELIGIBILITY_UNAVAILABLE")
    left_hash, right_hash = (
        content_hash(left.provider_payload()),
        content_hash(right.provider_payload()),
    )
    basis_refs = tuple(
        sorted(
            {
                left.formation_ref,
                right.formation_ref,
                content_hash(
                    [item.model_dump(mode="json") for item in left.assertion.material_contexts]
                ),
                content_hash(
                    [item.model_dump(mode="json") for item in right.assertion.material_contexts]
                ),
                content_hash(
                    left.composition.model_dump(mode="json") if left.composition else None
                ),
                content_hash(
                    right.composition.model_dump(mode="json") if right.composition else None
                ),
            }
        )
    )
    return PropositionPairBinding(
        run_ref=run_ref,
        pair_ref=content_hash(
            [
                PAIR_KIND,
                str(run_ref),
                str(left_ref),
                str(right_ref),
                left_hash,
                right_hash,
                PAIR_IDENTITY_SCHEMA,
                PAIR_IDENTITY_RUNTIME,
                PAIR_IDENTITY_PROMPT,
            ]
        ),
        left_proposition_ref=left_ref,
        right_proposition_ref=right_ref,
        left_input_hash=left_hash,
        right_input_hash=right_hash,
        basis_refs=basis_refs,
    )


class PropositionPairSemanticAlignmentFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alignment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding: PropositionPairBinding
    field_role: PairFieldRole
    alignment_status: PairIdentityStatus
    reason_code: PairIdentityReason

    @model_validator(mode="after")
    def validate_fact(self) -> PropositionPairSemanticAlignmentFact:
        allowed = {
            PairIdentityStatus.SAME: {PairIdentityReason.IDENTITY_ALIGNED},
            PairIdentityStatus.DIFFERENT: {PairIdentityReason.IDENTITY_DISTINCT},
            PairIdentityStatus.UNKNOWN: {
                PairIdentityReason.AMBIGUOUS,
                PairIdentityReason.INSUFFICIENT_CONTEXT,
            },
        }
        if self.reason_code not in allowed[self.alignment_status]:
            raise ValueError("PAIR_REASON_STATUS_MISMATCH")
        if self.alignment_ref != content_hash(
            [
                self.binding.model_dump(mode="json"),
                self.field_role.value,
                self.alignment_status.value,
                self.reason_code.value,
            ]
        ):
            raise ValueError("PAIR_ALIGNMENT_HASH_MISMATCH")
        return self


def project_pair_views(
    pair: PropositionPairInput, facts: tuple[PropositionPairSemanticAlignmentFact, ...]
) -> tuple[VerificationAssertion, VerificationAssertion]:
    if len(facts) != 2 or {item.field_role for item in facts} != set(PairFieldRole):
        raise ValueError("PAIR_FIELD_SET_MISMATCH")
    if any(item.binding != pair.binding for item in facts):
        raise ValueError("PAIR_FACT_BINDING_MISMATCH")
    left_updates: dict[str, SemanticIdentity] = {}
    right_updates: dict[str, SemanticIdentity] = {}
    for fact in facts:
        name = f"{fact.field_role.value.lower()}_identity"
        if fact.alignment_status is PairIdentityStatus.UNKNOWN:
            identities = (SemanticIdentity.unresolved(), SemanticIdentity.unresolved())
        else:
            common = {
                "status": "RESOLVED",
                "namespace": f"comparison:{PAIR_KIND}:{pair.binding.pair_ref}:{fact.field_role}",
                "normalization_version": PAIR_IDENTITY_SCHEMA,
                "basis": SemanticIdentityBasis.UPSTREAM_RESOLVER,
                "provenance_ref": fact.alignment_ref,
            }
            sides = (
                ("SAME", "SAME")
                if fact.alignment_status is PairIdentityStatus.SAME
                else ("LEFT", "RIGHT")
            )
            constructed = tuple(
                SemanticIdentity.model_validate(
                    {
                        **common,
                        "key": content_hash(
                            [
                                pair.binding.pair_ref,
                                fact.field_role.value,
                                PAIR_IDENTITY_RUNTIME,
                                side,
                            ]
                        ),
                    }
                )
                for side in sides
            )
            identities = (constructed[0], constructed[1])
        left_updates[name], right_updates[name] = identities
    return (
        pair.left.assertion.model_copy(update=left_updates),
        pair.right.assertion.model_copy(update=right_updates),
    )
