"""Pure, conservative proposition-level corroboration aggregation for V2.2-B2."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.atomic_proposition import EvidencePropositionRelation
from app.domain.proposition_support_group import (
    EvidenceOwnerIndependenceAssessment,
    EvidenceSourceIdentity,
    PropositionSupportGroup,
    assess_owner_independence,
)

INDEPENDENT_CORROBORATION_SCHEMA_VERSION = "independent-corroboration-v1"


class CorroborationEvidenceEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"


class OwnerCorroborationStatus(StrEnum):
    NO_ELIGIBLE_SUPPORT = "NO_ELIGIBLE_SUPPORT"
    SINGLE_KNOWN_OWNER = "SINGLE_KNOWN_OWNER"
    MULTI_KNOWN_OWNERS = "MULTI_KNOWN_OWNERS"
    UNKNOWN = "UNKNOWN"


class CorroborationReasonCode(StrEnum):
    ACCEPTED_SUPPORT = "ACCEPTED_SUPPORT"
    REJECTED_SUPPORT = "REJECTED_SUPPORT"
    SUPPORT_ELIGIBILITY_UNKNOWN = "SUPPORT_ELIGIBILITY_UNKNOWN"
    ELIGIBILITY_INPUT_MISSING = "ELIGIBILITY_INPUT_MISSING"
    NON_SUPPORT_DIRECTION = "NON_SUPPORT_DIRECTION"
    CONTRADICTING_EVIDENCE_RETAINED = "CONTRADICTING_EVIDENCE_RETAINED"
    OWNER_KEY_UNKNOWN = "OWNER_KEY_UNKNOWN"
    OWNER_KEYS_EQUAL = "OWNER_KEYS_EQUAL"
    OWNER_KEYS_DISTINCT = "OWNER_KEYS_DISTINCT"
    MULTIPLE_KNOWN_SUPPORT_OWNERS = "MULTIPLE_KNOWN_SUPPORT_OWNERS"


class CorroborationEvidenceEligibilityInput(BaseModel):
    """Per-evidence projection of the existing formal accepted/rejected result.

    ``accepted=None`` means no authoritative acceptance result was supplied.
    Scores, source roles, quote heuristics, and assertion-level grounding are
    intentionally not accepted by this contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ref: UUID
    accepted: bool | None


class NonSupportEvidenceDirection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: UUID
    relations: tuple[EvidencePropositionRelation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_relations(self) -> NonSupportEvidenceDirection:
        if self.relations != tuple(sorted(set(self.relations), key=lambda item: item.value)):
            raise ValueError("non_support_relations_must_be_unique_and_canonical")
        if EvidencePropositionRelation.SUPPORTS in self.relations:
            raise ValueError("support_relation_cannot_be_non_support_direction")
        return self


class IndependentCorroborationAssessment(BaseModel):
    """Derived evidence/owner facts only; deliberately contains no verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_evidence_refs: tuple[UUID, ...] = ()
    eligible_support_evidence_refs: tuple[UUID, ...] = ()
    ineligible_support_evidence_refs: tuple[UUID, ...] = ()
    unknown_eligibility_support_evidence_refs: tuple[UUID, ...] = ()
    non_support_evidence_refs: tuple[UUID, ...] = ()
    non_support_directions: tuple[NonSupportEvidenceDirection, ...] = ()
    known_support_owner_keys: tuple[str, ...] = ()
    unknown_owner_support_evidence_refs: tuple[UUID, ...] = ()
    support_evidence_count: int = Field(ge=0)
    eligible_support_evidence_count: int = Field(ge=0)
    known_owner_count: int = Field(ge=0)
    owner_corroboration_status: OwnerCorroborationStatus
    reason_codes: tuple[CorroborationReasonCode, ...] = ()
    schema_version: str = INDEPENDENT_CORROBORATION_SCHEMA_VERSION

    @model_validator(mode="after")
    def canonical_counts_and_refs(self) -> IndependentCorroborationAssessment:
        canonical_ref_fields = (
            self.support_evidence_refs,
            self.eligible_support_evidence_refs,
            self.ineligible_support_evidence_refs,
            self.unknown_eligibility_support_evidence_refs,
            self.non_support_evidence_refs,
            self.unknown_owner_support_evidence_refs,
        )
        if any(refs != tuple(sorted(set(refs), key=str)) for refs in canonical_ref_fields):
            raise ValueError("corroboration_evidence_refs_must_be_unique_and_canonical")
        if self.known_support_owner_keys != tuple(sorted(set(self.known_support_owner_keys))):
            raise ValueError("known_support_owner_keys_must_be_unique_and_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("corroboration_reason_codes_must_be_unique_and_canonical")
        if self.support_evidence_count != len(self.support_evidence_refs):
            raise ValueError("support_evidence_count_mismatch")
        if self.eligible_support_evidence_count != len(self.eligible_support_evidence_refs):
            raise ValueError("eligible_support_evidence_count_mismatch")
        if self.known_owner_count != len(self.known_support_owner_keys):
            raise ValueError("known_owner_count_mismatch")
        support_sets = (
            set(self.eligible_support_evidence_refs),
            set(self.ineligible_support_evidence_refs),
            set(self.unknown_eligibility_support_evidence_refs),
        )
        if any(
            left & right
            for index, left in enumerate(support_sets)
            for right in support_sets[index + 1 :]
        ):
            raise ValueError("support_eligibility_sets_must_be_disjoint")
        if set.union(*support_sets) != set(self.support_evidence_refs):
            raise ValueError("support_eligibility_sets_must_partition_support_evidence")
        if set(self.non_support_evidence_refs) & set(self.support_evidence_refs):
            raise ValueError("support_and_non_support_evidence_must_be_disjoint")
        return self


def resolve_evidence_eligibility(
    signals: Sequence[CorroborationEvidenceEligibility],
) -> CorroborationEvidenceEligibility:
    """Combine already-typed required signals with INELIGIBLE > UNKNOWN precedence."""
    if not signals:
        return CorroborationEvidenceEligibility.UNKNOWN
    if CorroborationEvidenceEligibility.INELIGIBLE in signals:
        return CorroborationEvidenceEligibility.INELIGIBLE
    if CorroborationEvidenceEligibility.UNKNOWN in signals:
        return CorroborationEvidenceEligibility.UNKNOWN
    return CorroborationEvidenceEligibility.ELIGIBLE


def _owner_key_is_unknown(value: str | None) -> bool:
    return value is None or value == "unknown"


def _validate_owner_assessments(
    supplied: Sequence[EvidenceOwnerIndependenceAssessment],
    source_identity_by_evidence: dict[UUID, EvidenceSourceIdentity],
) -> None:
    seen: dict[tuple[UUID, UUID], EvidenceOwnerIndependenceAssessment] = {}
    for assessment in supplied:
        if assessment.left_evidence_ref == assessment.right_evidence_ref:
            raise ValueError("owner_assessment_requires_distinct_evidence")
        pair_refs = sorted(
            (assessment.left_evidence_ref, assessment.right_evidence_ref), key=str
        )
        pair = (pair_refs[0], pair_refs[1])
        if any(ref not in source_identity_by_evidence for ref in pair):
            raise ValueError("owner_assessment_evidence_not_in_group")
        expected = assess_owner_independence(
            source_identity_by_evidence[pair[0]], source_identity_by_evidence[pair[1]]
        )
        normalized = assessment.model_copy(
            update={
                "left_evidence_ref": pair[0],
                "right_evidence_ref": pair[1],
                "left_source_ref": source_identity_by_evidence[pair[0]].source_ref,
                "right_source_ref": source_identity_by_evidence[pair[1]].source_ref,
            }
        )
        prior = seen.get(pair)
        if prior is not None and prior != normalized:
            raise ValueError("owner_assessment_pair_inconsistent")
        seen[pair] = normalized
        if normalized != expected:
            raise ValueError("owner_assessment_mismatches_canonical_owner_keys")


def aggregate_independent_corroboration(
    group: PropositionSupportGroup,
    eligibility_inputs: Sequence[CorroborationEvidenceEligibilityInput],
    owner_assessments: Sequence[EvidenceOwnerIndependenceAssessment] = (),
) -> IndependentCorroborationAssessment:
    """Aggregate only direct SUPPORTS memberships and formal acceptance results."""
    memberships = {item.evidence_ref: item for item in group.evidence_memberships}
    source_identity_by_evidence: dict[UUID, EvidenceSourceIdentity] = {}
    relations_by_evidence: dict[UUID, tuple[EvidencePropositionRelation, ...]] = {}

    for evidence_ref, membership in memberships.items():
        links = membership.provenance_links
        source_ids = {link.source_id for link in links}
        membership_owner_keys = {link.source_owner_key for link in links}
        if len(source_ids) != 1 or len(membership_owner_keys) != 1:
            raise ValueError("evidence_owner_or_source_inconsistent")
        relations = tuple(sorted({link.relation for link in links}, key=lambda item: item.value))
        if len(relations) != 1:
            raise ValueError("PROVENANCE_RELATION_INCONSISTENT")
        source_identity_by_evidence[evidence_ref] = EvidenceSourceIdentity(
            evidence_ref=evidence_ref,
            source_ref=next(iter(source_ids)),
            source_owner_key=next(iter(membership_owner_keys)),
        )
        relations_by_evidence[evidence_ref] = relations

    _validate_owner_assessments(owner_assessments, source_identity_by_evidence)

    eligibility_by_evidence: dict[UUID, bool | None] = {}
    for item in eligibility_inputs:
        if item.group_ref != group.group_ref or item.evidence_ref not in memberships:
            raise ValueError("evidence_eligibility_group_membership_mismatch")
        if item.evidence_ref in eligibility_by_evidence:
            if eligibility_by_evidence[item.evidence_ref] != item.accepted:
                raise ValueError("evidence_eligibility_input_inconsistent")
            continue
        eligibility_by_evidence[item.evidence_ref] = item.accepted

    support_refs: list[UUID] = []
    eligible_refs: list[UUID] = []
    ineligible_refs: list[UUID] = []
    unknown_eligibility_refs: list[UUID] = []
    non_support_refs: list[UUID] = []
    non_support_directions: list[NonSupportEvidenceDirection] = []
    unknown_owner_refs: list[UUID] = []
    owner_keys: set[str] = set()
    reasons: set[CorroborationReasonCode] = set()

    for evidence_ref in sorted(memberships, key=str):
        relation = relations_by_evidence[evidence_ref][0]
        if relation is not EvidencePropositionRelation.SUPPORTS:
            non_support_refs.append(evidence_ref)
            non_support_directions.append(
                NonSupportEvidenceDirection(
                    evidence_ref=evidence_ref,
                    relations=relations_by_evidence[evidence_ref],
                )
            )
            reasons.add(CorroborationReasonCode.NON_SUPPORT_DIRECTION)
            if relation is EvidencePropositionRelation.CONTRADICTS:
                reasons.add(CorroborationReasonCode.CONTRADICTING_EVIDENCE_RETAINED)
            continue

        support_refs.append(evidence_ref)
        if evidence_ref not in eligibility_by_evidence:
            eligibility = CorroborationEvidenceEligibility.UNKNOWN
            reasons.add(CorroborationReasonCode.ELIGIBILITY_INPUT_MISSING)
        else:
            accepted = eligibility_by_evidence[evidence_ref]
            if accepted is True:
                eligibility = CorroborationEvidenceEligibility.ELIGIBLE
            elif accepted is False:
                eligibility = CorroborationEvidenceEligibility.INELIGIBLE
            else:
                eligibility = CorroborationEvidenceEligibility.UNKNOWN
        eligibility = resolve_evidence_eligibility((eligibility,))
        if eligibility is CorroborationEvidenceEligibility.ELIGIBLE:
            eligible_refs.append(evidence_ref)
            reasons.add(CorroborationReasonCode.ACCEPTED_SUPPORT)
            owner_key = source_identity_by_evidence[evidence_ref].source_owner_key
            if _owner_key_is_unknown(owner_key):
                unknown_owner_refs.append(evidence_ref)
                reasons.add(CorroborationReasonCode.OWNER_KEY_UNKNOWN)
            else:
                assert owner_key is not None
                owner_keys.add(owner_key)
        elif eligibility is CorroborationEvidenceEligibility.INELIGIBLE:
            ineligible_refs.append(evidence_ref)
            reasons.add(CorroborationReasonCode.REJECTED_SUPPORT)
        else:
            unknown_eligibility_refs.append(evidence_ref)
            reasons.add(CorroborationReasonCode.SUPPORT_ELIGIBILITY_UNKNOWN)

    if len(owner_keys) >= 2:
        owner_status = OwnerCorroborationStatus.MULTI_KNOWN_OWNERS
        reasons.add(CorroborationReasonCode.MULTIPLE_KNOWN_SUPPORT_OWNERS)
    elif unknown_owner_refs or unknown_eligibility_refs:
        owner_status = OwnerCorroborationStatus.UNKNOWN
    elif len(owner_keys) == 1:
        owner_status = OwnerCorroborationStatus.SINGLE_KNOWN_OWNER
    else:
        owner_status = OwnerCorroborationStatus.NO_ELIGIBLE_SUPPORT

    if len(owner_keys) == 1:
        reasons.add(CorroborationReasonCode.OWNER_KEYS_EQUAL)
    elif len(owner_keys) >= 2:
        reasons.add(CorroborationReasonCode.OWNER_KEYS_DISTINCT)

    return IndependentCorroborationAssessment(
        group_ref=group.group_ref,
        support_evidence_refs=tuple(support_refs),
        eligible_support_evidence_refs=tuple(eligible_refs),
        ineligible_support_evidence_refs=tuple(ineligible_refs),
        unknown_eligibility_support_evidence_refs=tuple(unknown_eligibility_refs),
        non_support_evidence_refs=tuple(non_support_refs),
        non_support_directions=tuple(non_support_directions),
        known_support_owner_keys=tuple(sorted(owner_keys)),
        unknown_owner_support_evidence_refs=tuple(unknown_owner_refs),
        support_evidence_count=len(support_refs),
        eligible_support_evidence_count=len(eligible_refs),
        known_owner_count=len(owner_keys),
        owner_corroboration_status=owner_status,
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
    )
