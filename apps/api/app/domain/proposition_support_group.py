"""Pure B1 proposition grouping and owner-level independence contracts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.atomic_proposition import (
    ClaimPropositionLink,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.verification_assertion import PropositionPairAssessment

PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION = "proposition-support-group-v1"
EVIDENCE_OWNER_INDEPENDENCE_SCHEMA_VERSION = "evidence-owner-independence-v1"


class PropositionGroupingStatus(StrEnum):
    GROUPED = "GROUPED"
    SINGLETON = "SINGLETON"
    UNRESOLVED_MEMBERSHIP = "UNRESOLVED_MEMBERSHIP"


class PropositionGroupingReason(StrEnum):
    EXACT_PROPOSITION_REF = "EXACT_PROPOSITION_REF"
    PAIR_SAME = "PAIR_SAME"
    PAIR_NOT_SAME = "PAIR_NOT_SAME"
    PAIR_ASSESSMENT_MISSING = "PAIR_ASSESSMENT_MISSING"
    GROUPING_INCONSISTENT = "GROUPING_INCONSISTENT"


class OwnerIndependenceStatus(StrEnum):
    DISTINCT_OWNER = "DISTINCT_OWNER"
    SAME_OWNER = "SAME_OWNER"
    UNKNOWN = "UNKNOWN"


class OwnerIndependenceReason(StrEnum):
    OWNER_KEYS_DISTINCT = "OWNER_KEYS_DISTINCT"
    OWNER_KEYS_EQUAL = "OWNER_KEYS_EQUAL"
    OWNER_KEY_UNKNOWN = "OWNER_KEY_UNKNOWN"


class PropositionAssertionBinding(BaseModel):
    """Binds a proposition record ref to its already-projected A3 assertion ref."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_ref: UUID
    assertion_ref: str = Field(min_length=1, max_length=128)


class PropositionEvidenceMembership(BaseModel):
    """One unique Evidence membership with all direct provenance paths retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: UUID
    provenance_links: tuple[PropositionEvidenceLink, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_refs(self) -> PropositionEvidenceMembership:
        if any(link.evidence_id != self.evidence_ref for link in self.provenance_links):
            raise ValueError("membership_provenance_evidence_ref_mismatch")
        source_ids = {link.source_id for link in self.provenance_links}
        owner_keys = {link.source_owner_key for link in self.provenance_links}
        if len(source_ids) != 1 or len(owner_keys) != 1:
            raise ValueError("evidence_identity_provenance_conflict")
        return self


class PropositionSupportGroup(BaseModel):
    """A derived group; deliberately contains no verification/corroboration result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposition_refs: tuple[UUID, ...] = Field(min_length=1)
    evidence_memberships: tuple[PropositionEvidenceMembership, ...] = ()
    grouping_status: PropositionGroupingStatus
    reason_codes: tuple[PropositionGroupingReason, ...] = ()
    schema_version: str = PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION

    @model_validator(mode="after")
    def canonical_members(self) -> PropositionSupportGroup:
        if self.proposition_refs != tuple(sorted(set(self.proposition_refs), key=str)):
            raise ValueError("proposition_group_refs_must_be_unique_and_canonical")
        evidence_refs = tuple(item.evidence_ref for item in self.evidence_memberships)
        if evidence_refs != tuple(sorted(set(evidence_refs), key=str)):
            raise ValueError("proposition_group_evidence_must_be_unique_and_canonical")
        return self


class PropositionGroupingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groups: tuple[PropositionSupportGroup, ...]
    schema_version: str = PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION


class EvidenceSourceIdentity(BaseModel):
    """Minimal owner-comparison projection; diversity metadata is never authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: UUID
    source_ref: UUID
    source_owner_key: str | None = Field(default=None, min_length=1, max_length=255)
    source_diversity_key: str | None = Field(default=None, min_length=1, max_length=255)


class EvidenceOwnerIndependenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_evidence_ref: UUID
    right_evidence_ref: UUID
    left_source_ref: UUID
    right_source_ref: UUID
    status: OwnerIndependenceStatus
    reason_codes: tuple[OwnerIndependenceReason, ...]
    schema_version: str = EVIDENCE_OWNER_INDEPENDENCE_SCHEMA_VERSION

    @model_validator(mode="after")
    def distinct_evidence(self) -> EvidenceOwnerIndependenceAssessment:
        if self.left_evidence_ref == self.right_evidence_ref:
            raise ValueError("owner_independence_requires_distinct_evidence")
        return self


def _group_ref(proposition_refs: tuple[UUID, ...]) -> str:
    payload = {
        "schema_version": PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION,
        "proposition_refs": [str(ref) for ref in proposition_refs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _link_key(link: PropositionEvidenceLink) -> str:
    return json.dumps(link.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _assessment_index(
    assessments: Sequence[PropositionPairAssessment],
) -> dict[tuple[str, str], PropositionPairAssessment]:
    result: dict[tuple[str, str], PropositionPairAssessment] = {}
    for item in assessments:
        key = _pair_key(item.left_assertion_ref, item.right_assertion_ref)
        prior = result.get(key)
        if prior is not None and (
            prior.relation is not item.relation
            or prior.comparability is not item.comparability
        ):
            raise ValueError("conflicting_a3_pair_assessments")
        result[key] = item
    return result


def _cross_relation(
    left_refs: tuple[str, ...],
    right_refs: tuple[str, ...],
    assessments: dict[tuple[str, str], PropositionPairAssessment],
) -> PropositionRelation | None:
    """Return SAME only if every assertion pair has an explicit A3 SAME result."""
    if not left_refs or not right_refs:
        return None
    relations: list[PropositionRelation] = []
    for left in left_refs:
        for right in right_refs:
            assessment = assessments.get(_pair_key(left, right))
            if assessment is None:
                return None
            relations.append(assessment.relation)
    if all(relation is PropositionRelation.SAME for relation in relations):
        return PropositionRelation.SAME
    if PropositionRelation.CONTRADICTS in relations:
        return PropositionRelation.CONTRADICTS
    return PropositionRelation.UNKNOWN


def _components(
    proposition_refs: tuple[UUID, ...],
    same_edges: set[tuple[UUID, UUID]],
) -> tuple[tuple[UUID, ...], ...]:
    neighbors: dict[UUID, set[UUID]] = {ref: set() for ref in proposition_refs}
    for left, right in same_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    remaining = set(proposition_refs)
    result: list[tuple[UUID, ...]] = []
    while remaining:
        root = min(remaining, key=str)
        stack = [root]
        found: set[UUID] = set()
        while stack:
            item = stack.pop()
            if item in found:
                continue
            found.add(item)
            stack.extend(sorted(neighbors[item] - found, key=str, reverse=True))
        remaining -= found
        result.append(tuple(sorted(found, key=str)))
    return tuple(sorted(result, key=lambda component: tuple(map(str, component))))


def _evidence_memberships(
    proposition_refs: tuple[UUID, ...],
    provenance_by_proposition: dict[UUID, list[PropositionEvidenceLink]],
) -> tuple[PropositionEvidenceMembership, ...]:
    by_evidence: dict[UUID, dict[str, PropositionEvidenceLink]] = defaultdict(dict)
    for proposition_ref in proposition_refs:
        for link in provenance_by_proposition.get(proposition_ref, []):
            by_evidence[link.evidence_id][_link_key(link)] = link
    memberships = [
        PropositionEvidenceMembership(
            evidence_ref=evidence_ref,
            provenance_links=tuple(
                by_evidence[evidence_ref][key] for key in sorted(by_evidence[evidence_ref])
            ),
        )
        for evidence_ref in sorted(by_evidence, key=str)
    ]
    return tuple(memberships)


def group_propositions(
    claim_proposition_links: Sequence[ClaimPropositionLink],
    proposition_evidence_links: Sequence[PropositionEvidenceLink],
    assertion_bindings: Sequence[PropositionAssertionBinding],
    pair_assessments: Sequence[PropositionPairAssessment],
) -> PropositionGroupingResult:
    """Build clique-safe proposition groups using exact refs and frozen A3 results.

    A connected SAME component is merged only if it is a complete SAME clique.
    Non-clique components fail closed to unresolved singleton memberships; no
    bridge/transitive inference or input-order partitioning is performed.
    """
    claim_links = {
        (item.claim_id, item.proposition_id): item for item in claim_proposition_links
    }
    if not claim_links:
        return PropositionGroupingResult(groups=())

    proposition_refs = tuple(sorted({proposition_id for _, proposition_id in claim_links}, key=str))
    assertion_refs_by_proposition: dict[UUID, set[str]] = defaultdict(set)
    assertion_to_proposition: dict[str, UUID] = {}
    for binding in assertion_bindings:
        prior = assertion_to_proposition.get(binding.assertion_ref)
        if prior is not None and prior != binding.proposition_ref:
            raise ValueError("assertion_ref_bound_to_multiple_propositions")
        if binding.proposition_ref not in set(proposition_refs):
            raise ValueError("assertion_binding_proposition_not_in_input")
        assertion_to_proposition[binding.assertion_ref] = binding.proposition_ref
        assertion_refs_by_proposition[binding.proposition_ref].add(binding.assertion_ref)

    assessment_by_pair = _assessment_index(pair_assessments)
    same_edges: set[tuple[UUID, UUID]] = set()
    pair_relations: dict[tuple[UUID, UUID], PropositionRelation | None] = {}
    for index, left_ref in enumerate(proposition_refs):
        for right_ref in proposition_refs[index + 1 :]:
            relation = _cross_relation(
                tuple(sorted(assertion_refs_by_proposition.get(left_ref, set()))),
                tuple(sorted(assertion_refs_by_proposition.get(right_ref, set()))),
                assessment_by_pair,
            )
            pair_relations[(left_ref, right_ref)] = relation
            if relation is PropositionRelation.SAME:
                same_edges.add((left_ref, right_ref))

    components = _components(proposition_refs, same_edges)
    for component in components:
        if len(component) < 3:
            continue
        if any(
            pair_relations.get((left, right)) is PropositionRelation.CONTRADICTS
            for index, left in enumerate(component)
            for right in component[index + 1 :]
        ):
            raise ValueError("GROUPING_INCONSISTENT")

    provenance_by_proposition: dict[UUID, list[PropositionEvidenceLink]] = defaultdict(list)
    seen_links: set[str] = set()
    for link in proposition_evidence_links:
        if (link.claim_id, link.proposition_id) not in claim_links:
            raise ValueError("proposition_evidence_link_missing_claim_proposition_link")
        link_key = _link_key(link)
        if link_key not in seen_links:
            seen_links.add(link_key)
            provenance_by_proposition[link.proposition_id].append(link)

    groups: list[PropositionSupportGroup] = []
    for component in components:
        complete = all(
            pair_relations.get((left, right)) is PropositionRelation.SAME
            for index, left in enumerate(component)
            for right in component[index + 1 :]
        )
        if len(component) == 1:
            group_sets: tuple[tuple[UUID, ...], ...] = (component,)
            status = PropositionGroupingStatus.SINGLETON
            reasons: tuple[PropositionGroupingReason, ...] = ()
        elif complete:
            group_sets = (component,)
            status = PropositionGroupingStatus.GROUPED
            reasons = (PropositionGroupingReason.PAIR_SAME,)
        else:
            group_sets = tuple((ref,) for ref in component)
            status = PropositionGroupingStatus.UNRESOLVED_MEMBERSHIP
            component_relations = [
                pair_relations.get((left, right))
                for index, left in enumerate(component)
                for right in component[index + 1 :]
            ]
            reasons = tuple(
                [PropositionGroupingReason.PAIR_NOT_SAME]
                + (
                    [PropositionGroupingReason.PAIR_ASSESSMENT_MISSING]
                    if None in component_relations
                    else []
                )
            )
        for members in group_sets:
            canonical_members = tuple(sorted(members, key=str))
            groups.append(
                PropositionSupportGroup(
                    group_ref=_group_ref(canonical_members),
                    proposition_refs=canonical_members,
                    evidence_memberships=_evidence_memberships(
                        canonical_members, provenance_by_proposition
                    ),
                    grouping_status=status,
                    reason_codes=reasons,
                )
            )

    groups.sort(key=lambda group: tuple(map(str, group.proposition_refs)))
    return PropositionGroupingResult(groups=tuple(groups))


def assess_owner_independence(
    left: EvidenceSourceIdentity,
    right: EvidenceSourceIdentity,
) -> EvidenceOwnerIndependenceAssessment:
    """Compare only canonical source_owner_key values; other metadata is ignored."""
    if left.evidence_ref == right.evidence_ref:
        raise ValueError("owner_independence_requires_distinct_evidence")
    first, second = sorted((left, right), key=lambda item: str(item.evidence_ref))
    left_key = left.source_owner_key
    right_key = right.source_owner_key
    if (
        not left_key
        or not left_key.strip()
        or left_key == "unknown"
        or not right_key
        or not right_key.strip()
        or right_key == "unknown"
    ):
        status = OwnerIndependenceStatus.UNKNOWN
        reason = OwnerIndependenceReason.OWNER_KEY_UNKNOWN
    elif left_key == right_key:
        status = OwnerIndependenceStatus.SAME_OWNER
        reason = OwnerIndependenceReason.OWNER_KEYS_EQUAL
    else:
        status = OwnerIndependenceStatus.DISTINCT_OWNER
        reason = OwnerIndependenceReason.OWNER_KEYS_DISTINCT
    return EvidenceOwnerIndependenceAssessment(
        left_evidence_ref=first.evidence_ref,
        right_evidence_ref=second.evidence_ref,
        left_source_ref=first.source_ref,
        right_source_ref=second.source_ref,
        status=status,
        reason_codes=(reason,),
    )
