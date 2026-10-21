"""Pure, deterministic V2.2-C1 contradiction and conflict structures."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.atomic_proposition import EvidencePropositionRelation, PropositionRelation
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibility,
    IndependentCorroborationAssessment,
    resolve_evidence_eligibility,
)
from app.domain.proposition_support_group import (
    PropositionAssertionBinding,
    PropositionSupportGroup,
)
from app.domain.verification_assertion import (
    ComparabilityStatus,
    PropositionPairAssessment,
)

CONFLICT_ASSESSMENT_SCHEMA_VERSION = "proposition-conflict-assessment-v1"


class GroupPairConflictState(StrEnum):
    DIRECT_CONTRADICTION = "DIRECT_CONTRADICTION"
    NO_DIRECT_CONTRADICTION = "NO_DIRECT_CONTRADICTION"
    UNKNOWN = "UNKNOWN"
    GROUPING_UNRESOLVED = "GROUPING_UNRESOLVED"
    GROUPING_INCONSISTENCY = "GROUPING_INCONSISTENCY"
    PAIR_RELATION_INCONSISTENT = "PAIR_RELATION_INCONSISTENT"


class ConflictReasonCode(StrEnum):
    DIRECT_PROPOSITION_CONTRADICTION = "DIRECT_PROPOSITION_CONTRADICTION"
    NO_DIRECT_CONTRADICTION = "NO_DIRECT_CONTRADICTION"
    CONTRADICTION_COVERAGE_INCOMPLETE = "CONTRADICTION_COVERAGE_INCOMPLETE"
    COUNTEREVIDENCE_PRESENT = "COUNTEREVIDENCE_PRESENT"
    COUNTEREVIDENCE_ELIGIBILITY_UNKNOWN = "COUNTEREVIDENCE_ELIGIBILITY_UNKNOWN"
    GROUP_PAIR_ASSESSMENT_MISSING = "GROUP_PAIR_ASSESSMENT_MISSING"
    GROUPING_UNRESOLVED = "GROUPING_UNRESOLVED"
    GROUPING_INCONSISTENT = "GROUPING_INCONSISTENT"
    PAIR_ASSESSMENT_INCONSISTENT = "PAIR_ASSESSMENT_INCONSISTENT"


class PropositionPairRef(BaseModel):
    """Canonical proposition pair with the A3 assertion refs that identify it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_proposition_ref: UUID
    right_proposition_ref: UUID
    left_assertion_ref: str | None = Field(default=None, min_length=1, max_length=128)
    right_assertion_ref: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def canonical_order(self) -> PropositionPairRef:
        if str(self.left_proposition_ref) >= str(self.right_proposition_ref):
            raise ValueError("proposition_pair_refs_must_be_canonical_and_distinct")
        return self


class PropositionGroupContradictionEdge(BaseModel):
    """One direct group conflict, backed only by concrete A3 CONTRADICTS pairs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    contradiction_pair_refs: tuple[PropositionPairRef, ...] = Field(min_length=1)
    reason_codes: tuple[ConflictReasonCode, ...]
    schema_version: str = "proposition-group-contradiction-edge-v1"

    @model_validator(mode="after")
    def canonical_edge(self) -> PropositionGroupContradictionEdge:
        if self.left_group_ref >= self.right_group_ref:
            raise ValueError("group_pair_refs_must_be_canonical_and_distinct")
        if self.contradiction_pair_refs != tuple(
            sorted(
                set(self.contradiction_pair_refs),
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ):
            raise ValueError("contradiction_pair_refs_must_be_unique_and_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("conflict_reason_codes_must_be_unique_and_canonical")
        return self


class GroupPairRelationCoverage(BaseModel):
    """Coverage and classified A3 pair refs for one unordered group pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_cross_pair_count: int = Field(ge=1)
    assessed_pair_count: int = Field(ge=0)
    direct_contradiction_pair_refs: tuple[PropositionPairRef, ...] = ()
    same_pair_refs: tuple[PropositionPairRef, ...] = ()
    related_pair_refs: tuple[PropositionPairRef, ...] = ()
    not_comparable_pair_refs: tuple[PropositionPairRef, ...] = ()
    unknown_pair_refs: tuple[PropositionPairRef, ...] = ()
    missing_pair_refs: tuple[PropositionPairRef, ...] = ()
    inconsistent_pair_refs: tuple[PropositionPairRef, ...] = ()
    coverage_status: GroupPairConflictState
    reason_codes: tuple[ConflictReasonCode, ...] = ()
    schema_version: str = "group-pair-relation-coverage-v1"

    @model_validator(mode="after")
    def validate_coverage(self) -> GroupPairRelationCoverage:
        if self.left_group_ref >= self.right_group_ref:
            raise ValueError("group_pair_refs_must_be_canonical_and_distinct")
        categorized = (
            self.direct_contradiction_pair_refs,
            self.same_pair_refs,
            self.related_pair_refs,
            self.not_comparable_pair_refs,
            self.unknown_pair_refs,
            self.missing_pair_refs,
            self.inconsistent_pair_refs,
        )
        flattened = [pair for values in categorized for pair in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("group_pair_relation_refs_must_be_disjoint")
        if len(flattened) != self.expected_cross_pair_count:
            raise ValueError("group_pair_coverage_must_partition_expected_pairs")
        if self.assessed_pair_count != sum(
            len(values)
            for values in (
                self.direct_contradiction_pair_refs,
                self.same_pair_refs,
                self.related_pair_refs,
                self.not_comparable_pair_refs,
                self.unknown_pair_refs,
                self.inconsistent_pair_refs,
            )
        ):
            raise ValueError("group_pair_assessed_count_mismatch")
        for values in categorized:
            if values != tuple(
                sorted(
                    set(values),
                    key=lambda item: (
                        str(item.left_proposition_ref),
                        str(item.right_proposition_ref),
                    ),
                )
            ):
                raise ValueError("group_pair_refs_must_be_unique_and_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("conflict_reason_codes_must_be_unique_and_canonical")
        return self


class CounterEvidenceAssessment(BaseModel):
    """Evidence-direction counterevidence; never a proposition-pair conflict verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    contradicting_evidence_refs: tuple[UUID, ...] = ()
    eligible_counterevidence_refs: tuple[UUID, ...] = ()
    ineligible_counterevidence_refs: tuple[UUID, ...] = ()
    unknown_eligibility_counterevidence_refs: tuple[UUID, ...] = ()
    known_counterevidence_owner_keys: tuple[str, ...] = ()
    unknown_owner_counterevidence_refs: tuple[UUID, ...] = ()
    eligible_counterevidence_count: int = Field(ge=0)
    known_counterevidence_owner_count: int = Field(ge=0)
    reason_codes: tuple[ConflictReasonCode, ...] = ()
    schema_version: str = "counterevidence-assessment-v1"

    @model_validator(mode="after")
    def canonical_counterevidence(self) -> CounterEvidenceAssessment:
        ref_fields = (
            self.contradicting_evidence_refs,
            self.eligible_counterevidence_refs,
            self.ineligible_counterevidence_refs,
            self.unknown_eligibility_counterevidence_refs,
            self.unknown_owner_counterevidence_refs,
        )
        if any(refs != tuple(sorted(set(refs), key=str)) for refs in ref_fields):
            raise ValueError("counterevidence_refs_must_be_unique_and_canonical")
        if self.known_counterevidence_owner_keys != tuple(
            sorted(set(self.known_counterevidence_owner_keys))
        ):
            raise ValueError("counterevidence_owner_keys_must_be_unique_and_canonical")
        buckets = (
            set(self.eligible_counterevidence_refs),
            set(self.ineligible_counterevidence_refs),
            set(self.unknown_eligibility_counterevidence_refs),
        )
        if any(
            left & right for index, left in enumerate(buckets) for right in buckets[index + 1 :]
        ):
            raise ValueError("counterevidence_eligibility_sets_must_be_disjoint")
        if set.union(*buckets) != set(self.contradicting_evidence_refs):
            raise ValueError("counterevidence_eligibility_sets_must_partition_candidates")
        if not set(self.unknown_owner_counterevidence_refs).issubset(
            set(self.eligible_counterevidence_refs)
        ):
            raise ValueError("unknown_counterevidence_owners_must_be_eligible_refs")
        if self.eligible_counterevidence_count != len(self.eligible_counterevidence_refs):
            raise ValueError("eligible_counterevidence_count_mismatch")
        if self.known_counterevidence_owner_count != len(self.known_counterevidence_owner_keys):
            raise ValueError("counterevidence_owner_count_mismatch")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("conflict_reason_codes_must_be_unique_and_canonical")
        return self


class OwnerOppositionPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    support_owner_key: str = Field(min_length=1, max_length=255)
    counterevidence_owner_key: str = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def owners_must_differ(self) -> OwnerOppositionPair:
        if self.support_owner_key == self.counterevidence_owner_key:
            raise ValueError("owner_opposition_pair_requires_distinct_owners")
        return self


class CounterEvidenceEligibilityInput(BaseModel):
    """Existing acceptance result projected onto one group's evidence ref."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ref: UUID
    accepted: bool | None


class PropositionConflictSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    direct_conflicting_group_refs: tuple[str, ...] = ()
    support_owner_keys: tuple[str, ...] | None = None
    counterevidence: CounterEvidenceAssessment
    support_counter_owner_overlap: tuple[str, ...] | None = None
    cross_owner_opposition_pairs: tuple[OwnerOppositionPair, ...] | None = None
    reason_codes: tuple[ConflictReasonCode, ...] = ()
    schema_version: str = "proposition-conflict-summary-v1"

    @model_validator(mode="after")
    def canonical_summary(self) -> PropositionConflictSummary:
        if self.direct_conflicting_group_refs != tuple(
            sorted(set(self.direct_conflicting_group_refs))
        ):
            raise ValueError("conflicting_group_refs_must_be_unique_and_canonical")
        if self.support_owner_keys is not None and self.support_owner_keys != tuple(
            sorted(set(self.support_owner_keys))
        ):
            raise ValueError("support_owner_keys_must_be_unique_and_canonical")
        if (
            self.support_counter_owner_overlap is not None
            and self.support_counter_owner_overlap
            != tuple(sorted(set(self.support_counter_owner_overlap)))
        ):
            raise ValueError("owner_overlap_must_be_unique_and_canonical")
        if (
            self.cross_owner_opposition_pairs is not None
            and self.cross_owner_opposition_pairs
            != tuple(
                sorted(
                    set(self.cross_owner_opposition_pairs),
                    key=lambda item: (item.support_owner_key, item.counterevidence_owner_key),
                )
            )
        ):
            raise ValueError("owner_opposition_pairs_must_be_unique_and_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("conflict_reason_codes_must_be_unique_and_canonical")
        return self


class PropositionConflictAssessment(BaseModel):
    """Pure conflict structure for C2; contains no final verification status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_pair_coverages: tuple[GroupPairRelationCoverage, ...] = ()
    contradiction_edges: tuple[PropositionGroupContradictionEdge, ...] = ()
    group_summaries: tuple[PropositionConflictSummary, ...]
    reason_codes: tuple[ConflictReasonCode, ...] = ()
    schema_version: str = CONFLICT_ASSESSMENT_SCHEMA_VERSION

    @model_validator(mode="after")
    def canonical_assessment(self) -> PropositionConflictAssessment:
        coverage_keys = tuple(
            (item.left_group_ref, item.right_group_ref) for item in self.group_pair_coverages
        )
        edge_keys = tuple(
            (item.left_group_ref, item.right_group_ref) for item in self.contradiction_edges
        )
        summary_refs = tuple(item.group_ref for item in self.group_summaries)
        if coverage_keys != tuple(sorted(set(coverage_keys))):
            raise ValueError("group_pair_coverages_must_be_unique_and_canonical")
        if edge_keys != tuple(sorted(set(edge_keys))):
            raise ValueError("contradiction_edges_must_be_unique_and_canonical")
        if summary_refs != tuple(sorted(set(summary_refs))):
            raise ValueError("group_summaries_must_be_unique_and_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("conflict_reason_codes_must_be_unique_and_canonical")
        return self


def _canonical_pair_ref(
    left_proposition_ref: UUID,
    right_proposition_ref: UUID,
    assertion_by_proposition: dict[UUID, str],
) -> PropositionPairRef:
    if str(left_proposition_ref) < str(right_proposition_ref):
        left_ref, right_ref = left_proposition_ref, right_proposition_ref
    else:
        left_ref, right_ref = right_proposition_ref, left_proposition_ref
    return PropositionPairRef(
        left_proposition_ref=left_ref,
        right_proposition_ref=right_ref,
        left_assertion_ref=assertion_by_proposition.get(left_ref),
        right_assertion_ref=assertion_by_proposition.get(right_ref),
    )


def _a3_semantics(assessment: PropositionPairAssessment) -> tuple[object, ...]:
    dimensions = tuple(
        sorted(
            (
                item.dimension.value,
                item.compatibility.value,
                tuple(sorted(reason.value for reason in item.reason_codes)),
            )
            for item in assessment.dimension_assessments
        )
    )
    return (
        assessment.comparability,
        assessment.relation,
        assessment.relation_eligible,
        dimensions,
        tuple(sorted(reason.value for reason in assessment.reason_codes)),
        assessment.schema_version,
        assessment.normalization_version,
    )


def _counterevidence_for_group(
    group: PropositionSupportGroup,
    accepted_by_evidence: dict[UUID, bool | None],
) -> CounterEvidenceAssessment:
    candidates: list[UUID] = []
    eligible: list[UUID] = []
    ineligible: list[UUID] = []
    unknown: list[UUID] = []
    known_owners: set[str] = set()
    unknown_owners: list[UUID] = []
    reasons: set[ConflictReasonCode] = set()

    for membership in group.evidence_memberships:
        if any(
            link.proposition_id not in group.proposition_refs
            for link in membership.provenance_links
        ):
            raise ValueError("evidence_membership_proposition_not_in_group")
        directions = {link.relation for link in membership.provenance_links}
        if len(directions) != 1:
            raise ValueError("PROVENANCE_RELATION_INCONSISTENT")
        if next(iter(directions)) is not EvidencePropositionRelation.CONTRADICTS:
            continue

        evidence_ref = membership.evidence_ref
        candidates.append(evidence_ref)
        if evidence_ref not in accepted_by_evidence:
            eligibility = CorroborationEvidenceEligibility.UNKNOWN
        else:
            accepted = accepted_by_evidence[evidence_ref]
            eligibility = (
                CorroborationEvidenceEligibility.ELIGIBLE
                if accepted is True
                else CorroborationEvidenceEligibility.INELIGIBLE
                if accepted is False
                else CorroborationEvidenceEligibility.UNKNOWN
            )
        eligibility = resolve_evidence_eligibility((eligibility,))
        owner_key = membership.provenance_links[0].source_owner_key
        if eligibility is CorroborationEvidenceEligibility.ELIGIBLE:
            eligible.append(evidence_ref)
            if owner_key is None or owner_key == "unknown":
                unknown_owners.append(evidence_ref)
            else:
                known_owners.add(owner_key)
        elif eligibility is CorroborationEvidenceEligibility.INELIGIBLE:
            ineligible.append(evidence_ref)
        else:
            unknown.append(evidence_ref)

    if candidates:
        reasons.add(ConflictReasonCode.COUNTEREVIDENCE_PRESENT)
    if unknown:
        reasons.add(ConflictReasonCode.COUNTEREVIDENCE_ELIGIBILITY_UNKNOWN)
    return CounterEvidenceAssessment(
        group_ref=group.group_ref,
        contradicting_evidence_refs=tuple(sorted(candidates, key=str)),
        eligible_counterevidence_refs=tuple(sorted(eligible, key=str)),
        ineligible_counterevidence_refs=tuple(sorted(ineligible, key=str)),
        unknown_eligibility_counterevidence_refs=tuple(sorted(unknown, key=str)),
        known_counterevidence_owner_keys=tuple(sorted(known_owners)),
        unknown_owner_counterevidence_refs=tuple(sorted(unknown_owners, key=str)),
        eligible_counterevidence_count=len(eligible),
        known_counterevidence_owner_count=len(known_owners),
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
    )


def _validate_corroboration_input(
    group: PropositionSupportGroup,
    assessment: IndependentCorroborationAssessment,
) -> None:
    memberships = {item.evidence_ref: item for item in group.evidence_memberships}
    support_refs = set(assessment.support_evidence_refs)
    eligible_refs = set(assessment.eligible_support_evidence_refs)
    unknown_owner_refs = set(assessment.unknown_owner_support_evidence_refs)
    if assessment.group_ref != group.group_ref:
        raise ValueError("corroboration_assessment_group_mismatch")
    if not support_refs.issubset(memberships):
        raise ValueError("corroboration_evidence_not_in_group")
    for evidence_ref in support_refs:
        if any(
            link.relation is not EvidencePropositionRelation.SUPPORTS
            for link in memberships[evidence_ref].provenance_links
        ):
            raise ValueError("corroboration_support_ref_direction_mismatch")
    expected_owners: set[str] = set()
    expected_unknown_owner_refs: set[UUID] = set()
    for evidence_ref in eligible_refs:
        membership = memberships[evidence_ref]
        if any(
            link.relation is not EvidencePropositionRelation.SUPPORTS
            for link in membership.provenance_links
        ):
            raise ValueError("corroboration_eligible_evidence_not_support_direction")
        owner_key = membership.provenance_links[0].source_owner_key
        if owner_key is None or owner_key == "unknown":
            expected_unknown_owner_refs.add(evidence_ref)
        else:
            expected_owners.add(owner_key)
    if set(assessment.known_support_owner_keys) != expected_owners:
        raise ValueError("corroboration_owner_keys_mismatch_group_provenance")
    if unknown_owner_refs != expected_unknown_owner_refs:
        raise ValueError("corroboration_unknown_owner_refs_mismatch_group_provenance")


def _pair_coverage(
    left_group: PropositionSupportGroup,
    right_group: PropositionSupportGroup,
    assertion_by_proposition: dict[UUID, str],
    assessments_by_assertion_pair: dict[tuple[str, str], PropositionPairAssessment],
    inconsistent_assertion_pairs: set[tuple[str, str]],
) -> tuple[GroupPairRelationCoverage, PropositionGroupContradictionEdge | None]:
    if left_group.group_ref < right_group.group_ref:
        left, right = left_group, right_group
    else:
        left, right = right_group, left_group

    refs: dict[str, list[PropositionPairRef]] = {
        "contradiction": [],
        "same": [],
        "related": [],
        "not_comparable": [],
        "unknown": [],
        "missing": [],
        "inconsistent": [],
    }
    for left_prop in left.proposition_refs:
        for right_prop in right.proposition_refs:
            pair_ref = _canonical_pair_ref(left_prop, right_prop, assertion_by_proposition)
            left_assertion = assertion_by_proposition.get(pair_ref.left_proposition_ref)
            right_assertion = assertion_by_proposition.get(pair_ref.right_proposition_ref)
            if left_assertion is None or right_assertion is None:
                refs["missing"].append(pair_ref)
                continue
            assertion_refs = sorted((left_assertion, right_assertion))
            assertion_pair = (assertion_refs[0], assertion_refs[1])
            if assertion_pair in inconsistent_assertion_pairs:
                refs["inconsistent"].append(pair_ref)
                continue
            assessment = assessments_by_assertion_pair.get(assertion_pair)
            if assessment is None:
                refs["missing"].append(pair_ref)
            elif assessment.relation is PropositionRelation.CONTRADICTS:
                if assessment.comparability is not ComparabilityStatus.COMPARABLE:
                    refs["inconsistent"].append(pair_ref)
                else:
                    refs["contradiction"].append(pair_ref)
            elif assessment.relation is PropositionRelation.SAME:
                refs["same"].append(pair_ref)
            elif assessment.relation is PropositionRelation.RELATED:
                refs["related"].append(pair_ref)
            elif assessment.comparability is ComparabilityStatus.NOT_COMPARABLE:
                refs["not_comparable"].append(pair_ref)
            else:
                refs["unknown"].append(pair_ref)

    contradiction = refs["contradiction"]
    same = refs["same"]
    expected_count = len(left.proposition_refs) * len(right.proposition_refs)
    reasons: set[ConflictReasonCode] = set()
    if refs["inconsistent"]:
        reasons.add(ConflictReasonCode.PAIR_ASSESSMENT_INCONSISTENT)
    if refs["missing"]:
        reasons.add(ConflictReasonCode.GROUP_PAIR_ASSESSMENT_MISSING)
    if refs["missing"] or refs["unknown"]:
        reasons.add(ConflictReasonCode.CONTRADICTION_COVERAGE_INCOMPLETE)

    if same and contradiction:
        state = GroupPairConflictState.PAIR_RELATION_INCONSISTENT
        reasons.add(ConflictReasonCode.PAIR_ASSESSMENT_INCONSISTENT)
    elif same and len(same) == expected_count:
        state = GroupPairConflictState.GROUPING_INCONSISTENCY
        reasons.add(ConflictReasonCode.GROUPING_INCONSISTENT)
    elif same:
        state = GroupPairConflictState.GROUPING_UNRESOLVED
        reasons.add(ConflictReasonCode.GROUPING_UNRESOLVED)
    elif contradiction:
        state = GroupPairConflictState.DIRECT_CONTRADICTION
        reasons.add(ConflictReasonCode.DIRECT_PROPOSITION_CONTRADICTION)
    elif refs["inconsistent"]:
        state = GroupPairConflictState.PAIR_RELATION_INCONSISTENT
    elif refs["missing"]:
        state = GroupPairConflictState.UNKNOWN
        reasons.add(ConflictReasonCode.GROUP_PAIR_ASSESSMENT_MISSING)
    elif refs["unknown"]:
        state = GroupPairConflictState.UNKNOWN
    else:
        state = GroupPairConflictState.NO_DIRECT_CONTRADICTION
        reasons.add(ConflictReasonCode.NO_DIRECT_CONTRADICTION)

    coverage = GroupPairRelationCoverage(
        left_group_ref=left.group_ref,
        right_group_ref=right.group_ref,
        expected_cross_pair_count=expected_count,
        assessed_pair_count=expected_count - len(refs["missing"]),
        direct_contradiction_pair_refs=tuple(
            sorted(
                contradiction,
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        same_pair_refs=tuple(
            sorted(
                same,
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        related_pair_refs=tuple(
            sorted(
                refs["related"],
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        not_comparable_pair_refs=tuple(
            sorted(
                refs["not_comparable"],
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        unknown_pair_refs=tuple(
            sorted(
                refs["unknown"],
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        missing_pair_refs=tuple(
            sorted(
                refs["missing"],
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        inconsistent_pair_refs=tuple(
            sorted(
                refs["inconsistent"],
                key=lambda item: (str(item.left_proposition_ref), str(item.right_proposition_ref)),
            )
        ),
        coverage_status=state,
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
    )

    edge = None
    if state is GroupPairConflictState.DIRECT_CONTRADICTION:
        edge = PropositionGroupContradictionEdge(
            left_group_ref=left.group_ref,
            right_group_ref=right.group_ref,
            contradiction_pair_refs=coverage.direct_contradiction_pair_refs,
            reason_codes=(ConflictReasonCode.DIRECT_PROPOSITION_CONTRADICTION,),
        )
    return coverage, edge


def detect_proposition_conflicts(
    groups: Sequence[PropositionSupportGroup],
    pair_assessments: Sequence[PropositionPairAssessment],
    assertion_bindings: Sequence[PropositionAssertionBinding],
    *,
    corroboration_assessments: Sequence[IndependentCorroborationAssessment] = (),
    counterevidence_eligibility: Sequence[CounterEvidenceEligibilityInput] = (),
) -> PropositionConflictAssessment:
    """Build direct group conflicts and per-group counterevidence from frozen inputs.

    Missing counterevidence eligibility entries remain UNKNOWN. No evidence,
    prose, score, or owner heuristic is used to create an A3 relation.
    """
    ordered_groups = tuple(sorted(groups, key=lambda item: item.group_ref))
    if len({group.group_ref for group in ordered_groups}) != len(ordered_groups):
        raise ValueError("duplicate_proposition_group_ref")
    proposition_to_group: dict[UUID, str] = {}
    for group in ordered_groups:
        for proposition_ref in group.proposition_refs:
            if proposition_ref in proposition_to_group:
                raise ValueError("proposition_belongs_to_multiple_groups")
            proposition_to_group[proposition_ref] = group.group_ref
        for membership in group.evidence_memberships:
            if any(
                link.proposition_id not in group.proposition_refs
                for link in membership.provenance_links
            ):
                raise ValueError("evidence_membership_proposition_not_in_group")

    assertion_by_proposition: dict[UUID, str] = {}
    proposition_by_assertion: dict[str, UUID] = {}
    for binding in assertion_bindings:
        if binding.proposition_ref in assertion_by_proposition:
            raise ValueError("duplicate_proposition_assertion_binding")
        if binding.assertion_ref in proposition_by_assertion:
            raise ValueError("assertion_bound_to_multiple_propositions")
        assertion_by_proposition[binding.proposition_ref] = binding.assertion_ref
        proposition_by_assertion[binding.assertion_ref] = binding.proposition_ref

    assessment_by_pair: dict[tuple[str, str], PropositionPairAssessment] = {}
    inconsistent_pairs: set[tuple[str, str]] = set()
    pair_semantics: dict[tuple[str, str], tuple[object, ...]] = {}
    for assessment in pair_assessments:
        left_prop = proposition_by_assertion.get(assessment.left_assertion_ref)
        right_prop = proposition_by_assertion.get(assessment.right_assertion_ref)
        if left_prop is None or right_prop is None:
            raise ValueError("pair_assessment_assertion_binding_missing")
        if left_prop == right_prop:
            raise ValueError("pair_assessment_requires_distinct_propositions")
        left_group_ref = proposition_to_group.get(left_prop)
        right_group_ref = proposition_to_group.get(right_prop)
        if left_group_ref is None or right_group_ref is None:
            raise ValueError("pair_assessment_proposition_not_in_supplied_groups")
        assertion_refs = sorted((assessment.left_assertion_ref, assessment.right_assertion_ref))
        assertion_pair = (assertion_refs[0], assertion_refs[1])
        if (
            assessment.relation is PropositionRelation.CONTRADICTS
            and assessment.comparability is not ComparabilityStatus.COMPARABLE
        ):
            raise ValueError("a3_contradiction_requires_comparable_pair")
        if (
            assessment.relation
            in {
                PropositionRelation.SAME,
                PropositionRelation.CONTRADICTS,
                PropositionRelation.RELATED,
            }
            and assessment.relation_eligible is not True
        ):
            raise ValueError("a3_affirming_relation_must_be_eligible")
        if left_group_ref == right_group_ref:
            if assessment.relation is PropositionRelation.CONTRADICTS:
                raise ValueError("contradiction_inside_same_support_group")
            if assessment.relation is not PropositionRelation.SAME:
                raise ValueError("same_group_pair_must_have_a3_same_relation")
            continue
        semantic = _a3_semantics(assessment)
        prior_semantic = pair_semantics.get(assertion_pair)
        if prior_semantic is not None and prior_semantic != semantic:
            inconsistent_pairs.add(assertion_pair)
            assessment_by_pair.pop(assertion_pair, None)
            continue
        pair_semantics[assertion_pair] = semantic
        if assertion_pair not in inconsistent_pairs:
            assessment_by_pair[assertion_pair] = assessment

    group_by_ref = {item.group_ref: item for item in ordered_groups}
    assessments_by_group_ref: dict[str, IndependentCorroborationAssessment] = {}
    for corroboration in corroboration_assessments:
        if corroboration.group_ref not in group_by_ref:
            raise ValueError("corroboration_assessment_group_not_supplied")
        if corroboration.group_ref in assessments_by_group_ref:
            raise ValueError("duplicate_corroboration_assessment")
        group_for_corroboration = group_by_ref[corroboration.group_ref]
        _validate_corroboration_input(group_for_corroboration, corroboration)
        assessments_by_group_ref[corroboration.group_ref] = corroboration

    eligibility_by_group: dict[str, dict[UUID, bool | None]] = {}
    for item in counterevidence_eligibility:
        if item.group_ref not in group_by_ref:
            raise ValueError("counterevidence_eligibility_group_not_supplied")
        eligibility_group = group_by_ref[item.group_ref]
        matching_membership = next(
            (
                candidate
                for candidate in eligibility_group.evidence_memberships
                if candidate.evidence_ref == item.evidence_ref
            ),
            None,
        )
        if matching_membership is None or any(
            link.relation is not EvidencePropositionRelation.CONTRADICTS
            for link in matching_membership.provenance_links
        ):
            raise ValueError("counterevidence_eligibility_evidence_mismatch")
        bucket = eligibility_by_group.setdefault(item.group_ref, {})
        if item.evidence_ref in bucket and bucket[item.evidence_ref] != item.accepted:
            raise ValueError("counterevidence_eligibility_input_inconsistent")
        bucket[item.evidence_ref] = item.accepted
    known_group_refs = {group.group_ref for group in ordered_groups}
    if set(eligibility_by_group) - known_group_refs:
        raise ValueError("counterevidence_eligibility_group_not_supplied")

    coverages: list[GroupPairRelationCoverage] = []
    edges: list[PropositionGroupContradictionEdge] = []
    for index, left_group_obj in enumerate(ordered_groups):
        for right_group_obj in ordered_groups[index + 1 :]:
            coverage, edge = _pair_coverage(
                left_group_obj,
                right_group_obj,
                assertion_by_proposition,
                assessment_by_pair,
                inconsistent_pairs,
            )
            coverages.append(coverage)
            if edge is not None:
                edges.append(edge)

    direct_groups: dict[str, set[str]] = {group.group_ref: set() for group in ordered_groups}
    for edge in edges:
        direct_groups[edge.left_group_ref].add(edge.right_group_ref)
        direct_groups[edge.right_group_ref].add(edge.left_group_ref)

    summaries: list[PropositionConflictSummary] = []
    for group_summary_source in ordered_groups:
        support_assessment = assessments_by_group_ref.get(group_summary_source.group_ref)
        if support_assessment is not None:
            membership_refs = {
                item.evidence_ref for item in group_summary_source.evidence_memberships
            }
            if not set(support_assessment.eligible_support_evidence_refs).issubset(membership_refs):
                raise ValueError("corroboration_evidence_not_in_group")
            support_owner_keys: tuple[str, ...] | None = support_assessment.known_support_owner_keys
        else:
            support_owner_keys = None
        counter = _counterevidence_for_group(
            group_summary_source,
            eligibility_by_group.get(group_summary_source.group_ref, {}),
        )
        if support_owner_keys is None:
            overlap = None
            opposition = None
        else:
            support_owners = set(support_owner_keys)
            counter_owners = set(counter.known_counterevidence_owner_keys)
            overlap = tuple(sorted(support_owners & counter_owners))
            opposition = tuple(
                sorted(
                    (
                        OwnerOppositionPair(
                            support_owner_key=support_owner,
                            counterevidence_owner_key=counter_owner,
                        )
                        for support_owner in support_owners
                        for counter_owner in counter_owners
                        if support_owner != counter_owner
                    ),
                    key=lambda item: (item.support_owner_key, item.counterevidence_owner_key),
                )
            )
        summaries.append(
            PropositionConflictSummary(
                group_ref=group_summary_source.group_ref,
                direct_conflicting_group_refs=tuple(
                    sorted(direct_groups[group_summary_source.group_ref])
                ),
                support_owner_keys=support_owner_keys,
                counterevidence=counter,
                support_counter_owner_overlap=overlap,
                cross_owner_opposition_pairs=opposition,
                reason_codes=counter.reason_codes,
            )
        )

    assessment_reasons: set[ConflictReasonCode] = set()
    for coverage in coverages:
        assessment_reasons.update(coverage.reason_codes)
    for summary in summaries:
        assessment_reasons.update(summary.reason_codes)
    return PropositionConflictAssessment(
        group_pair_coverages=tuple(coverages),
        contradiction_edges=tuple(edges),
        group_summaries=tuple(summaries),
        reason_codes=tuple(sorted(assessment_reasons, key=lambda item: item.value)),
    )
