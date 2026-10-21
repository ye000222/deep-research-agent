"""Pure proposition-level verification aggregation for V2.2-C2.

This module consumes frozen B2 corroboration and C1 conflict assessments. It
does not inspect claims, evidence text, databases, or external services, and it
does not recompute any upstream semantic assessment.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.contradiction_detection import (
    CONFLICT_ASSESSMENT_SCHEMA_VERSION,
    GroupPairConflictState,
    PropositionConflictAssessment,
)
from app.domain.corroboration_aggregation import (
    INDEPENDENT_CORROBORATION_SCHEMA_VERSION,
    IndependentCorroborationAssessment,
    OwnerCorroborationStatus,
)
from app.domain.proposition_support_group import (
    PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION,
    PropositionGroupingStatus,
    PropositionSupportGroup,
)

PROPOSITION_VERIFICATION_SCHEMA_VERSION = "proposition-verification-assessment-v1"


class PropositionVerificationStatus(StrEnum):
    """Epistemic state for one proposition support group, never a whole Claim."""

    UNSUPPORTED = "UNSUPPORTED"
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    VERIFIED = "VERIFIED"
    UNKNOWN = "UNKNOWN"


class PositiveSupportState(StrEnum):
    NONE = "NONE"
    PRESENT = "PRESENT"
    UNKNOWN = "UNKNOWN"


class ConflictState(StrEnum):
    CLEAR = "CLEAR"
    ACTIVE = "ACTIVE"
    UNRESOLVED = "UNRESOLVED"


class ConflictScopeCompleteness(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNKNOWN = "UNKNOWN"


class VerificationReasonCode(StrEnum):
    ACCEPTED_SUPPORT_PRESENT = "ACCEPTED_SUPPORT_PRESENT"
    NO_ELIGIBLE_SUPPORT = "NO_ELIGIBLE_SUPPORT"
    SUPPORT_ELIGIBILITY_UNRESOLVED = "SUPPORT_ELIGIBILITY_UNRESOLVED"
    CORROBORATION_ASSESSMENT_MISSING = "CORROBORATION_ASSESSMENT_MISSING"
    CORROBORATION_SCHEMA_MISMATCH = "CORROBORATION_SCHEMA_MISMATCH"
    CONFLICT_ASSESSMENT_MISSING = "CONFLICT_ASSESSMENT_MISSING"
    CONFLICT_SCHEMA_MISMATCH = "CONFLICT_SCHEMA_MISMATCH"
    COUNTEREVIDENCE_PRESENT = "COUNTEREVIDENCE_PRESENT"
    COUNTEREVIDENCE_ELIGIBILITY_UNRESOLVED = "COUNTEREVIDENCE_ELIGIBILITY_UNRESOLVED"
    DIRECT_CONFLICTING_GROUP_SUPPORTED = "DIRECT_CONFLICTING_GROUP_SUPPORTED"
    DIRECT_CONFLICTING_GROUP_UNSUPPORTED = "DIRECT_CONFLICTING_GROUP_UNSUPPORTED"
    DIRECT_CONFLICTING_GROUP_UNRESOLVED = "DIRECT_CONFLICTING_GROUP_UNRESOLVED"
    CONFLICT_COVERAGE_INCOMPLETE = "CONFLICT_COVERAGE_INCOMPLETE"
    CONFLICT_SCOPE_NOT_COMPLETE = "CONFLICT_SCOPE_NOT_COMPLETE"
    OWNER_CORROBORATION_INSUFFICIENT = "OWNER_CORROBORATION_INSUFFICIENT"
    OWNER_CORROBORATION_UNKNOWN = "OWNER_CORROBORATION_UNKNOWN"
    MULTI_OWNER_SUPPORT = "MULTI_OWNER_SUPPORT"
    VERIFICATION_SCOPE_COMPLETE = "VERIFICATION_SCOPE_COMPLETE"
    GROUPING_UNRESOLVED = "GROUPING_UNRESOLVED"
    SEMANTIC_INPUT_INCONSISTENT = "SEMANTIC_INPUT_INCONSISTENT"
    CONFLICT_EDGE_INCONSISTENT = "CONFLICT_EDGE_INCONSISTENT"
    CONFLICT_SCOPE_MISMATCH = "CONFLICT_SCOPE_MISMATCH"


class VerificationConflictScope(BaseModel):
    """Caller-attested, bounded universe for conflict absence evaluation.

    COMPLETE means all relevant proposition groups in the named bounded
    candidate universe have been supplied to C1. It never means internet-wide
    or globally exhaustive truth.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    completeness: ConflictScopeCompleteness = ConflictScopeCompleteness.UNKNOWN
    scope_ref: str | None = Field(default=None, min_length=1, max_length=256)
    candidate_group_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def canonical_scope(self) -> VerificationConflictScope:
        if self.candidate_group_refs != tuple(sorted(set(self.candidate_group_refs))):
            raise ValueError("candidate_group_refs_must_be_unique_and_canonical")
        if any(len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref)
               for ref in self.candidate_group_refs):
            raise ValueError("candidate_group_refs_must_be_sha256_group_refs")
        if self.completeness is ConflictScopeCompleteness.COMPLETE:
            if self.scope_ref is None:
                raise ValueError("complete_conflict_scope_requires_scope_ref")
            if self.target_group_ref not in self.candidate_group_refs:
                raise ValueError("complete_conflict_scope_must_include_target_group")
        return self


class PropositionVerificationAssessment(BaseModel):
    """Auditable C2 result for exactly one B1 proposition support group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_status: PropositionVerificationStatus
    positive_support: PositiveSupportState
    conflict_state: ConflictState
    owner_corroboration_status: OwnerCorroborationStatus
    eligible_support_evidence_refs: tuple[str, ...] = ()
    known_support_owner_keys: tuple[str, ...] = ()
    known_support_owner_count: int = Field(ge=0)
    active_conflict_refs: tuple[str, ...] = ()
    unresolved_conflict_refs: tuple[str, ...] = ()
    active_counterevidence_refs: tuple[str, ...] = ()
    unresolved_counterevidence_refs: tuple[str, ...] = ()
    active_conflicting_group_refs: tuple[str, ...] = ()
    unsupported_conflicting_group_refs: tuple[str, ...] = ()
    unresolved_conflicting_group_refs: tuple[str, ...] = ()
    verification_scope: VerificationConflictScope
    reason_codes: tuple[VerificationReasonCode, ...]
    upstream_reason_codes: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = (
        "OWNER_INDEPENDENCE_IS_PUBLISHER_LEVEL",
        "ELIGIBILITY_INHERITS_EXISTING_ACCEPTANCE_CONTRACT",
        "VERIFIED_IS_BOUNDED_TO_DECLARED_SCOPE",
    )
    corroboration_schema_version: str = INDEPENDENT_CORROBORATION_SCHEMA_VERSION
    conflict_schema_version: str = CONFLICT_ASSESSMENT_SCHEMA_VERSION
    support_group_schema_version: str = PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION
    schema_version: str = PROPOSITION_VERIFICATION_SCHEMA_VERSION

    @model_validator(mode="after")
    def canonical_result(self) -> PropositionVerificationAssessment:
        for refs in (
            self.eligible_support_evidence_refs,
            self.known_support_owner_keys,
            self.active_conflict_refs,
            self.unresolved_conflict_refs,
            self.active_counterevidence_refs,
            self.unresolved_counterevidence_refs,
            self.active_conflicting_group_refs,
            self.unsupported_conflicting_group_refs,
            self.unresolved_conflicting_group_refs,
        ):
            if refs != tuple(sorted(set(refs))):
                raise ValueError("verification_refs_must_be_unique_and_canonical")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("verification_reason_codes_must_be_unique_and_canonical")
        if self.upstream_reason_codes != tuple(sorted(set(self.upstream_reason_codes))):
            raise ValueError("upstream_reason_codes_must_be_unique_and_canonical")
        if self.known_support_owner_keys != tuple(sorted(set(self.known_support_owner_keys))):
            raise ValueError("known_support_owner_keys_must_be_unique_and_canonical")
        return self


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _version_or_inconsistency(
    target_group: PropositionSupportGroup,
    corroboration: IndependentCorroborationAssessment | None,
    conflicts: PropositionConflictAssessment | None,
) -> VerificationReasonCode | None:
    if target_group.schema_version != PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION:
        return VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
    if corroboration is not None:
        if corroboration.group_ref != target_group.group_ref:
            raise ValueError("corroboration_assessment_group_mismatch")
        if corroboration.schema_version != INDEPENDENT_CORROBORATION_SCHEMA_VERSION:
            return VerificationReasonCode.CORROBORATION_SCHEMA_MISMATCH
    if conflicts is not None and conflicts.schema_version != CONFLICT_ASSESSMENT_SCHEMA_VERSION:
        return VerificationReasonCode.CONFLICT_SCHEMA_MISMATCH
    return None


def aggregate_proposition_verification(
    target_group: PropositionSupportGroup,
    target_corroboration: IndependentCorroborationAssessment | None,
    conflict_assessment: PropositionConflictAssessment | None,
    opposing_corroboration_assessments: tuple[IndependentCorroborationAssessment, ...] = (),
    conflict_scope: VerificationConflictScope | None = None,
) -> PropositionVerificationAssessment:
    """Aggregate only frozen proposition-level B2/C1 facts, deterministically.

    Structural identity mismatches are rejected. Missing/incomplete evidence is
    represented as UNKNOWN or UNRESOLVED and never repaired here.
    """

    if conflict_scope is not None and conflict_scope.target_group_ref != target_group.group_ref:
        raise ValueError("conflict_scope_target_group_mismatch")
    scope = conflict_scope or VerificationConflictScope(target_group_ref=target_group.group_ref)
    reasons: set[VerificationReasonCode] = set()
    upstream_reasons: set[str] = set()
    inconsistent = _version_or_inconsistency(
        target_group, target_corroboration, conflict_assessment
    )
    if inconsistent is not None:
        reasons.add(inconsistent)

    if target_group.grouping_status is PropositionGroupingStatus.UNRESOLVED_MEMBERSHIP:
        reasons.add(VerificationReasonCode.GROUPING_UNRESOLVED)

    if target_corroboration is None:
        positive_support = PositiveSupportState.UNKNOWN
        owner_status = OwnerCorroborationStatus.UNKNOWN
        owner_count = 0
        owner_keys: tuple[str, ...] = ()
        eligible_support_refs: tuple[str, ...] = ()
        reasons.add(VerificationReasonCode.CORROBORATION_ASSESSMENT_MISSING)
    else:
        owner_status = target_corroboration.owner_corroboration_status
        owner_count = target_corroboration.known_owner_count
        owner_keys = target_corroboration.known_support_owner_keys
        eligible_support_refs = tuple(
            sorted(str(ref) for ref in target_corroboration.eligible_support_evidence_refs)
        )
        upstream_reasons.update(reason.value for reason in target_corroboration.reason_codes)
        if target_corroboration.eligible_support_evidence_count > 0:
            positive_support = PositiveSupportState.PRESENT
            reasons.add(VerificationReasonCode.ACCEPTED_SUPPORT_PRESENT)
        elif target_corroboration.unknown_eligibility_support_evidence_refs:
            positive_support = PositiveSupportState.UNKNOWN
            reasons.add(VerificationReasonCode.SUPPORT_ELIGIBILITY_UNRESOLVED)
        elif (
            target_corroboration.owner_corroboration_status
            is OwnerCorroborationStatus.NO_ELIGIBLE_SUPPORT
        ):
            positive_support = PositiveSupportState.NONE
            reasons.add(VerificationReasonCode.NO_ELIGIBLE_SUPPORT)
        else:
            positive_support = PositiveSupportState.UNKNOWN
            reasons.add(VerificationReasonCode.SUPPORT_ELIGIBILITY_UNRESOLVED)

    conflict_state = ConflictState.CLEAR
    active_counter_refs: set[str] = set()
    unresolved_counter_refs: set[str] = set()
    active_group_refs: set[str] = set()
    unsupported_group_refs: set[str] = set()
    unresolved_group_refs: set[str] = set()

    if conflict_assessment is None:
        conflict_state = ConflictState.UNRESOLVED
        reasons.add(VerificationReasonCode.CONFLICT_ASSESSMENT_MISSING)
    else:
        summaries = {item.group_ref: item for item in conflict_assessment.group_summaries}
        if len(summaries) != len(conflict_assessment.group_summaries):
            inconsistent = VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
        if any(item.schema_version != "proposition-conflict-summary-v1"
               or item.counterevidence.schema_version != "counterevidence-assessment-v1"
               for item in conflict_assessment.group_summaries):
            inconsistent = VerificationReasonCode.CONFLICT_SCHEMA_MISMATCH
        if any(item.schema_version != "group-pair-relation-coverage-v1"
               for item in conflict_assessment.group_pair_coverages) or any(
            item.schema_version != "proposition-group-contradiction-edge-v1"
            for item in conflict_assessment.contradiction_edges
        ):
            inconsistent = VerificationReasonCode.CONFLICT_SCHEMA_MISMATCH
        target_summary = summaries.get(target_group.group_ref)
        if target_summary is None:
            conflict_state = ConflictState.UNRESOLVED
            reasons.add(VerificationReasonCode.CONFLICT_ASSESSMENT_MISSING)
        else:
            counter = target_summary.counterevidence
            upstream_reasons.update(reason.value for reason in target_summary.reason_codes)
            upstream_reasons.update(reason.value for reason in counter.reason_codes)
            active_counter_refs.update(str(ref) for ref in counter.eligible_counterevidence_refs)
            unresolved_counter_refs.update(
                str(ref) for ref in counter.unknown_eligibility_counterevidence_refs
            )
            if counter.group_ref != target_group.group_ref:
                inconsistent = VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
            elif counter.eligible_counterevidence_count > 0:
                reasons.add(VerificationReasonCode.COUNTEREVIDENCE_PRESENT)
            if counter.unknown_eligibility_counterevidence_refs:
                reasons.add(VerificationReasonCode.COUNTEREVIDENCE_ELIGIBILITY_UNRESOLVED)

            edge_neighbors: set[str] = set()
            for edge in conflict_assessment.contradiction_edges:
                if target_group.group_ref in (edge.left_group_ref, edge.right_group_ref):
                    edge_neighbors.add(
                        edge.right_group_ref
                        if edge.left_group_ref == target_group.group_ref
                        else edge.left_group_ref
                    )
            if edge_neighbors != set(target_summary.direct_conflicting_group_refs):
                inconsistent = VerificationReasonCode.CONFLICT_EDGE_INCONSISTENT

            summaries_for_groups = set(summaries)
            if edge_neighbors - summaries_for_groups:
                inconsistent = VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
            coverage_by_pair = {
                _pair_key(item.left_group_ref, item.right_group_ref): item
                for item in conflict_assessment.group_pair_coverages
            }
            edge_pairs = {
                _pair_key(edge.left_group_ref, edge.right_group_ref)
                for edge in conflict_assessment.contradiction_edges
            }
            if scope.completeness is ConflictScopeCompleteness.COMPLETE:
                if set(scope.candidate_group_refs) != summaries_for_groups:
                    reasons.add(VerificationReasonCode.CONFLICT_SCOPE_MISMATCH)
                    conflict_state = ConflictState.UNRESOLVED
                expected_pairs = {
                    _pair_key(target_group.group_ref, other_group_ref)
                    for other_group_ref in scope.candidate_group_refs
                    if other_group_ref != target_group.group_ref
                }
                if not expected_pairs.issubset(coverage_by_pair):
                    conflict_state = ConflictState.UNRESOLVED
                    reasons.add(VerificationReasonCode.CONFLICT_COVERAGE_INCOMPLETE)
            else:
                conflict_state = ConflictState.UNRESOLVED
                reasons.add(VerificationReasonCode.CONFLICT_SCOPE_NOT_COMPLETE)

            for pair, coverage in coverage_by_pair.items():
                if target_group.group_ref not in pair:
                    continue
                if coverage.coverage_status in {
                    GroupPairConflictState.GROUPING_INCONSISTENCY,
                    GroupPairConflictState.PAIR_RELATION_INCONSISTENT,
                }:
                    inconsistent = VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
                elif coverage.coverage_status in {
                    GroupPairConflictState.UNKNOWN,
                    GroupPairConflictState.GROUPING_UNRESOLVED,
                }:
                    conflict_state = ConflictState.UNRESOLVED
                    reasons.add(VerificationReasonCode.CONFLICT_COVERAGE_INCOMPLETE)
                elif (
                    coverage.coverage_status is GroupPairConflictState.DIRECT_CONTRADICTION
                    and pair not in edge_pairs
                ) or (
                    coverage.coverage_status is GroupPairConflictState.NO_DIRECT_CONTRADICTION
                    and pair in edge_pairs
                ):
                    inconsistent = VerificationReasonCode.CONFLICT_EDGE_INCONSISTENT

            if edge_neighbors:
                assessments_by_group = {
                    item.group_ref: item for item in opposing_corroboration_assessments
                }
                if len(assessments_by_group) != len(opposing_corroboration_assessments):
                    inconsistent = VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
                if set(assessments_by_group) - edge_neighbors:
                    raise ValueError("opposing_corroboration_not_for_direct_conflict")
                for opposing_ref in sorted(edge_neighbors):
                    opposing = assessments_by_group.get(opposing_ref)
                    if opposing is None:
                        unresolved_group_refs.add(opposing_ref)
                        reasons.add(VerificationReasonCode.DIRECT_CONFLICTING_GROUP_UNRESOLVED)
                        continue
                    if opposing.schema_version != INDEPENDENT_CORROBORATION_SCHEMA_VERSION:
                        inconsistent = VerificationReasonCode.CORROBORATION_SCHEMA_MISMATCH
                    if opposing.group_ref != opposing_ref:
                        raise ValueError("opposing_corroboration_group_mismatch")
                    upstream_reasons.update(reason.value for reason in opposing.reason_codes)
                    if opposing.eligible_support_evidence_count > 0:
                        active_group_refs.add(opposing_ref)
                        reasons.add(VerificationReasonCode.DIRECT_CONFLICTING_GROUP_SUPPORTED)
                    elif opposing.unknown_eligibility_support_evidence_refs:
                        unresolved_group_refs.add(opposing_ref)
                        reasons.add(VerificationReasonCode.DIRECT_CONFLICTING_GROUP_UNRESOLVED)
                    elif (
                        opposing.owner_corroboration_status
                        is OwnerCorroborationStatus.NO_ELIGIBLE_SUPPORT
                    ):
                        unsupported_group_refs.add(opposing_ref)
                        reasons.add(VerificationReasonCode.DIRECT_CONFLICTING_GROUP_UNSUPPORTED)
                    else:
                        unresolved_group_refs.add(opposing_ref)
                        reasons.add(VerificationReasonCode.DIRECT_CONFLICTING_GROUP_UNRESOLVED)

            if inconsistent is not None:
                reasons.add(inconsistent)
                conflict_state = ConflictState.UNRESOLVED
            elif (
                active_group_refs
                or (
                    positive_support is PositiveSupportState.PRESENT
                    and active_counter_refs
                )
            ):
                conflict_state = ConflictState.ACTIVE
            elif unresolved_counter_refs or unresolved_group_refs:
                conflict_state = ConflictState.UNRESOLVED

    if inconsistent is not None or positive_support is PositiveSupportState.UNKNOWN:
        verification_status = PropositionVerificationStatus.UNKNOWN
    elif positive_support is PositiveSupportState.NONE:
        verification_status = PropositionVerificationStatus.UNSUPPORTED
    elif conflict_state is ConflictState.ACTIVE:
        verification_status = PropositionVerificationStatus.CONFLICTED
    elif (
        owner_status is OwnerCorroborationStatus.MULTI_KNOWN_OWNERS
        and owner_count >= 2
        and conflict_state is ConflictState.CLEAR
        and scope.completeness is ConflictScopeCompleteness.COMPLETE
        and VerificationReasonCode.CONFLICT_SCOPE_MISMATCH not in reasons
        and VerificationReasonCode.GROUPING_UNRESOLVED not in reasons
    ):
        verification_status = PropositionVerificationStatus.VERIFIED
        reasons.add(VerificationReasonCode.MULTI_OWNER_SUPPORT)
        reasons.add(VerificationReasonCode.VERIFICATION_SCOPE_COMPLETE)
    else:
        verification_status = PropositionVerificationStatus.SUPPORTED
        if owner_status is OwnerCorroborationStatus.UNKNOWN:
            reasons.add(VerificationReasonCode.OWNER_CORROBORATION_UNKNOWN)
        elif owner_status is not OwnerCorroborationStatus.MULTI_KNOWN_OWNERS:
            reasons.add(VerificationReasonCode.OWNER_CORROBORATION_INSUFFICIENT)

    if tuple(sorted(eligible_support_refs)) != tuple(sorted(set(eligible_support_refs))):
        inconsistent = VerificationReasonCode.SEMANTIC_INPUT_INCONSISTENT
        reasons.add(inconsistent)
        verification_status = PropositionVerificationStatus.UNKNOWN

    active_conflict_refs = tuple(
        sorted(
            {f"EVIDENCE:{ref}" for ref in active_counter_refs}
            | {f"PROPOSITION_GROUP:{ref}" for ref in active_group_refs}
        )
    )
    unresolved_conflict_refs = tuple(
        sorted(
            {f"EVIDENCE:{ref}" for ref in unresolved_counter_refs}
            | {f"PROPOSITION_GROUP:{ref}" for ref in unresolved_group_refs}
        )
    )

    return PropositionVerificationAssessment(
        group_ref=target_group.group_ref,
        verification_status=verification_status,
        positive_support=positive_support,
        conflict_state=conflict_state,
        owner_corroboration_status=owner_status,
        eligible_support_evidence_refs=eligible_support_refs,
        known_support_owner_keys=owner_keys,
        known_support_owner_count=owner_count,
        active_conflict_refs=active_conflict_refs,
        unresolved_conflict_refs=unresolved_conflict_refs,
        active_counterevidence_refs=tuple(sorted(active_counter_refs)),
        unresolved_counterevidence_refs=tuple(sorted(unresolved_counter_refs)),
        active_conflicting_group_refs=tuple(sorted(active_group_refs)),
        unsupported_conflicting_group_refs=tuple(sorted(unsupported_group_refs)),
        unresolved_conflicting_group_refs=tuple(sorted(unresolved_group_refs)),
        verification_scope=scope,
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
        upstream_reason_codes=tuple(sorted(upstream_reasons)),
    )
