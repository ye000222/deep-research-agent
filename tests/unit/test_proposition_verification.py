from __future__ import annotations

from itertools import combinations
from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    EvidencePropositionRelation,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.contradiction_detection import (
    CounterEvidenceEligibilityInput,
    detect_proposition_conflicts,
)
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibilityInput,
    IndependentCorroborationAssessment,
    aggregate_independent_corroboration,
)
from app.domain.proposition_support_group import (
    PropositionAssertionBinding,
    PropositionEvidenceMembership,
    PropositionGroupingStatus,
    PropositionSupportGroup,
)
from app.domain.proposition_verification import (
    ConflictScopeCompleteness,
    ConflictState,
    PropositionVerificationStatus,
    VerificationConflictScope,
    aggregate_proposition_verification,
)
from app.domain.verification_assertion import (
    ComparabilityStatus,
    ComparisonReasonCode,
    PropositionPairAssessment,
)

CLAIM = UUID("20000000-0000-0000-0000-000000000001")
RUN = UUID("30000000-0000-0000-0000-000000000001")
TARGET_PROP = UUID("10000000-0000-0000-0000-000000000001")
OPPOSING_PROP = UUID("10000000-0000-0000-0000-000000000002")
TARGET_REF = "a" * 64
OPPOSING_REF = "b" * 64


def _group(
    group_ref: str,
    proposition_ref: UUID,
    evidence: tuple[tuple[int, str | None, EvidencePropositionRelation], ...] = (),
) -> PropositionSupportGroup:
    memberships = tuple(
        PropositionEvidenceMembership(
            evidence_ref=UUID(int=ordinal),
            provenance_links=(
                PropositionEvidenceLink(
                    proposition_id=proposition_ref,
                    evidence_id=UUID(int=ordinal),
                    claim_id=CLAIM,
                    relation=relation,
                    source_id=UUID(int=ordinal + 1000),
                    source_owner_key=owner,
                    quote_reference="quote",
                    span_start=0,
                    span_end=5,
                    method="fixture",
                    run_id=RUN,
                    question_id="q1",
                    dimension_key="d1",
                ),
            ),
        )
        for ordinal, owner, relation in evidence
    )
    return PropositionSupportGroup(
        group_ref=group_ref,
        proposition_refs=(proposition_ref,),
        evidence_memberships=tuple(sorted(memberships, key=lambda item: str(item.evidence_ref))),
        grouping_status=PropositionGroupingStatus.SINGLETON,
    )


def _corroboration(
    group: PropositionSupportGroup,
    accepted: dict[int, bool | None] | None = None,
) -> IndependentCorroborationAssessment:
    if accepted is None:
        accepted = {
            membership.evidence_ref.int: True
            for membership in group.evidence_memberships
            if membership.provenance_links[0].relation
            is EvidencePropositionRelation.SUPPORTS
        }
    inputs = tuple(
        CorroborationEvidenceEligibilityInput(
            group_ref=group.group_ref,
            evidence_ref=UUID(int=ordinal),
            accepted=value,
        )
        for ordinal, value in accepted.items()
    )
    return aggregate_independent_corroboration(group, inputs)


def _scope(*groups: PropositionSupportGroup) -> VerificationConflictScope:
    return VerificationConflictScope(
        target_group_ref=groups[0].group_ref,
        completeness=ConflictScopeCompleteness.COMPLETE,
        scope_ref="bounded-fixture-universe",
        candidate_group_refs=tuple(sorted(group.group_ref for group in groups)),
    )


def _conflicts(
    groups: tuple[PropositionSupportGroup, ...],
    *,
    contradictory: bool = False,
    unknown_relation: bool = False,
    counter_acceptance: dict[int, bool | None] | None = None,
) :
    bindings = tuple(
        PropositionAssertionBinding(
            proposition_ref=group.proposition_refs[0], assertion_ref=f"assertion-{index}"
        )
        for index, group in enumerate(groups, start=1)
    )
    pair_assessments = tuple(
        PropositionPairAssessment(
            left_assertion_ref=f"assertion-{left_index}",
            right_assertion_ref=f"assertion-{right_index}",
            comparability=(
                ComparabilityStatus.UNKNOWN
                if unknown_relation
                else ComparabilityStatus.COMPARABLE
            ),
            relation=(
                PropositionRelation.UNKNOWN
                if unknown_relation
                else PropositionRelation.CONTRADICTS
                if contradictory and left_index == 1
                else PropositionRelation.RELATED
            ),
            relation_eligible=None if unknown_relation else True,
            reason_codes=(
                (ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION,)
                if not unknown_relation and not (contradictory and left_index == 1)
                else ()
            ),
        )
        for left_index, right_index in combinations(range(1, len(groups) + 1), 2)
    )
    counter_inputs = tuple(
        CounterEvidenceEligibilityInput(
            group_ref=groups[0].group_ref,
            evidence_ref=UUID(int=ordinal),
            accepted=accepted,
        )
        for ordinal, accepted in (counter_acceptance or {}).items()
    )
    return detect_proposition_conflicts(
        groups,
        pair_assessments,
        bindings,
        counterevidence_eligibility=counter_inputs,
    )


def _evaluate(
    group: PropositionSupportGroup,
    corroboration: IndependentCorroborationAssessment | None = None,
    *,
    opposing: tuple[PropositionSupportGroup, ...] = (),
    conflicts=None,
    scope: VerificationConflictScope | None = None,
):
    all_groups = (group, *opposing)
    return aggregate_proposition_verification(
        group,
        corroboration,
        conflicts if conflicts is not None else _conflicts(all_groups),
        tuple(_corroboration(item) for item in opposing),
        scope if scope is not None else _scope(*all_groups),
    )


def _supported_group(owners: tuple[str | None, ...] = ("owner-a",)) -> PropositionSupportGroup:
    return _group(
        TARGET_REF,
        TARGET_PROP,
        tuple(
            (index + 1, owner, EvidencePropositionRelation.SUPPORTS)
            for index, owner in enumerate(owners)
        ),
    )


def test_unsupported_is_not_conflicted_without_positive_support() -> None:
    group = _group(
        TARGET_REF,
        TARGET_PROP,
        ((1, "owner-counter", EvidencePropositionRelation.CONTRADICTS),),
    )
    corroboration = _corroboration(group)
    result = _evaluate(group, corroboration)
    assert result.verification_status is PropositionVerificationStatus.UNSUPPORTED
    assert result.positive_support.value == "NONE"


def test_unknown_support_has_precedence_over_active_counterevidence() -> None:
    group = _group(
        TARGET_REF,
        TARGET_PROP,
        (
            (1, "owner-support", EvidencePropositionRelation.SUPPORTS),
            (2, "owner-counter", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    result = _evaluate(group, _corroboration(group, {1: None}))
    assert result.verification_status is PropositionVerificationStatus.UNKNOWN


def test_unknown_target_support_with_supported_opposition_remains_unknown_and_active() -> None:
    target = _group(
        TARGET_REF,
        TARGET_PROP,
        ((1, "owner-a", EvidencePropositionRelation.SUPPORTS),),
    )
    opposing = _group(
        OPPOSING_REF,
        OPPOSING_PROP,
        ((2, "owner-b", EvidencePropositionRelation.SUPPORTS),),
    )
    conflicts = _conflicts((target, opposing), contradictory=True)
    result = aggregate_proposition_verification(
        target,
        _corroboration(target, {1: None}),
        conflicts,
        (_corroboration(opposing),),
        _scope(target, opposing),
    )
    assert result.verification_status is PropositionVerificationStatus.UNKNOWN
    assert result.conflict_state is ConflictState.ACTIVE


def test_supported_with_one_or_unknown_owner_and_incomplete_scope() -> None:
    for owners in (("owner-a",), (None,)):
        group = _supported_group(owners)
        result = aggregate_proposition_verification(
            group,
            _corroboration(group),
            _conflicts((group,)),
            conflict_scope=VerificationConflictScope(
                target_group_ref=TARGET_REF,
                completeness=ConflictScopeCompleteness.INCOMPLETE,
                scope_ref="bounded-fixture-universe",
                candidate_group_refs=(TARGET_REF,),
            ),
        )
        assert result.verification_status is PropositionVerificationStatus.SUPPORTED


@pytest.mark.parametrize("owner_count", [2, 3])
def test_verified_requires_multiple_known_owners_and_complete_clear_scope(owner_count: int) -> None:
    group = _supported_group(tuple(f"owner-{index}" for index in range(owner_count)))
    result = _evaluate(group, _corroboration(group))
    assert result.verification_status is PropositionVerificationStatus.VERIFIED
    assert result.conflict_state is ConflictState.CLEAR


def test_extra_unknown_owner_does_not_erase_established_multi_owner_status() -> None:
    group = _supported_group(("owner-a", "owner-b", None))
    result = _evaluate(group, _corroboration(group))
    assert result.verification_status is PropositionVerificationStatus.VERIFIED


def test_unknown_conflict_scope_prevents_verified_but_preserves_support() -> None:
    group = _supported_group(("owner-a", "owner-b"))
    result = aggregate_proposition_verification(
        group,
        _corroboration(group),
        _conflicts((group,)),
        conflict_scope=VerificationConflictScope(target_group_ref=TARGET_REF),
    )
    assert result.verification_status is PropositionVerificationStatus.SUPPORTED
    assert result.conflict_state is ConflictState.UNRESOLVED


def test_c1_unknown_pair_coverage_preserves_support_but_blocks_verified() -> None:
    target = _supported_group(("owner-a", "owner-b"))
    other = _group(OPPOSING_REF, OPPOSING_PROP)
    conflicts = _conflicts((target, other), unknown_relation=True)
    result = aggregate_proposition_verification(
        target,
        _corroboration(target),
        conflicts,
        conflict_scope=_scope(target, other),
    )
    assert result.verification_status is PropositionVerificationStatus.SUPPORTED
    assert result.conflict_state is ConflictState.UNRESOLVED


def test_unknown_eligibility_does_not_erase_established_multi_owner_support() -> None:
    group = _supported_group(("owner-a", "owner-b", "owner-c"))
    corroboration = _corroboration(group, {1: True, 2: True, 3: None})
    result = _evaluate(group, corroboration)
    assert result.verification_status is PropositionVerificationStatus.VERIFIED


def test_supported_plus_eligible_counterevidence_is_conflicted() -> None:
    group = _group(
        TARGET_REF,
        TARGET_PROP,
        (
            (1, "owner-support", EvidencePropositionRelation.SUPPORTS),
            (2, "owner-counter", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    conflicts = _conflicts((group,), counter_acceptance={2: True})
    result = _evaluate(group, _corroboration(group), conflicts=conflicts)
    assert result.verification_status is PropositionVerificationStatus.CONFLICTED
    assert result.conflict_state is ConflictState.ACTIVE


def test_supported_plus_unresolved_counterevidence_remains_supported() -> None:
    group = _group(
        TARGET_REF,
        TARGET_PROP,
        (
            (1, "owner-support", EvidencePropositionRelation.SUPPORTS),
            (2, "owner-counter", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    conflicts = _conflicts((group,), counter_acceptance={2: None})
    result = _evaluate(group, _corroboration(group), conflicts=conflicts)
    assert result.verification_status is PropositionVerificationStatus.SUPPORTED
    assert result.conflict_state is ConflictState.UNRESOLVED


def test_active_and_unresolved_counterevidence_is_conflicted() -> None:
    group = _group(
        TARGET_REF,
        TARGET_PROP,
        (
            (1, "owner-support", EvidencePropositionRelation.SUPPORTS),
            (2, "owner-counter-a", EvidencePropositionRelation.CONTRADICTS),
            (3, "owner-counter-b", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    conflicts = _conflicts((group,), counter_acceptance={2: True, 3: None})
    result = _evaluate(group, _corroboration(group), conflicts=conflicts)
    assert result.verification_status is PropositionVerificationStatus.CONFLICTED
    assert result.conflict_state is ConflictState.ACTIVE
    assert result.unresolved_counterevidence_refs == (str(UUID(int=3)),)


def test_ineligible_counterevidence_does_not_block_verified() -> None:
    group = _group(
        TARGET_REF,
        TARGET_PROP,
        (
            (1, "owner-a", EvidencePropositionRelation.SUPPORTS),
            (2, "owner-b", EvidencePropositionRelation.SUPPORTS),
            (3, "owner-c", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    conflicts = _conflicts((group,), counter_acceptance={3: False})
    result = _evaluate(group, _corroboration(group), conflicts=conflicts)
    assert result.verification_status is PropositionVerificationStatus.VERIFIED


def test_direct_opposing_group_is_active_only_with_eligible_support() -> None:
    target = _supported_group()
    opposing = _group(
        OPPOSING_REF,
        OPPOSING_PROP,
        ((3, "owner-b", EvidencePropositionRelation.SUPPORTS),),
    )
    conflicts = _conflicts((target, opposing), contradictory=True)
    result = _evaluate(target, _corroboration(target), opposing=(opposing,), conflicts=conflicts)
    assert result.verification_status is PropositionVerificationStatus.CONFLICTED
    assert result.active_conflicting_group_refs == (OPPOSING_REF,)


def test_unsupported_opposing_group_does_not_block_verified() -> None:
    target = _supported_group(("owner-a", "owner-b"))
    opposing = _group(
        OPPOSING_REF,
        OPPOSING_PROP,
        ((3, "owner-c", EvidencePropositionRelation.SUPPORTS),),
    )
    conflicts = _conflicts((target, opposing), contradictory=True)
    opposing_corroboration = _corroboration(opposing, {3: False})
    result = aggregate_proposition_verification(
        target,
        _corroboration(target),
        conflicts,
        (opposing_corroboration,),
        _scope(target, opposing),
    )
    assert result.verification_status is PropositionVerificationStatus.VERIFIED
    assert result.unsupported_conflicting_group_refs == (OPPOSING_REF,)


def test_active_opposing_group_precedes_another_unresolved_opposing_group() -> None:
    third_ref = "c" * 64
    third_prop = UUID("10000000-0000-0000-0000-000000000003")
    target = _supported_group()
    active = _group(
        OPPOSING_REF,
        OPPOSING_PROP,
        ((3, "owner-b", EvidencePropositionRelation.SUPPORTS),),
    )
    unresolved = _group(
        third_ref,
        third_prop,
        ((4, "owner-c", EvidencePropositionRelation.SUPPORTS),),
    )
    conflicts = _conflicts((target, active, unresolved), contradictory=True)
    result = aggregate_proposition_verification(
        target,
        _corroboration(target),
        conflicts,
        (
            _corroboration(active),
            _corroboration(unresolved, {4: None}),
        ),
        _scope(target, active, unresolved),
    )
    assert result.verification_status is PropositionVerificationStatus.CONFLICTED
    assert result.conflict_state is ConflictState.ACTIVE
    assert result.active_conflicting_group_refs == (OPPOSING_REF,)
    assert result.unresolved_conflicting_group_refs == (third_ref,)


def test_missing_opposing_assessment_keeps_known_support_not_verified() -> None:
    target = _supported_group(("owner-a", "owner-b"))
    opposing = _group(OPPOSING_REF, OPPOSING_PROP)
    conflicts = _conflicts((target, opposing), contradictory=True)
    result = aggregate_proposition_verification(
        target, _corroboration(target), conflicts, (), _scope(target, opposing)
    )
    assert result.verification_status is PropositionVerificationStatus.SUPPORTED
    assert result.unresolved_conflicting_group_refs == (OPPOSING_REF,)


def test_unsupported_target_stays_unsupported_despite_supported_opposing_group() -> None:
    target = _group(TARGET_REF, TARGET_PROP)
    opposing = _group(
        OPPOSING_REF,
        OPPOSING_PROP,
        ((3, "owner-b", EvidencePropositionRelation.SUPPORTS),),
    )
    conflicts = _conflicts((target, opposing), contradictory=True)
    result = _evaluate(target, _corroboration(target), opposing=(opposing,), conflicts=conflicts)
    assert result.verification_status is PropositionVerificationStatus.UNSUPPORTED
    assert result.conflict_state is ConflictState.ACTIVE
    assert result.active_conflicting_group_refs == (OPPOSING_REF,)


def test_wrong_scope_target_and_wrong_corroboration_group_are_rejected() -> None:
    group = _supported_group()
    wrong_scope = VerificationConflictScope(target_group_ref=OPPOSING_REF)
    with pytest.raises(ValueError, match="conflict_scope_target_group_mismatch"):
        aggregate_proposition_verification(
            group, _corroboration(group), None, conflict_scope=wrong_scope
        )
    wrong = _corroboration(group).model_copy(update={"group_ref": OPPOSING_REF})
    with pytest.raises(ValueError, match="corroboration_assessment_group_mismatch"):
        aggregate_proposition_verification(group, wrong, None)


def test_missing_assessments_fail_closed_to_unknown() -> None:
    group = _supported_group()
    result = aggregate_proposition_verification(group, None, None)
    assert result.verification_status is PropositionVerificationStatus.UNKNOWN


def test_complete_scope_membership_mismatch_prevents_verified() -> None:
    target = _supported_group(("owner-a", "owner-b"))
    other = _group(OPPOSING_REF, OPPOSING_PROP)
    conflicts = _conflicts((target, other))
    incomplete_attestation = VerificationConflictScope(
        target_group_ref=TARGET_REF,
        completeness=ConflictScopeCompleteness.COMPLETE,
        scope_ref="incorrectly-truncated-universe",
        candidate_group_refs=(TARGET_REF,),
    )
    result = aggregate_proposition_verification(
        target,
        _corroboration(target),
        conflicts,
        conflict_scope=incomplete_attestation,
    )
    assert result.verification_status is PropositionVerificationStatus.SUPPORTED
    assert result.conflict_state is ConflictState.UNRESOLVED


def test_c1_pair_semantic_inconsistency_fails_closed_to_unknown() -> None:
    target = _supported_group(("owner-a", "owner-b"))
    other = _group(OPPOSING_REF, OPPOSING_PROP)
    conflicts = _conflicts((target, other))
    inconsistent_coverage = conflicts.group_pair_coverages[0].model_copy(
        update={"coverage_status": "PAIR_RELATION_INCONSISTENT"}
    )
    inconsistent_assessment = conflicts.model_copy(
        update={"group_pair_coverages": (inconsistent_coverage,)}
    )
    result = aggregate_proposition_verification(
        target,
        _corroboration(target),
        inconsistent_assessment,
        conflict_scope=_scope(target, other),
    )
    assert result.verification_status is PropositionVerificationStatus.UNKNOWN


def test_result_is_deterministic_and_does_not_mutate_upstream_inputs() -> None:
    group = _supported_group(("owner-a", "owner-b"))
    corroboration = _corroboration(group)
    conflicts = _conflicts((group,))
    before = (
        group.model_dump(mode="json"),
        corroboration.model_dump(mode="json"),
        conflicts.model_dump(mode="json"),
    )
    first = aggregate_proposition_verification(
        group, corroboration, conflicts, conflict_scope=_scope(group)
    )
    second = aggregate_proposition_verification(
        group, corroboration, conflicts, conflict_scope=_scope(group)
    )
    after = (
        group.model_dump(mode="json"),
        corroboration.model_dump(mode="json"),
        conflicts.model_dump(mode="json"),
    )
    assert first == second
    assert before == after
    assert "claim_id" not in first.model_dump()
    assert "claim_status" not in first.model_dump()
