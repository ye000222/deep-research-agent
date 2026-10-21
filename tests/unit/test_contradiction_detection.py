from __future__ import annotations

from itertools import permutations
from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    EvidencePropositionRelation,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.contradiction_detection import (
    CounterEvidenceEligibilityInput,
    GroupPairConflictState,
    PropositionConflictAssessment,
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
from app.domain.verification_assertion import (
    ComparabilityStatus,
    ComparisonReasonCode,
    PropositionPairAssessment,
)

GROUP_A = "a" * 64
GROUP_B = "b" * 64
CLAIM_ID = UUID("20000000-0000-0000-0000-000000000001")
RUN_ID = UUID("30000000-0000-0000-0000-000000000001")


def _uuid(prefix: int, ordinal: int) -> UUID:
    return UUID(f"{prefix:08d}-0000-0000-0000-{ordinal:012d}")


def _group(
    group_ref: str,
    propositions: tuple[UUID, ...],
    evidence: tuple[tuple[UUID, UUID, str | None, EvidencePropositionRelation], ...] = (),
) -> PropositionSupportGroup:
    memberships = tuple(
        PropositionEvidenceMembership(
            evidence_ref=evidence_ref,
            provenance_links=(
                PropositionEvidenceLink(
                    proposition_id=proposition_ref,
                    evidence_id=evidence_ref,
                    claim_id=CLAIM_ID,
                    relation=direction,
                    source_id=UUID(int=evidence_ref.int + 1),
                    source_owner_key=owner,
                    quote_reference="quote",
                    span_start=0,
                    span_end=5,
                    method="fixture",
                    run_id=RUN_ID,
                    question_id="q1",
                    dimension_key="d1",
                ),
            ),
        )
        for evidence_ref, proposition_ref, owner, direction in evidence
    )
    return PropositionSupportGroup(
        group_ref=group_ref,
        proposition_refs=tuple(sorted(propositions, key=str)),
        evidence_memberships=tuple(sorted(memberships, key=lambda item: str(item.evidence_ref))),
        grouping_status=(
            PropositionGroupingStatus.GROUPED
            if len(propositions) > 1
            else PropositionGroupingStatus.SINGLETON
        ),
    )


def _binding(proposition: UUID, ordinal: int) -> PropositionAssertionBinding:
    return PropositionAssertionBinding(
        proposition_ref=proposition,
        assertion_ref=f"assertion-{ordinal}",
    )


def _pair(
    left: str,
    right: str,
    relation: PropositionRelation,
    comparability: ComparabilityStatus = ComparabilityStatus.COMPARABLE,
) -> PropositionPairAssessment:
    return PropositionPairAssessment(
        left_assertion_ref=left,
        right_assertion_ref=right,
        comparability=comparability,
        relation=relation,
        relation_eligible=relation is not PropositionRelation.UNKNOWN,
        reason_codes=(
            (ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION,)
            if relation is PropositionRelation.RELATED
            else ()
        ),
    )


def _simple_inputs(
    evidence_a: tuple[tuple[UUID, UUID, str | None, EvidencePropositionRelation], ...] = (),
    evidence_b: tuple[tuple[UUID, UUID, str | None, EvidencePropositionRelation], ...] = (),
) -> tuple[
    PropositionSupportGroup, PropositionSupportGroup, tuple[PropositionAssertionBinding, ...]
]:
    prop_a, prop_b = _uuid(100, 1), _uuid(100, 2)
    group_a = _group(GROUP_A, (prop_a,), evidence_a)
    group_b = _group(GROUP_B, (prop_b,), evidence_b)
    return group_a, group_b, (_binding(prop_a, 1), _binding(prop_b, 2))


def _detect(
    groups: tuple[PropositionSupportGroup, ...],
    pairs: tuple[PropositionPairAssessment, ...],
    bindings: tuple[PropositionAssertionBinding, ...],
    *,
    corroboration_assessments: tuple[IndependentCorroborationAssessment, ...] = (),
    counterevidence_eligibility: tuple[CounterEvidenceEligibilityInput, ...] = (),
) -> PropositionConflictAssessment:
    return detect_proposition_conflicts(
        groups,
        pairs,
        bindings,
        corroboration_assessments=corroboration_assessments,
        counterevidence_eligibility=counterevidence_eligibility,
    )


def test_a3_contradiction_creates_one_canonical_group_edge_only() -> None:
    left, right, bindings = _simple_inputs()
    result = _detect(
        (right, left),
        (_pair("assertion-2", "assertion-1", PropositionRelation.CONTRADICTS),),
        bindings,
    )
    assert len(result.contradiction_edges) == 1
    edge = result.contradiction_edges[0]
    assert (edge.left_group_ref, edge.right_group_ref) == (GROUP_A, GROUP_B)
    assert edge.contradiction_pair_refs[0].left_assertion_ref == "assertion-1"
    assert (
        result.group_pair_coverages[0].coverage_status
        is GroupPairConflictState.DIRECT_CONTRADICTION
    )


def test_multiple_a3_contradictions_deduplicate_to_one_edge_with_all_pairs() -> None:
    a1, a2, b1 = _uuid(100, 1), _uuid(100, 2), _uuid(100, 3)
    ga = _group(GROUP_A, (a1, a2))
    gb = _group(GROUP_B, (b1,))
    bindings = (_binding(a1, 1), _binding(a2, 2), _binding(b1, 3))
    pairs = (
        _pair("assertion-1", "assertion-3", PropositionRelation.CONTRADICTS),
        _pair("assertion-2", "assertion-3", PropositionRelation.CONTRADICTS),
    )
    result = _detect((ga, gb), pairs, bindings)
    assert len(result.contradiction_edges) == 1
    assert len(result.contradiction_edges[0].contradiction_pair_refs) == 2
    assert result.group_pair_coverages[0].expected_cross_pair_count == 2


@pytest.mark.parametrize("other_relation", [PropositionRelation.UNKNOWN, None])
def test_proven_contradiction_is_monotonic_with_unknown_or_missing_other_pair(
    other_relation: PropositionRelation | None,
) -> None:
    a1, a2, b1 = _uuid(100, 1), _uuid(100, 2), _uuid(100, 3)
    ga, gb = _group(GROUP_A, (a1, a2)), _group(GROUP_B, (b1,))
    bindings = (_binding(a1, 1), _binding(a2, 2), _binding(b1, 3))
    pairs: tuple[PropositionPairAssessment, ...] = (
        _pair("assertion-1", "assertion-3", PropositionRelation.CONTRADICTS),
    )
    if other_relation is not None:
        pairs += (_pair("assertion-2", "assertion-3", other_relation),)
    result = _detect((ga, gb), pairs, bindings)
    assert (
        result.group_pair_coverages[0].coverage_status
        is GroupPairConflictState.DIRECT_CONTRADICTION
    )
    assert len(result.contradiction_edges) == 1


def test_no_contradiction_with_missing_or_unresolved_pair_is_unknown() -> None:
    a1, a2, b1 = _uuid(100, 1), _uuid(100, 2), _uuid(100, 3)
    ga, gb = _group(GROUP_A, (a1, a2)), _group(GROUP_B, (b1,))
    bindings = (_binding(a1, 1), _binding(a2, 2), _binding(b1, 3))
    pairs = (_pair("assertion-1", "assertion-3", PropositionRelation.RELATED),)
    result = _detect((ga, gb), pairs, bindings)
    coverage = result.group_pair_coverages[0]
    assert coverage.coverage_status is GroupPairConflictState.UNKNOWN
    assert len(coverage.missing_pair_refs) == 1


@pytest.mark.parametrize(
    "relation,comparability",
    [
        (PropositionRelation.UNKNOWN, ComparabilityStatus.COMPARABLE),
        (PropositionRelation.UNKNOWN, ComparabilityStatus.UNKNOWN),
    ],
)
def test_unknown_relation_is_not_no_contradiction(
    relation: PropositionRelation, comparability: ComparabilityStatus
) -> None:
    ga, gb, bindings = _simple_inputs()
    result = _detect(
        (ga, gb), (_pair("assertion-1", "assertion-2", relation, comparability),), bindings
    )
    assert result.group_pair_coverages[0].coverage_status is GroupPairConflictState.UNKNOWN


@pytest.mark.parametrize(
    "pairs",
    [
        (_pair("assertion-1", "assertion-2", PropositionRelation.RELATED),),
        (
            _pair(
                "assertion-1",
                "assertion-2",
                PropositionRelation.UNKNOWN,
                ComparabilityStatus.NOT_COMPARABLE,
            ),
        ),
    ],
)
def test_complete_definitive_noncontradiction_coverage(
    pairs: tuple[PropositionPairAssessment, ...],
) -> None:
    ga, gb, bindings = _simple_inputs()
    result = _detect((ga, gb), pairs, bindings)
    assert (
        result.group_pair_coverages[0].coverage_status
        is GroupPairConflictState.NO_DIRECT_CONTRADICTION
    )


def test_cross_group_local_same_is_grouping_unresolved() -> None:
    a1, a2, b1 = _uuid(100, 1), _uuid(100, 2), _uuid(100, 3)
    ga, gb = _group(GROUP_A, (a1, a2)), _group(GROUP_B, (b1,))
    bindings = (_binding(a1, 1), _binding(a2, 2), _binding(b1, 3))
    pairs = (_pair("assertion-1", "assertion-3", PropositionRelation.SAME),)
    result = _detect((ga, gb), pairs, bindings)
    assert (
        result.group_pair_coverages[0].coverage_status is GroupPairConflictState.GROUPING_UNRESOLVED
    )


def test_cross_group_all_pairs_same_is_grouping_inconsistency() -> None:
    a1, a2, b1, b2 = (_uuid(100, n) for n in range(1, 5))
    ga, gb = _group(GROUP_A, (a1, a2)), _group(GROUP_B, (b1, b2))
    bindings = tuple(_binding(prop, index) for index, prop in enumerate((a1, a2, b1, b2), 1))
    pairs = tuple(
        _pair(f"assertion-{left}", f"assertion-{right}", PropositionRelation.SAME)
        for left in (1, 2)
        for right in (3, 4)
    )
    result = _detect((ga, gb), pairs, bindings)
    assert (
        result.group_pair_coverages[0].coverage_status
        is GroupPairConflictState.GROUPING_INCONSISTENCY
    )
    assert not result.contradiction_edges


def test_cross_group_same_and_contradicts_fail_closed() -> None:
    a1, a2, b1 = _uuid(100, 1), _uuid(100, 2), _uuid(100, 3)
    ga, gb = _group(GROUP_A, (a1, a2)), _group(GROUP_B, (b1,))
    bindings = (_binding(a1, 1), _binding(a2, 2), _binding(b1, 3))
    pairs = (
        _pair("assertion-1", "assertion-3", PropositionRelation.SAME),
        _pair("assertion-2", "assertion-3", PropositionRelation.CONTRADICTS),
    )
    result = _detect((ga, gb), pairs, bindings)
    assert (
        result.group_pair_coverages[0].coverage_status
        is GroupPairConflictState.PAIR_RELATION_INCONSISTENT
    )
    assert not result.contradiction_edges


def test_consistent_duplicate_pair_is_deduplicated_and_conflicting_duplicate_fails_closed() -> None:
    ga, gb, bindings = _simple_inputs()
    contradiction = _pair("assertion-1", "assertion-2", PropositionRelation.CONTRADICTS)
    duplicate_reversed = _pair("assertion-2", "assertion-1", PropositionRelation.CONTRADICTS)
    result = _detect((ga, gb), (contradiction, duplicate_reversed), bindings)
    assert len(result.contradiction_edges[0].contradiction_pair_refs) == 1

    inconsistent = _pair("assertion-2", "assertion-1", PropositionRelation.RELATED)
    result = _detect((ga, gb), (contradiction, inconsistent), bindings)
    assert (
        result.group_pair_coverages[0].coverage_status
        is GroupPairConflictState.PAIR_RELATION_INCONSISTENT
    )
    assert not result.contradiction_edges


def test_assessment_for_unrelated_propositions_is_rejected() -> None:
    ga, gb, bindings = _simple_inputs()
    unrelated = _binding(_uuid(100, 99), 99)
    with pytest.raises(ValueError, match="pair_assessment_proposition_not_in_supplied_groups"):
        _detect(
            (ga, gb),
            (_pair("assertion-1", "assertion-99", PropositionRelation.CONTRADICTS),),
            (*bindings, unrelated),
        )


def test_missing_assertion_binding_is_reported_as_missing_coverage() -> None:
    ga, gb, bindings = _simple_inputs()
    result = _detect((ga, gb), (), bindings[:1])
    coverage = result.group_pair_coverages[0]
    assert coverage.expected_cross_pair_count == 1
    assert coverage.assessed_pair_count == 0
    assert coverage.coverage_status is GroupPairConflictState.UNKNOWN
    missing_ref = coverage.missing_pair_refs[0]
    assert missing_ref.left_assertion_ref is not None
    assert missing_ref.right_assertion_ref is None


def test_duplicate_conflicting_a3_pair_is_retained_as_inconsistent_coverage() -> None:
    ga, gb, bindings = _simple_inputs()
    pairs = (
        _pair("assertion-1", "assertion-2", PropositionRelation.CONTRADICTS),
        _pair("assertion-2", "assertion-1", PropositionRelation.RELATED),
    )
    result = _detect((ga, gb), pairs, bindings)
    coverage = result.group_pair_coverages[0]
    assert coverage.coverage_status is GroupPairConflictState.PAIR_RELATION_INCONSISTENT
    assert len(coverage.inconsistent_pair_refs) == 1
    assert not result.contradiction_edges


def test_evidence_contradicts_is_counterevidence_only_and_respects_acceptance() -> None:
    prop = _uuid(100, 1)
    evidence_ref = _uuid(400, 1)
    contra = (evidence_ref, prop, "owner-a", EvidencePropositionRelation.CONTRADICTS)
    group = _group(GROUP_A, (prop,), (contra,))
    binding = (_binding(prop, 1),)
    result = _detect(
        (group,),
        (),
        binding,
        counterevidence_eligibility=(
            CounterEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=evidence_ref, accepted=True
            ),
        ),
    )
    summary = result.group_summaries[0]
    assert not result.contradiction_edges
    assert summary.counterevidence.eligible_counterevidence_refs == (evidence_ref,)
    assert summary.counterevidence.known_counterevidence_owner_keys == ("owner-a",)


@pytest.mark.parametrize(
    "accepted,expected_field",
    [
        (False, "ineligible_counterevidence_refs"),
        (None, "unknown_eligibility_counterevidence_refs"),
    ],
)
def test_counterevidence_kept_but_eligibility_is_not_assumed(
    accepted: bool | None, expected_field: str
) -> None:
    prop, evidence_ref = _uuid(100, 1), _uuid(400, 1)
    group = _group(
        GROUP_A,
        (prop,),
        ((evidence_ref, prop, "owner-a", EvidencePropositionRelation.CONTRADICTS),),
    )
    result = _detect(
        (group,),
        (),
        (_binding(prop, 1),),
        counterevidence_eligibility=(
            CounterEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=evidence_ref, accepted=accepted
            ),
        ),
    )
    assert result.group_summaries[0].counterevidence.contradicting_evidence_refs == (evidence_ref,)
    assert getattr(result.group_summaries[0].counterevidence, expected_field) == (evidence_ref,)


def test_missing_counterevidence_eligibility_is_unknown() -> None:
    prop, evidence_ref = _uuid(100, 1), _uuid(400, 1)
    group = _group(
        GROUP_A,
        (prop,),
        ((evidence_ref, prop, "owner-a", EvidencePropositionRelation.CONTRADICTS),),
    )
    result = _detect((group,), (), (_binding(prop, 1),))
    counter = result.group_summaries[0].counterevidence
    assert counter.unknown_eligibility_counterevidence_refs == (evidence_ref,)
    assert counter.known_counterevidence_owner_keys == ()


@pytest.mark.parametrize(
    "direction",
    [
        EvidencePropositionRelation.SUPPORTS,
        EvidencePropositionRelation.MENTIONS,
        EvidencePropositionRelation.INSUFFICIENT,
    ],
)
def test_non_contradicts_direction_does_not_create_counterevidence(
    direction: EvidencePropositionRelation,
) -> None:
    prop, evidence_ref = _uuid(100, 1), _uuid(400, 1)
    group = _group(GROUP_A, (prop,), ((evidence_ref, prop, "owner-a", direction),))
    result = _detect((group,), (), (_binding(prop, 1),))
    assert result.group_summaries[0].counterevidence.contradicting_evidence_refs == ()


def test_duplicate_evidence_membership_is_counted_once() -> None:
    prop, evidence_ref, source_ref = _uuid(100, 1), _uuid(400, 1), _uuid(500, 1)
    first = PropositionEvidenceLink(
        proposition_id=prop,
        evidence_id=evidence_ref,
        claim_id=CLAIM_ID,
        relation=EvidencePropositionRelation.CONTRADICTS,
        source_id=source_ref,
        source_owner_key="owner-a",
        method="fixture",
        run_id=RUN_ID,
        question_id="q1",
        dimension_key="d1",
    )
    second = first.model_copy(update={"span_start": 2, "span_end": 4})
    group = PropositionSupportGroup(
        group_ref=GROUP_A,
        proposition_refs=(prop,),
        evidence_memberships=(
            PropositionEvidenceMembership(
                evidence_ref=evidence_ref, provenance_links=(first, second)
            ),
        ),
        grouping_status=PropositionGroupingStatus.SINGLETON,
    )
    result = _detect((group,), (), (_binding(prop, 1),))
    assert result.group_summaries[0].counterevidence.contradicting_evidence_refs == (evidence_ref,)


def test_support_counter_owner_overlap_and_cross_owner_opposition_are_retained() -> None:
    prop, support_ref, counter_ref = _uuid(100, 1), _uuid(400, 1), _uuid(400, 2)
    group = _group(
        GROUP_A,
        (prop,),
        (
            (support_ref, prop, "owner-a", EvidencePropositionRelation.SUPPORTS),
            (counter_ref, prop, "owner-b", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    support = aggregate_independent_corroboration(
        group,
        (
            CorroborationEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=support_ref, accepted=True
            ),
        ),
    )
    result = _detect(
        (group,),
        (),
        (_binding(prop, 1),),
        corroboration_assessments=(support,),
        counterevidence_eligibility=(
            CounterEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=counter_ref, accepted=True
            ),
        ),
    )
    summary = result.group_summaries[0]
    assert summary.support_counter_owner_overlap == ()
    assert summary.cross_owner_opposition_pairs is not None
    assert summary.cross_owner_opposition_pairs[0].support_owner_key == "owner-a"
    assert summary.cross_owner_opposition_pairs[0].counterevidence_owner_key == "owner-b"


def test_same_owner_support_and_counterevidence_overlap_is_preserved() -> None:
    prop, support_ref, counter_ref = _uuid(100, 1), _uuid(400, 1), _uuid(400, 2)
    group = _group(
        GROUP_A,
        (prop,),
        (
            (support_ref, prop, "owner-a", EvidencePropositionRelation.SUPPORTS),
            (counter_ref, prop, "owner-a", EvidencePropositionRelation.CONTRADICTS),
        ),
    )
    support = aggregate_independent_corroboration(
        group,
        (
            CorroborationEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=support_ref, accepted=True
            ),
        ),
    )
    result = _detect(
        (group,),
        (),
        (_binding(prop, 1),),
        corroboration_assessments=(support,),
        counterevidence_eligibility=(
            CounterEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=counter_ref, accepted=True
            ),
        ),
    )
    assert result.group_summaries[0].cross_owner_opposition_pairs is not None
    assert result.group_summaries[0].support_counter_owner_overlap == ("owner-a",)
    assert result.group_summaries[0].cross_owner_opposition_pairs == ()


def test_unknown_owner_counterevidence_does_not_create_an_owner_bucket() -> None:
    prop, evidence_ref = _uuid(100, 1), _uuid(400, 1)
    group = _group(
        GROUP_A,
        (prop,),
        ((evidence_ref, prop, None, EvidencePropositionRelation.CONTRADICTS),),
    )
    result = _detect(
        (group,),
        (),
        (_binding(prop, 1),),
        counterevidence_eligibility=(
            CounterEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=evidence_ref, accepted=True
            ),
        ),
    )
    counter = result.group_summaries[0].counterevidence
    assert counter.known_counterevidence_owner_keys == ()
    assert counter.unknown_owner_counterevidence_refs == (evidence_ref,)


def test_counterevidence_owner_set_deduplicates_same_owner_and_keeps_distinct_owners() -> None:
    prop = _uuid(100, 1)
    evidence = (
        (_uuid(400, 1), prop, "owner-a", EvidencePropositionRelation.CONTRADICTS),
        (_uuid(400, 2), prop, "owner-a", EvidencePropositionRelation.CONTRADICTS),
        (_uuid(400, 3), prop, "owner-b", EvidencePropositionRelation.CONTRADICTS),
    )
    group = _group(GROUP_A, (prop,), evidence)
    eligibility = tuple(
        CounterEvidenceEligibilityInput(group_ref=GROUP_A, evidence_ref=entry[0], accepted=True)
        for entry in evidence
    )
    counter = (
        _detect((group,), (), (_binding(prop, 1),), counterevidence_eligibility=eligibility)
        .group_summaries[0]
        .counterevidence
    )
    assert counter.eligible_counterevidence_count == 3
    assert counter.known_counterevidence_owner_count == 2
    assert counter.known_counterevidence_owner_keys == ("owner-a", "owner-b")


def test_direct_group_conflict_does_not_create_counterevidence_direction() -> None:
    ga, gb, bindings = _simple_inputs()
    result = _detect(
        (ga, gb),
        (_pair("assertion-1", "assertion-2", PropositionRelation.CONTRADICTS),),
        bindings,
    )
    assert result.contradiction_edges
    assert all(
        summary.counterevidence.contradicting_evidence_refs == ()
        for summary in result.group_summaries
    )


def test_counterevidence_does_not_create_group_conflict_edge() -> None:
    prop, evidence_ref = _uuid(100, 1), _uuid(400, 1)
    group = _group(
        GROUP_A,
        (prop,),
        ((evidence_ref, prop, "owner-a", EvidencePropositionRelation.CONTRADICTS),),
    )
    result = _detect(
        (group,),
        (),
        (_binding(prop, 1),),
        counterevidence_eligibility=(
            CounterEvidenceEligibilityInput(
                group_ref=GROUP_A, evidence_ref=evidence_ref, accepted=True
            ),
        ),
    )
    assert not result.contradiction_edges
    assert result.group_pair_coverages == ()
    assert result.group_summaries[0].counterevidence.eligible_counterevidence_count == 1


def test_b2_support_owner_input_must_match_group_provenance() -> None:
    prop, evidence_ref = _uuid(100, 1), _uuid(400, 1)
    group = _group(
        GROUP_A,
        (prop,),
        ((evidence_ref, prop, "owner-a", EvidencePropositionRelation.SUPPORTS),),
    )
    support = aggregate_independent_corroboration(group, ())
    wrong = support.model_copy(update={"known_support_owner_keys": ("owner-b",)})
    with pytest.raises(ValueError, match="corroboration_owner_keys_mismatch_group_provenance"):
        _detect((group,), (), (_binding(prop, 1),), corroboration_assessments=(wrong,))


def test_group_and_assessment_permutations_produce_identical_output() -> None:
    ga, gb, bindings = _simple_inputs()
    pair = _pair("assertion-2", "assertion-1", PropositionRelation.CONTRADICTS)
    expected = _detect((ga, gb), (pair,), bindings).model_dump(mode="json")
    for group_order in permutations((ga, gb)):
        for pair_order in permutations((pair,)):
            actual = _detect(group_order, pair_order, tuple(reversed(bindings))).model_dump(
                mode="json"
            )
            assert actual == expected


def test_c1_types_have_no_verification_or_truth_verdict_fields() -> None:
    ga, gb, bindings = _simple_inputs()
    result = _detect((ga, gb), (), bindings)
    field_names = set(type(result).model_fields)
    assert not field_names.intersection(
        {
            "verification_status",
            "verified",
            "supported",
            "conflicted",
            "truth_probability",
            "confidence",
        }
    )
