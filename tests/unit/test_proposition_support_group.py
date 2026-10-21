from __future__ import annotations

from itertools import permutations
from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    ClaimPropositionLink,
    DecompositionStatus,
    EvidencePropositionRelation,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.proposition_support_group import (
    EvidenceSourceIdentity,
    OwnerIndependenceStatus,
    PropositionAssertionBinding,
    PropositionGroupingReason,
    PropositionGroupingStatus,
    assess_owner_independence,
    group_propositions,
)
from app.domain.verification_assertion import (
    ComparabilityStatus,
    ComparisonReasonCode,
    PropositionPairAssessment,
)

PROP_A = UUID("10000000-0000-0000-0000-000000000001")
PROP_B = UUID("10000000-0000-0000-0000-000000000002")
PROP_C = UUID("10000000-0000-0000-0000-000000000003")
CLAIM_A = UUID("20000000-0000-0000-0000-000000000001")
CLAIM_B = UUID("20000000-0000-0000-0000-000000000002")
EVIDENCE_A = UUID("30000000-0000-0000-0000-000000000001")
EVIDENCE_B = UUID("30000000-0000-0000-0000-000000000002")
SOURCE_A = UUID("40000000-0000-0000-0000-000000000001")
SOURCE_B = UUID("40000000-0000-0000-0000-000000000002")


def _claim_link(claim: UUID, proposition: UUID) -> ClaimPropositionLink:
    return ClaimPropositionLink(
        claim_id=claim,
        proposition_id=proposition,
        proposition_index=0,
        decomposition_status=DecompositionStatus.SINGLE_PROPOSITION,
    )


def _evidence_link(
    evidence: UUID,
    source: UUID,
    claim: UUID,
    proposition: UUID,
    *,
    relation: EvidencePropositionRelation = EvidencePropositionRelation.SUPPORTS,
    owner: str | None = "owner-one",
) -> PropositionEvidenceLink:
    return PropositionEvidenceLink(
        proposition_id=proposition,
        evidence_id=evidence,
        claim_id=claim,
        relation=relation,
        source_id=source,
        source_owner_key=owner,
        quote_reference="quote-1",
        span_start=0,
        span_end=8,
        method="fixture",
        run_id=UUID("50000000-0000-0000-0000-000000000001"),
        question_id="q1",
        dimension_key="d1",
    )


def _binding(proposition: UUID, assertion: str) -> PropositionAssertionBinding:
    return PropositionAssertionBinding(proposition_ref=proposition, assertion_ref=assertion)


def _pair(left: str, right: str, relation: PropositionRelation) -> PropositionPairAssessment:
    comparable = relation in {PropositionRelation.SAME, PropositionRelation.CONTRADICTS}
    reasons = (
        (ComparisonReasonCode.SAME_TYPED_VALUE,)
        if relation is PropositionRelation.SAME
        else (ComparisonReasonCode.DIRECT_POLARITY_CONFLICT,)
        if relation is PropositionRelation.CONTRADICTS
        else (ComparisonReasonCode.RELATION_SEMANTICS_INSUFFICIENT,)
    )
    if relation is PropositionRelation.RELATED:
        reasons = (ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION,)
    return PropositionPairAssessment(
        left_assertion_ref=left,
        right_assertion_ref=right,
        comparability=(
            ComparabilityStatus.COMPARABLE
            if comparable
            else ComparabilityStatus.NOT_COMPARABLE
        ),
        relation=relation,
        relation_eligible=relation is not PropositionRelation.UNKNOWN,
        reason_codes=reasons,
    )


def _run(
    proposition_refs: tuple[UUID, ...],
    pairs: tuple[PropositionPairAssessment, ...] = (),
    *,
    evidence_links: tuple[PropositionEvidenceLink, ...] = (),
    claim_links: tuple[ClaimPropositionLink, ...] | None = None,
):
    claim_links = claim_links or tuple(
        _claim_link(CLAIM_A, ref) for ref in proposition_refs
    )
    bindings = tuple(
        _binding(ref, f"assertion-{ref.hex[-1]}") for ref in proposition_refs
    )
    return group_propositions(claim_links, evidence_links, bindings, pairs)


def test_exact_same_proposition_ref_is_one_group_across_claims() -> None:
    links = (
        _claim_link(CLAIM_A, PROP_A),
        _claim_link(CLAIM_B, PROP_A),
    )
    evidence = (
        _evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_A, PROP_A),
        _evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_B, PROP_A),
    )
    result = group_propositions(links, evidence, (_binding(PROP_A, "assertion-a"),), ())
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.proposition_refs == (PROP_A,)
    assert group.grouping_status is PropositionGroupingStatus.SINGLETON
    assert len(group.evidence_memberships) == 1
    assert len(group.evidence_memberships[0].provenance_links) == 2


def test_distinct_refs_merge_only_when_a3_says_same() -> None:
    pairs = (_pair("assertion-1", "assertion-2", PropositionRelation.SAME),)
    result = _run((PROP_A, PROP_B), pairs)
    assert len(result.groups) == 1
    assert result.groups[0].proposition_refs == (PROP_A, PROP_B)
    assert result.groups[0].grouping_status is PropositionGroupingStatus.GROUPED


@pytest.mark.parametrize(
    "relation",
    [
        PropositionRelation.RELATED,
        PropositionRelation.CONTRADICTS,
        PropositionRelation.UNKNOWN,
    ],
)
def test_non_same_a3_relations_never_merge(relation) -> None:
    result = _run((PROP_A, PROP_B), (_pair("assertion-1", "assertion-2", relation),))
    assert len(result.groups) == 2
    assert all(len(group.proposition_refs) == 1 for group in result.groups)


def test_complete_three_member_same_clique_forms_one_group() -> None:
    pairs = (
        _pair("assertion-1", "assertion-2", PropositionRelation.SAME),
        _pair("assertion-1", "assertion-3", PropositionRelation.SAME),
        _pair("assertion-2", "assertion-3", PropositionRelation.SAME),
    )
    result = _run((PROP_A, PROP_B, PROP_C), pairs)
    assert len(result.groups) == 1
    assert result.groups[0].proposition_refs == (PROP_A, PROP_B, PROP_C)


def test_missing_nontransitive_pair_fails_closed_without_three_member_merge() -> None:
    pairs = (
        _pair("assertion-1", "assertion-2", PropositionRelation.SAME),
        _pair("assertion-2", "assertion-3", PropositionRelation.SAME),
    )
    result = _run((PROP_A, PROP_B, PROP_C), pairs)
    assert len(result.groups) == 3
    assert all(
        group.grouping_status is PropositionGroupingStatus.UNRESOLVED_MEMBERSHIP
        for group in result.groups
    )
    assert all(
        PropositionGroupingReason.PAIR_ASSESSMENT_MISSING in group.reason_codes
        for group in result.groups
    )


def test_same_chain_with_contradictory_cross_pair_fails_closed() -> None:
    pairs = (
        _pair("assertion-1", "assertion-2", PropositionRelation.SAME),
        _pair("assertion-2", "assertion-3", PropositionRelation.SAME),
        _pair("assertion-1", "assertion-3", PropositionRelation.CONTRADICTS),
    )
    with pytest.raises(ValueError, match="GROUPING_INCONSISTENT"):
        _run((PROP_A, PROP_B, PROP_C), pairs)


def test_grouping_is_independent_of_all_input_order_permutations() -> None:
    proposition_refs = (PROP_A, PROP_B, PROP_C)
    pairs = (
        _pair("assertion-1", "assertion-2", PropositionRelation.SAME),
        _pair("assertion-1", "assertion-3", PropositionRelation.SAME),
        _pair("assertion-2", "assertion-3", PropositionRelation.SAME),
    )
    outputs = {
        tuple(
            (group.group_ref, group.proposition_refs, group.grouping_status)
            for group in _run(tuple(order), tuple(pair_order)).groups
        )
        for order in permutations(proposition_refs)
        for pair_order in permutations(pairs)
    }
    assert len(outputs) == 1


def test_duplicate_proposition_refs_and_evidence_paths_are_deduplicated() -> None:
    same = _pair("assertion-1", "assertion-2", PropositionRelation.SAME)
    link = _evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_A, PROP_A)
    result = _run(
        (PROP_A, PROP_B, PROP_A),
        (same,),
        evidence_links=(link, link),
    )
    assert len(result.groups) == 1
    assert len(result.groups[0].evidence_memberships) == 1
    assert len(result.groups[0].evidence_memberships[0].provenance_links) == 1


def test_same_evidence_from_multiple_paths_preserves_each_link_once() -> None:
    pairs = (_pair("assertion-1", "assertion-2", PropositionRelation.SAME),)
    evidence = (
        _evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_A, PROP_A),
        _evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_B, PROP_B),
    )
    claims = (_claim_link(CLAIM_A, PROP_A), _claim_link(CLAIM_B, PROP_B))
    result = _run((PROP_A, PROP_B), pairs, evidence_links=evidence, claim_links=claims)
    membership = result.groups[0].evidence_memberships[0]
    assert membership.evidence_ref == EVIDENCE_A
    assert len(membership.provenance_links) == 2
    assert {link.relation for link in membership.provenance_links} == {
        EvidencePropositionRelation.SUPPORTS
    }


def test_evidence_link_does_not_spread_to_same_claim_sibling_proposition() -> None:
    claim_links = (_claim_link(CLAIM_A, PROP_A), _claim_link(CLAIM_A, PROP_B))
    evidence = (_evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_A, PROP_A),)
    result = group_propositions(claim_links, evidence, (), ())
    evidence_by_prop = {
        group.proposition_refs[0]: [item.evidence_ref for item in group.evidence_memberships]
        for group in result.groups
    }
    assert evidence_by_prop[PROP_A] == [EVIDENCE_A]
    assert evidence_by_prop[PROP_B] == []


def test_evidence_link_requires_matching_claim_proposition_provenance() -> None:
    with pytest.raises(ValueError, match="missing_claim_proposition_link"):
        group_propositions(
            (_claim_link(CLAIM_A, PROP_A),),
            (_evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_B, PROP_A),),
            (),
            (),
        )


def test_membership_preserves_link_direction_without_inventing_support() -> None:
    claims = (_claim_link(CLAIM_A, PROP_A), _claim_link(CLAIM_B, PROP_A))
    evidence = (
        _evidence_link(EVIDENCE_A, SOURCE_A, CLAIM_A, PROP_A),
        _evidence_link(
            EVIDENCE_B,
            SOURCE_B,
            CLAIM_B,
            PROP_A,
            relation=EvidencePropositionRelation.MENTIONS,
        ),
    )
    group = group_propositions(claims, evidence, (_binding(PROP_A, "assertion-a"),), ()).groups[0]
    directions = {
        link.relation
        for membership in group.evidence_memberships
        for link in membership.provenance_links
    }
    assert directions == {
        EvidencePropositionRelation.SUPPORTS,
        EvidencePropositionRelation.MENTIONS,
    }


def test_distinct_claims_can_group_same_propositions_via_a3_same() -> None:
    claims = (_claim_link(CLAIM_A, PROP_A), _claim_link(CLAIM_B, PROP_B))
    pairs = (_pair("assertion-1", "assertion-2", PropositionRelation.SAME),)
    bindings = (_binding(PROP_A, "assertion-1"), _binding(PROP_B, "assertion-2"))
    result = group_propositions(claims, (), bindings, pairs)
    assert result.groups[0].proposition_refs == (PROP_A, PROP_B)


def _owner(
    evidence: UUID,
    source: UUID,
    owner: str | None,
    *,
    diversity: str | None = None,
) -> EvidenceSourceIdentity:
    return EvidenceSourceIdentity(
        evidence_ref=evidence,
        source_ref=source,
        source_owner_key=owner,
        source_diversity_key=diversity,
    )


def test_same_known_owner_is_same_owner_even_for_different_sources_and_diversity_keys() -> None:
    result = assess_owner_independence(
        _owner(EVIDENCE_A, SOURCE_A, "example.org", diversity="diversity-a"),
        _owner(EVIDENCE_B, SOURCE_B, "example.org", diversity="diversity-b"),
    )
    assert result.status is OwnerIndependenceStatus.SAME_OWNER


def test_different_known_owner_keys_are_distinct() -> None:
    result = assess_owner_independence(
        _owner(EVIDENCE_A, SOURCE_A, "owner-a"),
        _owner(EVIDENCE_B, SOURCE_A, "owner-b"),
    )
    assert result.status is OwnerIndependenceStatus.DISTINCT_OWNER


@pytest.mark.parametrize(
    ("left_owner", "right_owner"),
    [(None, "owner-a"), ("owner-a", None), (None, None), ("unknown", "owner-a")],
)
def test_missing_or_unknown_owner_never_implies_same_or_distinct(left_owner, right_owner) -> None:
    result = assess_owner_independence(
        _owner(EVIDENCE_A, SOURCE_A, left_owner),
        _owner(EVIDENCE_B, SOURCE_B, right_owner),
    )
    assert result.status is OwnerIndependenceStatus.UNKNOWN


def test_source_diversity_key_never_overrides_owner_key() -> None:
    same = assess_owner_independence(
        _owner(EVIDENCE_A, SOURCE_A, "owner-a", diversity="d1"),
        _owner(EVIDENCE_B, SOURCE_B, "owner-a", diversity="d2"),
    )
    distinct = assess_owner_independence(
        _owner(EVIDENCE_A, SOURCE_A, "owner-a", diversity="same"),
        _owner(EVIDENCE_B, SOURCE_B, "owner-b", diversity="same"),
    )
    assert same.status is OwnerIndependenceStatus.SAME_OWNER
    assert distinct.status is OwnerIndependenceStatus.DISTINCT_OWNER


def test_owner_assessment_is_symmetric_and_deterministic() -> None:
    left = _owner(EVIDENCE_A, SOURCE_A, "owner-a")
    right = _owner(EVIDENCE_B, SOURCE_B, "owner-b")
    forward = assess_owner_independence(left, right)
    reverse = assess_owner_independence(right, left)
    repeated = assess_owner_independence(left, right)
    assert forward.status is reverse.status is OwnerIndependenceStatus.DISTINCT_OWNER
    assert forward.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")


def test_duplicate_evidence_owner_comparison_is_rejected() -> None:
    identity = _owner(EVIDENCE_A, SOURCE_A, "owner-a")
    with pytest.raises(ValueError, match="requires_distinct_evidence"):
        assess_owner_independence(identity, identity)


def test_group_refs_and_members_are_deterministic() -> None:
    result_a = _run(
        (PROP_B, PROP_A),
        (_pair("assertion-1", "assertion-2", PropositionRelation.SAME),),
    )
    result_b = _run(
        (PROP_A, PROP_B),
        (_pair("assertion-2", "assertion-1", PropositionRelation.SAME),),
    )
    assert result_a.model_dump(mode="json") == result_b.model_dump(mode="json")
