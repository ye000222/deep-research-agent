from __future__ import annotations

from itertools import permutations
from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    EvidencePropositionRelation,
    PropositionEvidenceLink,
)
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibility,
    CorroborationEvidenceEligibilityInput,
    IndependentCorroborationAssessment,
    OwnerCorroborationStatus,
    aggregate_independent_corroboration,
    resolve_evidence_eligibility,
)
from app.domain.proposition_support_group import (
    EvidenceOwnerIndependenceAssessment,
    EvidenceSourceIdentity,
    OwnerIndependenceStatus,
    PropositionEvidenceMembership,
    PropositionGroupingStatus,
    PropositionSupportGroup,
    assess_owner_independence,
)

GROUP_REF = "a" * 64
PROP_REF = UUID("10000000-0000-0000-0000-000000000001")
CLAIM_REF = UUID("20000000-0000-0000-0000-000000000001")
SOURCE_A = UUID("30000000-0000-0000-0000-000000000001")
SOURCE_B = UUID("30000000-0000-0000-0000-000000000002")
EVIDENCE_A = UUID("40000000-0000-0000-0000-000000000001")
EVIDENCE_B = UUID("40000000-0000-0000-0000-000000000002")
EVIDENCE_C = UUID("40000000-0000-0000-0000-000000000003")


def _link(
    evidence_ref: UUID,
    source_ref: UUID,
    owner: str | None,
    relation: EvidencePropositionRelation = EvidencePropositionRelation.SUPPORTS,
    *,
    claim_ref: UUID = CLAIM_REF,
) -> PropositionEvidenceLink:
    return PropositionEvidenceLink(
        proposition_id=PROP_REF,
        evidence_id=evidence_ref,
        claim_id=claim_ref,
        relation=relation,
        source_id=source_ref,
        source_owner_key=owner,
        quote_reference="quote-1",
        span_start=0,
        span_end=10,
        method="fixture",
        run_id=UUID("50000000-0000-0000-0000-000000000001"),
        question_id="q1",
        dimension_key="d1",
    )


def _group(*links: PropositionEvidenceLink) -> PropositionSupportGroup:
    memberships = tuple(
        PropositionEvidenceMembership(
            evidence_ref=evidence_ref,
            provenance_links=tuple(link for link in links if link.evidence_id == evidence_ref),
        )
        for evidence_ref in sorted({link.evidence_id for link in links}, key=str)
    )
    return PropositionSupportGroup(
        group_ref=GROUP_REF,
        proposition_refs=(PROP_REF,),
        evidence_memberships=memberships,
        grouping_status=PropositionGroupingStatus.SINGLETON,
    )


def _input(
    evidence_ref: UUID, accepted: bool | None = True
) -> CorroborationEvidenceEligibilityInput:
    return CorroborationEvidenceEligibilityInput(
        group_ref=GROUP_REF,
        evidence_ref=evidence_ref,
        accepted=accepted,
    )


def _aggregate(
    links: tuple[PropositionEvidenceLink, ...],
    accepted: dict[UUID, bool | None] | None = None,
    *,
    owner_assessments: tuple[EvidenceOwnerIndependenceAssessment, ...] = (),
) -> IndependentCorroborationAssessment:
    group = _group(*links)
    eligibility = accepted if accepted is not None else {link.evidence_id: True for link in links}
    inputs = tuple(_input(ref, value) for ref, value in eligibility.items())
    return aggregate_independent_corroboration(group, inputs, owner_assessments)


def test_supports_accepted_enters_positive_support_and_known_owner_set() -> None:
    result = _aggregate((_link(EVIDENCE_A, SOURCE_A, "owner-a"),))
    assert result.eligible_support_evidence_refs == (EVIDENCE_A,)
    assert result.known_support_owner_keys == ("owner-a",)
    assert result.owner_corroboration_status is OwnerCorroborationStatus.SINGLE_KNOWN_OWNER


@pytest.mark.parametrize(
    "relation",
    [
        EvidencePropositionRelation.CONTRADICTS,
        EvidencePropositionRelation.MENTIONS,
        EvidencePropositionRelation.INSUFFICIENT,
    ],
)
def test_non_support_directions_are_preserved_but_never_counted(relation) -> None:
    result = _aggregate((_link(EVIDENCE_A, SOURCE_A, "owner-a", relation),))
    assert result.support_evidence_count == 0
    assert result.known_owner_count == 0
    assert result.non_support_evidence_refs == (EVIDENCE_A,)
    assert result.owner_corroboration_status is OwnerCorroborationStatus.NO_ELIGIBLE_SUPPORT
    assert result.non_support_directions[0].relations == (relation,)


def test_rejected_support_is_ineligible_and_does_not_contribute_owner() -> None:
    result = _aggregate((_link(EVIDENCE_A, SOURCE_A, "owner-a"),), {EVIDENCE_A: False})
    assert result.ineligible_support_evidence_refs == (EVIDENCE_A,)
    assert result.known_support_owner_keys == ()
    assert result.owner_corroboration_status is OwnerCorroborationStatus.NO_ELIGIBLE_SUPPORT


def test_missing_or_unknown_acceptance_is_retained_as_uncertainty() -> None:
    result = aggregate_independent_corroboration(
        _group(_link(EVIDENCE_A, SOURCE_A, "owner-a")), (),
    )
    assert result.unknown_eligibility_support_evidence_refs == (EVIDENCE_A,)
    assert result.owner_corroboration_status is OwnerCorroborationStatus.UNKNOWN


def test_unknown_acceptance_input_is_not_eligible() -> None:
    result = _aggregate((_link(EVIDENCE_A, SOURCE_A, "owner-a"),), {EVIDENCE_A: None})
    assert result.unknown_eligibility_support_evidence_refs == (EVIDENCE_A,)
    assert result.known_owner_count == 0
    assert result.owner_corroboration_status is OwnerCorroborationStatus.UNKNOWN


def test_eligibility_precedence_is_ineligible_then_unknown_then_eligible() -> None:
    assert resolve_evidence_eligibility(
        (CorroborationEvidenceEligibility.INELIGIBLE, CorroborationEvidenceEligibility.UNKNOWN)
    ) is CorroborationEvidenceEligibility.INELIGIBLE
    assert resolve_evidence_eligibility(
        (CorroborationEvidenceEligibility.ELIGIBLE, CorroborationEvidenceEligibility.UNKNOWN)
    ) is CorroborationEvidenceEligibility.UNKNOWN
    assert resolve_evidence_eligibility(
        (CorroborationEvidenceEligibility.ELIGIBLE, CorroborationEvidenceEligibility.ELIGIBLE)
    ) is CorroborationEvidenceEligibility.ELIGIBLE
    assert resolve_evidence_eligibility(()) is CorroborationEvidenceEligibility.UNKNOWN


@pytest.mark.parametrize("owners", [("owner-a",), ("owner-a", "owner-a", "owner-a")])
def test_multiple_eligible_evidence_from_same_owner_counts_one_owner(owners) -> None:
    evidence_refs = (EVIDENCE_A, EVIDENCE_B, EVIDENCE_C)
    source_refs = (SOURCE_A, SOURCE_B, SOURCE_A)
    links = tuple(
        _link(evidence_refs[index], source_refs[index], owner)
        for index, owner in enumerate(owners)
    )
    result = _aggregate(links)
    assert result.support_evidence_count == len(owners)
    assert result.eligible_support_evidence_count == len(owners)
    assert result.known_owner_count == 1
    assert result.owner_corroboration_status is OwnerCorroborationStatus.SINGLE_KNOWN_OWNER


def test_two_known_owners_with_eligible_support_are_multi_owner() -> None:
    result = _aggregate(
        (_link(EVIDENCE_A, SOURCE_A, "owner-a"), _link(EVIDENCE_B, SOURCE_B, "owner-b"))
    )
    assert result.known_support_owner_keys == ("owner-a", "owner-b")
    assert result.owner_corroboration_status is OwnerCorroborationStatus.MULTI_KNOWN_OWNERS


def test_eligible_support_with_unknown_owner_never_creates_synthetic_owner() -> None:
    result = _aggregate((_link(EVIDENCE_A, SOURCE_A, None),))
    assert result.unknown_owner_support_evidence_refs == (EVIDENCE_A,)
    assert result.known_support_owner_keys == ()
    assert result.owner_corroboration_status is OwnerCorroborationStatus.UNKNOWN


def test_one_known_owner_plus_unknown_owner_is_unknown() -> None:
    result = _aggregate(
        (_link(EVIDENCE_A, SOURCE_A, "owner-a"), _link(EVIDENCE_B, SOURCE_B, "unknown"))
    )
    assert result.known_support_owner_keys == ("owner-a",)
    assert result.unknown_owner_support_evidence_refs == (EVIDENCE_B,)
    assert result.owner_corroboration_status is OwnerCorroborationStatus.UNKNOWN


def test_known_multi_owner_remains_affirmative_with_unknown_owner_or_eligibility() -> None:
    links = (
        _link(EVIDENCE_A, SOURCE_A, "owner-a"),
        _link(EVIDENCE_B, SOURCE_B, "owner-b"),
        _link(EVIDENCE_C, SOURCE_A, None),
    )
    result = _aggregate(links)
    assert result.owner_corroboration_status is OwnerCorroborationStatus.MULTI_KNOWN_OWNERS

    uncertain = _aggregate(
        links,
        {EVIDENCE_A: True, EVIDENCE_B: True, EVIDENCE_C: None},
    )
    assert uncertain.owner_corroboration_status is OwnerCorroborationStatus.MULTI_KNOWN_OWNERS


def test_unknown_eligibility_for_second_known_owner_prevents_multi_owner() -> None:
    result = _aggregate(
        (_link(EVIDENCE_A, SOURCE_A, "owner-a"), _link(EVIDENCE_B, SOURCE_B, "owner-b")),
        {EVIDENCE_A: True, EVIDENCE_B: None},
    )
    assert result.known_support_owner_keys == ("owner-a",)
    assert result.owner_corroboration_status is OwnerCorroborationStatus.UNKNOWN


def test_unknown_eligibility_may_change_owner_multiplicity() -> None:
    result = _aggregate(
        (_link(EVIDENCE_A, SOURCE_A, "owner-a"),), {EVIDENCE_A: None}
    )
    assert result.owner_corroboration_status is OwnerCorroborationStatus.UNKNOWN


def test_duplicate_evidence_and_duplicate_provenance_do_not_double_count() -> None:
    link = _link(EVIDENCE_A, SOURCE_A, "owner-a")
    result = _aggregate((link, link))
    assert result.support_evidence_count == 1
    assert result.eligible_support_evidence_count == 1
    assert result.known_owner_count == 1


def test_multiple_consistent_support_paths_count_evidence_once() -> None:
    links = (
        _link(EVIDENCE_A, SOURCE_A, "owner-a", claim_ref=CLAIM_REF),
        _link(
            EVIDENCE_A,
            SOURCE_A,
            "owner-a",
            claim_ref=UUID("20000000-0000-0000-0000-000000000002"),
        ),
    )
    result = _aggregate(links)
    assert result.support_evidence_count == 1
    assert result.eligible_support_evidence_count == 1


def test_conflicting_directions_for_same_evidence_fail_closed() -> None:
    with pytest.raises(ValueError, match="PROVENANCE_RELATION_INCONSISTENT"):
        _aggregate(
            (
                _link(EVIDENCE_A, SOURCE_A, "owner-a", EvidencePropositionRelation.SUPPORTS),
                _link(EVIDENCE_A, SOURCE_A, "owner-a", EvidencePropositionRelation.CONTRADICTS),
            )
        )


def test_inconsistent_owner_for_same_evidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="evidence_identity_provenance_conflict"):
        _group(
            _link(EVIDENCE_A, SOURCE_A, "owner-a"),
            _link(EVIDENCE_A, SOURCE_A, "owner-b"),
        )


def test_eligibility_from_wrong_group_or_unknown_evidence_fails_closed() -> None:
    group = _group(_link(EVIDENCE_A, SOURCE_A, "owner-a"))
    with pytest.raises(ValueError, match="group_membership_mismatch"):
        aggregate_independent_corroboration(
            group,
            (
                CorroborationEvidenceEligibilityInput(
                    group_ref="b" * 64, evidence_ref=EVIDENCE_A, accepted=True
                ),
            ),
        )
    with pytest.raises(ValueError, match="group_membership_mismatch"):
        aggregate_independent_corroboration(group, (_input(EVIDENCE_B),))


def test_conflicting_eligibility_inputs_fail_closed() -> None:
    group = _group(_link(EVIDENCE_A, SOURCE_A, "owner-a"))
    with pytest.raises(ValueError, match="eligibility_input_inconsistent"):
        aggregate_independent_corroboration(
            group, (_input(EVIDENCE_A, True), _input(EVIDENCE_A, False))
        )


def test_b1_owner_pair_assessment_mismatch_fails_closed() -> None:
    group = _group(
        _link(EVIDENCE_A, SOURCE_A, "owner-a"),
        _link(EVIDENCE_B, SOURCE_B, "owner-b"),
    )
    correct = assess_owner_independence(
        EvidenceSourceIdentity(
            evidence_ref=EVIDENCE_A, source_ref=SOURCE_A, source_owner_key="owner-a"
        ),
        EvidenceSourceIdentity(
            evidence_ref=EVIDENCE_B, source_ref=SOURCE_B, source_owner_key="owner-b"
        ),
    )
    wrong = correct.model_copy(update={"status": OwnerIndependenceStatus.SAME_OWNER})
    with pytest.raises(ValueError, match="mismatches_canonical_owner_keys"):
        aggregate_independent_corroboration(
            group, (_input(EVIDENCE_A), _input(EVIDENCE_B)), (wrong,)
        )


def test_owner_pair_assessment_order_is_symmetric_and_input_is_not_mutated() -> None:
    group = _group(
        _link(EVIDENCE_A, SOURCE_A, "owner-a"),
        _link(EVIDENCE_B, SOURCE_B, "owner-b"),
    )
    owner_a = EvidenceSourceIdentity(
        evidence_ref=EVIDENCE_A, source_ref=SOURCE_A, source_owner_key="owner-a"
    )
    owner_b = EvidenceSourceIdentity(
        evidence_ref=EVIDENCE_B, source_ref=SOURCE_B, source_owner_key="owner-b"
    )
    assessment = assess_owner_independence(owner_a, owner_b)
    reverse = assessment.model_copy(
        update={
            "left_evidence_ref": EVIDENCE_B,
            "right_evidence_ref": EVIDENCE_A,
            "left_source_ref": SOURCE_B,
            "right_source_ref": SOURCE_A,
        }
    )
    before = reverse.model_dump(mode="json")
    result = aggregate_independent_corroboration(
        group, (_input(EVIDENCE_A), _input(EVIDENCE_B)), (reverse,)
    )
    assert result.owner_corroboration_status is OwnerCorroborationStatus.MULTI_KNOWN_OWNERS
    assert reverse.model_dump(mode="json") == before


def test_input_permutations_produce_identical_assessment() -> None:
    links = (
        _link(EVIDENCE_A, SOURCE_A, "owner-a"),
        _link(EVIDENCE_B, SOURCE_B, "owner-b"),
        _link(EVIDENCE_C, SOURCE_A, "owner-a", EvidencePropositionRelation.MENTIONS),
    )
    expected = _aggregate(links).model_dump(mode="json")
    for order in permutations(links):
        assert _aggregate(order).model_dump(mode="json") == expected


def test_assessment_has_no_verification_or_truth_verdict_fields() -> None:
    result = _aggregate((_link(EVIDENCE_A, SOURCE_A, "owner-a"),))
    fields = set(type(result).model_fields)
    assert "verification_status" not in fields
    assert "truth_probability" not in fields
    assert "confidence" not in fields


def test_only_ineligible_support_results_in_no_eligible_support() -> None:
    result = _aggregate(
        (_link(EVIDENCE_A, SOURCE_A, "owner-a"),), {EVIDENCE_A: False}
    )
    assert result.owner_corroboration_status is OwnerCorroborationStatus.NO_ELIGIBLE_SUPPORT
