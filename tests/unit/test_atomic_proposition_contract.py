"""Contract-only tests for the V2.1R atomic proposition design."""

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from apps.api.app.domain.atomic_proposition import (
    ATOMIC_PROPOSITION_SCHEMA_VERSION,
    AtomicProposition,
    ClaimPropositionLink,
    ContradictionComparability,
    DecompositionMetadata,
    DecompositionStatus,
    GuardResult,
    MaterialQualifier,
    NumericOperator,
    NumericValue,
    PropositionPolarity,
    PropositionValue,
    PropositionValueKind,
    QualifierCategory,
    QualifierMateriality,
    complete_link_same_cluster,
    contradiction_comparability_guard,
    exact_structured_identity_guard,
)


def make_proposition(
    *,
    proposition_id: UUID | None = None,
    subject: str = "Device A",
    predicate: str = "has rated output",
    value: PropositionValue | None = None,
    polarity: PropositionPolarity = PropositionPolarity.ASSERTED,
    qualifiers: tuple[MaterialQualifier, ...] = (),
    modality: str | None = None,
    attribution: str | None = None,
    status: DecompositionStatus = DecompositionStatus.SINGLE_PROPOSITION,
    reason: str | None = None,
) -> AtomicProposition:
    return AtomicProposition(
        proposition_id=proposition_id or uuid4(),
        original_text="Device A has rated output of 12 kW.",
        subject=subject,
        predicate=predicate,
        value=value
        or PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(
                operator=NumericOperator.EXACT,
                value=Decimal("12"),
                unit="kW",
            ),
        ),
        polarity=polarity,
        qualifiers=qualifiers,
        modality=modality,
        attribution=attribution,
        decomposition=DecompositionMetadata(status=status, reason=reason),
    )


def qualifier(
    category: QualifierCategory,
    key: str,
    value: str,
    materiality: QualifierMateriality = QualifierMateriality.MATERIAL,
) -> MaterialQualifier:
    return MaterialQualifier(
        category=category,
        key=key,
        value=value,
        materiality=materiality,
    )


def test_synthetic_fixture_is_generic_and_loadable() -> None:
    fixture = json.loads(
        Path("tests/fixtures/atomic_proposition_v2_1r.json").read_text(encoding="utf-8")
    )
    assert fixture["synthetic_only"] is True
    assert fixture["fixture_version"] == "atomic-proposition-contract-synthetic-v1"
    assert len(fixture["cases"]) >= 10


def test_one_claim_can_link_to_one_or_multiple_propositions() -> None:
    claim_id = uuid4()
    first, second = uuid4(), uuid4()
    one = ClaimPropositionLink(
        claim_id=claim_id,
        proposition_id=first,
        proposition_index=0,
        decomposition_status=DecompositionStatus.SINGLE_PROPOSITION,
    )
    links = [
        one,
        ClaimPropositionLink(
            claim_id=claim_id,
            proposition_id=second,
            proposition_index=1,
            decomposition_status=DecompositionStatus.DECOMPOSED,
        ),
    ]
    assert len(links) == 2
    assert {link.claim_id for link in links} == {claim_id}
    assert {link.proposition_id for link in links} == {first, second}


def test_many_claims_can_share_a_proposition_without_merging_claim_identity() -> None:
    shared = uuid4()
    links = [
        ClaimPropositionLink(
            claim_id=uuid4(),
            proposition_id=shared,
            proposition_index=0,
            decomposition_status=DecompositionStatus.DECOMPOSED,
        )
        for _ in range(2)
    ]
    assert links[0].claim_id != links[1].claim_id
    assert links[0].proposition_id == links[1].proposition_id


@pytest.mark.parametrize(
    "left,right",
    [
        (
            (QualifierCategory.TIME, "period", "2024"),
            (QualifierCategory.TIME, "period", "2025"),
        ),
        (
            (QualifierCategory.VERSION, "version", "v1"),
            (QualifierCategory.VERSION, "version", "v2"),
        ),
        (
            (QualifierCategory.SCOPE, "region", "local"),
            (QualifierCategory.SCOPE, "region", "global"),
        ),
    ],
)
def test_material_qualifiers_are_identity_and_contradiction_guards(
    left: tuple[QualifierCategory, str, str],
    right: tuple[QualifierCategory, str, str],
) -> None:
    first = make_proposition(qualifiers=(qualifier(*left),))
    second = make_proposition(qualifiers=(qualifier(*right),))
    assert exact_structured_identity_guard(first, second) == GuardResult.NOT_SAME
    assert (
        contradiction_comparability_guard(first, second) == ContradictionComparability.INCOMPARABLE
    )


def test_negation_is_explicit_and_prevents_same_but_is_comparable() -> None:
    asserted = make_proposition()
    negated = make_proposition(polarity=PropositionPolarity.NEGATED)
    assert exact_structured_identity_guard(asserted, negated) == GuardResult.NOT_SAME
    assert (
        contradiction_comparability_guard(asserted, negated)
        == ContradictionComparability.COMPARABLE
    )


def test_unit_mismatch_is_not_same_and_not_contradiction_comparable() -> None:
    km = make_proposition()
    miles = make_proposition(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(
                operator=NumericOperator.EXACT,
                value=Decimal("12"),
                unit="mi",
            ),
        )
    )
    assert exact_structured_identity_guard(km, miles) == GuardResult.NOT_SAME
    assert contradiction_comparability_guard(km, miles) == ContradictionComparability.INCOMPARABLE


def test_comparison_target_and_metric_are_preserved_in_value_contract() -> None:
    value = PropositionValue(
        kind=PropositionValueKind.COMPARISON,
        numeric=NumericValue(
            operator=NumericOperator.APPROX,
            value=Decimal("10"),
            unit="percent",
        ),
        comparison_target="Device B",
        comparison_metric="latency",
        comparison_metric_status="explicit",
    )
    assert value.comparison_target == "Device B"
    assert value.comparison_metric == "latency"


def test_enumeration_membership_does_not_imply_exhaustive_set_equality() -> None:
    members = PropositionValue(
        kind=PropositionValueKind.SET,
        members=("Application A", "Application B"),
    )
    assert members.members == ("Application A", "Application B")
    # The contract stores listed members only; exhaustiveness is not inferred.
    assert "exhaustive" not in members.model_dump(mode="json")


def test_general_and_conditioned_specific_are_related_not_same() -> None:
    general = make_proposition()
    specific = make_proposition(
        qualifiers=(qualifier(QualifierCategory.CONDITION, "mode", "mode-a"),)
    )
    assert exact_structured_identity_guard(general, specific) == GuardResult.NOT_SAME
    assert (
        contradiction_comparability_guard(general, specific)
        == ContradictionComparability.INCOMPARABLE
    )
    # No subsumption relation is emitted by the minimal relation contract.


def test_composite_claims_can_share_only_their_common_proposition() -> None:
    shared, only_left, only_right = uuid4(), uuid4(), uuid4()
    claim_a = {shared, only_left}
    claim_b = {shared, only_right}
    assert claim_a & claim_b == {shared}
    assert claim_a != claim_b


def test_different_subjects_are_not_contradiction_comparable() -> None:
    first = make_proposition(subject="Device A")
    second = make_proposition(subject="Device B")
    assert (
        contradiction_comparability_guard(first, second) == ContradictionComparability.INCOMPARABLE
    )


def test_same_scope_distinct_numeric_values_are_only_comparable_not_classified() -> None:
    first = make_proposition()
    second = make_proposition(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(
                operator=NumericOperator.EXACT,
                value=Decimal("14"),
                unit="kW",
            ),
        )
    )
    assert contradiction_comparability_guard(first, second) == ContradictionComparability.COMPARABLE
    assert exact_structured_identity_guard(first, second) == GuardResult.NOT_SAME


def test_unresolved_decomposition_fails_closed() -> None:
    unresolved = make_proposition(
        subject=None,
        predicate=None,
        status=DecompositionStatus.UNRESOLVED_DECOMPOSITION,
        reason="Ambiguous attachment in source sentence.",
    )
    resolved = make_proposition()
    assert exact_structured_identity_guard(unresolved, resolved) == GuardResult.UNKNOWN
    assert (
        contradiction_comparability_guard(unresolved, resolved)
        == ContradictionComparability.UNKNOWN
    )


def test_unknown_qualifier_materiality_fails_closed() -> None:
    uncertain = make_proposition(
        qualifiers=(
            qualifier(
                QualifierCategory.OTHER,
                "unsure_context",
                "context-x",
                QualifierMateriality.UNKNOWN,
            ),
        )
    )
    assert exact_structured_identity_guard(uncertain, uncertain) == GuardResult.UNKNOWN


def test_complete_link_cluster_rejects_transitive_only_same() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    direct_ab_bc = frozenset({frozenset({a, b}), frozenset({b, c})})
    assert not complete_link_same_cluster((a, b, c), direct_ab_bc)
    full = frozenset({frozenset({a, b}), frozenset({b, c}), frozenset({a, c})})
    assert complete_link_same_cluster((a, b, c), full)


def test_schema_contract_version_is_explicit() -> None:
    assert ATOMIC_PROPOSITION_SCHEMA_VERSION == "atomic-proposition-schema-v2"
