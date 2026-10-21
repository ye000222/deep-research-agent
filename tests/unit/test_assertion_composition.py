from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from app.domain.assertion_composition import (
    AssertionComposition,
    AssertionCompositionValidationError,
    AssertionGroup,
    AssertionGroupMember,
    AssertionGroupOperator,
    VerificationEligibility,
    proposition_verification_eligibility,
    validate_assertion_composition,
)
from app.domain.atomic_proposition import AtomicProposition, ClaimPropositionLink
from pydantic import ValidationError

FIXTURE = Path("tests/fixtures/assertion_composition_v2_1r.json")


def _fixture() -> dict[str, Any]:
    value = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert value["synthetic_only"] is True
    return value


def _composition(raw: dict[str, Any]) -> AssertionComposition:
    return AssertionComposition.model_validate(raw)


def _expected_map(raw: dict[str, str]) -> dict[UUID, VerificationEligibility]:
    return {UUID(key): VerificationEligibility(value) for key, value in raw.items()}


def test_synthetic_compositions_preserve_member_verification_eligibility() -> None:
    for case in _fixture()["cases"]:
        if case["composition"] is None:
            assert case["case_id"] in {
                "single_proposition_sparse",
                "conditional_is_one_conditioned_proposition",
                "causal_relation_is_not_boolean_group",
            }
            if case["case_id"] == "conditional_is_one_conditioned_proposition":
                assert "condition P" in case["handling"]
            if case["case_id"] == "causal_relation_is_not_boolean_group":
                assert "preserving cause" in case["handling"]
            continue
        composition = _composition(case["composition"])
        proposition_ids = {UUID(value) for value in case["proposition_ids"]}
        validate_assertion_composition(
            composition,
            proposition_ids=proposition_ids,
            claim_text=case["claim_text"],
        )
        expected = _expected_map(case["expected_eligibility"])
        actual = {
            proposition_id: proposition_verification_eligibility(composition, proposition_id)
            for proposition_id in expected
        }
        assert actual == expected, case["case_id"]


def test_contract_keeps_claim_and_atomic_proposition_as_first_class_types() -> None:
    assert "proposition_id" in ClaimPropositionLink.model_fields
    assert "claim_id" in ClaimPropositionLink.model_fields
    assert "proposition_id" in AtomicProposition.model_fields
    assert "assertion_composition" not in AtomicProposition.model_fields


def test_or_member_cannot_be_declared_independently_asserted() -> None:
    case = next(item for item in _fixture()["cases"] if item["case_id"] == "disjunction")
    raw = case["composition"]
    raw["groups"][0]["members"][0]["verification_eligibility"] = "INDEPENDENTLY_ASSERTED"
    composition = _composition(raw)
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition,
            proposition_ids={UUID(value) for value in case["proposition_ids"]},
            claim_text=case["claim_text"],
        )
    assert error.value.code == "verification_eligibility_mismatch"


def test_invalid_operator_and_missing_group_members_fail_schema_validation() -> None:
    case = next(item for item in _fixture()["cases"] if item["case_id"] == "disjunction")
    malformed = case["composition"].copy()
    malformed["groups"] = [case["composition"]["groups"][0] | {"operator": "NOR"}]
    with pytest.raises(ValidationError):
        _composition(malformed)

    malformed = case["composition"].copy()
    malformed["groups"] = [case["composition"]["groups"][0] | {"members": []}]
    with pytest.raises(ValidationError):
        _composition(malformed)


def test_duplicate_references_fail_closed() -> None:
    case = next(item for item in _fixture()["cases"] if item["case_id"] == "disjunction")
    group = case["composition"]["groups"][0]
    duplicate = group["members"][0] | {"position": 1}
    raw = case["composition"].copy()
    raw["groups"] = [group | {"members": [group["members"][0], duplicate]}]
    with pytest.raises(ValidationError, match="duplicate member references"):
        _composition(raw)

    # A leaf cannot be shared by two groups in one composition graph.
    root_id = UUID("20000000-0000-0000-0000-000000000091")
    left_id = UUID("20000000-0000-0000-0000-000000000092")
    right_id = UUID("20000000-0000-0000-0000-000000000093")
    shared_prop = UUID("00000000-0000-0000-0000-000000000091")
    left_group = AssertionGroup(
        group_id=left_id,
        operator=AssertionGroupOperator.AND,
        members=(
            AssertionGroupMember(
                member_type="proposition",
                position=0,
                proposition_id=shared_prop,
                verification_eligibility=VerificationEligibility.INDEPENDENTLY_ASSERTED,
            ),
            AssertionGroupMember(
                member_type="proposition",
                position=1,
                proposition_id=UUID("00000000-0000-0000-0000-000000000092"),
                verification_eligibility=VerificationEligibility.INDEPENDENTLY_ASSERTED,
            ),
        ),
    )
    right_group = left_group.model_copy(update={"group_id": right_id})
    root = AssertionGroup(
        group_id=root_id,
        operator=AssertionGroupOperator.AND,
        members=(
            AssertionGroupMember(member_type="group", position=0, group_id=left_id),
            AssertionGroupMember(member_type="group", position=1, group_id=right_id),
        ),
    )
    composition = AssertionComposition(
        claim_id=UUID("10000000-0000-0000-0000-000000000091"),
        root_group_id=root_id,
        groups=(root, left_group, right_group),
    )
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition,
            proposition_ids={shared_prop, UUID("00000000-0000-0000-0000-000000000092")},
            claim_text="A and B.",
        )
    assert error.value.code == "duplicate_proposition_reference"


def test_unknown_proposition_and_group_references_are_rejected() -> None:
    case = next(item for item in _fixture()["cases"] if item["case_id"] == "disjunction")
    composition = _composition(case["composition"])
    known_props = {UUID(value) for value in case["proposition_ids"]}
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition,
            proposition_ids={next(iter(known_props))},
            claim_text=case["claim_text"],
        )
    assert error.value.code == "unknown_proposition_reference"

    bad = case["composition"].copy()
    group = bad["groups"][0]
    group_member = {
        "member_type": "group",
        "position": 0,
        "group_id": "20000000-0000-0000-0000-000000000099",
    }
    bad["groups"] = [group | {"members": [group_member, group["members"][1]]}]
    bad["root_group_id"] = group["group_id"]
    composition = _composition(bad)
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition,
            proposition_ids=known_props,
            claim_text=case["claim_text"],
        )
    assert error.value.code == "unknown_group_reference"


def test_cycles_and_unreachable_groups_are_rejected() -> None:
    root_id = UUID("20000000-0000-0000-0000-000000000101")
    leaf_id = UUID("00000000-0000-0000-0000-000000000101")
    root = AssertionGroup(
        group_id=root_id,
        operator=AssertionGroupOperator.AND,
        members=(
            AssertionGroupMember(member_type="group", position=0, group_id=root_id),
            AssertionGroupMember(
                member_type="proposition",
                position=1,
                proposition_id=leaf_id,
                verification_eligibility=VerificationEligibility.INDEPENDENTLY_ASSERTED,
            ),
        ),
    )
    composition = AssertionComposition(
        claim_id=UUID("10000000-0000-0000-0000-000000000101"),
        root_group_id=root_id,
        groups=(root,),
    )
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition, proposition_ids={leaf_id}, claim_text="A and B."
        )
    assert error.value.code == "composition_cycle"

    unused_id = UUID("20000000-0000-0000-0000-000000000102")
    unused = AssertionGroup(
        group_id=unused_id,
        operator=AssertionGroupOperator.AND,
        members=(
            AssertionGroupMember(
                member_type="proposition",
                position=0,
                proposition_id=UUID("00000000-0000-0000-0000-000000000102"),
                verification_eligibility=VerificationEligibility.INDEPENDENTLY_ASSERTED,
            ),
            AssertionGroupMember(
                member_type="proposition",
                position=1,
                proposition_id=UUID("00000000-0000-0000-0000-000000000103"),
                verification_eligibility=VerificationEligibility.INDEPENDENTLY_ASSERTED,
            ),
        ),
    )
    root_without_edge = root.model_copy(update={"members": root.members[1:]})
    orphaned = composition.model_copy(update={"groups": (root_without_edge, unused)})
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            orphaned,
            proposition_ids={
                UUID("00000000-0000-0000-0000-000000000101"),
                UUID("00000000-0000-0000-0000-000000000102"),
                UUID("00000000-0000-0000-0000-000000000103"),
            },
            claim_text="A and B.",
        )
    assert error.value.code == "unreachable_group"


def test_source_spans_must_be_compatible_with_claim_and_parent_group() -> None:
    case = next(
        item
        for item in _fixture()["cases"]
        if item["case_id"] == "nested_conjunction_of_disjunction"
    )
    raw = case["composition"].copy()
    groups = list(raw["groups"])
    groups[1] = groups[1] | {"source_span": "Unit Z is offline"}
    raw["groups"] = groups
    composition = _composition(raw)
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition,
            proposition_ids={UUID(value) for value in case["proposition_ids"]},
            claim_text=case["claim_text"],
        )
    assert error.value.code == "source_span_not_in_claim"

    raw = case["composition"].copy()
    groups = list(raw["groups"])
    groups[0] = groups[0] | {"source_span": "Unit A is ready"}
    raw["groups"] = groups
    composition = _composition(raw)
    with pytest.raises(AssertionCompositionValidationError) as error:
        validate_assertion_composition(
            composition,
            proposition_ids={UUID(value) for value in case["proposition_ids"]},
            claim_text=case["claim_text"],
        )
    assert error.value.code == "nested_source_span_mismatch"


def test_negative_alternatives_are_independent_only_as_negated_conjuncts() -> None:
    case = next(
        item
        for item in _fixture()["cases"]
        if item["case_id"] == "negative_alternatives_as_asserted_negated_conjuncts"
    )
    assert case["proposition_polarities"] == {
        key: "NEGATED" for key in case["expected_eligibility"]
    }
    composition = _composition(case["composition"])
    assert composition.groups[0].operator is AssertionGroupOperator.AND
    validate_assertion_composition(
        composition,
        proposition_ids={UUID(value) for value in case["proposition_ids"]},
        claim_text=case["claim_text"],
    )


def test_future_verification_gate_reads_group_eligibility_not_leaf_text() -> None:
    case = next(item for item in _fixture()["cases"] if item["case_id"] == "disjunction")
    composition = _composition(case["composition"])
    member_ids = [UUID(value) for value in case["proposition_ids"]]
    assert all(
        proposition_verification_eligibility(composition, proposition_id)
        is VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
        for proposition_id in member_ids
    )
