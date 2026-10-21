from __future__ import annotations

from uuid import UUID

import pytest
from app.domain.assertion_composition import (
    AssertionComposition,
    AssertionCompositionValidationError,
    AssertionGroup,
    AssertionGroupMember,
    AssertionGroupOperator,
    CompositionSourceSpanTrace,
    VerificationEligibility,
    validate_assertion_composition,
)
from app.services.atomic_proposition_decomposer import _safe_structured_result
from app.services.logical_composition_v3 import validate_v3_decomposition_response

_CLAIM_ID = UUID("10000000-0000-4000-8000-000000000001")
_G0 = UUID("20000000-0000-4000-8000-000000000000")
_G1 = UUID("20000000-0000-4000-8000-000000000001")
_P0 = UUID("30000000-0000-4000-8000-000000000000")
_P1 = UUID("30000000-0000-4000-8000-000000000001")
_P2 = UUID("30000000-0000-4000-8000-000000000002")


def _proposition_ref(
    proposition_id: UUID,
    position: int,
    span: str | None,
    eligibility: VerificationEligibility,
) -> AssertionGroupMember:
    return AssertionGroupMember(
        member_type="proposition",
        position=position,
        proposition_id=proposition_id,
        verification_eligibility=eligibility,
        source_span=span,
    )


def _group_ref(group_id: UUID, position: int, span: str | None = None) -> AssertionGroupMember:
    return AssertionGroupMember(
        member_type="group", position=position, group_id=group_id, source_span=span
    )


def _composition(groups: tuple[AssertionGroup, ...]) -> AssertionComposition:
    return AssertionComposition(
        claim_id=_CLAIM_ID,
        root_expressions=(_group_ref(groups[-1].group_id, 0),),
        groups=groups,
    )


def _validate(
    composition: AssertionComposition,
    claim_text: str,
    proposition_source_spans: dict[UUID, str] | None = None,
    source_span_trace: CompositionSourceSpanTrace | None = None,
) -> None:
    proposition_ids = {
        member.proposition_id
        for group in composition.groups
        for member in group.members
        if member.member_type == "proposition" and member.proposition_id is not None
    }
    validate_assertion_composition(
        composition,
        proposition_ids=proposition_ids,
        claim_text=claim_text,
        proposition_source_spans=proposition_source_spans,
        source_span_trace=source_span_trace,
        reference_labels={
            **{
                group.group_id: (f"g{index}", "group")
                for index, group in enumerate(composition.groups)
            },
            **{
                proposition_id: (f"p{index}", "proposition")
                for index, proposition_id in enumerate(sorted(proposition_ids, key=str))
            },
        },
    )


def test_exact_group_and_member_spans_are_compatible() -> None:
    span = "North output reached 20 units"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=span,
        members=(
            _proposition_ref(_P0, 0, span, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, span, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )

    trace = CompositionSourceSpanTrace()
    _validate(_composition((group,)), f"Report: {span}.", {_P0: span, _P1: span}, trace)
    assert any(
        item["comparison_rule"] == "LITERAL_CLAIM_SUBSTRING_AND_PARENT_CONTAINS_CHILD"
        and item["child_source_span"] == span
        and item["compatibility_result"] == "PASS"
        for item in trace.checks
    )


def test_parent_group_may_contain_narrower_member_spans() -> None:
    parent = "North output reached 20 units or more than 20 units"
    left = "North output reached 20 units"
    right = "more than 20 units"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, left, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, right, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )

    _validate(_composition((group,)), f"Claim: {parent}.", {_P0: left, _P1: right})


def test_nested_group_uses_hierarchical_containment_not_ancestor_equality() -> None:
    parent = "North output reached 20 units and either South output reached 18 or 19"
    child = "either South output reached 18 or 19"
    p0_span = "North output reached 20 units"
    p1_span = "South output reached 18"
    p2_span = "19"
    outer = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.AND,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, p0_span, VerificationEligibility.INDEPENDENTLY_ASSERTED),
            _group_ref(_G1, 1, child),
        ),
    )
    inner = AssertionGroup(
        group_id=_G1,
        operator=AssertionGroupOperator.OR,
        source_span=child,
        members=(
            _proposition_ref(_P1, 0, p1_span, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P2, 1, p2_span, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )
    composition = _composition((inner, outer))

    _validate(composition, f"Claim: {parent}.", {_P0: p0_span, _P1: p1_span, _P2: p2_span})


def test_sibling_members_may_use_distinct_subspans() -> None:
    parent = "North production rose; South production remained stable"
    first = "North production rose"
    second = "South production remained stable"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.AND,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, first, VerificationEligibility.INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, second, VerificationEligibility.INDEPENDENTLY_ASSERTED),
        ),
    )

    _validate(_composition((group,)), parent, {_P0: first, _P1: second})


def test_shared_source_span_is_allowed_when_narrower_attribution_is_unavailable() -> None:
    shared = "The device was tested under controlled conditions"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.AND,
        source_span=shared,
        members=(
            _proposition_ref(_P0, 0, shared, VerificationEligibility.INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, shared, VerificationEligibility.INDEPENDENTLY_ASSERTED),
        ),
    )

    trace = CompositionSourceSpanTrace()
    _validate(_composition((group,)), shared, {_P0: shared, _P1: shared}, trace)
    assert any(
        item["parent_source_span"] == shared
        and item["child_source_span"] == shared
        and item["compatibility_result"] == "PASS"
        for item in trace.checks
    )


@pytest.mark.parametrize(
    ("claim", "parent", "child", "code"),
    [
        (
            "Alpha signal rose; Beta signal fell.",
            "Alpha signal rose",
            "Beta signal fell",
            "nested_source_span_mismatch",
        ),
        (
            "Alpha signal rose.",
            "Alpha signal rose",
            "Beta signal fell",
            "source_span_not_in_claim",
        ),
    ],
)
def test_disjoint_or_out_of_parent_child_span_fails_closed(
    claim: str, parent: str, child: str, code: str
) -> None:
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, child, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, parent, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )

    trace = CompositionSourceSpanTrace()
    with pytest.raises(AssertionCompositionValidationError, match=code):
        _validate(_composition((group,)), claim, {_P0: child, _P1: parent}, trace)
    failure = next(item for item in trace.checks if item["compatibility_result"] == "FAIL")
    assert failure["parent_source_span"] == parent
    assert failure["child_source_span"] == child
    assert failure["validation_reason_code"] == code


def test_nonexistent_text_range_is_rejected_as_invalid_source_span() -> None:
    claim = "Alpha signal rose and Beta signal fell."
    parent = "Alpha signal rose and Beta signal fell"
    invented_range = "Gamma signal rose"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.AND,
        source_span=parent,
        members=(
            _proposition_ref(
                _P0, 0, invented_range, VerificationEligibility.INDEPENDENTLY_ASSERTED
            ),
            _proposition_ref(
                _P1, 1, "Beta signal fell", VerificationEligibility.INDEPENDENTLY_ASSERTED
            ),
        ),
    )

    with pytest.raises(AssertionCompositionValidationError, match="source_span_not_in_claim"):
        _validate(_composition((group,)), claim)


def test_proposition_provenance_must_agree_with_explicit_member_span() -> None:
    claim = "Alpha signal rose; Beta signal fell."
    group_span = claim[:-1]
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=group_span,
        members=(
            _proposition_ref(
                _P0, 0, "Alpha signal rose", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
            _proposition_ref(
                _P1, 1, "Beta signal fell", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
        ),
    )

    with pytest.raises(
        AssertionCompositionValidationError, match="proposition_source_span_mismatch"
    ):
        _validate(
            _composition((group,)),
            claim,
            {_P0: "Beta signal fell", _P1: "Beta signal fell"},
        )


def test_missing_optional_span_is_not_fabricated_or_treated_as_mismatch() -> None:
    parent = "Alpha signal rose or Beta signal fell"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, None, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, None, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )

    _validate(_composition((group,)), parent)
    assert group.source_span == parent
    assert group.members[0].source_span is None


def test_spans_from_incompatible_source_material_fail_closed() -> None:
    source_claim = "A separate source says Gamma signal rose."
    active_claim = "Alpha signal rose or Beta signal fell."
    parent = "Alpha signal rose or Beta signal fell"
    foreign_span = "Gamma signal rose"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=parent,
        members=(
            _proposition_ref(
                _P0, 0, foreign_span, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
            _proposition_ref(
                _P1, 1, "Beta signal fell", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
        ),
    )

    assert foreign_span in source_claim
    with pytest.raises(AssertionCompositionValidationError, match="source_span_not_in_claim"):
        _validate(_composition((group,)), active_claim)


def test_successful_containment_records_bounded_pass_trace() -> None:
    parent = "North output reached 20 units or more than 20 units"
    left = "North output reached 20 units"
    right = "more than 20 units"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, left, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, right, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )
    trace = CompositionSourceSpanTrace(max_records=32)

    _validate(_composition((group,)), parent, {_P0: left, _P1: right}, trace)

    assert trace.truncated is False
    assert any(
        item["parent_ref"] == "g0"
        and item["child_ref"] == "p0"
        and item["parent_source_span"] == parent
        and item["child_source_span"] == left
        and item["compatibility_result"] == "PASS"
        for item in trace.checks
    )
    assert all("comparison_rule" in item and "field_path" in item for item in trace.checks)


def test_failed_parent_span_trace_retains_exact_comparison_and_reason() -> None:
    claim = "Alpha signal rose; Beta signal fell."
    parent = "Alpha signal rose"
    child = "Beta signal fell"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=parent,
        members=(
            _proposition_ref(_P0, 0, parent, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, child, VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )
    trace = CompositionSourceSpanTrace()

    with pytest.raises(AssertionCompositionValidationError, match="nested_source_span_mismatch"):
        _validate(_composition((group,)), claim, source_span_trace=trace)

    failed = next(item for item in trace.checks if item["compatibility_result"] == "FAIL")
    assert failed["parent_ref"] == "g0"
    assert failed["child_ref"] == "p1"
    assert failed["parent_source_span"] == parent
    assert failed["child_source_span"] == child
    assert failed["parent_compatibility_result"] == "FAIL"
    assert failed["validation_reason_code"] == "nested_source_span_mismatch"


def test_nested_group_failure_trace_identifies_graph_path() -> None:
    claim = "North result; South result."
    outer_span = "North result"
    inner_span = "South result"
    outer = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.AND,
        source_span=outer_span,
        members=(
            _proposition_ref(_P0, 0, outer_span, VerificationEligibility.INDEPENDENTLY_ASSERTED),
            _group_ref(_G1, 1, inner_span),
        ),
    )
    inner = AssertionGroup(
        group_id=_G1,
        operator=AssertionGroupOperator.OR,
        source_span=inner_span,
        members=(
            _proposition_ref(
                _P1, 0, "South result", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
            _proposition_ref(
                _P2, 1, "South result", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
        ),
    )
    trace = CompositionSourceSpanTrace()

    with pytest.raises(AssertionCompositionValidationError, match="nested_source_span_mismatch"):
        _validate(_composition((inner, outer)), claim, source_span_trace=trace)

    failed = next(item for item in trace.checks if item["compatibility_result"] == "FAIL")
    assert failed["parent_ref"] == "g1"
    assert failed["child_ref"] == "g0"
    assert failed["parent_source_span"] == outer_span
    assert failed["child_source_span"] == inner_span
    assert failed["validation_reason_code"] == "nested_source_span_mismatch"
    assert failed["graph_path"] == ["root[0]", "g1", "g0"]


def test_proposition_provenance_failure_trace_retains_resolved_span() -> None:
    claim = "Alpha signal rose; Beta signal fell."
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=claim[:-1],
        members=(
            _proposition_ref(
                _P0, 0, "Alpha signal rose", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
            _proposition_ref(
                _P1, 1, "Beta signal fell", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
        ),
    )
    trace = CompositionSourceSpanTrace()

    with pytest.raises(
        AssertionCompositionValidationError, match="proposition_source_span_mismatch"
    ):
        _validate(
            _composition((group,)),
            claim,
            {_P0: "Beta signal fell", _P1: "Beta signal fell"},
            trace,
        )

    failed = next(item for item in trace.checks if item["compatibility_result"] == "FAIL")
    assert failed["child_ref"] == "p0"
    assert failed["resolved_proposition_id"] == str(_P0)
    assert failed["child_source_span"] == "Alpha signal rose"
    assert failed["resolved_proposition_source_span"] == "Beta signal fell"
    assert failed["validation_reason_code"] == "proposition_source_span_mismatch"


def test_trace_limit_is_explicit_and_preserves_late_failure() -> None:
    claim = "Alpha; Beta; Gamma."
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span="Alpha; Beta",
        members=(
            _proposition_ref(_P0, 0, "Alpha", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
            _proposition_ref(_P1, 1, "Gamma", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ),
    )
    trace = CompositionSourceSpanTrace(max_records=2)

    with pytest.raises(AssertionCompositionValidationError, match="nested_source_span_mismatch"):
        _validate(_composition((group,)), claim, source_span_trace=trace)

    assert trace.truncated is True
    assert trace.omitted_check_count > 0
    assert len(trace.checks) == 2
    late_failure = trace.checks[-1]
    assert late_failure["child_ref"] == "p1"
    assert late_failure["child_source_span"] == "Gamma"
    assert late_failure["compatibility_result"] == "FAIL"
    assert late_failure["validation_reason_code"] == "nested_source_span_mismatch"


def test_diagnostic_recording_error_does_not_change_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = "Alpha signal rose or Beta signal fell"
    group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span=claim,
        members=(
            _proposition_ref(
                _P0,
                0,
                "Alpha signal rose",
                VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED,
            ),
            _proposition_ref(
                _P1,
                1,
                "Beta signal fell",
                VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED,
            ),
        ),
    )
    composition = _composition((group,))
    validate_assertion_composition(
        composition,
        proposition_ids={_P0, _P1},
        claim_text=claim,
        proposition_source_spans={_P0: "Alpha signal rose", _P1: "Beta signal fell"},
    )

    failing_collector = CompositionSourceSpanTrace()

    def fail_trace_record(_: dict[str, object]) -> None:
        raise RuntimeError("synthetic serialization failure")

    monkeypatch.setattr(failing_collector, "record", fail_trace_record)
    validate_assertion_composition(
        composition,
        proposition_ids={_P0, _P1},
        claim_text=claim,
        proposition_source_spans={_P0: "Alpha signal rose", _P1: "Beta signal fell"},
        source_span_trace=failing_collector,
    )
    assert failing_collector.recording_error_count > 0

    invalid_trace = CompositionSourceSpanTrace()
    monkeypatch.setattr(invalid_trace, "record", fail_trace_record)
    invalid_group = AssertionGroup(
        group_id=_G0,
        operator=AssertionGroupOperator.OR,
        source_span="Alpha signal rose",
        members=(
            _proposition_ref(
                _P0, 0, "Alpha signal rose", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
            _proposition_ref(
                _P1, 1, "Beta signal fell", VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
            ),
        ),
    )
    with pytest.raises(AssertionCompositionValidationError, match="nested_source_span_mismatch"):
        _validate(_composition((invalid_group,)), claim, source_span_trace=invalid_trace)
    assert invalid_trace.recording_error_count > 0


def test_safe_structured_diagnostics_retain_proposition_source_span() -> None:
    payload = {
        "schema_version": "atomic-proposition-schema-v3",
        "decomposition_status": "single",
        "propositions": [
            {
                "local_ref": "p0",
                "proposition_text": "Alpha signal rose",
                "subject": "Alpha",
                "predicate": "rose",
                "source_span": "Alpha signal rose",
                "typed_value": {"kind": "text", "text": "rose"},
            }
        ],
    }

    safe = _safe_structured_result(payload)

    assert safe is not None
    assert safe["propositions"][0]["local_ref"] == "p0"
    assert safe["propositions"][0]["source_span"] == "Alpha signal rose"


def test_v3_trace_is_observability_only_for_validity_and_eligibility() -> None:
    claim = "Alpha signal rose or Beta signal fell."
    structured = {
        "schema_version": "atomic-proposition-schema-v3",
        "decomposition_status": "decomposed",
        "propositions": [
            {
                "local_ref": "p0",
                "proposition_text": "Alpha signal rose",
                "subject": "Alpha",
                "predicate": "rose",
                "source_span": "Alpha signal rose",
                "typed_value": {"kind": "text", "text": "rose"},
                "polarity": "asserted",
            },
            {
                "local_ref": "p1",
                "proposition_text": "Beta signal fell",
                "subject": "Beta",
                "predicate": "fell",
                "source_span": "Beta signal fell",
                "typed_value": {"kind": "text", "text": "fell"},
                "polarity": "asserted",
            },
        ],
        "assertion_composition": {
            "root_expressions": [{"member_type": "group", "ref": "g0"}],
            "groups": [
                {
                    "local_ref": "g0",
                    "operator": "OR",
                    "source_span": claim[:-1],
                    "members": [
                        {"member_type": "proposition", "ref": "p0"},
                        {"member_type": "proposition", "ref": "p1"},
                    ],
                }
            ],
        },
    }
    traced = validate_v3_decomposition_response(
        structured, claim_text=claim, claim_id=str(_CLAIM_ID)
    )
    plain = validate_v3_decomposition_response(
        structured, claim_text=claim, claim_id=str(_CLAIM_ID)
    )

    assert traced.valid is True
    assert plain.valid is True
    assert traced.derived_member_eligibility == plain.derived_member_eligibility
    assert traced.derived_member_eligibility == {
        "p0": "NOT_INDEPENDENTLY_ASSERTED",
        "p1": "NOT_INDEPENDENTLY_ASSERTED",
    }
    assert traced.composition_source_span_trace
    assert traced.composition_source_span_trace_truncated is False
    resolved = next(
        item
        for item in traced.composition_source_span_trace
        if item["field_path"] == "propositions[p0].source_span"
    )
    assert resolved["resolved_proposition_id"] == traced.model_local_references["p0"]
    assert resolved["resolved_proposition_source_span"] == "Alpha signal rose"
