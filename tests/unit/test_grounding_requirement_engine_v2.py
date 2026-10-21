from __future__ import annotations

from typing import Any

from app.domain.grounding_requirements import (
    GROUNDING_REQUIREMENT_CONTRACT_VERSION,
    GroundingBindingCandidate,
    GroundingCardinalityPolicy,
    GroundingNodeInput,
    GroundingNodeKind,
    GroundingNodeUsability,
    GroundingObligationStatus,
    GroundingObligationType,
)
from app.domain.source_context_grounding_v2 import GroundingDimensionStatus as DS
from app.services.grounding_requirement_engine import derive_grounding_obligations
from app.services.grounding_requirement_engine_v2 import (
    GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
    derive_grounding_obligations_v2,
    evaluate_grounding_node_v2,
)


def _node(kind: GroundingNodeKind, payload: dict[str, Any], ref: str = "n1") -> GroundingNodeInput:
    return GroundingNodeInput(
        node_ref=ref, node_kind=kind, semantic_payload=payload, graph_role="synthetic"
    )


def _binding(
    ref: str,
    *,
    role: str = "ATTACHMENT",
    target_kind: str = "ATTACHMENT",
    target_ref: str = "AT1",
    payload: dict[str, Any] | None = None,
    semantic: DS = DS.VALID,
    structural: DS = DS.VALID,
    scope_ref: str | None = None,
) -> GroundingBindingCandidate:
    return GroundingBindingCandidate(
        binding_ref=ref,
        evidence_ref=f"E-{ref}",
        semantic_role=role,
        target_kind=target_kind,
        target_ref=target_ref,
        canonical_payload=payload or {},
        textual_grounding=DS.VALID,
        structural_grounding=structural,
        semantic_support=semantic,
        scope_ref=scope_ref,
    )


def _eval(derivation, bindings, *, target_support_overrides=None):
    return evaluate_grounding_node_v2(
        derivation,
        tuple(bindings),
        node_textual_grounding=DS.VALID,
        node_structural_grounding=DS.VALID,
        target_support_overrides=target_support_overrides,
    )


def _attachment(*, targets: list[str] | None = None) -> GroundingNodeInput:
    return _node(
        GroundingNodeKind.ATTACHMENT,
        {
            "context_ref": "C1",
            "relation": "APPLIES_TO",
            "target_refs": targets or ["A1"],
            "source_assertion_refs": targets or ["A1"],
        },
        "AT1",
    )


def _edge_binding(
    *,
    context: str = "C1",
    relation: str = "APPLIES_TO",
    targets: list[str] | None = None,
    semantic: DS = DS.VALID,
    target_ref: str = "AT1",
    scope_ref: str | None = None,
) -> GroundingBindingCandidate:
    return _binding(
        "B-edge",
        target_ref=target_ref,
        payload={
            "attachment_ref": "AT1",
            "context_ref": context,
            "relation": relation,
            "source_assertion_refs": targets or ["A1"],
            "status": "RESOLVED",
        },
        semantic=semantic,
        scope_ref=scope_ref,
    )


def test_v2_contract_is_explicit_and_v1_semantics_remain_versioned() -> None:
    node = _attachment()
    assert (
        derive_grounding_obligations(node).contract_version
        == GROUNDING_REQUIREMENT_CONTRACT_VERSION
    )
    result = derive_grounding_obligations_v2(node)
    assert result.contract_version == GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2
    assert all(
        item.contract_version == GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2
        for item in result.obligations
    )
    assert all(item.obligation_ref.startswith("obl2_") for item in result.obligations)


def test_structural_and_semantic_views_share_same_attachment_fact_without_starvation() -> None:
    assessment = _eval(derive_grounding_obligations_v2(_attachment()), [_edge_binding()])
    assert [item.status for item in assessment.obligation_evaluations] == [
        GroundingObligationStatus.SATISFIED,
        GroundingObligationStatus.SATISFIED,
    ]
    assert assessment.obligation_evaluations[0].match.matched_binding_refs == ("B-edge",)
    assert assessment.obligation_evaluations[1].match.matched_binding_refs == ("B-edge",)
    assert (
        assessment.obligation_evaluations[0].provenance[
            "structural_check_consumes_semantic_binding"
        ]
        is False
    )
    assert assessment.usability is GroundingNodeUsability.GROUNDED_USABLE


def test_attachment_edge_dereference_checks_context_relation_and_assertion_target() -> None:
    expected = derive_grounding_obligations_v2(_attachment())
    cases = [
        (_edge_binding(), True),
        (_edge_binding(context="C-other"), False),
        (_edge_binding(relation="REPORTS"), False),
        (_edge_binding(targets=["A-other"]), False),
    ]
    for binding, valid in cases:
        assessment = _eval(expected, [binding])
        application = assessment.obligation_evaluations[1]
        assert (application.status is GroundingObligationStatus.SATISFIED) is valid
        if valid:
            dereference = application.provenance["graph_dereference"][0]
            assert dereference["path"] == [
                "SemanticEvidenceBinding",
                "ATTACHMENT",
                "context_ref/relation/source_assertion_refs",
            ]
        else:
            assert application.match.excluded_candidates


def test_structural_reference_success_without_semantic_edge_stays_missing() -> None:
    assessment = _eval(
        derive_grounding_obligations_v2(_attachment()),
        [_binding("B-struct", payload={"context_ref": "C1", "relation": "APPLIES_TO"})],
    )
    assert assessment.obligation_evaluations[0].status is GroundingObligationStatus.SATISFIED
    assert (
        assessment.obligation_evaluations[1].status
        is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    )


def test_semantic_binding_cannot_rescue_invalid_graph_reference() -> None:
    assessment = _eval(
        derive_grounding_obligations_v2(_attachment()),
        [_edge_binding(target_ref="AT-wrong")],
    )
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    assert all(not item.match.matched_binding_refs for item in assessment.obligation_evaluations)


def test_method_intrinsic_value_and_attachment_edge_have_single_relation_owner() -> None:
    method = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "METHOD",
            "semantic_value": {"method": "visual grounding"},
            "method_value": "visual grounding",
            "target_relation": "APPLIES_TO",
            "target_refs": ["A1"],
        },
        "C1",
    )
    method_result = derive_grounding_obligations_v2(method)
    assert [item.obligation_type for item in method_result.obligations] == [
        GroundingObligationType.METHOD_VALUE
    ]
    edge_result = derive_grounding_obligations_v2(_attachment())
    assert (
        sum(
            item.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
            for item in edge_result.obligations
        )
        == 1
    )
    assert (
        sum(
            item.obligation_type is GroundingObligationType.METHOD_RELATION
            for item in method_result.obligations
        )
        == 0
    )
    method_binding = _binding(
        "B-method",
        role="METHOD_VALUE",
        target_kind="CONTEXT",
        target_ref="C1",
        payload={"semantic_value": "visual grounding"},
    )
    assert (
        _eval(method_result, [method_binding]).usability is GroundingNodeUsability.GROUNDED_USABLE
    )


def test_distinct_method_attachment_edges_remain_distinct_owners() -> None:
    for relation in ("APPLIES_TO", "USES_FOR"):
        att = _attachment()
        payload = dict(att.semantic_payload)
        payload["relation"] = relation
        result = derive_grounding_obligations_v2(
            att.model_copy(update={"semantic_payload": payload})
        )
        assert (
            sum(
                item.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
                for item in result.obligations
            )
            == 1
        )


def test_measurement_definition_and_metric_identity_do_not_demand_result_value() -> None:
    for value in ({"measurement": "pixel anomaly map"}, {"metric": "Image AUROC"}):
        result = derive_grounding_obligations_v2(
            _node(
                GroundingNodeKind.CONTEXT, {"context_type": "MEASUREMENT", "semantic_value": value}
            )
        )
        kinds = {item.obligation_type for item in result.obligations}
        assert GroundingObligationType.METRIC_IDENTITY in kinds
        assert GroundingObligationType.MEASUREMENT_VALUE not in kinds


def test_actual_numeric_result_derives_value_and_definition_separately() -> None:
    result = derive_grounding_obligations_v2(
        _node(
            GroundingNodeKind.CONTEXT,
            {
                "context_type": "MEASUREMENT",
                "semantic_value": {"metric": "AUROC", "value": 83.1, "unit": "%"},
            },
        )
    )
    kinds = {item.obligation_type for item in result.obligations}
    assert kinds == {
        GroundingObligationType.METRIC_IDENTITY,
        GroundingObligationType.MEASUREMENT_VALUE,
        GroundingObligationType.CONTEXT_VALUE,
    }
    assert {item.expected_semantic_role_family[0] for item in result.obligations} == {
        "MEASUREMENT_DEFINITION",
        "MEASUREMENT_VALUE",
        "MEASUREMENT_UNIT",
    }


def test_threshold_and_comparator_are_not_misclassified_as_observed_value() -> None:
    result = derive_grounding_obligations_v2(
        _node(
            GroundingNodeKind.CONTEXT,
            {
                "context_type": "MEASUREMENT",
                "semantic_value": {
                    "measure": "average AU-PRO",
                    "comparator": "below",
                    "threshold": "60%",
                },
            },
        )
    )
    kinds = {item.obligation_type for item in result.obligations}
    assert GroundingObligationType.METRIC_IDENTITY in kinds
    assert any(
        item.expected_semantic_role_family == ("MEASUREMENT_COMPARATOR",)
        for item in result.obligations
    )
    assert GroundingObligationType.MEASUREMENT_VALUE not in kinds


def test_single_target_attachment_has_one_target_specific_obligation() -> None:
    result = derive_grounding_obligations_v2(_attachment())
    target_rows = [item for item in result.obligations if item.expected_target_key]
    assert len(target_rows) == 1
    assert target_rows[0].cardinality_policy is GroundingCardinalityPolicy.PER_TARGET


def test_attachment_source_scope_constraint_is_dereferenced_and_enforced() -> None:
    node = _attachment().model_copy(
        update={
            "semantic_payload": {
                **_attachment().semantic_payload,
                "source_scope_ref": "S1",
            }
        }
    )
    result = derive_grounding_obligations_v2(node)
    matching = _edge_binding(scope_ref="S1")
    mismatch = _edge_binding(scope_ref="S2").model_copy(
        update={
            "canonical_payload": {
                **_edge_binding(scope_ref="S2").canonical_payload,
                "source_scope_ref": "S2",
            }
        }
    )
    assert (
        _eval(result, [matching]).obligation_evaluations[1].status
        is GroundingObligationStatus.SATISFIED
    )
    rejected = _eval(result, [mismatch]).obligation_evaluations[1]
    assert rejected.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    assert any(item.reason.value == "SCOPE_MISMATCH" for item in rejected.match.excluded_candidates)


def test_multi_target_attachment_allows_contract_defined_shared_edge_support() -> None:
    node = _attachment(targets=["A1", "A2"])
    derivation = derive_grounding_obligations_v2(node)
    target_rows = [item for item in derivation.obligations if item.expected_target_key]
    assert len(target_rows) == 2
    assert len({item.binding_reuse_group for item in target_rows}) == 1
    assert all(item.allow_binding_reuse for item in target_rows)
    assessment = _eval(derivation, [_edge_binding(targets=["A1", "A2"])])
    assert all(
        item.status is GroundingObligationStatus.SATISFIED
        for item in assessment.obligation_evaluations
    )


def test_multi_target_attachment_keeps_unsupported_target_unresolved() -> None:
    node = _attachment(targets=["A1", "A2"])
    binding = _edge_binding(targets=["A1", "A2"])
    assessment = _eval(
        derive_grounding_obligations_v2(node),
        [binding],
        target_support_overrides={("B-edge", "A1"): DS.VALID, ("B-edge", "A2"): DS.INVALID},
    )
    statuses = {
        item.obligation.expected_target_key: item.status
        for item in assessment.obligation_evaluations
        if item.obligation.expected_target_key
    }
    assert statuses["A1"] is GroundingObligationStatus.SATISFIED
    assert statuses["A2"] is GroundingObligationStatus.NOT_SUPPORTED


def test_unrestricted_reuse_between_distinct_semantic_obligations_is_rejected() -> None:
    derivation = derive_grounding_obligations_v2(
        _node(GroundingNodeKind.CONTEXT, {"context_type": "OTHER", "semantic_value": "x"})
    )
    first = derivation.obligations[0].model_copy(
        update={"obligation_ref": "first", "expected_semantic_role_family": ("CONTEXT_VALUE",)}
    )
    second = first.model_copy(
        update={
            "obligation_ref": "second",
            "target_constraint": first.target_constraint.model_copy(
                update={"semantic_key": "alternate", "expected_value": "y"}
            ),
        }
    )
    custom = derivation.model_copy(update={"obligations": (first, second)})
    candidate = _binding(
        "B-reuse",
        role="CONTEXT_VALUE",
        target_kind="CONTEXT",
        target_ref="n1",
        payload={"semantic_value": "x", "alternate": "y"},
    )
    assessment = _eval(custom, [candidate])
    assert assessment.obligation_evaluations[0].status is GroundingObligationStatus.SATISFIED
    assert (
        assessment.obligation_evaluations[1].status
        is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    )
    assert any(
        item.reason.value == "BINDING_ALREADY_CONSUMED"
        for item in assessment.obligation_evaluations[1].match.excluded_candidates
    )


def test_aggregation_remains_fail_closed_for_invalid_semantic_support() -> None:
    assessment = _eval(
        derive_grounding_obligations_v2(_attachment()), [_edge_binding(semantic=DS.INVALID)]
    )
    assert assessment.usability is GroundingNodeUsability.UNGROUNDED_UNUSABLE
    assert assessment.verification_assertion_ready is False
