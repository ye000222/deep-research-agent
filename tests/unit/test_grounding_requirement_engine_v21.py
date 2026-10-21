from __future__ import annotations

from typing import Any

from app.domain.grounding_requirements import (
    GroundingBindingCandidate,
    GroundingNodeInput,
    GroundingNodeKind,
    GroundingNodeUsability,
    GroundingObligation,
    GroundingObligationDerivationResult,
    GroundingObligationStatus,
    GroundingObligationType,
    GroundingRequirementLevel,
)
from app.domain.source_context_grounding_v2 import GroundingDimensionStatus as DS
from app.services.grounding_requirement_engine_v21 import (
    GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
    derive_grounding_obligations_v21,
    evaluate_grounding_node_v21,
    grounding_reporting_metrics,
)


def _node(kind: GroundingNodeKind, payload: dict[str, Any], ref: str = "N1") -> GroundingNodeInput:
    return GroundingNodeInput(
        node_ref=ref, node_kind=kind, semantic_payload=payload, graph_role="synthetic"
    )


def _binding(
    ref: str,
    *,
    role: str = "ASSERTION_EVIDENCE",
    target_kind: str = "ASSERTION",
    target_ref: str = "A1",
    payload: dict[str, Any] | None = None,
    semantic: DS = DS.VALID,
    structural: DS = DS.VALID,
) -> GroundingBindingCandidate:
    return GroundingBindingCandidate(
        binding_ref=ref,
        evidence_ref="E-" + ref,
        semantic_role=role,
        target_kind=target_kind,
        target_ref=target_ref,
        canonical_payload=payload or {},
        textual_grounding=DS.VALID,
        structural_grounding=structural,
        semantic_support=semantic,
    )


def _eval(derivation, bindings, **kwargs):
    return evaluate_grounding_node_v21(
        derivation,
        tuple(bindings),
        node_textual_grounding=kwargs.pop("textual", DS.VALID),
        node_structural_grounding=kwargs.pop("structural", DS.VALID),
        **kwargs,
    )


def test_same_assertion_anchor_reused_across_three_explicit_compatible_views() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(GroundingNodeKind.ASSERTION, {"assertion_content": ["fact"]}, "A1")
    )
    assessment = _eval(derivation, [_binding("B1", payload={"assertion_content": ["fact"]})])
    assert [row.status for row in assessment.obligation_evaluations] == [
        GroundingObligationStatus.SATISFIED,
        GroundingObligationStatus.SATISFIED,
        GroundingObligationStatus.SATISFIED,
    ]
    assert assessment.usability is GroundingNodeUsability.GROUNDED_USABLE
    reuse = assessment.obligation_evaluations[1].provenance["authorized_reuse"]
    assert reuse[0]["reuse_group"] == "assertion-anchor:A1"
    assert reuse[0]["canonical_fact_identity"]


def test_same_anchor_binding_is_not_reused_for_incompatible_semantic_fact() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(GroundingNodeKind.ASSERTION, {"assertion_content": ["fact"]}, "A1")
    )
    first, second = derivation.obligations[:2]
    incompatible = second.model_copy(
        update={
            "obligation_ref": "other",
            "target_constraint": second.target_constraint.model_copy(
                update={"expected_value": ["different fact"]}
            ),
        }
    )
    custom = derivation.model_copy(update={"obligations": (first, incompatible)})
    row = _eval(custom, [_binding("B1", payload={"assertion_content": ["fact"]})])
    assert (
        row.obligation_evaluations[1].status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    )


def test_reuse_group_mismatch_is_rejected() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(GroundingNodeKind.ASSERTION, {"assertion_content": ["fact"]}, "A1")
    )
    first, second = derivation.obligations[:2]
    custom = derivation.model_copy(
        update={"obligations": (first, second.model_copy(update={"binding_reuse_group": "other"}))}
    )
    assessment = _eval(custom, [_binding("B1", payload={"assertion_content": ["fact"]})])
    assert any(
        item.reason.value == "BINDING_ALREADY_CONSUMED"
        for item in assessment.obligation_evaluations[1].match.excluded_candidates
    )


def test_structural_and_semantic_anchor_views_have_separate_status_dimensions() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(GroundingNodeKind.ASSERTION, {"assertion_content": ["fact"]}, "A1")
    )
    binding = _binding("B1", payload={"assertion_content": ["fact"]}, structural=DS.VALID)
    assessment = _eval(derivation, [binding])
    structural = assessment.obligation_evaluations[2]
    assert structural.status is GroundingObligationStatus.SATISFIED
    assert structural.provenance["authorized_reuse"][0]["use_modes"] == ["semantic", "structural"]


def _canonical_scope(kind: str, members: list[str], ref: str = "S1") -> GroundingNodeInput:
    return _node(
        GroundingNodeKind.SCOPE,
        {"canonical_scope_owner": True, "kind": kind, "member_refs": members},
        ref,
    )


def test_scope_canonical_owner_one_member_is_structural_not_semantic() -> None:
    derivation = derive_grounding_obligations_v21(_canonical_scope("CLAIM", ["A1"]))
    assert derivation.contract_version == GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1
    assert not derivation.obligations
    assessment = _eval(derivation, [], structural=DS.VALID)
    assert assessment.usability is GroundingNodeUsability.STRUCTURALLY_GROUNDED


def test_scope_canonical_owner_multiple_members_is_structural_once() -> None:
    derivation = derive_grounding_obligations_v21(_canonical_scope("CLAIM", ["A1", "A2"]))
    assert not derivation.obligations
    assert "scope_kind_and_members_owned_by_source_scope_graph" in derivation.diagnostics


def test_scope_canonical_owner_invalid_graph_fails_closed() -> None:
    derivation = derive_grounding_obligations_v21(_canonical_scope("CLAIM", ["A1"]))
    assessment = _eval(derivation, [], structural=DS.INVALID)
    assert assessment.usability is GroundingNodeUsability.UNGROUNDED_UNUSABLE


def test_scope_semantic_context_without_evidence_remains_missing() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(
            GroundingNodeKind.CONTEXT, {"context_type": "SCOPE", "semantic_value": {"domain": "X"}}
        )
    )
    assessment = _eval(derivation, [])
    assert (
        assessment.obligation_evaluations[0].status
        is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    )


def test_scope_membership_is_satisfied_by_exact_valid_graph_owner() -> None:
    obligation = GroundingObligation(
        obligation_ref="scope-member-A1",
        node_ref="A1",
        node_kind=GroundingNodeKind.ASSERTION,
        obligation_type=GroundingObligationType.SCOPE_MEMBERSHIP,
        expected_semantic_role_family=("SCOPE_STRUCTURE",),
        requirement_level=GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
        target_constraint={"target_kind": "SCOPE", "target_ref": "S1"},
        cardinality_policy="ALL_EXPECTED_MEMBERS",
        expected_member_key="A1",
        contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
        derivation_rule_id="assertion.scope_membership.graph_owner.v2.1",
    )
    derivation = GroundingObligationDerivationResult(
        node_ref="A1",
        node_kind=GroundingNodeKind.ASSERTION,
        status="RESOLVED",
        obligations=(obligation,),
    )
    result = _eval(
        derivation,
        [],
        canonical_scope_memberships={"S1": {"kind": "CLAIM", "members": ("A1",), "valid": True}},
    )
    assert result.obligation_evaluations[0].status is GroundingObligationStatus.SATISFIED
    assert (
        result.obligation_evaluations[0].provenance["canonical_owner"]["owner_type"]
        == "SourceScopeV2"
    )


def test_scope_wrong_target_does_not_match_graph_owner() -> None:
    obligation = GroundingObligation(
        obligation_ref="scope-member-A2",
        node_ref="A2",
        node_kind=GroundingNodeKind.ASSERTION,
        obligation_type=GroundingObligationType.SCOPE_MEMBERSHIP,
        expected_semantic_role_family=("SCOPE_STRUCTURE",),
        requirement_level=GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
        target_constraint={"target_kind": "SCOPE", "target_ref": "S1"},
        cardinality_policy="ALL_EXPECTED_MEMBERS",
        expected_member_key="A2",
        contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
        derivation_rule_id="assertion.scope_membership.graph_owner.v2.1",
    )
    derivation = GroundingObligationDerivationResult(
        node_ref="A2",
        node_kind=GroundingNodeKind.ASSERTION,
        status="RESOLVED",
        obligations=(obligation,),
    )
    result = _eval(
        derivation,
        [],
        canonical_scope_memberships={"S1": {"kind": "CLAIM", "members": ("A1",), "valid": True}},
    )
    assert (
        result.obligation_evaluations[0].status
        is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    )


def _metric_context(value: dict[str, Any], ref: str = "C1") -> GroundingNodeInput:
    return _node(
        GroundingNodeKind.CONTEXT, {"context_type": "MEASUREMENT", "semantic_value": value}, ref
    )


def test_typed_metric_identity_matches_supported_definition_binding() -> None:
    derivation = derive_grounding_obligations_v21(
        _metric_context({"metric": "AUROC", "value": 0.99})
    )
    candidate = _binding(
        "B-metric",
        role="MEASUREMENT_DEFINITION",
        target_kind="CONTEXT",
        target_ref="C1",
        payload={"semantic_value": {"metric": "AUROC", "value": 0.99}},
    )
    result = _eval(derivation, [candidate])
    identity = next(
        item
        for item in result.obligation_evaluations
        if item.obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
    )
    value = next(
        item
        for item in result.obligation_evaluations
        if item.obligation.obligation_type is GroundingObligationType.MEASUREMENT_VALUE
    )
    assert identity.status is GroundingObligationStatus.SATISFIED
    assert value.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    assert not identity.obligation.allow_binding_reuse


def test_numeric_result_alone_never_satisfies_metric_identity() -> None:
    derivation = derive_grounding_obligations_v21(
        _metric_context({"metric": "AUROC", "value": 0.99})
    )
    candidate = _binding(
        "B-number",
        role="MEASUREMENT_VALUE",
        target_kind="CONTEXT",
        target_ref="C1",
        payload={"semantic_value": {"value": 0.99}},
    )
    result = _eval(derivation, [candidate])
    identity = next(
        item
        for item in result.obligation_evaluations
        if item.obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
    )
    assert identity.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING


def test_supported_result_value_cannot_mask_unsupported_metric_definition() -> None:
    value = {"metric": "AUROC", "value": 0.99}
    derivation = derive_grounding_obligations_v21(_metric_context(value))
    candidates = [
        _binding(
            "B-definition",
            role="MEASUREMENT_DEFINITION",
            target_kind="CONTEXT",
            target_ref="C1",
            payload={"semantic_value": value},
            semantic=DS.INVALID,
        ),
        _binding(
            "B-result",
            role="MEASUREMENT_VALUE",
            target_kind="CONTEXT",
            target_ref="C1",
            payload={"semantic_value": value},
            semantic=DS.VALID,
        ),
    ]
    result = _eval(derivation, candidates)
    identity = next(
        item
        for item in result.obligation_evaluations
        if item.obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
    )
    assert identity.status is GroundingObligationStatus.NOT_SUPPORTED
    assert identity.match.matched_binding_refs == ("B-definition",)


def test_metric_identity_wrong_target_is_rejected() -> None:
    derivation = derive_grounding_obligations_v21(
        _metric_context({"metric": "AUROC", "value": 0.99})
    )
    candidate = _binding(
        "B-other",
        role="MEASUREMENT_DEFINITION",
        target_kind="CONTEXT",
        target_ref="C2",
        payload={"semantic_value": {"metric": "AUROC", "value": 0.99}},
    )
    result = _eval(derivation, [candidate])
    identity = next(
        item
        for item in result.obligation_evaluations
        if item.obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
    )
    assert identity.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING


def test_supported_result_value_does_not_mask_rejected_formal_definition() -> None:
    value = {"metric": "AUROC", "value": 0.99}
    derivation = derive_grounding_obligations_v21(_metric_context(value))
    candidates = [
        _binding(
            "B-result",
            role="MEASUREMENT_VALUE",
            target_kind="CONTEXT",
            target_ref="C1",
            payload={"semantic_value": value},
        ),
        _binding(
            "B-definition",
            role="MEASUREMENT_DEFINITION",
            target_kind="CONTEXT",
            target_ref="C1",
            payload={"semantic_value": value},
            semantic=DS.INVALID,
        ),
    ]
    result = _eval(derivation, candidates)
    identity = next(
        item
        for item in result.obligation_evaluations
        if item.obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
    )
    assert identity.status is GroundingObligationStatus.NOT_SUPPORTED
    assert identity.match.matched_binding_refs == ("B-definition",)


def test_claim6_like_invalid_graph_cannot_become_ready_under_v21() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(
            GroundingNodeKind.ASSERTION,
            {"assertion_content": ["attribution only"], "scope_ref": "S-invalid"},
            "A1",
        )
    )
    result = _eval(
        derivation,
        [_binding("B-anchor", payload={"assertion_content": ["attribution only"]})],
        structural=DS.INVALID,
    )
    assert result.usability is GroundingNodeUsability.UNGROUNDED_UNUSABLE
    assert result.verification_assertion_ready is False


def test_shared_attachment_does_not_leak_support_to_missing_sibling() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(
            GroundingNodeKind.ATTACHMENT,
            {
                "context_ref": "C1",
                "relation": "APPLIES_TO",
                "target_refs": ["A1", "A2"],
                "source_assertion_refs": ["A1", "A2"],
            },
            "AT1",
        )
    )
    candidate = _binding(
        "B-edge",
        role="ATTACHMENT",
        target_kind="ATTACHMENT",
        target_ref="AT1",
        payload={
            "attachment_ref": "AT1",
            "context_ref": "C1",
            "relation": "APPLIES_TO",
            "source_assertion_refs": ["A1", "A2"],
        },
    )
    result = _eval(
        derivation,
        [candidate],
        target_support_overrides={
            ("B-edge", "A1"): DS.VALID,
            ("B-edge", "A2"): DS.MISSING_EVIDENCE,
        },
    )
    rows = {
        item.obligation.expected_target_key: item
        for item in result.obligation_evaluations
        if item.obligation.expected_target_key
    }
    assert rows["A1"].status is GroundingObligationStatus.SATISFIED
    assert rows["A2"].status is GroundingObligationStatus.INSUFFICIENT_EVIDENCE


def test_reporting_splits_mixed_readiness_and_reuse_attempts_without_changing_status() -> None:
    derivation = derive_grounding_obligations_v21(
        _node(GroundingNodeKind.ASSERTION, {"assertion_content": ["fact"]}, "A1")
    )
    assessment = _eval(derivation, [_binding("B1", payload={"assertion_content": ["fact"]})])
    metrics = grounding_reporting_metrics((assessment,))
    assert metrics["assertion_root_count"] == 1
    assert metrics["assertion_root_ready_count"] == 1
    assert metrics["all_nodes_grounded_usable_count"] == 1
    assert metrics["authorized_reuse_success_count"] == 2
    assert metrics["successful_undeclared_semantic_reuse_count"] == 0
    assert assessment.verification_assertion_ready is True


def test_derivation_version_and_provenance_are_explicit() -> None:
    result = derive_grounding_obligations_v21(
        _node(GroundingNodeKind.ASSERTION, {"assertion_content": ["fact"]}, "A1")
    )
    assessment = _eval(result, [_binding("B1", payload={"assertion_content": ["fact"]})])
    assert result.contract_version == GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1
    assert all(
        item.contract_version == GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1
        for item in assessment.derivation.obligations
    )
    assert all(
        item.provenance["matcher_version"].endswith("v2.1")
        for item in assessment.obligation_evaluations
    )
