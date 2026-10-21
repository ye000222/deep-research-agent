from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from app.domain.grounding_requirements import (
    GROUNDING_REQUIREMENT_CONTRACT_VERSION,
    GroundingBindingCandidate,
    GroundingCardinalityPolicy,
    GroundingDerivationStatus,
    GroundingNodeInput,
    GroundingNodeKind,
    GroundingNodeUsability,
    GroundingObligationStatus,
    GroundingRequirementLevel,
    ZeroObligationReason,
)
from app.domain.source_context_grounding_v2 import (
    DimensionGrounding,
    EvidenceTargetKind,
    SemanticEvidenceBinding,
    SemanticEvidenceRole,
)
from app.domain.source_context_grounding_v2 import (
    GroundingDimensionStatus as DS,
)
from app.services.grounding_requirement_engine import (
    derive_grounding_obligations,
    evaluate_grounding_node,
    project_semantic_evidence_bindings,
)

FIXTURE_PATH = Path("tests/fixtures/grounding_requirement_shadow_v1.json")
FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def _node(
    kind: GroundingNodeKind = GroundingNodeKind.ASSERTION,
    payload: dict[str, Any] | None = None,
    *,
    ref: str = "n0",
) -> GroundingNodeInput:
    return GroundingNodeInput(
        node_ref=ref,
        node_kind=kind,
        graph_role="synthetic_fixture_node",
        semantic_payload=payload or {"assertion_content": {"subject": "x", "value": 1}},
    )


def _binding(
    ref: str,
    *,
    evidence: str | None = None,
    role: str = "ASSERTION_EVIDENCE",
    target_kind: str = "ASSERTION",
    target_ref: str = "n0",
    payload: dict[str, Any] | None = None,
    relation: str | None = None,
    member_key: str | None = None,
    value_key: str | None = None,
    text: DS = DS.VALID,
    structure: DS = DS.VALID,
    semantic: DS = DS.VALID,
) -> GroundingBindingCandidate:
    return GroundingBindingCandidate(
        binding_ref=ref,
        evidence_ref=evidence or f"e-{ref}",
        semantic_role=role,
        target_kind=target_kind,
        target_ref=target_ref,
        canonical_payload=payload or {},
        relation=relation,
        member_key=member_key,
        value_key=value_key,
        textual_grounding=text,
        structural_grounding=structure,
        semantic_support=semantic,
    )


def _assertion_candidates(
    *,
    semantic: DS = DS.VALID,
    text: DS = DS.VALID,
    structure: DS = DS.VALID,
    evidence: str = "e-assertion",
) -> tuple[GroundingBindingCandidate, ...]:
    return (
        _binding(
            "b-assertion",
            evidence=evidence,
            payload={"assertion_content": {"subject": "x", "value": 1}},
            semantic=semantic,
            text=text,
            structure=structure,
        ),
    )


def _evaluate(
    derivation: Any,
    candidates: tuple[GroundingBindingCandidate, ...],
    *,
    text: DS = DS.VALID,
    structure: DS = DS.VALID,
):
    return evaluate_grounding_node(
        derivation,
        candidates,
        node_textual_grounding=text,
        node_structural_grounding=structure,
    )


def test_fixture_catalog_is_synthetic_and_covers_required_scenarios() -> None:
    expected_ids = {
        "assertion_fully_grounded",
        "assertion_without_irrelevant_scope",
        "method_all_relations_grounded",
        "method_relation_missing",
        "epistemic_agent_missing",
        "epistemic_wrong_target",
        "attachment_structural_and_semantic",
        "attachment_semantic_missing",
        "shared_context_two_targets",
        "shared_context_one_target_unsupported",
        "numeric_exact_value",
        "numeric_association_missing",
        "enumeration_count_only",
        "enumeration_count_and_members",
        "enumeration_member_missing",
        "optional_invalid_is_diagnostic",
        "required_not_supported",
        "required_ambiguous",
        "required_insufficient",
        "textual_invalid",
        "structural_invalid",
        "zero_structural_only",
        "derivation_unresolved",
        "multiple_distinct_candidates",
        "duplicate_evidence_candidates",
        "per_target_partial",
        "per_value_partial",
    }
    assert len(FIXTURE_CASES) == 27
    assert {case["id"] for case in FIXTURE_CASES} == expected_ids
    assert all(case["expected"] for case in FIXTURE_CASES)
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["synthetic_only"] is True
    assert fixture["contract_version"] == GROUNDING_REQUIREMENT_CONTRACT_VERSION


def test_assertion_obligations_are_deterministic_and_independent_of_labels() -> None:
    node = _node()
    first = derive_grounding_obligations(node)
    second = derive_grounding_obligations(node)
    assert first == second
    assert len(first.obligations) == 3
    assert len({item.obligation_ref for item in first.obligations}) == 3
    assert all(
        item.contract_version == GROUNDING_REQUIREMENT_CONTRACT_VERSION
        for item in first.obligations
    )
    assert all(item.allow_binding_reuse for item in first.obligations)
    # Labels may exist in surrounding evaluation metadata but never affect derivation.
    labelled = _node(
        payload={
            "assertion_content": {"subject": "x", "value": 1},
            "human_label": "SUPPORTED",
            "support_outcome": "VALID",
        }
    )
    assert derive_grounding_obligations(labelled) == first


def test_assertion_can_reuse_one_binding_only_with_explicit_group() -> None:
    derivation = derive_grounding_obligations(_node())
    assessment = _evaluate(derivation, _assertion_candidates())
    assert assessment.usability is GroundingNodeUsability.GROUNDED_USABLE
    assert all(
        item.match.matched_binding_refs == ("b-assertion",)
        for item in assessment.obligation_evaluations
    )
    assert assessment.verification_assertion_ready is True


def test_assertion_without_explicit_scope_does_not_invent_scope_obligation() -> None:
    derivation = derive_grounding_obligations(_node())
    assert not any(
        item.obligation_type.value == "SCOPE_MEMBERSHIP" for item in derivation.obligations
    )


def test_method_derives_value_and_each_target_relation_independently() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "METHOD",
            "method_value": "survey",
            "target_relation": "USES_FOR",
            "target_refs": ["a", "b"],
        },
    )
    derivation = derive_grounding_obligations(node)
    assert len(derivation.obligations) == 5
    target_obligations = [item for item in derivation.obligations if item.expected_target_key]
    assert {item.expected_target_key for item in target_obligations} == {"a", "b"}
    assert all(
        item.cardinality_policy is GroundingCardinalityPolicy.PER_TARGET
        for item in target_obligations
    )


def test_method_missing_relation_binding_is_reported_as_omission() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "METHOD",
            "method_value": "survey",
            "target_relation": "USES_FOR",
            "target_refs": ["a"],
        },
    )
    derivation = derive_grounding_obligations(node)
    candidates = (
        _binding(
            "method-value",
            role="METHOD_VALUE",
            target_kind="CONTEXT",
            payload={"semantic_value": "survey"},
        ),
    )
    assessment = _evaluate(derivation, candidates)
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    assert len(assessment.extraction_omissions) == 2


def test_epistemic_agent_is_separate_and_wrong_target_does_not_match() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "EPISTEMIC",
            "operator": "predicts",
            "agent_ref": "article",
            "target_refs": ["assertion-1"],
        },
    )
    derivation = derive_grounding_obligations(node)
    assert {item.obligation_type.value for item in derivation.obligations} >= {
        "EPISTEMIC_OPERATOR",
        "EPISTEMIC_AGENT",
    }
    candidates = (
        _binding(
            "op", role="EPISTEMIC_OPERATOR", target_kind="CONTEXT", payload={"operator": "predicts"}
        ),
        _binding(
            "target",
            role="ATTACHMENT",
            target_kind="CONTEXT",
            payload={"target_ref": "other"},
            relation="REPORTS",
        ),
    )
    assessment = _evaluate(derivation, candidates)
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    agent_eval = next(
        item
        for item in assessment.obligation_evaluations
        if item.obligation.obligation_type.value == "EPISTEMIC_AGENT"
    )
    target_eval = next(
        item
        for item in assessment.obligation_evaluations
        if item.obligation.obligation_type.value == "EPISTEMIC_TARGET_RELATIONSHIP"
    )
    assert agent_eval.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    assert target_eval.match.excluded_candidates


def test_attachment_requires_both_structural_reference_and_semantic_applicability() -> None:
    node = _node(
        GroundingNodeKind.ATTACHMENT,
        {"context_ref": "ctx", "relation": "APPLIES_TO", "target_refs": ["assertion-1"]},
    )
    derivation = derive_grounding_obligations(node)
    candidates = (
        _binding(
            "struct", role="ATTACHMENT", target_kind="ATTACHMENT", payload={"context_ref": "ctx"}
        ),
    )
    assessment = _evaluate(derivation, candidates)
    assert (
        derivation.obligations[0].requirement_level
        is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
    )
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    assert (
        sum(
            item.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
            for item in assessment.obligation_evaluations
        )
        == 1
    )


def test_shared_scope_requires_each_target_and_scope_member() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "SHARED_SCOPE",
            "semantic_value": "region-X",
            "member_refs": ["n1", "n2"],
            "target_refs": ["a1", "a2"],
        },
    )
    derivation = derive_grounding_obligations(node)
    assert len([item for item in derivation.obligations if item.expected_target_key]) == 2
    assert len([item for item in derivation.obligations if item.expected_member_key]) == 2
    candidates = [
        _binding(
            "scope-value",
            role="SCOPE_VALUE",
            target_kind="CONTEXT",
            payload={"semantic_value": "region-X"},
        ),
        _binding(
            "member-1",
            role="SCOPE_STRUCTURE",
            target_kind="SCOPE",
            payload={"member_ref": "n1"},
            member_key="n1",
        ),
        _binding(
            "member-2",
            role="SCOPE_STRUCTURE",
            target_kind="SCOPE",
            payload={"member_ref": "n2"},
            member_key="n2",
        ),
    ]
    for target in ("a1", "a2"):
        candidates.append(
            _binding(
                f"target-{target}",
                role="ATTACHMENT",
                target_kind="ATTACHMENT",
                payload={"target_ref": target},
                relation="APPLIES_TO",
            )
        )
    assessment = _evaluate(derivation, tuple(candidates))
    assert assessment.usability is GroundingNodeUsability.GROUNDED_USABLE


def test_scope_branch_preserves_value_plus_target_obligations() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {"context_type": "SCOPE", "semantic_value": "global", "target_refs": ["a"]},
    )
    derivation = derive_grounding_obligations(node)
    assert any(item.obligation_type.value == "SCOPE_VALUE" for item in derivation.obligations)
    assert any(
        item.obligation_type.value == "TARGET_APPLICABILITY" for item in derivation.obligations
    )


def test_measurement_branch_derives_measurement_metric_and_numeric_association() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "MEASUREMENT",
            "semantic_value": 99,
            "metric_identity": "accuracy",
            "associations": [{"value_key": "v99", "relation": "MEASURES"}],
        },
    )
    derivation = derive_grounding_obligations(node)
    assert {item.obligation_type.value for item in derivation.obligations} == {
        "MEASUREMENT_VALUE",
        "METRIC_IDENTITY",
        "NUMERIC_ASSOCIATION",
    }
    assoc = next(
        item
        for item in derivation.obligations
        if item.obligation_type.value == "NUMERIC_ASSOCIATION"
    )
    value_candidate = _binding(
        "measurement-value",
        role="MEASUREMENT_VALUE",
        target_kind="CONTEXT",
        payload={"semantic_value": 99},
    )
    metric_candidate = _binding(
        "metric-value",
        role="MEASUREMENT_DEFINITION",
        target_kind="CONTEXT",
        payload={"metric_identity": "accuracy"},
    )
    assessment = _evaluate(derivation, (value_candidate, metric_candidate))
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    value_result = next(
        item
        for item in assessment.obligation_evaluations
        if item.obligation.obligation_type.value == "MEASUREMENT_VALUE"
    )
    association_result = next(
        item
        for item in assessment.obligation_evaluations
        if item.obligation.obligation_type.value == "NUMERIC_ASSOCIATION"
    )
    assert value_result.status is GroundingObligationStatus.SATISFIED
    assert association_result.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
    assert assoc.cardinality_policy is GroundingCardinalityPolicy.PER_VALUE


def test_measurement_numeric_association_can_be_fully_matched() -> None:
    association = {"value_key": "v99", "relation": "MEASURES"}
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "MEASUREMENT",
            "semantic_value": 99,
            "metric_identity": "accuracy",
            "associations": [association],
        },
    )
    derivation = derive_grounding_obligations(node)
    candidates = (
        _binding(
            "measurement",
            role="MEASUREMENT_VALUE",
            target_kind="CONTEXT",
            payload={"semantic_value": 99},
        ),
        _binding(
            "metric",
            role="MEASUREMENT_DEFINITION",
            target_kind="CONTEXT",
            payload={"metric_identity": "accuracy"},
        ),
        _binding(
            "association",
            role="ATTACHMENT",
            target_kind="CONTEXT",
            payload={"association": association},
            relation="MEASURES",
            value_key="v99",
        ),
    )
    assert _evaluate(derivation, candidates).usability is GroundingNodeUsability.GROUNDED_USABLE


def test_enumeration_count_only_is_valid_and_members_are_not_invented() -> None:
    derivation = derive_grounding_obligations(
        _node(GroundingNodeKind.GROUP, {"group_type": "ENUMERATION", "count": 4})
    )
    assert len(derivation.obligations) == 1
    assert derivation.obligations[0].obligation_type.value == "ENUMERATION_COUNT"


def test_enumeration_requires_every_explicit_member() -> None:
    derivation = derive_grounding_obligations(
        _node(
            GroundingNodeKind.GROUP,
            {"group_type": "ENUMERATION", "count": 2, "members": ["a", "b"]},
        )
    )
    candidates = (
        _binding("count", role="CONTEXT_VALUE", target_kind="CONTEXT", payload={"count": 2}),
        _binding(
            "member-a",
            role="CONTEXT_VALUE",
            target_kind="CONTEXT",
            payload={"member": "a"},
            member_key="a",
        ),
    )
    assessment = _evaluate(derivation, candidates)
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    assert any(
        item.obligation.expected_member_key == "b"
        and item.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
        for item in assessment.obligation_evaluations
    )


def test_optional_invalid_binding_is_diagnostic_only() -> None:
    derivation = derive_grounding_obligations(_node())
    optional = derivation.obligations[0].model_copy(
        update={
            "obligation_ref": "optional",
            "requirement_level": GroundingRequirementLevel.OPTIONAL_CORROBORATION,
            "expected_semantic_role_family": ("CONTEXT_VALUE",),
            "target_constraint": derivation.obligations[0].target_constraint.model_copy(
                update={
                    "target_kind": "CONTEXT",
                    "semantic_key": "optional_support",
                    "expected_value": "corroboration",
                }
            ),
            "allow_binding_reuse": False,
            "binding_reuse_group": None,
        }
    )
    derivation = derivation.model_copy(update={"obligations": (*derivation.obligations, optional)})
    optional_candidate = _binding(
        "optional",
        role="CONTEXT_VALUE",
        target_kind="CONTEXT",
        payload={"optional_support": "corroboration"},
        semantic=DS.INVALID,
    )
    assessment = _evaluate(derivation, (*_assertion_candidates(), optional_candidate))
    assert assessment.usability is GroundingNodeUsability.GROUNDED_USABLE
    assert "optional_corroboration_failure_diagnostic_only" in assessment.diagnostics


@pytest.mark.parametrize(
    ("semantic", "expected"),
    [
        (DS.INVALID, GroundingNodeUsability.UNGROUNDED_UNUSABLE),
        (DS.AMBIGUOUS, GroundingNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS),
        (DS.MISSING_EVIDENCE, GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED),
        (DS.NOT_EVALUATED, GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED),
    ],
)
def test_required_support_states_have_distinct_aggregation(
    semantic: DS, expected: GroundingNodeUsability
) -> None:
    assessment = _evaluate(
        derive_grounding_obligations(_node()), _assertion_candidates(semantic=semantic)
    )
    assert assessment.usability is expected
    assert assessment.verification_assertion_ready is False


def test_textual_and_structural_failures_are_fail_closed() -> None:
    derivation = derive_grounding_obligations(_node())
    textual = _evaluate(derivation, _assertion_candidates(text=DS.INVALID))
    structural = _evaluate(derivation, _assertion_candidates(structure=DS.INVALID))
    assert textual.usability is GroundingNodeUsability.UNGROUNDED_UNUSABLE
    assert structural.usability is GroundingNodeUsability.UNGROUNDED_UNUSABLE


def test_explicit_zero_obligation_reasons_are_not_conflated() -> None:
    structural_node = _node(
        GroundingNodeKind.GROUP, {"group_type": "STRUCTURAL_GROUP", "member_refs": ["n1"]}
    )
    metadata_node = _node(
        GroundingNodeKind.METADATA, {"metadata_contract": "NON_SEMANTIC_METADATA"}
    )
    structural = derive_grounding_obligations(structural_node)
    metadata = derive_grounding_obligations(metadata_node)
    assert structural.zero_obligation_reason is ZeroObligationReason.STRUCTURAL_ONLY_NODE
    assert metadata.zero_obligation_reason is ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED
    assert (
        _evaluate(structural, (), text=DS.VALID, structure=DS.VALID).usability
        is GroundingNodeUsability.STRUCTURALLY_GROUNDED
    )
    assert (
        _evaluate(metadata, ()).usability is GroundingNodeUsability.NO_SEMANTIC_GROUNDING_REQUIRED
    )


def test_zero_structural_node_with_invalid_structure_is_not_usable() -> None:
    node = _node(GroundingNodeKind.GROUP, {"group_type": "STRUCTURAL_GROUP", "member_refs": ["n1"]})
    assessment = _evaluate(derive_grounding_obligations(node), (), structure=DS.INVALID)
    assert assessment.usability is GroundingNodeUsability.UNGROUNDED_UNUSABLE
    assert assessment.verification_assertion_ready is False


def test_unresolved_derivation_has_no_obligations_and_is_not_evaluated() -> None:
    result = derive_grounding_obligations(
        _node(GroundingNodeKind.CONTEXT, {"context_type": "UNKNOWN"})
    )
    assert result.status is GroundingDerivationStatus.UNSUPPORTED_NODE_SHAPE
    assert result.obligations == ()
    assessment = _evaluate(result, ())
    assert assessment.usability is GroundingNodeUsability.NOT_EVALUATED
    assert assessment.verification_assertion_ready is False


def test_multiple_candidates_are_ambiguous_but_duplicate_evidence_is_diagnosed() -> None:
    derivation = derive_grounding_obligations(_node())
    multiple = (
        _assertion_candidates(evidence="e1")[0],
        _assertion_candidates(evidence="e2")[0].model_copy(update={"binding_ref": "b2"}),
    )
    assessment = _evaluate(derivation, multiple)
    assert assessment.usability is GroundingNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS
    duplicate = (
        _assertion_candidates(evidence="same")[0],
        _assertion_candidates(evidence="same")[0].model_copy(update={"binding_ref": "b2"}),
    )
    duplicate_assessment = _evaluate(derivation, duplicate)
    assert duplicate_assessment.usability is GroundingNodeUsability.GROUNDED_USABLE
    assert all(
        "duplicate_binding_candidates_share_evidence_ref" in item.diagnostics
        for item in duplicate_assessment.obligation_evaluations
    )


def test_role_target_and_payload_mismatches_are_explicit_exclusions() -> None:
    derivation = derive_grounding_obligations(_node())
    wrong_role = _binding(
        "wrong-role",
        role="CONTEXT_VALUE",
        payload={"assertion_content": {"subject": "x", "value": 1}},
    )
    wrong_target = _binding(
        "wrong-target",
        target_ref="another",
        payload={"assertion_content": {"subject": "x", "value": 1}},
    )
    wrong_payload = _binding("wrong-payload", payload={"assertion_content": "wrong"})
    assessment = _evaluate(derivation, (wrong_role, wrong_target, wrong_payload))
    reasons = {
        item.reason.value
        for result in assessment.obligation_evaluations
        for item in result.match.excluded_candidates
    }
    assert {"ROLE_MISMATCH", "TARGET_REF_MISMATCH"}.issubset(reasons)
    assert "PAYLOAD_MISMATCH" in reasons
    assert assessment.extraction_omissions


def test_per_target_policy_reports_partial_target_coverage() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {"context_type": "METHOD", "target_relation": "USES_FOR", "target_refs": ["a", "b"]},
    )
    derivation = derive_grounding_obligations(node)
    candidate = _binding(
        "method-a",
        role="METHOD_RELATION",
        target_kind="CONTEXT",
        payload={"target_ref": "a"},
        relation="USES_FOR",
    )
    assessment = _evaluate(derivation, (candidate,))
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED


def test_per_value_policy_rejects_wrong_value_key() -> None:
    association = {"value_key": "wanted", "relation": "MEASURES"}
    node = _node(
        GroundingNodeKind.CONTEXT, {"context_type": "MEASUREMENT", "associations": [association]}
    )
    derivation = derive_grounding_obligations(node)
    candidate = _binding(
        "assoc",
        role="ATTACHMENT",
        target_kind="CONTEXT",
        payload={"association": association},
        relation="MEASURES",
        value_key="other",
    )
    assessment = _evaluate(derivation, (candidate,))
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    assert any(
        ex.reason.value == "VALUE_MISMATCH"
        for item in assessment.obligation_evaluations
        for ex in item.match.excluded_candidates
    )


def test_provenance_and_omission_diagnostics_preserve_contract_lineage() -> None:
    derivation = derive_grounding_obligations(_node())
    assessment = _evaluate(derivation, ())
    assert len(assessment.extraction_omissions) == 3
    for evaluation in assessment.obligation_evaluations:
        assert evaluation.provenance["contract_version"] == GROUNDING_REQUIREMENT_CONTRACT_VERSION
        assert (
            evaluation.provenance["derivation_rule_id"] == evaluation.obligation.derivation_rule_id
        )
        assert evaluation.provenance["obligation_ref"] == evaluation.obligation.obligation_ref
        assert evaluation.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING


def test_shadow_readiness_is_only_a_boolean_and_never_runs_verification() -> None:
    assessment = _evaluate(derive_grounding_obligations(_node()), _assertion_candidates())
    assert assessment.verification_assertion_ready is True
    assert not hasattr(assessment, "verification_result")
    assert not hasattr(assessment, "verdict")


def test_method_value_relation_and_target_can_all_be_satisfied() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "METHOD",
            "method_value": "survey",
            "target_relation": "USES_FOR",
            "target_refs": ["a"],
        },
    )
    derivation = derive_grounding_obligations(node)
    candidates = (
        _binding(
            "method-value",
            role="METHOD_VALUE",
            target_kind="CONTEXT",
            payload={"semantic_value": "survey"},
        ),
        _binding(
            "method-relation",
            role="METHOD_RELATION",
            target_kind="CONTEXT",
            payload={"target_ref": "a"},
            relation="USES_FOR",
        ),
        _binding(
            "method-target",
            role="ATTACHMENT",
            target_kind="ATTACHMENT",
            payload={"target_ref": "a"},
            relation="USES_FOR",
        ),
    )
    assert _evaluate(derivation, candidates).usability is GroundingNodeUsability.GROUNDED_USABLE


def test_attachment_valid_structure_and_semantic_binding_pass() -> None:
    node = _node(
        GroundingNodeKind.ATTACHMENT,
        {"context_ref": "ctx", "relation": "APPLIES_TO", "target_refs": ["assertion-1"]},
    )
    derivation = derive_grounding_obligations(node)
    candidates = (
        _binding(
            "struct",
            role="ATTACHMENT",
            target_kind="ATTACHMENT",
            payload={"context_ref": "ctx"},
        ),
        _binding(
            "applicability",
            role="ATTACHMENT",
            target_kind="ATTACHMENT",
            payload={"target_ref": "assertion-1"},
            relation="APPLIES_TO",
        ),
    )
    assert _evaluate(derivation, candidates).usability is GroundingNodeUsability.GROUNDED_USABLE


def test_requirement_type_is_separate_from_truth_condition_materiality() -> None:
    from app.domain.grounding_requirements import GroundingObligation

    assert (
        GroundingObligation.model_fields["requirement_level"].annotation
        is GroundingRequirementLevel
    )
    assert "materiality" not in GroundingObligation.model_fields


def test_candidate_outcomes_and_labels_cannot_change_requirement_derivation() -> None:
    node = _node(payload={"assertion_content": {"subject": "x", "value": 1}, "human_label": "PASS"})
    expected = derive_grounding_obligations(node)
    _evaluate(expected, _assertion_candidates(semantic=DS.INVALID))
    _evaluate(expected, _assertion_candidates(semantic=DS.AMBIGUOUS))
    assert derive_grounding_obligations(node) == expected


def test_graph_role_participates_in_deterministic_obligation_identity() -> None:
    payload = {"assertion_content": {"subject": "x", "value": 1}}
    left_node = _node(payload=payload, ref="graph-node")
    right_node = GroundingNodeInput(
        node_ref="graph-node",
        node_kind=GroundingNodeKind.ASSERTION,
        graph_role="different_graph_role",
        semantic_payload=payload,
    )
    left = derive_grounding_obligations(left_node)
    right = derive_grounding_obligations(right_node)
    assert [item.obligation_ref for item in left.obligations] != [
        item.obligation_ref for item in right.obligations
    ]


def test_existing_v2_binding_is_projected_without_changing_dimensions() -> None:
    source_binding = SemanticEvidenceBinding(
        binding_ref="b-v2",
        evidence_ref="e-v2",
        semantic_role=SemanticEvidenceRole.ASSERTION_EVIDENCE,
        target_kind=EvidenceTargetKind.ASSERTION,
        target_ref="n0",
        support_status=DS.VALID,
        dimensions=DimensionGrounding(
            textual_grounding=DS.VALID,
            structural_grounding=DS.VALID,
            semantic_support=DS.AMBIGUOUS,
        ),
    )
    projected = project_semantic_evidence_bindings(
        (source_binding,),
        canonical_payload_by_binding_ref={
            "b-v2": {"assertion_content": {"subject": "x", "value": 1}}
        },
    )
    assert len(projected) == 1
    assert projected[0].semantic_role == "ASSERTION_EVIDENCE"
    assert projected[0].textual_grounding is DS.VALID
    assert projected[0].structural_grounding is DS.VALID
    assert projected[0].semantic_support is DS.AMBIGUOUS


def test_shared_context_single_unsupported_target_is_not_hidden() -> None:
    node = _node(
        GroundingNodeKind.CONTEXT,
        {
            "context_type": "SHARED_SCOPE",
            "semantic_value": "region-X",
            "target_refs": ["a1", "a2"],
        },
    )
    derivation = derive_grounding_obligations(node)
    supported = _binding(
        "target-a1",
        role="ATTACHMENT",
        target_kind="ATTACHMENT",
        payload={"target_ref": "a1"},
        relation="APPLIES_TO",
    )
    value = _binding(
        "scope-value",
        role="SCOPE_VALUE",
        target_kind="CONTEXT",
        payload={"semantic_value": "region-X"},
    )
    assessment = _evaluate(derivation, (supported, value))
    assert assessment.usability is GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
    missing = [
        item
        for item in assessment.obligation_evaluations
        if item.obligation.expected_target_key == "a2"
    ]
    assert len(missing) == 1
    assert missing[0].status is GroundingObligationStatus.MISSING_REQUIRED_BINDING
