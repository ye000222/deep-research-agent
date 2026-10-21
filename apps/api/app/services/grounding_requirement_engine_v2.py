"""Versioned v2 grounding contract and matcher for offline/shadow evaluation.

The v1 functions remain untouched and retain their original semantics.  V2
changes only obligation ownership and deterministic graph matching.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from app.domain.grounding_requirements import (
    GroundingBindingCandidate,
    GroundingCandidateExclusion,
    GroundingCandidateExclusionReason,
    GroundingCardinalityPolicy,
    GroundingDerivationStatus,
    GroundingNodeInput,
    GroundingNodeKind,
    GroundingNodeUsability,
    GroundingObligation,
    GroundingObligationDerivationResult,
    GroundingObligationEvaluation,
    GroundingObligationMatch,
    GroundingObligationStatus,
    GroundingObligationType,
    GroundingRequirementDiagnostics,
    GroundingRequirementLevel,
    NodeGroundingAssessment,
    ZeroObligationReason,
)
from app.domain.source_context_grounding_v2 import GroundingDimensionStatus
from app.services.grounding_requirement_engine import (
    _combine_candidate_statuses,
    _constraint_exclusion,
    _emit,
    _resolved,
    _unresolved,
    derive_grounding_obligations,
)

GROUNDING_MATCHER_VERSION_V2 = "grounding-matcher.generic.v2"
GROUNDING_AGGREGATION_VERSION_V2 = "grounding-aggregation.generic.v1"
GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2 = "grounding-requirements.generic.v2"


def _reversion(obligations: list[GroundingObligation]) -> tuple[GroundingObligation, ...]:
    result: list[GroundingObligation] = []
    for obligation in obligations:
        payload = obligation.model_dump(mode="json")
        payload.pop("obligation_ref", None)
        payload["contract_version"] = GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result.append(
            obligation.model_copy(
                update={
                    "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
                    "obligation_ref": f"obl2_{digest[:24]}",
                }
            )
        )
    return tuple(result)


def _measurement_obligations(node: GroundingNodeInput) -> list[GroundingObligation] | None:
    payload = node.semantic_payload
    value = payload.get("semantic_value")
    if value is None and any(
        key in payload
        for key in ("metric_identity", "category_identity", "scope_identity", "associations")
    ):
        value = {}
    if not isinstance(value, dict):
        return None
    result: list[GroundingObligation] = []
    definition_fields = ("metric", "measurement", "measure")
    if any(key in value for key in definition_fields):
        result.append(
            _emit(
                node,
                kind=GroundingObligationType.METRIC_IDENTITY,
                roles=("MEASUREMENT_DEFINITION",),
                target_kind="CONTEXT",
                semantic_key="semantic_value",
                expected_value=value,
                rule_id="measurement.typed_definition.present.v2",
            )
        )
    # A value is an observed/reported result only when the payload has a typed
    # result field and does not instead represent a threshold/comparator.
    if "observed_value" in value or (
        "value" in value and not ("threshold" in value or "comparator" in value)
    ):
        result.append(
            _emit(
                node,
                kind=GroundingObligationType.MEASUREMENT_VALUE,
                roles=("MEASUREMENT_VALUE",),
                target_kind="CONTEXT",
                semantic_key="semantic_value",
                expected_value=value,
                rule_id="measurement.typed_result_value.present.v2",
            )
        )
    if "threshold" in value or "comparator" in value:
        result.append(
            _emit(
                node,
                kind=GroundingObligationType.CONTEXT_VALUE,
                roles=("MEASUREMENT_COMPARATOR",),
                target_kind="CONTEXT",
                semantic_key="semantic_value",
                expected_value=value,
                rule_id="measurement.typed_comparator_threshold.present.v2",
            )
        )
    if "unit" in value:
        result.append(
            _emit(
                node,
                kind=GroundingObligationType.CONTEXT_VALUE,
                roles=("MEASUREMENT_UNIT",),
                target_kind="CONTEXT",
                semantic_key="semantic_value",
                expected_value=value,
                rule_id="measurement.typed_unit.present.v2",
            )
        )
    for field, role, kind in (
        ("metric_identity", "MEASUREMENT_DEFINITION", GroundingObligationType.METRIC_IDENTITY),
        ("category_identity", "SCOPE_VALUE", GroundingObligationType.CATEGORY_IDENTITY),
        ("scope_identity", "SCOPE_VALUE", GroundingObligationType.SCOPE_VALUE),
    ):
        if field in payload:
            result.append(
                _emit(
                    node,
                    kind=kind,
                    roles=(role,),
                    target_kind="CONTEXT",
                    semantic_key=field,
                    expected_value=payload[field],
                    rule_id=f"measurement.{field}.present.v2",
                )
            )
    associations = payload.get("associations", [])
    if not isinstance(associations, list):
        return None
    for association in associations:
        if not isinstance(association, dict) or not association.get("value_key"):
            return None
        result.append(
            _emit(
                node,
                kind=GroundingObligationType.NUMERIC_ASSOCIATION,
                roles=("ATTACHMENT", "RELATION_OPERATOR"),
                target_kind="CONTEXT",
                relation=str(association.get("relation", "ASSOCIATED_WITH")),
                semantic_key="association",
                expected_value=association,
                cardinality=GroundingCardinalityPolicy.PER_VALUE,
                value_key=str(association["value_key"]),
                rule_id="measurement.association.per_value.v2",
            )
        )
    return result


def derive_grounding_obligations_v2(
    node: GroundingNodeInput,
) -> GroundingObligationDerivationResult:
    """Derive v2 obligations using intrinsic-node and semantic-edge ownership."""
    if node.node_kind is GroundingNodeKind.CONTEXT:
        context_type = node.semantic_payload.get("context_type") or node.semantic_payload.get(
            "semantic_kind"
        )
        if context_type == "MEASUREMENT":
            obligations = _measurement_obligations(node)
            if obligations is None:
                return _unresolved(
                    node,
                    GroundingDerivationStatus.CONTRACT_ERROR,
                    "measurement_typed_payload_invalid",
                )
            if not obligations:
                return _unresolved(
                    node,
                    GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                    "measurement_payload_has_no_typed_grounding_semantics",
                )
            base = _resolved(node, obligations)
        else:
            base = derive_grounding_obligations(node)
            if context_type == "METHOD" and base.status is GroundingDerivationStatus.RESOLVED:
                intrinsic = [
                    item
                    for item in base.obligations
                    if item.obligation_type is GroundingObligationType.METHOD_VALUE
                ]
                if intrinsic:
                    base = _resolved(node, intrinsic)
                else:
                    base = _resolved(
                        node,
                        [],
                        zero_reason=ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED,
                        diagnostics=("method_relation_owned_by_canonical_attachment_edge_v2",),
                    )
    else:
        base = derive_grounding_obligations(node)

    if base.status is not GroundingDerivationStatus.RESOLVED or not base.obligations:
        return base.model_copy(
            update={"contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2}
        )

    obligations = list(base.obligations)
    if node.node_kind is GroundingNodeKind.ATTACHMENT:
        targets = node.semantic_payload.get("target_refs", [])
        context_ref = node.semantic_payload.get("context_ref")
        relation = node.semantic_payload.get("relation")
        obligations = [
            item.model_copy(
                update={
                    "diagnostics": (
                        *item.diagnostics,
                        *(
                            (f"v2_source_scope_ref:{node.semantic_payload['source_scope_ref']}",)
                            if node.semantic_payload.get("source_scope_ref")
                            else ()
                        ),
                    ),
                    "target_constraint": item.target_constraint.model_copy(
                        update={
                            "scope_ref": str(context_ref),
                            "relation": str(relation),
                        }
                    )
                }
            )
            if item.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
            else item
            for item in obligations
        ]
        if isinstance(targets, list) and len(set(map(str, targets))) > 1:
            group = f"attachment-edge:{node.node_ref}"
            obligations = [
                item.model_copy(update={"allow_binding_reuse": True, "binding_reuse_group": group})
                if item.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
                else item
                for item in obligations
            ]
    return base.model_copy(
        update={
            "obligations": _reversion(obligations),
            "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
        }
    )


def _attachment_edge_exclusion(
    obligation: GroundingObligation, candidate: GroundingBindingCandidate
) -> GroundingCandidateExclusionReason | None:
    if candidate.semantic_role not in obligation.expected_semantic_role_family:
        return GroundingCandidateExclusionReason.ROLE_MISMATCH
    if candidate.target_kind != "ATTACHMENT" or candidate.target_ref != obligation.node_ref:
        return GroundingCandidateExclusionReason.TARGET_REF_MISMATCH
    payload = candidate.canonical_payload
    expected_context = obligation.target_constraint.scope_ref
    # The owning attachment node is the canonical graph edge.  Its binding
    # target is dereferenced to context + relation + assertion target(s).
    attachment_context = payload.get("context_ref")
    if expected_context and attachment_context != expected_context:
        return GroundingCandidateExclusionReason.PAYLOAD_MISMATCH
    expected_relation = obligation.target_constraint.relation
    actual_relation = payload.get("relation", candidate.relation)
    if expected_relation and actual_relation != expected_relation:
        return GroundingCandidateExclusionReason.RELATION_MISMATCH
    expected_source_scope = next(
        (
            diagnostic.partition(":")[2]
            for diagnostic in obligation.diagnostics
            if diagnostic.startswith("v2_source_scope_ref:")
        ),
        None,
    )
    if expected_source_scope and candidate.scope_ref != expected_source_scope:
        return GroundingCandidateExclusionReason.SCOPE_MISMATCH
    expected_target = obligation.target_constraint.expected_value
    targets = payload.get("source_assertion_refs", [])
    if not isinstance(targets, list | tuple) or expected_target not in targets:
        return GroundingCandidateExclusionReason.TARGET_REF_MISMATCH
    if obligation.target_constraint.target_kind and candidate.target_kind != "ATTACHMENT":
        return GroundingCandidateExclusionReason.TARGET_KIND_MISMATCH
    return None


def evaluate_grounding_node_v2(
    derivation: GroundingObligationDerivationResult,
    candidates: tuple[GroundingBindingCandidate, ...],
    *,
    node_textual_grounding: GroundingDimensionStatus,
    node_structural_grounding: GroundingDimensionStatus,
    target_support_overrides: Mapping[tuple[str, str], GroundingDimensionStatus] | None = None,
) -> NodeGroundingAssessment:
    """Evaluate v2 with structural checks non-exclusive and explicit edge dereference."""
    if derivation.status is not GroundingDerivationStatus.RESOLVED:
        return NodeGroundingAssessment(
            node_ref=derivation.node_ref,
            node_kind=derivation.node_kind,
            derivation=derivation,
            usability=GroundingNodeUsability.NOT_EVALUATED,
            aggregation_precedence=("DERIVATION_FAILURE",),
            diagnostics=(*derivation.diagnostics, "obligation_derivation_failed_closed"),
        )
    if not derivation.obligations:
        return NodeGroundingAssessment(
            node_ref=derivation.node_ref,
            node_kind=derivation.node_kind,
            derivation=derivation,
            usability=(
                GroundingNodeUsability.NO_SEMANTIC_GROUNDING_REQUIRED
                if derivation.zero_obligation_reason
                is ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED
                else GroundingNodeUsability.STRUCTURALLY_GROUNDED
                if derivation.zero_obligation_reason is ZeroObligationReason.STRUCTURAL_ONLY_NODE
                and node_textual_grounding is GroundingDimensionStatus.VALID
                and node_structural_grounding is GroundingDimensionStatus.VALID
                else GroundingNodeUsability.UNGROUNDED_UNUSABLE
                if derivation.zero_obligation_reason is ZeroObligationReason.STRUCTURAL_ONLY_NODE
                else GroundingNodeUsability.NOT_EVALUATED
            ),
            aggregation_precedence=("EXPLICIT_ZERO_OBLIGATION_POLICY_V2",),
            diagnostics=derivation.diagnostics,
        )

    evaluations: list[GroundingObligationEvaluation] = []
    omissions: list[GroundingRequirementDiagnostics] = []
    semantic_use: dict[str, GroundingObligation] = {}
    for obligation in derivation.obligations:
        matched: list[GroundingBindingCandidate] = []
        excluded: list[GroundingCandidateExclusion] = []
        for candidate in candidates:
            if (
                obligation.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
                and obligation.node_kind is GroundingNodeKind.ATTACHMENT
            ):
                reason = _attachment_edge_exclusion(obligation, candidate)
            else:
                reason = _constraint_exclusion(obligation, candidate)
            prior = semantic_use.get(candidate.binding_ref)
            if reason is None and prior is not None:
                same_explicit_attachment_edge = (
                    obligation.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
                    and prior.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
                    and obligation.node_kind is GroundingNodeKind.ATTACHMENT
                    and prior.node_ref == obligation.node_ref
                    and obligation.allow_binding_reuse
                    and prior.allow_binding_reuse
                    and obligation.binding_reuse_group == prior.binding_reuse_group
                )
                if not same_explicit_attachment_edge:
                    reason = GroundingCandidateExclusionReason.BINDING_ALREADY_CONSUMED
            if reason is None:
                support = (target_support_overrides or {}).get(
                    (candidate.binding_ref, str(obligation.target_constraint.expected_value)),
                    candidate.semantic_support,
                )
                matched.append(candidate.model_copy(update={"semantic_support": support}))
            else:
                excluded.append(
                    GroundingCandidateExclusion(binding_ref=candidate.binding_ref, reason=reason)
                )

        evidence_refs = [item.evidence_ref for item in matched]
        distinct_evidence = set(evidence_refs)
        duplicates = tuple(sorted({ref for ref in evidence_refs if evidence_refs.count(ref) > 1}))
        policy = obligation.cardinality_policy
        if policy is GroundingCardinalityPolicy.AT_LEAST_ONE:
            cardinality = bool(matched)
        elif policy is GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS or policy in {
            GroundingCardinalityPolicy.PER_TARGET,
            GroundingCardinalityPolicy.PER_VALUE,
        }:
            cardinality = bool(matched) and len(distinct_evidence) == 1
        else:
            cardinality = len(distinct_evidence) == 1
        match = GroundingObligationMatch(
            obligation_ref=obligation.obligation_ref,
            candidate_binding_refs=tuple(item.binding_ref for item in candidates),
            matched_binding_refs=tuple(item.binding_ref for item in matched),
            excluded_candidates=tuple(excluded),
            duplicate_evidence_refs=duplicates,
            cardinality_satisfied=cardinality,
            diagnostics=("duplicate_binding_candidates_share_evidence_ref",) if duplicates else (),
        )
        if (
            not matched
            and obligation.requirement_level is not GroundingRequirementLevel.OPTIONAL_CORROBORATION
        ):
            omissions.append(
                GroundingRequirementDiagnostics(
                    code="MISSING_REQUIRED_BINDING",
                    node_ref=obligation.node_ref,
                    obligation_ref=obligation.obligation_ref,
                    expected_role_family=obligation.expected_semantic_role_family,
                    target_constraint=obligation.target_constraint,
                    contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
                    reason="no candidate binding satisfied v2 graph and semantic constraints",
                )
            )
        status = _combine_candidate_statuses(obligation, matched)
        if not cardinality and matched:
            status = GroundingObligationStatus.AMBIGUOUS
        if obligation.requirement_level is not GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY:
            for candidate in matched:
                semantic_use.setdefault(candidate.binding_ref, obligation)
        binding_statuses = tuple(
            (
                item.binding_ref,
                item.textual_grounding.value,
                item.structural_grounding.value,
                item.semantic_support.value,
            )
            for item in matched
        )
        edge_provenance = []
        if (
            obligation.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
            and obligation.node_kind is GroundingNodeKind.ATTACHMENT
        ):
            edge_provenance = [
                {
                    "binding_ref": item.binding_ref,
                    "path": [
                        "SemanticEvidenceBinding",
                        "ATTACHMENT",
                        "context_ref/relation/source_assertion_refs",
                    ],
                    "attachment_ref": item.target_ref,
                    "context_ref": item.canonical_payload.get("context_ref"),
                    "relation": item.canonical_payload.get("relation"),
                    "assertion_targets": item.canonical_payload.get("source_assertion_refs", []),
                }
                for item in matched
            ]
        evaluations.append(
            GroundingObligationEvaluation(
                obligation=obligation,
                match=match,
                status=status,
                binding_statuses=binding_statuses,
                diagnostics=match.diagnostics,
                provenance={
                    "node_ref": obligation.node_ref,
                    "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2,
                    "matcher_version": GROUNDING_MATCHER_VERSION_V2,
                    "aggregation_version": GROUNDING_AGGREGATION_VERSION_V2,
                    "derivation_rule_id": obligation.derivation_rule_id,
                    "obligation_ref": obligation.obligation_ref,
                    "candidate_binding_refs": list(match.candidate_binding_refs),
                    "matched_binding_refs": list(match.matched_binding_refs),
                    "excluded_candidates": [item.model_dump(mode="json") for item in excluded],
                    "binding_reuse_permitted": obligation.allow_binding_reuse,
                    "binding_reuse_group": obligation.binding_reuse_group,
                    "structural_check_consumes_semantic_binding": False,
                    "graph_dereference": edge_provenance,
                    "obligation_status": status.value,
                },
            )
        )

    required = [
        item
        for item in evaluations
        if item.obligation.requirement_level
        is GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS
    ]
    structural = [
        item
        for item in evaluations
        if item.obligation.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
    ]
    states = {item.status for item in required + structural}
    dimensions = (node_textual_grounding, node_structural_grounding)
    if GroundingDimensionStatus.INVALID in dimensions or states.intersection(
        {
            GroundingObligationStatus.TEXTUALLY_INVALID,
            GroundingObligationStatus.STRUCTURALLY_INVALID,
        }
    ):
        usability, precedence = (
            GroundingNodeUsability.UNGROUNDED_UNUSABLE,
            "MANDATORY_TEXTUAL_OR_STRUCTURAL_FAILURE",
        )
    elif (
        GroundingDimensionStatus.AMBIGUOUS in dimensions
        or GroundingObligationStatus.AMBIGUOUS in states
    ):
        usability, precedence = (
            GroundingNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS,
            "REQUIRED_AMBIGUOUS",
        )
    elif GroundingObligationStatus.NOT_SUPPORTED in states:
        usability, precedence = GroundingNodeUsability.UNGROUNDED_UNUSABLE, "REQUIRED_NOT_SUPPORTED"
    elif (
        GroundingDimensionStatus.MISSING_EVIDENCE in dimensions
        or GroundingDimensionStatus.NOT_EVALUATED in dimensions
        or states.intersection(
            {
                GroundingObligationStatus.MISSING_REQUIRED_BINDING,
                GroundingObligationStatus.INSUFFICIENT_EVIDENCE,
                GroundingObligationStatus.DERIVATION_UNRESOLVED,
            }
        )
    ):
        usability, precedence = (
            GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED,
            "REQUIRED_MISSING_OR_INSUFFICIENT",
        )
    elif (
        required
        and all(item.status is GroundingObligationStatus.SATISFIED for item in required)
        and all(item.status is GroundingObligationStatus.SATISFIED for item in structural)
    ):
        usability, precedence = (
            GroundingNodeUsability.GROUNDED_USABLE,
            "ALL_REQUIRED_AND_STRUCTURAL_OBLIGATIONS_SATISFIED",
        )
    elif (
        structural
        and not required
        and all(item.status is GroundingObligationStatus.SATISFIED for item in structural)
    ):
        usability, precedence = (
            GroundingNodeUsability.STRUCTURALLY_GROUNDED,
            "STRUCTURAL_ONLY_OBLIGATIONS_SATISFIED",
        )
    else:
        usability, precedence = (
            GroundingNodeUsability.NOT_EVALUATED,
            "NO_REQUIRED_SEMANTIC_OBLIGATION",
        )
    diagnostics = tuple(
        f"obligation_state:{item.obligation.obligation_ref}:{item.status.value}"
        for item in evaluations
        if item.status is not GroundingObligationStatus.SATISFIED
    )
    return NodeGroundingAssessment(
        node_ref=derivation.node_ref,
        node_kind=derivation.node_kind,
        derivation=derivation,
        obligation_evaluations=tuple(evaluations),
        usability=usability,
        verification_assertion_ready=usability is GroundingNodeUsability.GROUNDED_USABLE,
        aggregation_precedence=(precedence,),
        diagnostics=diagnostics,
        extraction_omissions=tuple(omissions),
    )
