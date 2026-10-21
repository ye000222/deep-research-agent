"""Explicit v2.1 ownership contract for offline/shadow grounding evaluation.

The existing v1/v2 entry points retain their semantics. This module exposes
the D4-D revision as an opt-in versioned contract and does not wire it into the
ResearchLoop or persistence path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

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
from app.domain.source_context_grounding_v2 import GroundingDimensionStatus as DS
from app.services.grounding_requirement_engine import (
    _combine_candidate_statuses,
    _constraint_exclusion,
)
from app.services.grounding_requirement_engine_v2 import (
    GROUNDING_AGGREGATION_VERSION_V2,
    _attachment_edge_exclusion,
    derive_grounding_obligations_v2,
)

GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1 = "grounding-requirements.generic.v2.1"
GROUNDING_MATCHER_VERSION_V2_1 = "grounding-matcher.generic.v2.1"
CANONICAL_FACT_CONTRACT_VERSION = "grounding-canonical-fact.generic.v1"
REUSE_REPORT_VERSION = "grounding-reuse-metrics.generic.v2.1"
READINESS_REPORT_VERSION = "grounding-readiness-metrics.generic.v2.1"


def canonical_grounding_fact_identity(obligation: GroundingObligation) -> str:
    fact: dict[str, Any] = {
        "node_kind": obligation.node_kind.value,
        "target_kind": obligation.target_constraint.target_kind,
        "target_ref": obligation.target_constraint.target_ref,
        "relation": obligation.target_constraint.relation,
        "scope_ref": obligation.target_constraint.scope_ref,
        "semantic_key": obligation.target_constraint.semantic_key,
        "expected_value": obligation.target_constraint.expected_value,
    }
    # A multi-target attachment is one canonical graph edge. Its per-target
    # obligations are separate checks over that same edge, not different
    # semantic edge facts. The target member remains validated against the
    # candidate's explicit source_assertion_refs before this identity is used.
    if (
        obligation.node_kind is GroundingNodeKind.ATTACHMENT
        and obligation.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
    ):
        fact["semantic_key"] = "canonical_attachment_edge"
        fact.pop("expected_value", None)
    encoded = json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_ASSERTION_VIEW_TYPES = {
    GroundingObligationType.ASSERTION_TEXT_EVIDENCE,
    GroundingObligationType.ASSERTION_CONTENT_SUPPORT,
    GroundingObligationType.ASSERTION_GRAPH_MEMBERSHIP,
}


def _authorized_reuse(previous: GroundingObligation, current: GroundingObligation) -> bool:
    if not (
        previous.allow_binding_reuse
        and current.allow_binding_reuse
        and previous.binding_reuse_group
        and previous.binding_reuse_group == current.binding_reuse_group
        and previous.node_ref == current.node_ref
        and (
            previous.target_constraint == current.target_constraint
            or _same_canonical_attachment_edge(previous, current)
        )
        and canonical_grounding_fact_identity(previous)
        == canonical_grounding_fact_identity(current)
    ):
        return False
    assertion_views = (
        previous.node_kind is GroundingNodeKind.ASSERTION
        and current.node_kind is GroundingNodeKind.ASSERTION
        and previous.obligation_type in _ASSERTION_VIEW_TYPES
        and current.obligation_type in _ASSERTION_VIEW_TYPES
        and previous.expected_semantic_role_family == current.expected_semantic_role_family
    )
    attachment_views = _same_canonical_attachment_edge(previous, current)
    return assertion_views or attachment_views


def _same_canonical_attachment_edge(
    previous: GroundingObligation, current: GroundingObligation
) -> bool:
    return (
        previous.node_kind is GroundingNodeKind.ATTACHMENT
        and current.node_kind is GroundingNodeKind.ATTACHMENT
        and previous.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
        and current.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
        and previous.node_ref == current.node_ref
        and previous.target_constraint.target_kind == current.target_constraint.target_kind
        and previous.target_constraint.target_ref == current.target_constraint.target_ref
        and previous.target_constraint.relation == current.target_constraint.relation
        and previous.target_constraint.scope_ref == current.target_constraint.scope_ref
        and previous.target_constraint.semantic_key == "target_ref"
        and current.target_constraint.semantic_key == "target_ref"
    )


def _metric_identity_exclusion(
    obligation: GroundingObligation, candidate: GroundingBindingCandidate
) -> GroundingCandidateExclusionReason | None:
    constraint = obligation.target_constraint
    if candidate.semantic_role not in obligation.expected_semantic_role_family:
        return GroundingCandidateExclusionReason.ROLE_MISMATCH
    if constraint.target_kind and candidate.target_kind != constraint.target_kind:
        return GroundingCandidateExclusionReason.TARGET_KIND_MISMATCH
    if constraint.target_ref and candidate.target_ref != constraint.target_ref:
        return GroundingCandidateExclusionReason.TARGET_REF_MISMATCH
    typed_value = candidate.canonical_payload.get("semantic_value")
    if (
        not isinstance(typed_value, dict)
        or not typed_value.get("metric")
        or typed_value != constraint.expected_value
    ):
        return GroundingCandidateExclusionReason.PAYLOAD_MISMATCH
    return None


def _reuse_record(
    previous: GroundingObligation, current: GroundingObligation, binding_ref: str
) -> dict[str, Any]:
    modes = [
        "structural"
        if previous.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
        else "semantic",
        "structural"
        if current.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
        else "semantic",
    ]
    return {
        "binding_ref": binding_ref,
        "reuse_group": current.binding_reuse_group,
        "obligations_satisfied": [previous.obligation_ref, current.obligation_ref],
        "contract_rule": "same_explicit_group_and_canonical_fact_identity.v2.1",
        "canonical_fact_identity": canonical_grounding_fact_identity(current),
        "use_modes": modes,
    }


def _version_obligation(obligation: GroundingObligation) -> GroundingObligation:
    payload = obligation.model_dump(mode="json")
    payload.pop("obligation_ref", None)
    payload["contract_version"] = GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return obligation.model_copy(
        update={
            "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
            "obligation_ref": f"obl21_{digest[:24]}",
        }
    )


def derive_grounding_obligations_v21(
    node: GroundingNodeInput,
) -> GroundingObligationDerivationResult:
    """Derive v2.1 obligations, respecting explicit canonical scope owners.

    A SourceScope node's kind and member list are graph-owned facts. They are
    checked structurally from its validated graph state, not duplicated as
    binding-shaped semantic requirements. Natural-language scope meaning on
    a CONTEXT node remains semantic and still needs supported evidence.
    """
    if (
        node.node_kind is GroundingNodeKind.SCOPE
        and node.semantic_payload.get("canonical_scope_owner") is True
    ):
        scope_kind = node.semantic_payload.get("kind")
        members = node.semantic_payload.get("member_refs")
        if not isinstance(scope_kind, str) or not scope_kind or not isinstance(members, list):
            return GroundingObligationDerivationResult(
                node_ref=node.node_ref,
                node_kind=node.node_kind,
                status=GroundingDerivationStatus.CONTRACT_ERROR,
                diagnostics=("canonical_scope_owner_shape_invalid",),
                contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
            )
        if len(members) != len(set(map(str, members))):
            return GroundingObligationDerivationResult(
                node_ref=node.node_ref,
                node_kind=node.node_kind,
                status=GroundingDerivationStatus.CONTRACT_ERROR,
                diagnostics=("canonical_scope_owner_duplicate_members",),
                contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
            )
        return GroundingObligationDerivationResult(
            node_ref=node.node_ref,
            node_kind=node.node_kind,
            status=GroundingDerivationStatus.RESOLVED,
            zero_obligation_reason=ZeroObligationReason.STRUCTURAL_ONLY_NODE,
            contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
            diagnostics=(
                "scope_kind_and_members_owned_by_source_scope_graph",
                "scope_graph_does_not_establish_natural_language_applicability",
            ),
        )

    base = derive_grounding_obligations_v2(node)
    obligations: list[GroundingObligation] = []
    for obligation in base.obligations:
        updated = obligation
        semantic_value = node.semantic_payload.get("semantic_value", {})
        if obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY:
            metric_identity: Any = (
                semantic_value.get("metric") if isinstance(semantic_value, dict) else None
            )
            if metric_identity is None:
                metric_identity = node.semantic_payload.get("metric_identity")
            if metric_identity is not None:
                updated = obligation.model_copy(
                    update={
                        "expected_semantic_role_family": ("MEASUREMENT_DEFINITION",),
                        "target_constraint": obligation.target_constraint.model_copy(
                            update={
                                "semantic_key": "semantic_value",
                                "expected_value": semantic_value,
                            }
                        ),
                        "allow_binding_reuse": False,
                        "binding_reuse_group": None,
                        "cardinality_policy": GroundingCardinalityPolicy.AT_LEAST_ONE,
                        "derivation_rule_id": "measurement.canonical_metric_identity.v2.1",
                    }
                )
        obligations.append(_version_obligation(updated))
    return base.model_copy(
        update={
            "obligations": tuple(obligations),
            "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
        }
    )


def evaluate_grounding_node_v21(
    derivation: GroundingObligationDerivationResult,
    candidates: tuple[GroundingBindingCandidate, ...],
    *,
    node_textual_grounding: DS,
    node_structural_grounding: DS,
    canonical_scope_memberships: Mapping[str, Mapping[str, object]] | None = None,
    target_support_overrides: Mapping[tuple[str, str], DS] | None = None,
) -> NodeGroundingAssessment:
    """Evaluate using explicit compatible same-fact reuse and graph owners."""
    versioned = derivation.model_copy(
        update={
            "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
            "obligations": tuple(_version_obligation(item) for item in derivation.obligations),
        }
    )
    if versioned.status is not GroundingDerivationStatus.RESOLVED:
        return NodeGroundingAssessment(
            node_ref=versioned.node_ref,
            node_kind=versioned.node_kind,
            derivation=versioned,
            usability=GroundingNodeUsability.NOT_EVALUATED,
            aggregation_precedence=("DERIVATION_FAILURE",),
            diagnostics=(*versioned.diagnostics, "obligation_derivation_failed_closed"),
        )
    if not versioned.obligations:
        structural_only = (
            versioned.zero_obligation_reason is ZeroObligationReason.STRUCTURAL_ONLY_NODE
        )
        usable = (
            GroundingNodeUsability.NO_SEMANTIC_GROUNDING_REQUIRED
            if versioned.zero_obligation_reason
            is ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED
            else GroundingNodeUsability.STRUCTURALLY_GROUNDED
            if structural_only
            and node_textual_grounding is DS.VALID
            and node_structural_grounding is DS.VALID
            else GroundingNodeUsability.UNGROUNDED_UNUSABLE
            if structural_only
            else GroundingNodeUsability.NOT_EVALUATED
        )
        return NodeGroundingAssessment(
            node_ref=versioned.node_ref,
            node_kind=versioned.node_kind,
            derivation=versioned,
            usability=usable,
            aggregation_precedence=("EXPLICIT_ZERO_OBLIGATION_POLICY_V2_1",),
            diagnostics=versioned.diagnostics,
        )

    evaluations: list[GroundingObligationEvaluation] = []
    omissions: list[GroundingRequirementDiagnostics] = []
    semantic_use: dict[str, GroundingObligation] = {}
    for obligation in versioned.obligations:
        owner = (canonical_scope_memberships or {}).get(
            str(obligation.target_constraint.target_ref or "")
        )
        owner_members = owner.get("members", ()) if owner else ()
        scope_owner_match = bool(
            obligation.obligation_type is GroundingObligationType.SCOPE_MEMBERSHIP
            and obligation.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
            and owner
            and owner.get("valid") is True
            and obligation.expected_member_key
            and isinstance(owner_members, list | tuple)
            and obligation.expected_member_key in owner_members
        )
        matched: list[GroundingBindingCandidate] = []
        excluded: list[GroundingCandidateExclusion] = []
        for candidate in candidates:
            if scope_owner_match:
                continue
            reason = (
                _attachment_edge_exclusion(obligation, candidate)
                if obligation.obligation_type is GroundingObligationType.TARGET_APPLICABILITY
                and obligation.node_kind is GroundingNodeKind.ATTACHMENT
                else _metric_identity_exclusion(obligation, candidate)
                if obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
                else _constraint_exclusion(obligation, candidate)
            )
            prior = semantic_use.get(candidate.binding_ref)
            if reason is None and prior is not None and not _authorized_reuse(prior, obligation):
                reason = GroundingCandidateExclusionReason.BINDING_ALREADY_CONSUMED
            if reason is None:
                support_override = (target_support_overrides or {}).get(
                    (candidate.binding_ref, str(obligation.target_constraint.expected_value))
                )
                matched.append(
                    candidate.model_copy(update={"semantic_support": support_override})
                    if support_override is not None
                    else candidate
                )
            else:
                excluded.append(
                    GroundingCandidateExclusion(binding_ref=candidate.binding_ref, reason=reason)
                )

        evidence_refs = [item.evidence_ref for item in matched]
        distinct_evidence = set(evidence_refs)
        duplicates = tuple(sorted({ref for ref in evidence_refs if evidence_refs.count(ref) > 1}))
        policy = obligation.cardinality_policy
        cardinality = (
            bool(matched)
            if policy
            in {
                GroundingCardinalityPolicy.AT_LEAST_ONE,
                GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS,
                GroundingCardinalityPolicy.PER_TARGET,
                GroundingCardinalityPolicy.PER_VALUE,
            }
            else len(distinct_evidence) == 1
        )
        if scope_owner_match:
            cardinality = True
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
            and not scope_owner_match
            and obligation.requirement_level is not GroundingRequirementLevel.OPTIONAL_CORROBORATION
        ):
            omissions.append(
                GroundingRequirementDiagnostics(
                    code="MISSING_REQUIRED_BINDING",
                    node_ref=obligation.node_ref,
                    obligation_ref=obligation.obligation_ref,
                    expected_role_family=obligation.expected_semantic_role_family,
                    target_constraint=obligation.target_constraint,
                    contract_version=GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
                    reason="no candidate binding satisfied the v2.1 owner/matcher contract",
                )
            )
        status = (
            GroundingObligationStatus.SATISFIED
            if scope_owner_match
            else _combine_candidate_statuses(obligation, matched)
        )
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
        provenance = {
            "node_ref": obligation.node_ref,
            "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION_V2_1,
            "matcher_version": GROUNDING_MATCHER_VERSION_V2_1,
            "aggregation_version": GROUNDING_AGGREGATION_VERSION_V2,
            "canonical_fact_contract": CANONICAL_FACT_CONTRACT_VERSION,
            "canonical_fact_identity": canonical_grounding_fact_identity(obligation),
            "derivation_rule_id": obligation.derivation_rule_id,
            "obligation_ref": obligation.obligation_ref,
            "candidate_binding_refs": list(match.candidate_binding_refs),
            "matched_binding_refs": list(match.matched_binding_refs),
            "excluded_candidates": [item.model_dump(mode="json") for item in excluded],
            "binding_reuse_permitted": obligation.allow_binding_reuse,
            "binding_reuse_group": obligation.binding_reuse_group,
            "canonical_owner": (
                {
                    "owner_type": "SourceScopeV2",
                    "scope_ref": obligation.target_constraint.target_ref,
                    "kind": owner.get("kind") if owner else None,
                    "member_ref": obligation.expected_member_key,
                    "fact_identity": canonical_grounding_fact_identity(obligation),
                }
                if scope_owner_match
                else None
            ),
            "authorized_reuse": [
                _reuse_record(semantic_use[item.binding_ref], obligation, item.binding_ref)
                for item in matched
                if item.binding_ref in semantic_use
                and semantic_use[item.binding_ref].obligation_ref != obligation.obligation_ref
                and _authorized_reuse(semantic_use[item.binding_ref], obligation)
            ],
            "obligation_status": status.value,
        }
        evaluations.append(
            GroundingObligationEvaluation(
                obligation=obligation,
                match=match,
                status=status,
                binding_statuses=binding_statuses,
                diagnostics=match.diagnostics,
                provenance=provenance,
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
    if DS.INVALID in dimensions or states.intersection(
        {
            GroundingObligationStatus.TEXTUALLY_INVALID,
            GroundingObligationStatus.STRUCTURALLY_INVALID,
        }
    ):
        usability, precedence = (
            GroundingNodeUsability.UNGROUNDED_UNUSABLE,
            "MANDATORY_TEXTUAL_OR_STRUCTURAL_FAILURE",
        )
    elif DS.AMBIGUOUS in dimensions or GroundingObligationStatus.AMBIGUOUS in states:
        usability, precedence = (
            GroundingNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS,
            "REQUIRED_AMBIGUOUS",
        )
    elif GroundingObligationStatus.NOT_SUPPORTED in states:
        usability, precedence = GroundingNodeUsability.UNGROUNDED_UNUSABLE, "REQUIRED_NOT_SUPPORTED"
    elif (
        DS.MISSING_EVIDENCE in dimensions
        or DS.NOT_EVALUATED in dimensions
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
        node_ref=versioned.node_ref,
        node_kind=versioned.node_kind,
        derivation=versioned,
        obligation_evaluations=tuple(evaluations),
        usability=usability,
        verification_assertion_ready=usability is GroundingNodeUsability.GROUNDED_USABLE,
        aggregation_precedence=(precedence,),
        diagnostics=diagnostics,
        extraction_omissions=tuple(omissions),
    )


def grounding_reporting_metrics(
    node_assessments: tuple[NodeGroundingAssessment, ...],
) -> dict[str, int | str]:
    """Separate assertion readiness from mixed node totals; report reuse honestly."""
    nodes = list(node_assessments)
    excluded_reuse = sum(
        1
        for node in nodes
        for evaluation in node.obligation_evaluations
        for exclusion in evaluation.match.excluded_candidates
        if exclusion.reason.value == "BINDING_ALREADY_CONSUMED"
    )
    authorized_reuse = sum(
        len(evaluation.provenance.get("authorized_reuse", []))
        for node in nodes
        for evaluation in node.obligation_evaluations
    )
    reused_refs: dict[str, int] = {}
    authorized_refs: set[str] = set()
    for node in nodes:
        for evaluation in node.obligation_evaluations:
            for ref in evaluation.match.matched_binding_refs:
                reused_refs[ref] = reused_refs.get(ref, 0) + 1
            authorized_refs.update(
                str(item.get("binding_ref"))
                for item in evaluation.provenance.get("authorized_reuse", [])
            )
    undeclared_reuse = sum(
        count - 1 for ref, count in reused_refs.items() if count > 1 and ref not in authorized_refs
    )
    return {
        "report_version": READINESS_REPORT_VERSION,
        "all_nodes_grounded_usable_count": sum(
            node.usability.value == "GROUNDED_USABLE" for node in nodes
        ),
        "assertion_root_count": sum(
            node.node_kind is GroundingNodeKind.ASSERTION for node in nodes
        ),
        "assertion_root_ready_count": sum(
            node.node_kind is GroundingNodeKind.ASSERTION and node.verification_assertion_ready
            for node in nodes
        ),
        "context_ready_count": sum(
            node.node_kind is GroundingNodeKind.CONTEXT
            and node.usability.value == "GROUNDED_USABLE"
            for node in nodes
        ),
        "attachment_ready_count": sum(
            node.node_kind is GroundingNodeKind.ATTACHMENT
            and node.usability.value == "GROUNDED_USABLE"
            for node in nodes
        ),
        "scope_ready_count": sum(
            node.node_kind is GroundingNodeKind.SCOPE
            and node.usability.value in {"GROUNDED_USABLE", "STRUCTURALLY_GROUNDED"}
            for node in nodes
        ),
        "reuse_report_version": REUSE_REPORT_VERSION,
        "reuse_candidate_excluded_count": excluded_reuse,
        "authorized_reuse_success_count": authorized_reuse,
        "successful_undeclared_semantic_reuse_count": undeclared_reuse,
    }
