"""Pure shadow derivation, binding matching, and obligation aggregation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from app.domain.grounding_requirements import (
    GROUNDING_REQUIREMENT_CONTRACT_VERSION,
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
    GroundingTargetConstraint,
    NodeGroundingAssessment,
    ZeroObligationReason,
)
from app.domain.source_context_grounding_v2 import (
    GroundingDimensionStatus,
    SemanticEvidenceBinding,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def project_semantic_evidence_bindings(
    bindings: tuple[SemanticEvidenceBinding, ...],
    *,
    canonical_payload_by_binding_ref: Mapping[str, dict[str, Any]] | None = None,
    relation_by_binding_ref: Mapping[str, str] | None = None,
    scope_by_binding_ref: Mapping[str, str] | None = None,
    member_by_binding_ref: Mapping[str, str] | None = None,
    value_by_binding_ref: Mapping[str, str] | None = None,
) -> tuple[GroundingBindingCandidate, ...]:
    """Project existing v2 bindings to ephemeral requirement-engine inputs.

    The maps supply graph semantics that are deliberately not duplicated on
    ``SemanticEvidenceBinding``; dimension outcomes are copied unchanged.
    This is an offline adapter, not a production-flow hook.
    """
    payloads = canonical_payload_by_binding_ref or {}
    relations = relation_by_binding_ref or {}
    scopes = scope_by_binding_ref or {}
    members = member_by_binding_ref or {}
    values = value_by_binding_ref or {}
    return tuple(
        GroundingBindingCandidate(
            binding_ref=binding.binding_ref,
            evidence_ref=binding.evidence_ref,
            semantic_role=binding.semantic_role.value,
            target_kind=binding.target_kind.value,
            target_ref=binding.target_ref,
            canonical_payload=payloads.get(binding.binding_ref, {}),
            relation=relations.get(binding.binding_ref),
            scope_ref=scopes.get(binding.binding_ref),
            member_key=members.get(binding.binding_ref),
            value_key=values.get(binding.binding_ref),
            textual_grounding=binding.dimensions.textual_grounding,
            structural_grounding=binding.dimensions.structural_grounding,
            semantic_support=binding.dimensions.semantic_support,
        )
        for binding in bindings
    )


def _obligation_ref(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"obl_{digest[:24]}"


def _emit(
    node: GroundingNodeInput,
    *,
    kind: GroundingObligationType,
    roles: tuple[str, ...],
    level: GroundingRequirementLevel = GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS,
    target_kind: str | None = None,
    target_ref: str | None = None,
    relation: str | None = None,
    scope_ref: str | None = None,
    semantic_key: str | None = None,
    expected_value: Any | None = None,
    cardinality: GroundingCardinalityPolicy = GroundingCardinalityPolicy.EXACTLY_ONE,
    member_key: str | None = None,
    target_key: str | None = None,
    value_key: str | None = None,
    rule_id: str,
    diagnostics: tuple[str, ...] = (),
    reuse_group: str | None = None,
) -> GroundingObligation:
    constraint = GroundingTargetConstraint(
        target_kind=target_kind,
        target_ref=target_ref or node.node_ref,
        relation=relation,
        scope_ref=scope_ref
        or (
            str(node.semantic_payload["scope_ref"])
            if node.semantic_payload.get("scope_ref")
            else None
        ),
        semantic_key=semantic_key,
        expected_value=expected_value,
    )
    identity = {
        "node_ref": node.node_ref,
        "node_kind": node.node_kind.value,
        "graph_role": node.graph_role,
        "obligation_type": kind.value,
        "roles": list(roles),
        "requirement_level": level.value,
        "target_constraint": constraint.model_dump(mode="json"),
        "cardinality": cardinality.value,
        "member_key": member_key,
        "target_key": target_key,
        "value_key": value_key,
        "contract_version": GROUNDING_REQUIREMENT_CONTRACT_VERSION,
        "rule_id": rule_id,
    }
    return GroundingObligation(
        obligation_ref=_obligation_ref(identity),
        node_ref=node.node_ref,
        node_kind=node.node_kind,
        obligation_type=kind,
        expected_semantic_role_family=roles,
        requirement_level=level,
        target_constraint=constraint,
        cardinality_policy=cardinality,
        expected_member_key=member_key,
        expected_target_key=target_key,
        expected_value_key=value_key,
        derivation_rule_id=rule_id,
        diagnostics=diagnostics,
        allow_binding_reuse=reuse_group is not None,
        binding_reuse_group=reuse_group,
    )


def _resolved(
    node: GroundingNodeInput,
    obligations: list[GroundingObligation],
    *,
    zero_reason: ZeroObligationReason | None = None,
    diagnostics: tuple[str, ...] = (),
) -> GroundingObligationDerivationResult:
    return GroundingObligationDerivationResult(
        node_ref=node.node_ref,
        node_kind=node.node_kind,
        status=GroundingDerivationStatus.RESOLVED,
        obligations=tuple(obligations),
        zero_obligation_reason=zero_reason,
        diagnostics=diagnostics,
    )


def _unresolved(
    node: GroundingNodeInput,
    status: GroundingDerivationStatus,
    code: str,
) -> GroundingObligationDerivationResult:
    return GroundingObligationDerivationResult(
        node_ref=node.node_ref,
        node_kind=node.node_kind,
        status=status,
        zero_obligation_reason=ZeroObligationReason.DERIVATION_UNRESOLVED,
        diagnostics=(code,),
    )


def _context_value_obligation(
    node: GroundingNodeInput,
    *,
    role: str,
    value: Any,
    kind: GroundingObligationType = GroundingObligationType.CONTEXT_VALUE,
    target_kind: str = "CONTEXT",
    rule_id: str = "context.semantic_value.present",
) -> GroundingObligation:
    return _emit(
        node,
        kind=kind,
        roles=(role,),
        target_kind=target_kind,
        semantic_key="semantic_value",
        expected_value=value,
        rule_id=rule_id,
    )


def derive_grounding_obligations(
    node: GroundingNodeInput,
) -> GroundingObligationDerivationResult:
    """Derive obligations only from generic node semantics and graph role."""
    payload = node.semantic_payload
    out: list[GroundingObligation] = []

    if node.node_kind is GroundingNodeKind.ASSERTION:
        assertion_content = payload.get("assertion_content")
        if assertion_content in (None, "", {}, []):
            return _unresolved(
                node,
                GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                "assertion_semantic_content_missing",
            )
        reuse = f"assertion-anchor:{node.node_ref}"
        for obligation_type, level, rule_id in (
            (
                GroundingObligationType.ASSERTION_TEXT_EVIDENCE,
                GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS,
                "assertion.source_text.required",
            ),
            (
                GroundingObligationType.ASSERTION_CONTENT_SUPPORT,
                GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS,
                "assertion.content_support.required",
            ),
            (
                GroundingObligationType.ASSERTION_GRAPH_MEMBERSHIP,
                GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
                "assertion.graph_membership.structural",
            ),
        ):
            out.append(
                _emit(
                    node,
                    kind=obligation_type,
                    roles=("ASSERTION_EVIDENCE",),
                    level=level,
                    target_kind="ASSERTION",
                    semantic_key="assertion_content",
                    expected_value=assertion_content,
                    rule_id=rule_id,
                    reuse_group=reuse,
                )
            )
        if payload.get("scope_ref"):
            out.append(
                _emit(
                    node,
                    kind=GroundingObligationType.SCOPE_MEMBERSHIP,
                    roles=("SCOPE_STRUCTURE",),
                    level=GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
                    target_kind="SCOPE",
                    target_ref=str(payload["scope_ref"]),
                    rule_id="assertion.explicit_scope.structural",
                )
            )
        if "modality" in payload:
            out.append(
                _emit(
                    node,
                    kind=GroundingObligationType.MODALITY,
                    roles=("MODALITY",),
                    target_kind="ASSERTION",
                    semantic_key="modality",
                    expected_value=payload["modality"],
                    rule_id="assertion.explicit_modality.required",
                )
            )
        return _resolved(node, out)

    if node.node_kind is GroundingNodeKind.CONTEXT:
        context_type = payload.get("context_type") or payload.get("semantic_kind")
        if not isinstance(context_type, str):
            return _unresolved(
                node,
                GroundingDerivationStatus.UNSUPPORTED_NODE_SHAPE,
                "context_type_missing_or_invalid",
            )
        semantic_value = payload.get("semantic_value")
        value_present = semantic_value not in (None, "", {}, [])
        if context_type == "METHOD":
            if "method_value" in payload:
                out.append(
                    _context_value_obligation(
                        node,
                        role="METHOD_VALUE",
                        kind=GroundingObligationType.METHOD_VALUE,
                        value=payload["method_value"],
                        rule_id="context.method.value.present",
                    )
                )
            targets = payload.get("target_refs", [])
            relation = payload.get("target_relation")
            if targets and relation:
                for target_ref in sorted(set(map(str, targets))):
                    out.append(
                        _emit(
                            node,
                            kind=GroundingObligationType.METHOD_RELATION,
                            roles=("METHOD_RELATION",),
                            target_kind="CONTEXT",
                            relation=str(relation),
                            semantic_key="target_ref",
                            expected_value=target_ref,
                            cardinality=GroundingCardinalityPolicy.PER_TARGET,
                            target_key=target_ref,
                            rule_id="context.method.relation.per_target",
                        )
                    )
                    out.append(
                        _emit(
                            node,
                            kind=GroundingObligationType.TARGET_APPLICABILITY,
                            roles=("ATTACHMENT",),
                            target_kind="ATTACHMENT",
                            target_ref=node.node_ref,
                            relation=str(relation),
                            semantic_key="target_ref",
                            expected_value=target_ref,
                            cardinality=GroundingCardinalityPolicy.PER_TARGET,
                            target_key=target_ref,
                            rule_id="context.method.target_applicability.per_target",
                        )
                    )
            if not out:
                return _unresolved(
                    node,
                    GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                    "method_semantics_have_no_represented_value_or_relation",
                )
            return _resolved(node, out)

        if context_type == "EPISTEMIC":
            if "operator" in payload:
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.EPISTEMIC_OPERATOR,
                        roles=("EPISTEMIC_OPERATOR",),
                        target_kind="CONTEXT",
                        semantic_key="operator",
                        expected_value=payload["operator"],
                        rule_id="context.epistemic.operator.present",
                    )
                )
            if "agent_ref" in payload:
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.EPISTEMIC_AGENT,
                        roles=("EPISTEMIC_AGENT",),
                        target_kind="CONTEXT",
                        semantic_key="agent_ref",
                        expected_value=payload["agent_ref"],
                        rule_id="context.epistemic.agent.present",
                    )
                )
            for target_ref in sorted(set(map(str, payload.get("target_refs", [])))):
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.EPISTEMIC_TARGET_RELATIONSHIP,
                        roles=("ATTACHMENT",),
                        target_kind="CONTEXT",
                        relation=str(payload.get("target_relation", "REPORTS")),
                        semantic_key="target_ref",
                        expected_value=target_ref,
                        cardinality=GroundingCardinalityPolicy.PER_TARGET,
                        target_key=target_ref,
                        rule_id="context.epistemic.target.per_target",
                    )
                )
            if "modality" in payload:
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.MODALITY,
                        roles=("MODALITY",),
                        target_kind="CONTEXT",
                        semantic_key="modality",
                        expected_value=payload["modality"],
                        rule_id="context.epistemic.modality.present",
                    )
                )
            if not out:
                return _unresolved(
                    node,
                    GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                    "epistemic_semantics_empty",
                )
            return _resolved(node, out)

        simple_map = {
            "CONDITION": ("CONDITION", GroundingObligationType.CONDITION_SEMANTICS),
            "SCOPE": ("SCOPE_VALUE", GroundingObligationType.SCOPE_VALUE),
            "CONFIGURATION": (
                "CONFIGURATION_VALUE",
                GroundingObligationType.CONFIGURATION_VALUE,
            ),
            "MEASUREMENT": ("MEASUREMENT_VALUE", GroundingObligationType.MEASUREMENT_VALUE),
            "MODALITY": ("MODALITY", GroundingObligationType.MODALITY),
            "TEMPORAL": ("TEMPORAL_VALUE", GroundingObligationType.CONTEXT_VALUE),
            "VERSION": ("VERSION_VALUE", GroundingObligationType.CONTEXT_VALUE),
            "SCENARIO": ("SCENARIO_VALUE", GroundingObligationType.CONTEXT_VALUE),
            "SAMPLE_SCOPE": ("SAMPLE_SCOPE_VALUE", GroundingObligationType.CONTEXT_VALUE),
            "OTHER": ("CONTEXT_VALUE", GroundingObligationType.CONTEXT_VALUE),
        }
        if context_type == "MEASUREMENT":
            if value_present:
                out.append(
                    _context_value_obligation(
                        node,
                        role="MEASUREMENT_VALUE",
                        kind=GroundingObligationType.MEASUREMENT_VALUE,
                        value=semantic_value,
                        rule_id="context.measurement.semantic_value.present",
                    )
                )
            for field, role, obligation_type in (
                (
                    "metric_identity",
                    "MEASUREMENT_DEFINITION",
                    GroundingObligationType.METRIC_IDENTITY,
                ),
                ("category_identity", "SCOPE_VALUE", GroundingObligationType.CATEGORY_IDENTITY),
                ("scope_identity", "SCOPE_VALUE", GroundingObligationType.SCOPE_VALUE),
            ):
                if field in payload:
                    out.append(
                        _emit(
                            node,
                            kind=obligation_type,
                            roles=(role,),
                            target_kind="CONTEXT",
                            semantic_key=field,
                            expected_value=payload[field],
                            rule_id=f"measurement.{field}.present",
                        )
                    )
            associations = payload.get("associations", [])
            if associations and not isinstance(associations, list):
                return _unresolved(
                    node,
                    GroundingDerivationStatus.CONTRACT_ERROR,
                    "measurement_associations_must_be_list",
                )
            for association in associations:
                if not isinstance(association, dict) or not association.get("value_key"):
                    return _unresolved(
                        node,
                        GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                        "measurement_association_missing_value_key",
                    )
                assoc_key = str(association["value_key"])
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.NUMERIC_ASSOCIATION,
                        roles=("ATTACHMENT", "RELATION_OPERATOR"),
                        target_kind="CONTEXT",
                        relation=str(association.get("relation", "ASSOCIATED_WITH")),
                        semantic_key="association",
                        expected_value=association,
                        cardinality=GroundingCardinalityPolicy.PER_VALUE,
                        value_key=assoc_key,
                        rule_id="measurement.association.per_value",
                    )
                )
        elif context_type in {"SCOPE", "SHARED_SCOPE"}:
            if value_present:
                out.append(
                    _context_value_obligation(
                        node,
                        role="SCOPE_VALUE",
                        kind=GroundingObligationType.SCOPE_VALUE,
                        value=semantic_value,
                        rule_id="context.scope.semantic_value.present",
                    )
                )
            for member in sorted(set(map(str, payload.get("member_refs", [])))):
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.SCOPE_MEMBERSHIP,
                        roles=("SCOPE_STRUCTURE",),
                        level=GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
                        target_kind="SCOPE",
                        semantic_key="member_ref",
                        expected_value=member,
                        cardinality=GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS,
                        member_key=member,
                        rule_id="context.scope.member.per_expected_member",
                    )
                )
            for target_ref in sorted(set(map(str, payload.get("target_refs", [])))):
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.TARGET_APPLICABILITY,
                        roles=("ATTACHMENT",),
                        target_kind="ATTACHMENT",
                        target_ref=node.node_ref,
                        relation=str(payload.get("target_relation", "APPLIES_TO")),
                        semantic_key="target_ref",
                        expected_value=target_ref,
                        cardinality=GroundingCardinalityPolicy.PER_TARGET,
                        target_key=target_ref,
                        rule_id="context.scope.target.per_target",
                    )
                )
        elif context_type in simple_map:
            if value_present:
                role, obligation_type = simple_map[context_type]
                out.append(
                    _context_value_obligation(
                        node,
                        role=role,
                        kind=obligation_type,
                        value=semantic_value,
                        rule_id=f"context.{context_type.lower()}.semantic_value.present",
                    )
                )
        else:
            return _unresolved(
                node,
                GroundingDerivationStatus.UNSUPPORTED_NODE_SHAPE,
                f"unsupported_context_type:{context_type}",
            )
        if not out:
            return _unresolved(
                node,
                GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                "context_payload_does_not_represent_groundable_semantics",
            )
        targets = payload.get("target_refs", [])
        relation = payload.get("target_relation")
        if (
            context_type not in {"SCOPE", "SHARED_SCOPE", "METHOD", "EPISTEMIC"}
            and targets
            and relation
        ):
            for target_ref in sorted(set(map(str, targets))):
                out.append(
                    _emit(
                        node,
                        kind=GroundingObligationType.TARGET_APPLICABILITY,
                        roles=("ATTACHMENT",),
                        target_kind="ATTACHMENT",
                        target_ref=node.node_ref,
                        relation=str(relation),
                        semantic_key="target_ref",
                        expected_value=target_ref,
                        cardinality=GroundingCardinalityPolicy.PER_TARGET,
                        target_key=target_ref,
                        rule_id="context.explicit_target_relation.per_target",
                    )
                )
        return _resolved(node, out)

    if node.node_kind is GroundingNodeKind.ATTACHMENT:
        context_ref = payload.get("context_ref")
        targets = payload.get("target_refs")
        relation = payload.get("relation")
        if not context_ref or not isinstance(targets, list) or not targets or not relation:
            return _unresolved(
                node,
                GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                "attachment_requires_context_relation_and_targets",
            )
        out.append(
            _emit(
                node,
                kind=GroundingObligationType.STRUCTURAL_REFERENCE,
                roles=("ATTACHMENT",),
                level=GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
                target_kind="ATTACHMENT",
                semantic_key="context_ref",
                expected_value=context_ref,
                rule_id="attachment.references.structural",
            )
        )
        for target_ref in sorted(set(map(str, targets))):
            out.append(
                _emit(
                    node,
                    kind=GroundingObligationType.TARGET_APPLICABILITY,
                    roles=("ATTACHMENT",),
                    target_kind="ATTACHMENT",
                    relation=str(relation),
                    semantic_key="target_ref",
                    expected_value=target_ref,
                    cardinality=GroundingCardinalityPolicy.PER_TARGET,
                    target_key=target_ref,
                    rule_id="attachment.semantic_applicability.per_target",
                )
            )
        return _resolved(node, out)

    if node.node_kind is GroundingNodeKind.SCOPE:
        if "scope_value" not in payload:
            return _unresolved(
                node,
                GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                "scope_value_missing",
            )
        out.append(
            _emit(
                node,
                kind=GroundingObligationType.SCOPE_VALUE,
                roles=("SCOPE_VALUE",),
                target_kind="SCOPE",
                semantic_key="scope_value",
                expected_value=payload["scope_value"],
                rule_id="scope.value.required",
            )
        )
        for member in sorted(set(map(str, payload.get("member_refs", [])))):
            out.append(
                _emit(
                    node,
                    kind=GroundingObligationType.SCOPE_MEMBERSHIP,
                    roles=("SCOPE_STRUCTURE",),
                    level=GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY,
                    target_kind="SCOPE",
                    semantic_key="member_ref",
                    expected_value=member,
                    cardinality=GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS,
                    member_key=member,
                    rule_id="scope.membership.per_expected_member",
                )
            )
        for target_ref in sorted(set(map(str, payload.get("target_refs", [])))):
            out.append(
                _emit(
                    node,
                    kind=GroundingObligationType.TARGET_APPLICABILITY,
                    roles=("ATTACHMENT",),
                    target_kind="ATTACHMENT",
                    target_ref=node.node_ref,
                    relation=str(payload.get("target_relation", "APPLIES_TO")),
                    semantic_key="target_ref",
                    expected_value=target_ref,
                    cardinality=GroundingCardinalityPolicy.PER_TARGET,
                    target_key=target_ref,
                    rule_id="scope.applicability.per_target",
                )
            )
        return _resolved(node, out)

    if node.node_kind is GroundingNodeKind.GROUP:
        group_type = payload.get("group_type")
        if group_type == "STRUCTURAL_GROUP":
            if not payload.get("member_refs"):
                return _unresolved(
                    node,
                    GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                    "structural_group_members_missing",
                )
            return _resolved(
                node,
                [],
                zero_reason=ZeroObligationReason.STRUCTURAL_ONLY_NODE,
                diagnostics=("graph_membership_is_structural_not_semantic_support",),
            )
        if group_type == "ENUMERATION":
            if "count" not in payload:
                return _unresolved(
                    node,
                    GroundingDerivationStatus.DERIVATION_UNRESOLVED,
                    "enumeration_count_missing",
                )
            out.append(
                _emit(
                    node,
                    kind=GroundingObligationType.ENUMERATION_COUNT,
                    roles=("CONTEXT_VALUE",),
                    target_kind="CONTEXT",
                    semantic_key="count",
                    expected_value=payload["count"],
                    rule_id="enumeration.count.required",
                )
            )
            if "members" in payload:
                members = payload["members"]
                if not isinstance(members, list) or not members:
                    return _unresolved(
                        node,
                        GroundingDerivationStatus.CONTRACT_ERROR,
                        "enumeration_members_must_be_nonempty_list",
                    )
                for member in sorted(set(map(str, members))):
                    out.append(
                        _emit(
                            node,
                            kind=GroundingObligationType.ENUMERATION_MEMBER,
                            roles=("CONTEXT_VALUE", "SCOPE_VALUE"),
                            target_kind="CONTEXT",
                            semantic_key="member",
                            expected_value=member,
                            cardinality=GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS,
                            member_key=member,
                            rule_id="enumeration.member.per_expected_member",
                        )
                    )
            return _resolved(node, out)
        if group_type == "NON_SEMANTIC_METADATA":
            return _resolved(
                node,
                [],
                zero_reason=ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED,
                diagnostics=("versioned_metadata_contract_declares_no_semantic_grounding",),
            )
        return _unresolved(
            node,
            GroundingDerivationStatus.UNSUPPORTED_NODE_SHAPE,
            "unsupported_group_type",
        )

    if node.node_kind is GroundingNodeKind.METADATA:
        if payload.get("metadata_contract") == "NON_SEMANTIC_METADATA":
            return _resolved(
                node,
                [],
                zero_reason=ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED,
                diagnostics=("versioned_metadata_contract_declares_no_semantic_grounding",),
            )
        return _unresolved(
            node,
            GroundingDerivationStatus.UNSUPPORTED_NODE_SHAPE,
            "metadata_node_lacks_explicit_nonsemantic_contract",
        )

    return _unresolved(
        node,
        GroundingDerivationStatus.UNSUPPORTED_NODE_SHAPE,
        "unsupported_node_kind",
    )


def _constraint_exclusion(
    obligation: GroundingObligation,
    candidate: GroundingBindingCandidate,
) -> GroundingCandidateExclusionReason | None:
    constraint = obligation.target_constraint
    if candidate.semantic_role not in obligation.expected_semantic_role_family:
        return GroundingCandidateExclusionReason.ROLE_MISMATCH
    if constraint.target_kind and candidate.target_kind != constraint.target_kind:
        return GroundingCandidateExclusionReason.TARGET_KIND_MISMATCH
    if constraint.target_ref and candidate.target_ref != constraint.target_ref:
        return GroundingCandidateExclusionReason.TARGET_REF_MISMATCH
    if constraint.relation and candidate.relation != constraint.relation:
        return GroundingCandidateExclusionReason.RELATION_MISMATCH
    if constraint.scope_ref and candidate.scope_ref != constraint.scope_ref:
        return GroundingCandidateExclusionReason.SCOPE_MISMATCH
    if (
        constraint.semantic_key
        and candidate.canonical_payload.get(constraint.semantic_key) != constraint.expected_value
    ):
        return GroundingCandidateExclusionReason.PAYLOAD_MISMATCH
    if obligation.expected_member_key and candidate.member_key != obligation.expected_member_key:
        return GroundingCandidateExclusionReason.MEMBER_MISMATCH
    if obligation.expected_value_key and candidate.value_key != obligation.expected_value_key:
        return GroundingCandidateExclusionReason.VALUE_MISMATCH
    return None


def _match_obligation(
    obligation: GroundingObligation,
    candidates: tuple[GroundingBindingCandidate, ...],
    *,
    used_bindings: dict[str, GroundingObligation],
) -> GroundingObligationMatch:
    matched: list[GroundingBindingCandidate] = []
    excluded: list[GroundingCandidateExclusion] = []
    for candidate in candidates:
        reason = _constraint_exclusion(obligation, candidate)
        previous = used_bindings.get(candidate.binding_ref)
        if reason is None and previous is not None:
            reuse_ok = (
                obligation.allow_binding_reuse
                and previous.allow_binding_reuse
                and obligation.binding_reuse_group == previous.binding_reuse_group
            )
            if not reuse_ok:
                reason = GroundingCandidateExclusionReason.BINDING_ALREADY_CONSUMED
        if reason is None:
            matched.append(candidate)
        else:
            excluded.append(
                GroundingCandidateExclusion(binding_ref=candidate.binding_ref, reason=reason)
            )

    evidence_refs = [candidate.evidence_ref for candidate in matched]
    unique_evidence = set(evidence_refs)
    duplicates = tuple(sorted(ref for ref, qty in _counts(evidence_refs).items() if qty > 1))
    policy = obligation.cardinality_policy
    if policy is GroundingCardinalityPolicy.AT_LEAST_ONE:
        cardinality_satisfied = bool(matched)
    elif policy is GroundingCardinalityPolicy.ALL_EXPECTED_MEMBERS or policy in {
        GroundingCardinalityPolicy.PER_TARGET,
        GroundingCardinalityPolicy.PER_VALUE,
    }:
        cardinality_satisfied = bool(matched) and len(unique_evidence) == 1
    else:
        cardinality_satisfied = len(unique_evidence) == 1
    diagnostics: list[str] = []
    if duplicates:
        diagnostics.append("duplicate_binding_candidates_share_evidence_ref")
    if len(unique_evidence) > 1 and policy is not GroundingCardinalityPolicy.AT_LEAST_ONE:
        diagnostics.append("exactly_one_cardinality_has_multiple_distinct_evidence_items")
    return GroundingObligationMatch(
        obligation_ref=obligation.obligation_ref,
        candidate_binding_refs=tuple(item.binding_ref for item in candidates),
        matched_binding_refs=tuple(item.binding_ref for item in matched),
        excluded_candidates=tuple(excluded),
        duplicate_evidence_refs=duplicates,
        cardinality_satisfied=cardinality_satisfied,
        diagnostics=tuple(diagnostics),
    )


def _counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _binding_status(
    obligation: GroundingObligation,
    candidate: GroundingBindingCandidate,
) -> GroundingObligationStatus:
    if candidate.textual_grounding is not GroundingDimensionStatus.VALID:
        if candidate.textual_grounding is GroundingDimensionStatus.INVALID:
            return GroundingObligationStatus.TEXTUALLY_INVALID
        return GroundingObligationStatus.INSUFFICIENT_EVIDENCE
    if candidate.structural_grounding is not GroundingDimensionStatus.VALID:
        if candidate.structural_grounding is GroundingDimensionStatus.INVALID:
            return GroundingObligationStatus.STRUCTURALLY_INVALID
        return GroundingObligationStatus.INSUFFICIENT_EVIDENCE
    if obligation.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY:
        if candidate.structural_grounding is GroundingDimensionStatus.VALID:
            return GroundingObligationStatus.SATISFIED
        return GroundingObligationStatus.INSUFFICIENT_EVIDENCE
    return {
        GroundingDimensionStatus.VALID: GroundingObligationStatus.SATISFIED,
        GroundingDimensionStatus.INVALID: GroundingObligationStatus.NOT_SUPPORTED,
        GroundingDimensionStatus.AMBIGUOUS: GroundingObligationStatus.AMBIGUOUS,
        GroundingDimensionStatus.MISSING_EVIDENCE: GroundingObligationStatus.INSUFFICIENT_EVIDENCE,
        GroundingDimensionStatus.NOT_EVALUATED: GroundingObligationStatus.INSUFFICIENT_EVIDENCE,
    }[candidate.semantic_support]


def _combine_candidate_statuses(
    obligation: GroundingObligation,
    candidates: list[GroundingBindingCandidate],
) -> GroundingObligationStatus:
    if not candidates:
        if obligation.requirement_level is GroundingRequirementLevel.OPTIONAL_CORROBORATION:
            return GroundingObligationStatus.INSUFFICIENT_EVIDENCE
        return GroundingObligationStatus.MISSING_REQUIRED_BINDING
    outcomes = [_binding_status(obligation, item) for item in candidates]
    if obligation.cardinality_policy is GroundingCardinalityPolicy.AT_LEAST_ONE:
        if GroundingObligationStatus.SATISFIED in outcomes:
            return GroundingObligationStatus.SATISFIED
        if GroundingObligationStatus.TEXTUALLY_INVALID in outcomes:
            return GroundingObligationStatus.TEXTUALLY_INVALID
        if GroundingObligationStatus.STRUCTURALLY_INVALID in outcomes:
            return GroundingObligationStatus.STRUCTURALLY_INVALID
        if GroundingObligationStatus.NOT_SUPPORTED in outcomes:
            return GroundingObligationStatus.NOT_SUPPORTED
        if GroundingObligationStatus.AMBIGUOUS in outcomes:
            return GroundingObligationStatus.AMBIGUOUS
        return GroundingObligationStatus.INSUFFICIENT_EVIDENCE
    if len({item.evidence_ref for item in candidates}) > 1:
        return GroundingObligationStatus.AMBIGUOUS
    if len(set(outcomes)) > 1:
        return GroundingObligationStatus.AMBIGUOUS
    return outcomes[0]


def evaluate_grounding_node(
    derivation: GroundingObligationDerivationResult,
    candidates: tuple[GroundingBindingCandidate, ...],
    *,
    node_textual_grounding: GroundingDimensionStatus,
    node_structural_grounding: GroundingDimensionStatus,
) -> NodeGroundingAssessment:
    """Evaluate obligations and aggregate a shadow-only node readiness result."""
    if derivation.status is not GroundingDerivationStatus.RESOLVED:
        return NodeGroundingAssessment(
            node_ref=derivation.node_ref,
            node_kind=derivation.node_kind,
            derivation=derivation,
            usability=GroundingNodeUsability.NOT_EVALUATED,
            verification_assertion_ready=False,
            aggregation_precedence=("DERIVATION_FAILURE",),
            diagnostics=(*derivation.diagnostics, "obligation_derivation_failed_closed"),
        )

    if not derivation.obligations:
        if derivation.zero_obligation_reason is ZeroObligationReason.STRUCTURAL_ONLY_NODE:
            usable = (
                GroundingNodeUsability.STRUCTURALLY_GROUNDED
                if node_textual_grounding is GroundingDimensionStatus.VALID
                and node_structural_grounding is GroundingDimensionStatus.VALID
                else GroundingNodeUsability.UNGROUNDED_UNUSABLE
            )
        elif (
            derivation.zero_obligation_reason is ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED
        ):
            usable = GroundingNodeUsability.NO_SEMANTIC_GROUNDING_REQUIRED
        else:
            usable = GroundingNodeUsability.NOT_EVALUATED
        return NodeGroundingAssessment(
            node_ref=derivation.node_ref,
            node_kind=derivation.node_kind,
            derivation=derivation,
            usability=usable,
            verification_assertion_ready=False,
            aggregation_precedence=("EXPLICIT_ZERO_OBLIGATION_POLICY",),
            diagnostics=derivation.diagnostics,
        )

    used_bindings: dict[str, GroundingObligation] = {}
    evaluations: list[GroundingObligationEvaluation] = []
    omissions: list[GroundingRequirementDiagnostics] = []
    for obligation in derivation.obligations:
        match = _match_obligation(obligation, candidates, used_bindings=used_bindings)
        matched_candidates = [
            item for item in candidates if item.binding_ref in match.matched_binding_refs
        ]
        if (
            not matched_candidates
            and obligation.requirement_level is not GroundingRequirementLevel.OPTIONAL_CORROBORATION
        ):
            omissions.append(
                GroundingRequirementDiagnostics(
                    code="MISSING_REQUIRED_BINDING",
                    node_ref=obligation.node_ref,
                    obligation_ref=obligation.obligation_ref,
                    expected_role_family=obligation.expected_semantic_role_family,
                    target_constraint=obligation.target_constraint,
                    reason="no candidate binding matched the independently derived obligation",
                )
            )
        status = _combine_candidate_statuses(obligation, matched_candidates)
        if not match.cardinality_satisfied and matched_candidates:
            status = GroundingObligationStatus.AMBIGUOUS
        for candidate in matched_candidates:
            used_bindings.setdefault(candidate.binding_ref, obligation)
        binding_statuses = tuple(
            (
                item.binding_ref,
                item.textual_grounding.value,
                item.structural_grounding.value,
                item.semantic_support.value,
            )
            for item in matched_candidates
        )
        diagnostics = list(match.diagnostics)
        if status is GroundingObligationStatus.MISSING_REQUIRED_BINDING:
            diagnostics.append("extraction_omission_candidate_missing")
        evaluations.append(
            GroundingObligationEvaluation(
                obligation=obligation,
                match=match,
                status=status,
                binding_statuses=binding_statuses,
                diagnostics=tuple(diagnostics),
                provenance={
                    "node_ref": obligation.node_ref,
                    "contract_version": obligation.contract_version,
                    "derivation_rule_id": obligation.derivation_rule_id,
                    "obligation_ref": obligation.obligation_ref,
                    "candidate_binding_refs": list(match.candidate_binding_refs),
                    "matched_binding_refs": list(match.matched_binding_refs),
                    "excluded_candidates": [
                        item.model_dump(mode="json") for item in match.excluded_candidates
                    ],
                    "binding_grounding_and_support": [list(item) for item in binding_statuses],
                    "obligation_status": status.value,
                    "binding_reuse_permitted": obligation.allow_binding_reuse,
                    "binding_reuse_group": obligation.binding_reuse_group,
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
    node_dimension_states = {
        "textual": node_textual_grounding,
        "structural": node_structural_grounding,
    }
    node_failures = [
        f"node_{dimension}_grounding_{status.value.lower()}"
        for dimension, status in node_dimension_states.items()
        if status is not GroundingDimensionStatus.VALID
    ]
    all_mandatory = required + structural
    states = {item.status for item in all_mandatory}
    if (
        GroundingDimensionStatus.INVALID in node_dimension_states.values()
        or GroundingObligationStatus.TEXTUALLY_INVALID in states
        or GroundingObligationStatus.STRUCTURALLY_INVALID in states
    ):
        usability = GroundingNodeUsability.UNGROUNDED_UNUSABLE
        precedence = "MANDATORY_TEXTUAL_OR_STRUCTURAL_FAILURE"
    elif GroundingDimensionStatus.AMBIGUOUS in node_dimension_states.values():
        usability = GroundingNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS
        precedence = "NODE_GROUNDING_AMBIGUOUS"
    elif GroundingObligationStatus.NOT_SUPPORTED in states:
        usability = GroundingNodeUsability.UNGROUNDED_UNUSABLE
        precedence = "REQUIRED_NOT_SUPPORTED"
    elif GroundingObligationStatus.AMBIGUOUS in states:
        usability = GroundingNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS
        precedence = "REQUIRED_AMBIGUOUS"
    elif any(
        status
        in {
            GroundingDimensionStatus.MISSING_EVIDENCE,
            GroundingDimensionStatus.NOT_EVALUATED,
        }
        for status in node_dimension_states.values()
    ) or states.intersection(
        {
            GroundingObligationStatus.MISSING_REQUIRED_BINDING,
            GroundingObligationStatus.INSUFFICIENT_EVIDENCE,
            GroundingObligationStatus.DERIVATION_UNRESOLVED,
        }
    ):
        usability = GroundingNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED
        precedence = "REQUIRED_MISSING_OR_INSUFFICIENT"
    elif (
        required
        and all(item.status is GroundingObligationStatus.SATISFIED for item in required)
        and all(item.status is GroundingObligationStatus.SATISFIED for item in structural)
    ):
        usability = GroundingNodeUsability.GROUNDED_USABLE
        precedence = "ALL_REQUIRED_AND_STRUCTURAL_OBLIGATIONS_SATISFIED"
    elif (
        structural
        and not required
        and all(item.status is GroundingObligationStatus.SATISFIED for item in structural)
    ):
        usability = GroundingNodeUsability.STRUCTURALLY_GROUNDED
        precedence = "STRUCTURAL_ONLY_OBLIGATIONS_SATISFIED"
    else:
        usability = GroundingNodeUsability.NOT_EVALUATED
        precedence = "NO_REQUIRED_SEMANTIC_OBLIGATION"

    optional_failures = [
        item
        for item in evaluations
        if item.obligation.requirement_level is GroundingRequirementLevel.OPTIONAL_CORROBORATION
        and item.status is not GroundingObligationStatus.SATISFIED
    ]
    diagnostics = list(derivation.diagnostics)
    diagnostics.extend(node_failures)
    diagnostics.extend(
        f"obligation_state:{item.obligation.obligation_ref}:{item.status.value}"
        for item in evaluations
        if item.status is not GroundingObligationStatus.SATISFIED
    )
    if optional_failures:
        diagnostics.append("optional_corroboration_failure_diagnostic_only")
    return NodeGroundingAssessment(
        node_ref=derivation.node_ref,
        node_kind=derivation.node_kind,
        derivation=derivation,
        obligation_evaluations=tuple(evaluations),
        usability=usability,
        verification_assertion_ready=usability is GroundingNodeUsability.GROUNDED_USABLE,
        aggregation_precedence=(precedence,),
        diagnostics=tuple(diagnostics),
        extraction_omissions=tuple(omissions),
    )
