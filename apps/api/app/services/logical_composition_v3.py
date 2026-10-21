"""Offline V3 response contract and resolver for logical assertion composition.

This module is shadow-only. It reuses V2 proposition parsing and qualifier checks,
then resolves V3-local composition references to runtime proposition IDs.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
from app.domain.atomic_proposition import (
    ATOMIC_PROPOSITION_SCHEMA_VERSION,
    AtomicProposition,
    DecompositionMetadata,
)
from app.domain.atomic_proposition_contracts import (
    ATOMIC_DECOMPOSITION_PROMPT_V3,
    ATOMIC_PROPOSITION_SCHEMA_V3,
)
from app.services.atomic_proposition_validation import (
    PropositionPayload,
    QualifierCheck,
    ShadowDecompositionLabel,
    ValidationDiagnostic,
    ValidationIssue,
    validate_decomposition_response,
)

V3_COMPOSITION_INSTRUCTIONS = """
Decompose the supplied Claim conservatively into truth-condition-complete atomic
propositions. Keep a Claim that expresses one inseparable fact as one proposition.
Preserve all material qualifiers, attribution, modality, negation, time, scope, and
causal/relational meaning in the proposition fields and source spans.

Return the requested sparse assertion_composition only when the logical relationship
between propositions is material to the source truth conditions. A single ordinary
proposition needs no composition. For multiple independently asserted top-level
expressions, list each as a root expression; use an AND group only when conjunction is
materially expressed. Preserve OR and XOR rather than turning alternatives into separate
independent assertions. Use XOR only when exclusivity is explicit; otherwise use OR or
UNKNOWN as the source warrants. The model describes structure only and must not assign
verification eligibility.

Represent IF P THEN Q as Q with material condition P when safely expressible; do not
assert P independently. Preserve a material causal relation as a relational proposition,
not as a Boolean operator. Do not map ambiguous while, but, although, or similar
connectives to a guessed Boolean operator; keep the proposition inseparable or mark the
composition UNKNOWN. Do not use domain-specific rules or infer unstated relations.

Use explicit response-local proposition references p0, p1, ... and group references g0,
g1, ... . Composition roots identify exactly which expressions the Claim asserts.
Every returned proposition must be reachable from a root. Include source spans only when
they are literal source excerpts; never fabricate them. Ordinary single propositions
remain composition-free. If truth conditions cannot be safely represented, return an
unresolved decomposition with no propositions and a concise ambiguity reason.
""".strip()


class V3CompositionOperator(StrEnum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    UNKNOWN = "UNKNOWN"


class V3Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    member_type: Literal["proposition", "group"]
    ref: str = Field(pattern=r"^(p|g)[0-9]{1,3}$")
    source_span: str | None = Field(default=None, min_length=1, max_length=2_000)


class V3PropositionPayload(PropositionPayload):
    local_ref: str = Field(pattern=r"^p[0-9]{1,3}$")


class V3GroupPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    local_ref: str = Field(pattern=r"^g[0-9]{1,3}$")
    operator: V3CompositionOperator
    members: tuple[V3Reference, ...] = Field(min_length=2, max_length=32)
    source_span: str | None = Field(default=None, min_length=1, max_length=4_000)


class V3AssertionCompositionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    root_expressions: tuple[V3Reference, ...] = Field(min_length=1, max_length=32)
    groups: tuple[V3GroupPayload, ...] = Field(default=(), max_length=32)


class V3ShadowDecompositionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["atomic-proposition-schema-v3"]
    decomposition_status: ShadowDecompositionLabel
    propositions: tuple[V3PropositionPayload, ...] = Field(max_length=32)
    ambiguity_reason: str | None = Field(default=None, max_length=1_000)
    assertion_composition: V3AssertionCompositionPayload | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> V3ShadowDecompositionResponse:
        count = len(self.propositions)
        if self.decomposition_status is ShadowDecompositionLabel.SINGLE and count != 1:
            raise ValueError("single_proposition requires exactly one proposition")
        if self.decomposition_status is ShadowDecompositionLabel.DECOMPOSED and count < 2:
            raise ValueError("decomposed output requires at least two propositions")
        if self.decomposition_status is ShadowDecompositionLabel.UNRESOLVED:
            if count or self.assertion_composition is not None or not self.ambiguity_reason:
                raise ValueError("unresolved output requires no propositions or composition")
        elif self.ambiguity_reason is not None:
            raise ValueError("resolved output cannot contain an ambiguity reason")
        if count > 1 and self.assertion_composition is None:
            raise ValueError("multiple propositions require explicit asserted roots")
        return self


class CompositionFailureCategory(StrEnum):
    COMPOSITION_SCHEMA_INVALID = "COMPOSITION_SCHEMA_INVALID"
    COMPOSITION_REFERENCE_INVALID = "COMPOSITION_REFERENCE_INVALID"
    COMPOSITION_CYCLE = "COMPOSITION_CYCLE"
    COMPOSITION_SEMANTIC_INVALID = "COMPOSITION_SEMANTIC_INVALID"
    COMPOSITION_ELIGIBILITY_INCONSISTENT = "COMPOSITION_ELIGIBILITY_INCONSISTENT"
    COMPOSITION_UNRESOLVED = "COMPOSITION_UNRESOLVED"


class V3ValidatedDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: str = ATOMIC_PROPOSITION_SCHEMA_V3
    decomposition_status: ShadowDecompositionLabel | None = None
    propositions: tuple[AtomicProposition, ...] = ()
    assertion_composition: AssertionComposition | None = None
    root_assertions: tuple[dict[str, str | int], ...] = ()
    model_local_references: dict[str, str] = Field(default_factory=dict)
    derived_member_eligibility: dict[str, str] = Field(default_factory=dict)
    validation_issues: tuple[ValidationIssue, ...] = ()
    validation_diagnostics: tuple[ValidationDiagnostic, ...] = ()
    qualifier_checks: tuple[QualifierCheck, ...] = ()
    composition_source_span_trace: tuple[dict[str, Any], ...] = ()
    composition_source_span_trace_truncated: bool = False
    composition_source_span_trace_omitted_count: int = 0
    composition_source_span_trace_error_count: int = 0
    composition_failure_category: CompositionFailureCategory | None = None
    ambiguity_reason: str | None = None
    valid: bool


def validate_v3_decomposition_response(
    raw_response: dict[str, Any] | str | None,
    *,
    claim_text: str,
    claim_id: str | None,
    provenance: dict[str, str] | None = None,
) -> V3ValidatedDecomposition:
    """Parse V3, validate graph semantics, then reuse the frozen V2 proposition validator."""
    del provenance  # The enclosing shadow result retains safe provenance.
    try:
        payload: Any = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        response = V3ShadowDecompositionResponse.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        composition_error = any(
            "assertion_composition" in tuple(str(part) for part in item.get("loc", ()))
            for item in errors
        )
        if composition_error:
            return _composition_failure(
                CompositionFailureCategory.COMPOSITION_SCHEMA_INVALID,
                "composition_schema_invalid",
            )
        if any(
            item.get("msg") == "Value error, multiple propositions require explicit asserted roots"
            for item in errors
        ):
            return _composition_failure(
                CompositionFailureCategory.COMPOSITION_UNRESOLVED,
                "composition_roots_missing",
            )
        return V3ValidatedDecomposition(
            validation_issues=(ValidationIssue(code="schema_invalid", message="schema invalid"),),
            valid=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return _composition_failure(
            CompositionFailureCategory.COMPOSITION_SCHEMA_INVALID,
            "composition_schema_invalid",
        )

    if response.decomposition_status is ShadowDecompositionLabel.UNRESOLVED:
        return V3ValidatedDecomposition(
            decomposition_status=response.decomposition_status,
            ambiguity_reason=response.ambiguity_reason,
            valid=True,
        )

    source_span_trace = CompositionSourceSpanTrace()
    for item in response.propositions:
        span = item.source_span
        compatible = span is None or span in claim_text
        source_span_trace.try_record(
            {
                "parent_ref": "claim",
                "parent_kind": "claim",
                "parent_source_span": claim_text,
                "child_ref": item.local_ref,
                "child_kind": "proposition",
                "child_source_span": span,
                "resolved_proposition_id": None,
                "resolved_proposition_source_span": span,
                "comparison_rule": "LITERAL_PROPOSITION_SPAN_SUBSTRING_OF_CLAIM",
                "claim_compatibility_result": (
                    "NOT_CHECKED" if span is None else "PASS" if compatible else "FAIL"
                ),
                "parent_compatibility_result": "NOT_CHECKED",
                "compatibility_result": (
                    "NOT_CHECKED" if span is None else "PASS" if compatible else "FAIL"
                ),
                "validation_reason_code": (
                    "optional_span_missing"
                    if span is None
                    else None
                    if compatible
                    else "proposition_source_span_not_in_claim"
                ),
                "field_path": f"propositions[{item.local_ref}].source_span",
                "graph_path": [item.local_ref],
            }
        )
    if any(
        item.source_span is not None and item.source_span not in claim_text
        for item in response.propositions
    ):
        return _composition_failure(
            CompositionFailureCategory.COMPOSITION_SEMANTIC_INVALID,
            "proposition_source_span_not_in_claim",
            source_span_trace=source_span_trace,
        )

    local_propositions = {item.local_ref: item for item in response.propositions}
    if len(local_propositions) != len(response.propositions):
        return _composition_failure(
            CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID,
            "duplicate_local_proposition_ref",
        )

    composition_payload = response.assertion_composition
    if composition_payload is not None:
        issue = _validate_local_graph(composition_payload, set(local_propositions))
        if issue is not None:
            category, code = issue
            return _composition_failure(category, code)

    legacy_payload = {
        "schema_version": ATOMIC_PROPOSITION_SCHEMA_VERSION,
        "decomposition_status": response.decomposition_status.value,
        "propositions": [
            item.model_dump(mode="json", exclude={"local_ref"}) for item in response.propositions
        ],
        "ambiguity_reason": response.ambiguity_reason,
    }
    parsed = validate_decomposition_response(
        legacy_payload,
        claim_text=claim_text,
        claim_id=claim_id,
        schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
    )
    if not parsed.valid:
        return V3ValidatedDecomposition(
            decomposition_status=response.decomposition_status,
            validation_issues=parsed.validation_issues,
            validation_diagnostics=parsed.validation_diagnostics,
            qualifier_checks=parsed.qualifier_checks,
            ambiguity_reason=parsed.ambiguity_reason,
            valid=False,
        )

    propositions = tuple(
        _stamp_proposition_v3(item, proposition_index=index)
        for index, item in enumerate(parsed.propositions)
    )
    local_to_runtime = {
        payload_item.local_ref: str(proposition.proposition_id)
        for payload_item, proposition in zip(response.propositions, propositions, strict=True)
    }
    source_spans_by_ref = {
        item.local_ref: item.source_span for item in response.propositions
    }
    for check in source_span_trace.checks:
        local_ref = check.get("child_ref")
        if (
            check.get("child_kind") == "proposition"
            and local_ref in local_to_runtime
            and check.get("field_path", "").startswith("propositions[")
        ):
            check["resolved_proposition_id"] = local_to_runtime[local_ref]
            check["resolved_proposition_source_span"] = source_spans_by_ref[local_ref]
    if composition_payload is None:
        return V3ValidatedDecomposition(
            decomposition_status=response.decomposition_status,
            propositions=propositions,
            root_assertions=(
                {
                    "member_type": "proposition",
                    "ref": response.propositions[0].local_ref,
                    "position": 0,
                    **(
                        {"source_span": response.propositions[0].source_span}
                        if response.propositions[0].source_span is not None
                        else {}
                    ),
                },
            ),
            model_local_references=local_to_runtime,
            derived_member_eligibility={
                response.propositions[
                    0
                ].local_ref: VerificationEligibility.INDEPENDENTLY_ASSERTED.value
            },
            validation_issues=parsed.validation_issues,
            validation_diagnostics=parsed.validation_diagnostics,
            qualifier_checks=parsed.qualifier_checks,
            composition_source_span_trace=tuple(source_span_trace.checks),
            composition_source_span_trace_truncated=source_span_trace.truncated,
            composition_source_span_trace_omitted_count=source_span_trace.omitted_check_count,
            composition_source_span_trace_error_count=source_span_trace.recording_error_count,
            valid=True,
        )

    try:
        composition, eligibility = _resolve_composition(
            composition_payload,
            response.propositions,
            propositions,
            claim_text=claim_text,
            claim_id=claim_id,
            source_span_trace=source_span_trace,
        )
    except AssertionCompositionValidationError as exc:
        category, safe_code = _domain_failure(exc.code)
        return _composition_failure(
            category,
            safe_code,
            source_span_trace=source_span_trace,
        )

    roots = tuple(
        {
            "member_type": item.member_type,
            "ref": item.ref,
            "position": index,
            **({"source_span": item.source_span} if item.source_span is not None else {}),
        }
        for index, item in enumerate(composition_payload.root_expressions)
    )
    return V3ValidatedDecomposition(
        decomposition_status=response.decomposition_status,
        propositions=propositions,
        assertion_composition=composition,
        root_assertions=roots,
        model_local_references=local_to_runtime,
        derived_member_eligibility={
            local_ref: value.value for local_ref, value in eligibility.items()
        },
        validation_issues=parsed.validation_issues,
        validation_diagnostics=parsed.validation_diagnostics,
        qualifier_checks=parsed.qualifier_checks,
        composition_source_span_trace=tuple(source_span_trace.checks),
        composition_source_span_trace_truncated=source_span_trace.truncated,
        composition_source_span_trace_omitted_count=source_span_trace.omitted_check_count,
        composition_source_span_trace_error_count=source_span_trace.recording_error_count,
        valid=True,
    )


def _stamp_proposition_v3(
    proposition: AtomicProposition, *, proposition_index: int
) -> AtomicProposition:
    decomposition = DecompositionMetadata.model_validate(
        proposition.decomposition.model_dump(mode="python")
        | {
            "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
            "prompt_version": ATOMIC_DECOMPOSITION_PROMPT_V3,
            "proposition_index": proposition_index,
        }
    )
    return proposition.model_copy(
        update={
            "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
            "decomposition": decomposition,
        }
    )


def _validate_local_graph(
    composition: V3AssertionCompositionPayload, proposition_refs: set[str]
) -> tuple[CompositionFailureCategory, str] | None:
    group_map = {group.local_ref: group for group in composition.groups}
    if len(group_map) != len(composition.groups):
        return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, "duplicate_local_group_ref"
    root_group_refs = {
        root.ref for root in composition.root_expressions if root.member_type == "group"
    }
    if len(root_group_refs) != sum(
        root.member_type == "group" for root in composition.root_expressions
    ):
        return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, "duplicate_root_group_ref"
    seen_groups: set[str] = set()
    active_groups: set[str] = set()
    seen_props: set[str] = set()

    def add_prop(ref: str) -> tuple[CompositionFailureCategory, str] | None:
        if ref not in proposition_refs:
            return (
                CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID,
                "unknown_proposition_ref",
            )
        if ref in seen_props:
            return (
                CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID,
                "duplicate_proposition_ref",
            )
        seen_props.add(ref)
        return None

    def visit_group(group_ref: str) -> tuple[CompositionFailureCategory, str] | None:
        if group_ref in active_groups:
            return CompositionFailureCategory.COMPOSITION_CYCLE, "composition_cycle"
        group = group_map.get(group_ref)
        if group is None:
            return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, "unknown_group_ref"
        if group_ref in seen_groups:
            return (
                CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID,
                "group_referenced_more_than_once",
            )
        active_groups.add(group_ref)
        local_refs = [(member.member_type, member.ref) for member in group.members]
        if len(local_refs) != len(set(local_refs)):
            return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, "duplicate_member_ref"
        for member in group.members:
            failure = (
                add_prop(member.ref)
                if member.member_type == "proposition"
                else visit_group(member.ref)
            )
            if failure is not None:
                return failure
        active_groups.remove(group_ref)
        seen_groups.add(group_ref)
        return None

    root_pairs = [(root.member_type, root.ref) for root in composition.root_expressions]
    if len(root_pairs) != len(set(root_pairs)):
        return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, "duplicate_root_ref"
    for root in composition.root_expressions:
        failure = add_prop(root.ref) if root.member_type == "proposition" else visit_group(root.ref)
        if failure is not None:
            return failure
    if seen_groups != set(group_map):
        return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, "unreachable_group"
    if seen_props != proposition_refs:
        return CompositionFailureCategory.COMPOSITION_UNRESOLVED, "unreachable_proposition"
    return None


def _resolve_composition(
    payload: V3AssertionCompositionPayload,
    wire_props: tuple[V3PropositionPayload, ...],
    propositions: tuple[AtomicProposition, ...],
    *,
    claim_text: str,
    claim_id: str | None,
    source_span_trace: CompositionSourceSpanTrace,
) -> tuple[AssertionComposition, dict[str, VerificationEligibility]]:
    try:
        if claim_id is None:
            raise AssertionCompositionValidationError("invalid_claim_reference")
        parsed_claim_id = UUID(claim_id)
    except ValueError as exc:
        raise AssertionCompositionValidationError("invalid_claim_reference") from exc
    local_to_runtime = {
        wire.local_ref: proposition
        for wire, proposition in zip(wire_props, propositions, strict=True)
    }
    groups_by_ref = {group.local_ref: group for group in payload.groups}
    eligibility: dict[str, VerificationEligibility] = {}

    def effective(
        parent: VerificationEligibility, operator: V3CompositionOperator
    ) -> VerificationEligibility:
        if parent is VerificationEligibility.UNKNOWN or operator is V3CompositionOperator.UNKNOWN:
            return VerificationEligibility.UNKNOWN
        if operator in {V3CompositionOperator.OR, V3CompositionOperator.XOR}:
            return VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
        return parent

    def member(
        ref: V3Reference, position: int, expected: VerificationEligibility
    ) -> AssertionGroupMember:
        if ref.member_type == "group":
            return AssertionGroupMember(
                member_type="group",
                position=position,
                group_id=group_ids[ref.ref],
                source_span=ref.source_span,
            )
        eligibility[ref.ref] = expected
        return AssertionGroupMember(
            member_type="proposition",
            position=position,
            proposition_id=local_to_runtime[ref.ref].proposition_id,
            verification_eligibility=expected,
            source_span=ref.source_span,
        )

    group_ids = {
        local_ref: uuid5(parsed_claim_id, f"assertion-group:{local_ref}")
        for local_ref in groups_by_ref
    }

    def derive_group(group_ref: str, inherited: VerificationEligibility) -> None:
        group = groups_by_ref[group_ref]
        child_eligibility = effective(inherited, group.operator)
        for ref in group.members:
            if ref.member_type == "proposition":
                eligibility[ref.ref] = child_eligibility
            else:
                derive_group(ref.ref, child_eligibility)

    for root in payload.root_expressions:
        if root.member_type == "proposition":
            eligibility[root.ref] = VerificationEligibility.INDEPENDENTLY_ASSERTED
        else:
            derive_group(root.ref, VerificationEligibility.INDEPENDENTLY_ASSERTED)

    # The model never supplies eligibility; every proposition leaf is stamped from its
    # validated root-to-leaf operator path.
    domain_groups = tuple(
        AssertionGroup(
            group_id=group_ids[local_ref],
            operator=AssertionGroupOperator(wire_group.operator.value),
            members=tuple(
                member(
                    ref,
                    index,
                    eligibility[ref.ref]
                    if ref.member_type == "proposition"
                    else VerificationEligibility.UNKNOWN,
                )
                for index, ref in enumerate(wire_group.members)
            ),
            source_span=wire_group.source_span,
        )
        for local_ref, wire_group in groups_by_ref.items()
    )
    roots = tuple(
        member(
            ref,
            index,
            eligibility[ref.ref]
            if ref.member_type == "proposition"
            else VerificationEligibility.UNKNOWN,
        )
        for index, ref in enumerate(payload.root_expressions)
    )
    source_spans = {
        local_ref: proposition.decomposition.source_span
        for local_ref, proposition in local_to_runtime.items()
    }
    composition = AssertionComposition(
        claim_id=parsed_claim_id,
        root_expressions=roots,
        groups=domain_groups,
    )
    validate_assertion_composition(
        composition,
        proposition_ids={item.proposition_id for item in propositions},
        claim_text=claim_text,
        proposition_source_spans={
            local_to_runtime[local_ref].proposition_id: span
            for local_ref, span in source_spans.items()
            if span is not None
        },
        source_span_trace=source_span_trace,
        reference_labels={
            **{
                group_ids[local_ref]: (local_ref, "group")
                for local_ref in groups_by_ref
            },
            **{
                local_to_runtime[local_ref].proposition_id: (local_ref, "proposition")
                for local_ref in local_to_runtime
            },
        },
    )
    return composition, eligibility


def _domain_failure(code: str) -> tuple[CompositionFailureCategory, str]:
    if code == "composition_cycle":
        return CompositionFailureCategory.COMPOSITION_CYCLE, code
    if code == "verification_eligibility_mismatch":
        return CompositionFailureCategory.COMPOSITION_ELIGIBILITY_INCONSISTENT, code
    if code in {
        "unknown_root_group",
        "unknown_group_reference",
        "unknown_proposition_reference",
        "group_referenced_more_than_once",
        "duplicate_proposition_reference",
        "unreachable_group",
        "proposition_not_in_composition",
    }:
        return CompositionFailureCategory.COMPOSITION_REFERENCE_INVALID, code
    if code == "unreachable_proposition":
        return CompositionFailureCategory.COMPOSITION_UNRESOLVED, code
    return CompositionFailureCategory.COMPOSITION_SEMANTIC_INVALID, code


def _composition_failure(
    category: CompositionFailureCategory,
    code: str,
    *,
    schema_version: str = ATOMIC_PROPOSITION_SCHEMA_V3,
    source_span_trace: CompositionSourceSpanTrace | None = None,
) -> V3ValidatedDecomposition:
    return V3ValidatedDecomposition(
        schema_version=schema_version,
        validation_issues=(ValidationIssue(code=code, message=code.replace("_", " ")),),
        validation_diagnostics=(
            ValidationDiagnostic(
                validation_stage="ASSERTION_COMPOSITION",
                issue_code=code,
                field_path="assertion_composition",
                expected_contract=category.value,
                observed_type="structured_output",
            ),
        ),
        composition_source_span_trace=tuple(source_span_trace.checks)
        if source_span_trace is not None
        else (),
        composition_source_span_trace_truncated=(source_span_trace.truncated)
        if source_span_trace is not None
        else False,
        composition_source_span_trace_omitted_count=(source_span_trace.omitted_check_count)
        if source_span_trace is not None
        else 0,
        composition_source_span_trace_error_count=(source_span_trace.recording_error_count)
        if source_span_trace is not None
        else 0,
        composition_failure_category=category,
        valid=False,
    )
