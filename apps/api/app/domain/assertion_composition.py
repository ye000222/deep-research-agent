"""Passive, fail-closed composition contract for compound assertions.

This module is deliberately not imported by ResearchLoop or verification consumers.
It preserves truth-condition metadata for future shadow evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

ASSERTION_COMPOSITION_SCHEMA_VERSION: Literal["assertion-composition-v1"] = (
    "assertion-composition-v1"
)


class AssertionGroupOperator(StrEnum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    UNKNOWN = "UNKNOWN"


class VerificationEligibility(StrEnum):
    INDEPENDENTLY_ASSERTED = "INDEPENDENTLY_ASSERTED"
    NOT_INDEPENDENTLY_ASSERTED = "NOT_INDEPENDENTLY_ASSERTED"
    UNKNOWN = "UNKNOWN"


class AssertionGroupMember(BaseModel):
    """Ordered reference to either a proposition leaf or another group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member_type: Literal["proposition", "group"]
    position: int = Field(ge=0)
    proposition_id: UUID | None = None
    group_id: UUID | None = None
    verification_eligibility: VerificationEligibility | None = None
    source_span: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_reference_shape(self) -> AssertionGroupMember:
        if self.member_type == "proposition":
            if self.proposition_id is None or self.group_id is not None:
                raise ValueError("proposition member requires only proposition_id")
            if self.verification_eligibility is None:
                raise ValueError("proposition member requires verification eligibility")
        elif self.group_id is None or self.proposition_id is not None:
            raise ValueError("group member requires only group_id")
        elif self.verification_eligibility is not None:
            raise ValueError("group references do not have proposition eligibility")
        return self


class AssertionGroup(BaseModel):
    """A sparse, ordered composition node over propositions and nested groups."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: UUID
    operator: AssertionGroupOperator
    members: tuple[AssertionGroupMember, ...] = Field(min_length=2, max_length=32)
    source_span: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_member_order_and_uniqueness(self) -> AssertionGroup:
        if tuple(member.position for member in self.members) != tuple(range(len(self.members))):
            raise ValueError("group member positions must be unique and contiguous from zero")
        references = [
            (member.member_type, member.proposition_id or member.group_id)
            for member in self.members
        ]
        if len(references) != len(set(references)):
            raise ValueError("group cannot contain duplicate member references")
        return self


class AssertionComposition(BaseModel):
    """Composition metadata attached to a Claim without replacing propositions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: UUID
    schema_version: Literal["assertion-composition-v1"] = ASSERTION_COMPOSITION_SCHEMA_VERSION
    # root_group_id remains accepted for the original single-root domain prototype.
    # root_expressions represents a forest of independently asserted root expressions.
    root_group_id: UUID | None = None
    root_expressions: tuple[AssertionGroupMember, ...] = Field(default=(), max_length=32)
    groups: tuple[AssertionGroup, ...] = Field(default=(), max_length=32)
    source_reference: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_group_ids_and_root(self) -> AssertionComposition:
        has_legacy_root = self.root_group_id is not None
        has_root_expressions = bool(self.root_expressions)
        if has_legacy_root == has_root_expressions:
            raise ValueError("provide exactly one of root_group_id or root_expressions")
        if has_root_expressions:
            if tuple(item.position for item in self.root_expressions) != tuple(
                range(len(self.root_expressions))
            ):
                raise ValueError("root expression positions must be contiguous from zero")
            refs = [
                (item.member_type, item.proposition_id or item.group_id)
                for item in self.root_expressions
            ]
            if len(refs) != len(set(refs)):
                raise ValueError("root expressions cannot contain duplicate references")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("composition cannot contain duplicate group ids")
        if has_legacy_root and self.root_group_id not in set(group_ids):
            raise ValueError("root_group_id must reference a group in the composition")
        return self


class AssertionCompositionValidationError(ValueError):
    """Stable, non-content-bearing validation failure for offline callers."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass
class CompositionSourceSpanTrace:
    """Bounded, content-bearing trace for shadow-only composition diagnostics."""

    max_records: int = 256
    checks: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    omitted_check_count: int = 0
    recording_error_count: int = 0

    def __post_init__(self) -> None:
        if self.max_records < 1:
            raise ValueError("source-span trace limit must be positive")

    def record(self, item: dict[str, Any]) -> None:
        if len(self.checks) < self.max_records:
            self.checks.append(item)
            return
        self.truncated = True
        self.omitted_check_count += 1
        # Keep a failure even if it occurs after the normal diagnostic cap.
        if item.get("compatibility_result") == "FAIL":
            self.checks[-1] = item
            self.omitted_check_count += 1

    def try_record(self, item: dict[str, Any]) -> None:
        """Keep diagnostics best-effort so trace failures cannot change validation."""
        try:
            self.record(item)
        except Exception:
            self.recording_error_count += 1

    def safe_dict(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "truncated": self.truncated,
            "omitted_check_count": self.omitted_check_count,
            "recording_error_count": self.recording_error_count,
        }


def _effective_eligibility(
    inherited: VerificationEligibility, operator: AssertionGroupOperator
) -> VerificationEligibility:
    if inherited is VerificationEligibility.UNKNOWN or operator is AssertionGroupOperator.UNKNOWN:
        return VerificationEligibility.UNKNOWN
    if operator in {AssertionGroupOperator.OR, AssertionGroupOperator.XOR}:
        return VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED
    return inherited


def validate_assertion_composition(
    composition: AssertionComposition,
    *,
    proposition_ids: set[UUID] | frozenset[UUID],
    claim_text: str,
    proposition_source_spans: dict[UUID, str] | None = None,
    source_span_trace: CompositionSourceSpanTrace | None = None,
    reference_labels: dict[UUID, tuple[str, str]] | None = None,
) -> None:
    """Validate graph references, truth eligibility, and source-span provenance.

    Raises ``AssertionCompositionValidationError`` with a stable code. Callers must
    fail closed on every code; this function does not rewrite or repair the graph.
    """
    groups = {group.group_id: group for group in composition.groups}
    if composition.root_group_id is not None and composition.root_group_id not in groups:
        raise AssertionCompositionValidationError("unknown_root_group")

    visited: set[UUID] = set()
    active: set[UUID] = set()
    seen_propositions: set[UUID] = set()

    def label(reference: UUID | None, fallback_kind: str) -> tuple[str | None, str | None]:
        if reference is None:
            return None, None
        mapped = (reference_labels or {}).get(reference)
        return mapped if mapped is not None else (str(reference), fallback_kind)

    def record_check(
        *,
        parent_ref: str | None,
        parent_kind: str | None,
        parent_span: str | None,
        trace_parent_span: str | None = None,
        child_ref: str,
        child_kind: str,
        child_span: str | None,
        resolved_proposition_id: str | None = None,
        resolved_proposition_span: str | None = None,
        comparison_rule: str,
        compatible: bool | None,
        reason: str | None,
        field_path: str,
        graph_path: tuple[str, ...],
    ) -> None:
        if source_span_trace is None:
            return
        recorded_parent_span = trace_parent_span or parent_span
        source_for_claim_check = child_span is None or child_span in claim_text
        parent_contains_child = (
            recorded_parent_span is None
            or child_span is None
            or child_span in recorded_parent_span
        )
        result = "NOT_CHECKED" if compatible is None else "PASS" if compatible else "FAIL"
        source_result = (
            "NOT_CHECKED"
            if child_span is None
            else "PASS"
            if source_for_claim_check
            else "FAIL"
        )
        parent_result = (
            "NOT_CHECKED"
            if recorded_parent_span is None or child_span is None
            else "PASS"
            if parent_contains_child
            else "FAIL"
        )
        source_span_trace.try_record(
            {
                "parent_ref": parent_ref,
                "parent_kind": parent_kind,
                "parent_source_span": recorded_parent_span,
                "child_ref": child_ref,
                "child_kind": child_kind,
                "child_source_span": child_span,
                "resolved_proposition_id": resolved_proposition_id,
                "resolved_proposition_source_span": resolved_proposition_span,
                "comparison_rule": comparison_rule,
                "claim_compatibility_result": source_result,
                "parent_compatibility_result": parent_result,
                "compatibility_result": result,
                "validation_reason_code": reason,
                "field_path": field_path,
                "graph_path": list(graph_path),
            }
        )

    def check_span(
        span: str | None,
        parent_span: str | None,
        *,
        parent_ref: str | None,
        parent_kind: str | None,
        child_ref: str,
        child_kind: str,
        resolved_proposition_id: str | None = None,
        field_path: str,
        graph_path: tuple[str, ...],
    ) -> None:
        code = None
        if span is not None and span not in claim_text:
            code = "source_span_not_in_claim"
        elif span is not None and parent_span is not None and span not in parent_span:
            code = "nested_source_span_mismatch"
        record_check(
            parent_ref=parent_ref,
            parent_kind=parent_kind,
            parent_span=parent_span,
            trace_parent_span=claim_text if parent_kind == "claim" else parent_span,
            child_ref=child_ref,
            child_kind=child_kind,
            child_span=span,
            resolved_proposition_id=resolved_proposition_id,
            comparison_rule="LITERAL_CLAIM_SUBSTRING_AND_PARENT_CONTAINS_CHILD",
            compatible=None if span is None else code is None,
            reason=code or ("optional_span_missing" if span is None else None),
            field_path=field_path,
            graph_path=graph_path,
        )
        if code is not None:
            raise AssertionCompositionValidationError(code)

    def check_proposition_span(
        proposition_id: UUID,
        member_span: str | None,
        *,
        member_ref: str,
        parent_ref: str | None,
        parent_kind: str | None,
        parent_span: str | None,
        graph_path: tuple[str, ...],
    ) -> None:
        proposition_span = (proposition_source_spans or {}).get(proposition_id)
        compatible = member_span is None or proposition_span is None or (
            member_span in proposition_span or proposition_span in member_span
        )
        code = (
            "proposition_source_span_mismatch"
            if not compatible
            else None
        )
        record_check(
            parent_ref=parent_ref,
            parent_kind=parent_kind,
            parent_span=parent_span,
            trace_parent_span=claim_text if parent_kind == "claim" else parent_span,
            child_ref=member_ref,
            child_kind="proposition",
            child_span=member_span,
            resolved_proposition_id=str(proposition_id),
            resolved_proposition_span=proposition_span,
            comparison_rule="MEMBER_AND_RESOLVED_PROPOSITION_SPANS_CONTAINMENT_COMPATIBLE",
            compatible=None if member_span is None or proposition_span is None else compatible,
            reason=code
            or (
                "optional_span_missing"
                if member_span is None or proposition_span is None
                else None
            ),
            field_path="assertion_composition.member.source_span",
            graph_path=graph_path,
        )
        if member_span is None:
            return
        if code is not None:
            raise AssertionCompositionValidationError(code)

    def visit(
        group_id: UUID,
        inherited: VerificationEligibility,
        parent_span: str | None,
        *,
        parent_ref: str | None,
        parent_kind: str | None,
        graph_path: tuple[str, ...],
    ) -> None:
        if group_id in active:
            raise AssertionCompositionValidationError("composition_cycle")
        group = groups.get(group_id)
        if group is None:
            raise AssertionCompositionValidationError("unknown_group_reference")
        if group_id in visited:
            raise AssertionCompositionValidationError("group_referenced_more_than_once")
        active.add(group_id)
        effective = _effective_eligibility(inherited, group.operator)
        group_ref, _ = label(group_id, "group")
        current_path = (*graph_path, group_ref or str(group_id))
        check_span(
            group.source_span,
            parent_span,
            parent_ref=parent_ref,
            parent_kind=parent_kind,
            child_ref=group_ref or str(group_id),
            child_kind="group",
            field_path="assertion_composition.group.source_span",
            graph_path=current_path,
        )

        for member in group.members:
            member_id = member.group_id if member.member_type == "group" else member.proposition_id
            member_ref, _ = label(member_id, member.member_type)
            member_ref = member_ref or str(member_id)
            member_path = (*current_path, member_ref)
            check_span(
                member.source_span,
                group.source_span,
                parent_ref=group_ref,
                parent_kind="group",
                child_ref=member_ref,
                child_kind=member.member_type,
                field_path="assertion_composition.member.source_span",
                graph_path=member_path,
            )
            if member.member_type == "group":
                assert member.group_id is not None
                visit(
                    member.group_id,
                    effective,
                    group.source_span,
                    parent_ref=group_ref,
                    parent_kind="group",
                    graph_path=current_path,
                )
                continue

            assert member.proposition_id is not None
            if member.proposition_id not in proposition_ids:
                raise AssertionCompositionValidationError("unknown_proposition_reference")
            check_span(
                (proposition_source_spans or {}).get(member.proposition_id),
                group.source_span,
                parent_ref=group_ref,
                parent_kind="group",
                child_ref=member_ref,
                child_kind="proposition",
                resolved_proposition_id=str(member.proposition_id),
                field_path="proposition.decomposition.source_span",
                graph_path=member_path,
            )
            check_proposition_span(
                member.proposition_id,
                member.source_span,
                member_ref=member_ref,
                parent_ref=group_ref,
                parent_kind="group",
                parent_span=group.source_span,
                graph_path=member_path,
            )
            if member.proposition_id in seen_propositions:
                raise AssertionCompositionValidationError("duplicate_proposition_reference")
            seen_propositions.add(member.proposition_id)
            if member.verification_eligibility is not effective:
                raise AssertionCompositionValidationError("verification_eligibility_mismatch")

        active.remove(group_id)
        visited.add(group_id)

    roots: tuple[AssertionGroupMember, ...]
    if composition.root_group_id is not None:
        roots = (
            AssertionGroupMember(
                member_type="group", position=0, group_id=composition.root_group_id
            ),
        )
    else:
        roots = composition.root_expressions

    for root_index, root in enumerate(roots):
        root_id = root.group_id if root.member_type == "group" else root.proposition_id
        root_ref, _ = label(root_id, root.member_type)
        root_ref = root_ref or str(root_id)
        root_path = (f"root[{root_index}]", root_ref)
        check_span(
            root.source_span,
            None,
            parent_ref="claim",
            parent_kind="claim",
            child_ref=root_ref,
            child_kind=root.member_type,
            field_path="assertion_composition.root.source_span",
            graph_path=root_path,
        )
        if root.member_type == "group":
            assert root.group_id is not None
            visit(
                root.group_id,
                VerificationEligibility.INDEPENDENTLY_ASSERTED,
                root.source_span,
                parent_ref="claim",
                parent_kind="claim",
                graph_path=(f"root[{root_index}]",),
            )
            continue
        assert root.proposition_id is not None
        if root.proposition_id not in proposition_ids:
            raise AssertionCompositionValidationError("unknown_proposition_reference")
        check_span(
            (proposition_source_spans or {}).get(root.proposition_id),
            None,
            parent_ref="claim",
            parent_kind="claim",
            child_ref=root_ref,
            child_kind="proposition",
            resolved_proposition_id=str(root.proposition_id),
            field_path="proposition.decomposition.source_span",
            graph_path=root_path,
        )
        check_proposition_span(
            root.proposition_id,
            root.source_span,
            member_ref=root_ref,
            parent_ref="claim",
            parent_kind="claim",
            parent_span=None,
            graph_path=root_path,
        )
        if root.proposition_id in seen_propositions:
            raise AssertionCompositionValidationError("duplicate_proposition_reference")
        seen_propositions.add(root.proposition_id)
        if root.verification_eligibility is not VerificationEligibility.INDEPENDENTLY_ASSERTED:
            raise AssertionCompositionValidationError("verification_eligibility_mismatch")
    if visited != set(groups):
        raise AssertionCompositionValidationError("unreachable_group")
    if seen_propositions != set(proposition_ids):
        raise AssertionCompositionValidationError("unreachable_proposition")


def proposition_verification_eligibility(
    composition: AssertionComposition, proposition_id: UUID
) -> VerificationEligibility:
    """Return the explicitly declared eligibility for a referenced proposition.

    The caller must first validate the composition against the proposition inventory
    and source Claim. Unreferenced propositions are not made eligible by this layer.
    """
    groups = {group.group_id: group for group in composition.groups}

    def find(group_id: UUID) -> VerificationEligibility | None:
        group = groups[group_id]
        for member in group.members:
            if member.member_type == "proposition" and member.proposition_id == proposition_id:
                return member.verification_eligibility
            if member.member_type == "group" and member.group_id in groups:
                nested = find(member.group_id)
                if nested is not None:
                    return nested
        return None

    roots = (
        (composition.root_group_id,)
        if composition.root_group_id is not None
        else tuple(
            member.group_id
            for member in composition.root_expressions
            if member.member_type == "group" and member.group_id is not None
        )
    )
    direct_root = next(
        (
            member.verification_eligibility
            for member in composition.root_expressions
            if member.member_type == "proposition" and member.proposition_id == proposition_id
        ),
        None,
    )
    if direct_root is not None:
        return direct_root
    eligibility = next((found for root_id in roots if (found := find(root_id)) is not None), None)
    if eligibility is not None:
        return eligibility
    raise AssertionCompositionValidationError("proposition_not_in_composition")
