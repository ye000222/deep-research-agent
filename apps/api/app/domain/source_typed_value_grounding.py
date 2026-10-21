"""Versioned, fail-closed grounding for a bounded typed numeric value profile.

The parser consumes only explicitly bound SourceEvidenceSpan text. It never
searches a document and never performs unit or currency conversion.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.atomic_proposition import (
    NumericOperator,
    NumericValue,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.source_atomic_assertion import SourceGroundingState

SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION: Literal["source-typed-value-grounding-v1"] = (
    "source-typed-value-grounding-v1"
)
SOURCE_TYPED_VALUE_PARSER_VERSION: Literal["numeric-span-parser-ascii-v1"] = (
    "numeric-span-parser-ascii-v1"
)


class TypedValueComponentRole(StrEnum):
    NUMERIC_MAGNITUDE = "NUMERIC_MAGNITUDE"
    NUMERIC_OPERATOR = "NUMERIC_OPERATOR"
    LOWER_BOUND = "LOWER_BOUND"
    UPPER_BOUND = "UPPER_BOUND"
    UNIT = "UNIT"
    CURRENCY = "CURRENCY"


class TypedValueEvidenceBinding(BaseModel):
    """Provider-proposed value-component binding; never a grounding verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    value_field_role: Literal["VALUE"] = "VALUE"
    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    occurrence_index: int = Field(ge=0, le=1_000)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    component_role: TypedValueComponentRole
    source_ref: str = Field(min_length=1, max_length=2_000)
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_version: Literal["source-typed-value-grounding-v1"] = (
        SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION
    )

    @model_validator(mode="after")
    def nonempty_span(self) -> TypedValueEvidenceBinding:
        if self.char_end <= self.char_start:
            raise ValueError("typed_value_span_offsets_must_be_nonempty")
        return self


class TypedValueComponentAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_role: TypedValueComponentRole
    evidence_refs: tuple[str, ...] = ()
    textual_grounding: SourceGroundingState
    semantic_support: SourceGroundingState
    structural_grounding: SourceGroundingState
    normalized_component: str | None = None
    reason_code: str


class SourceTypedValueGroundingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assertion_ref: str = Field(min_length=1, max_length=128)
    contract_version: Literal["source-typed-value-grounding-v1"] = (
        SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION
    )
    parser_version: Literal["numeric-span-parser-ascii-v1"] = SOURCE_TYPED_VALUE_PARSER_VERSION
    status: SourceGroundingState
    components: tuple[TypedValueComponentAssessment, ...] = ()
    normalized_value: PropositionValue | None = None
    reason_codes: tuple[str, ...] = ()


class SourceEvidenceSpanInput(BaseModel):
    """Minimal span view needed by this pure domain validator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    exact_text: str = Field(min_length=1, max_length=4_000)
    occurrence_index: int = Field(default=0, ge=0, le=1_000)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_SUFFIX = r"(?:(?P<percent_unit>%)(?=\s|$)|\s+(?P<unit>[A-Za-z][A-Za-z0-9./]*))?"
_SCALAR = re.compile(rf"^\s*(?P<op>>=|<=|>|<|≈|~)?\s*(?P<number>{_NUMBER}){_SUFFIX}\s*$")
_RANGE = re.compile(rf"^\s*(?P<lower>{_NUMBER})\s*[\u2013]\s*(?P<upper>{_NUMBER}){_SUFFIX}\s*$")
_AMBIGUOUS_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _decimal(token: str) -> Decimal:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("NUMERIC_TOKEN_INVALID") from exc


def parse_numeric_span(exact_text: str) -> NumericValue | None:
    """Parse one entire explicitly designated span under the v1 ASCII grammar.

    Accepted syntax: ASCII integer/decimal with optional valid thousands
    grouping; scalar operators >=, <=, >, <, ~, ≈; en-dash ranges; optional
    contiguous percent or one-token unit/currency suffix. No conversion occurs.
    """

    text = unicodedata.normalize("NFC", exact_text)
    range_match = _RANGE.fullmatch(text)
    if range_match is not None:
        suffix = range_match.group("percent_unit") or range_match.group("unit")
        if suffix is not None and (
            suffix.casefold() == "percent" or _AMBIGUOUS_CURRENCY.fullmatch(suffix)
        ):
            return None
        unit, currency = _suffix_fields(suffix)
        return NumericValue(
            operator=NumericOperator.RANGE,
            lower_bound=_decimal(range_match.group("lower")),
            upper_bound=_decimal(range_match.group("upper")),
            unit=unit,
            currency=currency,
        )

    scalar_match = _SCALAR.fullmatch(text)
    if scalar_match is None:
        return None
    op = scalar_match.group("op")
    operator = {
        None: NumericOperator.EXACT,
        ">": NumericOperator.GT,
        ">=": NumericOperator.GTE,
        "<": NumericOperator.LT,
        "<=": NumericOperator.LTE,
        "~": NumericOperator.APPROX,
        "≈": NumericOperator.APPROX,
    }[op]
    suffix = scalar_match.group("percent_unit") or scalar_match.group("unit")
    if suffix is not None and (
        suffix.casefold() == "percent" or _AMBIGUOUS_CURRENCY.fullmatch(suffix)
    ):
        return None
    unit, currency = _suffix_fields(suffix)
    return NumericValue(
        operator=operator,
        value=_decimal(scalar_match.group("number")),
        unit=unit,
        currency=currency,
    )


def _suffix_fields(suffix: str | None) -> tuple[str | None, str | None]:
    if suffix is None:
        return None, None
    if suffix == "%":
        return "%", None
    return suffix, None


def _required_roles(value: PropositionValue) -> tuple[TypedValueComponentRole, ...] | None:
    if value.kind not in {PropositionValueKind.QUANTITY, PropositionValueKind.RANGE}:
        return None
    numeric = value.numeric
    if numeric is None:
        return None
    if value.kind is PropositionValueKind.RANGE:
        if numeric.operator is not NumericOperator.RANGE:
            return None
        roles = [TypedValueComponentRole.LOWER_BOUND, TypedValueComponentRole.UPPER_BOUND]
    else:
        if numeric.operator is NumericOperator.RANGE:
            return None
        roles = [TypedValueComponentRole.NUMERIC_MAGNITUDE]
        if numeric.operator is not NumericOperator.EXACT:
            roles.append(TypedValueComponentRole.NUMERIC_OPERATOR)
    if numeric.unit is not None:
        roles.append(TypedValueComponentRole.UNIT)
    if numeric.currency is not None:
        roles.append(TypedValueComponentRole.CURRENCY)
    return tuple(roles)


def assess_typed_value_grounding(
    *,
    assertion_ref: str,
    value: PropositionValue,
    bindings: tuple[TypedValueEvidenceBinding, ...],
    spans: tuple[SourceEvidenceSpanInput, ...],
    source_ref: str,
    snapshot_ref: str,
) -> SourceTypedValueGroundingAssessment:
    """Validate bindings, exact-span parses, typed equality and completeness."""

    required = _required_roles(value)
    if required is None:
        return _unknown_assessment(assertion_ref, "UNSUPPORTED_VALUE_KIND")
    span_by_ref = {span.evidence_ref: span for span in spans}
    if len(span_by_ref) != len(spans):
        return _invalid_assessment(assertion_ref, "DUPLICATE_EVIDENCE_SPAN_REF")
    relevant = bindings
    if any(
        binding.assertion_ref != assertion_ref
        or binding.value_field_role != "VALUE"
        or binding.source_ref != source_ref
        or binding.snapshot_ref != snapshot_ref
        for binding in relevant
    ):
        return _invalid_assessment(assertion_ref, "VALUE_BINDING_PROVENANCE_MISMATCH")
    roles = [item.component_role for item in relevant]
    if len(roles) != len(set(roles)):
        return _invalid_assessment(assertion_ref, "DUPLICATE_COMPONENT_BINDING")
    if set(roles) - set(required):
        return _invalid_assessment(assertion_ref, "UNEXPECTED_COMPONENT_BINDING")
    by_role = {item.component_role: item for item in relevant}
    components: list[TypedValueComponentAssessment] = []
    parsed_values: list[NumericValue] = []
    for role in required:
        binding = by_role.get(role)
        if binding is None:
            components.append(_component_unknown(role, "REQUIRED_COMPONENT_BINDING_MISSING"))
            continue
        span = span_by_ref.get(binding.evidence_ref)
        if (
            span is None
            or span.occurrence_index != binding.occurrence_index
            or span.char_start != binding.char_start
            or span.char_end != binding.char_end
        ):
            components.append(
                _component_invalid(role, binding.evidence_ref, "SPAN_REF_OR_OCCURRENCE_INVALID")
            )
            continue
        parsed = parse_numeric_span(span.exact_text)
        if parsed is None:
            components.append(
                _component_unknown(role, "EXACT_SPAN_NOT_IN_V1_GRAMMAR", binding.evidence_ref)
            )
            continue
        parsed_values.append(parsed)
        component = _component_value(parsed, role)
        expected = _expected_component(value.numeric, role)
        # Keep dimensions independent: the exact span can be syntactically
        # readable even when its meaning does not match the typed assertion.
        textual = SourceGroundingState.VALID
        semantic = (
            SourceGroundingState.VALID
            if component is not None and component == expected
            else SourceGroundingState.INVALID
        )
        structural = SourceGroundingState.VALID
        components.append(
            TypedValueComponentAssessment(
                component_role=role,
                evidence_refs=(binding.evidence_ref,),
                textual_grounding=textual,
                semantic_support=semantic,
                structural_grounding=structural,
                normalized_component=str(component),
                reason_code=(
                    "EXACT_SPAN_TYPED_COMPONENT_MATCH"
                    if semantic is SourceGroundingState.VALID
                    else "EXACT_SPAN_COMPONENT_VALUE_MISMATCH"
                ),
            )
        )

    component_states = [
        state
        for item in components
        for state in (item.textual_grounding, item.semantic_support, item.structural_grounding)
    ]
    if SourceGroundingState.INVALID in component_states:
        status = SourceGroundingState.INVALID
        reason = "TYPED_VALUE_COMPONENT_INVALID"
    elif not all(state is SourceGroundingState.VALID for state in component_states):
        status = SourceGroundingState.UNKNOWN
        reason = "TYPED_VALUE_COMPONENT_INCOMPLETE_OR_UNPARSEABLE"
    elif not parsed_values or any(parsed != value.numeric for parsed in parsed_values):
        status = SourceGroundingState.INVALID
        reason = "NORMALIZED_VALUE_MISMATCH"
    else:
        status = SourceGroundingState.VALID
        reason = "TYPED_VALUE_GROUNDED"
    return SourceTypedValueGroundingAssessment(
        assertion_ref=assertion_ref,
        status=status,
        components=tuple(components),
        normalized_value=value if status is SourceGroundingState.VALID else None,
        reason_codes=(reason, SOURCE_TYPED_VALUE_PARSER_VERSION),
    )


def _component_value(numeric: NumericValue, role: TypedValueComponentRole) -> object | None:
    return {
        TypedValueComponentRole.NUMERIC_MAGNITUDE: numeric.value,
        TypedValueComponentRole.NUMERIC_OPERATOR: numeric.operator,
        TypedValueComponentRole.LOWER_BOUND: numeric.lower_bound,
        TypedValueComponentRole.UPPER_BOUND: numeric.upper_bound,
        TypedValueComponentRole.UNIT: numeric.unit,
        TypedValueComponentRole.CURRENCY: numeric.currency,
    }[role]


def _expected_component(
    numeric: NumericValue | None, role: TypedValueComponentRole
) -> object | None:
    return _component_value(numeric, role) if numeric is not None else None


def _unknown_assessment(assertion_ref: str, reason: str) -> SourceTypedValueGroundingAssessment:
    return SourceTypedValueGroundingAssessment(
        assertion_ref=assertion_ref,
        status=SourceGroundingState.UNKNOWN,
        reason_codes=(reason,),
    )


def _invalid_assessment(assertion_ref: str, reason: str) -> SourceTypedValueGroundingAssessment:
    return SourceTypedValueGroundingAssessment(
        assertion_ref=assertion_ref,
        status=SourceGroundingState.INVALID,
        reason_codes=(reason,),
    )


def _component_unknown(
    role: TypedValueComponentRole, reason: str, evidence_ref: str | None = None
) -> TypedValueComponentAssessment:
    return TypedValueComponentAssessment(
        component_role=role,
        evidence_refs=(evidence_ref,) if evidence_ref else (),
        textual_grounding=SourceGroundingState.UNKNOWN,
        semantic_support=SourceGroundingState.UNKNOWN,
        structural_grounding=SourceGroundingState.UNKNOWN,
        reason_code=reason,
    )


def _component_invalid(
    role: TypedValueComponentRole, evidence_ref: str, reason: str
) -> TypedValueComponentAssessment:
    return TypedValueComponentAssessment(
        component_role=role,
        evidence_refs=(evidence_ref,),
        textual_grounding=SourceGroundingState.INVALID,
        semantic_support=SourceGroundingState.INVALID,
        structural_grounding=SourceGroundingState.INVALID,
        reason_code=reason,
    )
