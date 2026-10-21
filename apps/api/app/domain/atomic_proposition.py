"""Passive V2.1R Atomic Proposition contract.

This module is intentionally not imported by the ResearchLoop or any runtime
service. It defines validation contracts and conservative structural guards
for architecture tests; it does not decompose text or perform semantic matching.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

ATOMIC_PROPOSITION_SCHEMA_VERSION: Literal["atomic-proposition-schema-v2"] = (
    "atomic-proposition-schema-v2"
)


class PropositionValueKind(StrEnum):
    TEXT = "text"
    ENTITY = "entity"
    BOOLEAN = "boolean"
    QUANTITY = "quantity"
    SET = "set"
    RANGE = "range"
    COMPARISON = "comparison"
    TEMPORAL_CHANGE = "temporal_change"
    UNKNOWN = "unknown"


class NumericOperator(StrEnum):
    EXACT = "exact"
    APPROX = "approx"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    RANGE = "range"


class ComparisonMetricStatus(StrEnum):
    EXPLICIT = "explicit"
    CONTEXT_RESOLVED = "context_resolved"
    UNSPECIFIED = "unspecified"


class NumericMeasurementBasis(StrEnum):
    OBSERVED = "observed"
    REPORTED = "reported"
    ESTIMATED = "estimated"
    ATTAINABLE = "attainable"


class NumericExtremum(StrEnum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    PEAK = "peak"
    BEST = "best"


class NumericChangeType(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    RATIO = "ratio"


class PropositionPolarity(StrEnum):
    ASSERTED = "asserted"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class QualifierCategory(StrEnum):
    TIME = "time"
    DATE = "date"
    PERIOD = "period"
    VERSION = "version"
    LOCATION = "location"
    SCOPE = "scope"
    POPULATION = "population"
    DEFINITION = "definition"
    CONFIGURATION = "configuration"
    METHOD = "method"
    CONDITION = "condition"
    SCENARIO = "scenario"
    MEASUREMENT = "measurement"
    UNIT = "unit"
    CURRENCY = "currency"
    BASELINE = "baseline"
    COMPARISON_TARGET = "comparison_target"
    MODALITY = "modality"
    ATTRIBUTION = "attribution"
    OTHER = "other"


class QualifierMateriality(StrEnum):
    MATERIAL = "material"
    DESCRIPTIVE = "descriptive"
    UNKNOWN = "unknown"


class DecompositionStatus(StrEnum):
    SINGLE_PROPOSITION = "single_proposition"
    DECOMPOSED = "decomposed"
    UNRESOLVED_DECOMPOSITION = "unresolved_decomposition"


class DecompositionMethod(StrEnum):
    HUMAN = "human"
    STRUCTURED_MODEL = "structured_model"
    HYBRID = "hybrid"
    UNSPECIFIED = "unspecified"


class PropositionRelation(StrEnum):
    SAME = "same"
    RELATED = "related"
    CONTRADICTS = "contradicts"
    UNKNOWN = "unknown"


class EvidencePropositionRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MENTIONS = "mentions"
    INSUFFICIENT = "insufficient"


class GuardResult(StrEnum):
    SAME = "same"
    NOT_SAME = "not_same"
    UNKNOWN = "unknown"


class ContradictionComparability(StrEnum):
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"
    UNKNOWN = "unknown"


class NumericValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operator: NumericOperator
    value: Decimal | None = None
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    currency: str | None = Field(default=None, min_length=1, max_length=20)
    tolerance: Decimal | None = None
    measurement_basis: NumericMeasurementBasis | None = None
    extremum: NumericExtremum | None = None
    change_type: NumericChangeType | None = None

    @field_validator("value", "lower_bound", "upper_bound", "tolerance", mode="before")
    @classmethod
    def require_json_numeric_values(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float | Decimal)
        ):
            raise ValueError("numeric values must be JSON numbers")
        return value

    @model_validator(mode="after")
    def validate_operator_shape(self) -> NumericValue:
        if self.operator in {
            NumericOperator.EXACT,
            NumericOperator.APPROX,
            NumericOperator.LT,
            NumericOperator.LTE,
            NumericOperator.GT,
            NumericOperator.GTE,
        } and (self.value is None or self.lower_bound is not None or self.upper_bound is not None):
            raise ValueError("scalar numeric operators require value only")
        if self.operator == NumericOperator.RANGE and (
            self.value is not None
            or self.lower_bound is None
            or self.upper_bound is None
            or self.lower_bound > self.upper_bound
        ):
            raise ValueError("range requires ordered lower_bound and upper_bound")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("numeric tolerance cannot be negative")
        return self


class PropositionValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PropositionValueKind
    text: str | None = Field(default=None, max_length=2_000)
    boolean: StrictBool | None = None
    numeric: NumericValue | None = None
    members: tuple[str, ...] = ()
    comparison_target: str | None = Field(default=None, max_length=1_000)
    comparison_metric: str | None = Field(default=None, max_length=500)
    comparison_metric_status: ComparisonMetricStatus | None = None
    temporal_from: str | None = Field(default=None, max_length=200)
    temporal_to: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_value_shape(self) -> PropositionValue:
        allowed_fields = {
            PropositionValueKind.TEXT: {"text"},
            PropositionValueKind.ENTITY: {"text"},
            PropositionValueKind.BOOLEAN: {"boolean"},
            PropositionValueKind.QUANTITY: {"numeric"},
            PropositionValueKind.RANGE: {"numeric"},
            PropositionValueKind.SET: {"members"},
            PropositionValueKind.COMPARISON: {
                "comparison_target",
                "comparison_metric",
                "comparison_metric_status",
                "numeric",
            },
            PropositionValueKind.TEMPORAL_CHANGE: {
                "temporal_from",
                "temporal_to",
                "numeric",
            },
            PropositionValueKind.UNKNOWN: set(),
        }[self.kind]
        incompatible_fields = (self.model_fields_set - {"kind"}) - allowed_fields
        if incompatible_fields:
            raise ValueError("typed value contains fields incompatible with its kind")
        if self.kind in {PropositionValueKind.TEXT, PropositionValueKind.ENTITY}:
            if not self.text:
                raise ValueError("text/entity values require non-empty text")
        elif self.kind == PropositionValueKind.BOOLEAN:
            if self.boolean is None:
                raise ValueError("boolean value must be explicit, including false")
        elif self.kind in {PropositionValueKind.QUANTITY, PropositionValueKind.RANGE}:
            if self.numeric is None:
                raise ValueError("numeric and range values require numeric structure")
        elif self.kind == PropositionValueKind.SET:
            if not self.members or any(not member.strip() for member in self.members):
                raise ValueError("set values require non-empty members")
        elif self.kind == PropositionValueKind.COMPARISON:
            if not self.comparison_target or self.comparison_metric_status is None:
                raise ValueError("comparison requires a target and metric resolution status")
            if self.comparison_metric_status is ComparisonMetricStatus.UNSPECIFIED:
                if self.comparison_metric is not None:
                    raise ValueError("unspecified comparison metric cannot contain a metric")
            elif not self.comparison_metric:
                raise ValueError("resolved comparison metric status requires a metric")
        elif self.kind == PropositionValueKind.TEMPORAL_CHANGE and (
            not self.temporal_from or not self.temporal_to or self.numeric is None
        ):
            raise ValueError("temporal change requires from/to and a structured value")
        return self


class MaterialQualifier(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: QualifierCategory
    key: str = Field(min_length=1, max_length=100)
    value: str | int | Decimal | bool
    materiality: QualifierMateriality
    source_span: str | None = Field(default=None, max_length=2_000)


class DecompositionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DecompositionStatus
    method: DecompositionMethod = DecompositionMethod.UNSPECIFIED
    schema_version: str = ATOMIC_PROPOSITION_SCHEMA_VERSION
    prompt_version: str | None = None
    proposition_index: int | None = Field(default=None, ge=0)
    source_span: str | None = Field(default=None, max_length=2_000)
    reason: str | None = Field(default=None, max_length=1_000)
    confidence: Decimal | None = Field(default=None, ge=0, le=1)


class AtomicProposition(BaseModel):
    """One independently verifiable assertion with material context preserved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_id: UUID
    schema_version: Literal[
        "atomic-proposition-schema-v2", "atomic-proposition-schema-v3"
    ] = ATOMIC_PROPOSITION_SCHEMA_VERSION
    original_text: str = Field(min_length=1, max_length=4_000)
    canonical_text: str | None = Field(default=None, min_length=1, max_length=4_000)
    subject: str | None = Field(default=None, min_length=1, max_length=1_000)
    predicate: str | None = Field(default=None, min_length=1, max_length=1_000)
    value: PropositionValue
    polarity: PropositionPolarity
    qualifiers: tuple[MaterialQualifier, ...] = ()
    modality: str | None = Field(default=None, max_length=100)
    attribution: str | None = Field(default=None, max_length=1_000)
    decomposition: DecompositionMetadata

    @model_validator(mode="after")
    def validate_resolution(self) -> AtomicProposition:
        unresolved = self.decomposition.status == DecompositionStatus.UNRESOLVED_DECOMPOSITION
        if not unresolved and (self.subject is None or self.predicate is None):
            raise ValueError("resolved propositions require subject and predicate")
        if unresolved and self.decomposition.reason is None:
            raise ValueError("unresolved decomposition requires a reviewable reason")
        if self.decomposition.schema_version != self.schema_version:
            raise ValueError("decomposition and proposition schema versions must match")
        return self


class ClaimPropositionLink(BaseModel):
    """Assertion that a Claim expresses a proposition; not an identity edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: UUID
    proposition_id: UUID
    relation: Literal["expresses"] = "expresses"
    proposition_index: int = Field(ge=0)
    claim_span: str | None = Field(default=None, max_length=2_000)
    decomposition_status: DecompositionStatus


class PropositionEvidenceLink(BaseModel):
    """Source-specific assertion provenance for one proposition/evidence pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_id: UUID
    evidence_id: UUID
    claim_id: UUID
    relation: EvidencePropositionRelation
    source_id: UUID
    source_owner_key: str | None = Field(default=None, min_length=1, max_length=255)
    quote_reference: str | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    method: str = Field(min_length=1, max_length=100)
    confidence: Decimal | None = Field(default=None, ge=0, le=1)
    run_id: UUID
    question_id: str = Field(min_length=1, max_length=50)
    dimension_key: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_span(self) -> PropositionEvidenceLink:
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ValueError("evidence span end must not precede start")
        return self


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip().casefold()


def _material_qualifiers(
    proposition: AtomicProposition,
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for qualifier in proposition.qualifiers:
        if qualifier.materiality == QualifierMateriality.MATERIAL:
            result[(qualifier.category.value, _norm(qualifier.key))] = _norm(str(qualifier.value))
    return result


def exact_structured_identity_guard(
    left: AtomicProposition, right: AtomicProposition
) -> GuardResult:
    """Return SAME only for exact structured agreement; never infer semantics."""
    if (
        left.decomposition.status == DecompositionStatus.UNRESOLVED_DECOMPOSITION
        or right.decomposition.status == DecompositionStatus.UNRESOLVED_DECOMPOSITION
        or left.subject is None
        or right.subject is None
        or left.predicate is None
        or right.predicate is None
        or left.polarity == PropositionPolarity.UNKNOWN
        or right.polarity == PropositionPolarity.UNKNOWN
        or any(
            q.materiality == QualifierMateriality.UNKNOWN
            for q in (*left.qualifiers, *right.qualifiers)
        )
    ):
        return GuardResult.UNKNOWN
    if any(
        proposition.value.kind is PropositionValueKind.COMPARISON
        and proposition.value.comparison_metric_status is ComparisonMetricStatus.UNSPECIFIED
        for proposition in (left, right)
    ):
        return GuardResult.UNKNOWN
    if (
        _norm(left.subject) != _norm(right.subject)
        or _norm(left.predicate) != _norm(right.predicate)
        or left.polarity != right.polarity
        or left.value.model_dump(mode="json") != right.value.model_dump(mode="json")
        or _material_qualifiers(left) != _material_qualifiers(right)
        or _norm(left.modality or "") != _norm(right.modality or "")
        or _norm(left.attribution or "") != _norm(right.attribution or "")
    ):
        return GuardResult.NOT_SAME
    return GuardResult.SAME


def contradiction_comparability_guard(
    left: AtomicProposition, right: AtomicProposition
) -> ContradictionComparability:
    """Check whether a pair is in-scope for contradiction analysis, not its truth."""
    if (
        left.decomposition.status == DecompositionStatus.UNRESOLVED_DECOMPOSITION
        or right.decomposition.status == DecompositionStatus.UNRESOLVED_DECOMPOSITION
        or left.subject is None
        or right.subject is None
        or left.predicate is None
        or right.predicate is None
        or left.polarity == PropositionPolarity.UNKNOWN
        or right.polarity == PropositionPolarity.UNKNOWN
        or any(
            q.materiality == QualifierMateriality.UNKNOWN
            for q in (*left.qualifiers, *right.qualifiers)
        )
    ):
        return ContradictionComparability.UNKNOWN
    if any(
        proposition.value.kind is PropositionValueKind.COMPARISON
        and proposition.value.comparison_metric_status is ComparisonMetricStatus.UNSPECIFIED
        for proposition in (left, right)
    ):
        return ContradictionComparability.UNKNOWN
    if _norm(left.subject) != _norm(right.subject) or _norm(left.predicate) != _norm(
        right.predicate
    ):
        return ContradictionComparability.INCOMPARABLE
    if _material_qualifiers(left) != _material_qualifiers(right):
        return ContradictionComparability.INCOMPARABLE
    if _norm(left.modality or "") != _norm(right.modality or ""):
        return ContradictionComparability.INCOMPARABLE
    if _norm(left.attribution or "") != _norm(right.attribution or ""):
        return ContradictionComparability.INCOMPARABLE
    if left.value.kind != right.value.kind:
        return ContradictionComparability.INCOMPARABLE
    if left.value.kind in {PropositionValueKind.QUANTITY, PropositionValueKind.RANGE}:
        if left.value.numeric is None or right.value.numeric is None:
            return ContradictionComparability.UNKNOWN
        if (
            left.value.numeric.unit != right.value.numeric.unit
            or left.value.numeric.currency != right.value.numeric.currency
        ):
            return ContradictionComparability.INCOMPARABLE
    return ContradictionComparability.COMPARABLE


def complete_link_same_cluster(
    proposition_ids: tuple[UUID, ...], same_pairs: frozenset[frozenset[UUID]]
) -> bool:
    """Guard against transitive SAME: every pair must be directly SAME."""
    if len(set(proposition_ids)) != len(proposition_ids):
        return False
    return all(
        frozenset((left, right)) in same_pairs
        for index, left in enumerate(proposition_ids)
        for right in proposition_ids[index + 1 :]
    )
