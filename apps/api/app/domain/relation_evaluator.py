"""Pure, deterministic V2.2-A3 proposition-pair relation evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.atomic_proposition import (
    NumericOperator,
    PropositionPolarity,
    PropositionRelation,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.verification_assertion import (
    ComparabilityStatus,
    ComparisonReasonCode,
    PropositionPairAssessment,
    VerificationAssertion,
)

EXPLICIT_SEMANTIC_ASSOCIATION_SCHEMA_VERSION: Literal["explicit-semantic-association-v1"] = (
    "explicit-semantic-association-v1"
)


class ExplicitSemanticAssociation(BaseModel):
    """An upstream-established association bound to one unordered assertion pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_assertion_ref: str = Field(min_length=1, max_length=128)
    right_assertion_ref: str = Field(min_length=1, max_length=128)
    established: bool
    provenance_ref: str = Field(min_length=1, max_length=512)
    schema_version: Literal["explicit-semantic-association-v1"] = (
        EXPLICIT_SEMANTIC_ASSOCIATION_SCHEMA_VERSION
    )
    namespace: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def distinct_pair(self) -> ExplicitSemanticAssociation:
        if self.left_assertion_ref == self.right_assertion_ref:
            raise ValueError("semantic_association_requires_distinct_assertions")
        return self

    def matches(self, left_ref: str, right_ref: str) -> bool:
        return {self.left_assertion_ref, self.right_assertion_ref} == {left_ref, right_ref}


def _numeric_value_equal(left: PropositionValue, right: PropositionValue) -> bool:
    """Compare complete typed numeric representations, never stringified values."""
    if left.kind is not right.kind or left.numeric is None or right.numeric is None:
        return False
    if left.kind not in {PropositionValueKind.QUANTITY, PropositionValueKind.RANGE}:
        return False
    if left.kind is PropositionValueKind.RANGE and (
        left.numeric.operator is not NumericOperator.RANGE
        or right.numeric.operator is not NumericOperator.RANGE
    ):
        return False
    return left.numeric == right.numeric


def _typed_value_equal(left: PropositionValue, right: PropositionValue) -> bool:
    """Return equality only for typed families with explicit semantic equality."""
    if left.kind is not right.kind:
        return False
    if left.kind is PropositionValueKind.BOOLEAN:
        return left.boolean is not None and left.boolean is right.boolean
    return _numeric_value_equal(left, right)


def _direct_boolean_conflict(left: PropositionValue, right: PropositionValue) -> bool:
    return (
        left.kind is PropositionValueKind.BOOLEAN
        and right.kind is PropositionValueKind.BOOLEAN
        and left.boolean is not None
        and right.boolean is not None
        and left.boolean is not right.boolean
    )


def _reason_codes(
    assessment: PropositionPairAssessment,
    relation: PropositionRelation,
    relation_reason: ComparisonReasonCode,
) -> tuple[ComparisonReasonCode, ...]:
    retained: list[ComparisonReasonCode] = [
        code
        for code in assessment.reason_codes
        if code is not ComparisonReasonCode.RELATION_NOT_EVALUATED
    ]
    if relation is PropositionRelation.RELATED:
        retained.append(ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION)
    retained.append(relation_reason)
    return tuple(dict.fromkeys(retained))


class RelationEvaluator:
    """Derive a strict pair relation from assertions and an already-frozen A2 result."""

    @staticmethod
    def evaluate(
        left: VerificationAssertion,
        right: VerificationAssertion,
        comparability: PropositionPairAssessment,
        *,
        association: ExplicitSemanticAssociation | None = None,
    ) -> PropositionPairAssessment:
        if left.assertion_ref == right.assertion_ref:
            raise ValueError("relation_evaluation_requires_distinct_assertions")
        if (
            comparability.left_assertion_ref != left.assertion_ref
            or comparability.right_assertion_ref != right.assertion_ref
        ):
            raise ValueError("comparability_assessment_pair_mismatch")
        if comparability.schema_version != "proposition-comparison-contract-v1":
            raise ValueError("unsupported_comparability_assessment_version")
        if (
            left.schema_version != "verification-assertion-v1"
            or right.schema_version != "verification-assertion-v1"
            or left.normalization_version != right.normalization_version
            or comparability.normalization_version != left.normalization_version
        ):
            raise ValueError("relation_input_version_mismatch")
        if (
            comparability.relation is not PropositionRelation.UNKNOWN
            or comparability.relation_eligible is not None
        ):
            raise ValueError("relation_evaluator_requires_unevaluated_a2_assessment")
        if association is not None and not association.matches(
            left.assertion_ref, right.assertion_ref
        ):
            raise ValueError("semantic_association_pair_mismatch")

        relation = PropositionRelation.UNKNOWN
        relation_eligible = False
        relation_reason = ComparisonReasonCode.RELATION_SEMANTICS_INSUFFICIENT
        comparable = comparability.comparability is ComparabilityStatus.COMPARABLE

        if comparable:
            values_equal = _typed_value_equal(left.typed_value, right.typed_value)
            known_polarity = PropositionPolarity.UNKNOWN not in {
                left.polarity,
                right.polarity,
            }
            same_polarity = known_polarity and left.polarity is right.polarity
            opposite_polarity = known_polarity and left.polarity is not right.polarity

            if values_equal and same_polarity:
                relation = PropositionRelation.SAME
                relation_eligible = True
                relation_reason = ComparisonReasonCode.SAME_TYPED_VALUE
            elif values_equal and opposite_polarity:
                relation = PropositionRelation.CONTRADICTS
                relation_eligible = True
                relation_reason = ComparisonReasonCode.DIRECT_POLARITY_CONFLICT
            elif same_polarity and _direct_boolean_conflict(left.typed_value, right.typed_value):
                relation = PropositionRelation.CONTRADICTS
                relation_eligible = True
                relation_reason = ComparisonReasonCode.DIRECT_BOOLEAN_VALUE_CONFLICT

        if (
            relation is PropositionRelation.UNKNOWN
            and association is not None
            and association.established
        ):
            relation = PropositionRelation.RELATED
            relation_eligible = True
            relation_reason = ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION

        return comparability.model_copy(
            update={
                "relation": relation,
                "relation_eligible": relation_eligible,
                "reason_codes": _reason_codes(comparability, relation, relation_reason),
            }
        )
