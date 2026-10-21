"""Conservative, explainable semantic comparison for V2.1 shadow analysis.

This module is deliberately offline and has no persistence or research-loop
dependencies.  It recognizes a narrow set of deterministic equivalences and
returns UNKNOWN whenever the available structure is insufficient.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.claim_architecture import (
    CanonicalClaim,
    CanonicalClaimType,
    ClaimRelationType,
    exact_proposition_key,
)

MATCHER_VERSION = "claim-equivalence-v2-v1"


class SemanticMatchReason(StrEnum):
    EXACT_NORMALIZED_MATCH = "EXACT_NORMALIZED_MATCH"
    STRUCTURED_FIELDS_MATCH = "STRUCTURED_FIELDS_MATCH"
    SEMANTIC_PARAPHRASE = "SEMANTIC_PARAPHRASE"
    SUBJECT_MISMATCH = "SUBJECT_MISMATCH"
    TIME_MISMATCH = "TIME_MISMATCH"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    POLARITY_CONFLICT = "POLARITY_CONFLICT"
    CLAIM_TYPE_MISMATCH = "CLAIM_TYPE_MISMATCH"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    SEMANTIC_MODEL_UNCERTAIN = "SEMANTIC_MODEL_UNCERTAIN"


class QualifierCompatibility(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    INCOMPLETE = "incomplete"


class SemanticClaimComparison(BaseModel):
    """Auditable comparison result; never a verification/corroboration verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_claim_id: str
    right_claim_id: str
    relation: ClaimRelationType
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: tuple[SemanticMatchReason, ...]
    rationale: str
    signals: dict[str, str | bool | None]
    qualifier_compatibility: QualifierCompatibility
    method: str


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_QUALIFIER_KEY_ALIASES = {
    "year": "time",
    "date": "time",
    "time_period": "time",
    "geo_scope": "location",
    "geography": "location",
    "population": "scope",
}
_VALUE_RE = re.compile(
    r"^\s*(about\s+|approximately\s+|~\s*)?"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(billion|bn|b|million|mn|m|thousand|k)?\s*"
    r"(USD|US\$|EUR|GBP|JPY|CAD|AUD|people|persons|tokens|%|\$)?\s*$",
    flags=re.IGNORECASE,
)
_SCALES = {
    "billion": Decimal(1_000_000_000), "bn": Decimal(1_000_000_000),
    "b": Decimal(1_000_000_000), "million": Decimal(1_000_000),
    "mn": Decimal(1_000_000), "m": Decimal(1_000_000),
    "thousand": Decimal(1_000), "k": Decimal(1_000),
}
_UNIT_ALIASES = {"us$": "usd", "$": "usd", "persons": "people"}


def normalize_safe_text(value: str) -> str:
    """Normalize presentation only; preserve semantic tokens and qualifiers."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = (normalized.replace("\u2010", "-").replace("\u2011", "-")
                  .replace("\u2014", "-").replace("\u2013", "-"))
    return _SPACE_RE.sub(" ", normalized).strip().casefold()


def _token_key(value: str) -> str:
    return _PUNCT_RE.sub("", normalize_safe_text(value))


def _predicate_key(value: str) -> str:
    normalized = normalize_safe_text(value).strip(" .,!?:;")
    return _token_key(normalized)


def _normalized_qualifiers(claim: CanonicalClaim) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in claim.qualifiers.items():
        normalized_key = _QUALIFIER_KEY_ALIASES.get(key.casefold(), key.casefold())
        if value is None:
            continue
        normalized_value = normalize_safe_text(str(value))
        if normalized_key == "unit":
            normalized_value = _UNIT_ALIASES.get(normalized_value, normalized_value)
        result[normalized_key] = normalized_value
    return result


def _qualifier_comparison(
    left: CanonicalClaim, right: CanonicalClaim
) -> tuple[QualifierCompatibility, tuple[SemanticMatchReason, ...]]:
    left_values = _normalized_qualifiers(left)
    right_values = _normalized_qualifiers(right)
    for key in sorted(left_values.keys() & right_values.keys()):
        if left_values[key] != right_values[key]:
            reason = {
                "time": SemanticMatchReason.TIME_MISMATCH,
                "version": SemanticMatchReason.VERSION_MISMATCH,
                "scope": SemanticMatchReason.SCOPE_MISMATCH,
                "location": SemanticMatchReason.SCOPE_MISMATCH,
                "condition": SemanticMatchReason.SCOPE_MISMATCH,
                "population": SemanticMatchReason.SCOPE_MISMATCH,
                "unit": SemanticMatchReason.UNIT_MISMATCH,
                "methodology": SemanticMatchReason.SCOPE_MISMATCH,
            }.get(key, SemanticMatchReason.INSUFFICIENT_INFORMATION)
            return QualifierCompatibility.INCOMPATIBLE, (reason,)
    if left_values.keys() != right_values.keys():
        return QualifierCompatibility.INCOMPLETE, (SemanticMatchReason.INSUFFICIENT_INFORMATION,)
    return QualifierCompatibility.COMPATIBLE, ()


def _numeric_value(claim: CanonicalClaim) -> tuple[Decimal, str, bool] | None:
    raw = claim.object_or_value
    if raw is None:
        return None
    match = _VALUE_RE.fullmatch(raw)
    if match is None:
        return None
    try:
        amount = Decimal(match.group(2).replace(",", ""))
    except InvalidOperation:
        return None
    amount *= _SCALES.get((match.group(3) or "").casefold(), Decimal(1))
    unit = (match.group(4) or str(claim.qualifiers.get("unit") or "")).casefold()
    unit = _UNIT_ALIASES.get(unit, unit)
    return amount, unit, bool(match.group(1))


class SemanticClaimMatcher:
    """A bounded deterministic matcher; ambiguous language stays UNKNOWN."""

    version = MATCHER_VERSION

    def compare(self, left: CanonicalClaim, right: CanonicalClaim) -> SemanticClaimComparison:
        qualifier_state, qualifier_reasons = _qualifier_comparison(left, right)
        exact_text = exact_proposition_key(left) == exact_proposition_key(right)
        subject_left = _token_key(left.subject) if left.subject else None
        subject_right = _token_key(right.subject) if right.subject else None
        predicate_left = _predicate_key(left.predicate) if left.predicate else None
        predicate_right = _predicate_key(right.predicate) if right.predicate else None
        left_num = _numeric_value(left)
        right_num = _numeric_value(right)

        base_signals: dict[str, str | bool | None] = {
            "normalized_text_equal": exact_text,
            "subject_equal": (
                subject_left == subject_right if subject_left and subject_right else None
            ),
            "predicate_equal": (
                predicate_left == predicate_right if predicate_left and predicate_right else None
            ),
            "numeric_values_equal": (
                left_num[0] == right_num[0] if left_num and right_num else None
            ),
            "approximation_equal": (
                left_num[2] == right_num[2] if left_num and right_num else None
            ),
            "units_equal": (
                left_num[1] == right_num[1] if left_num and right_num else None
            ),
            "polarity_equal": (
                left.polarity == right.polarity
                if left.polarity is not None and right.polarity is not None
                else None
            ),
            "qualifiers_equal": qualifier_state is QualifierCompatibility.COMPATIBLE,
            "claim_types_equal": (
                left.claim_type == right.claim_type
                if left.claim_type is not CanonicalClaimType.UNKNOWN
                and right.claim_type is not CanonicalClaimType.UNKNOWN
                else None
            ),
        }

        if left.run_id != right.run_id:
            return self._result(
                left, right, ClaimRelationType.UNKNOWN, 0.0,
                (SemanticMatchReason.INSUFFICIENT_INFORMATION,),
                "Cross-run comparison is analyzer-only and not clustered.",
                base_signals, qualifier_state, "fallback_unknown",
            )
        frame_aligned = bool(
            subject_left and subject_right and subject_left == subject_right
            and predicate_left and predicate_right and predicate_left == predicate_right
        )
        if subject_left and subject_right and subject_left != subject_right:
            relation = (
                ClaimRelationType.RELATED
                if predicate_left and predicate_left == predicate_right
                else ClaimRelationType.UNKNOWN
            )
            method = (
                "deterministic_structured"
                if relation is ClaimRelationType.RELATED
                else "fallback_unknown"
            )
            return self._result(
                left, right, relation,
                0.96 if relation is ClaimRelationType.RELATED else 0.35,
                (SemanticMatchReason.SUBJECT_MISMATCH,),
                "Known subjects differ; SAME is prohibited. RELATED requires an aligned predicate.",
                base_signals, qualifier_state, method,
            )
        if qualifier_state is QualifierCompatibility.INCOMPATIBLE:
            relation = ClaimRelationType.RELATED if frame_aligned else ClaimRelationType.UNKNOWN
            method = "deterministic_structured" if frame_aligned else "fallback_unknown"
            return self._result(
                left, right, relation, 0.98 if frame_aligned else 0.4,
                qualifier_reasons,
                "A known qualifier differs; RELATED requires an aligned subject/predicate frame.",
                base_signals, qualifier_state, method,
            )
        if (left.polarity is not None and right.polarity is not None
                and left.polarity != right.polarity and frame_aligned):
            return self._result(
                left, right, ClaimRelationType.CONTRADICTS, 0.98,
                (SemanticMatchReason.POLARITY_CONFLICT,),
                "Polarity differs while the available scope is compatible.",
                base_signals, qualifier_state, "deterministic_structured",
            )
        if (left.claim_type is not CanonicalClaimType.UNKNOWN
                and right.claim_type is not CanonicalClaimType.UNKNOWN
                and left.claim_type != right.claim_type):
            return self._result(
                left, right, ClaimRelationType.UNKNOWN, 0.45,
                (SemanticMatchReason.CLAIM_TYPE_MISMATCH,),
                "Claim types differ; no equivalence is inferred.",
                base_signals, qualifier_state, "fallback_unknown",
            )

        if left_num and right_num:
            if left_num[1] != right_num[1]:
                return self._result(
                    left, right, ClaimRelationType.RELATED, 0.93,
                    (SemanticMatchReason.UNIT_MISMATCH,),
                    "Units differ and no conversion context is applied.",
                    base_signals, qualifier_state, "deterministic_structured",
                )
            if left_num[2] != right_num[2]:
                return self._result(
                    left, right, ClaimRelationType.UNKNOWN, 0.55,
                    (SemanticMatchReason.INSUFFICIENT_INFORMATION,),
                    "Approximate and exact numeric wording is not collapsed.",
                    base_signals, qualifier_state, "fallback_unknown",
                )
            if left_num[0] != right_num[0]:
                relation = (ClaimRelationType.CONTRADICTS
                    if qualifier_state is QualifierCompatibility.COMPATIBLE and frame_aligned
                    else ClaimRelationType.UNKNOWN)
                method = (
                    "deterministic_structured"
                    if relation is ClaimRelationType.CONTRADICTS
                    else "fallback_unknown"
                )
                return self._result(
                    left, right, relation,
                    0.94 if relation is ClaimRelationType.CONTRADICTS else 0.5,
                    (SemanticMatchReason.VALUE_MISMATCH,),
                    "Numeric values differ; contradiction needs aligned qualifiers.",
                    base_signals, qualifier_state, method,
                )
            if (left_num[2] == right_num[2]
                    and subject_left and subject_left == subject_right
                    and predicate_left and predicate_left == predicate_right
                    and qualifier_state is QualifierCompatibility.COMPATIBLE):
                return self._result(
                    left, right, ClaimRelationType.SAME, 0.95,
                    (SemanticMatchReason.STRUCTURED_FIELDS_MATCH,),
                    "Normalized value, unit, predicate, subject and qualifiers match.",
                    base_signals, qualifier_state, "deterministic_structured",
                )

        if exact_text:
            return self._result(
                left, right, ClaimRelationType.SAME, 0.99,
                (SemanticMatchReason.EXACT_NORMALIZED_MATCH,),
                "Claim text matches after safe presentation normalization with no known conflict.",
                base_signals, qualifier_state, "deterministic_exact",
            )

        fields_complete = all((subject_left, subject_right, predicate_left, predicate_right))
        if fields_complete and subject_left == subject_right and predicate_left == predicate_right:
            if (left.object_or_value is not None and right.object_or_value is not None
                    and _token_key(left.object_or_value) == _token_key(right.object_or_value)
                    and qualifier_state is QualifierCompatibility.COMPATIBLE):
                reason = (
                    SemanticMatchReason.SEMANTIC_PARAPHRASE
                    if normalize_safe_text(left.predicate or "")
                    != normalize_safe_text(right.predicate or "")
                    or normalize_safe_text(left.original_text)
                    != normalize_safe_text(right.original_text)
                    else SemanticMatchReason.STRUCTURED_FIELDS_MATCH
                )
                return self._result(
                    left, right, ClaimRelationType.SAME, 0.93,
                    (reason,),
                    "Subject, bounded predicate equivalence, value and observed qualifiers match.",
                    base_signals, qualifier_state, "deterministic_structured",
                )
            if (
                (left_num is None or right_num is None)
                and left.object_or_value is not None
                and right.object_or_value is not None
                and normalize_safe_text(left.object_or_value)
                == normalize_safe_text(right.object_or_value)
                and qualifier_state is QualifierCompatibility.COMPATIBLE
            ):
                return self._result(
                    left, right, ClaimRelationType.SAME, 0.91,
                    (SemanticMatchReason.STRUCTURED_FIELDS_MATCH,),
                    "Structured subject, predicate and object match.",
                    base_signals, qualifier_state, "deterministic_structured",
                )

        if qualifier_state is QualifierCompatibility.INCOMPLETE:
            return self._result(
                left, right, ClaimRelationType.UNKNOWN, 0.5,
                (SemanticMatchReason.INSUFFICIENT_INFORMATION,),
                "Qualifier metadata is incomplete; no equivalence is inferred.",
                base_signals, qualifier_state, "fallback_unknown",
            )
        return self._result(
            left, right, ClaimRelationType.UNKNOWN, 0.25,
            (SemanticMatchReason.INSUFFICIENT_INFORMATION,),
            "Available lexical and structured fields do not establish a safe relation.",
            base_signals, qualifier_state, "fallback_unknown",
        )

    @staticmethod
    def _result(
        left: CanonicalClaim,
        right: CanonicalClaim,
        relation: ClaimRelationType,
        confidence: float,
        reasons: tuple[SemanticMatchReason, ...],
        rationale: str,
        signals: dict[str, str | bool | None],
        qualifier_state: QualifierCompatibility,
        method: str,
    ) -> SemanticClaimComparison:
        if left.question_id != right.question_id or left.dimension_key != right.dimension_key:
            signals = {**signals, "same_question": left.question_id == right.question_id,
                "same_dimension": left.dimension_key == right.dimension_key}
            if relation is ClaimRelationType.SAME:
                relation = ClaimRelationType.RELATED
                reasons = (SemanticMatchReason.SCOPE_MISMATCH,)
                rationale = "Proposition text matches, but question/dimension scope differs."
        return SemanticClaimComparison(
            left_claim_id=str(left.claim_id), right_claim_id=str(right.claim_id),
            relation=relation, confidence=confidence, reason_codes=reasons,
            rationale=rationale, signals=signals,
            qualifier_compatibility=qualifier_state, method=method,
        )
