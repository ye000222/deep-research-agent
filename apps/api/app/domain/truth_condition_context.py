# ruff: noqa: RUF001
"""Deterministic, non-persistent V2.1R truth-condition context shadow projection.

This module consumes already-produced v3-shaped data. It never calls a model,
rewrites an Atomic Proposition, or participates in ResearchLoop decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

TRUTH_CONTEXT_SCHEMA_VERSION = "truth-condition-context-shadow-v1"
NUMERIC_NORMALIZATION_VERSION = "typed-decimal-exact-v1"


class ContextType(StrEnum):
    TEMPORAL = "TEMPORAL"
    VERSION = "VERSION"
    SCOPE = "SCOPE"
    CONFIGURATION = "CONFIGURATION"
    SCENARIO = "SCENARIO"
    CONDITION = "CONDITION"
    EPISTEMIC = "EPISTEMIC"
    MEASUREMENT = "MEASUREMENT"
    SAMPLE_SCOPE = "SAMPLE_SCOPE"
    METHOD = "METHOD"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ContextScopeKind(StrEnum):
    PROPOSITION = "PROPOSITION"
    PROPOSITION_SET = "PROPOSITION_SET"
    CLAUSE = "CLAUSE"
    ASSERTION_GROUP = "ASSERTION_GROUP"
    CLAIM = "CLAIM"
    UNKNOWN = "UNKNOWN"


class AttachmentRelation(StrEnum):
    APPLIES_TO = "APPLIES_TO"
    CONDITIONS = "CONDITIONS"
    ATTRIBUTES = "ATTRIBUTES"
    UNKNOWN = "UNKNOWN"


class AttachmentStatus(StrEnum):
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"


class ContextValidationOutcome(StrEnum):
    PRESERVED = "PRESERVED"
    MISSING = "MISSING"
    ATTACHMENT_AMBIGUOUS = "ATTACHMENT_AMBIGUOUS"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    SCOPE_UNDEREXPANSION = "SCOPE_UNDEREXPANSION"
    SCOPE_OVEREXPANSION = "SCOPE_OVEREXPANSION"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    DETECTOR_LIMIT = "DETECTOR_LIMIT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ContextFamilyCoverage(StrEnum):
    """Coverage is bounded by detector family, never a claim-wide semantic guarantee."""

    SUPPORTED_FOR_CONTRACT = "SUPPORTED_FOR_CONTRACT"
    PARTIAL_DETECTOR_LIMIT = "PARTIAL_DETECTOR_LIMIT"
    NOT_EVALUATED = "NOT_EVALUATED"


class ContextPresenceConclusion(StrEnum):
    CONTEXT_DETECTED = "CONTEXT_DETECTED"
    NO_CONTEXT_DETECTED = "NO_CONTEXT_DETECTED"
    NO_CONTEXT_EXISTS = "NO_CONTEXT_EXISTS"
    DETECTOR_CANNOT_DECIDE = "DETECTOR_CANNOT_DECIDE"


class ContextOrigin(StrEnum):
    SOURCE = "SOURCE"
    OUTPUT = "OUTPUT"


class ContextScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ContextScopeKind
    evidence: str | None = None
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)
    target_refs: tuple[str, ...] = ()


class ContextAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str
    target_ref: str | None = None
    target_kind: str
    relation: AttachmentRelation
    status: AttachmentStatus
    reason_code: str


class TruthConditionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str
    semantic_signature: str
    context_type: ContextType
    semantic_value: dict[str, Any]
    lexical_value: str | None = None
    source_span: str | None = None
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)
    materiality: str = "UNKNOWN"
    scope: ContextScope
    attachments: tuple[ContextAttachment, ...] = ()
    certainty: str = "DETERMINISTIC"
    detection_evidence: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    origin: ContextOrigin
    schema_version: str = TRUTH_CONTEXT_SCHEMA_VERSION
    normalization_version: str = NUMERIC_NORMALIZATION_VERSION


class ContextGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: ContextOrigin
    claim_ref: str
    contexts: tuple[TruthConditionContext, ...] = ()
    diagnostics: tuple[str, ...] = ()
    scope_expansions: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    source_detection_coverage: dict[str, ContextFamilyCoverage] = Field(default_factory=dict)
    context_presence: ContextPresenceConclusion = ContextPresenceConclusion.DETECTOR_CANNOT_DECIDE
    source_detection_complete: bool = False
    schema_version: str = TRUTH_CONTEXT_SCHEMA_VERSION


class ContextPreservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_context_ref: str
    output_context_refs: tuple[str, ...] = ()
    outcome: ContextValidationOutcome
    reason_codes: tuple[str, ...] = ()
    source_presence_conclusion: ContextPresenceConclusion | None = None


class VerificationAssertionView(BaseModel):
    """Read-only derived proposition view; deliberately not a persistence model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_ref: str
    proposition_core: dict[str, Any]
    material_context_signatures: tuple[str, ...] = ()
    material_context_unknown: bool = False
    unknown_context_details: tuple[str, ...] = ()
    resolved_context_refs: tuple[str, ...] = ()
    detector_limited_context_families: tuple[str, ...] = ()
    source_context_completeness: str = "NOT_PROVIDED"
    source_context_presence: str = "DETECTOR_CANNOT_DECIDE"
    composition_eligibility: str = "UNKNOWN"
    schema_version: str = TRUTH_CONTEXT_SCHEMA_VERSION
    normalization_version: str = NUMERIC_NORMALIZATION_VERSION


class LegacyContextAdapterResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_preservation_obligation_status: str = "ADAPT"
    proposition_semantic_view_status: str = "ADAPT"
    semantic_carrier_status: str = "DEPRECATE"
    source_graph: ContextGraph
    output_graph: ContextGraph
    diagnostic_only: bool = True
    reason_code: str = "legacy_inputs_adapted_to_context_graph_projection"


_CLAUSE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_DATE_RANGE = re.compile(
    r"(?<!\d)(\d{4}-\d{2}-\d{2})\s*(?:to|through|–|—|至|到)\s*(\d{4}-\d{2}-\d{2})(?!\d)",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_CN_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})年([01]?\d)月([0-3]?\d)日")
_CN_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})年([01]?\d)月(?![0-3]?\d日)")
_YEAR_RANGE = re.compile(
    r"(?i)(?<![A-Za-z0-9])((?:19|20)\d{2})\s*(?:to|through|–|—|至|到)\s*((?:19|20)\d{2})(?:年|年度)?"
)
_YEAR_PERIOD = re.compile(
    r"(?i)(?:during\s+|throughout\s+|in\s+)(Q[1-4]|FY)\s*((?:19|20)\d{2})|((?:19|20)\d{2})年(上半年|下半年|第[一二三四]季度)|(?<!\d)((?:19|20)\d{2})\s*Q([1-4])"
)
_YEAR = re.compile(r"(?<![A-Za-z0-9])((?:19|20)\d{2})(?:年|年度)?")
_VERSION = re.compile(r"(?i)(?<![\w.])(?:version\s+|v)(\d+(?:\.\d+){1,3})(?![\w.])")
_ATTRIBUTION_EN = re.compile(
    r"(?i)(?P<agent>[^,;.!?\n]{1,100}?)\s+(?P<operator>reports|reported|claims|claimed|predicts|predicted|estimates|estimated)\s+that\b"
)
_ATTRIBUTION_ZH = re.compile(
    r"(?P<agent>据|根据)?(?P<source>[^，。；\n]{1,80}?(?:报告|研究|综述|统计|机构))(?P<operator>预测|预计|估计|指出|显示|称)"
)
_MODALITY = re.compile(r"(?i)\b(may|might|possibly|likely|estimated|projected)\b|可能|或将|预计")
_CONDITION_EN = re.compile(r"(?i)\bif\s+(?P<value>[^,.;!?\n]+)")
_CONDITION_ZH = re.compile(r"(?:如果|若)(?P<value>[^，。；\n]+)")
_CONFIG_EN = re.compile(r"(?i)\bunder\s+(?:the\s+)?configuration\s+(?P<value>[^,.;!?\n]+)")
_CONFIG_ZH = re.compile(r"在(?P<value>[^，。；\n]{1,60}?配置)下")
_METHOD_EN = re.compile(r"(?i)\b(?:using|by applying)\s+(?P<value>[^,.;!?\n]+)")
_METHOD_ZH = re.compile(r"(?:通过|采用)(?P<value>[^，。；\n]{1,60}?)(?:方法|方式)")
_SAMPLE_EN = re.compile(r"(?i)\b(?:based on|from)\s+(?P<count>\d+)\s+(?P<unit>samples?|cases?)\b")
_SAMPLE_ZH = re.compile(r"(?P<count>\d+)个(?P<unit>样本|案例)")
_MEASURE_EN = re.compile(r"(?i)\bmeasured\s+by\s+(?P<value>[^,.;!?\n]+)")
_MEASURE_ZH = re.compile(r"以(?P<value>[^，。；\n]{1,40}?)(?:衡量|测量)")
_SCOPE_EN = re.compile(r"(?i)\b(?:within|among|for)\s+(?P<value>[^,.;!?\n]+)")
_SCOPE_ZH = re.compile(r"(?:在|针对)(?P<value>[^，。；\n]{1,40}?)(?:地区|区域|人群|群体|场景)")
_CLAIM_SCOPE_EN = re.compile(r"(?i)\bthroughout\s+(?:the\s+)?claim\b")
_CLAIM_SCOPE_ZH = re.compile(r"以下各项均|对全文所有断言")
_PROPOSITION_SET_SCOPE = re.compile(
    r"(?i)\b(?:in|for)\s+both\s+(?:cases|statements|propositions)\b|两项均|两者均"
)
_SCENARIO_EN = re.compile(r"(?i)\b(?:in|under)\s+(?:the\s+)?scenario\s+(?P<value>[^,.;!?\n]+)")
_SCENARIO_ZH = re.compile(r"在(?P<value>[^，。；\n]{1,40}?场景)(?:下|中)")
_PERCENT = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*$")

# Explicit bounded contract: these families have deterministic lexical paths,
# but none claims general natural-language semantic coverage.
_SOURCE_DETECTION_COVERAGE = {
    item.value: ContextFamilyCoverage.PARTIAL_DETECTOR_LIMIT
    for item in ContextType
    if item not in {ContextType.UNKNOWN, ContextType.OTHER}
}
_SOURCE_DETECTION_COVERAGE.update(
    {
        ContextType.TEMPORAL.value: ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT,
        ContextType.VERSION.value: ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT,
    }
)


def _qualifier_semantics(
    category: str, key: str, value: object
) -> tuple[ContextType, dict[str, Any]]:
    context_type = _QUALIFIER_TYPE.get(category, ContextType.UNKNOWN)
    semantic: dict[str, Any]
    if category in {"time", "date", "period"}:
        text = str(value).strip()
        year = re.fullmatch(r"(?:year[:= ]*)?((?:19|20)\d{2})(?:年|年度)?", text, re.IGNORECASE)
        semantic = {"year": int(year.group(1))} if year else {"key": key, "value": value}
    elif category == "version":
        semantic = {"version": str(value).strip().lower()}
    elif category in {"modality", "attribution"}:
        semantic = (
            {"operator": "ATTRIBUTED_TO", "agent": value}
            if category == "attribution"
            else {"operator": str(value).upper(), "agent": None}
        )
    else:
        semantic = {"key": key, "value": value}
    normalized = normalize_semantic_value(semantic)
    return context_type, normalized if isinstance(normalized, dict) else {"value": normalized}


def _decimal_text(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    normalized = number.normalize()
    return "0" if normalized == 0 else format(normalized, "f")


def normalize_semantic_value(value: object) -> object:
    """Canonicalize exact typed numbers without unit or currency conversion."""
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key in sorted(value):
            item = value[key]
            if key in {"value", "lower_bound", "upper_bound", "tolerance"}:
                decimal = _decimal_text(item)
                normalized[key] = decimal if decimal is not None else item
            else:
                normalized[key] = normalize_semantic_value(item)
        return normalized
    if isinstance(value, list | tuple):
        return [normalize_semantic_value(item) for item in value]
    if isinstance(value, str):
        percent = _PERCENT.fullmatch(value)
        if percent:
            decimal = _decimal_text(percent.group(1))
            return {"kind": "quantity", "operator": "exact", "value": decimal, "unit": "%"}
        return value
    return value


_EPISTEMIC_OPERATOR_ALIASES = {
    "REPORTS": "ATTRIBUTION",
    "REPORTED": "ATTRIBUTION",
    "CLAIMS": "ATTRIBUTION",
    "CLAIMED": "ATTRIBUTION",
    "ATTRIBUTED_TO": "ATTRIBUTION",
    "指出": "ATTRIBUTION",
    "显示": "ATTRIBUTION",
    "称": "ATTRIBUTION",
    "PREDICTS": "PREDICTION",
    "PREDICTED": "PREDICTION",
    "预测": "PREDICTION",
    "预计": "PREDICTION",
    "ESTIMATES": "ESTIMATE",
    "ESTIMATED": "ESTIMATE",
    "估计": "ESTIMATE",
    "可能": "POSSIBILITY",
    "MAY": "POSSIBILITY",
    "MIGHT": "POSSIBILITY",
    "LIKELY": "LIKELIHOOD",
}


def canonical_context_value(context: TruthConditionContext) -> dict[str, Any]:
    """Return a typed comparison value; semantic_signature remains diagnostic only."""
    value = context.semantic_value
    if context.context_type is ContextType.TEMPORAL:
        year = value.get("year")
        if isinstance(year, int) and 1000 <= year <= 9999:
            return {"family": "TEMPORAL", "kind": "YEAR", "start": year, "end": year}
        start = value.get("start_date") or value.get("start")
        end = value.get("end_date") or value.get("end")
        start_year = value.get("start_year")
        end_year = value.get("end_year")
        if isinstance(start_year, int) and isinstance(end_year, int):
            return {
                "family": "TEMPORAL",
                "kind": "YEAR_RANGE",
                "start": start_year,
                "end": end_year,
            }
        date = value.get("date")
        if isinstance(date, dict) and all(
            isinstance(date.get(key), int) for key in ("year", "month", "day")
        ):
            return {
                "family": "TEMPORAL",
                "kind": "DATE",
                "year": date["year"],
                "month": date["month"],
                "day": date["day"],
            }
        if isinstance(date, str):
            cn_date = re.fullmatch(r"((?:19|20)\d{2})年([01]?\d)月([0-3]?\d)日", date.strip())
            if cn_date:
                return {
                    "family": "TEMPORAL",
                    "kind": "DATE",
                    "year": int(cn_date.group(1)),
                    "month": int(cn_date.group(2)),
                    "day": int(cn_date.group(3)),
                }
            return {"family": "TEMPORAL", "kind": "DATE", "value": date.strip()}
        month = value.get("month")
        if isinstance(month, dict) and all(
            isinstance(month.get(key), int) for key in ("year", "month")
        ):
            return {
                "family": "TEMPORAL",
                "kind": "MONTH",
                "year": month["year"],
                "month": month["month"],
            }
        if isinstance(start, str) and isinstance(end, str):
            return {
                "family": "TEMPORAL",
                "kind": "RANGE",
                "start": start.strip().lower(),
                "end": end.strip().lower(),
            }
        period = value.get("period") or value.get("value") or context.lexical_value
        if isinstance(period, str):
            cleaned = re.sub(r"\s+", " ", period.strip().lower())
            named_period = re.fullmatch(
                r"(?:during\s+|throughout\s+|in\s+)?(q[1-4]|fy)\s+((?:19|20)\d{2})", cleaned
            )
            if named_period:
                return {
                    "family": "TEMPORAL",
                    "kind": "NAMED_PERIOD",
                    "period": named_period.group(1).upper(),
                    "year": int(named_period.group(2)),
                }
            compact_period = re.fullmatch(r"((?:19|20)\d{2})\s*q([1-4])", cleaned)
            if compact_period:
                return {
                    "family": "TEMPORAL",
                    "kind": "NAMED_PERIOD",
                    "period": f"Q{compact_period.group(2)}",
                    "year": int(compact_period.group(1)),
                }
            year_range = re.fullmatch(
                r"((?:19|20)\d{2})\s*(?:to|through|–|—|至|到)\s*((?:19|20)\d{2})(?:年|年度)?",
                cleaned,
            )
            if year_range:
                return {
                    "family": "TEMPORAL",
                    "kind": "YEAR_RANGE",
                    "start": int(year_range.group(1)),
                    "end": int(year_range.group(2)),
                }
            zh_period = re.fullmatch(r"((?:19|20)\d{2})年(上半年|下半年|第[一二三四]季度)", cleaned)
            if zh_period:
                return {
                    "family": "TEMPORAL",
                    "kind": "NAMED_PERIOD",
                    "period": zh_period.group(2),
                    "year": int(zh_period.group(1)),
                }
            date_range = re.fullmatch(
                r"(\d{4}-\d{2}-\d{2})\s*(?:to|through|–|—|至|到)\s*(\d{4}-\d{2}-\d{2})",
                cleaned,
            )
            if date_range:
                return {
                    "family": "TEMPORAL",
                    "kind": "DATE_RANGE",
                    "start": date_range.group(1),
                    "end": date_range.group(2),
                }
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
                return {"family": "TEMPORAL", "kind": "DATE", "value": cleaned}
            cn_month = re.fullmatch(r"((?:19|20)\d{2})年([01]?\d)月", cleaned)
            if cn_month:
                return {
                    "family": "TEMPORAL",
                    "kind": "MONTH",
                    "year": int(cn_month.group(1)),
                    "month": int(cn_month.group(2)),
                }
            year_match = re.fullmatch(r"(?:during\s+|in\s+)?((?:19|20)\d{2})(?:年|年度)?", cleaned)
            if year_match:
                year_value = int(year_match.group(1))
                return {
                    "family": "TEMPORAL",
                    "kind": "YEAR",
                    "start": year_value,
                    "end": year_value,
                }
            return {"family": "TEMPORAL", "kind": "PERIOD", "value": cleaned}
    if context.context_type is ContextType.EPISTEMIC:
        operator = str(value.get("operator", "UNKNOWN")).strip().upper()
        agent = value.get("agent")
        modality = value.get("modality") or value.get("certainty")
        return {
            "family": "EPISTEMIC",
            "operator": _EPISTEMIC_OPERATOR_ALIASES.get(operator, operator),
            "agent": re.sub(r"\s+", " ", str(agent).strip()).casefold() if agent else None,
            "modality": str(modality).strip().casefold() if modality else None,
        }
    if context.context_type is ContextType.SCOPE:
        subtype = value.get("subtype") or value.get("category") or value.get("key") or "UNKNOWN"
        payload = value.get("payload", value.get("value", value))
        return {
            "family": "SCOPE",
            "subtype": re.sub(r"\s+", " ", str(subtype).strip().casefold()),
            "payload": normalize_semantic_value(payload),
        }
    if context.context_type in {
        ContextType.METHOD,
        ContextType.SCENARIO,
        ContextType.CONDITION,
        ContextType.CONFIGURATION,
        ContextType.MEASUREMENT,
        ContextType.SAMPLE_SCOPE,
        ContextType.VERSION,
    }:
        subtype = value.get("subtype") or value.get("category") or value.get("key") or "UNKNOWN"
        payload = value.get("payload", value.get("value", value))
        return {
            "family": context.context_type.value,
            "subtype": re.sub(r"\s+", " ", str(subtype).strip().casefold()),
            "payload": normalize_semantic_value(payload),
        }
    return {"family": context.context_type.value, "value": normalize_semantic_value(value)}


def _canonical_values_compatible(
    source: TruthConditionContext,
    output: TruthConditionContext,
    output_contexts: tuple[TruthConditionContext, ...],
) -> bool:
    source_value = canonical_context_value(source)
    output_value = canonical_context_value(output)
    if source.context_type is ContextType.EPISTEMIC:
        if source_value["operator"] != output_value["operator"]:
            return False
        source_agent = source_value.get("agent")
        if source_agent and source_agent != output_value.get("agent"):
            # A source attribution operator may be represented by an explicit
            # attribution field alongside a modality field on the same target.
            return any(
                candidate.context_type is ContextType.EPISTEMIC
                and canonical_context_value(candidate).get("agent") == source_agent
                for candidate in output_contexts
            )
        source_modality = source_value.get("modality")
        return not source_modality or source_modality == output_value.get("modality")
    if source.context_type is ContextType.TEMPORAL:
        return source_value == output_value
    if source.context_type is ContextType.SCOPE:
        source_payload = source_value.get("payload")
        output_payload = output_value.get("payload")
        source_subtype = source_value.get("subtype", "UNKNOWN")
        output_subtype = output_value.get("subtype", "UNKNOWN")
        return source_payload == output_payload and (
            source_subtype == output_subtype or "unknown" in {source_subtype, output_subtype}
        )
    if source.context_type in {
        ContextType.METHOD,
        ContextType.SCENARIO,
        ContextType.CONDITION,
        ContextType.CONFIGURATION,
        ContextType.MEASUREMENT,
        ContextType.SAMPLE_SCOPE,
        ContextType.VERSION,
    }:
        return source_value.get("payload") == output_value.get("payload") and (
            source_value.get("subtype") == output_value.get("subtype")
            or "unknown" in {source_value.get("subtype"), output_value.get("subtype")}
        )
    return source_value == output_value


def _signature(context_type: ContextType, semantic_value: dict[str, Any]) -> str:
    payload = json.dumps(
        {"type": context_type.value, "value": normalize_semantic_value(semantic_value)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _local_ref(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def _proposition_span(proposition: dict[str, Any]) -> str | None:
    span = proposition.get("source_span")
    if isinstance(span, str) and span:
        return span
    decomposition = proposition.get("decomposition")
    if isinstance(decomposition, dict):
        nested = decomposition.get("source_span")
        if isinstance(nested, str) and nested:
            return nested
    return None


def _locate_spans(
    claim_text: str, propositions: list[dict[str, Any]]
) -> dict[str, tuple[int, int] | None]:
    result: dict[str, tuple[int, int] | None] = {}
    cursor = 0
    for index, proposition in enumerate(propositions):
        ref = str(proposition.get("local_ref") or f"p{index}")
        span = _proposition_span(proposition)
        if span is None:
            result[ref] = None
            continue
        start = claim_text.find(span, cursor)
        if start < 0:
            start = claim_text.find(span)
        if start < 0:
            result[ref] = None
            continue
        end = start + len(span)
        result[ref] = (start, end)
        cursor = end
    return result


def _clauses(text: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    start = 0
    for match in _CLAUSE_SPLIT.finditer(text):
        end = match.end()
        if text[start:end].strip():
            result.append((start, end, text[start:end]))
        start = end
    if text[start:].strip():
        result.append((start, len(text), text[start:]))
    return result or [(0, len(text), text)]


def _context_candidates(clause: str) -> list[dict[str, Any]]:
    """Return staged lexical candidates with semantic-role evidence."""
    candidates: list[dict[str, Any]] = []

    def add_matches(
        pattern: re.Pattern[str],
        context_type: ContextType,
        value_fn: Any,
        role_reason: str,
        relation: AttachmentRelation = AttachmentRelation.APPLIES_TO,
    ) -> None:
        for match in pattern.finditer(clause):
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "span": match.group(0),
                    "type": context_type,
                    "value": value_fn(match),
                    "relation": relation,
                    "evidence": role_reason,
                    "reason": "context_created",
                }
            )

    explicit_temporal_spans: list[tuple[int, int]] = []
    for match in _DATE_RANGE.finditer(clause):
        explicit_temporal_spans.append((match.start(), match.end()))
        candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "span": match.group(0),
                "type": ContextType.TEMPORAL,
                "value": {"start_date": match.group(1), "end_date": match.group(2)},
                "relation": AttachmentRelation.APPLIES_TO,
                "evidence": "bounded_iso_date_range_pattern",
                "reason": "context_created",
            }
        )
    for match in _ISO_DATE.finditer(clause):
        if any(
            start <= match.start() and match.end() <= end for start, end in explicit_temporal_spans
        ):
            continue
        explicit_temporal_spans.append((match.start(), match.end()))
        candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "span": match.group(0),
                "type": ContextType.TEMPORAL,
                "value": {"date": match.group(1)},
                "relation": AttachmentRelation.APPLIES_TO,
                "evidence": "bounded_iso_date_pattern",
                "reason": "context_created",
            }
        )
    for pattern, kind, fields in (
        (_CN_DATE, "DATE", ("year", "month", "day")),
        (_CN_MONTH, "MONTH", ("year", "month")),
    ):
        for match in pattern.finditer(clause):
            if any(
                start <= match.start() and match.end() <= end
                for start, end in explicit_temporal_spans
            ):
                continue
            explicit_temporal_spans.append((match.start(), match.end()))
            component_values = {
                key: int(raw) for key, raw in zip(fields, match.groups(), strict=True)
            }
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "span": match.group(0),
                    "type": ContextType.TEMPORAL,
                    "value": {"date" if kind == "DATE" else "month": component_values},
                    "relation": AttachmentRelation.APPLIES_TO,
                    "evidence": f"bounded_chinese_{kind.lower()}_pattern",
                    "reason": "context_created",
                }
            )
    for match in _YEAR_RANGE.finditer(clause):
        if any(
            start <= match.start() and match.end() <= end for start, end in explicit_temporal_spans
        ):
            continue
        explicit_temporal_spans.append((match.start(), match.end()))
        candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "span": match.group(0),
                "type": ContextType.TEMPORAL,
                "value": {"start_year": int(match.group(1)), "end_year": int(match.group(2))},
                "relation": AttachmentRelation.APPLIES_TO,
                "evidence": "bounded_year_range_pattern",
                "reason": "context_created",
            }
        )
    for match in _YEAR_PERIOD.finditer(clause):
        explicit_temporal_spans.append((match.start(), match.end()))
        candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "span": match.group(0),
                "type": ContextType.TEMPORAL,
                "value": {"period": re.sub(r"\s+", " ", match.group(0).strip())},
                "relation": AttachmentRelation.APPLIES_TO,
                "evidence": "bounded_named_period_pattern",
                "reason": "context_created",
            }
        )
    for match in _YEAR.finditer(clause):
        if any(
            start <= match.start() and match.end() <= end for start, end in explicit_temporal_spans
        ):
            continue
        candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "span": match.group(0),
                "type": ContextType.TEMPORAL,
                "value": {"year": int(match.group(1))},
                "relation": AttachmentRelation.APPLIES_TO,
                "evidence": "date_pattern_with_temporal_role",
                "reason": "context_created",
            }
        )
    add_matches(
        _VERSION, ContextType.VERSION, lambda m: {"version": m.group(1)}, "explicit_version_marker"
    )

    for pattern, language in ((_ATTRIBUTION_EN, "en"), (_ATTRIBUTION_ZH, "zh")):
        for match in pattern.finditer(clause):
            if language == "en":
                operator = match.group("operator").lower()
                agent = match.group("agent").strip()
            else:
                raw_operator = match.group("operator")
                operator = {
                    "预测": "PREDICTS",
                    "预计": "ESTIMATES",
                    "估计": "ESTIMATES",
                    "指出": "REPORTS",
                    "显示": "REPORTS",
                    "称": "CLAIMS",
                }[raw_operator]
                agent = match.group("source").strip()
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "span": match.group(0),
                    "type": ContextType.EPISTEMIC,
                    "value": {"operator": operator.upper(), "agent": agent or None},
                    "relation": AttachmentRelation.ATTRIBUTES,
                    "evidence": "attribution_predicate_and_agent_pattern",
                    "reason": "context_created",
                }
            )

    # A bare modality is retained even when no attribution agent is stated.
    for match in _MODALITY.finditer(clause):
        covered = any(
            item["type"] is ContextType.EPISTEMIC and item["start"] <= match.start() < item["end"]
            for item in candidates
        )
        if not covered:
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "span": match.group(0),
                    "type": ContextType.EPISTEMIC,
                    "value": {"operator": match.group(0).upper(), "agent": None},
                    "relation": AttachmentRelation.ATTRIBUTES,
                    "evidence": "modality_lexeme_in_assertion_context",
                    "reason": "context_created",
                }
            )

    for pattern, ctype, evidence, relation in (
        (
            _CONDITION_EN,
            ContextType.CONDITION,
            "conditional_construction",
            AttachmentRelation.CONDITIONS,
        ),
        (
            _CONDITION_ZH,
            ContextType.CONDITION,
            "conditional_construction",
            AttachmentRelation.CONDITIONS,
        ),
        (
            _CONFIG_EN,
            ContextType.CONFIGURATION,
            "explicit_configuration_relation",
            AttachmentRelation.APPLIES_TO,
        ),
        (
            _CONFIG_ZH,
            ContextType.CONFIGURATION,
            "explicit_configuration_relation",
            AttachmentRelation.APPLIES_TO,
        ),
        (_METHOD_EN, ContextType.METHOD, "method_relation_verb", AttachmentRelation.APPLIES_TO),
        (_METHOD_ZH, ContextType.METHOD, "method_relation_verb", AttachmentRelation.APPLIES_TO),
        (
            _MEASURE_EN,
            ContextType.MEASUREMENT,
            "measurement_relation_verb",
            AttachmentRelation.APPLIES_TO,
        ),
        (
            _MEASURE_ZH,
            ContextType.MEASUREMENT,
            "measurement_relation_verb",
            AttachmentRelation.APPLIES_TO,
        ),
        (_SCOPE_EN, ContextType.SCOPE, "explicit_scope_preposition", AttachmentRelation.APPLIES_TO),
        (_SCOPE_ZH, ContextType.SCOPE, "explicit_scope_marker", AttachmentRelation.APPLIES_TO),
        (
            _SCENARIO_EN,
            ContextType.SCENARIO,
            "explicit_scenario_marker",
            AttachmentRelation.APPLIES_TO,
        ),
        (
            _SCENARIO_ZH,
            ContextType.SCENARIO,
            "explicit_scenario_marker",
            AttachmentRelation.APPLIES_TO,
        ),
    ):
        add_matches(
            pattern,
            ctype,
            lambda m: {"value": m.groupdict().get("value", "").strip()},
            evidence,
            relation,
        )

    for pattern in (_SAMPLE_EN, _SAMPLE_ZH):
        for match in pattern.finditer(clause):
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "span": match.group(0),
                    "type": ContextType.SAMPLE_SCOPE,
                    "value": {"count": int(match.group("count")), "unit": match.group("unit")},
                    "relation": AttachmentRelation.APPLIES_TO,
                    "evidence": "explicit_sample_count_relation",
                    "reason": "context_created",
                }
            )

    # A lexical category noun without a relation is diagnostic-only, never a context.
    if re.search(r"(?i)\bmethod\b|方法", clause) and not any(
        item["type"] is ContextType.METHOD for item in candidates
    ):
        candidates.append(
            {
                "start": 0,
                "end": 0,
                "span": None,
                "type": ContextType.METHOD,
                "value": {},
                "relation": AttachmentRelation.UNKNOWN,
                "evidence": "category_noun_without_method_relation",
                "reason": "candidate_rejected_no_semantic_relation",
            }
        )
    return sorted(candidates, key=lambda item: (item["start"], item["end"], item["type"].value))


def project_source_context_graph(
    claim_text: str,
    propositions: list[dict[str, Any]],
    *,
    claim_ref: str = "claim",
    composition_groups: list[dict[str, Any]] | None = None,
) -> ContextGraph:
    """Build source contexts deterministically from source structure and spans."""
    spans = _locate_spans(claim_text, propositions)
    refs = [str(item.get("local_ref") or f"p{i}") for i, item in enumerate(propositions)]
    contexts: list[TruthConditionContext] = []
    diagnostics: list[str] = []
    scope_expansions: dict[str, tuple[str, ...]] = {
        str(group["local_ref"]): tuple(str(ref) for ref in group.get("member_refs", []))
        for group in composition_groups or []
        if isinstance(group.get("local_ref"), str) and isinstance(group.get("member_refs"), list)
    }
    for clause_start, clause_end, clause in _clauses(claim_text):
        clause_targets: list[str] = []
        for ref in refs:
            resolved_span = spans[ref]
            if (
                resolved_span is not None
                and resolved_span[0] >= clause_start
                and resolved_span[1] <= clause_end
            ):
                clause_targets.append(ref)
        explicit_claim_scope = bool(
            _CLAIM_SCOPE_EN.search(clause) or _CLAIM_SCOPE_ZH.search(clause)
        )
        group_scope: tuple[str, tuple[str, ...], str] | None = None
        for group in composition_groups or []:
            group_span = group.get("source_span")
            group_ref = group.get("local_ref")
            member_refs = group.get("member_refs")
            if (
                isinstance(group_span, str)
                and group_span
                and isinstance(group_ref, str)
                and group_ref
                and isinstance(member_refs, list)
                and member_refs
                and group_span in clause
            ):
                group_scope = (group_ref, tuple(str(ref) for ref in member_refs), group_span)
                break
        for candidate in _context_candidates(clause):
            if candidate["reason"] != "context_created":
                diagnostics.append(str(candidate["reason"]))
                continue
            candidate_start = clause_start + int(candidate["start"])
            candidate_end = clause_start + int(candidate["end"])
            local_span = candidate["span"]
            targets = clause_targets
            scope_target_kind = "proposition"
            if explicit_claim_scope and clause_targets:
                all_resolved = [ref for ref in refs if spans[ref] is not None]
                scope = ContextScope(
                    kind=ContextScopeKind.CLAIM,
                    evidence="explicit claim-wide scope marker in source",
                    source_start=clause_start,
                    source_end=clause_end,
                    target_refs=tuple(all_resolved),
                )
                targets = all_resolved
                attachment_status = AttachmentStatus.RESOLVED
                relation = candidate["relation"]
            elif _PROPOSITION_SET_SCOPE.search(clause) and len(clause_targets) > 1:
                scope = ContextScope(
                    kind=ContextScopeKind.PROPOSITION_SET,
                    evidence="explicit shared-set marker plus resolved proposition spans",
                    source_start=clause_start,
                    source_end=clause_end,
                    target_refs=tuple(clause_targets),
                )
                targets = clause_targets
                attachment_status = AttachmentStatus.RESOLVED
                relation = candidate["relation"]
            elif (
                group_scope is not None
                and clause_targets
                and set(group_scope[1]).issubset(set(clause_targets))
            ):
                group_ref, member_refs, group_span = group_scope
                scope = ContextScope(
                    kind=ContextScopeKind.ASSERTION_GROUP,
                    evidence="context occurs within explicit composition group source span",
                    source_start=clause_start + clause.find(group_span),
                    source_end=clause_start + clause.find(group_span) + len(group_span),
                    target_refs=(group_ref,),
                )
                targets = [group_ref]
                scope_target_kind = "assertion_group"
                attachment_status = AttachmentStatus.RESOLVED
                relation = candidate["relation"]
            elif not targets:
                scope = ContextScope(
                    kind=ContextScopeKind.UNKNOWN,
                    evidence="context detected but no proposition source span resolves in clause",
                    source_start=clause_start,
                    source_end=clause_end,
                )
                attachment_status = AttachmentStatus.UNKNOWN
                relation = AttachmentRelation.UNKNOWN
                diagnostics.append("attachment_unknown_no_resolved_proposition_span")
            elif len(targets) > 1:
                scope = ContextScope(
                    kind=ContextScopeKind.CLAUSE,
                    evidence="all resolved proposition spans lie in the same source clause",
                    source_start=clause_start,
                    source_end=clause_end,
                    target_refs=tuple(targets),
                )
                attachment_status = AttachmentStatus.RESOLVED
                relation = candidate["relation"]
            else:
                ref = targets[0]
                prop_span = spans[ref]
                assert prop_span is not None
                scope = ContextScope(
                    kind=ContextScopeKind.PROPOSITION,
                    evidence="one resolved proposition source span in source clause",
                    source_start=prop_span[0],
                    source_end=prop_span[1],
                    target_refs=(ref,),
                )
                attachment_status = AttachmentStatus.RESOLVED
                relation = candidate["relation"]
            semantic_value = dict(candidate["value"])
            signature = _signature(candidate["type"], semantic_value)
            index = len(contexts)
            context_ref = _local_ref("sc", index)
            attachments = tuple(
                ContextAttachment(
                    context_ref=context_ref,
                    target_ref=ref,
                    target_kind=scope_target_kind,
                    relation=relation,
                    status=attachment_status,
                    reason_code=(
                        "source_clause_targets_proposition"
                        if attachment_status is AttachmentStatus.RESOLVED
                        else "attachment_unknown"
                    ),
                )
                for ref in (targets if attachment_status is AttachmentStatus.RESOLVED else [None])
            )
            contexts.append(
                TruthConditionContext(
                    context_ref=context_ref,
                    semantic_signature=signature,
                    context_type=candidate["type"],
                    semantic_value=semantic_value,
                    lexical_value=str(candidate["span"]),
                    source_span=local_span,
                    source_start=candidate_start,
                    source_end=candidate_end,
                    materiality="MATERIAL",
                    scope=scope,
                    attachments=attachments,
                    detection_evidence=(str(candidate["evidence"]),),
                    reason_codes=(str(candidate["reason"]),),
                    origin=ContextOrigin.SOURCE,
                )
            )
    return ContextGraph(
        origin=ContextOrigin.SOURCE,
        claim_ref=claim_ref,
        contexts=tuple(contexts),
        diagnostics=tuple(diagnostics),
        scope_expansions=scope_expansions,
        source_detection_coverage=dict(_SOURCE_DETECTION_COVERAGE),
        context_presence=(
            ContextPresenceConclusion.CONTEXT_DETECTED
            if contexts
            else ContextPresenceConclusion.NO_CONTEXT_DETECTED
        ),
        source_detection_complete=all(
            status is ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT
            for status in _SOURCE_DETECTION_COVERAGE.values()
        ),
    )


_QUALIFIER_TYPE: dict[str, ContextType] = {
    "time": ContextType.TEMPORAL,
    "date": ContextType.TEMPORAL,
    "period": ContextType.TEMPORAL,
    "version": ContextType.VERSION,
    "location": ContextType.SCOPE,
    "scope": ContextType.SCOPE,
    "population": ContextType.SCOPE,
    "definition": ContextType.SCOPE,
    "configuration": ContextType.CONFIGURATION,
    "scenario": ContextType.SCENARIO,
    "condition": ContextType.CONDITION,
    "modality": ContextType.EPISTEMIC,
    "attribution": ContextType.EPISTEMIC,
    "measurement": ContextType.MEASUREMENT,
    "method": ContextType.METHOD,
    "sample_count": ContextType.SAMPLE_SCOPE,
    "unit": ContextType.OTHER,
    "currency": ContextType.OTHER,
    "comparison_target": ContextType.OTHER,
    "baseline": ContextType.OTHER,
    "other": ContextType.OTHER,
}


def project_output_context_graph(
    propositions: list[dict[str, Any]],
    *,
    claim_ref: str = "claim",
) -> ContextGraph:
    """Project only context information explicitly present in v3 output fields."""
    contexts: list[TruthConditionContext] = []
    diagnostics: list[str] = []

    def add(
        *,
        ref: str,
        ctype: ContextType,
        value: dict[str, Any],
        lexical: str | None,
        span: str | None,
        relation: AttachmentRelation,
        materiality: str,
        evidence: str,
    ) -> None:
        context_ref = _local_ref("oc", len(contexts))
        normalized_value = normalize_semantic_value(value)
        semantic_value = (
            normalized_value if isinstance(normalized_value, dict) else {"value": normalized_value}
        )
        signature = _signature(ctype, semantic_value)
        scope = ContextScope(
            kind=ContextScopeKind.PROPOSITION,
            evidence="v3 field explicitly belongs to this proposition",
            target_refs=(ref,),
        )
        contexts.append(
            TruthConditionContext(
                context_ref=context_ref,
                semantic_signature=signature,
                context_type=ctype,
                semantic_value=semantic_value,
                lexical_value=lexical,
                source_span=span,
                materiality=materiality,
                scope=scope,
                attachments=(
                    ContextAttachment(
                        context_ref=context_ref,
                        target_ref=ref,
                        target_kind="proposition",
                        relation=relation,
                        status=AttachmentStatus.RESOLVED,
                        reason_code="explicit_v3_field_projection",
                    ),
                ),
                detection_evidence=(evidence,),
                reason_codes=("projected_from_v3_output",),
                origin=ContextOrigin.OUTPUT,
            )
        )

    for index, proposition in enumerate(propositions):
        ref = str(proposition.get("local_ref") or f"p{index}")
        span = _proposition_span(proposition)
        qualifiers = proposition.get("material_qualifiers", proposition.get("qualifiers", []))
        if isinstance(qualifiers, list):
            for qualifier in qualifiers:
                if not isinstance(qualifier, dict):
                    diagnostics.append("output_qualifier_invalid_shape")
                    continue
                category = str(qualifier.get("category", "other")).lower()
                raw_value = qualifier.get("value")
                ctype, value = _qualifier_semantics(
                    category, str(qualifier.get("key", "")), raw_value
                )
                add(
                    ref=ref,
                    ctype=ctype,
                    value=value,
                    lexical=str(raw_value) if raw_value is not None else None,
                    span=qualifier.get("source_span")
                    if isinstance(qualifier.get("source_span"), str)
                    else span,
                    relation=(
                        AttachmentRelation.CONDITIONS
                        if ctype is ContextType.CONDITION
                        else AttachmentRelation.ATTRIBUTES
                        if ctype is ContextType.EPISTEMIC
                        else AttachmentRelation.APPLIES_TO
                    ),
                    materiality=str(qualifier.get("materiality", "UNKNOWN")).upper(),
                    evidence="v3 material_qualifiers field",
                )
        modality = proposition.get("modality")
        if isinstance(modality, str) and modality.strip():
            add(
                ref=ref,
                ctype=ContextType.EPISTEMIC,
                value={"operator": modality.strip().upper(), "agent": None},
                lexical=modality,
                span=span,
                relation=AttachmentRelation.ATTRIBUTES,
                materiality="MATERIAL",
                evidence="v3 modality field",
            )
        attribution = proposition.get("attribution")
        if isinstance(attribution, str) and attribution.strip():
            add(
                ref=ref,
                ctype=ContextType.EPISTEMIC,
                value={"operator": "ATTRIBUTED_TO", "agent": attribution.strip()},
                lexical=attribution,
                span=span,
                relation=AttachmentRelation.ATTRIBUTES,
                materiality="MATERIAL",
                evidence="v3 attribution field",
            )
        typed = proposition.get("typed_value", proposition.get("value"))
        if isinstance(typed, dict) and (
            "numeric" in typed or typed.get("kind") in {"quantity", "comparison", "range"}
        ):
            normalized = normalize_semantic_value(typed)
            diagnostics.append("typed_numeric_value_normalized_without_context_node")
            # Typed values stay in the proposition core; expose normalization only diagnostically.
            if not isinstance(normalized, dict):
                diagnostics.append("typed_value_normalization_unexpected_shape")
    return ContextGraph(
        origin=ContextOrigin.OUTPUT,
        claim_ref=claim_ref,
        contexts=tuple(contexts),
        diagnostics=tuple(diagnostics),
    )


def compare_context_graphs(
    source_graph: ContextGraph,
    output_graph: ContextGraph,
) -> tuple[ContextPreservationResult, ...]:
    """Compare source contexts against output contexts without carrier matching."""
    if not source_graph.contexts:
        coverage = source_graph.source_detection_coverage
        no_context_is_supported = bool(coverage) and all(
            status is ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT for status in coverage.values()
        )
        no_context_is_evaluable = bool(coverage) and all(
            status is not ContextFamilyCoverage.NOT_EVALUATED for status in coverage.values()
        )
        if no_context_is_supported or source_graph.source_detection_complete:
            outcome = ContextValidationOutcome.NOT_APPLICABLE
            reason = "bounded_contract_confirms_no_material_context"
        elif no_context_is_evaluable:
            outcome = ContextValidationOutcome.DETECTOR_LIMIT
            reason = "no_context_detected_but_detector_cannot_decide_absence"
        else:
            outcome = ContextValidationOutcome.DETECTOR_LIMIT
            reason = "context_family_not_evaluated"
        return (
            ContextPreservationResult(
                source_context_ref="",
                outcome=outcome,
                reason_codes=(reason,),
                source_presence_conclusion=(
                    ContextPresenceConclusion.NO_CONTEXT_EXISTS
                    if no_context_is_supported or source_graph.source_detection_complete
                    else ContextPresenceConclusion.DETECTOR_CANNOT_DECIDE
                ),
            ),
        )
    results: list[ContextPreservationResult] = []
    for source in source_graph.contexts:
        if any(item.status is AttachmentStatus.UNKNOWN for item in source.attachments):
            results.append(
                ContextPreservationResult(
                    source_context_ref=source.context_ref,
                    outcome=ContextValidationOutcome.ATTACHMENT_AMBIGUOUS,
                    reason_codes=("source_attachment_unknown",),
                )
            )
            continue
        semantic_matches = [
            output
            for output in output_graph.contexts
            if output.context_type is source.context_type
            and _canonical_values_compatible(source, output, output_graph.contexts)
        ]
        if not semantic_matches:
            same_type = any(
                item.context_type is source.context_type for item in output_graph.contexts
            )
            results.append(
                ContextPreservationResult(
                    source_context_ref=source.context_ref,
                    outcome=(
                        ContextValidationOutcome.SEMANTIC_MISMATCH
                        if same_type
                        else ContextValidationOutcome.MISSING
                    ),
                    reason_codes=(
                        ("same_context_type_but_different_semantic_value",)
                        if same_type
                        else ("no_output_context_of_source_type",)
                    ),
                )
            )
            continue
        if source.scope.kind is ContextScopeKind.UNKNOWN:
            results.append(
                ContextPreservationResult(
                    source_context_ref=source.context_ref,
                    output_context_refs=tuple(item.context_ref for item in semantic_matches),
                    outcome=ContextValidationOutcome.ATTACHMENT_AMBIGUOUS,
                    reason_codes=("source_scope_unknown",),
                )
            )
            continue
        source_targets = {
            member
            for target in source.scope.target_refs
            for member in source_graph.scope_expansions.get(target, (target,))
        }
        if any(
            edge.status is AttachmentStatus.UNKNOWN
            for item in semantic_matches
            for edge in item.attachments
        ):
            results.append(
                ContextPreservationResult(
                    source_context_ref=source.context_ref,
                    output_context_refs=tuple(item.context_ref for item in semantic_matches),
                    outcome=ContextValidationOutcome.ATTACHMENT_AMBIGUOUS,
                    reason_codes=("output_attachment_unknown",),
                )
            )
            continue
        output_targets = {
            member
            for item in semantic_matches
            for attachment in item.attachments
            if attachment.status is AttachmentStatus.RESOLVED and attachment.target_ref is not None
            for member in output_graph.scope_expansions.get(
                attachment.target_ref, (attachment.target_ref,)
            )
        }
        if source_targets and output_targets != source_targets:
            if output_targets and output_targets < source_targets:
                scope_outcome = ContextValidationOutcome.SCOPE_UNDEREXPANSION
                scope_reason = "output_target_set_is_proper_subset_of_source_targets"
            elif output_targets and source_targets < output_targets:
                scope_outcome = ContextValidationOutcome.SCOPE_OVEREXPANSION
                scope_reason = "output_target_set_is_proper_superset_of_source_targets"
            else:
                scope_outcome = ContextValidationOutcome.SCOPE_MISMATCH
                scope_reason = "source_and_output_target_sets_are_disjoint_or_unrelated"
            results.append(
                ContextPreservationResult(
                    source_context_ref=source.context_ref,
                    output_context_refs=tuple(item.context_ref for item in semantic_matches),
                    outcome=scope_outcome,
                    reason_codes=(scope_reason,),
                )
            )
            continue
        source_relations = {edge.relation for edge in source.attachments}
        output_relations = {edge.relation for item in semantic_matches for edge in item.attachments}
        if source.context_type is ContextType.EPISTEMIC:
            # Attribution/modality output fields are assertion-level context
            # carriers; legacy qualifier projection may have labeled the same
            # explicit context APPLIES_TO. Normalize only this typed relation.
            source_relations = {
                AttachmentRelation.ATTRIBUTES
                if relation is AttachmentRelation.APPLIES_TO
                else relation
                for relation in source_relations
            }
            output_relations = {
                AttachmentRelation.ATTRIBUTES
                if relation is AttachmentRelation.APPLIES_TO
                else relation
                for relation in output_relations
            }
        if source_relations != output_relations:
            results.append(
                ContextPreservationResult(
                    source_context_ref=source.context_ref,
                    output_context_refs=tuple(item.context_ref for item in semantic_matches),
                    outcome=ContextValidationOutcome.SEMANTIC_MISMATCH,
                    reason_codes=("attachment_relation_incompatible",),
                )
            )
            continue
        results.append(
            ContextPreservationResult(
                source_context_ref=source.context_ref,
                output_context_refs=tuple(item.context_ref for item in semantic_matches),
                outcome=ContextValidationOutcome.PRESERVED,
                reason_codes=("semantic_scope_and_attachment_compatible",),
            )
        )
    return tuple(results)


def build_verification_assertion_views(
    propositions: list[dict[str, Any]],
    output_graph: ContextGraph,
    *,
    eligibility_by_ref: dict[str, str] | None = None,
    source_graph: ContextGraph | None = None,
) -> tuple[VerificationAssertionView, ...]:
    """Derive assertion views; retain unknown material context rather than dropping it."""
    eligibility = eligibility_by_ref or {}
    result: list[VerificationAssertionView] = []
    for index, proposition in enumerate(propositions):
        ref = str(proposition.get("local_ref") or f"p{index}")
        attached = [
            context
            for context in output_graph.contexts
            if any(
                edge.target_ref == ref
                or ref in output_graph.scope_expansions.get(edge.target_ref or "", ())
                or (edge.status is AttachmentStatus.UNKNOWN and edge.target_ref is None)
                for edge in context.attachments
            )
            and context.materiality.upper() in {"MATERIAL", "UNKNOWN"}
        ]
        unknown = [
            context.context_ref
            for context in attached
            if context.scope.kind is ContextScopeKind.UNKNOWN
            or any(edge.status is AttachmentStatus.UNKNOWN for edge in context.attachments)
        ]
        source_attached = [
            context
            for context in (source_graph.contexts if source_graph is not None else ())
            if ref in context.scope.target_refs
            or any(edge.target_ref == ref for edge in context.attachments)
        ]
        source_coverage = source_graph.source_detection_coverage if source_graph is not None else {}
        limited_families = tuple(
            sorted(
                family
                for family, status in source_coverage.items()
                if status is ContextFamilyCoverage.PARTIAL_DETECTOR_LIMIT
            )
        )
        source_unknown = [
            context.context_ref
            for context in source_attached
            if context.scope.kind is ContextScopeKind.UNKNOWN
            or any(edge.status is AttachmentStatus.UNKNOWN for edge in context.attachments)
        ]
        source_coverage_complete = source_graph is not None and all(
            status is ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT
            for status in source_coverage.values()
        )
        core = {
            "subject": proposition.get("subject"),
            "predicate": proposition.get("predicate", proposition.get("relation")),
            "typed_value": normalize_semantic_value(
                proposition.get("typed_value", proposition.get("value"))
            ),
            "polarity": proposition.get("polarity"),
        }
        result.append(
            VerificationAssertionView(
                proposition_ref=ref,
                proposition_core=core,
                material_context_signatures=tuple(
                    sorted(
                        context.semantic_signature
                        for context in attached
                        if context.context_ref not in unknown
                    )
                ),
                material_context_unknown=bool(unknown or source_unknown),
                unknown_context_details=tuple(
                    sorted(set(unknown) | {f"source:{item}" for item in source_unknown})
                ),
                resolved_context_refs=tuple(
                    sorted(
                        context.context_ref
                        for context in attached
                        if context.context_ref not in unknown
                    )
                ),
                detector_limited_context_families=(
                    tuple(sorted({context.context_type.value for context in source_attached}))
                    if source_attached
                    and any(
                        source_coverage.get(context.context_type.value)
                        is ContextFamilyCoverage.PARTIAL_DETECTOR_LIMIT
                        for context in source_attached
                    )
                    else limited_families
                ),
                source_context_completeness=(
                    "NOT_PROVIDED"
                    if source_graph is None
                    else "SUPPORTED_FOR_CONTRACT"
                    if source_coverage_complete
                    else "DETECTOR_LIMIT"
                ),
                source_context_presence=(
                    source_graph.context_presence.value
                    if source_graph is not None
                    else ContextPresenceConclusion.DETECTOR_CANNOT_DECIDE.value
                ),
                composition_eligibility=eligibility.get(ref, "UNKNOWN"),
            )
        )
    return tuple(result)


def adapt_legacy_qualifier_inputs(
    obligations: list[dict[str, Any]],
    proposition_views: list[dict[str, Any]],
) -> LegacyContextAdapterResult:
    """Expose a bounded adapter summary; legacy objects are not a second truth source."""
    source_nodes: list[TruthConditionContext] = []
    output_nodes: list[TruthConditionContext] = []
    source_diagnostics = ["legacy_obligations_are_candidates_not_final_truth"]
    output_diagnostics = ["legacy_carriers_are_output_projection_candidates"]

    for index, obligation in enumerate(obligations):
        category = str(obligation.get("category", "other")).lower()
        context_type = _QUALIFIER_TYPE.get(category, ContextType.UNKNOWN)
        lexical = obligation.get("source_text")
        start = obligation.get("source_start")
        end = obligation.get("source_end")
        semantic_value = {
            "category": category,
            "value": obligation.get("normalized_value"),
        }
        context_ref = _local_ref("lc", index)
        source_nodes.append(
            TruthConditionContext(
                context_ref=context_ref,
                semantic_signature=_signature(context_type, semantic_value),
                context_type=context_type,
                semantic_value=semantic_value,
                lexical_value=str(lexical) if lexical is not None else None,
                source_span=str(lexical) if lexical is not None else None,
                source_start=start if isinstance(start, int) else None,
                source_end=end if isinstance(end, int) else None,
                materiality=str(obligation.get("materiality", "UNKNOWN")).upper(),
                scope=ContextScope(
                    kind=ContextScopeKind.UNKNOWN,
                    evidence="legacy obligation did not encode resolved scope targets",
                ),
                attachments=(
                    ContextAttachment(
                        context_ref=context_ref,
                        target_ref=None,
                        target_kind="unknown",
                        relation=AttachmentRelation.UNKNOWN,
                        status=AttachmentStatus.UNKNOWN,
                        reason_code="legacy_attachment_not_recoverable",
                    ),
                ),
                detection_evidence=(
                    str(obligation.get("detection_evidence", "legacy_obligation")),
                ),
                reason_codes=("legacy_obligation_adapted_scope_unknown",),
                origin=ContextOrigin.SOURCE,
            )
        )

    output_index = 0
    for view_index, view in enumerate(proposition_views):
        proposition_ref = str(
            view.get("proposition_ref") or f"p{view.get('proposition_index', view_index)}"
        )
        carriers = view.get("carriers", [])
        if not isinstance(carriers, list | tuple):
            output_diagnostics.append("legacy_proposition_view_carriers_invalid_shape")
            continue
        for carrier in carriers:
            if not isinstance(carrier, dict):
                output_diagnostics.append("legacy_semantic_carrier_invalid_shape")
                continue
            category = str(carrier.get("category", "other")).lower()
            context_type = _QUALIFIER_TYPE.get(category, ContextType.UNKNOWN)
            semantic_value = {
                "category": category,
                "value": carrier.get("normalized_value", carrier.get("value")),
            }
            context_ref = _local_ref("lo", output_index)
            output_index += 1
            span = carrier.get("source_span")
            output_nodes.append(
                TruthConditionContext(
                    context_ref=context_ref,
                    semantic_signature=_signature(context_type, semantic_value),
                    context_type=context_type,
                    semantic_value=semantic_value,
                    lexical_value=str(carrier.get("value"))
                    if carrier.get("value") is not None
                    else None,
                    source_span=span if isinstance(span, str) else None,
                    materiality="UNKNOWN",
                    scope=ContextScope(
                        kind=ContextScopeKind.PROPOSITION,
                        evidence="legacy carrier identifies one proposition view",
                        target_refs=(proposition_ref,),
                    ),
                    attachments=(
                        ContextAttachment(
                            context_ref=context_ref,
                            target_ref=proposition_ref,
                            target_kind="proposition",
                            relation=AttachmentRelation.APPLIES_TO,
                            status=AttachmentStatus.RESOLVED,
                            reason_code="legacy_carrier_proposition_ref_adapted",
                        ),
                    ),
                    detection_evidence=("legacy_semantic_carrier",),
                    reason_codes=("legacy_carrier_adapted_not_authoritative",),
                    origin=ContextOrigin.OUTPUT,
                )
            )

    return LegacyContextAdapterResult(
        source_graph=ContextGraph(
            origin=ContextOrigin.SOURCE,
            claim_ref="legacy-claim",
            contexts=tuple(source_nodes),
            diagnostics=tuple(source_diagnostics),
        ),
        output_graph=ContextGraph(
            origin=ContextOrigin.OUTPUT,
            claim_ref="legacy-claim",
            contexts=tuple(output_nodes),
            diagnostics=tuple(output_diagnostics),
        ),
    )
