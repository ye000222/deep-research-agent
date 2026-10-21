"""Strict, deterministic validation for non-persistent proposition decompositions."""

# Full-width punctuation is intentional in multilingual source-pattern boundaries.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    model_validator,
)

from app.domain.atomic_proposition import (
    ATOMIC_PROPOSITION_SCHEMA_VERSION,
    AtomicProposition,
    DecompositionMetadata,
    DecompositionMethod,
    DecompositionStatus,
    MaterialQualifier,
    PropositionPolarity,
    PropositionValue,
)
from app.domain.qualifier_preservation import (
    PreservationEvidence,
    PreservationEvidenceStrength,
    PropositionSemanticView,
    QualifierObligation,
    SemanticCarrier,
    SourceApplicability,
)

ATOMIC_DECOMPOSITION_PROMPT_VERSION = "atomic-decomposition.generic.v2"


class ShadowDecompositionLabel(StrEnum):
    SINGLE = "single_proposition"
    DECOMPOSED = "decomposed"
    UNRESOLVED = "unresolved_decomposition"


class QualifierCheckStatus(StrEnum):
    PRESERVED = "PRESERVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    POTENTIAL_LOSS = "POTENTIAL_LOSS"
    AMBIGUOUS = "AMBIGUOUS"


class _StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_JsonNumber = StrictInt | StrictFloat
_NonBlankString = Annotated[str, Field(min_length=1, pattern=r"[\s\S]*\S[\s\S]*")]


class _ScalarNumericPayload(_StrictContractModel):
    operator: Literal["exact", "approx", "lt", "lte", "gt", "gte"]
    value: _JsonNumber
    lower_bound: None = None
    upper_bound: None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    currency: str | None = Field(default=None, min_length=1, max_length=20)
    tolerance: _JsonNumber | None = Field(default=None, ge=0)
    measurement_basis: Literal["observed", "reported", "estimated", "attainable"] | None = None
    extremum: Literal["maximum", "minimum", "peak", "best"] | None = None
    change_type: Literal["absolute", "relative", "ratio"] | None = None


class _RangeNumericPayload(_StrictContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "For operator range, lower_bound must be ordered before upper_bound; "
                "runtime rejects reversed bounds."
            )
        },
    )

    operator: Literal["range"]
    value: None = None
    lower_bound: _JsonNumber
    upper_bound: _JsonNumber
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    currency: str | None = Field(default=None, min_length=1, max_length=20)
    tolerance: _JsonNumber | None = Field(default=None, ge=0)
    measurement_basis: Literal["observed", "reported", "estimated", "attainable"] | None = None
    extremum: Literal["maximum", "minimum", "peak", "best"] | None = None
    change_type: Literal["absolute", "relative", "ratio"] | None = None

    @model_validator(mode="after")
    def validate_order(self) -> _RangeNumericPayload:
        if self.lower_bound > self.upper_bound:
            raise ValueError("range bounds must be ordered")
        return self


_NumericPayload = Annotated[
    _ScalarNumericPayload | _RangeNumericPayload, Field(discriminator="operator")
]


class _TextValuePayload(_StrictContractModel):
    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=2_000)


class _EntityValuePayload(_StrictContractModel):
    kind: Literal["entity"]
    text: str = Field(min_length=1, max_length=2_000)


class _BooleanValuePayload(_StrictContractModel):
    kind: Literal["boolean"]
    boolean: StrictBool


class _QuantityValuePayload(_StrictContractModel):
    kind: Literal["quantity"]
    numeric: _NumericPayload


class _RangeValuePayload(_StrictContractModel):
    kind: Literal["range"]
    numeric: _NumericPayload


class _SetValuePayload(_StrictContractModel):
    kind: Literal["set"]
    members: tuple[_NonBlankString, ...] = Field(min_length=1)


class _ComparisonValuePayload(_StrictContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"comparison_metric_status": {"const": "unspecified"}},
                        "required": ["comparison_metric_status"],
                    },
                    "then": {"properties": {"comparison_metric": {"type": "null"}}},
                },
                {
                    "if": {
                        "properties": {
                            "comparison_metric_status": {"enum": ["explicit", "context_resolved"]}
                        },
                        "required": ["comparison_metric_status"],
                    },
                    "then": {
                        "properties": {"comparison_metric": {"type": "string", "minLength": 1}},
                        "required": ["comparison_metric"],
                    },
                },
            ]
        },
    )

    kind: Literal["comparison"]
    comparison_target: str = Field(min_length=1, max_length=1_000)
    comparison_metric: str | None = Field(default=None, min_length=1, max_length=500)
    comparison_metric_status: Literal["explicit", "context_resolved", "unspecified"]
    numeric: _NumericPayload | None = None

    @model_validator(mode="after")
    def validate_metric_resolution(self) -> _ComparisonValuePayload:
        if self.comparison_metric_status == "unspecified":
            if self.comparison_metric is not None:
                raise ValueError("unspecified comparison metric must not include a metric")
        elif not self.comparison_metric:
            raise ValueError("resolved comparison metric status requires a metric")
        return self


class _TemporalChangeValuePayload(_StrictContractModel):
    kind: Literal["temporal_change"]
    temporal_from: str = Field(min_length=1, max_length=200)
    temporal_to: str = Field(min_length=1, max_length=200)
    numeric: _NumericPayload


class _UnknownValuePayload(_StrictContractModel):
    kind: Literal["unknown"]


StructuredPropositionValue = Annotated[
    _TextValuePayload
    | _EntityValuePayload
    | _BooleanValuePayload
    | _QuantityValuePayload
    | _RangeValuePayload
    | _SetValuePayload
    | _ComparisonValuePayload
    | _TemporalChangeValuePayload
    | _UnknownValuePayload,
    Field(discriminator="kind"),
]


class PropositionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_text: str = Field(
        min_length=1,
        max_length=4_000,
        description=(
            "Truth-condition-complete human-readable rendering; material modality, "
            "attribution, time and conditions must be included when needed to interpret "
            "this proposition independently. Structured fields remain authoritative."
        ),
    )
    subject: str = Field(min_length=1, max_length=1_000)
    predicate: str = Field(min_length=1, max_length=1_000)
    typed_value: StructuredPropositionValue
    polarity: PropositionPolarity
    material_qualifiers: tuple[MaterialQualifier, ...] = ()
    modality: str | None = Field(default=None, max_length=100)
    attribution: str | None = Field(default=None, max_length=1_000)
    source_span: str | None = Field(default=None, max_length=2_000)
    confidence: Decimal | None = Field(default=None, ge=0, le=1)


class ShadowDecompositionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["atomic-proposition-schema-v2"]
    decomposition_status: ShadowDecompositionLabel
    propositions: tuple[PropositionPayload, ...] = Field(max_length=32)
    ambiguity_reason: str | None = Field(default=None, max_length=1_000)

    @classmethod
    def validate_shape(cls, response: ShadowDecompositionResponse) -> None:
        count = len(response.propositions)
        if response.decomposition_status is ShadowDecompositionLabel.SINGLE and count != 1:
            raise ValueError("single_proposition requires exactly one proposition")
        if response.decomposition_status is ShadowDecompositionLabel.DECOMPOSED and count < 2:
            raise ValueError("decomposed output requires at least two propositions")
        if (
            response.decomposition_status is not ShadowDecompositionLabel.UNRESOLVED
            and response.ambiguity_reason is not None
        ):
            raise ValueError("resolved output cannot contain an ambiguity reason")
        if response.decomposition_status is ShadowDecompositionLabel.UNRESOLVED and (
            count != 0 or not response.ambiguity_reason
        ):
            raise ValueError("unresolved output requires a reason and no propositions")


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)


class ValidationDiagnostic(BaseModel):
    """Safe field-level detail for explaining a validation rejection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_stage: str
    issue_code: str
    proposition_index: int | None = None
    field_path: str
    expected_contract: str
    observed_type: str
    observed_shape: dict[str, Any] = Field(default_factory=dict)


class QualifierCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    status: QualifierCheckStatus
    detected_text: str | None = None
    applicability: SourceApplicability = SourceApplicability.NO_SUPPORTED_SIGNAL
    evidence_strength: PreservationEvidenceStrength = PreservationEvidenceStrength.NONE
    obligation_count: int = Field(default=0, ge=0)
    reason_codes: tuple[str, ...] = ()
    preservation_evidence: tuple[PreservationEvidence, ...] = ()


class ValidatedDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: str
    decomposition_status: ShadowDecompositionLabel | None = None
    propositions: tuple[AtomicProposition, ...] = ()
    validation_issues: tuple[ValidationIssue, ...] = ()
    validation_diagnostics: tuple[ValidationDiagnostic, ...] = ()
    qualifier_checks: tuple[QualifierCheck, ...] = ()
    ambiguity_reason: str | None = None
    valid: bool


_QUALIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Unicode-aware structural signals. Python's \d intentionally accepts Unicode digits;
    # avoid ASCII word boundaries around CJK text.
    (
        "date_time",
        re.compile(
            r"(?:(?<!\d)\d{4}\s*[-/.]\s*\d{1,2}(?:\s*[-/.]\s*\d{1,2})?(?!\d)|"
            r"(?<!\d)\d{4}\s*年\s*(?:\d{1,2}\s*月?\s*(?:\d{1,2}\s*日?)?)?(?!\d)|"
            r"\b(?:in|during|by|since|until|throughout|as\s+of|fiscal\s+year|FY|"
            r"en|durante|desde|hasta)\s+\d{4}\b)",
            re.IGNORECASE,
        ),
    ),
    # A decimal is not, by itself, evidence of a version.  Require either an
    # explicit version/release marker, a product-attached identifier, or a
    # strongly version-like three-segment identifier (for example 1.2.3).
    # The named numeric groups let preservation compare the identifier value
    # without requiring the marker/product label to be duplicated in output.
    (
        "version",
        re.compile(
            r"(?:(?<![A-Za-z0-9_])(?:v(?:ersion)?|ver\.?|release|rel\.?)\s*[:=#-]?\s*"
            r"(?P<marked_version>\d+(?:\.\d+)+)"
            r"|版本(?:号|號)?\s*(?:为|為|是)?\s*(?P<cjk_version>\d+(?:\.\d+)+)"
            r"|バージョン\s*(?P<ja_version>\d+(?:\.\d+)+)"
            r"|versión\s*(?:de\s*)?(?P<es_version>\d+(?:\.\d+)+)"
            r"|(?<![A-Za-z0-9_])(?P<product_prefix>[A-Za-z][A-Za-z0-9_]*-)"
            r"(?P<product_version>\d+(?:\.\d+)+)"
            r"|(?<![\w.])(?P<multisegment_version>\d+(?:\.\d+){2,})"
            r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "currency",
        re.compile(
            r"(?:[$€£¥￥]\s?\d[\d,.]*"
            r"|\d[\d,.]*\s?(?:USD|EUR|GBP|JPY|CNY|RMB)\b"
            r"|\d[\d,.]*\s?(?:元|円|€|£)"
            r"|\d[\d,.]*\s?(?:万|億|千万|百万|十億)?(?:美元|欧元|英镑|日元|人民币))",
            re.I,
        ),
    ),
    (
        "unit",
        re.compile(
            r"(?<!\d)\d+(?:[.,]\d+)?\s?(?:%|\uFF05|percent(?:age)?|km|kilometers?|"
            r"m|meters?|cm|mm|kg|g|hours?|days?|years?|秒|分钟|小时|天|年|米|厘米|毫米|千克|公斤|克)"
            r"(?![A-Za-z0-9_])",
            re.I,
        ),
    ),
    (
        "comparison",
        re.compile(
            r"compared\s+(?:with|to)|versus|vs\.?|rather\s+than|\bthan\b|"
            r"比(?:较|起|之)|(?<=\S)比(?=\S{1,8}(?:高|低|大|小|多|少|优|優|快|慢))|"
            r"超过|超越|优于|高于|低于|相较|より|比較|上回る|下回る|"
            r"frente\s+a|más\s+que|menos\s+que",
            re.I,
        ),
    ),
    (
        "modality",
        re.compile(
            r"\b(?:may|might|could|possibly|likely|probably|expected|estimated|projected|"
            r"approximately|about|typically|usually|generally|often)\b|可能|预计|預計|预测|預測|"
            r"或许|或許|可望|見込み|可能性|予測|通常|一般に|一般而言|通常は|往往|经常|經常|"
            r"しばしば|通常|一般的に|suelen|suele|normalmente|generalmente|a\s+menudo|"
            r"por\s+lo\s+general|podría|podria|puede|posible|probablemente|se\s+espera",
            re.I,
        ),
    ),
    (
        "attribution",
        re.compile(
            r"\b(?:according\s+to|reported|claimed|stated|proposes|suggests|finds|concludes|"
            r"the\s+report\s+says|(?:the\s+)?(?:report|article|paper|study|source|author)\s+"
            r"(?:says|reports|claims|states|proposes|suggests|finds|concludes))\b|"
            r"(?:该|这)?(?:文章|论文|报告|研究|来源|作者|本文)(?:声称|宣称|指出|表示|发现|發現|表明|显示|預測|预测|认为|認為)|"
            r"(?:该|这)?(?:研究|论文|文章|报告|作者)(?:认为|認為|提出)|"
            r"(?:据|據)(?:该|該)?(?:报告|報告|文章|论文|研究)|"
            r"声称|宣称|据称|称|稱|报告称|報告稱|によると|報告では|述べる|報告する|"
            r"提案する|según|segun|afirma|informa|propone|sugiere|concluye|encuentra|"
            r"de\s+acuerdo\s+con",
            re.I,
        ),
    ),
    (
        "negation",
        re.compile(
            r"\b(?:not|never|no|cannot|can't|doesn't|didn't|without|neither)\b|"
            r"不(?!断|但|仅|僅|论|論|管|如|时|時|久)|未|没有|沒有|并非|並非|"
            r"无(?:法|法以|法可|法能)|無(?:法|法以|法可|法能)|ない|ません|ず|否定|"
            r"\bnunca\b|\bsin\b|\bno\b",
            re.I,
        ),
    ),
    (
        "condition",
        re.compile(
            r"\b(?:if|when|under|provided\s+that|in\s+the\s+case\s+of)\b|"
            r"在.{0,20}(?:条件|條件|输入|輸入|配置|设置|設定).{0,8}下|"
            r"条件下|條件下|若|如果|当.{0,12}时|"
            r"當.{0,12}時|条件为|條件為|場合|条件で|si\b|bajo\b|siempre\s+que",
            re.I,
        ),
    ),
    (
        "configuration",
        re.compile(
            r"\b(?:configured\s+with|using\s+setting|setting|mode|configuration|setup)\b|"
            r"配置|设定|設定|模式|设置|構成|configuración|configuracion|ajuste",
            re.I,
        ),
    ),
    (
        "sample_count",
        re.compile(
            r"(?<!\d)(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s?"
            r"(?:samples?|specimens?|trials?|runs?|items?|"
            r"muestras?|ejemplares?)(?![A-Za-z0-9_])|"
            r"(?<!\d)\d+\s?(?:个|個|张|張|例|件|份)\s?(?:参考|參考)?(?:样本|樣本|样品|樣品|試料)|"
            r"[一二三四五六七八九十百千万两]+\s?(?:个|個|张|張|例|件|份)?\s?"
            r"(?:参考|參考)?(?:样本|樣本|样品|樣品|試料)|"
            r"(?<!\d)\d+\s?(?:次|回)(?:试验|試驗|实验|實驗|运行|運行|测试|測試)?|"
            r"\b(?:sample\s+count|number\s+of\s+samples)\b|样本数|樣本數|サンプル数|"
            r"número\s+de\s+muestras",
            re.I,
        ),
    ),
    (
        "population",
        re.compile(
            r"\b(?:among|for|within)\s+(?:all\s+)?(?:users|patients|workers|households|"
            r"organizations|organisations|respondents)\b|所有|全部|仅限|僅限|用户群|用戶群|"
            r"患者|受访者|受訪者|対象者|全体|利用者|población|poblacion|usuarios|pacientes|"
            r"todos\s+los\s+hogares",
            re.I,
        ),
    ),
    (
        "measurement",
        re.compile(
            r"\b(?:measured|measurement|accuracy|precision|rate|score|average|median|mean|"
            r"parameter\s+count|param\s+count|gflops|tflops)\b|"
            r"(?<!\d)\d+(?:[.,]\d+)?\s*(?:million|billion|thousand|亿|億|万|萬|千)?"
            r"\s*(?:parameters?|参数)|"
            r"测量|測量|准确率|準確率|精度|比率|平均值|中央値|"
            r"(?![\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{0,3}概率)"
            r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{1,3}率|medición|medicion|"
            r"参数量|参数总量|计算量|浮点运算|precisión|precision|promedio|tasa",
            re.I,
        ),
    ),
    (
        "method",
        re.compile(
            r"\b(?:measured\s+by|using|uses|used|via|based\s+on|based\s+upon|by\s+means\s+of|"
            r"method|algorithm)\b|"
            r"采用(?=[^\uFF0C\u3002\uFF1B\uFF1A:]{0,12}(?:方法|模型|算法|協議|协议|技术|技術|方案|策略|提示|设置|設定|配置))|"
            r"採用(?=[^\uFF0C\u3002\uFF1B\uFF1A:]{0,12}(?:方法|模型|算法|協議|协议|技术|技術|方案|策略|提示|设置|設定|配置))|"
            r"通过|通過|使用|基于|基於|方法|算法|"
            r"を用いて|mediante|utilizando",
            re.I,
        ),
    ),
    (
        "scope",
        re.compile(
            r"\b(?:only|all|across|within|in\s+scope|limited\s+to)\b|"
            r"仅|僅|全部|范围|範圍|之内|之內|全て|全利用者|のみ|範囲|dentro\s+de|solo|todos",
            re.I,
        ),
    ),
)


def _is_cjk_or_kana(character: str) -> bool:
    """Return whether a character participates in a CJK lexical run."""
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF  # Hiragana and Katakana
        or 0x3400 <= codepoint <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= codepoint <= 0xFAFF  # CJK Compatibility Ideographs
    )


def _has_source_lexical_boundary(
    category: str,
    claim_text: str,
    match: re.Match[str],
    propositions: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> bool:
    """Reject a short CJK marker when it is embedded in a lexical run.

    CJK text has no whitespace token boundary.  The conservative rule covers
    single-character CJK attribution markers and numeric units embedded in hyphenated
    identifiers.  A marker preceded by another CJK/kana character is not independently
    evidenced as attribution.  A numeric unit after an identifier hyphen is not
    independently evidenced as a measurement.  These rules preserve clause-initial
    markers, script-transition forms such as ``R称``, and ordinary spaced units.

    A single CJK negation character inside a continuous CJK run is ambiguous without a
    tokenizer.  In that case it is accepted only when one attached proposition explicitly
    carries negated polarity / a negation qualifier.  This deliberately prefers precision
    over guessing from substrings such as ``不同`` or ``不规则``.  Multi-character negation
    constructions and clause-boundary occurrences retain their existing behavior.
    """
    start = match.start()
    marker = match.group(0)
    if category == "negation" and marker.casefold() == "no":
        end = match.end()
        if end < len(claim_text) and claim_text[end] in "-_":
            return False
    if category == "negation" and len(marker) == 1 and _is_cjk_or_kana(marker):
        before_is_cjk = start > 0 and _is_cjk_or_kana(claim_text[start - 1])
        end = match.end()
        after_is_cjk = end < len(claim_text) and _is_cjk_or_kana(claim_text[end])
        if before_is_cjk and after_is_cjk:
            supporting = [
                item
                for item in propositions
                if (
                    str(item.get("polarity", "")).casefold() == "negated"
                    or any(
                        isinstance(qualifier, dict)
                        and str(qualifier.get("category", "")).casefold() == "negation"
                        for qualifier in item.get("material_qualifiers", ())
                        if isinstance(item.get("material_qualifiers", ()), list | tuple)
                    )
                )
            ]
            if len(supporting) != 1:
                return False
    if category == "attribution" and len(marker) == 1 and _is_cjk_or_kana(marker):
        return start == 0 or not _is_cjk_or_kana(claim_text[start - 1])
    if category == "unit" and start >= 2 and claim_text[start - 1] == "-":
        return not claim_text[start - 2].isalnum() and claim_text[start - 2] != "_"
    return True


_QUALIFIER_FIELD_CATEGORIES: dict[str, frozenset[str]] = {
    "date_time": frozenset({"time", "date", "period"}),
    "version": frozenset({"version"}),
    "currency": frozenset({"currency"}),
    "unit": frozenset({"unit", "measurement"}),
    "comparison": frozenset({"comparison_target", "comparison"}),
    "modality": frozenset({"modality"}),
    "attribution": frozenset({"attribution"}),
    "negation": frozenset({"polarity", "negation"}),
    "condition": frozenset({"condition", "scenario"}),
    "configuration": frozenset({"configuration", "method"}),
    "sample_count": frozenset({"measurement", "configuration", "population"}),
    "population": frozenset({"population", "scope"}),
    "measurement": frozenset({"measurement", "unit"}),
    "method": frozenset({"method", "configuration"}),
    "scope": frozenset({"scope", "population", "location"}),
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _source_spans(item: dict[str, Any]) -> tuple[str, ...]:
    spans: list[str] = []
    top_level = item.get("source_span")
    if isinstance(top_level, str) and top_level:
        spans.append(top_level)
    qualifiers = item.get("material_qualifiers")
    if isinstance(qualifiers, list | tuple):
        spans.extend(
            str(qualifier["source_span"])
            for qualifier in qualifiers
            if isinstance(qualifier, dict)
            and isinstance(qualifier.get("source_span"), str)
            and qualifier["source_span"]
        )
    return tuple(dict.fromkeys(spans))


def inspect_explicit_qualifiers(
    claim_text: str, response: ShadowDecompositionResponse
) -> tuple[QualifierCheck, ...]:
    return inspect_qualifier_fields(
        claim_text,
        tuple(item.model_dump(mode="python") for item in response.propositions),
    )


def inspect_qualifier_fields(
    claim_text: str, propositions: tuple[dict[str, Any], ...] | list[dict[str, Any]]
) -> tuple[QualifierCheck, ...]:
    """Evaluate source obligations against proposition semantics, not field location.

    Detection remains source-only. Proposition fields can establish preservation but cannot
    create an obligation. Lexical coincidence is weak evidence and cannot override an
    incompatible attachment or value.
    """
    checks: list[QualifierCheck] = []
    views = tuple(
        _proposition_semantic_view(index, item) for index, item in enumerate(propositions)
    )
    for category, pattern in _QUALIFIER_PATTERNS:
        matches = [
            match
            for match in pattern.finditer(claim_text)
            if _has_source_lexical_boundary(category, claim_text, match, propositions)
        ]
        if category == "method":
            matches = [
                match
                for match in matches
                if _method_source_relation_supported(claim_text, match)
                and not _is_comparison_target_method_noun(claim_text, match)
            ]
        if category == "configuration":
            matches = [
                match
                for match in matches
                if _configuration_source_relation_supported(claim_text, match)
            ]
        if category == "version" and matches:
            date_pattern = dict(_QUALIFIER_PATTERNS)["date_time"]
            quantity_patterns = (
                dict(_QUALIFIER_PATTERNS)["unit"],
                dict(_QUALIFIER_PATTERNS)["currency"],
            )
            excluded_spans = [match.span() for match in date_pattern.finditer(claim_text)]
            excluded_spans.extend(
                match.span()
                for quantity_pattern in quantity_patterns
                for match in quantity_pattern.finditer(claim_text)
            )
            matches = [
                match
                for match in matches
                if not any(
                    _spans_overlap(_version_value_span(match), excluded)
                    for excluded in excluded_spans
                )
            ]
        if category == "unit":
            date_pattern = dict(_QUALIFIER_PATTERNS)["date_time"]
            date_spans = [match.span() for match in date_pattern.finditer(claim_text)]
            matches = [
                match
                for match in matches
                if not any(_spans_overlap(match.span(), date_span) for date_span in date_spans)
            ]
        if not matches:
            coverage_limited = _has_unsupported_letter_script(claim_text)
            checks.append(
                QualifierCheck(
                    category=category,
                    status=QualifierCheckStatus.NOT_APPLICABLE,
                    applicability=(
                        SourceApplicability.DETECTOR_COVERAGE_LIMIT
                        if coverage_limited
                        else SourceApplicability.NO_SUPPORTED_SIGNAL
                    ),
                    reason_codes=(
                        ("SOURCE_SCRIPT_OUTSIDE_DETECTOR_COVERAGE",)
                        if coverage_limited
                        else ("NO_SUPPORTED_SOURCE_SIGNAL",)
                    ),
                )
            )
            continue
        obligations = tuple(_qualifier_obligation(claim_text, category, match) for match in matches)
        decisions: list[tuple[QualifierCheckStatus, PreservationEvidenceStrength, str]] = []
        evidence_records: list[PreservationEvidence] = []
        for match in matches:
            obligation = _qualifier_obligation(claim_text, category, match)
            attached = [
                view
                for view in views
                if any(
                    _source_span_covers_obligation(span, obligation, claim_text)
                    for span in view.source_spans
                )
            ]
            candidates = attached if attached else list(views)
            compatible = [
                (view, _view_evidence_strength(obligation, view, claim_text=claim_text))
                for view in candidates
            ]
            compatible = [
                (view, strength)
                for view, strength in compatible
                if strength
                in {
                    PreservationEvidenceStrength.STRONG,
                    PreservationEvidenceStrength.MEDIUM,
                    PreservationEvidenceStrength.WEAK,
                }
            ]
            if len(compatible) > 1 and category in {
                "unit",
                "currency",
                "measurement",
                "sample_count",
            }:
                context_tokens = _semantic_tokens(obligation.attachment_context) - _semantic_tokens(
                    obligation.source_text
                )
                contextual = [
                    item
                    for item in compatible
                    if any(
                        context_tokens & _semantic_tokens(carrier.value)
                        for carrier in item[0].carriers
                        if carrier.category
                        in {
                            "subject",
                            "predicate",
                            "scope",
                            "population",
                            "configuration",
                            "method",
                            "condition",
                            "measurement",
                            "location",
                        }
                    )
                ]
                if contextual:
                    compatible = contextual
            if len(compatible) == 1:
                only_view, only_strength = compatible[0]
                if (
                    only_strength is PreservationEvidenceStrength.WEAK
                    and not attached
                    and len(views) > 1
                ):
                    reason = "WEAK_EVIDENCE_CANNOT_ESTABLISH_MULTI_PROPOSITION_ATTACHMENT"
                    decisions.append((QualifierCheckStatus.AMBIGUOUS, only_strength, reason))
                    evidence_records.append(
                        PreservationEvidence(
                            obligation=obligation,
                            evidence_strength=only_strength,
                            semantic_compatible=True,
                            attachment_compatible=False,
                            carrier_paths=("proposition_text",),
                            reason_code=reason,
                        )
                    )
                else:
                    reason = "SEMANTIC_AND_ATTACHMENT_MATCH"
                    decisions.append((QualifierCheckStatus.PRESERVED, only_strength, reason))
                    paths = tuple(
                        carrier.field_path
                        for carrier in only_view.carriers
                        if _carrier_preserves_obligation(obligation, carrier)
                    )
                    evidence_records.append(
                        PreservationEvidence(
                            obligation=obligation,
                            proposition_index=only_view.proposition_index,
                            evidence_strength=only_strength,
                            semantic_compatible=True,
                            attachment_compatible=bool(attached) or len(views) == 1,
                            carrier_paths=paths,
                            reason_code=reason,
                        )
                    )
            elif len(compatible) > 1 and (
                _shared_sentence_carrier(obligation.category, compatible)
                or _shared_group_comparison_carrier(obligation, compatible)
                or _shared_group_method_carrier(obligation, compatible, claim_text)
            ):
                shared_strength = min((item[1] for item in compatible), key=_strength_rank)
                reason = "SHARED_SENTENCE_LEVEL_SEMANTIC"
                decisions.append((QualifierCheckStatus.PRESERVED, shared_strength, reason))
                shared_paths = tuple(
                    f"p{view.proposition_index}.{carrier.field_path}"
                    for view, _ in compatible
                    for carrier in view.carriers
                    if (not carrier.source_span or _source_span_covers_obligation(
                        carrier.source_span, obligation, claim_text
                    ))
                    and _carrier_preserves_obligation(obligation, carrier)
                )
                evidence_records.append(
                    PreservationEvidence(
                        obligation=obligation,
                        evidence_strength=shared_strength,
                        semantic_compatible=True,
                        attachment_compatible=True,
                        carrier_paths=shared_paths,
                        reason_code=reason,
                    )
                )
            elif len(compatible) > 1:
                reason = "MULTIPLE_COMPATIBLE_ATTACHMENTS"
                decisions.append(
                    (QualifierCheckStatus.AMBIGUOUS, PreservationEvidenceStrength.NONE, reason)
                )
                evidence_records.append(
                    PreservationEvidence(
                        obligation=obligation,
                        evidence_strength=PreservationEvidenceStrength.NONE,
                        semantic_compatible=True,
                        attachment_compatible=False,
                        carrier_paths=tuple(f"p{view.proposition_index}" for view, _ in compatible),
                        reason_code=reason,
                    )
                )
            elif attached:
                reason = "ATTACHED_VIEW_MISSING_COMPATIBLE_CARRIER"
                decisions.append(
                    (QualifierCheckStatus.POTENTIAL_LOSS, PreservationEvidenceStrength.NONE, reason)
                )
                evidence_records.append(
                    PreservationEvidence(
                        obligation=obligation,
                        evidence_strength=PreservationEvidenceStrength.NONE,
                        semantic_compatible=False,
                        attachment_compatible=True,
                        carrier_paths=(),
                        reason_code=reason,
                    )
                )
            elif len(views) > 1:
                reason = "ATTACHMENT_NOT_ESTABLISHED"
                decisions.append(
                    (QualifierCheckStatus.AMBIGUOUS, PreservationEvidenceStrength.NONE, reason)
                )
                evidence_records.append(
                    PreservationEvidence(
                        obligation=obligation,
                        evidence_strength=PreservationEvidenceStrength.NONE,
                        semantic_compatible=False,
                        attachment_compatible=False,
                        carrier_paths=(),
                        reason_code=reason,
                    )
                )
            else:
                reason = "NO_COMPATIBLE_SEMANTIC_CARRIER"
                decisions.append(
                    (QualifierCheckStatus.POTENTIAL_LOSS, PreservationEvidenceStrength.NONE, reason)
                )
                evidence_records.append(
                    PreservationEvidence(
                        obligation=obligation,
                        evidence_strength=PreservationEvidenceStrength.NONE,
                        semantic_compatible=False,
                        attachment_compatible=len(views) == 1,
                        carrier_paths=(),
                        reason_code=reason,
                    )
                )
        match_states = [item[0] for item in decisions]
        status = (
            QualifierCheckStatus.AMBIGUOUS
            if QualifierCheckStatus.AMBIGUOUS in match_states
            else QualifierCheckStatus.POTENTIAL_LOSS
            if QualifierCheckStatus.POTENTIAL_LOSS in match_states
            else QualifierCheckStatus.PRESERVED
        )
        weakest_preserved = min(
            (item[1] for item in decisions if item[0] is QualifierCheckStatus.PRESERVED),
            key=_strength_rank,
            default=PreservationEvidenceStrength.NONE,
        )
        checks.append(
            QualifierCheck(
                category=category,
                status=status,
                detected_text="; ".join(dict.fromkeys(item.source_text for item in obligations))[
                    :500
                ],
                applicability=SourceApplicability.SUPPORTED_SIGNAL,
                evidence_strength=weakest_preserved,
                obligation_count=len(obligations),
                reason_codes=tuple(dict.fromkeys(item[2] for item in decisions)),
                preservation_evidence=tuple(evidence_records),
            )
        )
    return tuple(checks)


def _qualifier_obligation(
    claim_text: str, category: str, match: re.Match[str]
) -> QualifierObligation:
    start, end = match.span()
    if category == "version":
        # The version regex may include an attached product identifier in its
        # full match; obligation provenance must point at the version value.
        start, end = _version_value_span(match)
    left, right = _sentence_bounds(claim_text, start, end)
    separators = ("，", ",", "、", "和", "与", "及", " and ", " or ")
    comma_left = max(claim_text.rfind(mark, left, start) for mark in separators)
    comma_positions = [claim_text.find(mark, end, right) for mark in separators]
    comma_right = min((position for position in comma_positions if position >= 0), default=right)
    context_radius = 96 if category in {"unit", "currency"} else 48
    context_start = max(left, comma_left + 1, start - context_radius)
    context_end = min(right, comma_right, end + context_radius)
    source_text = _qualifier_match_value(category, match)
    return QualifierObligation(
        category=category,
        normalized_value=_normalized(source_text),
        source_text=source_text,
        source_start=start,
        source_end=end,
        attachment_context=claim_text[context_start:context_end],
        detection_evidence=f"source_pattern:{category}",
    )


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Find sentence bounds without mistaking decimal points for terminators."""
    boundaries = list(re.finditer(r"[。!?；;\n]|(?<!\d)\.(?!\d)", text))
    left = max((item.end() for item in boundaries if item.end() <= start), default=0)
    right = min((item.start() for item in boundaries if item.start() >= end), default=len(text))
    return left, right


def _proposition_semantic_view(index: int, item: dict[str, Any]) -> PropositionSemanticView:
    carriers: list[SemanticCarrier] = []
    proposition_ref = str(item.get("local_ref")) if item.get("local_ref") is not None else None
    qualifiers = item.get("material_qualifiers")
    qualifiers = qualifiers if isinstance(qualifiers, list | tuple) else ()
    for q_index, qualifier in enumerate(qualifiers):
        if not isinstance(qualifier, dict):
            continue
        category = str(qualifier.get("category", "")).casefold()
        value = str(qualifier.get("value") or "")
        if category and value:
            carriers.append(
                _semantic_carrier(
                    category,
                    f"material_qualifiers[{q_index}].value",
                    value,
                    PreservationEvidenceStrength.STRONG,
                    str(qualifier.get("source_span")) if qualifier.get("source_span") else None,
                    proposition_ref=proposition_ref,
                    semantic_hint=str(qualifier.get("key", "")),
                )
            )

    for field, category in (
        ("subject", "subject"),
        ("predicate", "predicate"),
        ("modality", "modality"),
        ("attribution", "attribution"),
    ):
        field_value = item.get(field)
        if isinstance(field_value, str) and field_value.strip():
            strength = (
                PreservationEvidenceStrength.STRONG
                if field in {"modality", "attribution"}
                else PreservationEvidenceStrength.MEDIUM
            )
            carriers.append(
                _semantic_carrier(
                    category, field, field_value, strength, proposition_ref=proposition_ref
                )
            )
    polarity = item.get("polarity")
    if isinstance(polarity, str) and polarity:
        carriers.append(
            _semantic_carrier(
                "polarity",
                "polarity",
                polarity,
                PreservationEvidenceStrength.STRONG,
                proposition_ref=proposition_ref,
            )
        )

    typed = item.get("typed_value")
    typed = typed if isinstance(typed, dict) else {}
    kind = str(typed.get("kind", "")).casefold()
    if kind:
        carriers.append(
            _semantic_carrier(
                "typed_value",
                "typed_value.kind",
                kind,
                PreservationEvidenceStrength.STRONG,
                proposition_ref=proposition_ref,
            )
        )
    for key in ("text", "comparison_target", "comparison_metric", "temporal_from", "temporal_to"):
        typed_field_value = typed.get(key)
        if isinstance(typed_field_value, str) and typed_field_value:
            category = (
                "comparison"
                if key == "comparison_target"
                else "date_time"
                if key.startswith("temporal_")
                else "typed_value"
            )
            carriers.append(
                _semantic_carrier(
                    category,
                    f"typed_value.{key}",
                    typed_field_value,
                    PreservationEvidenceStrength.STRONG,
                    proposition_ref=proposition_ref,
                )
            )
    numeric = typed.get("numeric")
    if isinstance(numeric, dict):
        carriers.append(
            _semantic_carrier(
                "measurement",
                "typed_value.numeric",
                json.dumps(numeric, ensure_ascii=False, sort_keys=True),
                PreservationEvidenceStrength.STRONG,
                proposition_ref=proposition_ref,
            )
        )
        for key, category in (
            ("unit", "unit"),
            ("currency", "currency"),
            ("measurement_basis", "measurement"),
            ("extremum", "measurement"),
            ("change_type", "measurement"),
        ):
            numeric_field_value = numeric.get(key)
            if numeric_field_value is not None:
                carriers.append(
                    _semantic_carrier(
                        category,
                        f"typed_value.numeric.{key}",
                        str(numeric_field_value),
                        PreservationEvidenceStrength.STRONG,
                        proposition_ref=proposition_ref,
                    )
                )
    if kind == "comparison":
        carriers.append(
            _semantic_carrier(
                "comparison",
                "typed_value.kind",
                kind,
                PreservationEvidenceStrength.STRONG,
                proposition_ref=proposition_ref,
            )
        )
    for field in ("subject", "predicate", "proposition_text"):
        text_field_value = item.get(field)
        if isinstance(text_field_value, str) and text_field_value.strip():
            strength = (
                PreservationEvidenceStrength.WEAK
                if field == "proposition_text"
                else PreservationEvidenceStrength.MEDIUM
            )
            carriers.append(
                _semantic_carrier(
                    "text",
                    field,
                    text_field_value,
                    strength,
                    proposition_ref=proposition_ref,
                    carrier_kind="weak_text" if field == "proposition_text" else "semantic_text",
                )
            )
    # Keep legal typed-value leaves available as carriers even when a new typed
    # field is introduced. Explicit projections above retain their category
    # semantics; this pass is deliberately field-generic.
    for path, leaf, leaf_kind in _structured_string_leaves(item.get("typed_value"), "typed_value"):
        if not leaf.strip() or any(carrier.field_path == path for carrier in carriers):
            continue
        carriers.append(
            _semantic_carrier(
                "typed_value",
                path,
                leaf,
                PreservationEvidenceStrength.STRONG,
                proposition_ref=proposition_ref,
                semantic_hint=path.rsplit(".", 1)[-1],
                carrier_kind=leaf_kind,
            )
        )
    return PropositionSemanticView(
        proposition_index=index,
        proposition_ref=proposition_ref,
        carriers=tuple(carriers),
        source_spans=_source_spans(item),
    )


def _semantic_carrier(
    category: str,
    path: str,
    value: str,
    strength: PreservationEvidenceStrength,
    source_span: str | None = None,
    *,
    proposition_ref: str | None = None,
    semantic_hint: str = "",
    carrier_kind: str = "structured",
) -> SemanticCarrier:
    semantic_role, related_roles = _semantic_roles(category, path, semantic_hint, value)
    return SemanticCarrier(
        category=category,
        semantic_role=semantic_role,
        field_path=path,
        value=value,
        normalized_value=_normalized(value),
        evidence_strength=strength,
        carrier_kind=carrier_kind,
        proposition_ref=proposition_ref,
        related_roles=related_roles,
        source_span=source_span,
    )


def _structured_string_leaves(value: Any, path: str) -> list[tuple[str, str, str]]:
    """Flatten structured semantic leaves without depending on field names."""
    leaves: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            leaves.extend(_structured_string_leaves(nested, f"{path}.{key}"))
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            leaves.extend(_structured_string_leaves(nested, f"{path}[{index}]"))
    elif isinstance(value, str) and value.strip():
        leaves.append((path, value, "structured_leaf"))
    return leaves


def _semantic_roles(
    category: str, field_path: str, semantic_hint: str, value: str
) -> tuple[str, tuple[str, ...]]:
    """Resolve semantic roles from both declared category and carrier content.

    Storage location is descriptive provenance only. Related roles require
    generic role evidence in a key/path or in the semantic payload.
    """
    role_aliases = {
        "comparison_target": "comparison_target",
        "comparison_metric": "comparison_metric",
        "comparison": "comparison",
        "method": "method",
        "configuration": "configuration",
        "condition": "condition",
        "scenario": "condition",
        "scope": "scope",
        "population": "population",
        "location": "location",
        "measurement": "measurement",
        "unit": "unit",
        "currency": "currency",
        "time": "date_time",
        "date": "date_time",
        "period": "date_time",
        "date_time": "date_time",
        "modality": "modality",
        "attribution": "attribution",
        "polarity": "polarity",
        "negation": "negation",
        "sample_count": "sample_count",
        "subject": "subject",
        "predicate": "predicate",
        "text": "text",
        "other": "other",
        "typed_value": "typed_value",
    }
    path_leaf = field_path.rsplit(".", 1)[-1].rsplit("]", 1)[-1].lstrip(".")
    path_role = role_aliases.get(path_leaf.casefold())
    primary = role_aliases.get(category.casefold(), category.casefold() or "unknown")
    if path_role and (category == "comparison" or category == "typed_value"):
        primary = path_role
    related: set[str] = set()
    hint_text = _normalized(semantic_hint)
    role_text = _normalized(" ".join((semantic_hint, field_path, value)))
    if primary == "other" or category == "other":
        if re.search(
            r"configuration|setting|setup|mode|配置|设定|設定|模式|设置|構成",
            hint_text,
            re.I,
        ) or _has_configuration_relation_phrase(value):
            related.add("configuration")
        if re.search(r"method|algorithm|方法|算法|手法", hint_text, re.I) or (
            _has_method_relation_phrase(value)
        ):
            related.add("method")
        if re.search(r"condition|scenario|under|条件|情况下|場合", role_text, re.I):
            related.add("condition")
        if re.search(r"scope|population|location|范围|範圍|人群|地区|地域", role_text, re.I):
            related.add("scope")
    if _has_method_relation_phrase(value):
        related.add("method")
    if primary in {"subject", "predicate"} and re.search(
        r"configuration|setting|setup|mode|配置|设定|設定|模式|设置|構成",
        value,
        re.I,
    ):
        related.add("configuration")
    if primary in {"condition", "typed_value"} and re.search(
        r"configuration|setting|setup|mode|配置|设定|設定|模式|设置|構成",
        value,
        re.I,
    ):
        related.add("configuration")
    return primary, tuple(sorted(role for role in related if role != primary))


def _has_method_relation_phrase(value: str) -> bool:
    """Whether text expresses an instrumental relation, not just a category noun."""
    return bool(
        re.search(
            r"\b(?:using|uses|used|via|based\s+on|based\s+upon|by\s+means\s+of|"
            r"measured\s+by|method\s+of|algorithm\s+for)\b|"
            r"采用|採用|通过|通過|使用|基于|基於|利用|依靠|借助|を用いて|mediante|utilizando",
            value,
            re.I,
        )
    )


def _has_configuration_relation_phrase(value: str) -> bool:
    return bool(
        re.search(
            r"\b(?:configured\s+with|configured\s+under|using\s+(?:the\s+)?setting|"
            r"under\s+(?:the\s+)?configuration|in\s+(?:the\s+)?(?:mode|configuration)|"
            r"set\s+to)\b|配置为|設定為|设定为|设置为|配置下|设定下|設定下",
            value,
            re.I,
        )
    )


def _configuration_source_relation_supported(
    claim_text: str, match: re.Match[str]
) -> bool:
    """Exclude anaphoric configuration nouns unless the source relates a setting."""
    sentence_start, sentence_end = _sentence_bounds(claim_text, match.start(), match.end())
    before = claim_text[max(sentence_start, match.start() - 18) : match.start()]
    after = claim_text[match.end() : min(sentence_end, match.end() + 48)]
    if re.search(
        r"(?:this|that|same|such|the\s+same)\s+(?:configuration|setting|mode)\s*$|"
        r"(?:该|此|同一|上述|这种|该项)\s*(?:配置|設定|设定|设置|模式)\s*$",
        before,
        re.I,
    ):
        return _has_configuration_relation_phrase(before + match.group(0) + after)
    return True


def _method_source_relation_supported(claim_text: str, match: re.Match[str]) -> bool:
    """Require a source-side method relation, including asserted method subjects."""
    marker = match.group(0)
    if not _is_method_noun(marker):
        return True
    left, _ = _sentence_bounds(claim_text, match.start(), match.end())
    separators = [
        claim_text.rfind(mark, left, match.start())
        for mark in (",", "，", ":", "：", ";", "；", "、")
    ]
    clause_start = max((position + 1 for position in separators if position >= 0), default=left)
    preceding = claim_text[max(clause_start, match.start() - 80) : match.start()]
    if re.search(
        r"\b(?:using|uses|used|via|based\s+on|based\s+upon|by\s+means\s+of|"
        r"measured\s+by|method\s+of|algorithm\s+for)\b|"
        r"采用|採用|通过|通過|使用|基于|基於|利用|依靠|借助|を用いて|mediante|utilizando",
        preceding,
        re.I,
    ):
        return True

    # A method noun can itself be the asserted subject (e.g. "traditional
    # methods are limited by ..."). This is a source relation, unlike a bare
    # occurrence in a comparison target. Keep the predicate window clause-local.
    following = claim_text[match.end() : match.end() + 80]
    return bool(
        re.search(
            r"受限于|受限於|依赖于|依賴於|存在.{0,8}(?:不足|限制|缺陷)|"
            r"(?:能力|效果|性能|准确率|準確率|泛化性|可解释性|可解釋性).{0,8}(?:不足|有限|较差|較差)|"
            r"(?:is|are|was|were)\s+(?:limited|constrained|restricted|inferior|less\s+effective)\b",
            following,
            re.I,
        )
    )


def _carrier_has_role(carrier: SemanticCarrier, role: str) -> bool:
    aliases = {
        "date_time": {"date_time", "time", "date", "period"},
        "comparison": {"comparison", "comparison_target", "comparison_metric"},
    }
    return bool(
        {role, *aliases.get(role, set())}
        & {carrier.semantic_role, *carrier.related_roles}
    )


def _view_evidence_strength(
    obligation: QualifierObligation,
    view: PropositionSemanticView,
    *,
    claim_text: str = "",
) -> PreservationEvidenceStrength:
    category = obligation.category
    explicit_categories = _QUALIFIER_FIELD_CATEGORIES.get(category, frozenset({category}))
    explicit_carriers = [
        carrier
        for carrier in view.carriers
        if carrier.field_path.startswith("material_qualifiers[")
        and (
            carrier.category in explicit_categories | {category}
            or _carrier_has_role(carrier, category)
        )
        and carrier.source_span
        and _source_span_covers_obligation(carrier.source_span, obligation, claim_text)
    ]
    if explicit_carriers and any(
        not _carrier_preserves_obligation(obligation, carrier) for carrier in explicit_carriers
    ):
        # A conflicting explicit value is evidence against preservation even if a
        # weaker carrier happens to repeat the source token.
        return PreservationEvidenceStrength.NONE
    if category == "attribution":
        explicit_attribution = [
            carrier for carrier in view.carriers if carrier.field_path == "attribution"
        ]
        if explicit_attribution and any(
            not _carrier_preserves_obligation(obligation, carrier)
            for carrier in explicit_attribution
        ):
            return PreservationEvidenceStrength.NONE
    best = PreservationEvidenceStrength.NONE
    for carrier in view.carriers:
        if category == "method" and carrier.field_path == "proposition_text":
            # Full text can corroborate a relation, but cannot by itself
            # establish that a method marker has the correct semantic role.
            continue
        if carrier.source_span and not _source_span_covers_obligation(
            carrier.source_span, obligation, claim_text
        ):
            continue
        if _carrier_preserves_obligation(obligation, carrier) and _strength_rank(
            carrier.evidence_strength
        ) > _strength_rank(best):
            best = carrier.evidence_strength
    proposition_text = next(
        (carrier.value for carrier in view.carriers if carrier.field_path == "proposition_text"), ""
    )
    context = obligation.attachment_context.strip()
    if (
        best is PreservationEvidenceStrength.NONE
        and category != "method"
        and context
        and _normalized(context) in _normalized(proposition_text)
        and not _has_conflicting_structured_evidence(obligation, view)
    ):
        best = PreservationEvidenceStrength.WEAK
    if (
        best is PreservationEvidenceStrength.NONE
        and category in {"measurement", "scope"}
        and obligation.source_text
        and _normalized(obligation.source_text) in _normalized(proposition_text)
        and not _has_conflicting_structured_evidence(obligation, view)
    ):
        best = PreservationEvidenceStrength.WEAK
    return best


def _span_anchors_obligation(span: str, obligation: QualifierObligation) -> bool:
    normalized_span = _normalized(span)
    normalized_context = _normalized(obligation.attachment_context)
    return (
        _normalized(obligation.source_text) in normalized_span
        and bool(normalized_context)
        and normalized_span in normalized_context
    )


def _source_span_covers_obligation(
    span: str, obligation: QualifierObligation, claim_text: str
) -> bool:
    """Require provenance to cover this occurrence, not merely the same token."""
    if (
        obligation.source_end <= len(claim_text)
        and claim_text[obligation.source_start : obligation.source_end]
        == obligation.source_text
    ):
        position = claim_text.find(span)
        while position >= 0:
            span_end = position + len(span)
            if position <= obligation.source_start and obligation.source_end <= span_end:
                return True
            position = claim_text.find(span, position + 1)
    # Conservative fallback for normalized spans: accept only a local span.
    return _span_anchors_obligation(span, obligation)


def _has_conflicting_structured_evidence(
    obligation: QualifierObligation, view: PropositionSemanticView
) -> bool:
    category = obligation.category
    typed = [carrier for carrier in view.carriers if carrier.field_path.startswith("typed_value")]
    if category == "attribution":
        attribution = [carrier for carrier in view.carriers if carrier.field_path == "attribution"]
        return bool(
            attribution
            and not any(
                _carrier_preserves_obligation(obligation, carrier) for carrier in attribution
            )
        )
    if category == "comparison":
        targets = [
            carrier for carrier in typed if carrier.field_path == "typed_value.comparison_target"
        ]
        return bool(
            targets
            and not any(_carrier_preserves_obligation(obligation, carrier) for carrier in targets)
        )
    if category in {"unit", "currency"}:
        measurements = [carrier for carrier in typed if carrier.field_path == "typed_value.numeric"]
        return bool(
            measurements
            and not any(
                _carrier_preserves_obligation(obligation, carrier) for carrier in measurements
            )
        )
    if category == "date_time":
        times = [
            carrier
            for carrier in view.carriers
            if carrier.category in {"time", "date", "period", "date_time"}
        ]
        return bool(
            times
            and not any(_carrier_preserves_obligation(obligation, carrier) for carrier in times)
        )
    if category == "version":
        versions = [carrier for carrier in view.carriers if carrier.category == "version"]
        return bool(
            versions
            and not any(_carrier_preserves_obligation(obligation, carrier) for carrier in versions)
        )
    if category == "method":
        method_values = [carrier for carrier in typed if carrier.field_path == "typed_value.text"]
        method_values.extend(
            carrier
            for carrier in view.carriers
            if carrier.category in {"method", "configuration"}
            or _carrier_has_role(carrier, "method")
        )
        return bool(
            method_values
            and not any(
                _carrier_preserves_obligation(obligation, carrier) for carrier in method_values
            )
        )
    return False


def _carrier_preserves_obligation(
    obligation: QualifierObligation, carrier: SemanticCarrier
) -> bool:
    category = obligation.category
    source = obligation.normalized_value
    value = carrier.normalized_value
    if category == "unit":
        source_number, source_unit = _numeric_unit(obligation.source_text)
        if source_number is not None and carrier.category == "measurement":
            try:
                payload = json.loads(carrier.value)
            except (TypeError, json.JSONDecodeError):
                return False
            return _numeric_payload_matches(
                payload,
                source_number,
                source_unit,
                attachment_context=obligation.attachment_context,
            )
        if source_number is not None and carrier.category == "unit":
            # A detached unit label cannot prove which numeric value it qualifies.
            return False
        return False
    if category == "currency":
        source_number, source_currency = _currency_amount(obligation.source_text)
        if carrier.category == "currency":
            # Currency alone cannot establish preservation of the source amount.
            return False
        if carrier.category == "measurement":
            try:
                return _numeric_payload_matches(
                    json.loads(carrier.value), source_number, source_currency, currency=True
                )
            except (TypeError, json.JSONDecodeError):
                return False
        return False
    if category == "date_time":
        if carrier.category not in {"date_time", "time", "date", "period"}:
            return False
        source_dates = _date_components(obligation.source_text)
        target_dates = _date_components(carrier.value)
        return bool(source_dates and any(item in target_dates for item in source_dates))
    if category == "version":
        versions = re.findall(r"\d+(?:\.\d+)+", obligation.source_text)
        return carrier.category == "version" and bool(versions) and versions[0] in carrier.value
    if category == "comparison":
        if carrier.category in {"comparison", "comparison_target"}:
            if carrier.value == "comparison":
                return False
            target = _normalized(carrier.value)
            context = _normalized(obligation.attachment_context)
            cue = obligation.normalized_value
            suffix = context.split(cue, 1)[-1]
            return bool(target and (target in context or target in suffix))
        return carrier.category == "predicate" and source in value
    if category == "negation":
        return (
            (carrier.category == "polarity" and carrier.value == "negated")
            or (carrier.category == "predicate" and source in value)
            or (carrier.category == "negation" and source in value)
        )
    if category == "attribution":
        if carrier.category == "attribution":
            target = carrier.normalized_value
            context = _normalized(obligation.attachment_context)
            target_tokens = _semantic_tokens(carrier.value)
            context_tokens = _semantic_tokens(obligation.attachment_context)
            return bool(target and (target in context or target_tokens & context_tokens))
        if carrier.category == "predicate":
            marker = re.search(
                r"(?:声称|宣称|指出|表示|发现|發現|表明|显示|預測|预测|认为|認為|提出|finds|states|reports|claims|suggests|concludes|propone)$",
                source,
                re.I,
            )
            source_tail = (
                marker.group(0) if marker else source.split()[-1] if source.split() else source
            )
            return source_tail in value
        return False
    if category == "modality":
        return (carrier.category == "modality" and (source in value or value in source)) or (
            carrier.category == "predicate" and source in value
        )
    if category == "method":
        if not (
            _carrier_has_role(carrier, "method")
            or carrier.category in {"method", "configuration", "typed_value"}
        ):
            return False
        target = _method_object(obligation.attachment_context)
        if carrier.category == "subject":
            return bool(source and target and source in value and target in value)
        if (
            carrier.category == "typed_value"
            and source
            and target
            and source in value
            and target in value
        ):
            return True
        if carrier.category in {"method", "configuration"} or _carrier_has_role(
            carrier, "method"
        ):
            carrier_object = _method_object(carrier.value)
            return bool(
                carrier_object
                and target
                and (carrier_object in target or target in carrier_object)
            )
        if carrier.category == "typed_value":
            return bool(target and target in value)
        if carrier.category == "predicate":
            return source in value and bool(target)
        return False

    if category == "configuration":
        # A structured carrier may live under another storage category, but it
        # must independently declare configuration semantics and carry the
        # source value. Source-span attachment is checked by the caller.
        return _carrier_has_role(carrier, "configuration") and bool(
            source and (source in value or value in source)
        )

    expected = _QUALIFIER_FIELD_CATEGORIES.get(category, frozenset({category}))
    if carrier.category in expected or carrier.category == category:
        if source in value or value in source:
            return True
        if _semantic_tokens(obligation.attachment_context) & _semantic_tokens(carrier.value):
            return True
    return carrier.category in {"subject", "predicate"} and (
        source in value
        or bool(
            _semantic_tokens(obligation.attachment_context)
            & _semantic_tokens(carrier.value)
        )
    )


def _semantic_tokens(value: str) -> set[str]:
    structural = {
        "a",
        "an",
        "the",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "under",
        "and",
        "or",
        "not",
        "only",
        "all",
        "region",
        "scope",
        "method",
        "setting",
        "configuration",
        "condition",
        "sample",
        "samples",
        "user",
        "users",
        "population",
    }
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%+-]*|[\u3400-\u4dbf\u4e00-\u9fff]{2,}", value)
    return {
        token.casefold()
        for token in raw
        if token.casefold() not in structural and (len(token) > 1 or token.isupper())
    }


def _has_unsupported_letter_script(value: str) -> bool:
    """Flag observable scripts outside the source detector's Latin/CJK/Kana coverage."""
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name or _is_cjk_or_kana(character):
            continue
        return True
    return False


def _shared_sentence_carrier(
    category: str,
    compatible: list[tuple[PropositionSemanticView, PreservationEvidenceStrength]],
) -> bool:
    if category not in {"modality", "attribution"}:
        return False
    values: list[str] = []
    for view, _ in compatible:
        category_values = [
            carrier.normalized_value
            for carrier in view.carriers
            if carrier.category == category and carrier.value.strip()
        ]
        if not category_values:
            return False
        values.append("|".join(sorted(category_values)))
    return len(set(values)) == 1


def _shared_group_comparison_carrier(
    obligation: QualifierObligation,
    compatible: list[tuple[PropositionSemanticView, PreservationEvidenceStrength]],
) -> bool:
    """Allow an identical comparison target shared by coordinated source predicates.

    Source-side coordination is mandatory. Distinct proposition source spans must
    lie within that same source context, and every member must carry the same
    validated comparison target. This does not infer group scope from output alone.
    """
    if obligation.category != "comparison" or len(compatible) < 2:
        return False
    context = _normalized(obligation.attachment_context)
    if not re.search(r"(?:并|以及|同时|且|和|与|及|\band\b)", context):
        return False
    spans = [
        _normalized(span)
        for view, _ in compatible
        for span in view.source_spans
        if span
    ]
    if len(spans) < len(compatible) or len(set(spans)) < len(compatible):
        return False
    if not all(span in context for span in spans):
        return False
    targets: list[str] = []
    for view, _ in compatible:
        values = {
            carrier.normalized_value
            for carrier in view.carriers
            if carrier.category in {"comparison", "comparison_target"}
            and carrier.value != "comparison"
        }
        if len(values) != 1:
            return False
        targets.extend(values)
    return len(set(targets)) == 1


def _shared_group_method_carrier(
    obligation: QualifierObligation,
    compatible: list[tuple[PropositionSemanticView, PreservationEvidenceStrength]],
    claim_text: str,
) -> bool:
    """Recognize one explicitly repeated method carrier on a source-side group.

    Every member must carry the same method value, and every carrier must have its
    own source span anchoring the detected method relation. Mere occurrence of the
    method in the claim or one neighboring proposition is insufficient.
    """
    if (
        obligation.category != "method"
        or _is_method_noun(obligation.source_text)
        or len(compatible) < 2
    ):
        return False
    values: list[str] = []
    for view, _ in compatible:
        attached_values = {
            carrier.normalized_value
            for carrier in view.carriers
            if (
                carrier.category in {"method", "configuration"}
                or _carrier_has_role(carrier, "method")
            )
            and carrier.source_span
            and _source_span_covers_obligation(carrier.source_span, obligation, claim_text)
            and _carrier_preserves_obligation(obligation, carrier)
        }
        if len(attached_values) != 1:
            return False
        values.extend(attached_values)
    return len(set(values)) == 1


def _method_object(value: str) -> str:
    normalized = _normalized(value)
    target = re.sub(
        r"^.*?(?:采用|採用|通过|通過|使用|基于|基於|using|uses|used|via|by\s+means\s+of)\s*",
        "",
        normalized,
    ).strip(" 的:：")
    # In Chinese ``based on X 的 method`` expresses X as the relation object;
    # trailing nominal material is not part of that object.
    return re.split(r"\s*的\s*", target, maxsplit=1)[0].strip()


def _is_method_noun(value: str) -> bool:
    return bool(
        re.fullmatch(r"(?:方法|算法|method|methods|algorithm|algorithms)", _normalized(value))
    )


def _is_comparison_target_method_noun(claim_text: str, match: re.Match[str]) -> bool:
    """Do not treat a method noun in a comparison target as a method qualifier."""
    marker = match.group(0)
    if not _is_method_noun(marker):
        return False
    sentence_start = max(
        claim_text.rfind(mark, 0, match.start()) + 1 for mark in ("。", ".", "!", "?", "；", ";")
    )
    left = _normalized(claim_text[sentence_start : match.start()])
    return bool(
        re.search(
            r"(?:超过|超越|优于|優於|高于|高於|低于|低於|相比|相较|相較|比|"
            r"compared\s+(?:with|to)|versus|vs\.?|\bthan\b)"
            r"[^。；;.!?]{0,48}$",
            left,
            re.I,
        )
    )


def _date_components(value: str) -> tuple[tuple[int, ...], ...]:
    """Parse date atoms, retaining each endpoint in a range independently."""
    normalized = unicodedata.normalize("NFKC", value)
    pattern = re.compile(
        r"(?<!\d)(\d{4})(?:(?:\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日?)?)"
        r"|(?:\s*[-/]\s*(\d{1,2})(?:\s*[-/]\s*(\d{1,2}))?))?"
    )
    results: list[tuple[int, ...]] = []
    for match in pattern.finditer(normalized):
        year, cjk_month, cjk_day, numeric_month, numeric_day = match.groups()
        month = cjk_month or numeric_month
        day = cjk_day or numeric_day
        components: tuple[int, ...] = (
            (int(year),) if month is None else (int(year), int(month))
        )
        if day is not None:
            components += (int(day),)
        results.append(components)
    return tuple(results)


def _numeric_unit(value: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"(\d+(?:[.,]\d+)*)(\s*[^\d\s,，。；;]*)", unicodedata.normalize("NFKC", value)
    )
    if not match:
        return None, None
    number = match.group(1).rstrip(".,")
    unit = match.group(2).strip().rstrip(".,")
    return number.replace(",", ""), _canonical_unit(unit)


def _currency_amount(value: str) -> tuple[str | None, str | None]:
    normalized = unicodedata.normalize("NFKC", value).strip()
    prefix = re.search(r"([$€£¥￥])\s*(\d+(?:[.,]\d+)*)", normalized)
    suffix = re.search(
        r"(\d+(?:[.,]\d+)*)\s*(USD|EUR|GBP|JPY|CNY|RMB|美元|欧元|英镑|日元|人民币|元|円)",
        normalized,
        re.I,
    )
    match = prefix or suffix
    if not match:
        return None, None
    if prefix is not None:
        number, currency = prefix.group(2), prefix.group(1)
    elif suffix is not None:
        number, currency = suffix.group(1), suffix.group(2)
    else:
        return None, None
    return number.rstrip(".,").replace(",", ""), _canonical_currency(currency)


def _canonical_unit(value: str | None) -> str | None:
    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip("% .")
    aliases = {
        "percent": "%",
        "percentage": "%",
        "年": "year",
        "years": "year",
        "秒": "second",
        "seconds": "second",
        "分钟": "minute",
        "minutes": "minute",
        "米": "meter",
        "meters": "meter",
        "公里": "km",
        "kilometers": "km",
    }
    if not normalized:
        return "%" if "%" in unicodedata.normalize("NFKC", value or "") else None
    return aliases.get(normalized, normalized)


def _canonical_currency(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    aliases = {
        "$": "USD",
        "US$": "USD",
        "USD": "USD",
        "¥": "CNY",
        "￥": "CNY",
        "CNY": "CNY",
        "RMB": "CNY",
        "元": "CNY",
        "円": "JPY",
        "€": "EUR",
        "EUR": "EUR",
        "£": "GBP",
        "GBP": "GBP",
        "JPY": "JPY",
        "日元": "JPY",
        "人民币": "CNY",
    }
    return aliases.get(normalized, normalized or None)


def _numeric_payload_matches(
    payload: Any,
    source_number: str | None,
    source_unit: str | None,
    *,
    currency: bool = False,
    attachment_context: str = "",
) -> bool:
    if not isinstance(payload, dict):
        return False
    numbers = (payload.get("value"), payload.get("lower_bound"), payload.get("upper_bound"))
    matches_number = source_number is None or any(
        value is not None and _normalized(str(value)).replace(",", "") == source_number
        for value in numbers
    )
    if currency:
        return (
            matches_number
            and _canonical_currency(str(payload.get("currency") or "")) == source_unit
        )
    raw_target_unit = unicodedata.normalize(
        "NFKC", str(payload.get("unit") or "")
    ).casefold().strip()
    target_unit = _canonical_unit(raw_target_unit)
    same_unit = target_unit == source_unit
    # A percent value may name its metric after the unit (for example, ``% AP``).
    # Accept that refinement only when the exact typed value matches and the metric
    # suffix is also present in the source-side attachment context.
    if source_unit == "%" and raw_target_unit.startswith("%"):
        metric_suffix = raw_target_unit[1:].strip(" .")
        if metric_suffix:
            same_unit = _normalized(metric_suffix) in _normalized(attachment_context)
    return matches_number and same_unit


def _strength_rank(value: PreservationEvidenceStrength) -> int:
    return {
        PreservationEvidenceStrength.NONE: 0,
        PreservationEvidenceStrength.WEAK: 1,
        PreservationEvidenceStrength.MEDIUM: 2,
        PreservationEvidenceStrength.STRONG: 3,
    }[value]


def _qualifier_match_value(category: str, match: re.Match[str]) -> str:
    if category == "version":
        return next(
            value
            for key, value in match.groupdict().items()
            if key.endswith("version") and value is not None
        )
    return match.group(0)


def _version_value_span(match: re.Match[str]) -> tuple[int, int]:
    for key, value in match.groupdict().items():
        if key.endswith("version") and value is not None:
            return match.span(key)
    return match.span()


def _proposition_supports(category: str, detected: str, proposition: dict[str, Any]) -> bool:
    """Require category-specific structured support, with exact source wording as backup."""
    typed_value = proposition.get("typed_value")
    typed_value = typed_value if isinstance(typed_value, dict) else {}
    qualifiers = proposition.get("material_qualifiers")
    qualifiers = qualifiers if isinstance(qualifiers, list | tuple) else ()
    qualifier_categories = {
        str(item.get("category", "")).casefold() for item in qualifiers if isinstance(item, dict)
    }
    expected = _QUALIFIER_FIELD_CATEGORIES[category]
    structured = False
    if category == "date_time":
        structured = bool(
            typed_value.get("temporal_from") or typed_value.get("temporal_to")
        ) or bool(qualifier_categories & expected)
    elif category == "currency":
        numeric = typed_value.get("numeric")
        structured = (
            bool(isinstance(numeric, dict) and numeric.get("currency"))
            or "currency" in qualifier_categories
        )
    elif category == "unit":
        numeric = typed_value.get("numeric")
        structured = bool(isinstance(numeric, dict) and numeric.get("unit")) or bool(
            qualifier_categories & expected
        )
    elif category == "comparison":
        structured = typed_value.get("kind") == "comparison" and bool(
            typed_value.get("comparison_target")
        )
    elif category == "modality":
        structured = bool(proposition.get("modality")) or "modality" in qualifier_categories
    elif category == "attribution":
        structured = bool(proposition.get("attribution")) or "attribution" in qualifier_categories
    elif category == "negation":
        structured = (
            str(proposition.get("polarity", "")).casefold() == "negated"
            or "negation" in qualifier_categories
        )
    elif category == "condition":
        structured = any(
            isinstance(qualifier, dict)
            and str(qualifier.get("category", "")).casefold() in expected
            and qualifier.get("value") not in (None, "")
            for qualifier in qualifiers
        )
    elif category == "attribution":
        structured = bool(proposition.get("attribution")) or "attribution" in qualifier_categories
    elif category == "method":
        # Configuration can coexist with a method, but it is not by itself
        # evidence that a source method marker was preserved.
        structured = "method" in qualifier_categories
    elif category == "measurement":
        comparison_target = typed_value.get("comparison_target")
        structured = bool(qualifier_categories & expected) or bool(
            typed_value.get("kind") == "comparison"
            and isinstance(comparison_target, str)
            and _normalized(detected) in _normalized(comparison_target)
        )
    else:
        structured = bool(qualifier_categories & expected)
    if not structured and category not in {"attribution", "negation", "measurement", "method"}:
        return False

    # For semantic markers, category-level structured values corroborate source-side
    # detection. For structural markers, require the actual value in a relevant field.
    if category in {"comparison", "modality"}:
        return True
    if category == "attribution":
        predicate = str(proposition.get("predicate", ""))
        return structured or bool(predicate and _normalized(detected) in _normalized(predicate))
    if category == "condition" and structured:
        # A structured condition value can preserve the semantic setting even
        # when the output appropriately omits the source's connective wording.
        return True
    if category == "method":
        predicate = str(proposition.get("predicate", ""))
        if structured:
            # An explicit method qualifier is already an attached canonical field; do
            # not require the source's method verb to be duplicated in proposition text.
            return any(
                isinstance(qualifier, dict)
                and str(qualifier.get("category", "")).casefold() == "method"
                and qualifier.get("value") not in (None, "")
                for qualifier in qualifiers
            )
        has_method_object = bool(
            typed_value.get("text")
            or typed_value.get("comparison_target")
            or any(
                isinstance(qualifier, dict)
                and str(qualifier.get("category", "")).casefold() in expected
                and qualifier.get("value") not in (None, "")
                for qualifier in qualifiers
            )
        )
        # The method relation itself may be represented by the predicate rather than
        # repeated as a material qualifier. Require same-proposition attachment and an
        # object/value so unrelated predicate text cannot count as preservation.
        return (
            bool(predicate)
            and _normalized(detected) in _normalized(predicate)
            and has_method_object
        )
    relevant_values: list[str] = []
    for qualifier in qualifiers:
        if (
            isinstance(qualifier, dict)
            and str(qualifier.get("category", "")).casefold() in expected
        ):
            relevant_values.extend(str(qualifier.get(key, "")) for key in ("value", "source_span"))
    for key in ("temporal_from", "temporal_to", "comparison_target", "comparison_metric"):
        value = typed_value.get(key)
        if isinstance(value, str):
            relevant_values.append(value)
    numeric = typed_value.get("numeric")
    if isinstance(numeric, dict):
        relevant_values.extend(str(numeric.get(key, "")) for key in ("unit", "currency"))
    relevant_values.extend(
        str(proposition.get(key, "")) for key in ("subject", "predicate", "proposition_text")
    )
    if structured and category == "negation":
        return True
    if structured:
        return any(
            _normalized(detected) in _normalized(value) for value in relevant_values if value
        )

    # A truth-condition negation may be carried by the predicate even when the
    # model left polarity asserted. Do not accept a stray occurrence in prose:
    # the predicate is the attached semantic field, while proposition_text can
    # contain quoted or unrelated source words.
    if category == "negation":
        predicate = str(proposition.get("predicate", ""))
        return bool(predicate) and _normalized(detected) in _normalized(predicate)

    # Measurement words are allowed to be represented by the predicate itself
    # (for example, “accuracy declined”) or by a typed numeric value/unit. This
    # is a category-specific semantic equivalence, not a universal token rule.
    if category == "measurement":
        predicate = str(proposition.get("predicate", ""))
        typed_text = str(typed_value.get("text", ""))
        return any(
            _normalized(detected) in _normalized(value)
            for value in (predicate, typed_text)
            if value
        )
    return False


def validate_decomposition_response(
    raw_response: dict[str, Any] | str | None,
    *,
    claim_text: str,
    claim_id: str | None = None,
    provenance: dict[str, str] | None = None,
    schema_version: str = ATOMIC_PROPOSITION_SCHEMA_VERSION,
) -> ValidatedDecomposition:
    """Parse strict model output and construct contract propositions without repair."""
    del claim_id, provenance  # Provenance is carried by the enclosing shadow record.
    if schema_version != ATOMIC_PROPOSITION_SCHEMA_VERSION:
        return _invalid(
            schema_version,
            "schema_version_mismatch",
            "Requested schema version is unknown.",
            diagnostics=(
                ValidationDiagnostic(
                    validation_stage="TOP_LEVEL_SCHEMA",
                    issue_code="schema_version_mismatch",
                    field_path="schema_version",
                    expected_contract=ATOMIC_PROPOSITION_SCHEMA_VERSION,
                    observed_type="str",
                    observed_shape={"source": "requested_schema_version"},
                ),
            ),
        )
    payload: Any
    try:
        if isinstance(raw_response, str):
            payload = json.loads(raw_response)
        elif isinstance(raw_response, dict):
            payload = raw_response
        else:
            return _invalid(
                schema_version,
                "structured_output_missing",
                "Structured response is missing.",
                diagnostics=(
                    ValidationDiagnostic(
                        validation_stage="JSON_DECODE",
                        issue_code="structured_output_missing",
                        field_path="$",
                        expected_contract="A JSON object matching the response contract.",
                        observed_type=_safe_type(raw_response),
                        observed_shape=_safe_shape(raw_response),
                    ),
                ),
            )
        response = ShadowDecompositionResponse.model_validate(payload)
        ShadowDecompositionResponse.validate_shape(response)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        if isinstance(exc, ValidationError):
            return _invalid(
                schema_version,
                "schema_invalid",
                "Structured response failed schema validation.",
                diagnostics=_pydantic_diagnostics(exc, payload),
            )
        if isinstance(exc, json.JSONDecodeError):
            return _invalid(
                schema_version,
                "structured_output_invalid",
                "Structured response is not valid JSON.",
                diagnostics=(
                    ValidationDiagnostic(
                        validation_stage="JSON_DECODE",
                        issue_code="json_decode_error",
                        field_path="$",
                        expected_contract="A JSON object matching the response contract.",
                        observed_type="str",
                        observed_shape={"json_error_position": exc.pos},
                    ),
                ),
            )
        return _invalid(
            schema_version,
            "structured_output_invalid",
            "Structured response shape is invalid.",
            diagnostics=(
                ValidationDiagnostic(
                    validation_stage="TOP_LEVEL_SHAPE",
                    issue_code="invalid_response_shape",
                    field_path="decomposition_status/propositions/ambiguity_reason",
                    expected_contract="Status, proposition count, and ambiguity reason must agree.",
                    observed_type=type(payload).__name__,
                    observed_shape=_safe_shape(payload),
                ),
            ),
        )

    normalized_texts = [_normalized(item.proposition_text) for item in response.propositions]
    if len(set(normalized_texts)) != len(normalized_texts):
        duplicate_index = next(
            index for index, text in enumerate(normalized_texts) if text in normalized_texts[:index]
        )
        return _invalid(
            schema_version,
            "duplicate_proposition",
            "Exact normalized duplicate proposition text was returned.",
            diagnostics=(
                ValidationDiagnostic(
                    validation_stage="DUPLICATE_CHECK",
                    issue_code="duplicate_proposition",
                    proposition_index=duplicate_index,
                    field_path=f"propositions[{duplicate_index}].proposition_text",
                    expected_contract="Each proposition_text must be unique after normalization.",
                    observed_type="str",
                    observed_shape={
                        "duplicate_of_index": normalized_texts.index(
                            normalized_texts[duplicate_index]
                        )
                    },
                ),
            ),
        )

    qualifier_checks = inspect_explicit_qualifiers(claim_text, response)
    issues = tuple(
        ValidationIssue(
            code=(
                f"potential_{item.category}_loss"
                if item.status is QualifierCheckStatus.POTENTIAL_LOSS
                else f"ambiguous_{item.category}_scope"
            ),
            message=(
                f"An explicit {item.category} marker may not be represented in the output."
                if item.status is QualifierCheckStatus.POTENTIAL_LOSS
                else f"The attachment of an explicit {item.category} marker is ambiguous."
            ),
        )
        for item in qualifier_checks
        if item.status in {QualifierCheckStatus.POTENTIAL_LOSS, QualifierCheckStatus.AMBIGUOUS}
    )
    qualifier_diagnostics = tuple(
        ValidationDiagnostic(
            validation_stage="QUALIFIER_PRESERVATION",
            issue_code=issue.code,
            field_path=f"claim.qualifier.{check.category}",
            expected_contract="Explicit material qualifiers should be represented in the output.",
            observed_type=(
                "not_detected_in_output"
                if check.status is QualifierCheckStatus.POTENTIAL_LOSS
                else "source_attachment_ambiguous"
            ),
            observed_shape={"category": check.category},
        )
        for check, issue in zip(
            (
                item
                for item in qualifier_checks
                if item.status
                in {QualifierCheckStatus.POTENTIAL_LOSS, QualifierCheckStatus.AMBIGUOUS}
            ),
            issues,
            strict=True,
        )
    )
    if response.decomposition_status is ShadowDecompositionLabel.UNRESOLVED:
        return ValidatedDecomposition(
            schema_version=schema_version,
            decomposition_status=response.decomposition_status,
            validation_issues=issues,
            validation_diagnostics=qualifier_diagnostics,
            qualifier_checks=qualifier_checks,
            ambiguity_reason=response.ambiguity_reason,
            valid=True,
        )

    decomposition_status = (
        DecompositionStatus.SINGLE_PROPOSITION
        if response.decomposition_status is ShadowDecompositionLabel.SINGLE
        else DecompositionStatus.DECOMPOSED
    )
    propositions = tuple(
        AtomicProposition(
            proposition_id=uuid4(),
            schema_version=schema_version,
            original_text=item.proposition_text,
            subject=item.subject,
            predicate=item.predicate,
            value=PropositionValue.model_validate(item.typed_value.model_dump(mode="python")),
            polarity=item.polarity,
            qualifiers=item.material_qualifiers,
            modality=item.modality,
            attribution=item.attribution,
            decomposition=DecompositionMetadata(
                status=decomposition_status,
                method=DecompositionMethod.STRUCTURED_MODEL,
                schema_version=schema_version,
                prompt_version=ATOMIC_DECOMPOSITION_PROMPT_VERSION,
                proposition_index=index,
                source_span=item.source_span,
                confidence=item.confidence,
            ),
        )
        for index, item in enumerate(response.propositions)
    )
    return ValidatedDecomposition(
        schema_version=schema_version,
        decomposition_status=response.decomposition_status,
        propositions=propositions,
        validation_issues=issues,
        validation_diagnostics=qualifier_diagnostics,
        qualifier_checks=qualifier_checks,
        valid=True,
    )


def _invalid(
    schema_version: str,
    code: str,
    message: str,
    *,
    diagnostics: tuple[ValidationDiagnostic, ...] = (),
) -> ValidatedDecomposition:
    return ValidatedDecomposition(
        schema_version=schema_version,
        validation_issues=(ValidationIssue(code=code, message=message or code),),
        validation_diagnostics=diagnostics,
        valid=False,
    )


def _pydantic_diagnostics(error: ValidationError, payload: Any) -> tuple[ValidationDiagnostic, ...]:
    diagnostics: list[ValidationDiagnostic] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = tuple(item.get("loc", ()))
        proposition_index = (
            location[1]
            if len(location) > 1 and location[0] == "propositions" and isinstance(location[1], int)
            else None
        )
        field_path = _safe_field_path(location)
        error_type = str(item.get("type", "validation_error"))[:100]
        stage = _validation_stage(location)
        observed = _value_at_location(payload, location)
        if (
            stage == "PROPOSITION_SCHEMA"
            and proposition_index is not None
            and error_type == "value_error"
            and isinstance(observed, dict)
            and isinstance(observed.get("typed_value"), dict)
        ):
            stage = "TYPED_VALUE_SHAPE"
            field_path = f"propositions[{proposition_index}].typed_value"
            observed = observed["typed_value"]
        diagnostics.append(
            ValidationDiagnostic(
                validation_stage=stage,
                issue_code=error_type,
                proposition_index=proposition_index,
                field_path=field_path,
                expected_contract=_expected_contract(location),
                observed_type=_safe_type(observed),
                observed_shape=_safe_shape(observed),
            )
        )
    return tuple(diagnostics)


def _validation_stage(location: tuple[Any, ...]) -> str:
    if not location:
        return "TOP_LEVEL_SCHEMA"
    field = location[2] if len(location) > 2 and isinstance(location[1], int) else location[0]
    if field == "typed_value" or (len(location) > 2 and location[2] == "typed_value"):
        return "TYPED_VALUE_SHAPE"
    if field == "polarity":
        return "POLARITY"
    if field == "material_qualifiers" or field == "qualifiers":
        return "QUALIFIER_SCHEMA"
    if field in {
        "proposition_text",
        "subject",
        "predicate",
        "source_span",
        "modality",
        "attribution",
    }:
        return "PROPOSITION_SCHEMA"
    if location[0] == "propositions":
        return "PROPOSITION_SCHEMA"
    return "TOP_LEVEL_SCHEMA"


def _safe_field_path(location: tuple[Any, ...]) -> str:
    allowed = {
        "schema_version",
        "decomposition_status",
        "propositions",
        "ambiguity_reason",
        "proposition_text",
        "subject",
        "predicate",
        "typed_value",
        "kind",
        "text",
        "boolean",
        "numeric",
        "operator",
        "value",
        "lower_bound",
        "upper_bound",
        "unit",
        "currency",
        "tolerance",
        "members",
        "comparison_target",
        "comparison_metric",
        "temporal_from",
        "temporal_to",
        "polarity",
        "material_qualifiers",
        "category",
        "key",
        "materiality",
        "source_span",
        "modality",
        "attribution",
        "confidence",
    }
    variant_names = {
        "text",
        "entity",
        "boolean",
        "quantity",
        "range",
        "set",
        "comparison",
        "temporal_change",
        "unknown",
        "exact",
        "approx",
        "lt",
        "lte",
        "gt",
        "gte",
    }
    parts: list[str] = []
    previous: object | None = None
    for segment in location:
        if isinstance(segment, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{segment}]"
            else:
                parts.append(f"[{segment}]")
        elif isinstance(segment, str):
            if segment in variant_names and previous in {"typed_value", "numeric"}:
                previous = segment
                continue
            safe_segment = segment if segment in allowed else "<unrecognized_field>"
            parts.append(("." if parts else "") + safe_segment)
            previous = segment
    return "".join(parts) or "$"


def _expected_contract(location: tuple[Any, ...]) -> str:
    field = location[-1] if location else None
    contracts = {
        "schema_version": ATOMIC_PROPOSITION_SCHEMA_VERSION,
        "decomposition_status": "single_proposition | decomposed | unresolved_decomposition",
        "proposition_text": "Non-empty string, at most 4000 characters.",
        "subject": "Non-empty string, at most 1000 characters.",
        "predicate": "Non-empty string, at most 1000 characters.",
        "polarity": "asserted | negated | unknown",
        "kind": (
            "text | entity | boolean | quantity | set | range | comparison | "
            "temporal_change | unknown"
        ),
        "materiality": "material | descriptive | unknown",
        "category": "A supported MaterialQualifier category.",
        "operator": "exact | approx | lt | lte | gt | gte | range",
        "propositions": "Array of at most 32 proposition objects.",
    }
    if "typed_value" in location:
        return "Kind-specific typed value and nested numeric operator contract."
    return contracts.get(str(field), "Field must satisfy the declared response model contract.")


def _value_at_location(payload: Any, location: tuple[Any, ...]) -> Any:
    value = payload
    variant_names = {
        "text",
        "entity",
        "boolean",
        "quantity",
        "range",
        "set",
        "comparison",
        "temporal_change",
        "unknown",
        "exact",
        "approx",
        "lt",
        "lte",
        "gt",
        "gte",
    }
    previous: object | None = None
    for segment in location:
        if (
            isinstance(segment, str)
            and segment in variant_names
            and previous
            in {
                "typed_value",
                "numeric",
            }
        ):
            previous = segment
            continue
        if isinstance(value, dict) and isinstance(segment, str):
            value = value.get(segment)
        elif (
            isinstance(value, list | tuple)
            and isinstance(segment, int)
            and 0 <= segment < len(value)
        ):
            value = value[segment]
        else:
            return None
        previous = segment
    return value


def _safe_type(value: Any) -> str:
    if value is None:
        return "null_or_missing"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int | float | Decimal):
        return "number"
    return type(value).__name__[:50]


def _safe_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        safe_keys = sorted(
            key if key in _DIAGNOSTIC_FIELD_NAMES else "<unrecognized_field>"
            for key in (str(item) for item in value)
        )
        return {"keys": safe_keys[:40]}
    if isinstance(value, list | tuple):
        return {
            "length": len(value),
            "item_types": sorted({_safe_type(item) for item in value[:20]}),
        }
    if isinstance(value, str):
        return {"length": len(value)}
    if value is None or isinstance(value, bool | int | float | Decimal):
        return {}
    return {"type": type(value).__name__[:50]}


_DIAGNOSTIC_FIELD_NAMES = frozenset(
    {
        "schema_version",
        "decomposition_status",
        "propositions",
        "ambiguity_reason",
        "proposition_text",
        "subject",
        "predicate",
        "typed_value",
        "kind",
        "text",
        "boolean",
        "numeric",
        "operator",
        "value",
        "lower_bound",
        "upper_bound",
        "unit",
        "currency",
        "tolerance",
        "members",
        "comparison_target",
        "comparison_metric",
        "temporal_from",
        "temporal_to",
        "polarity",
        "material_qualifiers",
        "category",
        "key",
        "materiality",
        "source_span",
        "modality",
        "attribution",
        "confidence",
    }
)
