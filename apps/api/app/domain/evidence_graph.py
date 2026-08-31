"""Pure helpers for building traceable Evidence Graph records."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)\s*(亿美元|亿元|万元|美元|%|\uff05|亿|万|元)?")
_GRAPH_PUNCTUATION = re.compile(r"[^0-9a-zA-Z%\uff05\u4e00-\u9fff<>]+")
_NEGATIVE_MARKERS = (
    "不支持",
    "无法",
    "没有",
    "尚未",
    "未能",
    "未进入",
    "未达到",
    "未实现",
    "下降",
    "减少",
    "降低",
    "落后",
)
_POSITIVE_MARKERS = ("支持", "可以", "已经", "增长", "上升", "增加", "提高", "领先")
_METRIC_PATTERNS = (
    ("market_size", re.compile(r"市场(?:在.{0,8}?年的?)?规模|市场规模")),
    ("sales", re.compile(r"销售额|营收|营业收入")),
    ("growth", re.compile(r"年均复合增长率|年复合增长率|复合增长率|增长率")),
    ("share", re.compile(r"市场份额|份额|占比")),
    ("quality", re.compile(r"准确率|精度|召回率|误检率|漏检率")),
    ("cost", re.compile(r"成本|价格")),
    ("quantity", re.compile(r"数量|装机量|出货量")),
)
_SCOPE_NOISE = re.compile(r"截至|当前|预计|未来|约|超过|达到|达|从|至|到|为|的|年")


@dataclass(frozen=True, slots=True)
class EvidenceChunkWindow:
    """A source window that preserves the exact quote and original offsets."""

    char_start: int
    char_end: int
    text: str
    token_count: int
    chunk_hash: str


@dataclass(frozen=True, slots=True)
class ClaimRelationDecision:
    relation: str
    confidence: float
    severity: float
    reason_code: str


@dataclass(frozen=True, slots=True)
class _NumericFact:
    value: float
    unit: str


class EvidenceGraphEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: UUID
    source_id: UUID
    snapshot_id: UUID | None = None
    chunk_id: UUID | None = None
    relation: str
    accepted: bool
    evidence_score: float = Field(ge=0.0, le=1.0)


class EvidenceGraphClaimNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: UUID
    question_id: str
    dimension_key: str
    atomic_claim: str
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceGraphEvidenceRef] = Field(default_factory=list)


class EvidenceGraphClaimEdgeView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: UUID
    from_claim_id: UUID
    to_claim_id: UUID
    relation: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceGraphConflictView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_id: UUID
    question_id: str
    entity: str
    attribute: str
    left_evidence_id: UUID
    right_evidence_id: UUID
    severity: float = Field(ge=0.0, le=1.0)
    status: str
    resolution_summary: str | None = None


class EvidenceGraphView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    claim_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    claims: list[EvidenceGraphClaimNode] = Field(default_factory=list)
    edges: list[EvidenceGraphClaimEdgeView] = Field(default_factory=list)
    conflicts: list[EvidenceGraphConflictView] = Field(default_factory=list)


def infer_claim_relation(
    left_claim: str,
    right_claim: str,
) -> ClaimRelationDecision | None:
    """Conservatively infer a reviewable relation between two atomic claims."""

    left = _relation_text(left_claim)
    right = _relation_text(right_claim)
    if not left or not right:
        return None
    if left == right:
        return ClaimRelationDecision("supports", 1.0, 0.0, "normalized_exact_match")

    left_numbers = _numeric_facts(left_claim)
    right_numbers = _numeric_facts(right_claim)
    skeleton_similarity = SequenceMatcher(
        None,
        _numeric_skeleton(left),
        _numeric_skeleton(right),
    ).ratio()
    text_similarity = SequenceMatcher(None, left, right).ratio()

    numeric_difference = _numeric_difference(left_numbers, right_numbers)
    left_numeric_scope = _numeric_relation_scope(left_claim)
    right_numeric_scope = _numeric_relation_scope(right_claim)
    same_numeric_scope = (
        left_numeric_scope is not None and left_numeric_scope == right_numeric_scope
    )
    if same_numeric_scope and skeleton_similarity >= 0.58 and numeric_difference is not None:
        if numeric_difference > 0.10:
            return ClaimRelationDecision(
                "contradicts",
                min(0.98, 0.78 + 0.20 * skeleton_similarity),
                min(1.0, max(0.4, numeric_difference)),
                "same_scope_numeric_difference",
            )
        return ClaimRelationDecision(
            "supports",
            min(0.96, 0.72 + 0.24 * skeleton_similarity),
            0.0,
            "same_scope_numeric_agreement",
        )

    if skeleton_similarity >= 0.60 and _opposite_polarity(left_claim, right_claim):
        return ClaimRelationDecision(
            "contradicts",
            min(0.92, 0.68 + 0.24 * skeleton_similarity),
            0.7,
            "opposite_polarity",
        )
    if text_similarity >= 0.86:
        return ClaimRelationDecision(
            "supports",
            min(0.94, text_similarity),
            0.0,
            "high_textual_agreement",
        )
    if text_similarity >= 0.58:
        return ClaimRelationDecision(
            "supplements",
            min(0.88, text_similarity),
            0.0,
            "related_scope_additional_detail",
        )
    return None


def _relation_text(value: str) -> str:
    normalized = normalize_graph_text(value).casefold()
    return _GRAPH_PUNCTUATION.sub("", normalized)


def _numeric_skeleton(value: str) -> str:
    return _NUMBER.sub("<number>", value)


def _numeric_facts(value: str) -> tuple[_NumericFact, ...]:
    return tuple(
        _NumericFact(
            value=float(match.group(1)),
            unit=(match.group(2) or "").replace("\uff05", "%"),
        )
        for match in _NUMBER.finditer(value)
    )


def _numeric_relation_scope(value: str) -> tuple[str, str] | None:
    """Return a conservative entity/metric signature for numeric comparison."""

    without_numbers = _NUMBER.sub("", normalize_graph_text(value))
    for metric_key, pattern in _METRIC_PATTERNS:
        match = pattern.search(without_numbers)
        if match is None:
            continue
        raw_scope = without_numbers[: match.start()]
        scope = _GRAPH_PUNCTUATION.sub("", _SCOPE_NOISE.sub("", raw_scope)).casefold()
        if scope:
            return scope, metric_key
    return None


def _numeric_difference(
    left: tuple[_NumericFact, ...],
    right: tuple[_NumericFact, ...],
) -> float | None:
    if not left or len(left) != len(right):
        return None
    if tuple(item.unit for item in left) != tuple(item.unit for item in right):
        return None
    differences = [
        abs(left_item.value - right_item.value)
        / max(abs(left_item.value), abs(right_item.value), 1e-9)
        for left_item, right_item in zip(left, right, strict=True)
    ]
    return max(differences, default=0.0)


def _opposite_polarity(left: str, right: str) -> bool:
    left_polarity = _polarity(left)
    right_polarity = _polarity(right)
    return left_polarity != 0 and right_polarity != 0 and left_polarity != right_polarity


def _polarity(value: str) -> int:
    if any(marker in value for marker in _NEGATIVE_MARKERS):
        return -1
    if any(marker in value for marker in _POSITIVE_MARKERS):
        return 1
    return 0


def derive_claim_status(
    *,
    has_accepted_evidence: bool,
    has_refuting_evidence: bool,
    independent_source_count: int,
) -> str:
    """Separate quote acceptance from claim-level corroboration."""

    if has_refuting_evidence:
        return "disputed"
    if not has_accepted_evidence:
        return "rejected"
    if independent_source_count >= 2:
        return "supported"
    return "partial"


def normalize_graph_text(value: str) -> str:
    """Normalize only whitespace so hashes remain stable without changing meaning."""

    return _WHITESPACE.sub(" ", value).strip()


def claim_fingerprint(claim: str) -> str:
    """Return the stable fingerprint used to deduplicate an atomic claim."""

    return hashlib.sha256(normalize_graph_text(claim).casefold().encode()).hexdigest()


def locate_quote(source_text: str, exact_quote: str) -> tuple[int, int] | None:
    """Locate a quote while tolerating equivalent whitespace in extracted HTML text."""

    direct_start = source_text.find(exact_quote)
    if direct_start >= 0:
        return direct_start, direct_start + len(exact_quote)

    normalized_source, source_positions = _normalize_with_positions(source_text)
    normalized_quote = normalize_graph_text(exact_quote)
    normalized_start = normalized_source.find(normalized_quote)
    if normalized_start < 0 or not normalized_quote:
        return None

    normalized_end = normalized_start + len(normalized_quote) - 1
    if normalized_end >= len(source_positions):
        return None
    return source_positions[normalized_start], source_positions[normalized_end] + 1


def build_evidence_chunk(
    source_text: str,
    exact_quote: str,
    *,
    context_chars: int = 1_800,
) -> EvidenceChunkWindow | None:
    """Build a bounded, reproducible source chunk around an exact quote."""

    location = locate_quote(source_text, exact_quote)
    if location is None:
        return None
    quote_start, quote_end = location
    char_start = max(0, quote_start - context_chars)
    char_end = min(len(source_text), quote_end + context_chars)
    text = source_text[char_start:char_end]
    chunk_hash = hashlib.sha256(f"{char_start}:{char_end}\n{text}".encode()).hexdigest()
    return EvidenceChunkWindow(
        char_start=char_start,
        char_end=char_end,
        text=text,
        token_count=max(1, len(text) // 4),
        chunk_hash=chunk_hash,
    )


def _normalize_with_positions(value: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    in_whitespace = False
    for index, character in enumerate(value):
        if character.isspace():
            if normalized and not in_whitespace:
                normalized.append(" ")
                positions.append(index)
            in_whitespace = True
            continue
        normalized.append(character)
        positions.append(index)
        in_whitespace = False

    if normalized and normalized[-1] == " ":
        normalized.pop()
        positions.pop()
    return "".join(normalized), positions
