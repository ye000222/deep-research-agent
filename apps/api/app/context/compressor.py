"""Deterministic hierarchical context compression with provenance retention."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.domain.context import CompressionLevel, ContextCandidate

_SENTENCE_SPLIT = re.compile(r"(?<=[\u3002\uFF01\uFF1F.!?])\s+|\n+")


def _estimate_tokens(content: str) -> int:
    return max(1, (len(content) + 2) // 3) if content else 0


_IMPORTANT = re.compile(r"\d|%|年|月|日|亿元|美元|CAGR|ID|citation|evidence", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CompressionResult:
    candidate: ContextCandidate
    input_hash: str
    output_hash: str
    token_before: int
    token_after: int
    validation_status: str


def compress_candidate(
    candidate: ContextCandidate, *, target_tokens: int
) -> CompressionResult | None:
    before = _estimate_tokens(candidate.content)
    if target_tokens <= 0 or before <= target_tokens:
        return None
    target_chars = max(96, target_tokens * 3)
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(candidate.content)
        if sentence.strip()
    ]
    if not sentences:
        return None
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (
            0 if _IMPORTANT.search(pair[1]) else 1,
            pair[0],
        ),
    )
    chosen: list[tuple[int, str]] = []
    used = 0
    for index, sentence in ranked:
        cost = len(sentence) + (1 if chosen else 0)
        if used + cost > target_chars:
            continue
        chosen.append((index, sentence))
        used += cost
    if not chosen:
        first = sentences[0][:target_chars].strip()
        if not first:
            return None
        chosen = [(0, first)]
    output = " ".join(sentence for _, sentence in sorted(chosen))
    after = _estimate_tokens(output)
    if not output or after >= before or after > target_tokens:
        return None
    level = (
        CompressionLevel.EXTRACTIVE if after >= max(1, before // 2) else CompressionLevel.SUMMARIZED
    )
    compressed = ContextCandidate(
        item_type=candidate.item_type,
        content=output,
        rank_score=candidate.rank_score,
        source_ref_type=candidate.source_ref_type,
        source_ref_id=candidate.source_ref_id,
        protected=candidate.protected,
        compression_level=level,
        selected_reason_code="compressed_to_fit_budget",
        provenance_refs=candidate.provenance_refs
        or tuple(
            value
            for value in (candidate.source_ref_type, candidate.source_ref_id)
            if value is not None
        ),
    )
    return CompressionResult(
        candidate=compressed,
        input_hash=hashlib.sha256(candidate.content.encode("utf-8")).hexdigest(),
        output_hash=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        token_before=before,
        token_after=after,
        validation_status="provenance_preserved",
    )
