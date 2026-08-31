"""Versioned lexical normalization for Evidence and Memory retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

RETRIEVAL_CONFIG_VERSION = "lexical-v1"
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+/-]*|[\u3400-\u9fff]")


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    latin_text: str
    cjk_lexemes: str
    fuzzy_text: str
    tokens: tuple[str, ...]


def normalize_text(value: str) -> NormalizedText:
    raw = " ".join(value.split())
    tokens = tuple(match.group(0).lower() for match in _WORD_RE.finditer(raw))
    cjk = tuple(item for item in tokens if len(item) == 1 and "\u3400" <= item <= "\u9fff")
    bigrams = tuple("".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1)))
    latin = tuple(item for item in tokens if not (len(item) == 1 and "\u3400" <= item <= "\u9fff"))
    lexemes = tuple(dict.fromkeys(cjk + bigrams))
    return NormalizedText(
        raw=raw,
        latin_text=" ".join(latin),
        cjk_lexemes=" ".join(lexemes),
        fuzzy_text=raw.lower(),
        tokens=tuple(dict.fromkeys(tokens + bigrams)),
    )


def reciprocal_rank_fusion(*ranked_lists: tuple[str, ...], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
