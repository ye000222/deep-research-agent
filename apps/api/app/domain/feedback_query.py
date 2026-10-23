"""Deterministic topic binding for legacy closure queries, before any Search."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.passage_relevance import events

_GENERIC = re.compile(
    r"\b(?:primary|source|sources|official|report|authoritative|evidence|reliable|"
    r"confirmation|verify|verification|find|need|needed|require|requires|the|a|an|"
    r"what|when|how|which|is|was|were|did|does|of|for|to|in|on|and|date)\b"
    r"|需要|要求|官方|来源|证据|报告|可靠|权威|验证|确认|何时|什么时候|日期|是什么",
    re.IGNORECASE,
)
_IDS = re.compile(r"\bq[\w-]*:d\d+\b", re.IGNORECASE)
_EVENT_WORDS = re.compile(
    r"正式|成功|开通服务|投入运营|投入运行|开始建设|建设|建成|开通|发射|何时|"
    r"\b(?:launch(?:ed)?|opened|commissioned|commissioning|construction|"
    r"completed?|successfully|officially|final|last|happened)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class FeedbackQuery:
    query: str
    topic_anchor_present: bool
    event_anchor_present: bool
    reason: str

    @property
    def allowed(self) -> bool:
        return self.topic_anchor_present and self.event_anchor_present


def _terms(text: str) -> set[str]:
    text = _EVENT_WORDS.sub(" ", _GENERIC.sub(" ", _IDS.sub(" ", text))).casefold()
    return set(re.findall(r"[a-z][a-z0-9-]+|[\u3400-\u9fff]{2,}", text))


def bind_feedback_query(
    query: str, *, current_question: str, original_query: str = "", requirement: str = ""
) -> FeedbackQuery:
    """Keep the current question's subject/event; never send a bare gap template.

    A compact current question is the anchor, not the entire original task.
    The original question is used only if that anchor is unavailable. Unknown
    event vocabularies must retain every informative question term instead of
    guessing an event or entity. This is routing, never a factual verifier.
    """
    anchor = current_question.strip()
    if not _terms(anchor):
        anchor = re.split(r"[\uff1f?。;\uff1b\n]", original_query, maxsplit=1)[0].strip()
    anchor = re.sub(r"\s+", " ", anchor)[:300].strip()
    terms = _terms(anchor)
    if not terms:
        return FeedbackQuery("", False, False, "feedback_query_missing_topic_anchor")

    def quality(value: str) -> tuple[bool, bool]:
        folded = value.casefold()
        topic = any(term in folded for term in terms)
        required_events = events(anchor)
        event = (
            bool(required_events & events(value))
            if required_events
            else all(term in folded for term in terms)
        )
        return topic, event

    cleaned = " ".join(_IDS.sub(" ", query).split())
    topic, event = quality(cleaned)
    if topic and event:
        return FeedbackQuery(cleaned, topic, event, "feedback_query_anchored")
    # Preserve the missing-evidence requirement, but not internal dimension IDs.
    gap = " ".join(_IDS.sub(" ", requirement).split())[:160]
    suffix = cleaned[:160] or "official source evidence"
    repaired = " ".join(dict.fromkeys(part for part in (anchor, gap, suffix) if part))
    topic, event = quality(repaired)
    return FeedbackQuery(repaired, topic, event, "feedback_query_anchor_repaired")
