"""Phase 14.1 evidence-aware research context.

One :class:`ResearchContext` carries the Phase 14.0 evidence-deficit
explanation (which failure reason, which missing evidence type, which refined
need, which query hints) from a ``ResearchNeed`` down the existing query
pipeline (intent -> plan -> candidate) so that a follow-up query can target
the *missing condition* instead of "more material".

Pure domain logic: it never touches the database, never triggers Search, and
reads only generic status strings produced by the analyzers.  A context built
from an empty mapping is ``is_empty`` and every consumer must behave exactly as
before Phase 14.1, which keeps the Phase 13.x artifacts unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from app.domain.gap_research_intent import GapResearchIntent

#: Metadata key written by ``feedback_analysis_metadata`` in Phase 14.0.
PRIMARY_FAILURE_REASON_KEY: Final[str] = "primary_failure_reason"
EVIDENCE_FAILURE_REASON_KEY: Final[str] = "evidence_failure_reason"
MISSING_EVIDENCE_TYPE_KEY: Final[str] = "missing_evidence_type"
REFINED_NEED_TYPE_KEY: Final[str] = "refined_need_type"
QUERY_HINTS_KEY: Final[str] = "query_hints"

#: Phase 15.1 Branch B — independent-source targeting exclusion metadata.  These
#: keys carry the canonical *owner* identities (registrable domains) that already
#: back a Requirement so a follow-up query can prefer a genuinely new publisher.
#: They are execution/eligibility metadata, never stitched into the query text.
EXISTING_SOURCE_OWNERS_KEY: Final[str] = "existing_source_owners"
EXISTING_DOMAINS_KEY: Final[str] = "existing_domains"
REQUIRED_SOURCE_COUNT_KEY: Final[str] = "required_source_count"
MISSING_SOURCE_COUNT_KEY: Final[str] = "missing_source_count"
TARGET_CLAIM_ID_KEY: Final[str] = "target_claim_id"
PROPOSITION_GAP_INTENTS_KEY: Final[str] = "proposition_gap_intents"


@dataclass(frozen=True, slots=True)
class ResearchContext:
    """The evidence-deficit explanation one research need carries downstream."""

    evidence_failure_reason: str | None = None
    missing_evidence_type: str | None = None
    refined_need_type: str | None = None
    query_hints: tuple[str, ...] = ()
    #: Phase 15.1 Branch B: owner/domain exclusion set plus the numeric source
    #: bar.  Empty/``None`` keeps the context byte-identical to Phase 14.x.
    existing_source_owners: tuple[str, ...] = ()
    existing_domains: tuple[str, ...] = ()
    required_source_count: int | None = None
    missing_source_count: int | None = None
    target_claim_id: str | None = None
    #: V2.4 Step 2: typed proposition-scoped advisory metadata. It is kept
    #: separate from legacy free-text hints and omitted from legacy payloads.
    proposition_gap_intents: tuple[GapResearchIntent, ...] = ()
    current_question: str | None = None
    original_query: str | None = None
    missing_requirement: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposition_gap_intents, tuple):
            raise TypeError("proposition_gap_intents_must_be_immutable_tuple")
        if any(not isinstance(item, GapResearchIntent) for item in self.proposition_gap_intents):
            raise TypeError("proposition_gap_intents_must_be_validated_intents")
        ordered = tuple(
            sorted(self.proposition_gap_intents, key=lambda item: item.priority_ordering_key)
        )
        refs = tuple(item.intent_ref for item in ordered)
        if ordered != self.proposition_gap_intents or len(refs) != len(set(refs)):
            raise ValueError("proposition_gap_intents_must_be_canonical_and_unique")

    @property
    def is_empty(self) -> bool:
        return not (
            self.evidence_failure_reason
            or self.missing_evidence_type
            or self.refined_need_type
            or self.query_hints
            or self.existing_source_owners
            or self.existing_domains
            or self.required_source_count
            or self.missing_source_count
            or self.target_claim_id
            or self.proposition_gap_intents
            or self.current_question
            or self.original_query
            or self.missing_requirement
        )

    @property
    def independent_source_shortage(self) -> bool:
        """True when this context targets a genuine post-A owner shortage."""

        return self.missing_source_count is not None and self.missing_source_count > 0

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence_failure_reason": self.evidence_failure_reason,
            "missing_evidence_type": self.missing_evidence_type,
            "refined_need_type": self.refined_need_type,
            "query_hints": list(self.query_hints),
            EXISTING_SOURCE_OWNERS_KEY: list(self.existing_source_owners),
            EXISTING_DOMAINS_KEY: list(self.existing_domains),
            REQUIRED_SOURCE_COUNT_KEY: self.required_source_count,
            MISSING_SOURCE_COUNT_KEY: self.missing_source_count,
            TARGET_CLAIM_ID_KEY: self.target_claim_id,
        }
        if self.proposition_gap_intents:
            result[PROPOSITION_GAP_INTENTS_KEY] = [
                intent.model_dump(mode="json")
                for intent in self.proposition_gap_intents
            ]
        for name in ("current_question", "original_query", "missing_requirement"):
            if getattr(self, name):
                result[name] = getattr(self, name)
        return result

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, object] | None) -> ResearchContext:
        """Read one context out of feedback/need metadata without validating identity.

        Unknown or malformed values are dropped instead of raising so that
        historical Phase 13.x payloads simply yield an empty context.
        """

        if not metadata:
            return EMPTY_RESEARCH_CONTEXT
        typed_intents: tuple[GapResearchIntent, ...] = ()
        if PROPOSITION_GAP_INTENTS_KEY in metadata:
            raw_intents = metadata[PROPOSITION_GAP_INTENTS_KEY]
            if not isinstance(raw_intents, (list, tuple)):
                raise ValueError("proposition_gap_intents_must_be_a_sequence")
            typed_intents = tuple(
                GapResearchIntent.model_validate(item) for item in raw_intents
            )
        reason = _text(metadata, EVIDENCE_FAILURE_REASON_KEY) or _text(
            metadata, PRIMARY_FAILURE_REASON_KEY
        )
        return cls(
            evidence_failure_reason=reason,
            missing_evidence_type=_text(metadata, MISSING_EVIDENCE_TYPE_KEY),
            refined_need_type=_text(metadata, REFINED_NEED_TYPE_KEY),
            query_hints=_texts(metadata, QUERY_HINTS_KEY),
            existing_source_owners=_texts(metadata, EXISTING_SOURCE_OWNERS_KEY),
            existing_domains=_texts(metadata, EXISTING_DOMAINS_KEY),
            required_source_count=_int(metadata, REQUIRED_SOURCE_COUNT_KEY),
            missing_source_count=_int(metadata, MISSING_SOURCE_COUNT_KEY),
            target_claim_id=_text(metadata, TARGET_CLAIM_ID_KEY),
            proposition_gap_intents=typed_intents,
            current_question=_text(metadata, "current_question"),
            original_query=_text(metadata, "original_query"),
            missing_requirement=_text(metadata, "missing_requirement"),
        )


EMPTY_RESEARCH_CONTEXT: Final[ResearchContext] = ResearchContext()


def meaningful_context(context: ResearchContext | None) -> ResearchContext | None:
    """Collapse an empty context into ``None`` so carriers keep their old shape."""

    if context is None or context.is_empty:
        return None
    return context


def query_hint_additions(
    query_text: str, hints: Iterable[str]
) -> tuple[str, tuple[str, ...]]:
    """Return ``(final text, hints actually appended)`` for one pass.

    The base query text is never rewritten, reordered, or removed; hints that
    already appear (case-insensitively) are skipped, so the result is stable
    and idempotent for the same input.  This is the single shared
    implementation behind :func:`apply_query_hints` (Phase 14.1) and the
    Phase 14.2 :class:`ResearchContextEnricher`, so the two can never diverge.
    """

    hint_tokens = tuple(hints)
    if not hint_tokens:
        # No hints means no adaptation: keep the base expression byte-for-byte.
        return query_text, ()
    text = " ".join(query_text.split())
    applied: list[str] = []
    for hint in hint_tokens:
        token = " ".join(hint.split())
        if not token or token.casefold() in text.casefold():
            continue
        text = f"{text} {token}"
        applied.append(token)
    return text, tuple(applied)


def apply_query_hints(query_text: str, hints: Iterable[str]) -> str:
    """Append missing hint terms to one query expression (Phase 14.1 carrier)."""

    return query_hint_additions(query_text, hints)[0]


def independent_source_eligibility(
    source_owner_key: str | None,
    existing_source_owners: Iterable[str],
) -> bool:
    """Whether one fetched source advances *this* independent-source requirement.

    Phase 15.1 Task O.  A candidate whose canonical owner is already counted for
    the Requirement is "not independent for this requirement" (returns False);
    a new owner returns True.  The verdict is deliberately requirement-scoped:
    the caller must keep a False result usable for *other* requirements — this
    helper never discards evidence, and it never inspects the owner for anything
    but membership in the existing set (no role / question / domain logic).
    """

    owner = (source_owner_key or "").strip().casefold()
    if not owner or owner == "unknown":
        return False
    existing = {
        str(value).strip().casefold()
        for value in existing_source_owners
        if str(value).strip()
    }
    return owner not in existing


def _text(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _texts(metadata: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        normalized = entry.strip()
        if normalized and normalized not in items:
            items.append(normalized)
    return tuple(items)


def _int(metadata: Mapping[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


__all__ = [
    "EMPTY_RESEARCH_CONTEXT",
    "EXISTING_DOMAINS_KEY",
    "EXISTING_SOURCE_OWNERS_KEY",
    "MISSING_SOURCE_COUNT_KEY",
    "PROPOSITION_GAP_INTENTS_KEY",
    "REQUIRED_SOURCE_COUNT_KEY",
    "TARGET_CLAIM_ID_KEY",
    "ResearchContext",
    "apply_query_hints",
    "independent_source_eligibility",
    "meaningful_context",
    "query_hint_additions",
]
