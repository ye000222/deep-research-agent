"""Phase 14.2 explicit ResearchContext resolution for the research loop.

The ``usage_snapshot`` key ``research_context_by_question`` is owned by this
module and nothing else:

* the write side (closure-feedback dispatch in ``research_tools``) calls
  :meth:`ResearchContextResolver.record` to project one refinement per
  question after a validated Query Candidate exists;
* the read side (Research Loop -> repository) calls
  :meth:`ResearchContextResolver.resolve` to turn a ``question_id`` back into
  a :class:`ResolvedResearchContext`.

Business code must never read ``run.usage_snapshot["research_context_by_
question"]`` directly; the loop only ever sees the resolver's typed result.
Pure domain logic: no database, no provider, no planner decisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from app.domain.research_context import ResearchContext

#: ``usage_snapshot`` key holding one refinement entry per question id.
SNAPSHOT_KEY: Final[str] = "research_context_by_question"
RESEARCH_NEED_ID_KEY: Final[str] = "research_need_id"
QUERY_PLAN_ID_KEY: Final[str] = "query_plan_id"
QUERY_CANDIDATE_ID_KEY: Final[str] = "query_candidate_id"


@dataclass(frozen=True, slots=True)
class ResolvedResearchContext:
    """One question's evidence-deficit context plus its originating pipeline ids."""

    context: ResearchContext
    research_need_id: str | None = None
    query_plan_id: str | None = None
    query_candidate_id: str | None = None


class ResearchContextResolver:
    """Single owner of the per-question ``research_context_by_question`` entry."""

    @staticmethod
    def resolve(
        snapshot: Mapping[str, object] | None,
        *,
        question_id: str,
    ) -> ResolvedResearchContext | None:
        """Read one question's refinement, or ``None`` for legacy behavior.

        Absent, malformed, or empty entries all resolve to ``None`` so the
        caller keeps the exact pre-14.2 (byte-identical, event-free) path.
        """

        if not snapshot:
            return None
        raw_entries = snapshot.get(SNAPSHOT_KEY)
        if not isinstance(raw_entries, Mapping):
            return None
        raw_entry = raw_entries.get(question_id)
        if not isinstance(raw_entry, Mapping):
            return None
        context = ResearchContext.from_mapping(raw_entry)
        if context.is_empty:
            return None
        return ResolvedResearchContext(
            context=context,
            research_need_id=_ref(raw_entry, RESEARCH_NEED_ID_KEY),
            query_plan_id=_ref(raw_entry, QUERY_PLAN_ID_KEY),
            query_candidate_id=_ref(raw_entry, QUERY_CANDIDATE_ID_KEY),
        )

    @staticmethod
    def record(
        snapshot: Mapping[str, object] | None,
        *,
        question_id: str,
        context: ResearchContext,
        research_need_id: UUID | str | None = None,
        query_plan_id: UUID | str | None = None,
        query_candidate_id: UUID | str | None = None,
    ) -> dict[str, object]:
        """Return a new usage snapshot with this question's entry replaced.

        One entry per question (latest refinement wins) so repeated feedback
        rounds cannot grow the snapshot without bound.
        """

        usage = dict(snapshot or {})
        raw_entries = usage.get(SNAPSHOT_KEY)
        entries = dict(raw_entries) if isinstance(raw_entries, Mapping) else {}
        payload: dict[str, object] = dict(context.as_dict())
        payload[RESEARCH_NEED_ID_KEY] = _as_ref(research_need_id)
        payload[QUERY_PLAN_ID_KEY] = _as_ref(query_plan_id)
        payload[QUERY_CANDIDATE_ID_KEY] = _as_ref(query_candidate_id)
        entries[question_id] = payload
        usage[SNAPSHOT_KEY] = entries
        return usage


def _as_ref(value: UUID | str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _ref(entry: Mapping[object, object], key: str) -> str | None:
    value = entry.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


__all__ = ["SNAPSHOT_KEY", "ResearchContextResolver", "ResolvedResearchContext"]
