"""Phase 14.2 evidence-aware query enrichment for the research loop.

One :class:`ResearchContextEnricher` pass is the only new transformation
between the Planner producing a ``SearchTarget`` and the Research Loop
handing ``SearchTarget.query`` to a real Provider:

    Planner -> SearchTarget -> ResearchContextEnricher -> enriched SearchTarget

It is a pure string layer by design.  It never decides the research
direction, the Search Strategy, the Provider, or any Budget; it only appends
the missing-evidence hint terms carried by an already-produced
``ResearchContext`` (Closure Feedback -> ResearchNeed -> ResearchContext).

Invariants enforced here:

* ``context is None`` (or hint-less) keeps the query byte-for-byte identical;
* the original query text is never removed or rewritten;
* enrichment is idempotent: ``enrich(enrich(q, c).enriched_query, c)``
  reproduces the same enriched text because applied hints are skipped later.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.research_context import ResearchContext, meaningful_context, query_hint_additions


@dataclass(frozen=True, slots=True)
class ResearchContextEnrichmentResult:
    """Outcome of one enrichment pass over a single query expression."""

    original_query: str
    enriched_query: str
    applied_hints: tuple[str, ...]
    changed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original_query,
            "enriched_query": self.enriched_query,
            "applied_hints": list(self.applied_hints),
            "changed": self.changed,
        }


class ResearchContextEnricher:
    """Append missing-evidence hints to one SearchTarget query expression.

    Stateless: one shared instance can serve every run and every question.
    The behavior is identical to the Phase 14.1 candidate-side hint injection
    because both call the same :func:`query_hint_additions` implementation.
    """

    def enrich(
        self,
        query: str,
        context: ResearchContext | None,
    ) -> ResearchContextEnrichmentResult:
        meaningful = meaningful_context(context)
        if meaningful is None or not meaningful.query_hints:
            # No context (or a context without hints): byte-identical pass.
            return ResearchContextEnrichmentResult(
                original_query=query,
                enriched_query=query,
                applied_hints=(),
                changed=False,
            )
        enriched, applied = query_hint_additions(query, meaningful.query_hints)
        if not applied:
            # Every hint already appears in the base query: keep the original
            # bytes so a no-op enrichment never touches the SearchTarget.
            return ResearchContextEnrichmentResult(
                original_query=query,
                enriched_query=query,
                applied_hints=(),
                changed=False,
            )
        return ResearchContextEnrichmentResult(
            original_query=query,
            enriched_query=enriched,
            applied_hints=applied,
            changed=enriched != query,
        )


__all__ = ["ResearchContextEnricher", "ResearchContextEnrichmentResult"]
