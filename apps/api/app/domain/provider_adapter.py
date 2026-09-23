"""Unified search provider adapter contract (Phase 12.4).

The :class:`SearchProviderAdapter` protocol is the single interface the Router
and (in Step 2) the research loop dispatch table depend on. Every search
provider — SearXNG, Bing, Brave, DuckDuckGo — exposes the same
``async search(query, limit=..., ...)`` call and returns a normalized
``list[SearchResult]``. Adding a provider therefore only requires a new adapter
that satisfies this protocol plus a :class:`~app.domain.provider_registry.ProviderCapability`
registration; no Router or dispatch code changes.

Design note on the return type: the Phase 12.4 spec sketches ``search()``
returning a ``ProviderExecutionResult``. That type already exists in
:mod:`app.domain.provider_executor` and models *executor telemetry* (status,
latency, classified failure, provider name) around a raw
:class:`~app.domain.provider_executor.ProviderCall`. The adapter contract here is
intentionally lower-level: it returns the same ``list[SearchResult]`` the
existing :meth:`SearXNGSearchProvider.search` returns so the loop can adopt a
uniform dispatch map in Step 2 without re-wrapping every provider and without
disturbing the ``record_search_results`` contract. The executor keeps wrapping
adapters into ``ProviderExecutionResult`` exactly as it does today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.domain.research_tools import SearchResult


@dataclass(frozen=True, slots=True)
class SearchRequestContext:
    """Per-attempt execution inputs a caller passes to every adapter.

    Only :class:`~app.tools.web_search.SearXNGSearchProvider` consumes the
    budget-aware fields (``provider_request_allowance`` / ``alternate_query``);
    the standalone HTTP adapters (Brave, DuckDuckGo) accept and ignore them,
    which keeps a single uniform dispatch call in the research loop while
    preserving the existing SearXNG budget/query inputs unchanged.
    """

    provider_request_allowance: int | None = None
    alternate_query: str | None = None


@runtime_checkable
class SearchProviderAdapter(Protocol):
    """Minimal async contract every search provider adapter must satisfy."""

    provider_name: str

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: SearchRequestContext | None = None,
    ) -> list[SearchResult]:
        """Run one web search and return normalized candidate results.

        Implementations raise on transport failure so the caller's failure
        classifier can record provider health; an empty (non-error) result set
        is returned as an empty list. ``exclude_domains`` / ``exclude_urls``
        let the caller avoid re-reading pages already consumed for this target.
        ``request_context`` carries provider-specific inputs (budget allowance,
        alternate query) that only some adapters consume.
        """
        ...


__all__ = ["SearchProviderAdapter", "SearchRequestContext"]
