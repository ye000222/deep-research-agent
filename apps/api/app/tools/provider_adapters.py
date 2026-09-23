"""Adapter wrappers that unify the legacy providers under one contract (Phase 12.4).

:class:`SearXNGSearchProvider` predates the :class:`SearchProviderAdapter`
protocol: its ``search``/``bing_only_search`` methods take provider-specific
inputs and expose ``last_*`` telemetry attributes that the research loop reads
for budget accounting. These thin wrappers let the loop dispatch SearXNG and
Bing through the *same* adapter map as Brave / DuckDuckGo — with no
provider-string branching — while delegating straight to the original methods so
search strategy, budget inputs and telemetry stay byte-for-byte identical.

``execution_source`` exposes the underlying provider so the loop keeps reading
its ``last_request_count`` / ``last_*`` counters from the real object instead of
the wrapper, preserving the exact pre-12.4 budget accounting numbers.
"""

from __future__ import annotations

from app.domain.provider_adapter import SearchRequestContext
from app.domain.research_tools import SearchResult
from app.tools.web_search import SearXNGSearchProvider

_SEARXNG_NAME = "SearXNG"
_BING_NAME = "Bing"


class SearXNGAdapter:
    """Route the health-selected SearXNG call through the adapter contract."""

    provider_name = _SEARXNG_NAME

    def __init__(self, provider: SearXNGSearchProvider) -> None:
        self._provider = provider

    @property
    def execution_source(self) -> SearXNGSearchProvider:
        return self._provider

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: SearchRequestContext | None = None,
    ) -> list[SearchResult]:
        allowance = request_context.provider_request_allowance if request_context else None
        alternate = request_context.alternate_query if request_context else None
        return await self._provider.search(
            query,
            limit=limit,
            max_provider_requests=allowance,
            alternate_query=alternate,
            exclude_domains=exclude_domains,
            exclude_urls=exclude_urls,
            single_provider_only=True,
        )


class BingAdapter:
    """Route the health-selected Bing call through the adapter contract."""

    provider_name = _BING_NAME

    def __init__(self, provider: SearXNGSearchProvider) -> None:
        self._provider = provider

    @property
    def execution_source(self) -> SearXNGSearchProvider:
        return self._provider

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: SearchRequestContext | None = None,
    ) -> list[SearchResult]:
        del request_context  # Bing HTML fallback has no per-request budget input.
        return await self._provider.bing_only_search(
            query,
            limit=limit,
            exclude_domains=exclude_domains,
            exclude_urls=exclude_urls,
        )


__all__ = ["BingAdapter", "SearXNGAdapter"]
