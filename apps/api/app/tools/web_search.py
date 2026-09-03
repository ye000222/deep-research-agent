"""SearXNG JSON search adapter. Search snippets remain candidate metadata only."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.domain.research_tools import SearchResult
from app.tools.errors import ToolExecutionError

_SEARCH_STRATEGIES: tuple[tuple[dict[str, str], int], ...] = (
    ({}, 1),
    # The default general engines are frequently rate-limited or blocked on
    # Chinese networks. Sogou provides a genuinely independent general-web
    # fallback instead of repeatedly asking the same degraded engine group.
    ({"engines": "sogou"}, 2),
    # Public research indexes are especially useful for technical questions
    # and usually expose readable abstract pages that the evidence reader can
    # verify without a paywall.
    ({"engines": "arxiv,openairepublications"}, 2),
)


class SearXNGSearchProvider:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def search(self, query: str, *, limit: int = 8) -> list[SearchResult]:
        normalized = " ".join(query.split())
        if not normalized:
            raise ToolExecutionError("SEARCH_QUERY_INVALID", retryable=False)

        results_by_url: dict[str, SearchResult] = {}
        healthy_strategy_seen = False
        for strategy, attempts in _SEARCH_STRATEGIES:
            for attempt in range(attempts):
                payload = await self._request(normalized, strategy=strategy)
                unresponsive = payload.get("unresponsive_engines")
                strategy_unresponsive = isinstance(unresponsive, list) and bool(unresponsive)
                healthy_strategy_seen = healthy_strategy_seen or not strategy_unresponsive

                raw_results = payload.get("results")
                if not isinstance(raw_results, list):
                    raise ToolExecutionError("SEARCH_RESPONSE_INVALID", retryable=False)
                for raw in raw_results:
                    if not isinstance(raw, dict):
                        continue
                    mapped = _map_result(raw, rank=len(results_by_url) + 1)
                    if mapped is None or mapped.url in results_by_url:
                        continue
                    results_by_url[mapped.url] = mapped
                    if len(results_by_url) >= limit:
                        break
                if raw_results or len(results_by_url) >= limit:
                    break
                if strategy_unresponsive and attempt + 1 < attempts:
                    await asyncio.sleep(0.5)
            if len(results_by_url) >= limit:
                break

        if not results_by_url and not healthy_strategy_seen:
            raise ToolExecutionError("SEARCH_PROVIDER_DEGRADED", retryable=True)
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(results_by_url.values(), start=1)
        ]

    async def _request(self, query: str, *, strategy: dict[str, str]) -> dict[str, Any]:
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "safesearch": 1,
            "language": "auto",
            **strategy,
        }
        try:
            response = await self._client.get(
                f"{self._base_url}/search",
                params=params,
                timeout=20.0,
            )
        except httpx.TimeoutException as exc:
            raise ToolExecutionError("SEARCH_TIMEOUT", retryable=True) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError("SEARCH_NETWORK_ERROR", retryable=True) from exc
        if response.status_code >= 500:
            raise ToolExecutionError("SEARCH_PROVIDER_UNAVAILABLE", retryable=True)
        if response.status_code >= 400:
            raise ToolExecutionError("SEARCH_REQUEST_REJECTED", retryable=False)
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise ToolExecutionError("SEARCH_RESPONSE_INVALID", retryable=False) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ToolExecutionError("SEARCH_RESPONSE_INVALID", retryable=False)
        return payload


def _map_result(raw: dict[str, Any], *, rank: int) -> SearchResult | None:
    title = raw.get("title")
    url = raw.get("url")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    snippet = raw.get("content")
    published = raw.get("publishedDate")
    return SearchResult(
        title=title.strip(),
        url=url.strip(),
        snippet=snippet.strip() if isinstance(snippet, str) else "",
        published_at=published.strip() if isinstance(published, str) else None,
        rank=rank,
    )
