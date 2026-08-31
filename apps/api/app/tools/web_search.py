"""SearXNG JSON search adapter. Search snippets remain candidate metadata only."""

from __future__ import annotations

from typing import Any

import httpx

from app.domain.research_tools import SearchResult
from app.tools.errors import ToolExecutionError


class SearXNGSearchProvider:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def search(self, query: str, *, limit: int = 8) -> list[SearchResult]:
        normalized = " ".join(query.split())
        if not normalized:
            raise ToolExecutionError("SEARCH_QUERY_INVALID", retryable=False)
        try:
            response = await self._client.get(
                f"{self._base_url}/search",
                params={
                    "q": normalized,
                    "format": "json",
                    "safesearch": 1,
                    "language": "auto",
                },
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

        results: list[SearchResult] = []
        for raw in payload["results"]:
            if not isinstance(raw, dict):
                continue
            mapped = _map_result(raw, rank=len(results) + 1)
            if mapped is not None:
                results.append(mapped)
            if len(results) >= limit:
                break
        return results


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
