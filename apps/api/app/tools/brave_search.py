"""Brave Search provider adapter (Phase 12.4 — third provider).

Implements :class:`~app.domain.provider_adapter.SearchProviderAdapter` against
the Brave Search API (``GET /res/v1/web/search``). Brave is registered in the
:class:`~app.domain.provider_registry.SearchProviderRegistry` only when a
``BRAVE_API_KEY`` is configured; without a key the registry omits it and the
Router never selects it, so this adapter is never constructed on a deployment
that lacks credentials.

The adapter performs a single HTTP request per search and raises
:class:`~app.tools.errors.ToolExecutionError` on any transport failure so the
outer provider-health classifier records the failure consistently with the
existing SearXNG / Bing paths. It never mutates provider health or budget.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.domain.provider_adapter import SearchRequestContext
from app.domain.research_tools import SearchResult
from app.domain.source_policy import (
    is_stable_read_url,
    normalize_source_url,
    source_owner_key,
)
from app.tools.errors import ToolExecutionError

_PROVIDER_NAME = "Brave"
_DEFAULT_BASE_URL = "https://api.search.brave.com"
_MAX_RESULTS = 20


class BraveSearchAdapter:
    """Async Brave Search API adapter satisfying ``SearchProviderAdapter``."""

    provider_name = _PROVIDER_NAME

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        if not api_key:
            raise ValueError("BraveSearchAdapter requires a non-empty api_key")
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: SearchRequestContext | None = None,
    ) -> list[SearchResult]:
        del request_context  # Brave issues a single request; no budget inputs.
        excluded_owners = {d.strip().lower() for d in exclude_domains if d.strip()}
        excluded_urls = {normalize_source_url(u) for u in exclude_urls if u.strip()}
        params: dict[str, str | int] = {
            "q": query,
            "count": max(1, min(limit * 2, _MAX_RESULTS)),
        }
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        try:
            response = await self._client.get(
                f"{self._base_url}/res/v1/web/search",
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_TIMEOUT",
                retryable=True,
                details=self._failure_details(query, "timeout"),
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_NETWORK_ERROR",
                retryable=True,
                details=self._failure_details(query, "network_error"),
            ) from exc

        if response.status_code == 429:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_RATE_LIMIT",
                retryable=True,
                details=self._failure_details(query, "rate_limited"),
            )
        if response.status_code >= 400:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_HTTP_ERROR",
                retryable=True,
                details=self._failure_details(query, "http_error"),
            )

        payload = self._decode(response)
        return self._normalize(payload, query, limit, excluded_owners, excluded_urls)

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_BAD_PAYLOAD",
                retryable=True,
                details={"provider": _PROVIDER_NAME, "failure_type": "bad_payload"},
            ) from exc
        return data if isinstance(data, dict) else {}

    def _normalize(
        self,
        payload: dict[str, Any],
        query: str,
        limit: int,
        excluded_owners: set[str],
        excluded_urls: set[str],
    ) -> list[SearchResult]:
        web = payload.get("web")
        raw_rows = web.get("results") if isinstance(web, dict) else None
        results: list[SearchResult] = []
        seen: set[str] = set()
        rank = 0
        for row in raw_rows or []:
            if not isinstance(row, dict):
                continue
            raw_url = str(row.get("url", "")).strip()
            normalized_url = normalize_source_url(raw_url)
            if (
                not normalized_url
                or normalized_url in seen
                or normalized_url in excluded_urls
                or not is_stable_read_url(normalized_url)
            ):
                continue
            if source_owner_key(normalized_url) in excluded_owners:
                continue
            title = " ".join(str(row.get("title", "")).split())
            snippet = " ".join(str(row.get("description", "")).split())
            page_age = row.get("page_age")
            rank += 1
            results.append(
                SearchResult(
                    title=title or normalized_url,
                    url=normalized_url,
                    snippet=snippet[:4000],
                    published_at=str(page_age)[:100] if page_age else None,
                    rank=rank,
                )
            )
            seen.add(normalized_url)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _failure_details(query: str, failure_type: str) -> dict[str, object]:
        return {
            "provider": _PROVIDER_NAME,
            "failure_type": failure_type,
            "context": {
                "strategy": "brave_api",
                "query": query[:500],
                "fallback_attempted": False,
                "fallback_provider": None,
                "fallback_success": False,
            },
            "failures": [],
        }


__all__ = ["BraveSearchAdapter"]
