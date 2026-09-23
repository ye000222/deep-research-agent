"""DuckDuckGo HTML provider adapter (Phase 12.4 — third/fallback provider).

A key-free fallback search provider used for development and benchmark runs so
the Router has a third real provider to reroute to when both SearXNG and Bing
are unhealthy. It scrapes the lightweight ``html.duckduckgo.com/html/``
endpoint — no API key, no external SDK — mirroring the existing direct-Bing
HTML fallback pattern in :mod:`app.tools.web_search`.

DuckDuckGo is *not* a production-preferred provider; it sits below Bing and
above SearXNG in the registry priority so it only receives traffic when the
two primary providers are degraded. Like the Brave adapter it raises
:class:`~app.tools.errors.ToolExecutionError` on transport failure and never
mutates provider health or budget.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from app.domain.provider_adapter import SearchRequestContext
from app.domain.research_tools import SearchResult
from app.domain.source_policy import (
    is_stable_read_url,
    normalize_source_url,
    source_owner_key,
)
from app.tools.errors import ToolExecutionError

_PROVIDER_NAME = "DuckDuckGo"
_DEFAULT_ENDPOINT = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# One result block: title link (result__a) plus an optional snippet link.
_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>'
    r'(?P<title>.*?)</a>'
    r'(?:.*?<a[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>'
    r'(?P<snippet>.*?)</a>)?',
    flags=re.IGNORECASE | re.DOTALL,
)


class DuckDuckGoAdapter:
    """Async DuckDuckGo HTML adapter satisfying ``SearchProviderAdapter``."""

    provider_name = _PROVIDER_NAME

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
    ) -> None:
        self._client = client
        self._endpoint = endpoint

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: SearchRequestContext | None = None,
    ) -> list[SearchResult]:
        del request_context  # DuckDuckGo issues a single request; no budget inputs.
        excluded_owners = {d.strip().lower() for d in exclude_domains if d.strip()}
        excluded_urls_set = {
            normalize_source_url(u) for u in exclude_urls if u.strip()
        }
        try:
            response = await self._client.post(
                self._endpoint,
                data={"q": query},
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                },
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
        body = response.text
        if not body.strip():
            raise ToolExecutionError(
                "SEARCH_PROVIDER_EMPTY_RESPONSE",
                retryable=True,
                details=self._failure_details(query, "empty_response"),
            )
        return self._parse(
            body, limit, excluded_owners, excluded_urls_set
        )

    def _parse(
        self,
        body: str,
        limit: int,
        excluded_owners: set[str],
        excluded_urls: set[str],
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        seen: set[str] = set()
        rank = 0
        for match in _RESULT_RE.finditer(body):
            raw_url = _unwrap_ddg_redirect(html.unescape(match.group("url")).strip())
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
            title = _strip_tags(match.group("title"))
            snippet = _strip_tags(match.group("snippet") or "")
            rank += 1
            results.append(
                SearchResult(
                    title=title or normalized_url,
                    url=normalized_url,
                    snippet=snippet[:4000],
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
                "strategy": "duckduckgo_html_fallback",
                "query": query[:500],
                "fallback_attempted": False,
                "fallback_provider": None,
                "fallback_success": False,
            },
            "failures": [],
        }


def _unwrap_ddg_redirect(raw_href: str) -> str:
    """Resolve DuckDuckGo's ``/l/?uddg=<encoded>`` redirect wrapper."""

    if not raw_href:
        return raw_href
    href = raw_href
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if parts.path.startswith("/l/") and "uddg" in parts.query:
        targets = parse_qs(parts.query).get("uddg")
        if targets:
            return unquote(targets[0])
    return href


def _strip_tags(fragment: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(fragment)).split())


__all__ = ["DuckDuckGoAdapter"]
