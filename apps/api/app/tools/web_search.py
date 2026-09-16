"""SearXNG JSON search adapter. Search snippets remain candidate metadata only."""

from __future__ import annotations

import asyncio
import base64
import html
import os
import re
import time
from collections.abc import Set
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx

from app.domain.research_tools import SearchResult
from app.domain.source_policy import is_stable_read_url, normalize_source_url, source_owner_key
from app.tools.errors import ToolExecutionError

_ACADEMIC_ENGINES = (
    "google scholar,semantic scholar,openalex,openairepublications,crossref"
)
_DEFAULT_SEARCH_STRATEGIES: tuple[tuple[dict[str, str], int], ...] = (
    # Public research indexes are especially useful for technical questions
    # and usually expose readable abstract pages that the evidence reader can
    # verify without a paywall. Put this before broad fallbacks so a three-call
    # allowance cannot permanently starve the authoritative lane.
    ({"engines": _ACADEMIC_ENGINES}, 1),
    # Use the bilingual planner hint on an independent general-web lane.
    # Repeating the Chinese query across two engine groups produced the same
    # publisher/DOI redirects and starved readable English manufacturer and
    # university pages.
    ({"language": "en"}, 1),
    # The default general engines are frequently rate-limited or blocked on
    # Chinese networks. Sogou is a final independent fallback; one attempt is
    # enough because run-level query variants provide bounded retries.
    ({"engines": "sogou"}, 1),
)
# Prefer an operator-selected engine when the deployment has a known stable
# egress path.  Keep the existing strategy chain as a fallback so local and
# test environments remain unchanged when the variable is absent.
_PREFERRED_ENGINES = os.getenv("SEARXNG_PREFERRED_ENGINES", "").strip()
_SEARCH_STRATEGIES: tuple[tuple[dict[str, str], int], ...] = (
    (
        _DEFAULT_SEARCH_STRATEGIES[0],
        ({"engines": _PREFERRED_ENGINES}, 1),
        *_DEFAULT_SEARCH_STRATEGIES[1:],
    )
    if _PREFERRED_ENGINES
    else _DEFAULT_SEARCH_STRATEGIES
)
_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SECONDS = 15.0
_HEDGE_DELAY_SECONDS = 1.8
_STRATEGY_CIRCUIT_COOLDOWN_SECONDS = 120.0
_MAX_RESULTS_PER_OWNER = 2
_TARGET_RESULTS_PER_QUERY = 8
_MAX_ACADEMIC_RESULTS_PER_QUERY = 4
_MAX_EXCLUDED_DOMAINS = 6
_PROVIDER_NAME = "SearXNG"
_FALLBACK_PROVIDER_NAME = "Bing"
_PROVIDER_HEALTHY = "healthy"
_PROVIDER_DEGRADED = "degraded"
_PROVIDER_COOLDOWN = "cooldown"


def _state_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _state_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
_GENERAL_FIRST_QUERY_MARKERS = (
    "厂商",
    "厂家",
    "产品",
    "供应商",
    "客户案例",
    "市场规模",
    "增长率",
    "市场份额",
    "竞争格局",
    "vendor",
    "manufacturer",
    "supplier",
    "product",
    "customer case",
    "market size",
    "growth rate",
    "market share",
    "competitive landscape",
)
_GENERIC_QUERY_TOKENS = {
    "official",
    "documentation",
    "primary",
    "source",
    "government",
    "university",
    "paper",
    "association",
    "manufacturer",
    "report",
    "study",
    "case",
    "site",
    "官方",
    "原始",
    "来源",
    "政府",
    "高校",
    "论文",
    "行业",
    "协会",
    "制造",
    "官网",
    "文档",
}
_TOPIC_DOMAIN_ANCHORS = {
    "工业",
    "缺陷",
    "机器",
    "视觉",
    "质检",
    "表面",
    "industrial",
    "defect",
    "machine",
    "vision",
    "inspection",
}
_INDUSTRIAL_CONTEXT_ANCHORS = {
    "工业",
    "industrial",
    "制造",
    "manufacturing",
    "产线",
    "生产线",
    "factory",
}
_DEFECT_INSPECTION_ANCHORS = {
    "缺陷",
    "defect",
    "质检",
    "检测",
    "inspection",
    "异常",
    "anomaly",
    "表面",
    "surface",
}
_VISUAL_TOPIC_ANCHORS = {
    "视觉",
    "图像",
    "影像",
    "相机",
    "光学",
    "机器视觉",
    "vision",
    "visual",
    "image",
    "imaging",
    "camera",
    "optical",
}
_NON_VISUAL_ANOMALY_ANCHORS = {
    "控制系统",
    "工业控制",
    "网络异常",
    "入侵检测",
    "过程异常",
    "控制网络",
    "control system",
    "industrial control",
    "network anomaly",
    "intrusion detection",
    "process anomaly",
    "control network",
}


class SearXNGSearchProvider:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        fallback_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client
        # SearXNG is internal/direct, while public fallback traffic must use
        # the configured egress proxy. The worker passes its public reader
        # client here; tests and local adapters can omit it.
        self._fallback_client = fallback_client or client
        self._base_url = base_url.rstrip("/")
        self._failure_streak = 0
        self._circuit_open_until = 0.0
        self._last_request_count = 0
        self._last_timeout_count = 0
        self._last_fallback_count = 0
        self._last_healthy_response_count = 0
        self._last_unresponsive_response_count = 0
        self._last_productive_response_count = 0
        self._last_network_error_count = 0
        self._last_http_error_count = 0
        self._last_empty_response_count = 0
        self._last_invalid_response_count = 0
        self._last_fallback_attempt_count = 0
        self._last_fallback_success_count = 0
        self._last_circuit_open_count = 0
        self._failure_events: list[dict[str, object]] = []
        self._events: list[dict[str, object]] = []
        self._provider_states: dict[str, dict[str, object]] = {
            _PROVIDER_NAME: {
                "state": _PROVIDER_HEALTHY,
                "failure_streak": 0,
                "cooldown_until": 0.0,
            },
            _FALLBACK_PROVIDER_NAME: {
                "state": _PROVIDER_HEALTHY,
                "failure_streak": 0,
                "cooldown_until": 0.0,
            },
        }
        self._last_failure_type: str | None = None
        self._strategy_circuit_open_until: dict[str, float] = {}

    @property
    def last_request_count(self) -> int:
        """Number of upstream HTTP attempts made by the last logical search."""

        return self._last_request_count

    @property
    def last_timeout_count(self) -> int:
        return self._last_timeout_count

    @property
    def last_fallback_count(self) -> int:
        return self._last_fallback_count

    @property
    def last_healthy_response_count(self) -> int:
        return self._last_healthy_response_count

    @property
    def last_unresponsive_response_count(self) -> int:
        return self._last_unresponsive_response_count

    @property
    def last_productive_response_count(self) -> int:
        return self._last_productive_response_count

    @property
    def last_network_error_count(self) -> int:
        return self._last_network_error_count

    @property
    def last_http_error_count(self) -> int:
        return self._last_http_error_count

    @property
    def last_empty_response_count(self) -> int:
        return self._last_empty_response_count

    @property
    def last_invalid_response_count(self) -> int:
        return self._last_invalid_response_count

    @property
    def last_fallback_attempt_count(self) -> int:
        return self._last_fallback_attempt_count

    @property
    def last_fallback_success_count(self) -> int:
        return self._last_fallback_success_count

    @property
    def last_circuit_open_count(self) -> int:
        return self._last_circuit_open_count

    @property
    def state(self) -> str:
        return str(self._provider_states[_PROVIDER_NAME]["state"])

    @state.setter
    def state(self, value: str) -> None:
        self._provider_states[_PROVIDER_NAME]["state"] = value

    @property
    def failure_streak(self) -> int:
        return _state_int(self._provider_states[_PROVIDER_NAME]["failure_streak"])

    @property
    def cooldown_until(self) -> float:
        return _state_float(self._provider_states[_PROVIDER_NAME]["cooldown_until"])

    @cooldown_until.setter
    def cooldown_until(self, value: float) -> None:
        self._provider_states[_PROVIDER_NAME]["cooldown_until"] = float(value)
        self._circuit_open_until = float(value)

    @property
    def last_failure_type(self) -> str | None:
        return self._last_failure_type

    @property
    def events(self) -> list[dict[str, object]]:
        return list(self._events)

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        max_provider_requests: int | None = None,
        alternate_query: str | None = None,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
    ) -> list[SearchResult]:
        self._last_request_count = 0
        self._last_timeout_count = 0
        self._last_fallback_count = 0
        self._last_healthy_response_count = 0
        self._last_unresponsive_response_count = 0
        self._last_productive_response_count = 0
        self._last_network_error_count = 0
        self._last_http_error_count = 0
        self._last_empty_response_count = 0
        self._last_invalid_response_count = 0
        self._last_fallback_attempt_count = 0
        self._last_fallback_success_count = 0
        self._last_circuit_open_count = 0
        self._failure_events = []
        normalized = " ".join(query.split())
        normalized_alternate = " ".join((alternate_query or "").split()) or normalized
        normalized_excluded_owners = frozenset(
            domain.casefold().strip().lstrip(".")
            for domain in exclude_domains
            if domain and domain != "unknown"
        )
        normalized_excluded_urls = frozenset(
            normalize_source_url(url) for url in exclude_urls if url
        )
        exclusions = " ".join(
            f"-site:{domain}"
            for domain in tuple(normalized_excluded_owners)[:_MAX_EXCLUDED_DOMAINS]
            if domain and domain != "unknown" and all(part for part in domain.split("."))
        )
        if exclusions:
            query_room = max(1, 400 - len(exclusions) - 1)
            normalized = f"{normalized[:query_room]} {exclusions}"
            normalized_alternate = f"{normalized_alternate[:query_room]} {exclusions}"
        if not normalized:
            raise ToolExecutionError("SEARCH_QUERY_INVALID", retryable=False)
        now = time.monotonic()
        if now < self._circuit_open_until:
            self._last_circuit_open_count += 1
            raise ToolExecutionError(
                "SEARCH_PROVIDER_DEGRADED",
                retryable=True,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="circuit_open",
                        strategy="global",
                        query=normalized,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                        extra={
                            "retry_after_seconds": round(
                                self._circuit_open_until - now, 1
                            )
                        },
                    )
                ),
            )

        results_by_url: dict[str, SearchResult] = {}
        results_by_owner: dict[str, int] = {}
        failures: list[dict[str, object]] = []
        hedged_strategy_indexes: set[int] = set()
        strategies = _strategies_for_query(normalized, normalized_alternate)
        for strategy_index, (strategy, attempts) in enumerate(strategies):
            if strategy_index in hedged_strategy_indexes:
                continue
            strategy_key = _strategy_key(strategy)
            if now < self._strategy_circuit_open_until.get(strategy_key, 0.0):
                self._last_circuit_open_count += 1
                self._record_failure(
                    provider=_PROVIDER_NAME,
                    failure_type="circuit_open",
                    strategy=strategy_key,
                    query=normalized,
                    fallback_attempted=False,
                    fallback_provider=_FALLBACK_PROVIDER_NAME,
                    fallback_success=False,
                )
                continue
            strategy_query = _query_for_strategy(
                strategy,
                default_query=normalized,
                alternate_query=normalized_alternate,
            )
            academic_strategy = _is_academic_strategy(strategy)
            strategy_result_cap = (
                min(limit, _MAX_ACADEMIC_RESULTS_PER_QUERY)
                if academic_strategy and len(strategies) > 1
                else limit
            )
            strategy_added = 0
            for attempt in range(attempts):
                if (
                    max_provider_requests is not None
                    and self._last_request_count >= max(0, max_provider_requests)
                ):
                    break
                if strategy_index > 0 or attempt > 0:
                    self._last_fallback_count += 1
                hedge_fallback_query = (
                    _query_for_strategy(
                        strategies[1][0],
                        default_query=normalized,
                        alternate_query=normalized_alternate,
                    )
                    if len(strategies) > 1
                    else strategy_query
                )
                try:
                    if (
                        strategy_index == 0
                        and attempt == 0
                        and len(strategies) > 1
                        and hedge_fallback_query == strategy_query
                        and (
                            max_provider_requests is None
                            or self._last_request_count + 2 <= max(0, max_provider_requests)
                        )
                    ):
                        # Start the independent fallback after a short hedge
                        # window. A slow default engine therefore cannot hold
                        # the whole search chain behind its 20s timeout.
                        requests_before_hedge = self._last_request_count
                        payload = await self._request_hedged(
                            strategy_query,
                            primary=strategy,
                            fallback=strategies[1][0],
                            fallback_query=hedge_fallback_query,
                        )
                        if self._last_request_count - requests_before_hedge > 1:
                            self._last_fallback_count += 1
                            # The next configured strategy has already been
                            # requested by the hedge. Do not issue the same
                            # upstream request again in the sequential loop.
                            hedged_strategy_indexes.add(1)
                    else:
                        payload = await self._request(strategy_query, strategy=strategy)
                except ToolExecutionError as exc:
                    if not exc.retryable:
                        raise
                    failures.append(
                        {
                            "strategy": strategy.get("engines", "general"),
                            "attempt": attempt + 1,
                            "code": exc.code,
                        }
                    )
                    self._strategy_circuit_open_until[strategy_key] = (
                        time.monotonic() + _STRATEGY_CIRCUIT_COOLDOWN_SECONDS
                    )
                    # A timeout or transient upstream error in one engine group
                    # must not suppress independent fallback groups.
                    if attempt + 1 < attempts:
                        await asyncio.sleep(0.5)
                    continue
                unresponsive = payload.get("unresponsive_engines")
                strategy_unresponsive = isinstance(unresponsive, list) and bool(unresponsive)
                usable_payload = _payload_has_usable_results(strategy_query, payload)
                if strategy_unresponsive and not usable_payload:
                    self._strategy_circuit_open_until[strategy_key] = (
                        time.monotonic() + _STRATEGY_CIRCUIT_COOLDOWN_SECONDS
                    )

                raw_results = payload.get("results")
                if not isinstance(raw_results, list):
                    raise ToolExecutionError(
                        "SEARCH_RESPONSE_INVALID",
                        retryable=False,
                        details=self._failure_details(
                            self._record_failure(
                                provider=_PROVIDER_NAME,
                                failure_type="invalid_response",
                                strategy=strategy_key,
                                query=strategy_query,
                                fallback_attempted=False,
                                fallback_provider=_FALLBACK_PROVIDER_NAME,
                                fallback_success=False,
                            )
                        ),
                    )
                for raw in raw_results:
                    if not isinstance(raw, dict):
                        continue
                    mapped = _map_result(raw, rank=len(results_by_url) + 1)
                    if (
                        mapped is None
                        or not is_stable_read_url(mapped.url)
                        or not _is_relevant_candidate(strategy_query, mapped)
                    ):
                        continue
                    identity_url = normalize_source_url(mapped.url)
                    if (
                        identity_url in results_by_url
                        or identity_url in normalized_excluded_urls
                    ):
                        continue
                    owner = source_owner_key(mapped.url)
                    if owner in normalized_excluded_owners:
                        continue
                    if results_by_owner.get(owner, 0) >= _MAX_RESULTS_PER_OWNER:
                        continue
                    results_by_url[identity_url] = mapped
                    results_by_owner[owner] = results_by_owner.get(owner, 0) + 1
                    strategy_added += 1
                    if (
                        len(results_by_url) >= limit
                        or strategy_added >= strategy_result_cap
                    ):
                        break
                if usable_payload or len(results_by_url) >= limit:
                    break
                if strategy_unresponsive and attempt + 1 < attempts:
                    await asyncio.sleep(0.5)
            if len(results_by_url) >= min(limit, _TARGET_RESULTS_PER_QUERY):
                break
            if (
                max_provider_requests is not None
                and self._last_request_count >= max(0, max_provider_requests)
            ):
                break

        if not results_by_url:
            # SearXNG can report every configured engine as suspended (CAPTCHA,
            # rate limit, or temporary circuit-open) even though the container
            # still has working outbound HTTPS.  Use Bing's public HTML endpoint
            # as a bounded transport fallback so a transient metasearch outage
            # does not turn a valid research run into a retry-pending loop.
            # Try both the planner's native query and its alternate-language
            # form. Bing may silently broaden a long CJK query (returning
            # generic results), while the alternate often preserves the
            # domain nouns and yields usable candidates.
            direct_results: list[SearchResult] = []
            seen_direct: set[str] = set()
            fallback_queries = [normalized, normalized_alternate]
            compact_query = _compact_bing_query(normalized)
            if compact_query:
                fallback_queries.append(compact_query)
            for fallback_query in dict.fromkeys(fallback_queries):
                candidates = await self._direct_bing_fallback(
                    fallback_query,
                    limit=limit,
                    excluded_urls=normalized_excluded_urls,
                    excluded_owners=normalized_excluded_owners,
                )
                for candidate in candidates:
                    if candidate.url not in seen_direct:
                        seen_direct.add(candidate.url)
                        direct_results.append(candidate)
                if len(direct_results) >= limit:
                    break
            if direct_results:
                self._last_productive_response_count += 1
                return direct_results
            raise ToolExecutionError(
                "SEARCH_PROVIDER_DEGRADED",
                retryable=True,
                details=self._failure_details(
                    {
                        "provider": _PROVIDER_NAME,
                        "failure_type": "fallback_failure"
                        if self._last_fallback_attempt_count
                        else "empty_response",
                        "context": {
                            "strategy": "direct_bing_fallback",
                            "query": normalized[:500],
                            "fallback_attempted": bool(self._last_fallback_attempt_count),
                            "fallback_provider": _FALLBACK_PROVIDER_NAME,
                            "fallback_success": False,
                        },
                        "failures": failures,
                    }
                ),
            )
        self._failure_streak = 0
        self._circuit_open_until = 0.0
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(results_by_url.values(), start=1)
        ]

    async def _direct_bing_fallback(
        self,
        query: str,
        *,
        limit: int,
        excluded_urls: Set[str],
        excluded_owners: Set[str],
    ) -> list[SearchResult]:
        """Read a small set of Bing HTML results when SearXNG is unavailable.

        This fallback only returns normal web candidates; all downstream URL
        safety, page-fetch, extraction, and evidence checks remain unchanged.
        """

        # Keep mocked/non-SearXNG adapters deterministic; production uses the
        # Docker service hostname (``searxng``). This also avoids silently
        # turning an arbitrary configured provider into a Bing dependency.
        host = (urlsplit(self._base_url).hostname or "").casefold()
        if host not in {"searxng", "localhost", "127.0.0.1"}:
            return []

        bing_state = self._provider_states[_FALLBACK_PROVIDER_NAME]
        now = time.monotonic()
        if (
            bing_state["state"] == _PROVIDER_COOLDOWN
            and _state_float(bing_state["cooldown_until"]) > now
        ):
            self._last_circuit_open_count += 1
            self._record_failure(
                provider=_FALLBACK_PROVIDER_NAME,
                failure_type="circuit_open",
                strategy="direct_bing_fallback",
                query=query,
                fallback_attempted=True,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
            return []

        self._last_fallback_attempt_count += 1
        self._last_fallback_count += 1

        try:
            response = await self._fallback_client.get(
                "https://www.bing.com/search",
                params={
                    "q": query,
                    "count": min(max(limit, 1), 10),
                    "setlang": "zh-cn",
                    "cc": "cn",
                    "mkt": "zh-CN",
                },
                headers={
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
                    "User-Agent": "Mozilla/5.0",
                },
                timeout=20.0,
            )
        except httpx.TimeoutException:
            self._record_failure(
                provider=_FALLBACK_PROVIDER_NAME,
                failure_type="timeout",
                strategy="direct_bing_fallback",
                query=query,
                fallback_attempted=True,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
            return []
        except httpx.RequestError:
            self._record_failure(
                provider=_FALLBACK_PROVIDER_NAME,
                failure_type="network_error",
                strategy="direct_bing_fallback",
                query=query,
                fallback_attempted=True,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
            return []
        if response.status_code >= 400:
            self._record_failure(
                provider=_FALLBACK_PROVIDER_NAME,
                failure_type="http_error",
                strategy="direct_bing_fallback",
                query=query,
                fallback_attempted=True,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
                extra={"http_status": response.status_code},
            )
            return []
        body = response.text
        if not body.strip():
            self._record_failure(
                provider=_FALLBACK_PROVIDER_NAME,
                failure_type="empty_response",
                strategy="direct_bing_fallback",
                query=query,
                fallback_attempted=True,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
            return []
        rows = re.findall(
            # Bing changes attribute order and adds classes to h2/a nodes
            # between locales. Match the semantic nodes instead of assuming
            # href is the first attribute or h2 has no attributes.
            r'<li[^>]+class="b_algo"[^>]*>.*?<h2[^>]*>\s*<a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?</h2>(?:.*?<p[^>]*>(?P<snippet>.*?)</p>)?',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: list[SearchResult] = []
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            raw_url = _unwrap_bing_redirect(html.unescape(row[0]).strip())
            normalized_url = normalize_source_url(raw_url)
            if (
                not normalized_url
                or normalized_url in seen
                or normalized_url in excluded_urls
                or not is_stable_read_url(normalized_url)
            ):
                continue
            owner = source_owner_key(normalized_url)
            if owner in excluded_owners:
                continue
            title = re.sub(r"<[^>]+>", " ", html.unescape(row[1]))
            snippet = re.sub(r"<[^>]+>", " ", html.unescape(row[2] or ""))
            title = " ".join(title.split())
            snippet = " ".join(snippet.split())
            candidate = SearchResult(
                title=title or raw_url,
                url=normalized_url,
                snippet=snippet,
                rank=rank,
            )
            if not _is_relevant_candidate(query, candidate):
                continue
            seen.add(normalized_url)
            results.append(candidate)
            if len(results) >= limit:
                break
        if results:
            self._last_fallback_success_count += 1
            self._mark_provider_healthy(_FALLBACK_PROVIDER_NAME)
            self._events.append(
                {
                    "event_type": "provider.fallback_succeeded",
                    "provider": _FALLBACK_PROVIDER_NAME,
                    "primary_provider": _PROVIDER_NAME,
                    "primary_provider_state": self.state,
                    "fallback_success": True,
                }
            )
        else:
            self._record_failure(
                provider=_FALLBACK_PROVIDER_NAME,
                failure_type="empty_response",
                strategy="direct_bing_fallback",
                query=query,
                fallback_attempted=True,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
        return results

    async def _request_hedged(
        self,
        query: str,
        *,
        primary: dict[str, str],
        fallback: dict[str, str],
        fallback_query: str | None = None,
    ) -> dict[str, Any]:
        primary_task = asyncio.create_task(self._request(query, strategy=primary))
        fallback_task: asyncio.Task[dict[str, Any]] | None = None
        try:
            done, _ = await asyncio.wait({primary_task}, timeout=_HEDGE_DELAY_SECONDS)
            if done:
                try:
                    primary_payload = await primary_task
                except ToolExecutionError as exc:
                    self._record_hedged_error(exc)
                    primary_payload = None
                if primary_payload is not None and _payload_has_usable_results(
                    query, primary_payload
                ):
                    return primary_payload
                # A fast response, even when unusable, is handed back to the
                # strategy loop. The next strategy is the immediate fallback;
                # hedging is reserved for a primary that remains pending past
                # the hedge window so a fast empty response does not double
                # provider traffic.
                return primary_payload or {"results": []}
            fallback_task = asyncio.create_task(
                self._request(fallback_query or query, strategy=fallback)
            )
            pending: set[asyncio.Task[dict[str, Any]]] = {primary_task, fallback_task}
            first_success: dict[str, Any] | None = None
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    try:
                        payload = await task
                    except ToolExecutionError as exc:
                        self._record_hedged_error(exc)
                        continue
                    if _payload_has_usable_results(query, payload):
                        return payload
                    if first_success is None:
                        first_success = payload
            if first_success is not None:
                return first_success
            raise ToolExecutionError(
                "SEARCH_PROVIDER_DEGRADED",
                retryable=True,
                details=self._failure_details(
                    {
                        "provider": _PROVIDER_NAME,
                        "failure_type": "fallback_failure",
                        "context": {
                            "strategy": _strategy_key(primary),
                            "query": query[:500],
                            "fallback_attempted": True,
                            "fallback_provider": _strategy_key(fallback),
                            "fallback_success": False,
                        },
                    }
                ),
            )
        finally:
            cleanup_tasks: list[asyncio.Task[dict[str, Any]]] = [primary_task]
            if fallback_task is not None:
                cleanup_tasks.append(fallback_task)
            for task in cleanup_tasks:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def _record_hedged_error(self, exc: ToolExecutionError) -> None:
        # ``_request`` records transport failures before raising.  Hedged
        # requests only observe that already-recorded error and must not count
        # it a second time.
        del exc

    async def recovery_probe(self, query: str) -> list[SearchResult]:
        """Probe SearXNG once after cooldown without invoking fallback paths."""

        now = time.monotonic()
        if self.state != _PROVIDER_COOLDOWN or now < self.cooldown_until:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_COOLDOWN",
                retryable=True,
                details={
                    "provider": _PROVIDER_NAME,
                    "failure_type": "circuit_open",
                    "context": {
                        "strategy": "recovery_probe",
                        "query": " ".join(query.split())[:500],
                        "fallback_attempted": False,
                        "fallback_provider": _FALLBACK_PROVIDER_NAME,
                        "fallback_success": False,
                    },
                },
            )
        self._events.append(
            {
                "event_type": "provider.probe_started",
                "provider": _PROVIDER_NAME,
                "failure_type": None,
            }
        )
        try:
            payload = await self._request(query, strategy={})
            unresponsive = payload.get("unresponsive_engines")
            if isinstance(unresponsive, list) and unresponsive:
                raise ToolExecutionError(
                    "SEARCH_PROVIDER_DEGRADED",
                    retryable=True,
                    details=self._failure_details(
                        {
                            "provider": _PROVIDER_NAME,
                            "failure_type": "unresponsive_response",
                            "context": {
                                "strategy": "recovery_probe",
                                "query": " ".join(query.split())[:500],
                                "fallback_attempted": False,
                                "fallback_provider": _FALLBACK_PROVIDER_NAME,
                                "fallback_success": False,
                            },
                        }
                    ),
                )
            self._mark_provider_healthy(_PROVIDER_NAME)
            raw_results = payload.get("results")
            if not isinstance(raw_results, list):
                return []
            return [
                mapped
                for rank, raw in enumerate(raw_results, start=1)
                if isinstance(raw, dict)
                and (mapped := _map_result(raw, rank=rank)) is not None
            ]
        except ToolExecutionError as exc:
            failure_type = str(exc.details.get("failure_type", exc.code))
            self._extend_provider_cooldown(_PROVIDER_NAME, failure_type)
            raise

    async def _request(self, query: str, *, strategy: dict[str, str]) -> dict[str, Any]:
        self._last_request_count += 1
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
            raise ToolExecutionError(
                "SEARCH_TIMEOUT",
                retryable=True,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="timeout",
                        strategy=_strategy_key(strategy),
                        query=query,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                    )
                ),
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(
                "SEARCH_NETWORK_ERROR",
                retryable=True,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="network_error",
                        strategy=_strategy_key(strategy),
                        query=query,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                    )
                ),
            ) from exc
        if response.status_code >= 500:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_UNAVAILABLE",
                retryable=True,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="http_error",
                        strategy=_strategy_key(strategy),
                        query=query,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                        extra={"http_status": response.status_code},
                    )
                ),
            )
        if response.status_code >= 400:
            raise ToolExecutionError(
                "SEARCH_REQUEST_REJECTED",
                retryable=False,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="http_error",
                        strategy=_strategy_key(strategy),
                        query=query,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                        extra={"http_status": response.status_code},
                    )
                ),
            )
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                "SEARCH_RESPONSE_INVALID",
                retryable=False,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="invalid_response",
                        strategy=_strategy_key(strategy),
                        query=query,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                    )
                ),
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ToolExecutionError(
                "SEARCH_RESPONSE_INVALID",
                retryable=False,
                details=self._failure_details(
                    self._record_failure(
                        provider=_PROVIDER_NAME,
                        failure_type="invalid_response",
                        strategy=_strategy_key(strategy),
                        query=query,
                        fallback_attempted=False,
                        fallback_provider=_FALLBACK_PROVIDER_NAME,
                        fallback_success=False,
                    )
                ),
            )
        unresponsive = payload.get("unresponsive_engines")
        raw_results = payload.get("results")
        if isinstance(unresponsive, list) and unresponsive:
            self._last_unresponsive_response_count += 1
            self._record_failure(
                provider=_PROVIDER_NAME,
                failure_type="unresponsive_response",
                strategy=_strategy_key(strategy),
                query=query,
                fallback_attempted=False,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
        else:
            self._last_healthy_response_count += 1
            self._mark_provider_healthy(_PROVIDER_NAME)
        usable = _payload_has_usable_results(query, payload)
        if not raw_results and not (isinstance(unresponsive, list) and unresponsive):
            self._record_failure(
                provider=_PROVIDER_NAME,
                failure_type="empty_response",
                strategy=_strategy_key(strategy),
                query=query,
                fallback_attempted=False,
                fallback_provider=_FALLBACK_PROVIDER_NAME,
                fallback_success=False,
            )
        if usable:
            self._last_productive_response_count += 1
        elif isinstance(unresponsive, list) and unresponsive:
            self._strategy_circuit_open_until[_strategy_key(strategy)] = (
                time.monotonic() + _STRATEGY_CIRCUIT_COOLDOWN_SECONDS
            )
        return payload

    def _record_failure(
        self,
        *,
        provider: str,
        failure_type: str,
        strategy: str,
        query: str,
        fallback_attempted: bool,
        fallback_provider: str,
        fallback_success: bool,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        counter_by_type = {
            "timeout": "_last_timeout_count",
            "network_error": "_last_network_error_count",
            "http_error": "_last_http_error_count",
            "empty_response": "_last_empty_response_count",
            "invalid_response": "_last_invalid_response_count",
        }
        counter_name = counter_by_type.get(failure_type)
        if counter_name is not None:
            setattr(self, counter_name, getattr(self, counter_name) + 1)
        context: dict[str, object] = {
            "strategy": strategy,
            "query": " ".join(query.split())[:500],
            "fallback_attempted": fallback_attempted,
            "fallback_provider": fallback_provider,
            "fallback_success": fallback_success,
        }
        if extra:
            context.update(extra)
        telemetry: dict[str, object] = {
            "provider": provider,
            "failure_type": failure_type,
            "context": context,
        }
        self._last_failure_type = failure_type
        self._failure_events.append(telemetry)
        self._events.append(
            {
                "event_type": "provider.failure",
                "provider": provider,
                "failure_type": failure_type,
                "context": context,
            }
        )
        if failure_type != "circuit_open":
            self._mark_provider_degraded(provider, failure_type)
        return telemetry

    def _mark_provider_degraded(self, provider: str, failure_type: str) -> None:
        state = self._provider_states[provider]
        now = time.monotonic()
        if (
            state["state"] == _PROVIDER_COOLDOWN
            and _state_float(state["cooldown_until"]) > now
        ):
            return
        streak = _state_int(state["failure_streak"]) + 1
        state["failure_streak"] = streak
        previous = str(state["state"])
        state["state"] = _PROVIDER_DEGRADED
        if provider == _PROVIDER_NAME:
            self._failure_streak = streak
        if streak < _CIRCUIT_FAILURE_THRESHOLD:
            if previous != _PROVIDER_DEGRADED:
                self._events.append(
                    {
                        "event_type": "provider.degraded",
                        "provider": provider,
                        "failure_type": failure_type,
                    }
                )
            return
        cooldown_until = max(
            now + _CIRCUIT_COOLDOWN_SECONDS,
            _state_float(state["cooldown_until"]),
        )
        state["state"] = _PROVIDER_COOLDOWN
        state["cooldown_until"] = cooldown_until
        if provider == _PROVIDER_NAME:
            self._circuit_open_until = cooldown_until
        self._events.append(
            {
                "event_type": "provider.cooldown_started",
                "provider": provider,
                "failure_type": failure_type,
                "cooldown_until": cooldown_until,
            }
        )

    def _mark_provider_healthy(self, provider: str, *, emit_recovered: bool = True) -> None:
        state = self._provider_states[provider]
        previous = str(state["state"])
        state["state"] = _PROVIDER_HEALTHY
        state["failure_streak"] = 0
        state["cooldown_until"] = 0.0
        if provider == _PROVIDER_NAME:
            self._failure_streak = 0
            self._circuit_open_until = 0.0
        if emit_recovered and previous != _PROVIDER_HEALTHY:
            self._events.append(
                {
                    "event_type": "provider.recovered",
                    "provider": provider,
                    "failure_type": None,
                }
            )

    def _extend_provider_cooldown(self, provider: str, failure_type: str) -> None:
        state = self._provider_states[provider]
        now = time.monotonic()
        previous_until = _state_float(state["cooldown_until"])
        cooldown_until = max(
            now + _CIRCUIT_COOLDOWN_SECONDS,
            previous_until + _CIRCUIT_COOLDOWN_SECONDS,
        )
        state["state"] = _PROVIDER_COOLDOWN
        state["cooldown_until"] = cooldown_until
        if provider == _PROVIDER_NAME:
            self._circuit_open_until = cooldown_until
        self._events.append(
            {
                "event_type": "provider.cooldown_extended",
                "provider": provider,
                "failure_type": failure_type,
                "cooldown_until": cooldown_until,
            }
        )

    def _telemetry_snapshot(self) -> dict[str, int]:
        return {
            "provider_requests": self._last_request_count,
            "healthy_responses": self._last_healthy_response_count,
            "unresponsive_responses": self._last_unresponsive_response_count,
            "timeout_count": self._last_timeout_count,
            "network_error_count": self._last_network_error_count,
            "http_error_count": self._last_http_error_count,
            "empty_response_count": self._last_empty_response_count,
            "invalid_response_count": self._last_invalid_response_count,
            "fallback_attempts": self._last_fallback_attempt_count,
            "fallback_successes": self._last_fallback_success_count,
            "circuit_open_count": self._last_circuit_open_count,
        }

    def _failure_details(self, primary: dict[str, object]) -> dict[str, object]:
        details = dict(primary)
        details["metrics"] = self._telemetry_snapshot()
        details["failure_events"] = self._failure_events[-20:]
        return details


def _strategy_key(strategy: dict[str, str]) -> str:
    return strategy.get("engines", "default") or "default"


def _is_academic_strategy(strategy: dict[str, str]) -> bool:
    return strategy.get("engines", "").casefold() == _ACADEMIC_ENGINES.casefold()


def _strategies_for_query(
    default_query: str, alternate_query: str
) -> tuple[tuple[dict[str, str], int], ...]:
    """Route commercial and market questions to public web results first."""

    combined = f"{default_query}\n{alternate_query}".casefold()
    if not any(marker in combined for marker in _GENERAL_FIRST_QUERY_MARKERS):
        return _SEARCH_STRATEGIES
    general = [
        entry
        for entry in _SEARCH_STRATEGIES
        if entry[0].get("engines", "").casefold() != _ACADEMIC_ENGINES.casefold()
    ]
    academic = [
        entry
        for entry in _SEARCH_STRATEGIES
        if entry[0].get("engines", "").casefold() == _ACADEMIC_ENGINES.casefold()
    ]
    return tuple([*general, *academic])


def _query_for_strategy(
    strategy: dict[str, str],
    *,
    default_query: str,
    alternate_query: str,
) -> str:
    engines = strategy.get("engines", "").casefold()
    if any(
        engine in engines
        for engine in (
            "arxiv",
            "crossref",
            "google scholar",
            "openalex",
            "openaire",
            "semantic scholar",
        )
    ):
        return (
            alternate_query
            if "filetype:pdf" in alternate_query.casefold()
            else f"{alternate_query} filetype:pdf"
        )
    if strategy.get("language", "").casefold() == "en":
        return alternate_query
    return default_query


def _payload_has_usable_results(query: str, payload: dict[str, Any]) -> bool:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return False
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        mapped = _map_result(raw, rank=1)
        if (
            mapped is not None
            and is_stable_read_url(mapped.url)
            and _is_relevant_candidate(query, mapped)
        ):
            return True
    return False


def _map_result(raw: dict[str, Any], *, rank: int) -> SearchResult | None:
    title = raw.get("title")
    url = raw.get("url")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    clean_url = url.strip()
    if len(clean_url) > 4_000:
        return None
    snippet = raw.get("content")
    published = raw.get("publishedDate")
    return SearchResult(
        title=title.strip()[:1_000],
        url=_normalize_academic_landing_url(clean_url),
        snippet=snippet.strip()[:4_000] if isinstance(snippet, str) else "",
        published_at=(published.strip()[:100] if isinstance(published, str) else None),
        rank=rank,
    )


def _normalize_academic_landing_url(url: str) -> str:
    """Prefer attributable HTML landing pages over unsupported PDF downloads."""

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold()
    path = parsed.path
    if hostname in {"arxiv.org", "www.arxiv.org"} and path.startswith("/pdf/"):
        identifier = path.removeprefix("/pdf/").removesuffix(".pdf").strip("/")
        if identifier:
            return urlunsplit(("https", "arxiv.org", f"/abs/{identifier}", "", ""))
    return url


def _is_relevant_candidate(query: str, result: SearchResult) -> bool:
    """Remove obvious lexical mismatches only for sufficiently specific queries."""

    query_tokens = _search_tokens(query)
    if len(query_tokens) < 6:
        return True
    candidate_tokens = _search_tokens(f"{result.title} {result.snippet}")
    required_anchors = query_tokens & _TOPIC_DOMAIN_ANCHORS
    query_requires_visual_topic = bool(query_tokens & _VISUAL_TOPIC_ANCHORS)
    candidate_text = f"{result.title} {result.snippet}".casefold()
    candidate_has_visual_topic = bool(candidate_tokens & _VISUAL_TOPIC_ANCHORS)
    candidate_is_non_visual_anomaly = any(
        phrase in candidate_text for phrase in _NON_VISUAL_ANOMALY_ANCHORS
    )
    # Derive the guard from the current query. A control-system question is
    # still allowed to return control-system sources; only visually scoped
    # questions reject those false friends.
    if query_requires_visual_topic and (
        not candidate_has_visual_topic or candidate_is_non_visual_anomaly
    ):
        return False
    if (
        query_tokens & _INDUSTRIAL_CONTEXT_ANCHORS
        and query_tokens & _DEFECT_INSPECTION_ANCHORS
        and not (
            candidate_tokens & _INDUSTRIAL_CONTEXT_ANCHORS
            and candidate_tokens & _DEFECT_INSPECTION_ANCHORS
        )
    ):
        return False
    if required_anchors and not (required_anchors & candidate_tokens):
        return False
    required_overlap = min(4, max(2, (len(query_tokens) + 11) // 12))
    return len(query_tokens & candidate_tokens) >= required_overlap


def _search_tokens(value: str) -> set[str]:
    normalized = "".join(char.casefold() if char.isalnum() else " " for char in value)
    words = {word for word in normalized.split() if len(word) > 1}
    cjk = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if all("\u4e00" <= char <= "\u9fff" for char in normalized[index : index + 2])
    }
    return (words | cjk) - _GENERIC_QUERY_TOKENS


def _unwrap_bing_redirect(raw_url: str) -> str:
    """Return the destination URL from Bing's ``/ck/a?...&u=a1...`` link."""

    parsed = urlsplit(raw_url)
    if parsed.hostname not in {"www.bing.com", "bing.com"} or not parsed.path.startswith("/ck/"):
        return raw_url
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return raw_url
    try:
        padded = encoded[2:] + "=" * (-len(encoded[2:]) % 4)
        destination = base64.urlsafe_b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return raw_url
    return destination if destination.startswith(("http://", "https://")) else raw_url


def _compact_bing_query(query: str) -> str:
    """Drop planner boilerplate before querying a general web endpoint."""

    boilerplate = {
        "至少两个独立来源",
        "至少两个独立来源给出",
        "官方",
        "官方文档",
        "标准",
        "政府",
        "原始数据",
        "论文",
        "独立来源",
        "来源",
    }
    tokens = [token for token in query.split() if token not in boilerplate]
    compact = " ".join(tokens[:12]).strip()
    return compact if compact != query else ""
