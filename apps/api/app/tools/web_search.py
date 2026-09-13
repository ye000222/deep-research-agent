"""SearXNG JSON search adapter. Search snippets remain candidate metadata only."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from app.domain.research_tools import SearchResult
from app.domain.source_policy import is_stable_read_url, normalize_source_url, source_owner_key
from app.tools.errors import ToolExecutionError

_DEFAULT_SEARCH_STRATEGIES: tuple[tuple[dict[str, str], int], ...] = (
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
# Prefer an operator-selected engine when the deployment has a known stable
# egress path.  Keep the existing strategy chain as a fallback so local and
# test environments remain unchanged when the variable is absent.
_PREFERRED_ENGINES = os.getenv("SEARXNG_PREFERRED_ENGINES", "").strip()
_SEARCH_STRATEGIES: tuple[tuple[dict[str, str], int], ...] = (
    (({"engines": _PREFERRED_ENGINES}, 2), *_DEFAULT_SEARCH_STRATEGIES)
    if _PREFERRED_ENGINES
    else _DEFAULT_SEARCH_STRATEGIES
)
_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SECONDS = 15.0
_HEDGE_DELAY_SECONDS = 1.8
_MAX_RESULTS_PER_OWNER = 2
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
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._failure_streak = 0
        self._circuit_open_until = 0.0
        self._last_request_count = 0
        self._last_timeout_count = 0
        self._last_fallback_count = 0

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

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        max_provider_requests: int | None = None,
    ) -> list[SearchResult]:
        self._last_request_count = 0
        self._last_timeout_count = 0
        self._last_fallback_count = 0
        normalized = " ".join(query.split())
        if not normalized:
            raise ToolExecutionError("SEARCH_QUERY_INVALID", retryable=False)
        now = time.monotonic()
        if now < self._circuit_open_until:
            raise ToolExecutionError(
                "SEARCH_PROVIDER_DEGRADED",
                retryable=True,
                details={
                    "circuit": "open",
                    "retry_after_seconds": round(self._circuit_open_until - now, 1),
                },
            )

        results_by_url: dict[str, SearchResult] = {}
        results_by_owner: dict[str, int] = {}
        healthy_strategy_seen = False
        failures: list[dict[str, object]] = []
        hedged_strategy_indexes: set[int] = set()
        for strategy_index, (strategy, attempts) in enumerate(_SEARCH_STRATEGIES):
            if strategy_index in hedged_strategy_indexes:
                continue
            for attempt in range(attempts):
                if (
                    max_provider_requests is not None
                    and self._last_request_count >= max(0, max_provider_requests)
                ):
                    break
                if strategy_index > 0 or attempt > 0:
                    self._last_fallback_count += 1
                try:
                    if (
                        strategy_index == 0
                        and attempt == 0
                        and len(_SEARCH_STRATEGIES) > 1
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
                            normalized,
                            primary=strategy,
                            fallback=_SEARCH_STRATEGIES[1][0],
                        )
                        if self._last_request_count - requests_before_hedge > 1:
                            self._last_fallback_count += 1
                            # The next configured strategy has already been
                            # requested by the hedge. Do not issue the same
                            # upstream request again in the sequential loop.
                            hedged_strategy_indexes.add(1)
                    else:
                        payload = await self._request(normalized, strategy=strategy)
                except ToolExecutionError as exc:
                    if exc.code == "SEARCH_TIMEOUT":
                        self._last_timeout_count += 1
                    if not exc.retryable:
                        raise
                    failures.append(
                        {
                            "strategy": strategy.get("engines", "general"),
                            "attempt": attempt + 1,
                            "code": exc.code,
                        }
                    )
                    # A timeout or transient upstream error in one engine group
                    # must not suppress independent fallback groups.
                    if attempt + 1 < attempts:
                        await asyncio.sleep(0.5)
                    continue
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
                    if (
                        mapped is None
                        or not is_stable_read_url(mapped.url)
                        or not _is_relevant_candidate(normalized, mapped)
                    ):
                        continue
                    identity_url = normalize_source_url(mapped.url)
                    if identity_url in results_by_url:
                        continue
                    owner = source_owner_key(mapped.url)
                    if results_by_owner.get(owner, 0) >= _MAX_RESULTS_PER_OWNER:
                        continue
                    results_by_url[identity_url] = mapped
                    results_by_owner[owner] = results_by_owner.get(owner, 0) + 1
                    if len(results_by_url) >= limit:
                        break
                if raw_results or len(results_by_url) >= limit:
                    break
                if strategy_unresponsive and attempt + 1 < attempts:
                    await asyncio.sleep(0.5)
            if len(results_by_url) >= limit:
                break
            if (
                max_provider_requests is not None
                and self._last_request_count >= max(0, max_provider_requests)
            ):
                break

        if not results_by_url and not healthy_strategy_seen:
            self._failure_streak += 1
            if self._failure_streak >= _CIRCUIT_FAILURE_THRESHOLD:
                self._circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_SECONDS
            raise ToolExecutionError(
                "SEARCH_PROVIDER_DEGRADED",
                retryable=True,
                details={"attempts": failures},
            )
        self._failure_streak = 0
        self._circuit_open_until = 0.0
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(results_by_url.values(), start=1)
        ]

    async def _request_hedged(
        self,
        query: str,
        *,
        primary: dict[str, str],
        fallback: dict[str, str],
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
            fallback_task = asyncio.create_task(self._request(query, strategy=fallback))
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
            raise ToolExecutionError("SEARCH_PROVIDER_DEGRADED", retryable=True)
        finally:
            cleanup_tasks: list[asyncio.Task[dict[str, Any]]] = [primary_task]
            if fallback_task is not None:
                cleanup_tasks.append(fallback_task)
            for task in cleanup_tasks:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def _record_hedged_error(self, exc: ToolExecutionError) -> None:
        if exc.code == "SEARCH_TIMEOUT":
            self._last_timeout_count += 1

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
    snippet = raw.get("content")
    published = raw.get("publishedDate")
    return SearchResult(
        title=title.strip(),
        url=url.strip(),
        snippet=snippet.strip() if isinstance(snippet, str) else "",
        published_at=published.strip() if isinstance(published, str) else None,
        rank=rank,
    )


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
