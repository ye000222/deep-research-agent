import asyncio

import httpx
import pytest
import respx
from app.domain.research_tools import SearchResult
from app.tools.errors import ToolExecutionError
from app.tools.web_search import (
    _ACADEMIC_ENGINES,
    SearXNGSearchProvider,
    _is_relevant_candidate,
    _map_result,
    _strategies_for_query,
)


def test_authority_boilerplate_cannot_make_an_unrelated_result_relevant() -> None:
    query = (
        "工业缺陷检测 深度学习 技术路线 "
        "官方 原始来源 政府 高校 论文 行业协会 primary source government paper"
    )
    unrelated = SearchResult(
        title="A federated architecture for sector-led AI governance",
        url="https://arxiv.org/abs/2603.26865",
        snippet="Government policy and public software institutions.",
        rank=1,
    )

    assert _is_relevant_candidate(query, unrelated) is False


def test_provider_metadata_is_bounded_before_search_result_validation() -> None:
    mapped = _map_result(
        {
            "title": "Industrial inspection " * 200,
            "url": "https://example.org/paper",
            "content": "<jats:p>industrial defect inspection</jats:p>" * 500,
            "publishedDate": "2026-09-15" * 20,
        },
        rank=1,
    )

    assert mapped is not None
    assert len(mapped.title) == 1_000
    assert len(mapped.snippet) == 4_000
    assert mapped.published_at is not None
    assert len(mapped.published_at) == 100


def test_provider_result_with_oversized_url_is_discarded() -> None:
    assert (
        _map_result(
            {
                "title": "Industrial inspection",
                "url": "https://example.org/" + "a" * 4_001,
            },
            rank=1,
        )
        is None
    )


def test_vendor_and_market_queries_use_general_search_before_academic() -> None:
    vendor = _strategies_for_query(
        "工业视觉缺陷检测 厂商 产品 客户案例",
        "industrial visual inspection vendors products customer cases",
    )
    technical = _strategies_for_query(
        "工业视觉缺陷检测 深度学习方法",
        "industrial visual defect inspection deep learning methods",
    )

    assert vendor[0][0].get("engines", "").casefold() != _ACADEMIC_ENGINES.casefold()
    assert technical[0][0]["engines"] == _ACADEMIC_ENGINES


def test_industrial_defect_query_rejects_generic_vision_method_overlap() -> None:
    query = "工业视觉缺陷检测技术路线 深度学习"
    unrelated = SearchResult(
        title="视觉光流计算与行人跟踪算法综述",
        url="https://example.com/optical-flow",
        snippet="视觉、检测、算法和深度学习方法的综述。",
        rank=1,
    )
    direct = SearchResult(
        title="工业视觉表面缺陷检测技术路线",
        url="https://example.com/industrial-defect",
        snippet="面向产线质量检测的工业缺陷识别与异常检测方法。",
        rank=2,
    )

    assert _is_relevant_candidate(query, unrelated) is False
    assert _is_relevant_candidate(query, direct) is True


def test_visual_guard_is_scoped_to_visual_queries() -> None:
    control_query = "industrial control system network anomaly detection"
    control_result = SearchResult(
        title="Anomaly Detection for Industrial Control Systems",
        url="https://example.com/control",
        snippet="Network intrusion and process anomaly detection in industrial control systems.",
        rank=1,
    )

    assert _is_relevant_candidate(control_query, control_result) is True


def test_arxiv_pdf_result_is_normalized_to_readable_abstract_page() -> None:
    result = _map_result(
        {
            "title": "Industrial defect detection paper",
            "url": "https://arxiv.org/pdf/2406.00501.pdf",
        },
        rank=1,
    )

    assert result is not None
    assert result.url == "https://arxiv.org/abs/2406.00501"


@pytest.mark.asyncio
@respx.mock
async def test_searxng_maps_ranked_candidate_metadata() -> None:
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Official product page",
                        "url": "https://example.com/product",
                        "content": "Candidate snippet only.",
                        "publishedDate": "2026-08-01",
                    },
                    {"title": "Missing URL"},
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search(
            " industrial   inspection ", limit=5
        )

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].snippet == "Candidate snippet only."


@pytest.mark.asyncio
@respx.mock
async def test_searxng_deduplicates_transport_and_tracking_url_variants() -> None:
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Industrial inspection source",
                        "url": "http://example.com/source?utm_source=search#section",
                        "content": "industrial inspection source evidence",
                    },
                    {
                        "title": "Same source over HTTPS",
                        "url": "https://example.com/source",
                        "content": "industrial inspection source evidence",
                    },
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search(
            "industrial inspection source evidence", limit=5
        )

    assert len(results) == 1


@pytest.mark.asyncio
@respx.mock
async def test_search_filters_low_value_domains_and_uses_fallback_sources() -> None:
    route = respx.get("http://searxng.test/search").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "工业视觉检测趋势预测报告",
                            "url": "https://www.docin.com/p-123.html",
                            "content": "工业视觉检测趋势预测",
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "工业视觉检测技术路线图",
                            "url": "https://www.nist.gov/industrial-vision-roadmap",
                            "content": "工业视觉检测未来技术趋势与路线图",
                        }
                    ]
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search(
            "工业视觉检测未来技术趋势路线图", limit=1
        )

    assert route.call_count == 2
    assert [result.url for result in results] == [
        "https://www.nist.gov/industrial-vision-roadmap"
    ]


@pytest.mark.asyncio
@respx.mock
async def test_search_caps_duplicate_owners_before_filling_from_fallback() -> None:
    route = respx.get("http://searxng.test/search").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": f"Industrial inspection result {index}",
                            "url": f"https://example.com/{index}",
                            "content": "industrial inspection benchmark result",
                        }
                        for index in range(4)
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Independent industrial inspection benchmark",
                            "url": "https://nist.gov/benchmark",
                            "content": "industrial inspection benchmark result",
                        }
                    ]
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search(
            "industrial inspection benchmark result dataset metrics", limit=3
        )

    assert route.call_count == 2
    assert [result.url for result in results] == [
        "https://example.com/0",
        "https://example.com/1",
        "https://nist.gov/benchmark",
    ]


@pytest.mark.asyncio
@respx.mock
async def test_searxng_normalizes_invalid_payload_to_safe_error() -> None:
    respx.get("http://searxng.test/search").mock(return_value=httpx.Response(200, json=[]))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolExecutionError) as caught:
            await SearXNGSearchProvider(client, "http://searxng.test").search("topic")

    assert caught.value.code == "SEARCH_RESPONSE_INVALID"
    assert caught.value.retryable is False


@pytest.mark.asyncio
@respx.mock
async def test_searxng_falls_back_to_general_when_academic_engines_are_degraded() -> None:
    route = respx.get("http://searxng.test/search").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "results": [],
                    "unresponsive_engines": [["brave", "too many requests"]],
                },
            ),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Sogou result",
                            "url": "https://example.cn/inspection",
                            "content": "Public result page.",
                        }
                    ],
                    "unresponsive_engines": [],
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search(
            "topic", limit=1
        )

    assert route.call_count == 2
    assert results[0].url == "https://example.cn/inspection"
    assert route.calls[0].request.url.params["engines"] == _ACADEMIC_ENGINES
    assert route.calls[1].request.url.params.get("engines") is None


@pytest.mark.asyncio
@respx.mock
async def test_searxng_reports_provider_degradation_instead_of_false_empty_success() -> None:
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [],
                "unresponsive_engines": [["engine", "timeout"]],
            },
        )
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolExecutionError) as caught:
            await SearXNGSearchProvider(client, "http://searxng.test").search("topic")

    assert caught.value.code == "SEARCH_PROVIDER_DEGRADED"
    assert caught.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_searxng_rotates_unhealthy_strategies_without_same_engine_retry() -> None:
    route = respx.get("http://searxng.test/search").mock(
        side_effect=[
            httpx.Response(
                200,
                json={"results": [], "unresponsive_engines": [["general", "timeout"]]},
            ),
            httpx.Response(
                200,
                json={"results": [], "unresponsive_engines": [["sogou", "timeout"]]},
            ),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Recovered result",
                            "url": "https://arxiv.org/abs/2109.11304",
                        }
                    ],
                    "unresponsive_engines": [],
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search("topic")

    assert route.call_count == 3
    assert results[0].title == "Recovered result"


@pytest.mark.asyncio
@respx.mock
async def test_searxng_continues_fallback_after_network_timeout() -> None:
    route = respx.get("http://searxng.test/search").mock(
        side_effect=[
            httpx.ConnectTimeout("general timeout"),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Fallback result",
                            "url": "https://example.org/fallback",
                        }
                    ],
                    "unresponsive_engines": [],
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        results = await SearXNGSearchProvider(client, "http://searxng.test").search(
            "topic", limit=1
        )

    assert route.call_count == 2
    assert results[0].url == "https://example.org/fallback"


@pytest.mark.asyncio
async def test_fast_hedged_primary_timeout_is_counted() -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        engine = request.url.params.get("engines") or "default"
        requests.append(engine)
        if engine == _ACADEMIC_ENGINES:
            raise httpx.ReadTimeout("primary timed out", request=request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Industrial machine vision inspection",
                        "url": "https://fallback.example/inspection",
                        "content": "industrial machine vision inspection",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        results = await provider.search(
            "industrial machine vision inspection",
            limit=1,
            max_provider_requests=2,
        )

    assert results
    assert requests == [_ACADEMIC_ENGINES, "default"]
    assert provider.last_request_count == 2
    assert provider.last_timeout_count == 1
    assert provider.last_fallback_count == 1


@pytest.mark.asyncio
async def test_academic_strategy_uses_alternate_language_query() -> None:
    requests: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            (request.url.params.get("engines") or "default", request.url.params["q"])
        )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Industrial visual defect inspection benchmark",
                        "url": "https://arxiv.org/abs/1234.5678",
                        "content": "industrial visual defect inspection methods benchmark",
                    }
                ],
                "unresponsive_engines": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        results = await provider.search(
            "工业视觉缺陷检测 方法",
            alternate_query="industrial visual defect inspection methods",
            max_provider_requests=1,
        )

    assert results
    assert requests == [
        (
            _ACADEMIC_ENGINES,
            "industrial visual defect inspection methods filetype:pdf",
        )
    ]
    assert provider.last_healthy_response_count == 1
    assert provider.last_productive_response_count == 1


@pytest.mark.asyncio
async def test_search_filters_excluded_owners_from_payload_not_only_query_text() -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.params["q"])
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Blocked industrial inspection source",
                        "url": "https://papers.blocked.example/article/1",
                        "content": "industrial machine vision defect inspection methods",
                    },
                    {
                        "title": "Readable industrial inspection source",
                        "url": "https://public.example.org/inspection",
                        "content": "industrial machine vision defect inspection methods",
                    },
                ],
                "unresponsive_engines": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        results = await provider.search(
            "industrial machine vision defect inspection methods",
            limit=1,
            max_provider_requests=1,
            exclude_domains=("blocked.example",),
        )

    assert [result.url for result in results] == [
        "https://public.example.org/inspection"
    ]
    assert "-site:blocked.example" in requests[0]


@pytest.mark.asyncio
async def test_search_filters_failed_redirect_entry_url_from_fresh_results() -> None:
    failed_doi = "https://doi.org/10.1000/failed"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Failed redirected industrial inspection paper",
                        "url": failed_doi,
                        "content": "industrial machine vision defect inspection methods",
                    },
                    {
                        "title": "Alternative industrial inspection paper",
                        "url": "https://open.example.org/paper",
                        "content": "industrial machine vision defect inspection methods",
                    },
                ],
                "unresponsive_engines": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        results = await provider.search(
            "industrial machine vision defect inspection methods",
            limit=1,
            max_provider_requests=1,
            exclude_urls=(failed_doi,),
        )

    assert [result.url for result in results] == ["https://open.example.org/paper"]


@pytest.mark.asyncio
async def test_technical_search_reserves_room_for_non_academic_sources() -> None:
    requested_engines: list[str] = []
    requested_queries: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        engine = request.url.params.get("engines") or "default"
        requested_engines.append(engine)
        requested_queries.append(request.url.params["q"])
        prefix = "academic" if engine == _ACADEMIC_ENGINES else "public"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": f"Industrial defect inspection {prefix} source {index}",
                        "url": f"https://{prefix}{index}.source{index}.org/inspection",
                        "content": "industrial machine vision defect inspection methods",
                    }
                    for index in range(8)
                ],
                "unresponsive_engines": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        results = await provider.search(
            "工业视觉缺陷检测 方法",
            alternate_query="industrial machine vision defect inspection methods",
            limit=8,
            max_provider_requests=2,
        )

    assert requested_engines == [_ACADEMIC_ENGINES, "default"]
    assert requested_queries == [
        "industrial machine vision defect inspection methods filetype:pdf",
        "industrial machine vision defect inspection methods",
    ]
    assert sum("academic" in result.url for result in results) == 4
    assert sum("public" in result.url for result in results) == 4


@pytest.mark.asyncio
async def test_unresponsive_strategies_are_circuit_broken_for_later_questions() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"results": [], "unresponsive_engines": [["engine", "captcha"]]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        with pytest.raises(ToolExecutionError):
            await provider.search("industrial inspection")
        assert calls == 3
        assert provider.last_unresponsive_response_count == 3

        with pytest.raises(ToolExecutionError):
            await provider.search("another industrial inspection question")
        assert calls == 3
        assert provider.last_request_count == 0


@pytest.mark.asyncio
async def test_hedged_fallback_is_not_requested_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.web_search._HEDGE_DELAY_SECONDS", 0.001)
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        engine = request.url.params.get("engines") or "default"
        requests.append(engine)
        if engine == "default":
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={"results": []})
        if engine == "sogou":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Industrial machine vision inspection source",
                            "url": "https://sogou-result.example/inspection",
                            "content": "industrial machine vision inspection",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Industrial machine vision inspection paper one",
                        "url": "https://papers.example/one",
                        "content": "industrial machine vision inspection",
                    },
                    {
                        "title": "Industrial machine vision inspection paper two",
                        "url": "https://papers-two.example/two",
                        "content": "industrial machine vision inspection",
                    },
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        results = await provider.search(
            "industrial machine vision inspection",
            limit=3,
            max_provider_requests=4,
        )

    assert len(results) == 3
    assert requests.count("sogou") == 1
    assert provider.last_request_count == 3
    assert provider.last_fallback_count == 2
