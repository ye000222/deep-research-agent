import httpx
import pytest
import respx
from app.tools.errors import ToolExecutionError
from app.tools.web_search import SearXNGSearchProvider


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
async def test_searxng_normalizes_invalid_payload_to_safe_error() -> None:
    respx.get("http://searxng.test/search").mock(return_value=httpx.Response(200, json=[]))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolExecutionError) as caught:
            await SearXNGSearchProvider(client, "http://searxng.test").search("topic")

    assert caught.value.code == "SEARCH_RESPONSE_INVALID"
    assert caught.value.retryable is False


@pytest.mark.asyncio
@respx.mock
async def test_searxng_falls_back_to_sogou_when_general_engines_are_degraded() -> None:
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
    assert route.calls[1].request.url.params["engines"] == "sogou"


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
async def test_searxng_retries_transient_fallback_degradation_once() -> None:
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

    assert route.call_count == 4
    assert results[0].title == "Recovered result"
