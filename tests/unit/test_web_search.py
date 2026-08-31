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
