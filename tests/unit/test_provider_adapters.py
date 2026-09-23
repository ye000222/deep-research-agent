"""Phase 12.4 B/C: Brave and DuckDuckGo adapter unit tests (mock transport)."""

from __future__ import annotations

import httpx
import pytest
from app.domain.provider_adapter import SearchProviderAdapter
from app.tools.brave_search import BraveSearchAdapter
from app.tools.duckduckgo_search import DuckDuckGoAdapter
from app.tools.errors import ToolExecutionError


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_brave_normalizes_results_and_sends_token() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("X-Subscription-Token", "")
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Acme Vision Defect",
                            "url": "https://acme-vision.com/products/defect",
                            "description": "Industrial  defect   detection",
                            "page_age": "2024-05-01T00:00:00Z",
                        },
                        {
                            "title": "Nova Cameras",
                            "url": "https://nova-cameras.com/inspection",
                            "description": "Machine vision inspection",
                        },
                    ]
                }
            },
        )

    adapter = BraveSearchAdapter(_client(handler), api_key="secret-token")
    results = await adapter.search("industrial vision defect", limit=5)

    assert seen["auth"] == "secret-token"
    assert "/res/v1/web/search" in seen["url"]
    assert len(results) == 2
    assert results[0].title == "Acme Vision Defect"
    assert results[0].url == "https://acme-vision.com/products/defect"
    assert results[0].snippet == "Industrial defect detection"
    assert results[0].published_at == "2024-05-01T00:00:00Z"
    assert [r.rank for r in results] == [1, 2]
    assert isinstance(adapter, SearchProviderAdapter)


async def test_brave_honors_limit_and_excludes_owner() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "A", "url": "https://skip-me.com/a"},
                        {"title": "B", "url": "https="},
                        {"title": "C", "url": "https://keep-me.com/c"},
                        {"title": "D", "url": "https://keep-me.com/d"},
                    ]
                }
            },
        )

    adapter = BraveSearchAdapter(_client(handler), api_key="k")
    results = await adapter.search(
        "q", limit=5, exclude_domains=("skip-me.com",)
    )

    urls = [r.url for r in results]
    assert "https://skip-me.com/a" not in urls
    assert urls == ["https://keep-me.com/c", "https://keep-me.com/d"]


@pytest.mark.parametrize(
    "status",
    [429, 500, 503],
)
async def test_brave_raises_on_error_status(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    adapter = BraveSearchAdapter(_client(handler), api_key="k")
    with pytest.raises(ToolExecutionError):
        await adapter.search("q")


async def test_brave_requires_api_key() -> None:
    with pytest.raises(ValueError):
        BraveSearchAdapter(_client(lambda r: httpx.Response(200)), api_key="")


async def test_duckduckgo_parses_and_unwraps_redirect() -> None:
    html_body = """
    <div class="result">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnovus-vision.com%2Fdefect&rut=abc">
           Novus Defect</a>
      </h2>
      <a class="result__snippet" href="//d/l/">Industrial <b>defect</b> detection</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://peak-inspect.com/quality">Peak Quality</a>
      <a class="result__snippet">Automated quality checks</a>
    </div>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_body)

    adapter = DuckDuckGoAdapter(_client(handler))
    results = await adapter.search("industrial vision defect", limit=5)

    assert len(results) == 2
    assert results[0].url == "https://novus-vision.com/defect"
    assert results[0].title == "Novus Defect"
    assert results[0].snippet == "Industrial defect detection"
    assert results[1].url == "https://peak-inspect.com/quality"
    assert [r.rank for r in results] == [1, 2]
    assert isinstance(adapter, SearchProviderAdapter)


async def test_duckduckgo_respects_limit_and_exclusions() -> None:
    html_body = (
        '<a class="result__a" href="https://one.com/a">One</a>'
        '<a class="result__a" href="https://two.com/b">Two</a>'
        '<a class="result__a" href="https://three.com/c">Three</a>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_body)

    adapter = DuckDuckGoAdapter(_client(handler))
    results = await adapter.search(
        "q", limit=2, exclude_domains=("one.com",)
    )

    urls = [r.url for r in results]
    assert "https://one.com/a" not in urls
    assert len(urls) == 2


async def test_duckduckgo_raises_on_empty_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="   ")

    adapter = DuckDuckGoAdapter(_client(handler))
    with pytest.raises(ToolExecutionError):
        await adapter.search("q")


async def test_duckduckgo_raises_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("ddg timeout", request=request)

    adapter = DuckDuckGoAdapter(_client(handler))
    with pytest.raises(ToolExecutionError):
        await adapter.search("q")
