"""Fail-first regression tests for the Provider Recovery state machine.

These tests intentionally describe the Phase 3.3 recovery contract.  The
current adapter exposes typed failure telemetry, but it does not yet expose a
durable provider state, cooldown timestamp, recovery probe, or recovery event
stream.  They are expected to fail until that Tool Layer contract exists.
"""

import httpx
import pytest
from app.tools.errors import ToolExecutionError
from app.tools.web_search import SearXNGSearchProvider


def _state(provider: SearXNGSearchProvider) -> str | None:
    return getattr(provider, "state", None)


def _events(provider: SearXNGSearchProvider) -> list[dict[str, object]]:
    events = getattr(provider, "events", None)
    return events if isinstance(events, list) else []


@pytest.mark.asyncio
async def test_timeout_reaches_cooldown_after_failure_threshold() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("search provider timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")

        for _ in range(3):
            with pytest.raises(ToolExecutionError):
                await provider.search(
                    "industrial inspection",
                    limit=1,
                    max_provider_requests=1,
                )

        assert _state(provider) == "cooldown"
        assert getattr(provider, "last_failure_type", None) == "timeout"
        assert getattr(provider, "cooldown_until", None) is not None
        assert any(
            event.get("event_type") == "provider.cooldown_started"
            and event.get("failure_type") == "timeout"
            for event in _events(provider)
        )


@pytest.mark.asyncio
async def test_bing_fallback_success_does_not_recover_searxng() -> None:
    bing_html = (
        '<li class="b_algo"><h2><a href="https://example.org/inspection">'
        "Industrial inspection source</a></h2><p>industrial inspection</p></li>"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "searxng":
            return httpx.Response(
                200,
                json={"results": [], "unresponsive_engines": [["engine", "down"]]},
            )
        return httpx.Response(200, text=bing_html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "http://searxng")
        results = await provider.search(
            "industrial inspection",
            limit=1,
            max_provider_requests=1,
        )

    assert results
    assert getattr(provider, "last_fallback_success_count", 0) == 1
    assert _state(provider) == "degraded"
    assert any(
        event.get("event_type") == "provider.fallback_succeeded"
        and event.get("provider") == "Bing"
        and event.get("primary_provider_state") == "degraded"
        for event in _events(provider)
    )


@pytest.mark.asyncio
async def test_cooldown_expiry_probe_success_returns_to_healthy() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Industrial inspection source",
                        "url": "https://example.org/inspection",
                        "content": "industrial inspection",
                    }
                ],
                "unresponsive_engines": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        provider.state = "cooldown"  # type: ignore[attr-defined]
        provider.cooldown_until = 0.0  # type: ignore[attr-defined]
        probe = getattr(provider, "recovery_probe", None)
        assert callable(probe)

        await probe("industrial inspection")

    assert _state(provider) == "healthy"
    assert any(
        event.get("event_type") == "provider.recovered"
        and event.get("failure_type") is None
        for event in _events(provider)
    )


@pytest.mark.asyncio
async def test_cooldown_expiry_probe_failure_extends_cooldown() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "https://search.example")
        provider.state = "cooldown"  # type: ignore[attr-defined]
        provider.cooldown_until = 0.0  # type: ignore[attr-defined]
        probe = getattr(provider, "recovery_probe", None)
        assert callable(probe)

        with pytest.raises(ToolExecutionError):
            await probe("industrial inspection")

    assert _state(provider) == "cooldown"
    assert getattr(provider, "cooldown_until", 0.0) > 0.0
    assert any(
        event.get("event_type") == "provider.cooldown_extended"
        and event.get("failure_type") == "network_error"
        for event in _events(provider)
    )
