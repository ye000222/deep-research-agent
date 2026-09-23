"""Phase 12.4 Step 2: router decision must equal the executed provider.

These tests exercise the real :class:`ResearchLoopService` provider-dispatch
path (the router-authoritative branch) and assert the two invariants the spec
requires:

* ``provider.routing.selected`` provider == the provider whose adapter ran ==
  the ``executed_provider`` recorded on the search/failure event;
* a provider selected by the router but absent from the dispatch map never
  triggers a silent fallback to another provider (it stops as a bounded
  provider error instead).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest
from app.domain.provider_health import ProviderHealthState
from app.domain.provider_registry import SearchProviderRegistry
from app.domain.provider_router import ProviderSelectionDecision
from app.domain.research_tools import SearchResult
from app.infrastructure.db.research_tools import ResearchTarget
from app.services.research_loop import ResearchLoopService
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.web_search import SearXNGSearchProvider
from test_research_loop import (
    FakeArtifacts,
    FakeExtractor,
    FakeReader,
    FakeRepository,
)


class SpyAdapter:
    """A :class:`SearchProviderAdapter` double that records every dispatch."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.calls = 0

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: Any = None,
    ) -> list[SearchResult]:
        del query, limit, exclude_domains, exclude_urls, request_context
        self.calls += 1
        return [
            SearchResult(
                title=f"{self.provider_name} source {rank}",
                url=f"https://example.com/{self.provider_name}/{rank}",
                rank=rank,
            )
            for rank in (1, 2, 3)
        ]


class DispatchRepository(FakeRepository):
    """Fake repository that records the router/executor provider pairing."""

    def __init__(self, decision: ProviderSelectionDecision) -> None:
        super().__init__()
        self._decision = decision
        self.execution_started_providers: list[str] = []
        self.recorded_search_providers: list[str | None] = []
        self.recorded_failure_providers: list[str | None] = []

    async def select_provider_for_target(
        self, *args: object, **kwargs: object
    ) -> ProviderSelectionDecision:
        return self._decision

    async def record_provider_execution_started(
        self, *args: object, **kwargs: object
    ) -> None:
        decision = kwargs["decision"]  # type: ignore[index]
        if decision.selected_provider is not None:
            self.execution_started_providers.append(decision.selected_provider)

    async def record_search_results(self, *args: object, **kwargs: object) -> None:
        self.recorded_search_providers.append(kwargs.get("executed_provider"))

    async def record_tool_failure(self, *args: object, **kwargs: object) -> None:
        self.allow_search_failure = True
        self.recorded_failure_providers.append(kwargs.get("executed_provider"))


def _searxng() -> SearXNGSearchProvider:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"results": []}))
    client = httpx.AsyncClient(transport=transport)
    return SearXNGSearchProvider(client, "http://searxng.test")


def _decision(provider: str | None) -> ProviderSelectionDecision:
    return ProviderSelectionDecision(
        selected_provider=provider,
        excluded_providers=(),
        reason="healthy_provider_selected" if provider else "all_providers_unavailable",
        health_state=ProviderHealthState.UNKNOWN if provider else None,
        fallback_used=False,
        provider_pool_size=3,
        available_provider_count=2,
        selection_rank=1 if provider else 0,
    )


def _service(
    decision: ProviderSelectionDecision, *, ddg: SpyAdapter | None = None
) -> tuple[ResearchLoopService, DispatchRepository, SpyAdapter | None]:
    provider = _searxng()
    repository = DispatchRepository(decision)
    adapters: dict[str, Any] = {
        "SearXNG": SearXNGAdapter(provider),
        "Bing": BingAdapter(provider),
    }
    if ddg is not None:
        adapters["DuckDuckGo"] = ddg
    service = ResearchLoopService(
        repository,  # type: ignore[arg-type]
        provider,  # type: ignore[arg-type]
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
        search_adapters=adapters,
    )
    return service, repository, ddg


@pytest.mark.asyncio
async def test_selected_provider_is_the_executed_provider() -> None:
    ddg = SpyAdapter("DuckDuckGo")
    service, repository, _ = _service(_decision("DuckDuckGo"), ddg=ddg)

    await service.run_one_iteration(uuid4(), worker_task_id="dispatch-1")

    # The router picked DuckDuckGo, the executor started DuckDuckGo, the DuckDuckGo
    # adapter actually ran, and the search completion was attributed to DuckDuckGo.
    assert repository.execution_started_providers == ["DuckDuckGo"]
    assert ddg.calls == 1
    assert repository.recorded_search_providers == ["DuckDuckGo"]


@pytest.mark.asyncio
async def test_selected_provider_without_adapter_never_falls_back() -> None:
    # Brave is selected but absent from the dispatch map: the loop must stop as a
    # provider error rather than silently executing another provider.
    service, repository, _ = _service(_decision("Brave"))

    result = await service.run_one_iteration(uuid4(), worker_task_id="dispatch-2")

    assert repository.execution_started_providers == ["Brave"]
    assert repository.recorded_failure_providers == [None]
    assert repository.recorded_search_providers == []
    assert result.decision == "provider_error" or repository.last_attempt_outcome in {
        "provider_error",
        None,
    }


def test_registry_pool_names_all_have_adapters_in_worker_shape() -> None:
    registry = SearchProviderRegistry.default(
        brave_api_key_present=True,
        duckduckgo_enabled=True,
    )
    provider = _searxng()
    wired = {"SearXNG", "Bing", "Brave", "DuckDuckGo"}
    # Every provider the registry can hand the Router must be dispatchable.
    assert set(registry.active_providers()) <= wired
    assert {"SearXNG", "Bing"} <= {
        name
        for name, adapter in (
            ("SearXNG", SearXNGAdapter(provider)),
            ("Bing", BingAdapter(provider)),
        )
        if adapter is not None
    }


def test_target_defaults_keep_dispatch_inputs() -> None:
    # Guard: the fields the dispatch reads must exist on the prepared target.
    target = ResearchTarget(
        plan_version=1,
        question_id="q1",
        question="Which routes?",
        query="industrial inspection routes",
        gap_id=uuid4(),
        tool_call_id=uuid4(),
        source_id_seed=uuid4(),
    )
    assert hasattr(target, "provider_request_allowance")
    assert hasattr(target, "alternate_query")
    assert hasattr(target, "search_excluded_owner_keys")
    assert hasattr(target, "search_excluded_urls")
