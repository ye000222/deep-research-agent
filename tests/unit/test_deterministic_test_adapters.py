from __future__ import annotations

from app.core.config import Settings
from app.domain.identifiers import uuid7
from app.domain.provider_registry import SearchProviderRegistry
from app.domain.providers import CanonicalModelRequest

from tests.support.deterministic_research import (
    FOUNDING_QUOTE,
    DeterministicFixtureLLMGateway,
    DeterministicFixtureSearchProvider,
    deterministic_result_order,
)


def test_fixture_search_order_is_stable_for_same_input() -> None:
    import asyncio

    provider = DeterministicFixtureSearchProvider()
    first = asyncio.run(provider.search("generic fixture query"))
    second = asyncio.run(provider.search("generic fixture query"))
    assert [item.url for item in first] == list(deterministic_result_order())
    assert [item.url for item in second] == [item.url for item in first]


def test_fixture_extractor_is_deterministic_for_same_input() -> None:
    import asyncio

    from app.domain.providers import ContentPart

    request = CanonicalModelRequest(
        task_kind="evidence_extraction",
        role="extractor",
        model="fixture-model",
        instructions="test only",
        content_parts=(ContentPart(kind="text", value=FOUNDING_QUOTE),),
        response_contract={"type": "object"},
        max_output_tokens=100,
        context_manifest_id=uuid7(),
    )
    gateway = DeterministicFixtureLLMGateway()
    first = asyncio.run(gateway.generate_structured(request=request))
    second = asyncio.run(gateway.generate_structured(request=request))
    assert first.parsed_object == second.parsed_object
    assert first.parsed_object is not None
    assert first.parsed_object["items"][0]["claim"] == FOUNDING_QUOTE


def test_production_defaults_cannot_select_fixture_provider() -> None:
    Settings(app_env="production")
    registry = SearchProviderRegistry.default()
    names = set(registry.active_providers())
    assert "DeterministicFixture" not in names
    assert "deterministic_fixture_provider_enabled" not in Settings.model_fields
    assert "tests.support.deterministic_research" not in {
        module for module in __import__("sys").modules if module.startswith("app.")
    }
