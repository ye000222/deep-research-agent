"""Phase 12.4 provider registry unit tests."""

from __future__ import annotations

from app.domain.provider_registry import (
    CostClass,
    ProviderCapability,
    SearchProviderRegistry,
)


def test_default_registry_orders_by_priority_without_brave_key() -> None:
    registry = SearchProviderRegistry.default(
        brave_api_key_present=False,
        duckduckgo_enabled=True,
    )
    # Production priority: SearXNG > Bing > DuckDuckGo (no Brave key).
    assert registry.active_providers() == ("SearXNG", "Bing", "DuckDuckGo")
    assert registry.pool_size() == 3


def test_default_registry_includes_brave_when_key_present() -> None:
    registry = SearchProviderRegistry.default(
        brave_api_key_present=True,
        duckduckgo_enabled=True,
    )
    # Full production priority: SearXNG > Bing > Brave > DuckDuckGo.
    assert registry.active_providers() == ("SearXNG", "Bing", "Brave", "DuckDuckGo")
    assert registry.pool_size() == 4


def test_default_registry_can_disable_duckduckgo() -> None:
    registry = SearchProviderRegistry.default(
        brave_api_key_present=False,
        duckduckgo_enabled=False,
    )
    assert registry.active_providers() == ("SearXNG", "Bing")


def test_custom_capabilities_sort_by_priority_not_declaration() -> None:
    registry = SearchProviderRegistry(
        [
            ProviderCapability(provider_name="Low", priority=90),
            ProviderCapability(provider_name="High", priority=10),
            ProviderCapability(provider_name="Mid", priority=50),
        ]
    )
    assert [cap.provider_name for cap in registry.capabilities] == [
        "High",
        "Mid",
        "Low",
    ]


def test_disabled_capability_is_excluded_from_active_providers() -> None:
    registry = SearchProviderRegistry(
        [
            ProviderCapability(provider_name="A", priority=1, enabled=True),
            ProviderCapability(provider_name="B", priority=2, enabled=False),
        ]
    )
    assert registry.active_providers() == ("A",)
    assert registry.is_registered("A") is True
    assert registry.is_registered("B") is False
    assert "B" in registry  # present in the table but not active


def test_capability_as_dict_is_serializable() -> None:
    cap = ProviderCapability(
        provider_name="Brave",
        priority=1,
        cost_class=CostClass.LOW,
        rate_limit_per_minute=2000,
    )
    payload = cap.as_dict()
    assert payload["provider_name"] == "Brave"
    assert payload["cost_class"] == "low"
    assert payload["rate_limit_per_minute"] == 2000


def test_registry_requires_at_least_one_capability() -> None:
    try:
        SearchProviderRegistry([])
    except ValueError:
        return
    raise AssertionError("empty registry must be rejected")


def test_registry_rejects_duplicate_names() -> None:
    try:
        SearchProviderRegistry(
            [
                ProviderCapability(provider_name="Dup", priority=1),
                ProviderCapability(provider_name="Dup", priority=2),
            ]
        )
    except ValueError:
        return
    raise AssertionError("duplicate provider names must be rejected")
