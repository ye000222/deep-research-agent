"""Search Provider Registry (Phase 12.4).

Centralizes the metadata for every search provider that the Router may
consider: identity, declared priority, capability flags, cost class, and
whether the provider is currently *registered* (has credentials / enabled).
The Router consumes the registry to obtain an ordered candidate list instead
of a hard-coded tuple, so adding a new provider only requires registering a
new capability — no Router or research_tools code change.

Provider priority order (highest → lowest):
    Brave > Bing > DuckDuckGo > SearXNG

The registry never performs network calls, never mutates provider health, and
never triggers a provider switch. It is a pure lookup / configuration table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class CostClass(StrEnum):
    """Coarse cost classification for provider budget planning."""

    FREE = "free"
    LOW = "low"
    STANDARD = "standard"


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    """Static metadata describing one search provider."""

    provider_name: str
    supports_web_search: bool = True
    supports_language: tuple[str, ...] = ("en", "zh")
    rate_limit_per_minute: int | None = None
    cost_class: CostClass = CostClass.FREE
    priority: int = 100  # lower number = more preferred
    enabled: bool = True  # runtime: False when credentials/config missing

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "supports_web_search": self.supports_web_search,
            "supports_language": list(self.supports_language),
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "cost_class": self.cost_class.value,
            "priority": self.priority,
            "enabled": self.enabled,
        }


class SearchProviderRegistry:
    """Ordered set of registered provider capabilities.

    The registry stores providers sorted by (priority asc, declaration order)
    so that Router candidate enumeration always follows the configured
    preference. Only ``enabled`` providers appear in :meth:`active_providers`,
    which is the tuple the Router consumes.
    """

    def __init__(self, capabilities: Sequence[ProviderCapability]) -> None:
        if not capabilities:
            raise ValueError("provider registry requires at least one capability")
        # Sort by priority then declaration index for stable ordering.
        indexed = list(enumerate(capabilities))
        indexed.sort(key=lambda pair: (pair[1].priority, pair[0]))
        self._capabilities: tuple[ProviderCapability, ...] = tuple(
            cap for _, cap in indexed
        )
        self._by_name: dict[str, ProviderCapability] = {
            cap.provider_name: cap for cap in self._capabilities
        }
        if len(self._by_name) != len(self._capabilities):
            raise ValueError("provider registry capability names must be unique")

    # ------------------------------------------------------------------ query
    def __contains__(self, provider_name: object) -> bool:
        return isinstance(provider_name, str) and provider_name in self._by_name

    def __len__(self) -> int:
        return len(self._capabilities)

    @property
    def capabilities(self) -> tuple[ProviderCapability, ...]:
        return self._capabilities

    def capability(self, provider_name: str) -> ProviderCapability | None:
        return self._by_name.get(provider_name)

    def is_registered(self, provider_name: str) -> bool:
        cap = self._by_name.get(provider_name)
        return cap is not None and cap.enabled

    def active_providers(self) -> tuple[str, ...]:
        """Ordered provider names eligible for routing (enabled only)."""

        return tuple(cap.provider_name for cap in self._capabilities if cap.enabled)

    def pool_size(self) -> int:
        """Total count of enabled providers in the pool."""

        return sum(1 for cap in self._capabilities if cap.enabled)

    # ----------------------------------------------------------- factory

    @classmethod
    def default(
        cls,
        *,
        brave_api_key_present: bool = False,
        duckduckgo_enabled: bool = True,
    ) -> SearchProviderRegistry:
        """Return the standard multi-provider registry.

        Production priority (mainland-China deployment where SearXNG is the
        primary, controllable entry point):
        ``SearXNG > Bing > Brave > DuckDuckGo``. SearXNG and Bing are always
        present because they underpin the current infrastructure. Brave is
        registered only when an API key is present (it must never be the
        default first choice in a key-less environment). DuckDuckGo is enabled
        in development / benchmark environments by default and can be turned
        off via configuration; it sits last as the third-party failover.
        """

        capabilities: list[ProviderCapability] = []
        capabilities.append(
            ProviderCapability(
                provider_name="SearXNG",
                supports_web_search=True,
                supports_language=("en", "zh"),
                rate_limit_per_minute=None,
                cost_class=CostClass.FREE,
                priority=1,
                enabled=True,
            )
        )
        capabilities.append(
            ProviderCapability(
                provider_name="Bing",
                supports_web_search=True,
                supports_language=("en", "zh"),
                rate_limit_per_minute=None,
                cost_class=CostClass.FREE,
                priority=2,
                enabled=True,
            )
        )
        if brave_api_key_present:
            capabilities.append(
                ProviderCapability(
                    provider_name="Brave",
                    supports_web_search=True,
                    supports_language=("en", "zh"),
                    rate_limit_per_minute=2000,
                    cost_class=CostClass.LOW,
                    priority=3,
                    enabled=True,
                )
            )
        if duckduckgo_enabled:
            capabilities.append(
                ProviderCapability(
                    provider_name="DuckDuckGo",
                    supports_web_search=True,
                    supports_language=("en", "zh"),
                    rate_limit_per_minute=None,
                    cost_class=CostClass.FREE,
                    priority=4,
                    enabled=True,
                )
            )
        return cls(capabilities)


__all__ = [
    "CostClass",
    "ProviderCapability",
    "SearchProviderRegistry",
]
