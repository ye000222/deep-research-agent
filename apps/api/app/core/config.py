from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration.

    Provider API keys are deliberately absent. User-supplied credentials are stored through
    the credential service and referenced by opaque handles; they must never become process
    configuration, graph state, Celery payloads, or logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "DeepResearch Agent"
    app_env: str = "development"
    app_version: str = "1.0.0-rc.1"
    source_revision: str = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    database_url: str = (
        "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research"
    )
    checkpoint_database_uri: str = (
        "postgresql://deep_research:deep_research@localhost:5432/deep_research_checkpoint"
    )
    redis_url: str = "redis://localhost:6379/0"
    searxng_base_url: str = "http://localhost:8080"
    # Phase 12.4 provider pool. Brave is registered as a first-priority
    # provider only when an API key is present; otherwise it is omitted from
    # the registry. DuckDuckGo is a key-free fallback provider used in
    # development / benchmark runs and can be disabled via configuration.
    brave_api_key: SecretStr | None = None
    duckduckgo_enabled: bool = True
    # Phase 14.3 A/B switch: controls ONLY whether the Research Loop applies
    # the Phase 14.2 ResearchContextEnricher transformation to SearchTarget
    # queries.  Disabled keeps the Planner -> SearchTarget -> Provider path
    # byte-identical to the Phase 12.4 baseline (no enrichment, no event).
    # Phase 14.5 closeout: the 14.3 A/B showed no quality gain and 14.4
    # attribution showed hint overlap 1.0 with the 14.1 candidate path, so the
    # V1 default is disabled.  The flag stays available for experiments.
    evidence_aware_context_enabled: bool = False
    # Phase 15.1 Branch B targeting is part of the frozen V1 RC reference
    # configuration. Explicitly set false only for historical/A-B runs.
    # It controls ONLY whether a still-open
    # independent-source / claim-verification requirement, after Branch A's
    # identity-correct closure, carries its already-counted source *owners*
    # (canonical registrable domains) as exclusion metadata so a follow-up query
    # targets a genuinely NEW publisher and post-search candidates are scored for
    # requirement-scoped independence. This is a *declared experiment factor*
    # (see app.domain.benchmark_comparability), not baseline identity.
    independent_source_targeting_enabled: bool = True
    # Phase 15.2 experiment factor.  False preserves Phase 15.1 extraction
    # behavior; true enables claim-aware input preparation and diagnostics.
    evidence_input_quality_enabled: bool = False
    # Consecutive failed cooldown re-probes before a provider is disabled for
    # the whole run (bounds the probe -> fail -> probe loop).
    provider_max_probe_failures: int = Field(default=3, ge=1)
    # Optional Docker-reachable egress proxy for external model providers.
    # Search traffic has its own SearXNG proxy configuration.
    model_http_proxy: str | None = None
    # Optional Docker-reachable egress proxy for public page reads.  Keep this
    # separate from model_http_proxy: model and web traffic may intentionally
    # use different routes (for example, direct model access plus proxied web).
    public_web_http_proxy: str | None = None
    artifact_root: Path = Path("artifacts")

    langgraph_strict_msgpack: bool = True
    persist_provider_credentials: bool = False
    secret_master_key_base64: SecretStr | None = None
    secret_master_key_file: Path = Path("artifacts/.secrets/provider_master_key")
    provider_session_cookie_name: str = "dr_client_session"
    allow_insecure_provider_endpoints: bool = False
    cors_origins: str = "http://localhost:5173,http://localhost:5174"
    external_probes_enabled: bool = True

    checkpoint_pool_min_size: int = Field(default=1, ge=1)
    checkpoint_pool_max_size: int = Field(default=4, ge=1)
    checkpoint_pool_timeout_seconds: float = Field(default=10.0, gt=0)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def validate_security_invariants(self) -> Settings:
        if self.checkpoint_pool_max_size < self.checkpoint_pool_min_size:
            raise ValueError("checkpoint_pool_max_size must be >= checkpoint_pool_min_size")
        if (
            self.persist_provider_credentials
            and self.secret_master_key_base64 is None
            and self.app_env.lower() not in {"development", "test"}
        ):
            raise ValueError(
                "SECRET_MASTER_KEY_BASE64 is required when persisted credentials are enabled"
            )
        if self.app_env.lower() != "test" and not self.langgraph_strict_msgpack:
            raise ValueError("LANGGRAPH_STRICT_MSGPACK must remain enabled outside tests")
        if "*" in self.cors_origin_list:
            raise ValueError("CORS_ORIGINS must list explicit origins when credentials are enabled")
        return self

    def public_snapshot(self) -> dict[str, str | bool]:
        """Return safe configuration fields for diagnostics."""

        return {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "app_version": self.app_version,
            "source_revision": self.source_revision,
            "strict_checkpoint_serialization": self.langgraph_strict_msgpack,
            "external_probes_enabled": self.external_probes_enabled,
            "persist_provider_credentials": self.persist_provider_credentials,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
