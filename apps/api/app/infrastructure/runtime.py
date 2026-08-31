"""Application-owned external resource lifecycle."""

from dataclasses import dataclass

import httpx
from redis.asyncio import Redis

from app.context.manager import ContextBudgetManager
from app.core.config import Settings
from app.core.readiness import ReadinessRegistry
from app.infrastructure.checkpoints.lifecycle import CheckpointRuntime
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.state_runtime import ResearchStateRuntimeRepository
from app.memory.manager import ResearchMemoryManager
from app.security.client_sessions import ClientSessionManager
from app.security.secrets import SecretCipher, load_or_create_master_key
from app.services.provider_profiles import ProviderProfileService
from app.services.research_runs import ResearchRunService
from app.tools.analyze_data import AnalyzeDataTool
from app.tools.gateway import ControlledToolGateway
from app.tools.search_evidence import SearchEvidenceTool


@dataclass(slots=True)
class ApplicationRuntime:
    business_db: PostgresRuntime
    checkpoint_db: CheckpointRuntime
    redis: Redis
    http: httpx.AsyncClient
    searxng_base_url: str
    profile_service: ProviderProfileService | None
    client_sessions: ClientSessionManager | None
    research_run_service: ResearchRunService

    @classmethod
    def build(cls, settings: Settings) -> "ApplicationRuntime":
        business_db = PostgresRuntime(settings.database_url)
        profile_service: ProviderProfileService | None = None
        client_sessions: ClientSessionManager | None = None
        if settings.persist_provider_credentials:
            master_key = load_or_create_master_key(settings)
            cipher = SecretCipher(master_key)
            profile_service = ProviderProfileService(
                ProviderProfileRepository(business_db.session_factory),
                cipher,
                allow_insecure_endpoints=settings.allow_insecure_provider_endpoints,
            )
            client_sessions = ClientSessionManager(
                master_key,
                cookie_name=settings.provider_session_cookie_name,
            )
        controlled_tools = ControlledToolGateway(
            SearchEvidenceTool(business_db.session_factory),
            AnalyzeDataTool(business_db.session_factory),
        )
        research_run_service = ResearchRunService(
            ResearchRunRepository(business_db.session_factory),
            ResearchToolRepository(business_db.session_factory),
            ReportRepository(business_db.session_factory),
            ResearchStateRuntimeRepository(business_db.session_factory),
            ContextBudgetManager(business_db.session_factory),
            ResearchMemoryManager(business_db.session_factory),
            controlled_tools,
        )
        return cls(
            business_db=business_db,
            checkpoint_db=CheckpointRuntime(
                settings.checkpoint_database_uri,
                min_size=settings.checkpoint_pool_min_size,
                max_size=settings.checkpoint_pool_max_size,
            ),
            redis=Redis.from_url(settings.redis_url, decode_responses=True),
            http=httpx.AsyncClient(timeout=10.0, trust_env=False),
            searxng_base_url=settings.searxng_base_url.rstrip("/"),
            profile_service=profile_service,
            client_sessions=client_sessions,
            research_run_service=research_run_service,
        )

    def register_readiness(self, registry: ReadinessRegistry) -> None:
        registry.register("postgres", self.business_db.ping)
        registry.register("checkpoint_postgres", self.checkpoint_db.ping)
        registry.register("redis", self._ping_redis)
        registry.register("searxng", self._ping_searxng)

    async def _ping_redis(self) -> None:
        await self.redis.ping()

    async def _ping_searxng(self) -> None:
        response = await self.http.get(f"{self.searxng_base_url}/healthz")
        if response.status_code == 404:
            response = await self.http.get(self.searxng_base_url)
        response.raise_for_status()

    async def close(self) -> None:
        await self.http.aclose()
        await self.redis.aclose()
        await self.checkpoint_db.close()
        await self.business_db.close()
