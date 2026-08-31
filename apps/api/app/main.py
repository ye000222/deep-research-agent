"""DeepResearch Agent API application factory."""

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

if sys.platform == "win32":
    # psycopg async requires SelectorEventLoop; Windows defaults to ProactorEventLoop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.api.v1.router import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.readiness import ReadinessRegistry
from app.infrastructure.runtime import ApplicationRuntime

CallNext = Callable[[Request], Awaitable[Response]]


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        registry = ReadinessRegistry()
        application.state.settings = resolved
        application.state.readiness = registry
        runtime = ApplicationRuntime.build(resolved)
        application.state.runtime = runtime
        application.state.profile_service = runtime.profile_service
        application.state.client_sessions = runtime.client_sessions
        application.state.research_run_service = runtime.research_run_service
        if resolved.external_probes_enabled:
            runtime.register_readiness(registry)

        try:
            yield
        finally:
            await runtime.close()

    application = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "Last-Event-ID",
            "X-Request-ID",
        ],
    )

    @application.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: CallNext,
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    application.include_router(router)
    return application


app = create_app()
