"""Request-scoped API dependencies."""

from fastapi import HTTPException, Request, Response, status

from app.security.client_sessions import ClientSession, ClientSessionManager
from app.services.provider_profiles import ProviderProfileServiceProtocol
from app.services.research_runs import ResearchRunServiceProtocol


def get_profile_service(request: Request) -> ProviderProfileServiceProtocol:
    service: ProviderProfileServiceProtocol | None = request.app.state.profile_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "provider_profile_persistence_disabled"},
        )
    return service


def get_research_run_service(request: Request) -> ResearchRunServiceProtocol:
    service: ResearchRunServiceProtocol | None = request.app.state.research_run_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "research_run_service_unavailable"},
        )
    return service


def get_client_session(request: Request, response: Response) -> ClientSession:
    manager: ClientSessionManager | None = request.app.state.client_sessions
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "provider_profile_persistence_disabled"},
        )
    session = manager.resolve(request.cookies.get(manager.cookie_name))
    settings = request.app.state.settings
    manager.issue_cookie(
        response,
        session,
        secure=settings.app_env.lower() not in {"development", "test"},
    )
    return session
