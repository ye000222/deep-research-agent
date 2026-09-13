from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.api.dependencies import get_client_session, get_research_run_service
from app.core.config import Settings
from app.domain.research_runs import RunStatus
from app.infrastructure.db.research_runs import ResearchRunNotFoundError
from app.main import create_app
from app.security.client_sessions import ClientSession
from fastapi.testclient import TestClient


def test_performance_preserves_unknown_usage_and_checks_owner() -> None:
    run_id = uuid4()
    service = SimpleNamespace(
        get_run=AsyncMock(
            return_value=SimpleNamespace(
                run_id=run_id,
                status=RunStatus.COMPLETED_WITH_LIMITATIONS,
                created_at=datetime.now(UTC),
                finished_at=None,
                usage_snapshot={},
                budget_snapshot={},
                quality_snapshot={},
                termination_reason=None,
            )
        ),
        list_llm_calls=AsyncMock(return_value=[{"usage": {}, "latency_ms": 10}]),
    )
    app = create_app(
        Settings(app_env="test", external_probes_enabled=False, langgraph_strict_msgpack=False)
    )
    app.dependency_overrides[get_research_run_service] = lambda: service
    app.dependency_overrides[get_client_session] = lambda: ClientSession(
        client_id=uuid4(),
        owner_hash="owner",
        is_new=False,
    )
    with TestClient(app) as client:
        response = client.get(f"/api/v1/research-runs/{run_id}/performance")
        assert response.status_code == 200
        assert response.json()["unavailable_usage_calls"] == 1
        assert response.json()["measurement_complete"] is False
        assert response.json()["dependency_wait_ms"] is None
        service.get_run.assert_awaited_with("owner", run_id)
        service.get_run.side_effect = ResearchRunNotFoundError(str(run_id))
        service.list_llm_calls.reset_mock()
        assert client.get(f"/api/v1/research-runs/{run_id}/performance").status_code == 404
        service.list_llm_calls.assert_not_awaited()
