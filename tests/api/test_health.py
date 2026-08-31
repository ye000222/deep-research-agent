from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


def make_test_settings() -> Settings:
    return Settings(
        app_env="test",
        external_probes_enabled=False,
        langgraph_strict_msgpack=False,
    )


def test_liveness_and_readiness_without_external_probes() -> None:
    with TestClient(create_app(make_test_settings())) as client:
        live = client.get("/healthz")
        ready = client.get("/readyz")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "checks": []}


def test_request_id_is_returned() -> None:
    with TestClient(create_app(make_test_settings())) as client:
        response = client.get("/healthz", headers={"X-Request-ID": "test-request"})

    assert response.headers["X-Request-ID"] == "test-request"


def test_research_run_preflight_allows_idempotency_key() -> None:
    with TestClient(create_app(make_test_settings())) as client:
        response = client.options(
            "/api/v1/research-runs",
            headers={
                "Origin": "http://localhost:5174",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,idempotency-key,last-event-id",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    assert "idempotency-key" in allowed_headers
    assert "last-event-id" in allowed_headers
