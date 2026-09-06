from __future__ import annotations

import json

import httpx
import respx
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


@respx.mock
def test_quick_connection_test_returns_redacted_capability_result() -> None:
    route = respx.post("https://provider.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "probe-123",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"ok": True})},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
        )
    )
    settings = Settings(app_env="test", external_probes_enabled=False)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/llm/providers/connections/test",
            json={
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://provider.example/v1",
                "model": "probe-model",
                "api_key": "secret-probe-key",
                "test_type": "full",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["passed"] is True
    assert payload["provider_request_id"] == "probe-123"
    assert payload["capability_matrix"]["basic_generation"] is True
    assert "secret-probe-key" not in response.text
    assert route.call_count == 1


def test_connection_test_rejects_non_https_endpoint() -> None:
    settings = Settings(app_env="test", external_probes_enabled=False)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/llm/providers/connections/test",
            json={
                "adapter_type": "openai_responses",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "probe-model",
                "api_key": "secret-probe-key",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INVALID_BASE_URL"
