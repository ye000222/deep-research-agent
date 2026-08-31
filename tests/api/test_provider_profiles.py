from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.api.dependencies import get_client_session, get_profile_service
from app.core.config import Settings
from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.main import create_app
from app.security.client_sessions import ClientSession
from fastapi.testclient import TestClient
from pydantic import SecretStr


class FakeProfileService:
    def __init__(self) -> None:
        self.received_secret: str | None = None
        self.view = ProviderProfileView(
            profile_id=uuid4(),
            name="Main model",
            adapter_type=AdapterType.OPENAI_RESPONSES,
            base_url="https://api.openai.com/v1",
            endpoint_host="api.openai.com",
            model="chosen-model",
            status="active",
            version=1,
            is_default=True,
            credential_version_id=uuid4(),
            credential_version=1,
            credential_last_four="3456",
            credential_fingerprint="a" * 64,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    async def list_profiles(self, owner_hash: str) -> list[ProviderProfileView]:
        return [self.view]

    async def create_profile(
        self,
        owner_hash: str,
        *,
        name: str,
        adapter_type: AdapterType,
        base_url: str,
        model: str,
        api_key: SecretStr,
        is_default: bool,
    ) -> ProviderProfileView:
        self.received_secret = api_key.get_secret_value()
        return self.view

    async def update_profile(
        self,
        owner_hash: str,
        profile_id: UUID,
        *,
        adapter_type: AdapterType | None,
        name: str | None,
        base_url: str | None,
        model: str | None,
        is_default: bool | None,
    ) -> ProviderProfileView:
        return self.view

    async def rotate_credential(
        self,
        owner_hash: str,
        profile_id: UUID,
        api_key: SecretStr,
    ) -> ProviderProfileView:
        self.received_secret = api_key.get_secret_value()
        return self.view

    async def delete_profile(self, owner_hash: str, profile_id: UUID) -> None:
        return None


def make_client(fake: FakeProfileService) -> TestClient:
    settings = Settings(
        app_env="test",
        external_probes_enabled=False,
        langgraph_strict_msgpack=False,
    )
    application = create_app(settings)
    application.dependency_overrides[get_profile_service] = lambda: fake
    application.dependency_overrides[get_client_session] = lambda: ClientSession(
        client_id=uuid4(),
        owner_hash="owner",
        is_new=False,
    )
    return TestClient(application)


def test_create_profile_never_returns_api_key() -> None:
    fake = FakeProfileService()
    with make_client(fake) as client:
        response = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "Main model",
                "adapter_type": "openai_responses",
                "base_url": "https://api.openai.com/v1",
                "model": "chosen-model",
                "api_key": "sk-secret-123456",
                "is_default": True,
            },
        )

    assert response.status_code == 201
    assert fake.received_secret == "sk-secret-123456"
    body = response.json()
    assert "api_key" not in body
    assert "sk-secret" not in response.text
    assert body["credential_last_four"] == "3456"
    assert body["credential_fingerprint"] == "a" * 12


def test_list_profiles_returns_refreshable_non_secret_configuration() -> None:
    fake = FakeProfileService()
    with make_client(fake) as client:
        response = client.get("/api/v1/llm/profiles")

    assert response.status_code == 200
    assert response.json()[0]["model"] == "chosen-model"
    assert response.json()[0]["has_saved_credential"] is True
