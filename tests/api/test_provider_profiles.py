from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.api.dependencies import (
    get_client_session,
    get_existing_client_session,
    get_profile_service,
    get_research_run_service,
)
from app.core.config import Settings
from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.domain.research_runs import ResearchRunView, RunPhase, RunStatus
from app.main import create_app
from app.security.client_sessions import ClientSession
from app.infrastructure.db.research_runs import CredentialVersionNotFoundError
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


class FakeResearchRunService:
    def __init__(self, *, expected_owner: str = "owner", association_valid: bool = True) -> None:
        self.expected_owner = expected_owner
        self.association_valid = association_valid
        self.received: dict[str, object] | None = None

    async def create_run(self, owner_hash: str, **kwargs):
        self.received = {"owner_hash": owner_hash, **kwargs}
        if (
            owner_hash != self.expected_owner
            or not self.association_valid
            or kwargs["saved_profile_version_id"] != profile_view().credential_version_id
        ):
            raise CredentialVersionNotFoundError("not available")
        now = datetime.now(UTC)
        return (
            ResearchRunView(
                run_id=uuid4(),
                original_query=kwargs["query"],
                normalized_goal=" ".join(kwargs["query"].split()),
                status=RunStatus.QUEUED,
                phase=RunPhase.INITIALIZING,
                state_version=1,
                plan_version=0,
                next_event_seq=2,
                credential_status="ready",
                saved_profile_id=profile_view().profile_id,
                credential_version_id=profile_view().credential_version_id,
                llm_config_snapshot={},
                budget_snapshot={},
                usage_snapshot={},
                quality_snapshot={},
                termination_reason=None,
                created_at=now,
                updated_at=now,
                started_at=None,
                finished_at=None,
            ),
            True,
        )


def profile_view() -> ProviderProfileView:
    return _PROFILE_VIEW


_PROFILE_VIEW = ProviderProfileView(
    profile_id=UUID("10000000-0000-4000-8000-000000000001"),
    name="Main model",
    adapter_type=AdapterType.OPENAI_RESPONSES,
    base_url="https://api.openai.com/v1",
    endpoint_host="api.openai.com",
    model="chosen-model",
    status="active",
    version=1,
    is_default=True,
    credential_version_id=UUID("20000000-0000-4000-8000-000000000002"),
    credential_version=1,
    credential_last_four="3456",
    credential_fingerprint="a" * 64,
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC),
)


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


def test_get_profile_credential_version_id_is_accepted_by_run_create() -> None:
    fake_profile = FakeProfileService()
    fake_profile.view = _PROFILE_VIEW
    run_service = FakeResearchRunService()
    application = create_app(
        Settings(app_env="test", external_probes_enabled=False, langgraph_strict_msgpack=False)
    )
    application.dependency_overrides[get_profile_service] = lambda: fake_profile
    application.dependency_overrides[get_research_run_service] = lambda: run_service
    application.dependency_overrides[get_client_session] = lambda: ClientSession(
        client_id=UUID("30000000-0000-4000-8000-000000000003"),
        owner_hash="owner",
        is_new=False,
    )
    application.dependency_overrides[get_existing_client_session] = lambda: ClientSession(
        client_id=UUID("30000000-0000-4000-8000-000000000003"),
        owner_hash="owner",
        is_new=False,
    )

    with TestClient(application) as client:
        listed = client.get("/api/v1/llm/profiles")
        profile = listed.json()[0]
        accepted = client.post(
            "/api/v1/research-runs",
            headers={"Idempotency-Key": "profile-version-contract"},
            json={
                "query": "test query",
                "saved_profile_version_id": profile["credential_version_id"],
                "budget_tier": "quick",
                "orchestrator_version": "V3",
            },
        )

    assert listed.status_code == 200
    assert profile["profile_id"] != profile["credential_version_id"]
    assert accepted.status_code == 202
    assert run_service.received is not None
    assert run_service.received["owner_hash"] == "owner"
    assert run_service.received["saved_profile_version_id"] == _PROFILE_VIEW.credential_version_id
    assert run_service.received["orchestrator_version"] == "V3"


@pytest.mark.parametrize(
    ("expected_owner", "association_valid", "credential_id"),
    [
        ("owner", True, "90000000-0000-4000-8000-000000000009"),
        ("another-owner", True, str(_PROFILE_VIEW.credential_version_id)),
        ("owner", False, str(_PROFILE_VIEW.credential_version_id)),
    ],
    ids=["missing-version", "different-owner", "broken-profile-association"],
)
def test_run_create_rejects_unresolvable_owner_scoped_credential(
    expected_owner: str, association_valid: bool, credential_id: str
) -> None:
    run_service = FakeResearchRunService(
        expected_owner=expected_owner,
        association_valid=association_valid,
    )
    application = create_app(
        Settings(app_env="test", external_probes_enabled=False, langgraph_strict_msgpack=False)
    )
    application.dependency_overrides[get_profile_service] = lambda: FakeProfileService()
    application.dependency_overrides[get_research_run_service] = lambda: run_service
    application.dependency_overrides[get_client_session] = lambda: ClientSession(
        client_id=UUID("30000000-0000-4000-8000-000000000003"),
        owner_hash="owner",
        is_new=False,
    )
    application.dependency_overrides[get_existing_client_session] = lambda: ClientSession(
        client_id=UUID("30000000-0000-4000-8000-000000000003"),
        owner_hash="owner",
        is_new=False,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/research-runs",
            headers={"Idempotency-Key": "unresolvable-profile-version"},
            json={
                "query": "test query",
                "saved_profile_version_id": credential_id,
                "budget_tier": "quick",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "credential_version_not_found"
    assert "this session" in response.json()["detail"]["message"]


def test_run_create_requires_an_existing_session() -> None:
    from fastapi import HTTPException

    application = create_app(
        Settings(app_env="test", external_probes_enabled=False, langgraph_strict_msgpack=False)
    )

    def reject_missing_session() -> None:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTHENTICATED_CLIENT_SESSION_REQUIRED"},
        )

    application.dependency_overrides[get_existing_client_session] = reject_missing_session
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/research-runs",
            headers={"Idempotency-Key": "missing-session"},
            json={
                "query": "test query",
                "saved_profile_version_id": str(_PROFILE_VIEW.credential_version_id),
                "budget_tier": "quick",
                "orchestrator_version": "V3",
            },
        )

    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "AUTHENTICATED_CLIENT_SESSION_REQUIRED"
