from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.v1.prospective_acquisition import get_acquisition_profile_repository
from app.core.config import Settings
from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.infrastructure.db.models import CredentialVersionRow
from app.main import create_app
from app.security.client_sessions import ClientSessionManager
from fastapi import Response
from fastapi.testclient import TestClient

_ENDPOINT = "/api/v1/internal/prospective-acquisition/execution-context/freeze"
_ORIGIN = "http://localhost:5174"


class FakeProfileService:
    def __init__(self, profiles: list[ProviderProfileView]) -> None:
        self.profiles = profiles
        self.scopes: list[str] = []

    async def list_profiles(self, owner_hash: str) -> list[ProviderProfileView]:
        self.scopes.append(owner_hash)
        return self.profiles


class FakeProfileRepository:
    def __init__(self, view: ProviderProfileView, owner_hash: str) -> None:
        self.owner_hash = owner_hash
        self.row = SimpleNamespace(
            id=view.profile_id,
            owner_hash=owner_hash,
            status="active",
            is_default=True,
            deleted_at=None,
            version=view.version,
            adapter_type=view.adapter_type.value,
            model=view.model,
            normalized_base_url=view.base_url,
            auth_type="bearer",
            non_secret_settings={"timeout_seconds": 20},
        )
        self.credential = CredentialVersionRow(
            id=view.credential_version_id,
            profile_id=view.profile_id,
            scope="saved_profile",
            credential_version=view.credential_version,
            ciphertext=b"fixture-ciphertext-not-a-secret",
            nonce=b"0" * 12,
            key_version=1,
            aad_version=1,
            hmac_fingerprint="d" * 64,
            last_four="0000",
            revoked_at=None,
            deleted_at=None,
        )
        self.calls: list[tuple[str, object]] = []

    async def get_active(self, owner_hash: str, profile_id: object) -> tuple[object, object]:
        self.calls.append((owner_hash, profile_id))
        if owner_hash != self.owner_hash or profile_id != self.row.id:
            raise AssertionError("profile repository received a different owner scope")
        return self.row, self.credential


def _view(*, is_default: bool = True) -> ProviderProfileView:
    return ProviderProfileView(
        profile_id=uuid4(),
        name="Test profile",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.example.test/v1",
        endpoint_host="api.example.test",
        model="test-model",
        status="active",
        version=7,
        is_default=is_default,
        credential_version_id=uuid4(),
        credential_version=3,
        credential_last_four="1234",
        credential_fingerprint="b" * 64,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _client(
    tmp_path: Path,
    *,
    profiles: list[ProviderProfileView] | None = None,
    with_cookie: bool = True,
    cookie_valid: bool = True,
) -> tuple[TestClient, FakeProfileService, FakeProfileRepository, str | None]:
    manager = ClientSessionManager(b"s" * 32, cookie_name="dr_client_session")
    session = manager.resolve(None)
    response_cookie = Response()
    manager.issue_cookie(response_cookie, session, secure=False)
    token = response_cookie.headers["set-cookie"].split("dr_client_session=", 1)[1].split(";", 1)[0]
    profile_views = profiles if profiles is not None else [_view()]
    service = FakeProfileService(profile_views)
    repository = (
        FakeProfileRepository(profile_views[0], session.owner_hash) if profile_views else None
    )

    settings = Settings(
        app_env="test",
        external_probes_enabled=False,
        langgraph_strict_msgpack=False,
        artifact_root=tmp_path,
        cors_origins=_ORIGIN,
    )
    app = create_app(settings)
    app.state.settings = settings
    app.state.client_sessions = manager
    app.state.profile_service = service
    app.state.runtime = SimpleNamespace(business_db=SimpleNamespace(session_factory=object()))
    app.dependency_overrides[get_acquisition_profile_repository] = lambda: repository
    client = TestClient(app)
    if with_cookie:
        client.cookies.set("dr_client_session", token if cookie_valid else "invalid-token")
    return client, service, repository, token


def _post(client: TestClient, body: dict[str, object] | None = None):
    return client.post(
        _ENDPOINT,
        json=body
        or {
            "population_id": "v2_1r_d4r_prospective_population_001",
            "batch_id": "v2_1r_d4r_prospective_population_001_batch_001",
            "protocol_version": "prospective-development-population.v1",
        },
        headers={"Origin": _ORIGIN},
    )


def test_session_bound_freeze_uses_scoped_service_and_redacts_secrets(tmp_path: Path) -> None:
    client, service, repository, _ = _client(tmp_path)
    response = _post(client)

    assert response.status_code == 200
    assert len(service.scopes) == 1
    assert len(service.scopes[0]) == 64
    assert repository is not None and repository.calls[0][0] == service.scopes[0]
    body = response.json()
    artifact = body["execution_context"]
    assert body["status"] == "FROZEN"
    assert artifact["owner_scope_source"] == "AUTHENTICATED_CLIENT_SESSION"
    assert "owner_hash" not in json.dumps(artifact)
    assert "api_key" not in json.dumps(artifact)
    assert "1234" not in json.dumps(artifact)
    assert artifact["provider_binding"]["fingerprint_contract"] == (
        "provider-safe-config-fingerprint.v1"
    )
    assert len(list((tmp_path / "v2_1r_d4r_a6_closure" / "execution_contexts").glob("*.json"))) == 1


@pytest.mark.parametrize("with_cookie,cookie_valid", [(False, True), (True, False)])
def test_missing_or_invalid_session_fails_closed_without_set_cookie(
    tmp_path: Path, with_cookie: bool, cookie_valid: bool
) -> None:
    client, _, _, _ = _client(
        tmp_path,
        with_cookie=with_cookie,
        cookie_valid=cookie_valid,
    )
    response = _post(client)

    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "AUTHENTICATED_CLIENT_SESSION_REQUIRED"
    assert "set-cookie" not in response.headers


def test_request_cannot_select_owner_or_provider(tmp_path: Path) -> None:
    client, _, _, _ = _client(tmp_path)

    response = _post(
        client,
        {
            "population_id": "p",
            "batch_id": "b",
            "protocol_version": "v1",
            "owner_hash": "not-accepted",
        },
    )

    assert response.status_code == 422


def test_zero_and_multiple_defaults_fail_closed(tmp_path: Path) -> None:
    no_default, _, _, _ = _client(tmp_path / "zero", profiles=[_view(is_default=False)])
    zero_response = _post(no_default)
    assert zero_response.status_code == 409
    assert (
        zero_response.json()["detail"]["error_code"]
        == "NO_DEFAULT_PROFILE_FOR_ACQUISITION_OWNER"
    )

    first, second = _view(), _view()
    multiple, _, _, _ = _client(tmp_path / "multiple", profiles=[first, second])
    multiple_response = _post(multiple)
    assert multiple_response.status_code == 409
    assert (
        multiple_response.json()["detail"]["error_code"]
        == "MULTIPLE_DEFAULT_PROFILES_FOR_ACQUISITION_OWNER"
    )


def test_identical_freeze_is_idempotent(tmp_path: Path) -> None:
    client, _, _, _ = _client(tmp_path)
    first = _post(client).json()
    second = _post(client).json()

    assert first["status"] == "FROZEN"
    assert second["status"] == "ALREADY_FROZEN"
    assert first["execution_context"] == second["execution_context"]


def test_different_session_cannot_reuse_the_same_population_batch(tmp_path: Path) -> None:
    shared_profile = _view()
    first_client, _, _, _ = _client(tmp_path, profiles=[shared_profile])
    first = _post(first_client).json()["execution_context"]

    second_client, _, _, _ = _client(tmp_path, profiles=[shared_profile])
    response = _post(second_client)

    assert response.status_code == 409
    assert (
        response.json()["detail"]["error_code"]
        == "EXECUTION_CONTEXT_ALREADY_FROZEN_DIFFERENT_IDENTITY"
    )
    files = list((tmp_path / "v2_1r_d4r_a6_closure" / "execution_contexts").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8")) == first


def test_provider_drift_does_not_overwrite_frozen_context(tmp_path: Path) -> None:
    client, service, repository, _ = _client(tmp_path)
    first = _post(client).json()["execution_context"]
    changed = _view()
    service.profiles = [changed]
    repository.row.id = changed.profile_id
    repository.row.normalized_base_url = changed.base_url
    repository.row.model = changed.model
    repository.row.version = changed.version
    repository.credential.id = changed.credential_version_id
    repository.credential.credential_version = changed.credential_version

    response = _post(client)

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "EXECUTION_PROVIDER_BINDING_CHANGED"
    artifact_files = list((tmp_path / "v2_1r_d4r_a6_closure" / "execution_contexts").glob("*.json"))
    assert len(artifact_files) == 1
    assert json.loads(artifact_files[0].read_text(encoding="utf-8")) == first


def test_invalid_credential_version_returns_safe_structured_failure(tmp_path: Path) -> None:
    client, _, repository, _ = _client(tmp_path)
    repository.credential.credential_version = None

    response = _post(client)

    assert response.status_code == 409
    assert response.json() == {
        "detail": {"error_code": "PROVIDER_CREDENTIAL_VERSION_INVALID"}
    }
    assert "ciphertext" not in response.text
    assert "fixture-ciphertext" not in response.text


def test_freeze_route_is_not_registered_outside_development_and_test() -> None:
    app = create_app(
        Settings(
            app_env="production",
            external_probes_enabled=False,
        )
    )

    assert _ENDPOINT not in app.openapi()["paths"]
