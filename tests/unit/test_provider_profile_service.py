from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.domain.providers import AdapterType
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.security.secrets import SecretCipher
from app.services.provider_profiles import ProviderProfileService
from pydantic import SecretStr


def make_service(*, allow_insecure: bool = False) -> ProviderProfileService:
    repository = AsyncMock(spec=ProviderProfileRepository)
    return ProviderProfileService(
        repository,
        SecretCipher(b"k" * 32),
        allow_insecure_endpoints=allow_insecure,
    )


def test_provider_url_rejects_credentials_and_query() -> None:
    service = make_service()

    with pytest.raises(ValueError, match="credentials"):
        service._normalize_base_url("https://user:pass@example.com/v1")
    with pytest.raises(ValueError, match="query"):
        service._normalize_base_url("https://example.com/v1?key=secret")


def test_provider_url_requires_https_except_explicit_local_development() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        make_service()._normalize_base_url("http://localhost:11434/v1")

    normalized, host = make_service(allow_insecure=True)._normalize_base_url(
        "http://localhost:11434/v1/"
    )

    assert normalized == "http://localhost:11434/v1"
    assert host == "localhost:11434"


@pytest.mark.asyncio
async def test_adapter_change_reencrypts_existing_secret_with_new_aad() -> None:
    cipher = SecretCipher(b"z" * 32)
    repository = AsyncMock(spec=ProviderProfileRepository)
    profile_id = uuid4()
    credential_id = uuid4()
    encrypted = cipher.encrypt(
        SecretStr("preserved-provider-secret"),
        credential_id=credential_id,
        adapter_type=AdapterType.OPENAI_RESPONSES.value,
        credential_version=1,
    )
    profile = ProviderProfileRow(
        id=profile_id,
        owner_hash="owner",
        name="DeepSeek",
        adapter_type=AdapterType.OPENAI_RESPONSES.value,
        normalized_base_url="https://api.deepseek.com",
        endpoint_host="api.deepseek.com",
        model="deepseek-model",
        status="active",
        version=1,
        is_default=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    credential = CredentialVersionRow(
        id=credential_id,
        profile_id=profile_id,
        scope="saved_profile",
        credential_version=1,
        ciphertext=encrypted.ciphertext,
        nonce=encrypted.nonce,
        key_version=encrypted.key_version,
        aad_version=encrypted.aad_version,
        hmac_fingerprint=encrypted.hmac_fingerprint,
        last_four=encrypted.last_four,
    )
    repository.get_active.return_value = (profile, credential)

    async def replace_adapter(*args: object, **kwargs: object) -> object:
        replacement = kwargs["credential"]
        assert isinstance(replacement, CredentialVersionRow)
        profile.adapter_type = str(kwargs["adapter_type"])
        profile.normalized_base_url = str(kwargs["normalized_base_url"])
        profile.endpoint_host = str(kwargs["endpoint_host"])
        profile.version += 1
        return profile, replacement

    repository.replace_adapter.side_effect = replace_adapter
    service = ProviderProfileService(
        repository,
        cipher,
        allow_insecure_endpoints=False,
    )

    view = await service.update_profile(
        "owner",
        profile_id,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        name=None,
        base_url="https://api.deepseek.com",
        model=None,
        is_default=None,
    )

    replacement = repository.replace_adapter.await_args.kwargs["credential"]
    assert isinstance(replacement, CredentialVersionRow)
    decrypted = cipher.decrypt(
        service._encrypted_secret(replacement),
        credential_id=replacement.id,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT.value,
        credential_version=2,
    )
    assert decrypted.get_secret_value() == "preserved-provider-secret"
    assert view.adapter_type == AdapterType.OPENAI_COMPATIBLE_CHAT
    assert view.credential_version == 2
