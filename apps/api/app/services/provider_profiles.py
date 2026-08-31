"""Provider profile service with URL policy and credential encryption."""

from __future__ import annotations

import ipaddress
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

from pydantic import SecretStr

from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.security.secrets import EncryptedSecret, SecretCipher


class ProviderProfileServiceProtocol(Protocol):
    async def list_profiles(self, owner_hash: str) -> list[ProviderProfileView]: ...

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
    ) -> ProviderProfileView: ...

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
    ) -> ProviderProfileView: ...

    async def rotate_credential(
        self, owner_hash: str, profile_id: UUID, api_key: SecretStr
    ) -> ProviderProfileView: ...

    async def delete_profile(self, owner_hash: str, profile_id: UUID) -> None: ...


class ProviderProfileService:
    def __init__(
        self,
        repository: ProviderProfileRepository,
        cipher: SecretCipher,
        *,
        allow_insecure_endpoints: bool,
    ) -> None:
        self._repository = repository
        self._cipher = cipher
        self._allow_insecure_endpoints = allow_insecure_endpoints

    async def list_profiles(self, owner_hash: str) -> list[ProviderProfileView]:
        rows = await self._repository.list_active(owner_hash)
        return [self._view(*row) for row in rows]

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
        normalized_url, endpoint_host = self._normalize_base_url(base_url)
        profile_id = uuid4()
        credential_id = uuid4()
        encrypted = self._cipher.encrypt(
            api_key,
            credential_id=credential_id,
            adapter_type=adapter_type.value,
            credential_version=1,
        )
        profile = ProviderProfileRow(
            id=profile_id,
            owner_hash=owner_hash,
            name=name,
            adapter_type=adapter_type.value,
            normalized_base_url=normalized_url,
            endpoint_host=endpoint_host,
            model=model,
            is_default=is_default,
        )
        credential = self._credential_row(credential_id, profile_id, 1, encrypted)
        return self._view(*await self._repository.create(profile, credential))

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
        profile: ProviderProfileRow | None = None
        current: CredentialVersionRow | None = None
        if adapter_type is not None:
            profile, current = await self._repository.get_active(owner_hash, profile_id)
        if (
            adapter_type is not None
            and profile is not None
            and current is not None
            and adapter_type.value != profile.adapter_type
        ):
            target_url = base_url if base_url is not None else profile.normalized_base_url
            replacement_url, replacement_host = self._normalize_base_url(target_url)
            plaintext = self._cipher.decrypt(
                self._encrypted_secret(current),
                credential_id=current.id,
                adapter_type=profile.adapter_type,
                credential_version=current.credential_version,
            )
            credential_id = uuid4()
            credential_version = current.credential_version + 1
            encrypted = self._cipher.encrypt(
                plaintext,
                credential_id=credential_id,
                adapter_type=adapter_type.value,
                credential_version=credential_version,
            )
            credential = self._credential_row(
                credential_id,
                profile_id,
                credential_version,
                encrypted,
            )
            rows = await self._repository.replace_adapter(
                owner_hash,
                profile_id,
                expected_credential_id=current.id,
                adapter_type=adapter_type.value,
                normalized_base_url=replacement_url,
                endpoint_host=replacement_host,
                name=name,
                model=model,
                is_default=is_default,
                credential=credential,
            )
            return self._view(*rows)

        normalized_url: str | None = None
        endpoint_host: str | None = None
        if base_url is not None:
            normalized_url, endpoint_host = self._normalize_base_url(base_url)
        rows = await self._repository.update_non_secret(
            owner_hash,
            profile_id,
            name=name,
            normalized_base_url=normalized_url,
            endpoint_host=endpoint_host,
            model=model,
            is_default=is_default,
        )
        return self._view(*rows)

    async def rotate_credential(
        self, owner_hash: str, profile_id: UUID, api_key: SecretStr
    ) -> ProviderProfileView:
        profile, current = await self._repository.get_active(owner_hash, profile_id)
        next_version = current.credential_version + 1
        credential_id = uuid4()
        encrypted = self._cipher.encrypt(
            api_key,
            credential_id=credential_id,
            adapter_type=profile.adapter_type,
            credential_version=next_version,
        )
        credential = self._credential_row(
            credential_id,
            profile_id,
            next_version,
            encrypted,
        )
        rows = await self._repository.rotate(owner_hash, profile_id, credential)
        return self._view(*rows)

    async def delete_profile(self, owner_hash: str, profile_id: UUID) -> None:
        await self._repository.delete(owner_hash, profile_id)

    def _normalize_base_url(self, raw_url: str) -> tuple[str, str]:
        parsed = urlsplit(raw_url.strip())
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("provider base URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("provider base URL may not contain credentials, query, or fragment")
        host = parsed.hostname.lower()
        if parsed.scheme != "https":
            local = host in {"localhost", "127.0.0.1", "::1"}
            if not (self._allow_insecure_endpoints and local):
                raise ValueError("provider base URL must use HTTPS")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if (
            address is not None
            and not address.is_global
            and host
            not in {
                "127.0.0.1",
                "::1",
            }
        ):
            raise ValueError("private or metadata IP endpoints are not allowed")
        normalized = urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                "",
            )
        )
        return normalized, parsed.netloc.lower()

    @staticmethod
    def _encrypted_secret(credential: CredentialVersionRow) -> EncryptedSecret:
        return EncryptedSecret(
            ciphertext=credential.ciphertext,
            nonce=credential.nonce,
            key_version=credential.key_version,
            aad_version=credential.aad_version,
            hmac_fingerprint=credential.hmac_fingerprint,
            last_four=credential.last_four,
        )

    @staticmethod
    def _credential_row(
        credential_id: UUID,
        profile_id: UUID,
        credential_version: int,
        encrypted: EncryptedSecret,
    ) -> CredentialVersionRow:
        return CredentialVersionRow(
            id=credential_id,
            profile_id=profile_id,
            scope="saved_profile",
            credential_version=credential_version,
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            key_version=encrypted.key_version,
            aad_version=encrypted.aad_version,
            hmac_fingerprint=encrypted.hmac_fingerprint,
            last_four=encrypted.last_four,
        )

    @staticmethod
    def _view(profile: ProviderProfileRow, credential: CredentialVersionRow) -> ProviderProfileView:
        return ProviderProfileView(
            profile_id=profile.id,
            name=profile.name,
            adapter_type=AdapterType(profile.adapter_type),
            base_url=profile.normalized_base_url,
            endpoint_host=profile.endpoint_host,
            model=profile.model,
            status=profile.status,
            version=profile.version,
            is_default=profile.is_default,
            credential_version_id=credential.id,
            credential_version=credential.credential_version,
            credential_last_four=credential.last_four,
            credential_fingerprint=credential.hmac_fingerprint,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )
