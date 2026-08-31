"""Transactional repository for provider profiles and credential versions."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import Select

from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow


class ProfileNotFoundError(LookupError):
    pass


class ProviderProfileRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_active(
        self, owner_hash: str
    ) -> list[tuple[ProviderProfileRow, CredentialVersionRow]]:
        async with self._sessions() as session:
            result = await session.execute(
                self._active_query(owner_hash).order_by(
                    ProviderProfileRow.is_default.desc(),
                    ProviderProfileRow.updated_at.desc(),
                )
            )
            return list(result.tuples().all())

    async def get_active(
        self, owner_hash: str, profile_id: UUID
    ) -> tuple[ProviderProfileRow, CredentialVersionRow]:
        async with self._sessions() as session:
            result = await session.execute(
                self._active_query(owner_hash).where(ProviderProfileRow.id == profile_id)
            )
            row = result.tuples().first()
            if row is None:
                raise ProfileNotFoundError(str(profile_id))
            return row

    async def create(
        self,
        profile: ProviderProfileRow,
        credential: CredentialVersionRow,
    ) -> tuple[ProviderProfileRow, CredentialVersionRow]:
        async with self._sessions() as session, session.begin():
            if profile.is_default:
                await session.execute(
                    update(ProviderProfileRow)
                    .where(
                        ProviderProfileRow.owner_hash == profile.owner_hash,
                        ProviderProfileRow.deleted_at.is_(None),
                    )
                    .values(is_default=False)
                )
            session.add(profile)
            await session.flush()
            session.add(credential)
        return profile, credential

    async def update_non_secret(
        self,
        owner_hash: str,
        profile_id: UUID,
        *,
        name: str | None,
        normalized_base_url: str | None,
        endpoint_host: str | None,
        model: str | None,
        is_default: bool | None,
    ) -> tuple[ProviderProfileRow, CredentialVersionRow]:
        async with self._sessions() as session, session.begin():
            profile = await self._lock_profile(session, owner_hash, profile_id)
            if is_default:
                await session.execute(
                    update(ProviderProfileRow)
                    .where(
                        ProviderProfileRow.owner_hash == owner_hash,
                        ProviderProfileRow.id != profile_id,
                        ProviderProfileRow.deleted_at.is_(None),
                    )
                    .values(is_default=False)
                )
            if name is not None:
                profile.name = name
            if normalized_base_url is not None:
                profile.normalized_base_url = normalized_base_url
            if endpoint_host is not None:
                profile.endpoint_host = endpoint_host
            if model is not None:
                profile.model = model
            if is_default is not None:
                profile.is_default = is_default
            profile.version += 1
            profile.updated_at = datetime.now(UTC)
            credential = await self._active_credential(session, profile_id)
        return profile, credential

    async def replace_adapter(
        self,
        owner_hash: str,
        profile_id: UUID,
        *,
        expected_credential_id: UUID,
        adapter_type: str,
        normalized_base_url: str,
        endpoint_host: str,
        name: str | None,
        model: str | None,
        is_default: bool | None,
        credential: CredentialVersionRow,
    ) -> tuple[ProviderProfileRow, CredentialVersionRow]:
        async with self._sessions() as session, session.begin():
            profile = await self._lock_profile(session, owner_hash, profile_id)
            current = await session.scalar(
                select(CredentialVersionRow)
                .where(
                    CredentialVersionRow.profile_id == profile_id,
                    CredentialVersionRow.revoked_at.is_(None),
                    CredentialVersionRow.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if current is None or current.id != expected_credential_id:
                raise ProfileNotFoundError("active credential changed during adapter update")
            if is_default:
                await session.execute(
                    update(ProviderProfileRow)
                    .where(
                        ProviderProfileRow.owner_hash == owner_hash,
                        ProviderProfileRow.id != profile_id,
                        ProviderProfileRow.deleted_at.is_(None),
                    )
                    .values(is_default=False)
                )
            now = datetime.now(UTC)
            current.revoked_at = now
            profile.adapter_type = adapter_type
            profile.normalized_base_url = normalized_base_url
            profile.endpoint_host = endpoint_host
            if name is not None:
                profile.name = name
            if model is not None:
                profile.model = model
            if is_default is not None:
                profile.is_default = is_default
            profile.version += 1
            profile.updated_at = now
            session.add(credential)
        return profile, credential

    async def rotate(
        self,
        owner_hash: str,
        profile_id: UUID,
        credential: CredentialVersionRow,
    ) -> tuple[ProviderProfileRow, CredentialVersionRow]:
        async with self._sessions() as session, session.begin():
            profile = await self._lock_profile(session, owner_hash, profile_id)
            now = datetime.now(UTC)
            await session.execute(
                update(CredentialVersionRow)
                .where(
                    CredentialVersionRow.profile_id == profile_id,
                    CredentialVersionRow.revoked_at.is_(None),
                    CredentialVersionRow.deleted_at.is_(None),
                )
                .values(revoked_at=now)
            )
            profile.version += 1
            profile.updated_at = now
            session.add(credential)
        return profile, credential

    async def delete(self, owner_hash: str, profile_id: UUID) -> None:
        async with self._sessions() as session, session.begin():
            profile = await self._lock_profile(session, owner_hash, profile_id)
            now = datetime.now(UTC)
            profile.status = "deleted"
            profile.deleted_at = now
            profile.updated_at = now
            await session.execute(
                update(CredentialVersionRow)
                .where(
                    CredentialVersionRow.profile_id == profile_id,
                    CredentialVersionRow.deleted_at.is_(None),
                )
                .values(revoked_at=now, deleted_at=now)
            )

    @staticmethod
    def _active_query(
        owner_hash: str,
    ) -> Select[tuple[ProviderProfileRow, CredentialVersionRow]]:
        return (
            select(ProviderProfileRow, CredentialVersionRow)
            .join(
                CredentialVersionRow,
                CredentialVersionRow.profile_id == ProviderProfileRow.id,
            )
            .where(
                ProviderProfileRow.owner_hash == owner_hash,
                ProviderProfileRow.status == "active",
                ProviderProfileRow.deleted_at.is_(None),
                CredentialVersionRow.revoked_at.is_(None),
                CredentialVersionRow.deleted_at.is_(None),
            )
        )

    @staticmethod
    async def _lock_profile(
        session: AsyncSession, owner_hash: str, profile_id: UUID
    ) -> ProviderProfileRow:
        profile = await session.scalar(
            select(ProviderProfileRow)
            .where(
                ProviderProfileRow.id == profile_id,
                ProviderProfileRow.owner_hash == owner_hash,
                ProviderProfileRow.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if profile is None:
            raise ProfileNotFoundError(str(profile_id))
        return profile

    @staticmethod
    async def _active_credential(session: AsyncSession, profile_id: UUID) -> CredentialVersionRow:
        credential = await session.scalar(
            select(CredentialVersionRow).where(
                CredentialVersionRow.profile_id == profile_id,
                CredentialVersionRow.revoked_at.is_(None),
                CredentialVersionRow.deleted_at.is_(None),
            )
        )
        if credential is None:
            raise ProfileNotFoundError(f"active credential for {profile_id}")
        return credential
