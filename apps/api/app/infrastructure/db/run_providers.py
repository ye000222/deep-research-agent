"""Resolve immutable run model bindings and encrypted credential versions."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.providers import AdapterType
from app.infrastructure.db.models import CredentialVersionRow, ProviderProfileRow
from app.infrastructure.db.run_models import ResearchRunRow
from app.security.secrets import EncryptedSecret


class RunCredentialUnavailableError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class RunProviderBinding:
    run_id: UUID
    goal: str
    adapter_type: AdapterType
    base_url: str
    model: str
    credential_id: UUID
    credential_version: int
    encrypted_secret: EncryptedSecret
    context_window: int = 16_000
    max_output_tokens: int | None = None


class RunProviderBindingRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, run_id: UUID) -> RunProviderBinding:
        async with self._sessions() as session:
            result = await session.execute(
                select(ResearchRunRow, ProviderProfileRow, CredentialVersionRow)
                .join(ProviderProfileRow, ProviderProfileRow.id == ResearchRunRow.saved_profile_id)
                .join(
                    CredentialVersionRow,
                    CredentialVersionRow.id == ResearchRunRow.credential_version_id,
                )
                .where(ResearchRunRow.id == run_id)
            )
            row = result.tuples().first()
            if row is None:
                raise RunCredentialUnavailableError(str(run_id))
            run, profile, credential = row
            if credential.deleted_at is not None:
                raise RunCredentialUnavailableError(str(run_id))
            adapter_type = AdapterType(
                str(run.llm_config_snapshot.get("adapter_type", profile.adapter_type))
            )
            base_url = str(run.llm_config_snapshot.get("base_url", profile.normalized_base_url))
            model = str(run.llm_config_snapshot.get("model", profile.model))
            context_window = _positive_int(
                run.llm_config_snapshot.get(
                    "context_window", profile.non_secret_settings.get("context_window")
                ),
                default=16_000,
            )
            max_output_tokens = _optional_positive_int(
                run.llm_config_snapshot.get(
                    "max_output_tokens", profile.non_secret_settings.get("max_output_tokens")
                )
            )
            return RunProviderBinding(
                run_id=run.id,
                goal=run.normalized_goal,
                adapter_type=adapter_type,
                base_url=base_url,
                model=model,
                credential_id=credential.id,
                credential_version=credential.credential_version,
                encrypted_secret=EncryptedSecret(
                    ciphertext=credential.ciphertext,
                    nonce=credential.nonce,
                    key_version=credential.key_version,
                    aad_version=credential.aad_version,
                    hmac_fingerprint=credential.hmac_fingerprint,
                    last_four=credential.last_four,
                ),
                context_window=context_window,
                max_output_tokens=max_output_tokens,
            )


def _positive_int(value: object, *, default: int) -> int:
    parsed = _optional_positive_int(value)
    return parsed if parsed is not None else default


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
