"""Business persistence models for provider profiles and credential versions."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    func,
    text,
)
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ProviderProfileRow(Base):
    __tablename__ = "llm_provider_profiles"
    __table_args__ = (
        Index("ix_llm_profiles_owner_status", "owner_hash", "status"),
        Index("ix_llm_profiles_owner_default", "owner_hash", "is_default"),
        Index(
            "uq_llm_profiles_owner_default_active",
            "owner_hash",
            unique=True,
            postgresql_where=text(
                "is_default IS TRUE AND deleted_at IS NULL AND status = 'active'"
            ),
        ),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    owner_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    adapter_type: Mapped[str] = mapped_column(String(50), nullable=False)
    normalized_base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    endpoint_host: Mapped[str] = mapped_column(String(255), nullable=False)
    auth_type: Mapped[str] = mapped_column(String(30), default="bearer", nullable=False)
    non_secret_settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="active", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CredentialVersionRow(Base):
    __tablename__ = "llm_credential_versions"
    __table_args__ = (
        Index(
            "ix_llm_credentials_profile_active",
            "profile_id",
            "revoked_at",
            "deleted_at",
        ),
        Index(
            "uq_llm_credentials_one_active",
            "profile_id",
            unique=True,
            postgresql_where=text("revoked_at IS NULL AND deleted_at IS NULL"),
        ),
        CheckConstraint(
            "octet_length(nonce) = 12",
            name="ck_llm_credentials_nonce_12",
        ),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    profile_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("llm_provider_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    scope: Mapped[str] = mapped_column(String(30), default="saved_profile", nullable=False)
    credential_version: Mapped[int] = mapped_column(Integer, nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    key_version: Mapped[int] = mapped_column(Integer, nullable=False)
    aad_version: Mapped[int] = mapped_column(Integer, nullable=False)
    hmac_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    last_four: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
