"""Persisted provider capability probe results."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class LLMCapabilityTestRow(Base):
    __tablename__ = "llm_capability_tests"
    __table_args__ = (
        Index("ix_llm_capability_tests_profile_verified", "profile_id", "verified_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("llm_provider_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    credential_version_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("llm_credential_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    adapter_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    test_type: Mapped[str] = mapped_column(String(10), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    capability_matrix: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selected_fallbacks: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    latency_ms: Mapped[int] = mapped_column(nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(String(200))
    error_code: Mapped[str | None] = mapped_column(String(80))
    detail_code: Mapped[str | None] = mapped_column(String(120))
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
