"""Persistent audit rows for provider model calls."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class LLMCallRow(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (
        Index("ix_llm_calls_run_created", "run_id", "created_at"),
        Index("ix_llm_calls_run_node", "run_id", "node"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    node: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(String(255))
    context_manifest_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("context_manifests.id", ondelete="SET NULL")
    )
    finish_reason: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    usage: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    retry_mode: Mapped[str] = mapped_column(String(80), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(100))
    detail_code: Mapped[str | None] = mapped_column(String(160))
    diagnostics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
