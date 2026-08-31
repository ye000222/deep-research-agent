"""Working, episodic, and semantic Research Memory persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class MemoryItemRow(Base):
    __tablename__ = "memory_items"
    __table_args__ = (
        Index(
            "uq_memory_item_run_type_fingerprint",
            "origin_run_id",
            "memory_type",
            "fingerprint",
            unique=True,
        ),
        Index("ix_memory_items_owner_status_type", "owner_hash", "status", "memory_type"),
        Index("ix_memory_items_run_status", "origin_run_id", "status"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_memory_confidence"),
        CheckConstraint("importance >= 0 AND importance <= 1", name="ck_memory_importance"),
        CheckConstraint("access_count >= 0", name="ck_memory_access_count"),
        CheckConstraint("utility_count >= 0", name="ck_memory_utility_count"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    origin_run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(30), default="run", nullable=False)
    scope_id: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)
    content_summary: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="active", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    utility_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemoryAccessLogRow(Base):
    __tablename__ = "memory_access_logs"
    __table_args__ = (Index("ix_memory_access_logs_run_created", "run_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    requested_types: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    candidate_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    selected_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    score_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[str] = mapped_column(String(30), nullable=False)
    revalidation_required_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
