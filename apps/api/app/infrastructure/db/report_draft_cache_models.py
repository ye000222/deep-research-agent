"""Persistent report-draft cache keyed by the assembled writing context.

A hit is only valid within the same run; any change to the selected evidence,
questions, quality snapshot, or writer/template version changes the context
hash and therefore misses. The final citation verifier still runs on every
save regardless of the cache.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ReportDraftCacheRow(Base):
    __tablename__ = "report_draft_cache"
    __table_args__ = (
        Index(
            "uq_report_draft_cache_context",
            "run_id",
            "context_hash",
            unique=True,
        ),
        Index("ix_report_draft_cache_run_created", "run_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    writer_version: Mapped[str] = mapped_column(String(80), nullable=False)
    draft: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
