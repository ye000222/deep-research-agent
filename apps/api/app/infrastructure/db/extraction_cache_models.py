"""Persistent evidence-extraction cache.

Only successful, non-empty structured extractions are stored. A cache hit must
still pass the normal evidence-acceptance rules on replay; an empty result is
never treated as a successful cache entry.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ExtractionCacheRow(Base):
    __tablename__ = "extraction_cache"
    __table_args__ = (
        Index(
            "uq_extraction_cache_key",
            "run_id",
            "question_hash",
            "snapshot_hash",
            "adapter",
            "model",
            "extractor_version",
            unique=True,
        ),
        Index("ix_extraction_cache_run_created", "run_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    adapter: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(80), nullable=False)
    evidence: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
