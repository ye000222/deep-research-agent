"""Persisted deterministic evaluation snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class EvaluationSnapshotRow(Base):
    __tablename__ = "evaluation_snapshots"
    __table_args__ = (
        Index("ix_evaluation_snapshots_run_scope_created", "run_id", "scope", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    scope: Mapped[str] = mapped_column(String(30), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    coverage: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_sufficiency: Mapped[float] = mapped_column(Float, nullable=False)
    source_quality: Mapped[float] = mapped_column(Float, nullable=False)
    source_diversity: Mapped[float] = mapped_column(Float, nullable=False)
    source_independence: Mapped[float] = mapped_column(Float, nullable=False)
    cross_validation: Mapped[float] = mapped_column(Float, nullable=False)
    freshness: Mapped[float] = mapped_column(Float, nullable=False)
    conflict_resolution: Mapped[float] = mapped_column(Float, nullable=False)
    citation_completeness: Mapped[float] = mapped_column(Float, nullable=False)
    citation_support: Mapped[float] = mapped_column(Float, nullable=False)
    weak_claim_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    missing_dimension_keys: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    unresolved_conflict_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
