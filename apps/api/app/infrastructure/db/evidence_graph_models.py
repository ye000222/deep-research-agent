"""Relational Evidence Graph persistence models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ResearchClaimRow(Base):
    __tablename__ = "research_claims"
    __table_args__ = (
        Index("uq_research_claim_hash", "run_id", "question_id", "claim_hash", unique=True),
        Index("ix_research_claims_run_status", "run_id", "status"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_research_claim_confidence",
        ),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    dimension_key: Mapped[str] = mapped_column(String(100), nullable=False)
    atomic_claim: Mapped[str] = mapped_column(Text, nullable=False)
    claim_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    claim_type: Mapped[str] = mapped_column(String(50), default="factual", nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="candidate", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchSourceSnapshotRow(Base):
    __tablename__ = "research_source_snapshots"
    __table_args__ = (
        Index(
            "uq_research_source_snapshot_hash",
            "source_id",
            "content_hash",
            unique=True,
        ),
        Index("ix_research_source_snapshots_run_fetched", "run_id", "fetched_at"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    final_url: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ResearchSourceChunkRow(Base):
    __tablename__ = "research_source_chunks"
    __table_args__ = (
        Index("uq_research_source_chunk_hash", "snapshot_id", "chunk_hash", unique=True),
        Index("ix_research_source_chunks_run_snapshot", "run_id", "snapshot_id"),
        CheckConstraint("char_start >= 0", name="ck_research_chunk_char_start"),
        CheckConstraint("char_end >= char_start", name="ck_research_chunk_char_end"),
        CheckConstraint("token_count >= 0", name="ck_research_chunk_token_count"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_source_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    heading_path: Mapped[str | None] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ResearchClaimEdgeRow(Base):
    __tablename__ = "research_claim_edges"
    __table_args__ = (
        Index(
            "uq_research_claim_edge",
            "from_claim_id",
            "to_claim_id",
            "relation",
            unique=True,
        ),
        Index("ix_research_claim_edges_run_relation", "run_id", "relation"),
        CheckConstraint(
            "from_claim_id <> to_claim_id",
            name="ck_research_claim_edge_not_self",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_research_claim_edge_confidence",
        ),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    from_claim_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_claims.id", ondelete="CASCADE"),
        nullable=False,
    )
    to_claim_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_claims.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(String(30), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchConflictRow(Base):
    __tablename__ = "research_conflicts"
    __table_args__ = (
        Index("ix_research_conflicts_run_status", "run_id", "status"),
        Index("ix_research_conflicts_question", "run_id", "question_id"),
        Index(
            "uq_research_conflict_evidence_pair",
            "left_evidence_id",
            "right_evidence_id",
            unique=True,
        ),
        CheckConstraint(
            "left_evidence_id <> right_evidence_id",
            name="ck_research_conflict_distinct_evidence",
        ),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    entity: Mapped[str] = mapped_column(Text, nullable=False)
    attribute: Mapped[str] = mapped_column(String(100), nullable=False)
    time_scope: Mapped[str | None] = mapped_column(String(100))
    geo_scope: Mapped[str | None] = mapped_column(String(100))
    definition_scope: Mapped[str | None] = mapped_column(Text)
    left_evidence_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_evidence.id", ondelete="CASCADE"),
        nullable=False,
    )
    right_evidence_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_evidence.id", ondelete="CASCADE"),
        nullable=False,
    )
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="open", nullable=False)
    resolution_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
