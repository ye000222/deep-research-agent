"""Persistence models for research gaps, tool calls, sources, and evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ResearchGapRow(Base):
    __tablename__ = "research_gaps"
    __table_args__ = (
        Index(
            "uq_research_gap_question",
            "run_id",
            "plan_version",
            "question_id",
            unique=True,
        ),
        Index("ix_research_gaps_run_status", "run_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    gap_type: Mapped[str] = mapped_column(String(30), default="missing", nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    acceptance_criteria: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="open", nullable=False)
    resolution_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchToolCallRow(Base):
    __tablename__ = "research_tool_calls"
    __table_args__ = (
        Index(
            "uq_research_tool_call_dedupe",
            "run_id",
            "tool_name",
            "duplicate_key",
            unique=True,
        ),
        Index("ix_research_tool_calls_run_status", "run_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    gap_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_gaps.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(50), nullable=False)
    duplicate_key: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result_refs: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(100))
    retryable: Mapped[bool | None] = mapped_column(Boolean)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SearchQueryRow(Base):
    __tablename__ = "research_search_queries"
    __table_args__ = (
        Index("uq_research_search_query_hash", "run_id", "normalized_hash", unique=True),
        Index("ix_research_search_queries_run", "run_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    tool_call_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_tool_calls.id", ondelete="CASCADE"),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SearchResultRow(Base):
    __tablename__ = "research_search_results"
    __table_args__ = (
        Index("uq_research_search_result_rank", "search_query_id", "rank", unique=True),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    search_query_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_search_queries.id", ondelete="CASCADE"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[str | None] = mapped_column(String(100))


class ResearchSourceRow(Base):
    __tablename__ = "research_sources"
    __table_args__ = (
        Index("uq_research_source_url_hash", "run_id", "url_hash", unique=True),
        Index("ix_research_sources_run_domain", "run_id", "domain"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    source_owner_key: Mapped[str] = mapped_column(String(255), nullable=False)
    original_source_id: Mapped[UUID | None] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_sources.id", ondelete="SET NULL"),
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    reliability: Mapped[float] = mapped_column(Float, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchEvidenceRow(Base):
    __tablename__ = "research_evidence"
    __table_args__ = (
        Index("uq_research_evidence_hash", "run_id", "evidence_hash", unique=True),
        Index("ix_research_evidence_run_accepted", "run_id", "accepted"),
        Index("ix_research_evidence_question", "run_id", "question_id"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_id: Mapped[UUID | None] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_claims.id", ondelete="RESTRICT"),
    )
    snapshot_id: Mapped[UUID | None] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_source_snapshots.id", ondelete="RESTRICT"),
    )
    chunk_id: Mapped[UUID | None] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_source_chunks.id", ondelete="RESTRICT"),
    )
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    exact_quote: Mapped[str] = mapped_column(Text, nullable=False)
    relation: Mapped[str] = mapped_column(String(30), nullable=False)
    relevance: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source_reliability: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(String(100))
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
