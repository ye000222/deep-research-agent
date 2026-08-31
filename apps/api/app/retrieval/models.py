"""Retrieval projection ORM models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class RetrievalConfigVersionRow(Base):
    __tablename__ = "retrieval_config_versions"
    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    version: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    normalizer_name: Mapped[str] = mapped_column(String(80), nullable=False)
    tokenizer_name: Mapped[str] = mapped_column(String(80), nullable=False)
    dictionary_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ranking_rule_version: Mapped[str] = mapped_column(String(80), nullable=False)
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvidenceSearchDocumentRow(Base):
    __tablename__ = "evidence_search_documents"
    __table_args__ = (
        Index("ix_evidence_search_documents_run_config", "run_id", "retrieval_config_version_id"),
        Index("ix_evidence_search_documents_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_evidence_search_documents_fuzzy",
            "fuzzy_text",
            postgresql_using="gin",
            postgresql_ops={"fuzzy_text": "gin_trgm_ops"},
        ),
    )
    evidence_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_evidence.id", ondelete="CASCADE"),
        primary_key=True,
    )
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    dimension_key: Mapped[str | None] = mapped_column(String(100))
    retrieval_config_version_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("retrieval_config_versions.id"), nullable=False
    )
    raw_search_text: Mapped[str] = mapped_column(Text, nullable=False)
    latin_text: Mapped[str] = mapped_column(Text, nullable=False)
    cjk_lexemes: Mapped[str] = mapped_column(Text, nullable=False)
    fuzzy_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[object] = mapped_column(TSVECTOR, nullable=False)
    claim_status: Mapped[str] = mapped_column(String(30), nullable=False)
    source_owner_key: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    evidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemorySearchDocumentRow(Base):
    __tablename__ = "memory_search_documents"
    __table_args__ = (
        Index("ix_memory_search_documents_scope_status", "scope_type", "scope_id", "status"),
        Index("ix_memory_search_documents_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_memory_search_documents_fuzzy",
            "fuzzy_text",
            postgresql_using="gin",
            postgresql_ops={"fuzzy_text": "gin_trgm_ops"},
        ),
    )
    memory_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("memory_items.id", ondelete="CASCADE"), primary_key=True
    )
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    retrieval_config_version_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("retrieval_config_versions.id"), nullable=False
    )
    raw_search_text: Mapped[str] = mapped_column(Text, nullable=False)
    latin_text: Mapped[str] = mapped_column(Text, nullable=False)
    cjk_lexemes: Mapped[str] = mapped_column(Text, nullable=False)
    fuzzy_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[object] = mapped_column(TSVECTOR, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
