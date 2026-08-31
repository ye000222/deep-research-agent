"""Persistence models for Context Budget Manager manifests and item decisions."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db import run_models as _run_models  # noqa: F401
from app.infrastructure.db.base import Base


class ContextManifestRow(Base):
    __tablename__ = "context_manifests"
    __table_args__ = (
        Index("ix_context_manifests_run_created", "run_id", "created_at"),
        Index("ix_context_manifests_run_node", "run_id", "node_name"),
        CheckConstraint("context_window > 0", name="ck_context_manifest_window_positive"),
        CheckConstraint("input_budget > 0", name="ck_context_manifest_input_positive"),
        CheckConstraint("output_reserve > 0", name="ck_context_manifest_output_positive"),
        CheckConstraint("token_after <= input_budget", name="ck_context_manifest_within_budget"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    node_name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_adapter: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    input_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    output_reserve: Mapped[int] = mapped_column(Integer, nullable=False)
    safety_margin: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    protected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    compressed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_before: Mapped[int] = mapped_column(Integer, nullable=False)
    token_after: Mapped[int] = mapped_column(Integer, nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rendered_prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_template_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    items: Mapped[list[ContextItemRow]] = relationship(
        back_populates="manifest", cascade="all, delete-orphan", order_by="ContextItemRow.ordinal"
    )
    compression_artifacts: Mapped[list[CompressionArtifactRow]] = relationship(
        back_populates="manifest",
        cascade="all, delete-orphan",
        order_by="CompressionArtifactRow.created_at",
    )


class ContextItemRow(Base):
    __tablename__ = "context_items"
    __table_args__ = (
        UniqueConstraint("context_manifest_id", "ordinal", name="uq_context_item_ordinal"),
        Index("ix_context_items_manifest_selected", "context_manifest_id", "selected", "ordinal"),
        CheckConstraint("token_count >= 0", name="ck_context_item_token_nonnegative"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    context_manifest_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("context_manifests.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    item_type: Mapped[str] = mapped_column(String(40), nullable=False)
    source_ref_type: Mapped[str | None] = mapped_column(String(50))
    source_ref_id: Mapped[str | None] = mapped_column(String(200))
    rank_score: Mapped[float] = mapped_column(Float, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    compression_level: Mapped[str] = mapped_column(String(20), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    protected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    selected_reason_code: Mapped[str] = mapped_column(String(80), nullable=False)
    compression_artifact_id: Mapped[UUID | None] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("compression_artifacts.id", ondelete="SET NULL"),
    )
    manifest: Mapped[ContextManifestRow] = relationship(back_populates="items")


class CompressionArtifactRow(Base):
    __tablename__ = "compression_artifacts"
    __table_args__ = (
        Index("ix_compression_artifacts_manifest", "context_manifest_id", "created_at"),
        CheckConstraint("token_after <= token_before", name="ck_compression_reduces_tokens"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    context_manifest_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("context_manifests.id", ondelete="CASCADE"),
        nullable=False,
    )
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    compression_level: Mapped[str] = mapped_column(String(20), nullable=False)
    token_before: Mapped[int] = mapped_column(Integer, nullable=False)
    token_after: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_status: Mapped[str] = mapped_column(String(30), nullable=False)
    provenance_refs: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    manifest: Mapped[ContextManifestRow] = relationship(back_populates="compression_artifacts")
