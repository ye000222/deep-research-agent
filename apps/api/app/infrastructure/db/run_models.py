"""Business persistence models for Research Runs, Events, and Outbox."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ResearchRunRow(Base):
    __tablename__ = "research_runs"
    __table_args__ = (
        CheckConstraint(
            "(latest_run_state_ref IS NULL AND latest_run_state_hash IS NULL "
            "AND latest_run_state_version IS NULL) OR "
            "(latest_run_state_ref IS NOT NULL AND latest_run_state_hash IS NOT NULL "
            "AND latest_run_state_version IS NOT NULL AND latest_run_state_version >= 1)",
            name="ck_research_run_state_pointer_consistent",
        ),
        UniqueConstraint(
            "owner_hash",
            "idempotency_key",
            name="uq_research_runs_owner_idempotency",
        ),
        Index(
            "ix_research_runs_owner_created",
            "owner_hash",
            text("created_at DESC"),
        ),
        Index("ix_research_runs_status_lease", "status", "lease_until"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    owner_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_goal: Mapped[str] = mapped_column(Text, nullable=False)
    constraints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    run_kind: Mapped[str] = mapped_column(
        String(40), nullable=False, default="research", server_default="research"
    )
    orchestrator_version: Mapped[str] = mapped_column(
        String(20), nullable=False, default="V2_LEGACY", server_default="V2_LEGACY"
    )
    latest_run_state_ref: Mapped[str | None] = mapped_column(Text)
    latest_run_state_hash: Mapped[str | None] = mapped_column(String(64))
    latest_run_state_version: Mapped[int | None] = mapped_column(Integer)
    phase: Mapped[str] = mapped_column(String(30), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    plan_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    scoring_rule_version: Mapped[str] = mapped_column(String(50), default="v1", nullable=False)
    prompt_bundle_version: Mapped[str] = mapped_column(String(50), default="v1", nullable=False)
    graph_schema_revision: Mapped[str] = mapped_column(String(50), default="v1", nullable=False)
    next_event_seq: Mapped[int] = mapped_column(BigInteger, default=1, nullable=False)
    credential_status: Mapped[str] = mapped_column(String(30), nullable=False)
    saved_profile_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("llm_provider_profiles.id"),
        nullable=False,
    )
    credential_version_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("llm_credential_versions.id"),
        nullable=False,
    )
    llm_config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    budget_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    usage_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    quality_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    termination_reason: Mapped[str | None] = mapped_column(String(100))
    lease_owner: Mapped[str | None] = mapped_column(String(200))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_task_id: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentEventRow(Base):
    __tablename__ = "agent_events"
    __table_args__ = (
        UniqueConstraint("run_id", "run_seq", name="uq_agent_events_run_seq"),
        Index("ix_agent_events_run_seq", "run_id", "run_seq"),
    )

    global_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    phase: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    public_summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    refs: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TaskDispatchOutboxRow(Base):
    __tablename__ = "task_dispatch_outbox"
    __table_args__ = (
        UniqueConstraint("dispatch_key", name="uq_task_dispatch_outbox_key"),
        Index("ix_task_dispatch_outbox_pending", "status", "next_attempt_at"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    dispatch_type: Mapped[str] = mapped_column(String(30), nullable=False)
    dispatch_key: Mapped[str] = mapped_column(String(255), nullable=False)
    payload_ref: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(1000))


class ResearchPlanItemRow(Base):
    __tablename__ = "research_plan_items"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "plan_version",
            "question_id",
            name="uq_research_plan_item_version_question",
        ),
        Index("ix_research_plan_items_run_version", "run_id", "plan_version"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_requirements: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    search_hints: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResearchWorkRow(Base):
    """Durable logical research work owned by the V3 orchestrator."""

    __tablename__ = "research_work"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "producer_kind", "producer_ref", "schema_version",
            name="uq_research_work_producer_identity",
        ),
        Index("ix_research_work_run_status_priority", "run_id", "status", "priority"),
    )

    work_ref: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    producer_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    producer_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    work_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    objective: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    objective_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    constraints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    execution_bounds: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    budget_attribution: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    source_run_state_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_run_state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_run_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class WorkAttemptRow(Base):
    """Durable reservation/outcome; external dispatch is not exactly-once."""

    __tablename__ = "research_work_attempt"
    __table_args__ = (
        UniqueConstraint("work_ref", "attempt_index", name="uq_work_attempt_index"),
        UniqueConstraint("reservation_identity", name="uq_work_attempt_reservation_identity"),
        Index("ix_work_attempt_work_state", "work_ref", "state"),
    )

    attempt_ref: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    work_ref: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_work.work_ref", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(30), nullable=False)
    reservation_identity: Mapped[str] = mapped_column(String(255), nullable=False)
    source_run_state_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_run_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    query_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    tool_call_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    produced_evidence_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    reused_evidence_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    budget_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    usage_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    failure_classification: Mapped[str | None] = mapped_column(String(100))
    resume_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
