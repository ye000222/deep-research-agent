"""Atomic per-attempt model token reservation ledger.

A reservation is created before a provider call leaves the process, is
settled once with the audited usage, and never refunds an attempt whose
billing outcome is unknown (status ``uncertain`` keeps occupying its
conservative upper bound until reconciliation).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class ModelBudgetReservationRow(Base):
    __tablename__ = "model_budget_reservations"
    __table_args__ = (
        Index(
            "uq_model_budget_reservation_attempt",
            "run_id",
            "attempt_id",
            unique=True,
        ),
        Index("ix_model_budget_reservations_run_status", "run_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        SQLUuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    node: Mapped[str] = mapped_column(String(100), nullable=False)
    attempt_id: Mapped[UUID] = mapped_column(SQLUuid(as_uuid=True), nullable=False)
    estimated_input: Mapped[int] = mapped_column(Integer, nullable=False)
    max_output: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_total: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_total: Mapped[int | None] = mapped_column(Integer)
    # reserved -> settled | released | uncertain
    status: Mapped[str] = mapped_column(String(20), default="reserved", nullable=False)
    usage_estimated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
