"""Atomic per-attempt model token reservation ledger."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260909_0020"
down_revision = "20260906_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_budget_reservations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("node", sa.String(length=100), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("estimated_input", sa.Integer(), nullable=False),
        sa.Column("max_output", sa.Integer(), nullable=False),
        sa.Column("reserved_total", sa.Integer(), nullable=False),
        sa.Column("actual_total", sa.Integer()),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("usage_estimated", sa.Boolean(), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_model_budget_reservation_attempt",
        "model_budget_reservations",
        ["run_id", "attempt_id"],
        unique=True,
    )
    op.create_index(
        "ix_model_budget_reservations_run_status",
        "model_budget_reservations",
        ["run_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_budget_reservations_run_status", table_name="model_budget_reservations"
    )
    op.drop_index(
        "uq_model_budget_reservation_attempt", table_name="model_budget_reservations"
    )
    op.drop_table("model_budget_reservations")
