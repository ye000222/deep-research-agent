"""Persist canonical GapRequirement state beside legacy research_gaps."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260918_0024"
down_revision = "20260912_0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gap_requirements",
        sa.Column("gap_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("dimension_key", sa.String(length=100), nullable=False),
        sa.Column("requirement_type", sa.String(length=40), nullable=False),
        sa.Column("criterion", sa.Text(), nullable=False),
        sa.Column("required_evidence_count", sa.Integer(), nullable=False),
        sa.Column("required_independent_sources", sa.Integer(), nullable=False),
        sa.Column("current_evidence_count", sa.Integer(), nullable=False),
        sa.Column("current_independent_sources", sa.Integer(), nullable=False),
        sa.Column("current_coverage", sa.Float(), nullable=False),
        sa.Column("required_coverage", sa.Float(), nullable=True),
        sa.Column("verification_status", sa.String(length=30), nullable=False),
        sa.Column("closure_status", sa.String(length=30), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("transition_reason", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("gap_id"),
    )
    op.create_index(
        "uq_gap_requirement_dimension",
        "gap_requirements",
        ["run_id", "plan_version", "dimension_key"],
        unique=True,
    )
    op.create_index(
        "ix_gap_requirements_run_status",
        "gap_requirements",
        ["run_id", "closure_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_gap_requirements_run_status", table_name="gap_requirements")
    op.drop_index("uq_gap_requirement_dimension", table_name="gap_requirements")
    op.drop_table("gap_requirements")
