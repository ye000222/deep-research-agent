"""Persist deterministic evaluation snapshots for research runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260831_0016"
down_revision = "20260831_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evaluation_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(30), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("evidence_sufficiency", sa.Float(), nullable=False),
        sa.Column("source_quality", sa.Float(), nullable=False),
        sa.Column("source_diversity", sa.Float(), nullable=False),
        sa.Column("source_independence", sa.Float(), nullable=False),
        sa.Column("cross_validation", sa.Float(), nullable=False),
        sa.Column("freshness", sa.Float(), nullable=False),
        sa.Column("conflict_resolution", sa.Float(), nullable=False),
        sa.Column("citation_completeness", sa.Float(), nullable=False),
        sa.Column("citation_support", sa.Float(), nullable=False),
        sa.Column("weak_claim_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "missing_dimension_keys", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "unresolved_conflict_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("verdict", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evaluation_snapshots_run_scope_created",
        "evaluation_snapshots",
        ["run_id", "scope", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evaluation_snapshots_run_scope_created",
        table_name="evaluation_snapshots",
    )
    op.drop_table("evaluation_snapshots")
