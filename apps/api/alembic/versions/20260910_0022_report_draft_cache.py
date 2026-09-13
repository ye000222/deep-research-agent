"""Persist per-run report drafts keyed by assembled writing context."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260910_0022"
down_revision = "20260910_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "report_draft_cache",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("context_hash", sa.String(length=64), nullable=False),
        sa.Column("writer_version", sa.String(length=80), nullable=False),
        sa.Column("draft", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_report_draft_cache_context",
        "report_draft_cache",
        ["run_id", "context_hash"],
        unique=True,
    )
    op.create_index(
        "ix_report_draft_cache_run_created",
        "report_draft_cache",
        ["run_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_report_draft_cache_run_created", table_name="report_draft_cache")
    op.drop_index("uq_report_draft_cache_context", table_name="report_draft_cache")
    op.drop_table("report_draft_cache")