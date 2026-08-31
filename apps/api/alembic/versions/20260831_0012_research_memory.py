"""Add Working, Episodic, and Semantic Research Memory.

Revision ID: 20260831_0012
Revises: 20260831_0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260831_0012"
down_revision: str | None = "20260831_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("origin_run_id", sa.Uuid(), nullable=False),
        sa.Column("owner_hash", sa.String(length=64), nullable=False),
        sa.Column("scope_type", sa.String(length=30), nullable=False),
        sa.Column("scope_id", sa.String(length=100), nullable=False),
        sa.Column("memory_type", sa.String(length=30), nullable=False),
        sa.Column("content_summary", sa.Text(), nullable=False),
        sa.Column("source_ref_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("keywords", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer(), nullable=False),
        sa.Column("utility_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("access_count >= 0", name="ck_memory_access_count"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_memory_confidence"),
        sa.CheckConstraint("importance >= 0 AND importance <= 1", name="ck_memory_importance"),
        sa.CheckConstraint("utility_count >= 0", name="ck_memory_utility_count"),
        sa.ForeignKeyConstraint(["origin_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_memory_item_run_type_fingerprint",
        "memory_items",
        ["origin_run_id", "memory_type", "fingerprint"],
        unique=True,
    )
    op.create_index(
        "ix_memory_items_owner_status_type",
        "memory_items",
        ["owner_hash", "status", "memory_type"],
    )
    op.create_index(
        "ix_memory_items_run_status",
        "memory_items",
        ["origin_run_id", "status"],
    )
    op.create_table(
        "memory_access_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("requested_types", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("candidate_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("selected_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("score_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", sa.String(length=30), nullable=False),
        sa.Column("revalidation_required_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_memory_access_logs_run_created",
        "memory_access_logs",
        ["run_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_access_logs_run_created", table_name="memory_access_logs")
    op.drop_table("memory_access_logs")
    op.drop_index("ix_memory_items_run_status", table_name="memory_items")
    op.drop_index("ix_memory_items_owner_status_type", table_name="memory_items")
    op.drop_index("uq_memory_item_run_type_fingerprint", table_name="memory_items")
    op.drop_table("memory_items")
