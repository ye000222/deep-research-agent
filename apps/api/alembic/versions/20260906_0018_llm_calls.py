"""Persist redacted per-run LLM call diagnostics."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260906_0018"
down_revision = "20260902_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("node", sa.String(length=100), nullable=False),
        sa.Column("adapter", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("strategy", sa.String(length=100), nullable=False),
        sa.Column("provider_request_id", sa.String(length=255)),
        sa.Column("context_manifest_id", sa.Uuid()),
        sa.Column("finish_reason", sa.String(length=80)),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("usage", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("retry_mode", sa.String(length=80), nullable=False),
        sa.Column("error_code", sa.String(length=100)),
        sa.Column("detail_code", sa.String(length=160)),
        sa.Column("diagnostics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["context_manifest_id"], ["context_manifests.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_calls_run_created", "llm_calls", ["run_id", "created_at"])
    op.create_index("ix_llm_calls_run_node", "llm_calls", ["run_id", "node"])


def downgrade() -> None:
    op.drop_index("ix_llm_calls_run_node", table_name="llm_calls")
    op.drop_index("ix_llm_calls_run_created", table_name="llm_calls")
    op.drop_table("llm_calls")
