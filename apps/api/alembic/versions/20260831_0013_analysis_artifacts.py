"""Add declarative analysis artifacts and input provenance.

Revision ID: 20260831_0013
Revises: 20260831_0012
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260831_0013"
down_revision: str | None = "20260831_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("tool_call_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("operation", sa.String(length=50), nullable=False),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("formula", sa.Text(), nullable=False),
        sa.Column("formula_version", sa.String(length=50), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("warnings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tool_call_id"], ["research_tool_calls.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tool_call_id"),
    )
    op.create_index(
        "ix_analysis_artifacts_run_created", "analysis_artifacts", ["run_id", "created_at"]
    )
    op.create_table(
        "analysis_inputs",
        sa.Column("analysis_artifact_id", sa.Uuid(), nullable=False),
        sa.Column("evidence_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_artifact_id"], ["analysis_artifacts.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["evidence_id"], ["research_evidence.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("analysis_artifact_id", "evidence_id"),
    )
    op.create_index("ix_analysis_inputs_evidence", "analysis_inputs", ["evidence_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_inputs_evidence", table_name="analysis_inputs")
    op.drop_table("analysis_inputs")
    op.drop_index("ix_analysis_artifacts_run_created", table_name="analysis_artifacts")
    op.drop_table("analysis_artifacts")
