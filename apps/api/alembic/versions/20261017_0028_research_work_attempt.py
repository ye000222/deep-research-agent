"""Add durable V3 ResearchWork and WorkAttempt lifecycle tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261017_0028"
down_revision = "20261016_0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_work",
        sa.Column("work_ref", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("producer_kind", sa.String(40), nullable=False),
        sa.Column("producer_ref", sa.String(255), nullable=False),
        sa.Column("work_kind", sa.String(40), nullable=False),
        sa.Column("objective", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("objective_hash", sa.String(64), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("constraints", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("execution_bounds", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("budget_attribution", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("source_run_state_ref", sa.Text(), nullable=False),
        sa.Column("source_run_state_hash", sa.String(64), nullable=False),
        sa.Column("source_run_state_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("schema_version", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("run_id", "producer_kind", "producer_ref", "schema_version", name="uq_research_work_producer_identity"),
    )
    op.create_index("ix_research_work_run_status_priority", "research_work", ["run_id", "status", "priority"])
    op.create_table(
        "research_work_attempt",
        sa.Column("attempt_ref", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("work_ref", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_work.work_ref", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(30), nullable=False),
        sa.Column("reservation_identity", sa.String(255), nullable=False),
        sa.Column("source_run_state_ref", sa.Text(), nullable=False),
        sa.Column("source_run_state_version", sa.Integer(), nullable=False),
        sa.Column("query_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("tool_call_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("produced_evidence_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("reused_evidence_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("budget_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("usage_summary", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("failure_classification", sa.String(100), nullable=True),
        sa.Column("resume_metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("schema_version", sa.String(50), nullable=False),
        sa.UniqueConstraint("work_ref", "attempt_index", name="uq_work_attempt_index"),
        sa.UniqueConstraint("reservation_identity", name="uq_work_attempt_reservation_identity"),
    )
    op.create_index("ix_work_attempt_work_state", "research_work_attempt", ["work_ref", "state"])


def downgrade() -> None:
    op.drop_index("ix_work_attempt_work_state", table_name="research_work_attempt")
    op.drop_table("research_work_attempt")
    op.drop_index("ix_research_work_run_status_priority", table_name="research_work")
    op.drop_table("research_work")
