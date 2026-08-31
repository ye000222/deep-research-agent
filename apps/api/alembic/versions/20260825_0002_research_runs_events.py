"""Create Research Runs, Agent Events, and dispatch Outbox."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260825_0002"
down_revision: str | None = "20260825_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_hash", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("normalized_goal", sa.Text(), nullable=False),
        sa.Column(
            "constraints",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("phase", sa.String(length=30), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("plan_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "scoring_rule_version",
            sa.String(length=50),
            nullable=False,
            server_default="v1",
        ),
        sa.Column(
            "prompt_bundle_version",
            sa.String(length=50),
            nullable=False,
            server_default="v1",
        ),
        sa.Column(
            "graph_schema_revision",
            sa.String(length=50),
            nullable=False,
            server_default="v1",
        ),
        sa.Column("next_event_seq", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("credential_status", sa.String(length=30), nullable=False),
        sa.Column("saved_profile_id", sa.Uuid(), nullable=False),
        sa.Column("credential_version_id", sa.Uuid(), nullable=False),
        sa.Column(
            "llm_config_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "budget_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "usage_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "quality_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("termination_reason", sa.String(length=100), nullable=True),
        sa.Column("lease_owner", sa.String(length=200), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_task_id", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["saved_profile_id"], ["llm_provider_profiles.id"]),
        sa.ForeignKeyConstraint(
            ["credential_version_id"],
            ["llm_credential_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_hash",
            "idempotency_key",
            name="uq_research_runs_owner_idempotency",
        ),
    )
    op.create_index(
        "ix_research_runs_owner_created",
        "research_runs",
        ["owner_hash", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_research_runs_status_lease",
        "research_runs",
        ["status", "lease_until"],
    )

    op.create_table(
        "agent_events",
        sa.Column(
            "global_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("run_seq", sa.BigInteger(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("phase", sa.String(length=50), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("public_summary", sa.String(length=1000), nullable=False),
        sa.Column(
            "refs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["research_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("global_id"),
        sa.UniqueConstraint(
            "run_id",
            "run_seq",
            name="uq_agent_events_run_seq",
        ),
    )
    op.create_index(
        "ix_agent_events_run_seq",
        "agent_events",
        ["run_id", "run_seq"],
    )

    op.create_table(
        "task_dispatch_outbox",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("dispatch_type", sa.String(length=30), nullable=False),
        sa.Column("dispatch_key", sa.String(length=255), nullable=False),
        sa.Column(
            "payload_ref",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=1000), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["research_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dispatch_key",
            name="uq_task_dispatch_outbox_key",
        ),
    )
    op.create_index(
        "ix_task_dispatch_outbox_pending",
        "task_dispatch_outbox",
        ["status", "next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_task_dispatch_outbox_pending",
        table_name="task_dispatch_outbox",
    )
    op.drop_table("task_dispatch_outbox")
    op.drop_index("ix_agent_events_run_seq", table_name="agent_events")
    op.drop_table("agent_events")
    op.drop_index("ix_research_runs_status_lease", table_name="research_runs")
    op.drop_index("ix_research_runs_owner_created", table_name="research_runs")
    op.drop_table("research_runs")
