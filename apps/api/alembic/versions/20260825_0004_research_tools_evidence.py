"""Persist research gaps, tool calls, sources, and evidence."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260825_0004"
down_revision: str | None = "20260825_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_gaps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("gap_type", sa.String(length=30), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("acceptance_criteria", sa.Text(), nullable=False),
        sa.Column("severity", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("resolution_attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_gap_question",
        "research_gaps",
        ["run_id", "plan_version", "question_id"],
        unique=True,
    )
    op.create_index("ix_research_gaps_run_status", "research_gaps", ["run_id", "status"])

    op.create_table(
        "research_tool_calls",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("gap_id", sa.Uuid(), nullable=False),
        sa.Column("action_id", sa.Uuid(), nullable=False),
        sa.Column("tool_name", sa.String(length=50), nullable=False),
        sa.Column("duplicate_key", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("arguments", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result_refs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["gap_id"], ["research_gaps.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_tool_call_dedupe",
        "research_tool_calls",
        ["run_id", "tool_name", "duplicate_key"],
        unique=True,
    )
    op.create_index(
        "ix_research_tool_calls_run_status",
        "research_tool_calls",
        ["run_id", "status"],
    )

    op.create_table(
        "research_search_queries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("tool_call_id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("normalized_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tool_call_id"], ["research_tool_calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_search_query_hash",
        "research_search_queries",
        ["run_id", "normalized_hash"],
        unique=True,
    )
    op.create_index(
        "ix_research_search_queries_run",
        "research_search_queries",
        ["run_id", "created_at"],
    )

    op.create_table(
        "research_search_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("search_query_id", sa.Uuid(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("published_at", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(
            ["search_query_id"], ["research_search_queries.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_search_result_rank",
        "research_search_results",
        ["search_query_id", "rank"],
        unique=True,
    )

    op.create_table(
        "research_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("url_hash", sa.String(length=64), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("reliability", sa.Float(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_source_url_hash",
        "research_sources",
        ["run_id", "url_hash"],
        unique=True,
    )
    op.create_index("ix_research_sources_run_domain", "research_sources", ["run_id", "domain"])

    op.create_table(
        "research_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("exact_quote", sa.Text(), nullable=False),
        sa.Column("relation", sa.String(length=30), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source_reliability", sa.Float(), nullable=False),
        sa.Column("evidence_score", sa.Float(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("rejection_reason", sa.String(length=100), nullable=True),
        sa.Column("evidence_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["research_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_evidence_hash",
        "research_evidence",
        ["run_id", "evidence_hash"],
        unique=True,
    )
    op.create_index(
        "ix_research_evidence_run_accepted",
        "research_evidence",
        ["run_id", "accepted"],
    )
    op.create_index(
        "ix_research_evidence_question",
        "research_evidence",
        ["run_id", "question_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_research_evidence_question", table_name="research_evidence")
    op.drop_index("ix_research_evidence_run_accepted", table_name="research_evidence")
    op.drop_index("uq_research_evidence_hash", table_name="research_evidence")
    op.drop_table("research_evidence")
    op.drop_index("ix_research_sources_run_domain", table_name="research_sources")
    op.drop_index("uq_research_source_url_hash", table_name="research_sources")
    op.drop_table("research_sources")
    op.drop_index("uq_research_search_result_rank", table_name="research_search_results")
    op.drop_table("research_search_results")
    op.drop_index("ix_research_search_queries_run", table_name="research_search_queries")
    op.drop_index("uq_research_search_query_hash", table_name="research_search_queries")
    op.drop_table("research_search_queries")
    op.drop_index("ix_research_tool_calls_run_status", table_name="research_tool_calls")
    op.drop_index("uq_research_tool_call_dedupe", table_name="research_tool_calls")
    op.drop_table("research_tool_calls")
    op.drop_index("ix_research_gaps_run_status", table_name="research_gaps")
    op.drop_index("uq_research_gap_question", table_name="research_gaps")
    op.drop_table("research_gaps")
