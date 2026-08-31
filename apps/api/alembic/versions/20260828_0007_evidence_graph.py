"""Add the relational Evidence Graph compatibility layer."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260828_0007"
down_revision: str | None = "20260826_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_sources",
        sa.Column("source_owner_key", sa.String(length=255), nullable=True),
    )
    op.add_column("research_sources", sa.Column("original_source_id", sa.Uuid()))
    op.create_foreign_key(
        "fk_research_sources_original_source",
        "research_sources",
        "research_sources",
        ["original_source_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.execute(
        "UPDATE research_sources SET source_owner_key = domain "
        "WHERE source_owner_key IS NULL"
    )
    op.alter_column("research_sources", "source_owner_key", nullable=False)
    op.create_index(
        "ix_research_sources_run_owner",
        "research_sources",
        ["run_id", "source_owner_key"],
    )

    op.create_table(
        "research_claims",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("dimension_key", sa.String(length=100), nullable=False),
        sa.Column("atomic_claim", sa.Text(), nullable=False),
        sa.Column("claim_hash", sa.String(length=64), nullable=False),
        sa.Column("claim_type", sa.String(length=50), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_research_claim_confidence",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_claim_hash",
        "research_claims",
        ["run_id", "question_id", "claim_hash"],
        unique=True,
    )
    op.create_index(
        "ix_research_claims_run_status",
        "research_claims",
        ["run_id", "status"],
    )

    op.create_table(
        "research_source_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("final_url", sa.Text(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("parser_version", sa.String(length=50), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["research_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_source_snapshot_hash",
        "research_source_snapshots",
        ["source_id", "content_hash"],
        unique=True,
    )
    op.create_index(
        "ix_research_source_snapshots_run_fetched",
        "research_source_snapshots",
        ["run_id", "fetched_at"],
    )

    op.create_table(
        "research_source_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("heading_path", sa.Text()),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("chunk_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint("char_start >= 0", name="ck_research_chunk_char_start"),
        sa.CheckConstraint("char_end >= char_start", name="ck_research_chunk_char_end"),
        sa.CheckConstraint("token_count >= 0", name="ck_research_chunk_token_count"),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["research_source_snapshots.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_source_chunk_hash",
        "research_source_chunks",
        ["snapshot_id", "chunk_hash"],
        unique=True,
    )
    op.create_index(
        "ix_research_source_chunks_run_snapshot",
        "research_source_chunks",
        ["run_id", "snapshot_id"],
    )

    for column_name, target in (
        ("claim_id", "research_claims"),
        ("snapshot_id", "research_source_snapshots"),
        ("chunk_id", "research_source_chunks"),
    ):
        op.add_column("research_evidence", sa.Column(column_name, sa.Uuid()))
        op.create_foreign_key(
            f"fk_research_evidence_{column_name.removesuffix('_id')}",
            "research_evidence",
            target,
            [column_name],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_index(
        "ix_research_evidence_claim",
        "research_evidence",
        ["run_id", "claim_id"],
    )

    op.create_table(
        "research_claim_edges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("from_claim_id", sa.Uuid(), nullable=False),
        sa.Column("to_claim_id", sa.Uuid(), nullable=False),
        sa.Column("relation", sa.String(length=30), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "from_claim_id <> to_claim_id",
            name="ck_research_claim_edge_not_self",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_research_claim_edge_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["from_claim_id"],
            ["research_claims.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["to_claim_id"],
            ["research_claims.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_research_claim_edge",
        "research_claim_edges",
        ["from_claim_id", "to_claim_id", "relation"],
        unique=True,
    )
    op.create_index(
        "ix_research_claim_edges_run_relation",
        "research_claim_edges",
        ["run_id", "relation"],
    )

    op.create_table(
        "research_conflicts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("entity", sa.Text(), nullable=False),
        sa.Column("attribute", sa.String(length=100), nullable=False),
        sa.Column("time_scope", sa.String(length=100)),
        sa.Column("geo_scope", sa.String(length=100)),
        sa.Column("definition_scope", sa.Text()),
        sa.Column("left_evidence_id", sa.Uuid(), nullable=False),
        sa.Column("right_evidence_id", sa.Uuid(), nullable=False),
        sa.Column("severity", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("resolution_summary", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "left_evidence_id <> right_evidence_id",
            name="ck_research_conflict_distinct_evidence",
        ),
        sa.ForeignKeyConstraint(
            ["left_evidence_id"],
            ["research_evidence.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["right_evidence_id"],
            ["research_evidence.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_conflicts_run_status",
        "research_conflicts",
        ["run_id", "status"],
    )
    op.create_index(
        "ix_research_conflicts_question",
        "research_conflicts",
        ["run_id", "question_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_research_conflicts_question", table_name="research_conflicts")
    op.drop_index("ix_research_conflicts_run_status", table_name="research_conflicts")
    op.drop_table("research_conflicts")
    op.drop_index("ix_research_claim_edges_run_relation", table_name="research_claim_edges")
    op.drop_index("uq_research_claim_edge", table_name="research_claim_edges")
    op.drop_table("research_claim_edges")
    op.drop_index("ix_research_evidence_claim", table_name="research_evidence")
    for column_name in ("chunk_id", "snapshot_id", "claim_id"):
        op.drop_constraint(
            f"fk_research_evidence_{column_name.removesuffix('_id')}",
            "research_evidence",
            type_="foreignkey",
        )
        op.drop_column("research_evidence", column_name)
    op.drop_index(
        "ix_research_source_chunks_run_snapshot",
        table_name="research_source_chunks",
    )
    op.drop_index(
        "uq_research_source_chunk_hash",
        table_name="research_source_chunks",
    )
    op.drop_table("research_source_chunks")
    op.drop_index(
        "ix_research_source_snapshots_run_fetched",
        table_name="research_source_snapshots",
    )
    op.drop_index(
        "uq_research_source_snapshot_hash",
        table_name="research_source_snapshots",
    )
    op.drop_table("research_source_snapshots")
    op.drop_index("ix_research_claims_run_status", table_name="research_claims")
    op.drop_index("uq_research_claim_hash", table_name="research_claims")
    op.drop_table("research_claims")
    op.drop_index("ix_research_sources_run_owner", table_name="research_sources")
    op.drop_constraint(
        "fk_research_sources_original_source",
        "research_sources",
        type_="foreignkey",
    )
    op.drop_column("research_sources", "original_source_id")
    op.drop_column("research_sources", "source_owner_key")
