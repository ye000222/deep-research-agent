"""Persist successful evidence extractions for cross-question reuse."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260910_0021"
down_revision = "20260909_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "extraction_cache",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_hash", sa.String(length=64), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("adapter", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("extractor_version", sa.String(length=80), nullable=False),
        sa.Column(
            "evidence",
            sa.dialects.postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_extraction_cache_run_created",
        "extraction_cache",
        ["run_id", "created_at"],
    )
    op.create_index(
        "uq_extraction_cache_key",
        "extraction_cache",
        [
            "run_id",
            "question_hash",
            "snapshot_hash",
            "adapter",
            "model",
            "extractor_version",
        ],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_extraction_cache_key", table_name="extraction_cache")
    op.drop_index("ix_extraction_cache_run_created", table_name="extraction_cache")
    op.drop_table("extraction_cache")