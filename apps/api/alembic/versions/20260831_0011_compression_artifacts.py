"""Add auditable Context compression artifacts.

Revision ID: 20260831_0011
Revises: 20260828_0010
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260831_0011"
down_revision = "20260828_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compression_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("context_manifest_id", sa.Uuid(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("output_hash", sa.String(length=64), nullable=False),
        sa.Column("compression_level", sa.String(length=20), nullable=False),
        sa.Column("token_before", sa.Integer(), nullable=False),
        sa.Column("token_after", sa.Integer(), nullable=False),
        sa.Column("validation_status", sa.String(length=30), nullable=False),
        sa.Column(
            "provenance_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("token_after <= token_before", name="ck_compression_reduces_tokens"),
        sa.ForeignKeyConstraint(
            ["context_manifest_id"], ["context_manifests.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_compression_artifacts_manifest",
        "compression_artifacts",
        ["context_manifest_id", "created_at"],
    )
    op.add_column("context_items", sa.Column("compression_artifact_id", sa.Uuid()))
    op.create_foreign_key(
        "fk_context_items_compression_artifact",
        "context_items",
        "compression_artifacts",
        ["compression_artifact_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_context_items_compression_artifact", "context_items", type_="foreignkey")
    op.drop_column("context_items", "compression_artifact_id")
    op.drop_index("ix_compression_artifacts_manifest", table_name="compression_artifacts")
    op.drop_table("compression_artifacts")
