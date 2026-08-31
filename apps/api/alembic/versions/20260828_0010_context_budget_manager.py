"""Add Context Budget Manager manifests and item audit trail.

Revision ID: 20260828_0010
Revises: 20260828_0009
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260828_0010"
down_revision = "20260828_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "context_manifests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("node_name", sa.String(length=100), nullable=False),
        sa.Column("provider_adapter", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=False),
        sa.Column("input_budget", sa.Integer(), nullable=False),
        sa.Column("output_reserve", sa.Integer(), nullable=False),
        sa.Column("safety_margin", sa.Integer(), nullable=False),
        sa.Column("selected_count", sa.Integer(), nullable=False),
        sa.Column("rejected_count", sa.Integer(), nullable=False),
        sa.Column("protected_count", sa.Integer(), nullable=False),
        sa.Column("compressed_count", sa.Integer(), nullable=False),
        sa.Column("token_before", sa.Integer(), nullable=False),
        sa.Column("token_after", sa.Integer(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("rendered_prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt_template_version", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("context_window > 0", name="ck_context_manifest_window_positive"),
        sa.CheckConstraint("input_budget > 0", name="ck_context_manifest_input_positive"),
        sa.CheckConstraint("output_reserve > 0", name="ck_context_manifest_output_positive"),
        sa.CheckConstraint("token_after <= input_budget", name="ck_context_manifest_within_budget"),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_context_manifests_run_created", "context_manifests", ["run_id", "created_at"]
    )
    op.create_index("ix_context_manifests_run_node", "context_manifests", ["run_id", "node_name"])
    op.create_table(
        "context_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("context_manifest_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("item_type", sa.String(length=40), nullable=False),
        sa.Column("source_ref_type", sa.String(length=50), nullable=True),
        sa.Column("source_ref_id", sa.String(length=200), nullable=True),
        sa.Column("rank_score", sa.Float(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("compression_level", sa.String(length=20), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("protected", sa.Boolean(), nullable=False),
        sa.Column("selected_reason_code", sa.String(length=80), nullable=False),
        sa.CheckConstraint("token_count >= 0", name="ck_context_item_token_nonnegative"),
        sa.ForeignKeyConstraint(
            ["context_manifest_id"], ["context_manifests.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("context_manifest_id", "ordinal", name="uq_context_item_ordinal"),
    )
    op.create_index(
        "ix_context_items_manifest_selected",
        "context_items",
        ["context_manifest_id", "selected", "ordinal"],
    )


def downgrade() -> None:
    op.drop_index("ix_context_items_manifest_selected", table_name="context_items")
    op.drop_table("context_items")
    op.drop_index("ix_context_manifests_run_node", table_name="context_manifests")
    op.drop_index("ix_context_manifests_run_created", table_name="context_manifests")
    op.drop_table("context_manifests")
