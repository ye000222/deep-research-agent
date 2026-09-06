"""Persist saved-profile capability probe results."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260906_0019"
down_revision = "20260906_0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_capability_tests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("credential_version_id", sa.Uuid(), nullable=False),
        sa.Column("adapter_type", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("test_type", sa.String(length=10), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("capability_matrix", sa.JSON(), nullable=False),
        sa.Column("selected_fallbacks", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("provider_request_id", sa.String(length=200), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("detail_code", sa.String(length=120), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["llm_provider_profiles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["credential_version_id"],
            ["llm_credential_versions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_capability_tests_profile_verified",
        "llm_capability_tests",
        ["profile_id", "verified_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_capability_tests_profile_verified", table_name="llm_capability_tests")
    op.drop_table("llm_capability_tests")
