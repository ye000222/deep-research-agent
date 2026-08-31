"""Create encrypted LLM provider profiles and credential versions."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_provider_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_hash", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("adapter_type", sa.String(length=50), nullable=False),
        sa.Column("normalized_base_url", sa.String(length=500), nullable=False),
        sa.Column("endpoint_host", sa.String(length=255), nullable=False),
        sa.Column("auth_type", sa.String(length=30), nullable=False, server_default="bearer"),
        sa.Column(
            "non_secret_settings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.true()),
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_profiles_owner_status",
        "llm_provider_profiles",
        ["owner_hash", "status"],
    )
    op.create_index(
        "ix_llm_profiles_owner_default",
        "llm_provider_profiles",
        ["owner_hash", "is_default"],
    )
    op.create_index(
        "uq_llm_profiles_owner_default_active",
        "llm_provider_profiles",
        ["owner_hash"],
        unique=True,
        postgresql_where=sa.text("is_default IS TRUE AND deleted_at IS NULL AND status = 'active'"),
    )

    op.create_table(
        "llm_credential_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column(
            "scope",
            sa.String(length=30),
            nullable=False,
            server_default="saved_profile",
        ),
        sa.Column("credential_version", sa.Integer(), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column("key_version", sa.Integer(), nullable=False),
        sa.Column("aad_version", sa.Integer(), nullable=False),
        sa.Column("hmac_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("last_four", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("octet_length(nonce) = 12", name="ck_llm_credentials_nonce_12"),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["llm_provider_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_credentials_profile_active",
        "llm_credential_versions",
        ["profile_id", "revoked_at", "deleted_at"],
    )
    op.create_index(
        "uq_llm_credentials_one_active",
        "llm_credential_versions",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_llm_credentials_one_active", table_name="llm_credential_versions")
    op.drop_index(
        "ix_llm_credentials_profile_active",
        table_name="llm_credential_versions",
    )
    op.drop_table("llm_credential_versions")
    op.drop_index(
        "uq_llm_profiles_owner_default_active",
        table_name="llm_provider_profiles",
    )
    op.drop_index("ix_llm_profiles_owner_default", table_name="llm_provider_profiles")
    op.drop_index("ix_llm_profiles_owner_status", table_name="llm_provider_profiles")
    op.drop_table("llm_provider_profiles")
