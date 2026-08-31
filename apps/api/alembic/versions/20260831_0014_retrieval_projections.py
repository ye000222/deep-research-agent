"""Add versioned PostgreSQL hybrid retrieval projections."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260831_0014"
down_revision = "20260831_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "retrieval_config_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.String(80), nullable=False),
        sa.Column("normalizer_name", sa.String(80), nullable=False),
        sa.Column("tokenizer_name", sa.String(80), nullable=False),
        sa.Column("dictionary_hash", sa.String(64), nullable=False),
        sa.Column("ranking_rule_version", sa.String(80), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version"),
    )
    op.create_table(
        "evidence_search_documents",
        sa.Column("evidence_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(50), nullable=False),
        sa.Column("dimension_key", sa.String(100)),
        sa.Column("retrieval_config_version_id", sa.Uuid(), nullable=False),
        sa.Column("raw_search_text", sa.Text(), nullable=False),
        sa.Column("latin_text", sa.Text(), nullable=False),
        sa.Column("cjk_lexemes", sa.Text(), nullable=False),
        sa.Column("fuzzy_text", sa.Text(), nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR(), nullable=False),
        sa.Column("claim_status", sa.String(30), nullable=False),
        sa.Column("source_owner_key", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("evidence_score", sa.Float(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["evidence_id"], ["research_evidence.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["retrieval_config_version_id"], ["retrieval_config_versions.id"]),
        sa.PrimaryKeyConstraint("evidence_id"),
    )
    op.create_index(
        "ix_evidence_search_documents_run_config",
        "evidence_search_documents",
        ["run_id", "retrieval_config_version_id"],
    )
    op.create_index(
        "ix_evidence_search_documents_vector",
        "evidence_search_documents",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_evidence_search_documents_fuzzy",
        "evidence_search_documents",
        ["fuzzy_text"],
        postgresql_using="gin",
        postgresql_ops={"fuzzy_text": "gin_trgm_ops"},
    )
    op.create_table(
        "memory_search_documents",
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(100), nullable=False),
        sa.Column("memory_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("retrieval_config_version_id", sa.Uuid(), nullable=False),
        sa.Column("raw_search_text", sa.Text(), nullable=False),
        sa.Column("latin_text", sa.Text(), nullable=False),
        sa.Column("cjk_lexemes", sa.Text(), nullable=False),
        sa.Column("fuzzy_text", sa.Text(), nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_items.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["retrieval_config_version_id"], ["retrieval_config_versions.id"]),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index(
        "ix_memory_search_documents_scope_status",
        "memory_search_documents",
        ["scope_type", "scope_id", "status"],
    )
    op.create_index(
        "ix_memory_search_documents_vector",
        "memory_search_documents",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_memory_search_documents_fuzzy",
        "memory_search_documents",
        ["fuzzy_text"],
        postgresql_using="gin",
        postgresql_ops={"fuzzy_text": "gin_trgm_ops"},
    )
    op.execute(
        "INSERT INTO retrieval_config_versions "
        "(id, version, normalizer_name, tokenizer_name, dictionary_hash, "
        "ranking_rule_version, activated_at) "
        "VALUES ('01a05800-0000-7000-8000-000000000001', 'lexical-v1', 'unicode-space-v1', "
        "'cjk-bigram-v1', repeat('0', 64), 'rrf-v1', now())"
    )


def downgrade() -> None:
    op.drop_index("ix_memory_search_documents_fuzzy", table_name="memory_search_documents")
    op.drop_index("ix_memory_search_documents_vector", table_name="memory_search_documents")
    op.drop_index("ix_memory_search_documents_scope_status", table_name="memory_search_documents")
    op.drop_table("memory_search_documents")
    op.drop_index("ix_evidence_search_documents_fuzzy", table_name="evidence_search_documents")
    op.drop_index("ix_evidence_search_documents_vector", table_name="evidence_search_documents")
    op.drop_index("ix_evidence_search_documents_run_config", table_name="evidence_search_documents")
    op.drop_table("evidence_search_documents")
    op.drop_table("retrieval_config_versions")
