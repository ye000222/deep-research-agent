"""Bind report citations to Evidence Graph snapshots and guard conflict pairs.

Revision ID: 20260828_0008
Revises: 20260828_0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260828_0008"
down_revision = "20260828_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("report_citations", sa.Column("claim_id", sa.Uuid(), nullable=True))
    op.add_column("report_citations", sa.Column("snapshot_id", sa.Uuid(), nullable=True))
    op.add_column("report_citations", sa.Column("chunk_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_report_citations_claim_id",
        "report_citations",
        "research_claims",
        ["claim_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_report_citations_snapshot_id",
        "report_citations",
        "research_source_snapshots",
        ["snapshot_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_report_citations_chunk_id",
        "report_citations",
        "research_source_chunks",
        ["chunk_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.execute(
        """
        UPDATE report_citations AS citation
        SET claim_id = evidence.claim_id,
            snapshot_id = evidence.snapshot_id,
            chunk_id = evidence.chunk_id
        FROM research_evidence AS evidence
        WHERE evidence.id = citation.evidence_id
        """
    )
    op.create_index(
        "ix_report_citations_snapshot",
        "report_citations",
        ["snapshot_id"],
    )
    op.create_index(
        "uq_research_conflict_evidence_pair",
        "research_conflicts",
        ["left_evidence_id", "right_evidence_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_research_conflict_evidence_pair",
        table_name="research_conflicts",
    )
    op.drop_index("ix_report_citations_snapshot", table_name="report_citations")
    op.drop_constraint(
        "fk_report_citations_chunk_id",
        "report_citations",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_report_citations_snapshot_id",
        "report_citations",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_report_citations_claim_id",
        "report_citations",
        type_="foreignkey",
    )
    op.drop_column("report_citations", "chunk_id")
    op.drop_column("report_citations", "snapshot_id")
    op.drop_column("report_citations", "claim_id")
