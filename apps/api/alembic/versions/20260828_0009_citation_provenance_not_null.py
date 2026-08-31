"""Require complete Citation provenance after Evidence Graph backfill.

Revision ID: 20260828_0009
Revises: 20260828_0008
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260828_0009"
down_revision = "20260828_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE report_citations AS citation
        SET claim_id = evidence.claim_id,
            snapshot_id = evidence.snapshot_id,
            chunk_id = evidence.chunk_id
        FROM research_evidence AS evidence
        WHERE citation.evidence_id = evidence.id
          AND (
              citation.claim_id IS NULL
              OR citation.snapshot_id IS NULL
              OR citation.chunk_id IS NULL
          )
        """
    )
    op.alter_column(
        "report_citations",
        "claim_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )
    op.alter_column(
        "report_citations",
        "snapshot_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )
    op.alter_column(
        "report_citations",
        "chunk_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "report_citations",
        "chunk_id",
        existing_type=sa.Uuid(),
        nullable=True,
    )
    op.alter_column(
        "report_citations",
        "snapshot_id",
        existing_type=sa.Uuid(),
        nullable=True,
    )
    op.alter_column(
        "report_citations",
        "claim_id",
        existing_type=sa.Uuid(),
        nullable=True,
    )
