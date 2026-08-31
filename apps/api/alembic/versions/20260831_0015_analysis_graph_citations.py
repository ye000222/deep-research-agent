"""Link declarative analysis artifacts into the evidence graph and citations."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260831_0015"
down_revision = "20260831_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_artifact_claims",
        sa.Column("analysis_artifact_id", sa.Uuid(), nullable=False),
        sa.Column("claim_id", sa.Uuid(), nullable=False),
        sa.Column("relation", sa.String(30), nullable=False, server_default="derived_from"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_artifact_id"], ["analysis_artifacts.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["claim_id"], ["research_claims.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("analysis_artifact_id", "claim_id"),
    )
    op.create_index(
        "ix_analysis_artifact_claims_claim",
        "analysis_artifact_claims",
        ["claim_id"],
    )
    op.add_column(
        "report_citations",
        sa.Column("analysis_artifact_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_report_citations_analysis_artifact",
        "report_citations",
        "analysis_artifacts",
        ["analysis_artifact_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_report_citations_analysis_artifact",
        "report_citations",
        type_="foreignkey",
    )
    op.drop_column("report_citations", "analysis_artifact_id")
    op.drop_index("ix_analysis_artifact_claims_claim", table_name="analysis_artifact_claims")
    op.drop_table("analysis_artifact_claims")
