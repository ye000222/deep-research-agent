"""Add provenance for canonical GapRequirement rows."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260918_0025"
down_revision = "20260918_0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "gap_requirements",
        sa.Column(
            "migration_source",
            sa.String(length=80),
            nullable=False,
            server_default="runtime",
        ),
    )


def downgrade() -> None:
    op.drop_column("gap_requirements", "migration_source")
