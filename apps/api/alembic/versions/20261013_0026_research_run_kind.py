"""Add a typed lane discriminator to research runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20261013_0026"
down_revision = "20260918_0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_runs",
        sa.Column("run_kind", sa.String(length=40), nullable=False, server_default="research"),
    )


def downgrade() -> None:
    op.drop_column("research_runs", "run_kind")
