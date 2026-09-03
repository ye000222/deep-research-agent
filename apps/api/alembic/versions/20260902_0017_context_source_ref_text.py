"""Allow unbounded context source references for legacy URL manifests."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260902_0017"
down_revision = "20260831_0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "context_items",
        "source_ref_id",
        existing_type=sa.String(length=200),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "context_items",
        "source_ref_id",
        existing_type=sa.Text(),
        type_=sa.String(length=200),
        existing_nullable=True,
    )
