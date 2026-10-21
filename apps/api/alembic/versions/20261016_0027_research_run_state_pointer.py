"""Add versioned canonical run-state pointer metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20261016_0027"
down_revision = "20261013_0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_runs",
        sa.Column(
            "orchestrator_version",
            sa.String(length=20),
            nullable=False,
            server_default="V2_LEGACY",
        ),
    )
    op.add_column("research_runs", sa.Column("latest_run_state_ref", sa.Text(), nullable=True))
    op.add_column(
        "research_runs", sa.Column("latest_run_state_hash", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "research_runs", sa.Column("latest_run_state_version", sa.Integer(), nullable=True)
    )
    op.create_check_constraint(
        "ck_research_run_state_pointer_consistent",
        "research_runs",
        "(latest_run_state_ref IS NULL AND latest_run_state_hash IS NULL "
        "AND latest_run_state_version IS NULL) OR "
        "(latest_run_state_ref IS NOT NULL AND latest_run_state_hash IS NOT NULL "
        "AND latest_run_state_version IS NOT NULL AND latest_run_state_version >= 1)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_research_run_state_pointer_consistent", "research_runs", type_="check"
    )
    op.drop_column("research_runs", "latest_run_state_version")
    op.drop_column("research_runs", "latest_run_state_hash")
    op.drop_column("research_runs", "latest_run_state_ref")
    op.drop_column("research_runs", "orchestrator_version")
