"""Scope search-query idempotency to the research question."""

from collections.abc import Sequence

from alembic import op

revision = "20260912_0023"
down_revision = "20260910_0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("uq_research_search_query_hash", table_name="research_search_queries")
    op.create_index(
        "uq_research_search_query_question_hash",
        "research_search_queries",
        ["run_id", "question_id", "normalized_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_research_search_query_question_hash",
        table_name="research_search_queries",
    )
    op.create_index(
        "uq_research_search_query_hash",
        "research_search_queries",
        ["run_id", "normalized_hash"],
        unique=True,
    )
