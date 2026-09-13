"""Dependency-aware incremental evaluation planning.

The planner is deliberately pure: persistence and model evaluation remain in
the service layer, while this module makes the invalidation boundary explicit
and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvaluationScope = Literal["evidence", "question", "global", "section", "report"]


@dataclass(frozen=True, slots=True)
class EvaluationInvalidation:
    touched_claim_ids: frozenset[str]
    touched_question_ids: frozenset[str]
    scopes: tuple[EvaluationScope, ...]
    invalidate_report_draft: bool
    reason: str


def plan_incremental_invalidation(
    *,
    claim_ids: set[str] | frozenset[str] = frozenset(),
    question_ids: set[str] | frozenset[str] = frozenset(),
    evidence_changed: bool = False,
    evidence_revoked: bool = False,
    conflict_changed: bool = False,
) -> EvaluationInvalidation:
    """Return the smallest safe evaluation/cache invalidation boundary.

    Any evidence mutation affects the question and global quality scores. A
    revoke or conflict change also affects report sections and the final report;
    this intentionally errs toward invalidation so stale citations cannot be
    reused.
    """

    claims = frozenset(claim_ids)
    questions = frozenset(question_ids)
    evidence_mutation = evidence_changed or evidence_revoked or conflict_changed
    scopes: list[EvaluationScope] = []
    if claims or questions or evidence_mutation:
        scopes.extend(("evidence", "question", "global"))
    if evidence_revoked or conflict_changed:
        scopes.extend(("section", "report"))
    # preserve declaration order while avoiding duplicates
    ordered = tuple(dict.fromkeys(scopes))
    return EvaluationInvalidation(
        touched_claim_ids=claims,
        touched_question_ids=questions,
        scopes=ordered,
        invalidate_report_draft=bool(evidence_mutation),
        reason=(
            "evidence_revoked_or_conflict_changed"
            if evidence_revoked or conflict_changed
            else "evidence_changed"
            if evidence_changed
            else "claim_or_question_touched"
            if claims or questions
            else "no_dependency_change"
        ),
    )
