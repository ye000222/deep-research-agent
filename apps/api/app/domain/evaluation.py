from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class EvaluationScope(StrEnum):
    EVIDENCE = "evidence"
    QUESTION = "question"
    GLOBAL = "global"
    SECTION = "section"
    REPORT = "report"


class EvaluationVerdict(StrEnum):
    CONTINUE = "continue"
    REPLAN = "replan"
    WRITE = "write"
    FAIL = "fail"


class EvaluationSnapshot(BaseModel):
    evaluation_id: UUID
    run_id: UUID
    scope: EvaluationScope
    state_version: int = Field(ge=0)
    plan_version: int = Field(ge=1)
    coverage: Score
    evidence_sufficiency: Score
    source_quality: Score
    source_diversity: Score
    source_independence: Score
    cross_validation: Score
    freshness: Score
    conflict_resolution: Score
    citation_completeness: Score
    citation_support: Score
    weak_claim_ids: tuple[UUID, ...] = ()
    missing_dimension_keys: tuple[str, ...] = ()
    unresolved_conflict_ids: tuple[UUID, ...] = ()
    recommended_action_ids: tuple[UUID, ...] = ()
    verdict: EvaluationVerdict
