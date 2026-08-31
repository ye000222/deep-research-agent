"""Validated research-plan contracts produced by the Planner."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q[1-9][0-9]*$")
    question: str = Field(min_length=5, max_length=500)
    priority: int = Field(ge=1, le=3)
    rationale: str = Field(min_length=5, max_length=500)
    evidence_requirements: list[str] = Field(min_length=1, max_length=5)
    search_hints: list[str] = Field(default_factory=list, max_length=3)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=5, max_length=2000)
    scope_summary: str = Field(min_length=5, max_length=1000)
    questions: list[ResearchQuestion] = Field(min_length=5, max_length=8)
    completion_criteria: list[str] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> ResearchPlan:
        identifiers = [question.id for question in self.questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research question ids must be unique")
        return self
