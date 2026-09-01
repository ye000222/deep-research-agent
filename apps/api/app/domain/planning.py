"""Validated research-plan contracts produced by the Planner."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_DYNAMIC_QUESTIONS = 12
MAX_DYNAMIC_APPEND = 3


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
    questions: list[ResearchQuestion] = Field(min_length=5, max_length=MAX_DYNAMIC_QUESTIONS)
    completion_criteria: list[str] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> ResearchPlan:
        identifiers = [question.id for question in self.questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research question ids must be unique")
        return self


def append_dynamic_questions(
    plan: ResearchPlan,
    additions: list[ResearchQuestion],
) -> ResearchPlan:
    """Return a validated plan with bounded, non-duplicate gap-driven questions."""
    if len(additions) > MAX_DYNAMIC_APPEND:
        raise ValueError("at most three dynamic questions may be added per revision")
    existing_text = {" ".join(item.question.split()).casefold() for item in plan.questions}
    merged = list(plan.questions)
    for question in additions:
        normalized = " ".join(question.question.split())
        if normalized.casefold() in existing_text:
            continue
        if len(merged) >= MAX_DYNAMIC_QUESTIONS:
            raise ValueError("dynamic question budget exhausted")
        merged.append(question.model_copy(update={"question": normalized}))
        existing_text.add(normalized.casefold())
    return plan.model_copy(update={"questions": merged})


def build_gap_driven_questions(
    plan: ResearchPlan,
    gaps: list[tuple[str, str, tuple[str, ...], int]],
) -> list[ResearchQuestion]:
    """Turn evaluator-visible coverage gaps into bounded follow-up questions."""

    used_ids = {item.id for item in plan.questions}
    next_number = max(int(item.id[1:]) for item in plan.questions) + 1
    additions: list[ResearchQuestion] = []
    available = min(MAX_DYNAMIC_APPEND, MAX_DYNAMIC_QUESTIONS - len(plan.questions))
    for dimension_key, question, missing_reasons, priority in gaps:
        if len(additions) >= available:
            break
        while f"q{next_number}" in used_ids:
            next_number += 1
        reason_text = "; ".join(missing_reasons) or "缺少满足验收标准的独立证据"
        follow_up = (
            f"针对研究问题「{question}」, 还需哪些独立公开证据补齐缺口: {reason_text}?"
        )[:500]
        additions.append(
            ResearchQuestion(
                id=f"q{next_number}",
                question=follow_up,
                priority=max(1, min(priority, 3)),
                rationale=f"Evaluator 发现维度 {dimension_key} 尚未满足验收标准.",
                evidence_requirements=[
                    *missing_reasons[:4],
                    "至少一个此前未使用的独立来源",
                ][:5],
                search_hints=[
                    question[:300],
                    f"{question[:260]} independent source",
                ],
            )
        )
        used_ids.add(f"q{next_number}")
        next_number += 1
    return additions
