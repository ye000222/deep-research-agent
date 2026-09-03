"""Validated research-plan contracts produced by the Planner."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_DYNAMIC_QUESTIONS = 12
MAX_DYNAMIC_APPEND = 3
DEFAULT_RESEARCH_RESERVE_RATIO = 0.20

ShortRequirement = Annotated[str, Field(min_length=2, max_length=60)]
ShortHint = Annotated[str, Field(min_length=2, max_length=80)]
ShortCriterion = Annotated[str, Field(min_length=2, max_length=60)]
CompactRequirement = Annotated[str, Field(min_length=2, max_length=50)]
CompactHint = Annotated[str, Field(min_length=2, max_length=50)]


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


class ResearchQuestionDraft(BaseModel):
    """Bounded model-authored fields; stable identifiers are server-owned."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=120)
    priority: int = Field(ge=1, le=3)
    rationale: str = Field(min_length=5, max_length=80)
    evidence_requirements: list[ShortRequirement] = Field(min_length=1, max_length=2)
    search_hints: list[ShortHint] = Field(min_length=1, max_length=2)


class ResearchPlanDraft(BaseModel):
    """Normal Planner contract sized for a concise 5-8 question outline."""

    model_config = ConfigDict(extra="forbid")

    scope_summary: str = Field(min_length=5, max_length=160)
    questions: list[ResearchQuestionDraft] = Field(min_length=5, max_length=8)
    completion_criteria: list[ShortCriterion] = Field(min_length=2, max_length=4)


class CompactResearchQuestionDraft(BaseModel):
    """Strict length-recovery contract used only after the first attempt fails."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=100)
    priority: int = Field(ge=1, le=3)
    rationale: str = Field(min_length=5, max_length=50)
    evidence_requirements: list[CompactRequirement] = Field(min_length=1, max_length=1)
    search_hints: list[CompactHint] = Field(min_length=0, max_length=1)


class CompactResearchPlanDraft(BaseModel):
    """Exactly five questions and two criteria for a sub-1,000-token retry."""

    model_config = ConfigDict(extra="forbid")

    scope_summary: str = Field(min_length=5, max_length=100)
    questions: list[CompactResearchQuestionDraft] = Field(min_length=5, max_length=5)
    completion_criteria: list[ShortCriterion] = Field(min_length=2, max_length=2)


def materialize_research_plan(
    goal: str,
    draft: ResearchPlanDraft | CompactResearchPlanDraft,
) -> ResearchPlan:
    """Attach immutable goal and deterministic q1..qN identifiers server-side."""

    draft_questions: list[ResearchQuestionDraft | CompactResearchQuestionDraft]
    if isinstance(draft, ResearchPlanDraft):
        draft_questions = list(draft.questions)
    else:
        draft_questions = list(draft.questions)
    return ResearchPlan(
        goal=goal,
        scope_summary=draft.scope_summary,
        questions=[
            ResearchQuestion(
                id=f"q{index}",
                question=item.question,
                priority=item.priority,
                rationale=item.rationale,
                evidence_requirements=list(item.evidence_requirements),
                search_hints=list(item.search_hints),
            )
            for index, item in enumerate(draft_questions, start=1)
        ],
        completion_criteria=list(draft.completion_criteria),
    )


def normalize_research_plan_draft_payload(
    payload: object,
    *,
    compact: bool,
) -> object:
    """Deterministically fit otherwise valid Planner content to its size contract.

    Compatible providers do not always enforce JSON Schema string/array maxima even
    when JSON mode is active. Length overflow is safe to repair server-side because
    it does not invent a research fact: preserve order and meaning, clip prose, and
    keep only the contract's highest-ranked leading items. Missing fields, invalid
    priorities, wrong types, extra fields, and too-few questions remain strict
    Pydantic failures.
    """

    if not isinstance(payload, dict):
        return payload
    normalized: dict[object, object] = dict(payload)
    scope_limit = 100 if compact else 160
    question_limit = 100 if compact else 120
    rationale_limit = 50 if compact else 80
    requirement_limit = 50 if compact else 60
    hint_limit = 50 if compact else 80
    question_count = 5 if compact else 8
    requirement_count = 1 if compact else 2
    hint_count = 1 if compact else 2
    criteria_count = 2 if compact else 4

    normalized["scope_summary"] = _clip_text(
        normalized.get("scope_summary"), scope_limit
    )
    questions = normalized.get("questions")
    if isinstance(questions, list):
        bounded_questions: list[object] = []
        for raw_question in questions[:question_count]:
            if not isinstance(raw_question, dict):
                bounded_questions.append(raw_question)
                continue
            question: dict[object, object] = dict(raw_question)
            question["question"] = _clip_text(
                question.get("question"), question_limit
            )
            question["rationale"] = _clip_text(
                question.get("rationale"), rationale_limit
            )
            question["evidence_requirements"] = _clip_text_list(
                question.get("evidence_requirements"),
                max_items=requirement_count,
                max_chars=requirement_limit,
            )
            question["search_hints"] = _clip_text_list(
                question.get("search_hints"),
                max_items=hint_count,
                max_chars=hint_limit,
            )
            bounded_questions.append(question)
        normalized["questions"] = bounded_questions
    normalized["completion_criteria"] = _clip_text_list(
        normalized.get("completion_criteria"),
        max_items=criteria_count,
        max_chars=60,
    )
    return normalized


def _clip_text(value: object, max_chars: int) -> object:
    if not isinstance(value, str):
        return value
    return " ".join(value.split())[:max_chars]


def _clip_text_list(value: object, *, max_items: int, max_chars: int) -> object:
    if not isinstance(value, list):
        return value
    return [_clip_text(item, max_chars) for item in value[:max_items]]


def fit_plan_to_budget(
    plan: ResearchPlan,
    *,
    max_iterations: int,
    reserve_ratio: float = DEFAULT_RESEARCH_RESERVE_RATIO,
) -> ResearchPlan:
    """Keep the initial plan executable within the immutable run budget."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    reserve = 0 if max_iterations < 10 else max(1, math.ceil(max_iterations * reserve_ratio))
    attempts_per_question = 1 if max_iterations < 10 else 2
    executable_questions = max(5, (max_iterations - reserve) // attempts_per_question)
    if len(plan.questions) <= executable_questions:
        return plan
    ranked = sorted(
        enumerate(plan.questions),
        key=lambda pair: (pair[1].priority, pair[0]),
    )[:executable_questions]
    retained_indexes = {index for index, _question in ranked}
    retained = [
        question
        for index, question in enumerate(plan.questions)
        if index in retained_indexes
    ]
    return plan.model_copy(update={"questions": retained})


def build_gap_resolution_hints(
    question: ResearchQuestion,
    missing_reasons: tuple[str, ...],
) -> list[str]:
    """Create novel gap-specific queries without increasing the coverage denominator."""

    reason_text = " ".join(missing_reasons).casefold()
    base = " ".join(question.question.split())
    if "独立" in reason_text or "second" in reason_text:
        suffixes = (
            "official report benchmark independent source",
            "survey comparative study evidence",
            "标准 数据集 权威报告",
        )
    elif "原文" in reason_text or "网页" in reason_text or "证据" in reason_text:
        suffixes = (
            "official documentation case study",
            "survey benchmark dataset",
            "公开报告 实证 数据",
        )
    else:
        suffixes = (
            "systematic review evidence",
            "official report statistics",
            "benchmark case study",
        )
    return [f"{base[:360]} {suffix}"[:500] for suffix in suffixes]


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
