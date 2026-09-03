import pytest
from app.domain.planning import (
    ResearchPlan,
    ResearchQuestion,
    append_dynamic_questions,
    build_gap_driven_questions,
    build_gap_resolution_hints,
    fit_plan_to_budget,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="研究工业视觉缺陷检测发展",
        scope_summary="技术、厂商和趋势",
        questions=[
            ResearchQuestion(
                id=f"q{index}",
                question=f"研究维度 {index} 的公开证据",
                priority=1,
                rationale="用于覆盖研究目标",
                evidence_requirements=["至少一个公开来源"],
            )
            for index in range(1, 6)
        ],
        completion_criteria=["覆盖核心维度", "记录证据来源"],
    )


def test_dynamic_questions_are_bounded_and_deduplicated() -> None:
    plan = append_dynamic_questions(
        _plan(),
        [
            ResearchQuestion(
                id="q6",
                question="  补充研究中国厂商  ",
                priority=2,
                rationale="发现厂商信息缺口",
                evidence_requirements=["至少两个公开来源"],
            ),
            ResearchQuestion(
                id="q7",
                question="研究维度 1 的公开证据",
                priority=2,
                rationale="重复问题应被忽略",
                evidence_requirements=["至少一个公开来源"],
            ),
        ],
    )

    assert len(plan.questions) == 6
    assert plan.questions[-1].question == "补充研究中国厂商"


def test_dynamic_question_append_limit_is_enforced() -> None:
    with pytest.raises(ValueError, match="at most three"):
        append_dynamic_questions(_plan(), [_question(f"q{index}") for index in range(6, 10)])


def test_evaluator_gap_builds_a_bounded_replan_question() -> None:
    additions = build_gap_driven_questions(
        _plan(),
        [
            (
                "q1",
                "研究国内工业视觉厂商",
                ("缺少第二个独立来源",),
                1,
            )
        ],
    )

    assert [item.id for item in additions] == ["q6"]
    assert "独立公开证据" in additions[0].question
    assert additions[0].priority == 1


def test_standard_budget_keeps_only_executable_high_priority_questions() -> None:
    plan = _plan().model_copy(
        update={
            "questions": [
                *(_plan().questions),
                _question("q6").model_copy(update={"priority": 3}),
                _question("q7").model_copy(update={"priority": 1}),
                _question("q8").model_copy(update={"priority": 2}),
            ]
        }
    )

    fitted = fit_plan_to_budget(plan, max_iterations=15)

    assert len(fitted.questions) == 6
    assert "q7" in {question.id for question in fitted.questions}
    assert "q6" not in {question.id for question in fitted.questions}


def test_gap_resolution_hints_target_independent_sources_without_new_question() -> None:
    question = _question("q6")

    hints = build_gap_resolution_hints(question, ("缺少第二个独立来源",))

    assert len(hints) == 3
    assert all(question.question in hint for hint in hints)
    assert any("independent source" in hint for hint in hints)


def _question(identifier: str) -> ResearchQuestion:
    return ResearchQuestion(
        id=identifier,
        question=f"动态问题 {identifier}",
        priority=2,
        rationale="由信息缺口触发",
        evidence_requirements=["公开来源"],
    )
