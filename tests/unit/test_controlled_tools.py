from uuid import uuid4

import pytest
from app.domain.controlled_tools import (
    AnalysisDataPoint,
    AnalysisOperation,
    AnalyzeDataInput,
    ControlledToolName,
    ToolDecisionRequest,
)
from app.tools.analyze_data import _analyze
from app.tools.policy import ToolPolicyGuard


def _decision(**kwargs: object) -> ToolDecisionRequest:
    return ToolDecisionRequest(
        action_id=uuid4(),
        tool_name=kwargs.pop("tool_name", ControlledToolName.WEB_SEARCH),
        target_gap_ids=(uuid4(),),
        duplicate_key="unique",
        **kwargs,
    )


def test_web_search_requires_evidence_preflight() -> None:
    result = ToolPolicyGuard().authorize(_decision())
    assert result.verdict.value == "fallback"
    assert result.reason_code == "EVIDENCE_SEARCH_REQUIRED"


def test_web_search_is_blocked_when_unread_source_exists() -> None:
    result = ToolPolicyGuard().authorize(_decision(evidence_checked=True, unread_candidate_count=1))
    assert result.verdict.value == "fallback"
    assert result.reason_code == "UNREAD_SOURCE_AVAILABLE"


def test_analysis_is_declarative_and_recomputable() -> None:
    evidence_id = uuid4()
    request = AnalyzeDataInput(
        run_id=uuid4(),
        action_id=uuid4(),
        target_gap_ids=(uuid4(),),
        question_id="q1",
        operation=AnalysisOperation.CAGR,
        data=(
            AnalysisDataPoint(
                evidence_id=evidence_id, label="2024", value=100, unit="USD", period="2024"
            ),
            AnalysisDataPoint(
                evidence_id=evidence_id, label="2026", value=121, unit="USD", period="2026"
            ),
        ),
        parameters={"periods": 2},
    )
    result, formula, warnings = _analyze(request)
    assert result["cagr"] == pytest.approx(0.1)
    assert formula.startswith("(last / first)")
    assert warnings == ()
