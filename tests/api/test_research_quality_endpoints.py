from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.api.dependencies import get_client_session, get_research_run_service
from app.core.config import Settings
from app.domain.evaluation import EvaluationScope, EvaluationSnapshot, EvaluationVerdict
from app.domain.reports import VerificationView
from app.domain.research_runs import AgentEventView
from app.domain.state import (
    ActionType,
    CoverageDimensionSnapshot,
    GapType,
    NextAction,
    ResearchGap,
    ResearchState,
)
from app.infrastructure.db.reports import ReportNotFoundError
from app.main import create_app
from app.security.client_sessions import ClientSession
from fastapi.testclient import TestClient


class FakeResearchRunService:
    def __init__(self) -> None:
        self.run_id = uuid4()
        self.question_id = uuid4()
        self.gap = ResearchGap(
            gap_id=uuid4(),
            question_id=self.question_id,
            dimension_key="market",
            gap_type=GapType.MISSING,
            description="缺少市场规模",
            acceptance_criteria="至少两个独立来源",
            severity=0.9,
        )
        self.action = NextAction(
            action_id=uuid4(),
            action_type=ActionType.SEARCH_WEB,
            target_gap_ids=(self.gap.gap_id,),
            tool_name="web_search",
            expected_output="市场规模一手数据",
            public_decision_summary="市场维度仍为空, 因此继续检索.",
        )
        self.state = ResearchState(
            run_id=self.run_id,
            state_version=4,
            gaps=(self.gap,),
            next_action=self.action,
            coverage_map=(
                CoverageDimensionSnapshot(
                    dimension_key="market",
                    question="市场规模是多少?",
                    priority=1,
                    coverage=0.0,
                    accepted_evidence=0,
                    independent_sources=0,
                    missing_reasons=("缺少独立来源",),
                ),
            ),
        )
        self.evaluation = EvaluationSnapshot(
            evaluation_id=uuid4(),
            run_id=self.run_id,
            scope=EvaluationScope.GLOBAL,
            state_version=4,
            plan_version=1,
            coverage=0.4,
            evidence_sufficiency=0.3,
            source_quality=0.8,
            source_diversity=0.5,
            source_independence=0.5,
            cross_validation=0.2,
            freshness=0.9,
            conflict_resolution=1.0,
            citation_completeness=0.0,
            citation_support=0.0,
            missing_dimension_keys=("market",),
            verdict=EvaluationVerdict.CONTINUE,
        )
        self.verification = VerificationView(
            run_id=self.run_id,
            report_id=uuid4(),
            report_status="verified",
            verified=True,
            citation_count=3,
            analysis_artifact_citation_count=1,
            verification_result={
                "verified": True,
                "citation_completeness": 1.0,
                "semantic_support_rate": 1.0,
            },
        )
        self.report_ready = True

    async def get_state(self, owner_hash: str, run_id: UUID) -> ResearchState:
        assert owner_hash == "owner"
        assert run_id == self.run_id
        return self.state

    async def list_events(
        self,
        owner_hash: str,
        run_id: UUID,
        *,
        after_seq: int,
    ) -> list[AgentEventView]:
        assert after_seq == 0
        return [
            AgentEventView(
                global_id=1,
                run_id=run_id,
                seq=1,
                schema_version=1,
                timestamp=datetime.now(UTC),
                phase="researching",
                event_type="action.selected",
                public_summary="选择 Web Search。",
                refs={"action_id": str(self.action.action_id)},
                metrics={"expected_information_gain": 0.5},
            ),
            AgentEventView(
                global_id=2,
                run_id=run_id,
                seq=2,
                schema_version=1,
                timestamp=datetime.now(UTC),
                phase="researching",
                event_type="context.built",
                public_summary="上下文已构建。",
                refs={},
                metrics=None,
            ),
        ]

    async def list_evaluations(
        self,
        owner_hash: str,
        run_id: UUID,
    ) -> list[EvaluationSnapshot]:
        assert owner_hash == "owner"
        assert run_id == self.run_id
        return [self.evaluation]

    async def get_verification(self, owner_hash: str, run_id: UUID) -> VerificationView:
        assert owner_hash == "owner"
        assert run_id == self.run_id
        if not self.report_ready:
            raise ReportNotFoundError
        return self.verification


def make_client(fake: FakeResearchRunService) -> TestClient:
    application = create_app(
        Settings(
            app_env="test",
            external_probes_enabled=False,
            langgraph_strict_msgpack=False,
        )
    )
    application.dependency_overrides[get_research_run_service] = lambda: fake
    application.dependency_overrides[get_client_session] = lambda: ClientSession(
        client_id=uuid4(),
        owner_hash="owner",
        is_new=False,
    )
    return TestClient(application)


def test_gap_api_returns_unknown_registry_and_open_count() -> None:
    fake = FakeResearchRunService()
    with make_client(fake) as client:
        response = client.get(f"/api/v1/research-runs/{fake.run_id}/gaps")

    assert response.status_code == 200
    assert response.json()["open_count"] == 1
    assert response.json()["gaps"][0]["dimension_key"] == "market"


def test_action_api_filters_public_action_and_tool_events() -> None:
    fake = FakeResearchRunService()
    with make_client(fake) as client:
        response = client.get(f"/api/v1/research-runs/{fake.run_id}/actions")

    assert response.status_code == 200
    body = response.json()
    assert body["next_action"]["action_type"] == "search_web"
    assert [event["event_type"] for event in body["events"]] == ["action.selected"]


def test_evaluation_api_returns_versioned_quality_snapshot() -> None:
    fake = FakeResearchRunService()
    with make_client(fake) as client:
        response = client.get(f"/api/v1/research-runs/{fake.run_id}/evaluations")

    assert response.status_code == 200
    body = response.json()[0]
    assert body["scope"] == "global"
    assert body["missing_dimension_keys"] == ["market"]
    assert body["verdict"] == "continue"


def test_verification_api_returns_persisted_report_gate() -> None:
    fake = FakeResearchRunService()
    with make_client(fake) as client:
        response = client.get(f"/api/v1/research-runs/{fake.run_id}/verification")

    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert body["citation_count"] == 3
    assert body["analysis_artifact_citation_count"] == 1
    assert body["verification_result"]["semantic_support_rate"] == 1.0


def test_verification_api_distinguishes_report_not_ready() -> None:
    fake = FakeResearchRunService()
    fake.report_ready = False
    with make_client(fake) as client:
        response = client.get(f"/api/v1/research-runs/{fake.run_id}/verification")

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "REPORT_NOT_READY"
