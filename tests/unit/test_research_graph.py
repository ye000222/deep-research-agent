from uuid import UUID, uuid4

import pytest
from app.agent.reducers.state_reducer import apply_patch
from app.domain.identifiers import uuid7
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.state import (
    CoverageDimensionSnapshot,
    ResearchPhase,
    ResearchState,
    RunStatus,
    StatePatch,
)
from app.services.research_graph import ResearchGraphService
from app.services.research_loop import ResearchIterationResult
from langgraph.checkpoint.memory import InMemorySaver


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Research a traceable industrial inspection market question.",
        scope_summary="Technology, vendors, products, and future trends.",
        questions=[
            ResearchQuestion(
                id=f"q{index}",
                question=f"What verifiable evidence answers research dimension {index}?",
                priority=1 if index < 3 else 2,
                rationale="The dimension is required by the research objective.",
                evidence_requirements=["At least one exact public quote."],
                search_hints=[f"industrial inspection dimension {index}"],
            )
            for index in range(1, 6)
        ],
        completion_criteria=["Every question has evidence.", "All citations resolve."],
    )


class FakeRuns:
    def __init__(self) -> None:
        self.plan: ResearchPlan | None = None
        self.saved = 0
        self.replanned = 0
        self.gap_tasks: list[tuple[str, tuple[str, ...]]] = []

    async def get_plan_for_execution(self, run_id: UUID) -> ResearchPlan | None:
        return self.plan

    async def save_generated_plan(self, run_id: UUID, **kwargs: object) -> bool:
        self.plan = kwargs["plan"]  # type: ignore[assignment]
        self.saved += 1
        return True

    async def record_model_retry(self, *args: object, **kwargs: object) -> bool:
        return True

    async def save_replanned_questions(self, run_id: UUID, **kwargs: object) -> bool:
        assert self.plan is not None
        additions = kwargs["additions"]
        self.plan = self.plan.model_copy(
            update={"questions": [*self.plan.questions, *additions]}  # type: ignore[misc]
        )
        self.replanned += 1
        return True

    async def save_gap_resolution_plan(self, run_id: UUID, **kwargs: object) -> bool:
        self.gap_tasks = kwargs["gaps"]  # type: ignore[assignment]
        self.replanned += 1
        return True


class FakeStates:
    def __init__(self, run_id: UUID) -> None:
        self.state = ResearchState(run_id=run_id)
        self.nodes: list[str] = []

    async def ensure_initialized(self, run_id: UUID) -> ResearchState:
        return self.state

    async def synchronize(
        self,
        run_id: UUID,
        *,
        node_name: str,
        worker_task_id: str | None,
    ) -> ResearchState:
        self.nodes.append(node_name)
        if node_name in {"planner", "research_iteration", "replan"}:
            phase, status = ResearchPhase.RESEARCH, RunStatus.RESEARCHING
        else:
            phase, status = ResearchPhase.FINALIZE, RunStatus.COMPLETED
        self.state = apply_patch(
            self.state,
            StatePatch(
                patch_id=uuid7(),
                base_version=self.state.state_version,
                phase=phase,
                status=status,
                coverage_map=(
                    (
                        CoverageDimensionSnapshot(
                            dimension_key="q1",
                            question="Which independent source closes the market gap?",
                            priority=1,
                            coverage=0.5,
                            accepted_evidence=1,
                            independent_sources=1,
                            missing_reasons=("A second independent source is required.",),
                        ),
                    )
                    if node_name == "replan"
                    else self.state.coverage_map
                ),
            ),
        )
        return self.state


class FakePlanner:
    async def generate(self, run_id: UUID) -> tuple[ResearchPlan, TokenUsage]:
        return _plan(), TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            accuracy=UsageAccuracy.EXACT,
        )


class FakeResearchLoop:
    def __init__(self) -> None:
        self.calls = 0

    async def run_one_iteration(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> ResearchIterationResult:
        self.calls += 1
        should_continue = self.calls == 1
        return ResearchIterationResult(
            outcome=(
                "research_stopped:continue_plan"
                if should_continue
                else "research_stopped:ready_to_write"
            ),
            continue_research=should_continue,
            decision="continue_plan" if should_continue else "ready_to_write",
            pages_read=2,
            accepted_evidence=1,
            coverage=0.5 if should_continue else 0.9,
            information_gain=0.4 if should_continue else 0.05,
            low_information_gain_streak=0 if should_continue else 1,
        )


class FakeReportWriter:
    async def write(self, run_id: UUID, *, worker_task_id: str) -> str:
        return "report_completed:completed:citations=5"


class FakeReplanResearchLoop(FakeResearchLoop):
    async def run_one_iteration(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> ResearchIterationResult:
        self.calls += 1
        replan = self.calls == 1
        return ResearchIterationResult(
            outcome="research_stopped:replan" if replan else "research_stopped:ready_to_write",
            continue_research=replan,
            decision="replan" if replan else "ready_to_write",
            pages_read=1,
            accepted_evidence=1,
            coverage=0.5 if replan else 0.9,
            information_gain=0.2,
            low_information_gain_streak=0,
        )


@pytest.mark.asyncio
async def test_graph_checkpoints_all_stage_boundaries_and_applies_state_patches() -> None:
    run_id = uuid4()
    runs = FakeRuns()
    states = FakeStates(run_id)
    saver = InMemorySaver()
    research_loop = FakeResearchLoop()
    service = ResearchGraphService(  # type: ignore[arg-type]
        runs,
        states,
        FakePlanner(),
        research_loop,
        FakeReportWriter(),
    )

    outcome = await service.execute(
        run_id,
        worker_task_id="worker-1",
        checkpointer=saver,
    )

    checkpoint = await saver.aget_tuple({"configurable": {"thread_id": str(run_id)}})
    assert outcome == "report_completed:completed:citations=5"
    assert runs.saved == 1
    assert states.nodes == [
        "planner",
        "research_iteration",
        "research_iteration",
        "report_writer",
    ]
    assert research_loop.calls == 2
    assert states.state.state_version == 4
    assert states.state.status is RunStatus.COMPLETED
    assert checkpoint is not None


@pytest.mark.asyncio
async def test_graph_routes_evaluator_replan_through_persisted_plan_revision() -> None:
    run_id = uuid4()
    runs = FakeRuns()
    states = FakeStates(run_id)
    research_loop = FakeReplanResearchLoop()
    service = ResearchGraphService(  # type: ignore[arg-type]
        runs,
        states,
        FakePlanner(),
        research_loop,
        FakeReportWriter(),
    )

    outcome = await service.execute(
        run_id,
        worker_task_id="worker-1",
        checkpointer=InMemorySaver(),
    )

    assert outcome == "report_completed:completed:citations=5"
    assert runs.replanned == 1
    assert runs.plan is not None
    assert len(runs.plan.questions) == 5
    assert runs.gap_tasks == [
        ("q1", ("A second independent source is required.",)),
    ]
    assert states.nodes == [
        "planner",
        "research_iteration",
        "replan",
        "replan",
        "research_iteration",
        "report_writer",
    ]
