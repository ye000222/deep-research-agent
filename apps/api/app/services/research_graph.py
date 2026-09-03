"""LangGraph orchestration for Planner, Research, and Writer stage boundaries."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.domain.memory import MemoryItemView
from app.domain.planning import ResearchPlan
from app.domain.providers import TokenUsage
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.state_runtime import ResearchStateRuntimeRepository
from app.llm.adapters import ModelGatewayError
from app.memory.manager import ResearchMemoryManager
from app.services.planner import PlannerService
from app.services.report_writer import ReportWriterService
from app.services.research_loop import ResearchLoopService

_PLANNER_RETRY_DELAYS_SECONDS = (3.0, 10.0)


class ResearchGraphState(TypedDict):
    run_id: str
    research_state: dict[str, Any]
    outcome: str
    continue_research: bool
    iteration: int
    replan_required: bool


class ResearchGraphService:
    """Run coarse Agent stages as durable LangGraph nodes.

    The business database remains the fact source. Each node projects those facts through a
    validated StatePatch, while the official checkpointer persists graph super-step boundaries.
    """

    def __init__(
        self,
        runs: ResearchRunRepository,
        states: ResearchStateRuntimeRepository,
        planner: PlannerService,
        research_loop: ResearchLoopService,
        report_writer: ReportWriterService,
        memories: ResearchMemoryManager | None = None,
    ) -> None:
        self._runs = runs
        self._states = states
        self._planner = planner
        self._research_loop = research_loop
        self._report_writer = report_writer
        self._memories = memories

    async def execute(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> str:
        initial = await self._states.ensure_initialized(run_id)
        workflow = StateGraph(ResearchGraphState)

        async def planner_node(_graph_state: ResearchGraphState) -> ResearchGraphState:
            plan = await self._runs.get_plan_for_execution(run_id)
            if plan is None:
                memory_leads: tuple[MemoryItemView, ...] = ()
                if self._memories is not None:
                    memory_result = await self._memories.retrieve_for_run(run_id)
                    memory_leads = memory_result.items
                plan, usage = await self._generate_plan_with_retry(
                    run_id,
                    worker_task_id=worker_task_id,
                    memory_leads=memory_leads,
                )
                saved = await self._runs.save_generated_plan(
                    run_id,
                    worker_task_id=worker_task_id,
                    plan=plan,
                    usage=usage,
                )
                if not saved:
                    raise RuntimeError("planner lost the worker lease before persisting the plan")
            state = await self._states.synchronize(
                run_id,
                node_name="planner",
                worker_task_id=worker_task_id,
            )
            if self._memories is not None:
                await self._memories.capture_state(state, node_name="planner")
            return {
                "run_id": str(run_id),
                "research_state": state.model_dump(mode="json"),
                "outcome": "planned",
                "continue_research": True,
                "iteration": 0,
                "replan_required": False,
            }

        async def research_node(graph_state: ResearchGraphState) -> ResearchGraphState:
            iteration = int(graph_state.get("iteration", 0)) + 1
            result = await self._research_loop.run_one_iteration(
                run_id,
                worker_task_id=worker_task_id,
            )
            state = await self._states.synchronize(
                run_id,
                node_name="research_iteration",
                worker_task_id=worker_task_id,
            )
            if self._memories is not None:
                await self._memories.capture_state(state, node_name="research_iteration")
            return {
                "run_id": str(run_id),
                "research_state": state.model_dump(mode="json"),
                "outcome": result.outcome,
                "continue_research": result.continue_research,
                "iteration": iteration,
                "replan_required": result.decision == "replan",
            }

        async def replan_node(graph_state: ResearchGraphState) -> ResearchGraphState:
            plan = await self._runs.get_plan_for_execution(run_id)
            if plan is None:
                raise RuntimeError("REPLAN requires a persisted plan")
            state = await self._states.synchronize(
                run_id,
                node_name="replan",
                worker_task_id=worker_task_id,
            )
            gaps = [
                (
                    dimension.dimension_key,
                    dimension.missing_reasons,
                )
                for dimension in sorted(
                    state.coverage_map,
                    key=lambda item: (item.priority, item.coverage, item.dimension_key),
                )
                if dimension.coverage < 1.0
            ]
            if not gaps:
                raise RuntimeError("REPLAN was selected without an actionable coverage gap")
            saved = await self._runs.save_gap_resolution_plan(
                run_id,
                worker_task_id=worker_task_id,
                gaps=gaps,
            )
            if not saved:
                raise RuntimeError("REPLAN lost the worker lease or produced no gap task")
            state = await self._states.synchronize(
                run_id,
                node_name="replan",
                worker_task_id=worker_task_id,
            )
            return {
                "run_id": str(run_id),
                "research_state": state.model_dump(mode="json"),
                "outcome": f"replanned:gaps={len(gaps[:3])}",
                "continue_research": True,
                "iteration": int(graph_state.get("iteration", 0)),
                "replan_required": False,
            }

        async def writer_node(_graph_state: ResearchGraphState) -> ResearchGraphState:
            outcome = await self._report_writer.write(
                run_id,
                worker_task_id=worker_task_id,
            )
            state = await self._states.synchronize(
                run_id,
                node_name="report_writer",
                worker_task_id=None,
            )
            if self._memories is not None:
                await self._memories.capture_state(state, node_name="report_writer")
            return {
                "run_id": str(run_id),
                "research_state": state.model_dump(mode="json"),
                "outcome": outcome,
                "continue_research": False,
                "iteration": int(_graph_state.get("iteration", 0)),
                "replan_required": False,
            }

        def route_after_research(
            graph_state: ResearchGraphState,
        ) -> Literal["research_iteration", "replan", "report_writer"]:
            if graph_state.get("replan_required", False):
                return "replan"
            return (
                "research_iteration"
                if graph_state.get("continue_research", False)
                else "report_writer"
            )

        # LangGraph's callable overload currently rejects async functions returning
        # this total TypedDict under strict MyPy, although they are valid runtime nodes.
        workflow.add_node("planner", cast(Any, planner_node))
        workflow.add_node("research_iteration", cast(Any, research_node))
        workflow.add_node("replan", cast(Any, replan_node))
        workflow.add_node("report_writer", cast(Any, writer_node))
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "research_iteration")
        workflow.add_conditional_edges(
            "research_iteration",
            route_after_research,
            {
                "research_iteration": "research_iteration",
                "replan": "replan",
                "report_writer": "report_writer",
            },
        )
        workflow.add_edge("replan", "research_iteration")
        workflow.add_edge("report_writer", END)
        graph = workflow.compile(checkpointer=checkpointer, name="deep_research_v1")
        result = await graph.ainvoke(
            {
                "run_id": str(run_id),
                "research_state": initial.model_dump(mode="json"),
                "outcome": "initialized",
                "continue_research": True,
                "iteration": 0,
                "replan_required": False,
            },
            config={"configurable": {"thread_id": str(run_id)}},
            durability="sync",
        )
        return str(cast(dict[str, Any], result).get("outcome", "completed"))

    async def _generate_plan_with_retry(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        memory_leads: tuple[MemoryItemView, ...] = (),
    ) -> tuple[ResearchPlan, TokenUsage]:
        max_attempts = len(_PLANNER_RETRY_DELAYS_SECONDS) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                if memory_leads:
                    return await self._planner.generate(run_id, memory_leads=memory_leads)
                return await self._planner.generate(run_id)
            except ModelGatewayError as exc:
                if not exc.retryable or attempt >= max_attempts:
                    raise
                delay_seconds = _PLANNER_RETRY_DELAYS_SECONDS[attempt - 1]
                recorded = await self._runs.record_model_retry(
                    run_id,
                    worker_task_id=worker_task_id,
                    role="planner",
                    failed_attempt=attempt,
                    max_attempts=max_attempts,
                    error_code=exc.code,
                    detail_code=exc.detail_code,
                    delay_seconds=delay_seconds,
                )
                if not recorded:
                    raise RuntimeError(
                        "planner lost the worker lease while scheduling retry"
                    ) from exc
                await asyncio.sleep(delay_seconds)
        raise RuntimeError("planner retry loop exhausted without a result")
