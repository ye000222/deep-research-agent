"""Planner retry policy tests."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest
from app.domain.planning import ResearchPlan
from app.domain.providers import TokenUsage
from app.llm.adapters import ModelGatewayError
from app.services import research_graph


class FakePlanner:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def generate(self, run_id: object) -> tuple[ResearchPlan, TokenUsage]:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return cast(tuple[ResearchPlan, TokenUsage], outcome)


class FakeRepository:
    def __init__(self) -> None:
        self.retries: list[dict[str, object]] = []

    async def record_model_retry(self, *args: object, **kwargs: object) -> bool:
        self.retries.append(kwargs)
        return True


@pytest.mark.asyncio
async def test_planner_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    result = cast(tuple[ResearchPlan, TokenUsage], (object(), object()))
    planner = FakePlanner(
        [
            ModelGatewayError("MODEL_NETWORK_ERROR", retryable=True),
            ModelGatewayError("MODEL_TIMEOUT", retryable=True),
            result,
        ]
    )
    repository = FakeRepository()
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(research_graph.asyncio, "sleep", fake_sleep)

    service = research_graph.ResearchGraphService(  # type: ignore[arg-type]
        repository,
        object(),
        planner,
        object(),
        object(),
    )
    actual = await service._generate_plan_with_retry(
        uuid4(),
        worker_task_id="worker-1",
    )

    assert actual == result
    assert planner.calls == 3
    assert delays == [3.0, 10.0]
    assert [item["failed_attempt"] for item in repository.retries] == [1, 2]


@pytest.mark.asyncio
async def test_planner_does_not_retry_permanent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = FakePlanner(
        [ModelGatewayError("MODEL_AUTHENTICATION_FAILED", retryable=False)]
    )
    repository = FakeRepository()

    async def fail_if_called(delay: float) -> None:
        raise AssertionError(f"unexpected retry delay: {delay}")

    monkeypatch.setattr(research_graph.asyncio, "sleep", fail_if_called)

    with pytest.raises(ModelGatewayError, match="MODEL_AUTHENTICATION_FAILED"):
        service = research_graph.ResearchGraphService(  # type: ignore[arg-type]
            repository,
            object(),
            planner,
            object(),
            object(),
        )
        await service._generate_plan_with_retry(
            uuid4(),
            worker_task_id="worker-1",
        )

    assert planner.calls == 1
    assert repository.retries == []
