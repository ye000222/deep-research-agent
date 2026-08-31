"""Research run and public agent-event contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    COMPLETED_WITH_LIMITATIONS = "completed_with_limitations"
    CREDENTIALS_REQUIRED = "credentials_required"


class RunPhase(StrEnum):
    INITIALIZING = "initializing"
    PLANNING = "planning"
    RESEARCHING = "researching"
    EVALUATING = "evaluating"
    WRITING = "writing"
    VERIFYING = "verifying"
    TERMINAL = "terminal"


TERMINAL_RUN_STATUSES = {
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.COMPLETED,
    RunStatus.COMPLETED_WITH_LIMITATIONS,
}


@dataclass(frozen=True, slots=True)
class ResearchRunView:
    run_id: UUID
    original_query: str
    normalized_goal: str
    status: RunStatus
    phase: RunPhase
    state_version: int
    plan_version: int
    next_event_seq: int
    credential_status: str
    saved_profile_id: UUID
    credential_version_id: UUID
    llm_config_snapshot: dict[str, Any]
    budget_snapshot: dict[str, Any]
    usage_snapshot: dict[str, Any]
    quality_snapshot: dict[str, Any]
    termination_reason: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class AgentEventView:
    global_id: int
    run_id: UUID
    seq: int
    schema_version: int
    timestamp: datetime
    phase: str
    event_type: str
    public_summary: str
    refs: dict[str, Any]
    metrics: dict[str, Any] | None
