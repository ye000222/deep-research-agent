"""Auditable Research Memory contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


class MemoryItemView(BaseModel):
    memory_id: UUID
    origin_run_id: UUID
    memory_type: MemoryType
    content_summary: str
    source_ref_ids: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    status: MemoryStatus
    revalidation_required: bool = False
    access_count: int = 0
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class MemoryAccessView(BaseModel):
    access_id: UUID
    run_id: UUID
    query: str
    requested_types: tuple[MemoryType, ...]
    candidate_ids: tuple[UUID, ...]
    selected_ids: tuple[UUID, ...]
    result: str
    revalidation_required_count: int
    created_at: datetime


class MemoryRetrievalResult(BaseModel):
    access_id: UUID
    items: tuple[MemoryItemView, ...]
    result: str
