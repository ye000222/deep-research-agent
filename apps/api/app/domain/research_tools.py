"""Validated contracts for Web research tools and extracted evidence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=1000)
    url: str = Field(min_length=1, max_length=4000)
    snippet: str = Field(default="", max_length=4000)
    published_at: str | None = Field(default=None, max_length=100)
    rank: int = Field(ge=1)


class ReadPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_url: str = Field(min_length=1, max_length=4000)
    title: str = Field(min_length=1, max_length=1000)
    clean_text: str = Field(min_length=100)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fetched_at: datetime
    truncated: bool = False


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    CONTEXT = "context"


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=10, max_length=1200)
    exact_quote: str = Field(min_length=10, max_length=2000)
    dimension_key: str | None = Field(default=None, min_length=1, max_length=100)
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS
    relevance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EvidenceCandidate] = Field(default_factory=list, max_length=5)


class ScoredEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: EvidenceCandidate
    source_reliability: float = Field(ge=0.0, le=1.0)
    evidence_score: float = Field(ge=0.0, le=1.0)
    accepted: bool
    rejection_reason: str | None = None


class EvidenceView(BaseModel):
    evidence_id: UUID
    claim_id: UUID | None = None
    snapshot_id: UUID | None = None
    chunk_id: UUID | None = None
    question_id: str
    claim: str
    exact_quote: str
    relation: EvidenceRelation
    source_title: str
    source_url: str
    source_domain: str
    source_reliability: float
    evidence_score: float
    accepted: bool
    rejection_reason: str | None = None
