"""Report, section, and citation contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ReportCitationView(BaseModel):
    citation_number: int
    evidence_id: UUID
    claim_id: UUID
    snapshot_id: UUID
    chunk_id: UUID
    question_id: str
    claim: str
    exact_quote: str
    source_title: str
    source_url: str
    source_domain: str
    source_content_hash: str
    snapshot_content_hash: str
    chunk_char_start: int
    chunk_char_end: int
    accessed_at: datetime
    analysis_artifact_id: UUID | None = None


class ReportSectionView(BaseModel):
    outline_order: int
    section_key: str
    title: str
    draft_markdown: str
    status: str
    verification_result: dict[str, Any]


class ReportView(BaseModel):
    report_id: UUID
    run_id: UUID
    version: int
    title: str
    final_markdown: str
    limitations: list[str]
    verification_result: dict[str, Any]
    status: str
    created_at: datetime
    sections: list[ReportSectionView]
    citations: list[ReportCitationView]
