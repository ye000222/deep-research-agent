"""Deterministic document, processing-window, evidence-span, and context mapping.

All character coordinates in this module refer to the canonical readable text
stored on a ReadPage, never raw HTML bytes or a model prompt string.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

DOCUMENT_CONTEXT_SCHEMA = "document-evidence-context.v1"
CHUNKING_VERSION = "canonical-char-chunks.v1"
_WHITESPACE = re.compile(r"\s+")


class DocumentCompleteness(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class SourceLocatorType(StrEnum):
    PDF_PAGE = "PDF_PAGE"
    HTML_SECTION = "HTML_SECTION"
    DOCUMENT_BLOCK = "DOCUMENT_BLOCK"
    UNKNOWN = "UNKNOWN"


class LocalizationStatus(StrEnum):
    LOCATED = "LOCATED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    UNMAPPED_INPUT = "UNMAPPED_INPUT"


class SourceLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    locator_type: SourceLocatorType
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    block_ordinal: int | None = Field(default=None, ge=0)


class DocumentSourceSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(ge=0)
    locator: SourceLocator
    ordinal: int = Field(ge=0)
    segment_type: str = "TEXT_BLOCK"


class ReadableDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = DOCUMENT_CONTEXT_SCHEMA
    document_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ref: str
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_text: str
    canonical_text_length: int = Field(ge=0)
    available_text_length: int = Field(ge=0)
    document_type: str
    completeness: DocumentCompleteness
    was_truncated: bool
    truncation_boundary: int | None = Field(default=None, ge=0)
    truncation_reason: str | None = None
    source_segments: tuple[DocumentSourceSegment, ...] = ()


class ProcessingChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = DOCUMENT_CONTEXT_SCHEMA
    chunk_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(ge=0)
    text: str
    ordinal: int = Field(ge=0)
    chunking_version: str = CHUNKING_VERSION


class RelevantSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(ge=0)
    processing_chunk_refs: tuple[str, ...]
    selection_reason: str


class CanonicalEvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = DOCUMENT_CONTEXT_SCHEMA
    span_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ref: str
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    exact_text: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_locator: SourceLocator


class ContextWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = DOCUMENT_CONTEXT_SCHEMA
    context_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_start: int = Field(ge=0)
    window_end: int = Field(ge=0)
    evidence_span_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str


@dataclass(frozen=True, slots=True)
class MappedTextWindow:
    """A prompt window proven to be an exact contiguous canonical-text slice."""

    text: str
    canonical_start: int
    canonical_end: int
    chunk_ref: str
    selection_reason: str = "selected_for_extraction"


@dataclass(frozen=True, slots=True)
class SpanLocalization:
    status: LocalizationStatus
    span: CanonicalEvidenceSpan | None
    occurrence_count: int


def build_readable_document(
    *,
    source_ref: str,
    snapshot_content_hash: str,
    canonical_text: str,
    available_text_length: int | None,
    document_type: str,
    was_truncated: bool,
    truncation_reason: str | None = None,
    source_segments: Iterable[DocumentSourceSegment] = (),
    completeness_known: bool = True,
) -> ReadableDocument:
    snapshot_ref = _sha256(snapshot_content_hash)
    document_ref = _identity_hash(
        "document", {"source_ref": source_ref, "snapshot_ref": snapshot_ref}
    )
    return ReadableDocument(
        document_ref=document_ref,
        source_ref=source_ref,
        snapshot_ref=snapshot_ref,
        canonical_text=canonical_text,
        canonical_text_length=len(canonical_text),
        available_text_length=max(available_text_length or 0, len(canonical_text)),
        document_type=document_type,
        completeness=(
            DocumentCompleteness.UNKNOWN
            if not completeness_known
            else DocumentCompleteness.PARTIAL
            if was_truncated
            else DocumentCompleteness.COMPLETE
        ),
        was_truncated=was_truncated,
        truncation_boundary=len(canonical_text) if was_truncated else None,
        truncation_reason=truncation_reason,
        source_segments=tuple(source_segments),
    )


def build_processing_chunks(
    document: ReadableDocument,
    *,
    chunk_chars: int,
    overlap_chars: int = 0,
    max_chunks: int = 64,
) -> tuple[ProcessingChunk, ...]:
    if chunk_chars <= 0 or max_chunks <= 0 or overlap_chars < 0:
        raise ValueError("chunk size and limit must be positive; overlap must be non-negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk size")
    stride = chunk_chars - overlap_chars
    chunks: list[ProcessingChunk] = []
    start = 0
    while start < document.canonical_text_length and len(chunks) < max_chunks:
        end = min(document.canonical_text_length, start + chunk_chars)
        text = document.canonical_text[start:end]
        chunk_ref = _identity_hash(
            "processing-chunk",
            {
                "document_ref": document.document_ref,
                "start": start,
                "end": end,
                "version": CHUNKING_VERSION,
            },
        )
        chunks.append(
            ProcessingChunk(
                chunk_ref=chunk_ref,
                document_ref=document.document_ref,
                canonical_start=start,
                canonical_end=end,
                text=text,
                ordinal=len(chunks),
            )
        )
        if end == document.canonical_text_length:
            break
        start += stride
    return tuple(chunks)


def planned_processing_chunk_count(
    text_length: int, *, chunk_chars: int, overlap_chars: int = 0
) -> int:
    if text_length < 0 or chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("invalid processing chunk dimensions")
    if text_length == 0:
        return 0
    stride = chunk_chars - overlap_chars
    return max(1, (max(0, text_length - chunk_chars) + stride - 1) // stride + 1)


def processing_chunk_for_range(
    document: ReadableDocument,
    *,
    start: int,
    end: int,
    ordinal: int,
    chunking_version: str = CHUNKING_VERSION,
) -> ProcessingChunk:
    if start < 0 or end < start or end > document.canonical_text_length:
        raise ValueError("processing chunk range is outside canonical document")
    chunk_ref = _identity_hash(
        "processing-chunk",
        {
            "document_ref": document.document_ref,
            "start": start,
            "end": end,
            "version": chunking_version,
        },
    )
    return ProcessingChunk(
        chunk_ref=chunk_ref,
        document_ref=document.document_ref,
        canonical_start=start,
        canonical_end=end,
        text=document.canonical_text[start:end],
        ordinal=ordinal,
        chunking_version=chunking_version,
    )


def mapped_window(
    chunk: ProcessingChunk, *, selection_reason: str = "selected_for_extraction"
) -> MappedTextWindow:
    return MappedTextWindow(
        text=chunk.text,
        canonical_start=chunk.canonical_start,
        canonical_end=chunk.canonical_end,
        chunk_ref=chunk.chunk_ref,
        selection_reason=selection_reason,
    )


def locate_canonical_evidence_span(
    document: ReadableDocument,
    exact_quote: str,
    windows: Iterable[MappedTextWindow],
) -> SpanLocalization:
    occurrences: set[tuple[int, int]] = set()
    saw_unmapped = False
    for window in windows:
        if (
            window.canonical_start < 0
            or window.canonical_end > document.canonical_text_length
            or window.canonical_end - window.canonical_start != len(window.text)
            or document.canonical_text[window.canonical_start : window.canonical_end] != window.text
        ):
            saw_unmapped = True
            continue
        for start, end in _find_quote_occurrences(window.text, exact_quote):
            occurrences.add((window.canonical_start + start, window.canonical_start + end))
    if not occurrences:
        status = LocalizationStatus.UNMAPPED_INPUT if saw_unmapped else LocalizationStatus.NOT_FOUND
        return SpanLocalization(status=status, span=None, occurrence_count=0)
    if len(occurrences) != 1:
        return SpanLocalization(
            status=LocalizationStatus.AMBIGUOUS,
            span=None,
            occurrence_count=len(occurrences),
        )
    start, end = next(iter(occurrences))
    text = document.canonical_text[start:end]
    locator = _locator_for_range(document.source_segments, start, end)
    span = CanonicalEvidenceSpan(
        span_ref=_identity_hash(
            "canonical-evidence-span",
            {
                "document_ref": document.document_ref,
                "start": start,
                "end": end,
                "content_sha256": _sha256(text),
            },
        ),
        document_ref=document.document_ref,
        source_ref=document.source_ref,
        snapshot_ref=document.snapshot_ref,
        char_start=start,
        char_end=end,
        exact_text=text,
        content_sha256=_sha256(text),
        source_locator=locator,
    )
    return SpanLocalization(
        status=LocalizationStatus.LOCATED,
        span=span,
        occurrence_count=1,
    )


def build_context_window(
    document: ReadableDocument,
    span: CanonicalEvidenceSpan,
    *,
    context_chars: int = 1_800,
) -> ContextWindow:
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")
    start = max(0, span.char_start - context_chars)
    end = min(document.canonical_text_length, span.char_end + context_chars)
    text = document.canonical_text[start:end]
    return ContextWindow(
        context_ref=_identity_hash(
            "context-window",
            {"document_ref": document.document_ref, "start": start, "end": end},
        ),
        document_ref=document.document_ref,
        window_start=start,
        window_end=end,
        evidence_span_ref=span.span_ref,
        text=text,
    )


def _locator_for_range(
    segments: tuple[DocumentSourceSegment, ...], start: int, end: int
) -> SourceLocator:
    matched = tuple(
        segment
        for segment in segments
        if segment.canonical_start < end and segment.canonical_end > start
    )
    if not matched:
        return SourceLocator(locator_type=SourceLocatorType.UNKNOWN)
    if all(segment.locator.locator_type is SourceLocatorType.PDF_PAGE for segment in matched):
        pages = [
            page
            for segment in matched
            for page in (segment.locator.page_start, segment.locator.page_end)
            if page is not None
        ]
        return SourceLocator(
            locator_type=SourceLocatorType.PDF_PAGE,
            page_start=min(pages),
            page_end=max(pages),
        )
    if len(matched) == 1:
        return matched[0].locator
    return SourceLocator(locator_type=SourceLocatorType.UNKNOWN)


def _find_quote_occurrences(text: str, quote: str) -> tuple[tuple[int, int], ...]:
    if not quote:
        return ()
    direct: list[tuple[int, int]] = []
    start = 0
    while True:
        found = text.find(quote, start)
        if found < 0:
            break
        direct.append((found, found + len(quote)))
        start = found + 1
    if direct:
        return tuple(direct)
    normalized_text, positions = _normalize_with_positions(text)
    normalized_quote = _WHITESPACE.sub(" ", quote).strip()
    if not normalized_quote:
        return ()
    normalized_matches: list[tuple[int, int]] = []
    start = 0
    while True:
        found = normalized_text.find(normalized_quote, start)
        if found < 0:
            break
        final = found + len(normalized_quote) - 1
        if final < len(positions):
            normalized_matches.append((positions[found], positions[final] + 1))
        start = found + 1
    return tuple(normalized_matches)


def _normalize_with_positions(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    in_whitespace = False
    for index, character in enumerate(text):
        if character.isspace():
            if normalized and not in_whitespace:
                normalized.append(" ")
                positions.append(index)
            in_whitespace = True
        else:
            normalized.append(character)
            positions.append(index)
            in_whitespace = False
    if normalized and normalized[-1] == " ":
        normalized.pop()
        positions.pop()
    return "".join(normalized), positions


def _identity_hash(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"kind": kind, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
