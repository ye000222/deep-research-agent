import json
from itertools import pairwise

from app.domain.document_evidence_context import (
    DocumentCompleteness,
    DocumentSourceSegment,
    LocalizationStatus,
    MappedTextWindow,
    SourceLocator,
    SourceLocatorType,
    build_context_window,
    build_processing_chunks,
    build_readable_document,
    locate_canonical_evidence_span,
    mapped_window,
    planned_processing_chunk_count,
    processing_chunk_for_range,
)
from app.domain.research_tools import EvidenceCandidate, ScoredEvidence
from app.services.evidence_extractor import _attach_canonical_spans, _processing_window_manifest


def _document(
    text: str,
    *,
    snapshot_hash: str = "a" * 64,
    source_segments: tuple[DocumentSourceSegment, ...] = (),
    truncated: bool = False,
    completeness_known: bool = True,
):
    return build_readable_document(
        source_ref="source-1",
        snapshot_content_hash=snapshot_hash,
        canonical_text=text,
        available_text_length=len(text),
        document_type="HTML",
        was_truncated=truncated,
        source_segments=source_segments,
        completeness_known=completeness_known,
    )


def test_readable_document_identity_is_deterministic_and_snapshot_bound() -> None:
    text = "A canonical readable document."
    first = _document(text)
    repeated = _document(text)
    changed_snapshot = _document(text, snapshot_hash="b" * 64)

    assert first.document_ref == repeated.document_ref
    assert first.document_ref != changed_snapshot.document_ref
    assert first.canonical_text == text
    assert first.canonical_text_length == len(text)


def test_legacy_metadata_is_unknown_not_falsely_complete() -> None:
    document = _document("Legacy readable text.", completeness_known=False)
    partial = _document("Truncated text.", truncated=True)

    assert document.completeness is DocumentCompleteness.UNKNOWN
    assert partial.completeness is DocumentCompleteness.PARTIAL
    assert partial.truncation_boundary == len(partial.canonical_text)


def test_processing_chunks_are_stable_slices_with_bounded_overlap() -> None:
    text = "0123456789" * 7
    document = _document(text)
    chunks = build_processing_chunks(document, chunk_chars=20, overlap_chars=4, max_chunks=20)
    repeated = build_processing_chunks(document, chunk_chars=20, overlap_chars=4, max_chunks=20)

    assert [chunk.chunk_ref for chunk in chunks] == [chunk.chunk_ref for chunk in repeated]
    assert all(chunk.text == text[chunk.canonical_start : chunk.canonical_end] for chunk in chunks)
    assert all(left.canonical_end - right.canonical_start == 4 for left, right in pairwise(chunks))
    assert chunks[-1].canonical_end == len(text)
    assert planned_processing_chunk_count(len(text), chunk_chars=20, overlap_chars=4) == len(chunks)


def test_relevant_processing_chunk_preserves_canonical_offsets() -> None:
    text = "first paragraph\n\nselected factual quote\n\nlast paragraph"
    document = _document(text)
    start = text.index("selected factual quote")
    chunk = processing_chunk_for_range(
        document, start=start, end=start + len("selected factual quote"), ordinal=2
    )

    assert chunk.text == "selected factual quote"
    assert mapped_window(chunk).canonical_start == start
    assert chunk.text == document.canonical_text[chunk.canonical_start : chunk.canonical_end]


def test_unique_quote_maps_to_canonical_span_and_context_stays_separate() -> None:
    text = "Prior context. Exact quote. Following context."
    document = _document(text)
    start = text.index("Exact quote.")
    result = locate_canonical_evidence_span(
        document,
        "Exact quote.",
        (
            MappedTextWindow(
                text=text, canonical_start=0, canonical_end=len(text), chunk_ref="full"
            ),
        ),
    )

    assert result.status is LocalizationStatus.LOCATED
    assert result.span is not None
    assert result.span.char_start == start
    assert (
        document.canonical_text[result.span.char_start : result.span.char_end]
        == result.span.exact_text
    )
    context = build_context_window(document, result.span, context_chars=20)
    assert context.text == document.canonical_text[context.window_start : context.window_end]
    assert "Prior context." in context.text
    assert context.evidence_span_ref == result.span.span_ref
    assert result.span.exact_text == "Exact quote."


def test_same_quote_at_two_document_locations_is_ambiguous() -> None:
    text = "First: repeated quote. Middle. Second: repeated quote."
    document = _document(text)
    result = locate_canonical_evidence_span(
        document,
        "repeated quote.",
        (
            MappedTextWindow(
                text=text, canonical_start=0, canonical_end=len(text), chunk_ref="full"
            ),
        ),
    )

    assert result.status is LocalizationStatus.AMBIGUOUS
    assert result.span is None
    assert result.occurrence_count == 2


def test_quote_in_second_selected_span_maps_to_second_occurrence() -> None:
    text = "Repeated fact.\n\nUnrelated block.\n\nRepeated fact."
    document = _document(text)
    start = text.rindex("Repeated fact.")
    second_span = processing_chunk_for_range(
        document,
        start=start,
        end=start + len("Repeated fact."),
        ordinal=1,
    )
    result = locate_canonical_evidence_span(
        document,
        "Repeated fact.",
        (mapped_window(second_span),),
    )

    assert result.status is LocalizationStatus.LOCATED
    assert result.span is not None
    assert result.span.char_start == start


def test_overlapping_windows_collapse_same_span_but_not_distinct_occurrences() -> None:
    text = "x" * 25 + "The quote is exact." + "y" * 25
    document = _document(text)
    first = MappedTextWindow(text=text[:45], canonical_start=0, canonical_end=45, chunk_ref="a")
    second = MappedTextWindow(
        text=text[20:], canonical_start=20, canonical_end=len(text), chunk_ref="b"
    )
    result = locate_canonical_evidence_span(document, "The quote is exact.", (first, second))
    assert result.status is LocalizationStatus.LOCATED

    duplicated = "The quote is exact. and The quote is exact."
    duplicate_document = _document(duplicated)
    duplicate_result = locate_canonical_evidence_span(
        duplicate_document,
        "The quote is exact.",
        (
            MappedTextWindow(
                text=duplicated, canonical_start=0, canonical_end=len(duplicated), chunk_ref="c"
            ),
        ),
    )
    assert duplicate_result.status is LocalizationStatus.AMBIGUOUS


def test_unmapped_compressed_input_fails_closed() -> None:
    document = _document("Source quote is present here.")
    result = locate_canonical_evidence_span(
        document,
        "Source quote",
        (
            MappedTextWindow(
                text="Compressed source quote",
                canonical_start=0,
                canonical_end=24,
                chunk_ref="compressed",
            ),
        ),
    )

    assert result.status is LocalizationStatus.UNMAPPED_INPUT
    assert result.span is None


def test_pdf_locator_is_preserved_without_text_regex_guessing() -> None:
    text = "first page quote\n\nsecond page quote"
    first_end = len("first page quote")
    second_start = first_end + 2
    segments = (
        DocumentSourceSegment(
            canonical_start=0,
            canonical_end=first_end,
            locator=SourceLocator(
                locator_type=SourceLocatorType.PDF_PAGE,
                page_start=1,
                page_end=1,
            ),
            ordinal=0,
            segment_type="PDF_PAGE_TEXT",
        ),
        DocumentSourceSegment(
            canonical_start=second_start,
            canonical_end=len(text),
            locator=SourceLocator(
                locator_type=SourceLocatorType.PDF_PAGE,
                page_start=2,
                page_end=2,
            ),
            ordinal=1,
            segment_type="PDF_PAGE_TEXT",
        ),
    )
    document = _document(text, source_segments=segments)
    start = text.index("second page quote")
    result = locate_canonical_evidence_span(
        document,
        "second page quote",
        (
            MappedTextWindow(
                text=text, canonical_start=0, canonical_end=len(text), chunk_ref="full"
            ),
        ),
    )

    assert result.span is not None
    assert result.span.char_start == start
    assert result.span.source_locator.page_start == 2
    assert result.span.source_locator.page_end == 2


def test_extractor_attachment_uses_selected_mapping_and_records_selection_provenance() -> None:
    text = "First source quote.\n\nOther paragraph.\n\nSecond source quote."
    document = _document(text)
    start = text.rindex("Second source quote.")
    selected = processing_chunk_for_range(
        document,
        start=start,
        end=start + len("Second source quote."),
        ordinal=1,
    )
    candidate = EvidenceCandidate(
        claim="The report states a second source quote.",
        exact_quote="Second source quote.",
        relevance=0.9,
        confidence=0.9,
    )
    scored = ScoredEvidence(
        candidate=candidate,
        source_reliability=0.9,
        evidence_score=0.9,
        accepted=True,
    )

    attached, counts = _attach_canonical_spans(
        [scored], document=document, windows=(mapped_window(selected),)
    )
    manifest = json.loads(
        _processing_window_manifest((mapped_window(selected, selection_reason="query_mmr_rank"),))
    )

    assert attached[0].canonical_span is not None
    assert attached[0].canonical_span.char_start == start
    assert counts["canonical_span_localized_count"] == 1
    assert manifest[0]["canonical_start"] == start
    assert manifest[0]["selection_reason"] == "query_mmr_rank"
