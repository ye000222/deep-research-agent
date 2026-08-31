"""Deterministic report citation and evidence support verifier."""

from __future__ import annotations

import re
from collections.abc import Iterable
from uuid import UUID

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def verify_citation_integrity(
    factual_lines: Iterable[str],
    numeric_lines: Iterable[str],
    citation_numbers: Iterable[int],
    *,
    evidence_ids: Iterable[UUID] = (),
) -> dict[str, float | int | bool | list[int]]:
    """Verify citation markers against the persisted citation registry.

    This verifier is deliberately deterministic: it never asks an LLM to decide
    whether a marker exists or whether a citation number is registered.
    """
    lines = [line for line in factual_lines if line.strip()]
    numeric = [line for line in numeric_lines if line.strip()]
    registered = {int(number) for number in citation_numbers}
    unresolved: list[int] = []
    cited_count = 0
    for line in lines:
        markers = [int(item) for item in _CITATION_PATTERN.findall(line)]
        if markers and all(item in registered for item in markers):
            cited_count += 1
        else:
            unresolved.extend(item for item in markers if item not in registered)
    numeric_supported = sum(
        bool(_CITATION_PATTERN.search(line))
        and all(int(item) in registered for item in _CITATION_PATTERN.findall(line))
        for line in numeric
    )
    unique_evidence = len(set(evidence_ids))
    return {
        "verified": bool(lines) and cited_count == len(lines) and not unresolved,
        "factual_paragraphs": len(lines),
        "citation_completeness": cited_count / len(lines) if lines else 0.0,
        "numeric_citation_rate": numeric_supported / len(numeric) if numeric else 1.0,
        "citation_registry_size": len(registered),
        "unresolved_citations": sorted(set(unresolved)),
        "unique_evidence_refs": unique_evidence,
    }
