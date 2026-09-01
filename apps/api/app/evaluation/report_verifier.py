"""Deterministic report citation and evidence support verifier."""

from __future__ import annotations

import re
from collections.abc import Iterable
from uuid import UUID

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


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


def verify_evidence_support(
    factual_lines: Iterable[str],
    citation_evidence: dict[int, tuple[str, str]],
    *,
    minimum_overlap: float = 0.12,
) -> dict[str, float | int | list[int]]:
    """Check that cited evidence shares substantive terms with each factual line.

    This is a deterministic semantic-support gate, not an LLM self-judgement. It
    deliberately uses claim and exact-quote text only; unsupported or unregistered
    citation markers are reported rather than silently accepted.
    """
    lines = [line for line in factual_lines if line.strip()]
    supported = 0
    unsupported: list[int] = []
    for line in lines:
        markers = [int(item) for item in _CITATION_PATTERN.findall(line)]
        body = _CITATION_PATTERN.sub("", line)
        body_tokens = set(_TOKEN_PATTERN.findall(body.lower()))
        evidence_tokens = set()
        for marker in markers:
            claim_quote = citation_evidence.get(marker)
            if claim_quote is not None:
                evidence_tokens.update(_TOKEN_PATTERN.findall(" ".join(claim_quote).lower()))
        overlap = len(body_tokens & evidence_tokens) / len(body_tokens) if body_tokens else 0.0
        if markers and evidence_tokens and overlap >= minimum_overlap:
            supported += 1
        else:
            unsupported.extend(markers)
    return {
        "semantic_support_rate": supported / len(lines) if lines else 0.0,
        "semantic_supported_lines": supported,
        "semantic_unsupported_citations": sorted(set(unsupported)),
    }
