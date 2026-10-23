"""Deterministic Writer packing. Never truncate accepted claims or quotes."""

from __future__ import annotations

import json
from typing import Any

from app.context.manager import ContextBudgetInsufficientError, estimate_tokens
from app.domain.event_coverage import event_types, project_event_coverage


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pack_writer_context(
    context: Any,
    evidence: list[Any],
    *,
    instructions: str,
    schema: dict[str, Any],
    input_budget: int,
) -> dict[str, Any]:
    coverage = project_event_coverage(context.goal, context.evidence)
    critical = [
        {"target_id": t["target_id"], "status": t["coverage_status"], "gap": t["gap_description"]}
        for t in coverage["targets"]
        if t["coverage_status"] != "SUPPORTED"
    ]
    # Keep unresolved canonical requirements without copying risk-state trees,
    # per-claim projections, raw pages, memories, or event histories.
    gaps = context.quality_snapshot.get("gap_requirements", [])
    critical += [
        {"gap_id": str(g["gap_id"]), "criterion": g["criterion"]}
        for g in gaps
        if isinstance(g, dict)
        and g.get("closure_status") != "closed"
        and g.get("gap_id")
        and g.get("criterion")
    ]
    payload = {
        "goal": context.goal,
        "questions": [
            {"question_id": q.question_id, "question": q.question} for q in context.questions
        ],
        "coverage_targets": [
            {"target_id": t["target_id"], "label": t["label"], "status": t["coverage_status"]}
            for t in coverage["targets"]
        ],
        "critical_unresolved_gaps": critical,
        "stop_reason": context.stop_reason,
        "sources": {},
        "evidence_cards": [],
    }
    overhead = estimate_tokens(instructions) + estimate_tokens(compact_json(schema))

    def fits(value: dict[str, Any]) -> bool:
        return overhead + estimate_tokens(compact_json(value)) + 64 <= input_budget

    if not fits(payload):
        raise ContextBudgetInsufficientError("WRITER_REQUIRED_CONTEXT_EXCEEDS_INPUT_BUDGET")
    remaining = list(evidence)
    seen, covered, questions = set(), set(), set()
    while remaining:
        card = max(
            remaining,
            key=lambda c: (
                len((event_types(c.exact_quote) & event_types(context.goal)) - covered),
                c.question_id not in questions,
                c.evidence_score,
            ),
        )
        remaining.remove(card)
        identity = (card.claim, card.exact_quote, card.source_url)
        if identity in seen:
            continue
        seen.add(identity)
        sources = dict(payload["sources"])
        source_ref = next(
            (k for k, v in sources.items() if v["url"] == card.source_url), f"s{len(sources) + 1}"
        )
        sources[source_ref] = {
            "url": card.source_url,
            "title": card.source_title,
            "domain": card.source_domain,
        }
        packed = {
            "evidence_id": str(card.evidence_id),
            "question_id": card.question_id,
            "claim": card.claim,
            "exact_quote": card.exact_quote,
            "source_ref": source_ref,
        }
        if card.analysis_artifact_id is not None:
            packed["analysis"] = {
                "artifact_id": str(card.analysis_artifact_id),
                "operation": card.analysis_operation,
                "formula": card.analysis_formula,
                "result": card.analysis_result,
            }
        candidate = {
            **payload,
            "sources": sources,
            "evidence_cards": [*payload["evidence_cards"], packed],
        }
        if fits(candidate):
            payload = candidate
            covered.update(event_types(card.exact_quote))
            questions.add(card.question_id)
    if not payload["evidence_cards"]:
        raise ContextBudgetInsufficientError("REPORT_EVIDENCE_CARDS_PRUNED")
    return payload
