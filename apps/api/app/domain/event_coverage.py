"""Conservative coverage of explicitly requested event nodes, never new facts.

Accepted rows remain unchanged. This projection only decides which requested
event their literal support can cover. Unrecognized paraphrases remain gaps.
"""

from __future__ import annotations

import re
from typing import Any

from app.domain.source_policy import source_owner_key

EVENTS = {
    "START_OF_CONSTRUCTION": (
        "开始建设",
        r"开始建设|启动建设|开工|construction (?:began|started|start)|began construction",
    ),
    "COMPLETION": (
        "建成",
        r"建成|竣工|完成建设|construction (?:was )?completed|completion|system was completed",
    ),
    "OFFICIAL_SERVICE_OPENING": (
        "正式开通服务",
        r"正式开通|开通服务|inaugurat\w*|official(?:ly)? (?:service )?open\w*"
        r"|service (?:was )?opened",
    ),
    "OPERATIONAL_ENTRY": (
        "投入运营",
        r"投入运营|投入运行|开始运营|entered (?:commercial )?operation"
        r"|became operational|operational entry",
    ),
}
MISSING_EVENT_TEXT = "当前检索未获得足以确认该节点的独立原始来源证据。"
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:年)?(?!\d)")
_DATE = re.compile(
    r"(?:19|20)\d{2}(?:年\d{1,2}月(?:\d{1,2}日)?|[-/]\d{1,2}(?:[-/]\d{1,2})?)?|\d{1,2}月\d{1,2}日"
)
_NEGATED = re.compile(
    r"未|并非|不代表|不能|不等于|计划|拟于|预计|将于|\bnot\b|planned|will be", re.I
)


def event_types(text: str) -> set[str]:
    return {key for key, (_, pattern) in EVENTS.items() if re.search(pattern, text, re.I)}


def _subject_anchor(goal: str) -> str:
    return re.split(r"何时|请|[\uFF1F?]", goal, maxsplit=1)[0][:160].strip() or goal[:160]


def temporal_support(text: str) -> bool:
    """A historical date is not complete without a supported year in its text."""
    has_partial_date = bool(
        re.search(
            r"\d{1,2}月\d{1,2}日|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d",
            text,
            re.I,
        )
    )
    return not has_partial_date or bool(_YEAR.search(text))


def _value(record: Any, key: str, default: Any = "") -> Any:
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _event_support(quote: str, event: str) -> tuple[bool, bool, set[str]]:
    pattern = EVENTS[event][1]
    dates: set[str] = set()
    supported = complete = False
    for clause in re.split(r"[。\uFF1B;\n]", quote):
        for match in re.finditer(pattern, clause, re.I):
            if _NEGATED.search(clause[max(0, match.start() - 16) : match.start()]):
                continue
            supported = True
            found = list(_DATE.finditer(clause))
            if found and _YEAR.search(clause):
                complete = True
                preceding = [d for d in found if d.start() <= match.start()]
                date = preceding[-1] if preceding else found[0]
                dates.add(date.group())
    return supported, complete, dates


def project_event_coverage(goal: str, evidence: Any, *, minimum_sources: int = 2) -> dict[str, Any]:
    targets = []
    for event in EVENTS:
        if event not in event_types(goal):
            continue
        ids, complete_ids, owners, dates, conflicts = set(), set(), set(), set(), set()
        for card in evidence:
            if not _value(card, "accepted", True):
                continue
            supported, complete, found_dates = _event_support(
                str(_value(card, "exact_quote")), event
            )
            if not supported:
                continue
            identifier = str(_value(card, "evidence_id", _value(card, "id")))
            if _value(card, "relation", "supports") in {"contradicts", "refutes"}:
                conflicts.add(identifier)
                continue
            ids.add(identifier)
            if complete:
                complete_ids.add(identifier)
                owner = _value(card, "source_owner_key") or source_owner_key(
                    str(_value(card, "source_url"))
                )
                if owner:
                    owners.add(owner)
                dates.update(found_dates)
        # Only incompatible full calendar dates establish a temporal conflict;
        # a year-only statement can coexist with a more specific date.
        calendar_dates = {
            tuple(map(int, re.findall(r"\d+", d))) for d in dates if len(re.findall(r"\d+", d)) == 3
        }
        status = (
            "CONFLICTING"
            if conflicts or len(calendar_dates) > 1
            else (
                "SUPPORTED"
                if complete_ids and len(owners) >= minimum_sources
                else "PARTIALLY_SUPPORTED"
                if ids
                else "UNRESOLVED"
            )
        )
        label = EVENTS[event][0]
        targets.append(
            {
                "target_id": event,
                "semantic_event_type": event,
                "label": label,
                "required": True,
                "critical": True,
                "coverage_status": status,
                "supporting_accepted_evidence_ids": sorted(ids),
                "temporally_complete_evidence_ids": sorted(complete_ids),
                "conflicting_evidence_ids": sorted(conflicts),
                "source_diversity": len(owners),
                "required_sources": minimum_sources,
                "temporal_support": "COMPLETE"
                if complete_ids
                else "PARTIAL"
                if ids
                else "UNRESOLVED",
                "gap_description": (
                    f"缺少{_subject_anchor(goal)}{label}时间的独立原始来源及完整年份证据"
                ),
                "gap_query": f"{goal[:300]} 仅核实{label} 日期 原始公告 独立来源",
            }
        )
    return {
        "version": "event_coverage.v1",
        "targets": targets,
        "critical_unresolved_count": sum(t["coverage_status"] != "SUPPORTED" for t in targets),
        "complete": all(t["coverage_status"] == "SUPPORTED" for t in targets),
    }


def ensure_event_plan(plan: Any) -> Any:
    """Keep user-required event questions ahead of optional elaboration.

    The existing plan fitter and question budgets still enforce their limits.
    A missing event is a question, not an invented answer or extra API call.
    """
    from app.domain.planning import ResearchQuestion

    requested = event_types(plan.goal)
    # Never allocate extra initial execution slots. If the model's bounded
    # outline is shorter than the required event set, missing events remain
    # explicit canonical gaps instead of silently changing question budgets.
    if len(requested) < 2 or len(plan.questions) < len(requested):
        return plan
    available = list(plan.questions)
    required = []
    next_id = max(int(q.id[1:]) for q in available) + 1
    anchor = _subject_anchor(plan.goal)
    for event, (label, _) in EVENTS.items():
        if event not in requested:
            continue
        candidates = [
            q
            for q in available
            if event in event_types(" ".join(q.evidence_requirements))
            or event in event_types(q.question)
        ]
        if candidates:
            chosen = min(candidates, key=lambda q: len(event_types(q.question)))
            available.remove(chosen)
            chosen = chosen.model_copy(update={"priority": 1})
        else:
            chosen = ResearchQuestion(
                id=f"q{next_id}",
                question=f"{anchor}的{label}时间是什么? 请提供原始来源和完整年份。",
                priority=1,
                rationale="保留用户明确要求且不能被其他事件替代的时间节点。",
                evidence_requirements=[f"明确{label}的原始来源及含年份的时间"],
                search_hints=[f"{anchor[:40]} {label} 日期 原始公告"],
            )
            next_id += 1
        required.append(chosen)
    original_order = {q.id: i for i, q in enumerate(plan.questions)}
    required.sort(key=lambda q: original_order.get(q.id, len(original_order)))
    capacity = min(12, max(len(plan.questions), len(required)))
    return plan.model_copy(update={"questions": (required + available)[:capacity]})


def reconcile_coverage_map(coverage_map: list[dict[str, Any]], projection: dict[str, Any]) -> None:
    """Add independent event dimensions to the existing GapRequirement input."""
    for target in projection["targets"]:
        if not coverage_map:
            break
        event = target["target_id"]
        dimension = next(
            (d for d in coverage_map if event in event_types(d["question"])), coverage_map[0]
        )
        statuses = dimension["requirement_statuses"]
        score = (
            1.0
            if target["coverage_status"] == "SUPPORTED"
            else (0.5 if target["coverage_status"] == "PARTIALLY_SUPPORTED" else 0.0)
        )
        # A generic existing dimension must not claim completion from opening evidence.
        for dimension_entry in coverage_map:
            for status in dimension_entry["requirement_statuses"]:
                if event in event_types(str(status["criterion"])):
                    status["coverage"] = min(float(status["coverage"]), score)
        statuses.append(
            {
                "dimension_key": f"{dimension['dimension_key']}:event:{event}",
                "criterion": target["gap_description"],
                "coverage": score,
                "required_sources": target["required_sources"],
                "accepted_evidence": len(target["supporting_accepted_evidence_ids"]),
                "independent_sources": target["source_diversity"],
                "max_source_reliability": 0.0,
            }
        )
    for dimension in coverage_map:
        statuses = dimension["requirement_statuses"]
        dimension["coverage"] = sum(s["coverage"] for s in statuses) / max(1, len(statuses))
        dimension["missing_reasons"] = [
            str(s["criterion"]) + ": 缺少对应事件的完整时间或独立证据"
            for s in statuses
            if s["coverage"] < 1.0
        ]
