"""Bounded bilingual retrieval hints, not fact verification or evidence acceptance."""

import re

_EVENTS = {
    "launch": r"发射|\blaunch(?:ed|es|ing)?\b",
    "opening": (
        r"开通|启用|inaugurat\w*|(?:began|started|opened|launched).{0,35}service"
        r"|service.{0,35}(?:began|opened|launched)"
    ),
    "completion": r"建成|竣工|完成建设|completed|completion",
    "operation": r"投入运营|投入运行|operational|began operating",
    "construction": r"开始建设|开工|construction began|construction start",
}
_ALIASES = {
    "satellite": r"卫星|\bsatellites?\b",
    "beidou": r"北斗|bei\s?dou|\bbds\b",
    "date": r"何时|日期|哪一天|\bwhen\b|\bdate\b",
    "final": r"最后|最终|\bfinal\b|\blast\b",
}
_PLAN = re.compile(r"计划|拟于|预计|将于|\bplanned\b|\bplans?\b|scheduled|will be launched", re.I)
_DATE = re.compile(
    r"\d{4}年|\d{1,2}月\d{1,2}日|\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
    re.I,
)


def events(text: str) -> set[str]:
    return {key for key, pattern in _EVENTS.items() if re.search(pattern, text, re.I)}


def semantic_terms(text: str) -> set[str]:
    result = set(re.findall(r"[a-z0-9][a-z0-9_.-]+", text.casefold()))
    for phrase in re.findall(r"[\u3400-\u9fff]+", text):
        result.update(phrase[i : i + 2] for i in range(max(1, len(phrase) - 1)))
    result.update("event:" + key for key in events(text))
    result.update(
        "anchor:" + key for key, pattern in _ALIASES.items() if re.search(pattern, text, re.I)
    )
    return result


def passage_score(text: str, question: str) -> tuple[float, str]:
    objective, observed = semantic_terms(question), semantic_terms(text)
    overlap = min(1.0, len(objective & observed) / max(1, min(12, len(objective))))
    requested, found = events(question), events(text)
    anchors = {t for t in objective if t.startswith("anchor:") and t != "anchor:date"}
    anchored = bool(anchors & observed) if anchors else overlap > 0.1
    if (
        "launch" in requested
        and _PLAN.search(text)
        and not re.search(
            r"成功发射|发射成功|successfully launched|has launched|launched on", text, re.I
        )
    ):
        return min(overlap, 0.25), "planned_only"
    if requested and anchored and requested & found:
        return min(1.0, 0.55 + overlap * 0.3 + 0.15 * bool(_DATE.search(text))), "direct"
    if requested and anchored and found and not requested & found:
        return min(0.3, overlap), "background_only"
    if requested and "anchor:date" in objective and not requested & found:
        return min(0.3, overlap), "background_only" if anchored else "off_topic"
    if not overlap:
        return 0.0, "off_topic"
    return overlap, "supporting" if overlap >= 0.3 else "background_only"
