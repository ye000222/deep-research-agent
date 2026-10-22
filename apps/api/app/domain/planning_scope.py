"""Conservative deterministic scope checks, not a semantic truth classifier.

Explicit constraints and narrow requested fact types are enforceable offline.
Paraphrases without recognizable anchors remain a documented limitation.
Search hints/rationale never authorize an otherwise contradictory objective.
"""

import re

from app.domain.planning import ResearchPlan, ResearchQuestion

_FACTS = {
    "date": r"何时|哪[一天年]|日期|时间|when|\bdate\b",
    "performance": r"性能|定位精度|测速|授时|performance|positioning accuracy|timing precision",
    "roadmap": r"后续|升级计划|未来计划|roadmap|future plans?",
    "market": r"市场规模|市场份额|market size|market share",
    "mechanism": r"技术原理|工作原理|how .* works?|mechanism",
}
_STATES = {
    "construction": r"开始建设|启动建设|开工|initiates? construction|construction start",
    "completion": r"建成|竣工|完成建设|construction complet|\bcompleted\b",
    "opening": r"开通|启用|正式发布|inaugurat|launch.*service|service.*launch",
    "operation": r"投入运营|投入运行|商业运营|operational|operation start",
}
_CONTRAST = re.compile(r"区别|不同|区分|衔接|对比|比较|difference|distinguish|versus|compare", re.I)
_VERIFY = re.compile(r"交叉验证|交叉核验|相互印证|独立来源|原始公告|cross.check|corroborat", re.I)
_REFERENCE = re.compile(
    r"该(?:系统|主体|事件|日期|地区)|此(?:日期|事件)|\b(?:this|that|its)\b", re.I
)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_INDEXED = re.compile(r"([\u4e00-\u9fff]{2,8}?)([一二三四五六七八九十\d]+)(号|代|期)")
_REGION = re.compile(
    r"(?:在|面向|针对)([\u4e00-\u9fff]{2,8}?)(?:地区|境内|市场|范围)|"
    r"\b(?:in|within) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b"
)
_GENERIC = re.compile(
    r"研究问题|研究目标|研究|问题|请问|请|是什么|有哪些|如何|发展情况|"
    r"\b(?:what|when|where|which|does|the|and|for|research|question|find|about|source)\b",
    re.I,
)


class PlanScopeError(ValueError):
    pass


def _matches(patterns: dict[str, str], text: str) -> set[str]:
    return {key for key, pattern in patterns.items() if re.search(pattern, text, re.I)}


def _tokens(text: str) -> set[str]:
    text = _GENERIC.sub(" ", text).lower()
    words = set(re.findall(r"[a-z][a-z0-9-]{2,}", text))
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(part[i : i + 2] for i in range(len(part) - 1))
    return words


def question_scope_error(goal: str, question: ResearchQuestion) -> str | None:
    text = question.question
    # Numeric generations are identities, not interchangeable topic keywords.
    subjects: dict[tuple[str, str], set[str]] = {}
    for base, number, suffix in _INDEXED.findall(goal):
        subjects.setdefault((_GENERIC.sub("", base), suffix), set()).add(number)
    for (base, suffix), numbers in subjects.items():
        if not base:
            continue
        variants = re.findall(re.escape(base) + r"([一二三四五六七八九十\d]+)" + suffix, text)
        if variants and not set(variants).issubset(numbers):
            return "subject_identity_changed"
        if not any(base + number + suffix in text for number in numbers) and not _REFERENCE.search(
            text
        ):
            return "subject_identity_missing"
    goal_regions = {a or b for a, b in _REGION.findall(goal)}
    regions = {a or b for a, b in _REGION.findall(text)}
    if goal_regions and regions and not regions.issubset(goal_regions):
        return "region_changed"
    years, requested_years = set(_YEAR.findall(text)), set(_YEAR.findall(goal))
    if requested_years and years:
        lo, hi = min(requested_years), max(requested_years)
        outside = {year for year in years if not lo <= year <= hi}
        inside = {year for year in years if lo <= year <= hi}
        if outside and not (_CONTRAST.search(text) and inside):
            return "time_scope_changed"
    facts, requested_facts = _matches(_FACTS, text), _matches(_FACTS, goal)
    states, requested_states = _matches(_STATES, text), _matches(_STATES, goal)
    if requested_states and states and not states & requested_states:
        return "requested_event_replaced"
    if requested_facts == {"date"}:
        if facts - requested_facts:
            return "unrequested_fact_type"
        if (
            "date" not in facts
            and not _VERIFY.search(text)
            and not (_CONTRAST.search(text) and states & requested_states)
        ):
            return "missing_requested_date_relation"
    anchors = _tokens(goal)
    if anchors and not anchors & _tokens(text) and not _REFERENCE.search(text):
        return "unrelated_objective"
    return None


def enforce_plan_scope(plan: ResearchPlan, goal: str, *, repair: bool = True) -> ResearchPlan:
    if plan.goal != goal:
        raise PlanScopeError("immutable_goal_changed")
    # Completion criteria feed later gap scheduling. Retaining an explicit
    # off-topic objective here would silently reintroduce a removed question.
    # Reject rather than weaken or invent replacement acceptance criteria.
    requested_facts = _matches(_FACTS, goal)
    if requested_facts == {"date"}:
        for criterion in plan.completion_criteria:
            if _matches(_FACTS, criterion) - requested_facts:
                raise PlanScopeError("completion_criterion_out_of_scope")
        if _matches(_FACTS, plan.scope_summary) - requested_facts:
            if not repair:
                raise PlanScopeError("scope_summary_out_of_scope")
            plan = plan.model_copy(update={"scope_summary": goal})
    rejected = [(q.id, question_scope_error(goal, q)) for q in plan.questions]
    rejected = [(qid, reason) for qid, reason in rejected if reason is not None]
    if not rejected:
        return plan
    if not repair:
        raise PlanScopeError(";".join(f"{qid}:{reason}" for qid, reason in rejected))
    rejected_ids = {qid for qid, _ in rejected}
    retained = [q for q in plan.questions if q.id not in rejected_ids]
    # Never fabricate a factual substitute or accept a wholly unrelated outline.
    if not retained:
        raise PlanScopeError("no_in_scope_question")
    return plan.model_copy(
        update={
            "questions": retained,
            "scope_summary": goal,
        }
    )
