"""Deterministic V2 scheduling, triage, and claim-quality primitives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit


class QueryFamily(StrEnum):
    SCOPE = "scope"
    AUTHORITATIVE = "authoritative"
    ALTERNATE = "alternate"
    CONTRADICTION = "contradiction"


QUERY_FAMILIES = tuple(QueryFamily)


def query_family_for_attempt(attempt_index: int) -> QueryFamily | None:
    """Return one of four bounded families; no implicit suffix cycling."""

    if attempt_index < 0 or attempt_index >= len(QUERY_FAMILIES):
        return None
    return QUERY_FAMILIES[attempt_index]


def _compact_search_hint(hint: str) -> str:
    """Keep planner hints query-shaped instead of carrying audit prose.

    Replanning can feed the previous criterion back into ``search_hints``.
    That text is useful for evaluation but harmful as a search anchor (it often
    contains a whole question followed by requirements such as "independent
    source" or "official report").
    """

    compact = " ".join(hint.strip().split())
    if not compact:
        return ""
    # A question mark is a reliable boundary between a topic and appended
    # audit instructions in both Chinese and English planner output.
    compact = re.split(r"[?\uff1f]", compact, maxsplit=1)[0].strip()
    # Replan hint banks intentionally rotate the *semantic* search angle.
    # Keep the useful nouns from those banks while dropping only the audit
    # qualifiers; otherwise every rotated hint collapses back to the same
    # topic anchor and the run appears to exhaust its source space without
    # ever issuing a new query.
    compact = re.sub(
        r"\bindependent\s+market\s+report\s+methodology\b",
        "market report methodology",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"\bofficial\s+report\s+benchmark\s+independent\s+source\b",
        "report benchmark source",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"\bsurvey\s+comparative\s+study\s+evidence\b",
        "comparative study evidence",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"\bgovernment\s+forecast\s+primary\s+data\b",
        "forecast primary data",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"\bpeer[- ]reviewed\s+evaluation\s+dataset\b",
        "evaluation dataset",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"\bcomparative\s+benchmark\s+results\s+study\b",
        "benchmark results study",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"\bindustry\s+outlook\s+adoption\s+statistics\b",
        "industry adoption statistics",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.split(
        r"(?:原文|逐字|直接)(?:中)?(?:列出|描述|说明|给出|包含|提供)?|"
        r"(?:official|authoritative|independent|benchmark|comparative|survey)\s+"
        r"(?:report|source|study|evidence|data)?|"
        r"(?:industry|market)\s+outlook|adoption\s+statistics|"
        r"government\s+forecast\s+primary\s+data|"
        r"independent\s+market\s+report\s+methodology|"
        r"official\s+report\s+benchmark\s+independent\s+source|"
        r"survey\s+comparative\s+study\s+evidence|"
        r"systematic\s+review\s+validation\s+evidence|"
        r"peer[- ]reviewed\s+evaluation\s+dataset|"
        r"comparative\s+benchmark\s+results\s+study|"
        r"industry\s+outlook\s+adoption\s+statistics|"
        r"manufacturer\s+product(?:\s+page)?|vendor\s+solution|"
        r"customer\s+case(?:\s+study)?|field\s+trial|evaluation\s+report|"
        r"technical\s+specification|peer[- ]reviewed",
        compact,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    compact = " ".join(compact.split())
    return compact[:160]


def build_family_query(
    *,
    question: str,
    criterion: str = "",
    hints: tuple[str, ...] = (),
    family: QueryFamily,
    prefer_alternate_hint: bool = False,
) -> str:
    usable_hints = tuple(
        compact
        for compact in (_compact_search_hint(hint) for hint in hints)
        if compact
    )
    use_alternate_hint = prefer_alternate_hint or family is QueryFamily.ALTERNATE
    hint_index = 0
    if use_alternate_hint and usable_hints:
        first_language = any("\u4e00" <= char <= "\u9fff" for char in usable_hints[0])
        alternate_indexes = [
            index
            for index, hint in enumerate(usable_hints[1:], start=1)
            if any("\u4e00" <= char <= "\u9fff" for char in hint) != first_language
        ]
        # Prefer a genuinely different language; if the planner supplied only
        # one language, the shortest compact hint is the least polluted anchor.
        hint_index = alternate_indexes[0] if alternate_indexes else min(
            range(len(usable_hints)), key=lambda index: len(usable_hints[index])
        )
    base = (
        usable_hints[min(hint_index, len(usable_hints) - 1)]
        if usable_hints
        else question.strip()
    )
    base_is_chinese = any("\u4e00" <= char <= "\u9fff" for char in base)
    chinese = base_is_chinese if use_alternate_hint and usable_hints else any(
        "\u4e00" <= char <= "\u9fff" for char in question
    )
    suffixes = {
        QueryFamily.SCOPE: "范围 定义 数据" if chinese else "scope definition data",
        QueryFamily.AUTHORITATIVE: (
            "官方 标准 政府 原始数据 论文"
            if chinese
            else "official standard government primary paper"
        ),
        QueryFamily.ALTERNATE: (
            "同义词 英文术语 国际来源"
            if chinese
            else "synonyms alternate terminology international"
        ),
        QueryFamily.CONTRADICTION: (
            "失败 局限 反例 争议" if chinese else "failure limitations counterexample contradiction"
        ),
    }
    # Planner hints already carry the topic anchor.  A scope query should stay
    # compact: appending prose such as "原文列出……的表述" substantially lowers
    # recall in Chinese engines.  Follow-up families retain only the useful
    # middle of that criterion, with audit-oriented boilerplate removed.
    compact_criterion = re.sub(
        r"(?:原文|逐字|直接)(?:中)?|(?:列出|描述|说明|给出|包含|提供)|(?:的)?表述|"
        r"at\s+least\s+two\s+independent\s+sources|original\s+(?:text|source)|"
        r"(?:describe|list|state|show|provide)(?:s|d|ed|ing)?",
        " ",
        criterion,
        flags=re.IGNORECASE,
    )
    compact_criterion = " ".join(compact_criterion.split())[:100]
    if use_alternate_hint and usable_hints and base_is_chinese != any(
        "\u4e00" <= char <= "\u9fff" for char in compact_criterion
    ):
        compact_criterion = ""
    parts = (
        (base,)
        if family is QueryFamily.SCOPE and usable_hints
        else (base, compact_criterion, suffixes[family])
    )
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        normalized = " ".join(part.casefold().split())
        if normalized and normalized not in seen:
            unique.append(part)
            seen.add(normalized)
    return " ".join(unique)[:320]


@dataclass(frozen=True, slots=True)
class ActionCost:
    estimated_tokens: int = 0
    latency_ms: int = 0
    pages: int = 0
    provider_requests: int = 0


def expected_utility(
    *,
    priority: int,
    gap_risk: float,
    probability_of_accepted_evidence: float,
    expected_coverage_delta: float,
    source_novelty: float,
    cost: ActionCost,
) -> float:
    priority_weight = {1: 1.5, 2: 1.0, 3: 0.7}.get(priority, 0.7)
    denominator = (
        max(1.0, cost.estimated_tokens / 1_000)
        + max(0.0, cost.latency_ms / 2_000)
        + max(0, cost.pages) * 1.5
        + max(0, cost.provider_requests) * 1.0
    )
    numerator = (
        priority_weight
        * _unit(gap_risk)
        * _unit(probability_of_accepted_evidence)
        * _unit(expected_coverage_delta)
        * _unit(source_novelty)
    )
    return round(numerator / denominator, 6)


@dataclass(frozen=True, slots=True)
class TriageResult:
    accepted: bool
    score: float
    reason: str
    source_role: str


_INJECTION_RE = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|system)|reveal.{0,30}(?:secret|api.?key)|"
    r"忽略.{0,20}(?:系统|之前).{0,20}(?:指令|提示)",
    re.IGNORECASE,
)
_LOW_VALUE_RE = re.compile(
    r"(?:sign\s*in|log\s*in|enable javascript|cookie policy|access denied|验证码|登录后查看|"
    r"版权所有.{0,20}广告)",
    re.IGNORECASE,
)
# Match standalone numeric values, not digits embedded in semantic tokens
# such as ``3D``, ``2D`` or ``3C``. Those tokens are common in qualitative
# research criteria and must not become numeric-scope requirements.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:19|20)\d{2}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)?\s*(?:亿美元|亿元|万元|人民币|美元|欧元|英镑|%|\uff05|ms|s|秒|kg|吨|万|亿|元)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def cheap_triage(
    *,
    question: str,
    criteria: tuple[str, ...],
    query_hints: tuple[str, ...] = (),
    text: str,
    url: str,
) -> TriageResult:
    """Reject obvious low-yield pages without invoking a model."""

    role = classify_source_role(url, text=text[:2_000])
    if len(text.strip()) < 300:
        return TriageResult(False, 0.0, "body_too_short", role)
    if _INJECTION_RE.search(text[:20_000]):
        return TriageResult(False, 0.0, "prompt_injection_detected", role)
    if _LOW_VALUE_RE.search(text[:4_000]) and len(text) < 2_000:
        return TriageResult(False, 0.1, "login_or_navigation_page", role)
    # The plan may be authored in Chinese while the authoritative source is
    # English. Include the exact bilingual search anchors that retrieved the
    # page; otherwise a relevant paper is deterministically rejected before
    # the model ever sees it.
    objective = _tokens(" ".join((question, *criteria, *query_hints)))
    observed = _tokens(text[:40_000])
    overlap = len(objective & observed) / max(1, min(12, len(objective)))
    traceability = 1.0 if _traceability_markers(text) >= 2 else 0.6
    role_score = {
        "government": 1.0,
        "academic": 0.95,
        "standard": 0.95,
        "manufacturer": 0.82,
        "association": 0.82,
        "independent_research": 0.78,
        "publisher": 0.68,
        "unknown": 0.58,
        "aggregator": 0.35,
    }.get(role, 0.55)
    score = round(min(1.0, 0.65 * overlap + 0.20 * traceability + 0.15 * role_score), 4)
    return TriageResult(
        score >= 0.42, score, "accepted" if score >= 0.42 else "topic_mismatch", role
    )


def classify_source_role(url: str, *, text: str = "") -> str:
    host = (urlsplit(url).hostname or "").casefold()
    path = urlsplit(url).path.casefold()
    sample = text.casefold()
    if host.endswith(".gov") or ".gov." in host:
        return "government"
    if host.endswith(".edu") or ".edu." in host or host.endswith(".ac.cn"):
        return "academic"
    if any(
        part in host
        for part in (
            "arxiv.org",
            "pubmed",
            "acm.org",
            "ieee.org",
            "nature.com",
            "springer.com",
            "sciencedirect.com",
            "frontiersin.org",
            "mdpi.com",
            "cnki.",
        )
    ):
        return "academic"
    if (
        any(part in path for part in ("/bitstream/", "/portalfiles/", "/eprints/"))
        or (
            path.endswith(".pdf")
            and "abstract" in sample
            and any(marker in sample for marker in ("doi", "keywords", "references"))
        )
    ):
        # Institutional repositories frequently use country-code domains
        # rather than .edu (for example university portals under .dk/.my).
        # Repository paths or a conventional scholarly PDF front matter are
        # stronger role signals than the TLD alone.
        return "academic"
    if any(part in host for part in ("iso.org", "iec.ch", "nist.gov", "standards.")):
        return "standard"
    if any(part in host for part in ("blog", "wenku", "baike", "zhihu", "csdn", "medium.com")):
        return "aggregator"
    if any(
        part in host
        for part in (
            "cognex.",
            "keyence.",
            "baslerweb.",
            "teledynevisionsolutions.",
            "mvtec.",
            "omron.",
            "hikrobotics.",
            "daheng-imaging.",
            "deepvai.",
            "shuangyi-tech.",
        )
    ) or any(part in path for part in ("/product", "/products", "/specification")):
        return "manufacturer"
    if any(
        marker in sample
        for marker in ("manufacturer", "product specification", "产品规格", "制造商")
    ):
        return "manufacturer"
    if host.endswith(".org"):
        return "association"
    if any(
        marker in sample for marker in ("doi:", "methodology", "方法", "references", "参考文献")
    ):
        return "independent_research"
    # Market-statistic pages are often hosted by ordinary publisher domains
    # rather than a dedicated research-index host.  Promote only pages whose
    # own title/body explicitly carries both a market-report signal and a
    # quantitative market signal; generic news/blog pages remain publishers.
    if (
        any(
            marker in sample
            for marker in (
                "market report",
                "market research",
                "行业报告",
                "研究报告",
                "研究机构",
            )
        )
        and any(
            marker in sample
            for marker in ("market size", "cagr", "市场规模", "增长率", "年复合")
        )
    ):
        return "independent_research"
    # Some publishers expose numeric market pages whose body omits the
    # literal phrase "market report" (the title/URL carries that context).
    # Treat an explicit market path plus quantitative forecast language as an
    # independent research source; this remains narrower than promoting every
    # ordinary .com publisher and works for arbitrary market questions.
    if (
        any(
            marker in f"{host}{path}"
            for marker in ("market", "forecast", "industry-report")
        )
        and any(
            marker in sample
            for marker in (
                "cagr",
                "market size",
                "market share",
                "forecast",
                "%",
                "亿美元",
                "million",
                "billion",
            )
        )
    ):
        return "independent_research"
    if host.endswith(".com") or host.endswith(".cn"):
        return "publisher"
    return "unknown"


def source_reliability_score(url: str, *, text: str = "", title: str = "") -> float:
    role = classify_source_role(url, text=f"{title}\n{text}")
    base = {
        "government": 0.93,
        "standard": 0.92,
        "academic": 0.90,
        "manufacturer": 0.84,
        "association": 0.83,
        "independent_research": 0.82,
        "publisher": 0.72,
        "unknown": 0.66,
        "aggregator": 0.54,
    }[role]
    sample = f"{title}\n{text[:8_000]}"
    folded = sample.casefold()
    trace = _traceability_markers(sample)
    feature_adjustment = min(0.06, trace * 0.02)
    if any(marker in folded for marker in ("author", "作者", "机构", "organization")):
        feature_adjustment += 0.015
    if any(marker in folded for marker in ("published", "发布日期", "发布时间", "datepublished")):
        feature_adjustment += 0.015
    if any(marker in folded for marker in ("method", "dataset", "doi", "方法", "数据集")):
        feature_adjustment += 0.015
    if any(marker in folded for marker in ("转载", "转自", "reposted", "syndicated")):
        feature_adjustment -= 0.08
    if len(text.strip()) < 500 or "truncated" in folded or "正文截断" in folded:
        feature_adjustment -= 0.05
    return round(min(0.97, max(0.45, base + feature_adjustment)), 4)


def infer_claim_type(value: str) -> str:
    folded = value.casefold()
    if any(token in folded for token in ("market share", "市场份额", "规模", "增长率")):
        return "market_statistic"
    if any(token in folded for token in ("accuracy", "f1", "precision", "性能", "准确率")):
        return "academic_performance"
    vendor_marked = any(
        token in folded
        for token in ("vendor", "manufacturer", "supplier", "厂商", "厂家", "供应商")
    )
    product_marked = any(
        token in folded
        for token in ("product", "platform", "产品", "平台", "型号")
    )
    if vendor_marked and product_marked:
        return "vendor_product"
    if any(token in folded for token in ("specification", "规格", "参数", "型号")):
        return "product_specification"
    if any(token in folded for token in ("compare", "versus", "领先", "优于", "比较")):
        return "comparative"
    if _NUMBER_RE.search(value):
        return "numeric"
    return "factual"


def source_role_fits_claim(*, claim_type: str, source_role: str) -> bool:
    allowed = {
        "market_statistic": {"government", "academic", "independent_research", "association"},
        "academic_performance": {"academic", "independent_research"},
        "product_specification": {"manufacturer", "standard", "government"},
        "vendor_product": {"manufacturer"},
        "comparative": {"academic", "independent_research", "government", "standard"},
        "numeric": {"government", "academic", "standard", "manufacturer", "independent_research"},
        "factual": {
            "government",
            "academic",
            "standard",
            "manufacturer",
            "association",
            "independent_research",
            "publisher",
        },
    }
    return source_role in allowed.get(claim_type, allowed["factual"])


def claim_quote_entails(claim: str, quote: str) -> bool:
    claim_tokens = _tokens(claim)
    quote_tokens = _tokens(quote)
    claim_numbers = {_scope_number(item) for item in _NUMBER_RE.findall(claim)}
    quote_numbers = {_scope_number(item) for item in _NUMBER_RE.findall(quote)}
    if _different_dominant_scripts(claim, quote):
        # A lexical entailment check is undefined for a translated claim and
        # its source-language quotation.  This path is reached only after the
        # caller has proved that the quotation occurs in the fetched source;
        # keep numeric assertions strict and let the existing relevance,
        # confidence, role and score gates validate the extracted card.
        return bool(quote_tokens) and claim_numbers.issubset(quote_numbers)
    lexical = len(claim_tokens & quote_tokens) / max(1, min(10, len(claim_tokens)))
    return lexical >= 0.45 and claim_numbers.issubset(quote_numbers)


def _different_dominant_scripts(left: str, right: str) -> bool:
    def profile(value: str) -> tuple[int, int]:
        cjk = sum("\u3400" <= character <= "\u9fff" for character in value)
        latin = sum("a" <= character.casefold() <= "z" for character in value)
        return cjk, latin

    left_cjk, left_latin = profile(left)
    right_cjk, right_latin = profile(right)
    return (left_cjk >= 2 and right_latin >= 4 and right_cjk < 2) or (
        right_cjk >= 2 and left_latin >= 4 and left_cjk < 2
    )


def numeric_scope_consistent(*, criterion: str, claim: str, quote: str) -> bool:
    observed_text = f"{claim}\n{quote}"
    required = {_scope_number(item) for item in _NUMBER_RE.findall(criterion)}
    observed = {_scope_number(item) for item in _NUMBER_RE.findall(observed_text)}
    if not required:
        return True
    if required and not required.issubset(observed):
        return False

    # Numeric claims are only interchangeable when their units are
    # interchangeable.  A bare number in a criterion remains permissive for
    # legacy criteria, while an explicit unit is a hard requirement.
    required_units = {
        unit for _value, unit in required if unit and not unit.isdigit()
    }
    observed_units = {unit for _value, unit in observed if unit and not unit.isdigit()}
    if required_units and not required_units.issubset(observed_units):
        return False

    folded_criterion = criterion.casefold()
    folded_observed = observed_text.casefold()
    scope_markers = (
        "每", "per", "样本", "sample", "分母", "denominator", "平均", "average",
        "全球", "中国", "美国", "欧洲", "北美", "亚太", "日本", "韩国", "印度",
        "德国", "英国",
    )
    required_scope = {marker for marker in scope_markers if marker in folded_criterion}
    return required_scope.issubset(
        {marker for marker in scope_markers if marker in folded_observed}
    )


def normalized_gain(
    *, gain: float, tokens: int, pages: int, requests: int, latency_ms: int
) -> dict[str, float]:
    return {
        "gain_per_1k_tokens": round(gain / max(1.0, tokens / 1_000), 6),
        "gain_per_page": round(gain / max(1, pages), 6),
        "gain_per_provider_request": round(gain / max(1, requests), 6),
        "gain_per_second": round(gain / max(0.001, latency_ms / 1_000), 6),
    }


def _tokens(value: str) -> set[str]:
    folded = value.casefold()
    words = {word for word in re.findall(r"[a-z0-9][a-z0-9._%-]+", folded) if len(word) > 1}
    cjk = "".join(char for char in folded if "\u3400" <= char <= "\u9fff")
    words.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    return words


def _traceability_markers(value: str) -> int:
    folded = value.casefold()
    markers = (
        "author",
        "published",
        "method",
        "dataset",
        "doi",
        "作者",
        "发布",
        "方法",
        "数据集",
        "标准号",
    )
    return sum(marker in folded for marker in markers)


def _normalize_number(value: str) -> str:
    compact = "".join(value.casefold().replace("\uff0c", ",").split())
    # Treat a comma followed by a three-digit group as a thousands separator;
    # retain decimal commas such as 1,5 for locales that use them.
    return re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", compact)


def _scope_number(value: str) -> tuple[str, str]:
    """Normalize a numeric value together with its explicit unit/scope."""

    normalized = _normalize_number(value)
    normalized = normalized.replace("percent", "%").replace("百分比", "%")
    match = re.match(r"(?P<number>(?:19|20)\d{2}|[-+]?\d+(?:[.,]\d+)?)(?P<unit>.*)", normalized)
    if match is None:
        return normalized, ""
    unit = match.group("unit")
    aliases = {"\uff05": "%", "公斤": "kg", "千克": "kg", "人民币": "元"}
    return match.group("number"), aliases.get(unit, unit)


def _unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
