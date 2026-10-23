"""Explicit quote-local calendar/time/ordinal facts; never infer from metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_MONTH_NAMES = (
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may",),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sep", "sept"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
)
_MONTHS = {name: i for i, names in enumerate(_MONTH_NAMES, 1) for name in names}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DATE_EN = re.compile(
    rf"\b(?:(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"|(?P<day_first>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month_last>{_MONTH}))"
    r"(?:,?\s+(?P<year>(?:19|20)\d{2}))?\b",
    re.I,
)
_DATE_ZH = re.compile(r"(?:(?P<year>\d{4})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})[日号]")
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?:\s*(a\.?m\.?|p\.?m\.?))?", re.I)
_TIME_ZH = re.compile(r"(上午|下午|晚上|凌晨)?\s*(\d{1,2})[时点](\d{1,2})分")
_ORDINAL = re.compile(r"第(\d+)(?:颗|个|次|号)?|\b(\d+)(st|nd|rd|th)\b", re.I)
_PLANNED = re.compile(
    r"计划|拟于|预计|将于|scheduled|\bplanned\b|\bplans?\b|\bmay\s+launch|will be launched", re.I
)
_ACTUAL_LAUNCH = re.compile(r"成功发射|发射成功|已发射|\blaunched\b", re.I)
_OPENING = re.compile(
    r"正式开通|开通服务|officially opened|service.{0,12}(?:opened|commissioned)|commissioning", re.I
)
_COMPLETE = re.compile(
    r"建成|完成组网|星座部署.*完成|completed? constellation|"
    r"constellation.{0,30}complet|complet.{0,30}constellation",
    re.I,
)
_NEGATION = re.compile(r"未能|未成功|未发射|未开通|未完成|没有|尚未|never|\bnot\b|didn't", re.I)


@dataclass(frozen=True)
class NormalizedFacts:
    residual: str
    signals: frozenset[str]
    applied: tuple[str, ...]


def normalize_quote_facts(text: str) -> NormalizedFacts:
    signals: set[str] = set()
    applied: set[str] = set()

    def calendar(match: re.Match[str]) -> str:
        groups = match.groupdict()
        raw_month = groups.get("month") or groups.get("month_last") or ""
        month = int(raw_month) if raw_month.isdigit() else _MONTHS[raw_month.rstrip(".").lower()]
        day = int(groups.get("day") or groups.get("day_first") or "0")
        year = groups.get("year")
        try:
            date(int(year) if year else 2000, month, day)
        except ValueError:
            signals.add("invalid_date")
            return match.group()
        signals.update((f"month:{month}", f"day:{day}", f"date:{month:02}-{day:02}"))
        if year:
            signals.update((f"year:{year}", f"date:{year}-{month:02}-{day:02}"))
        applied.add("calendar_date")
        return " "

    residual = _DATE_EN.sub(calendar, text)
    residual = _DATE_ZH.sub(calendar, residual)

    def clock(match: re.Match[str]) -> str:
        if match.re is _TIME_ZH:
            marker, hour, minute = match.groups()
            marker = "pm" if marker in {"下午", "晚上"} else "am" if marker else ""
        else:
            hour, minute, marker = match.groups()
            marker = (marker or "").replace(".", "").lower()
        h, m = int(hour), int(minute)
        if m > 59 or h > 23 or (marker and not 1 <= h <= 12):
            signals.add("invalid_time")
            return match.group()
        if marker:
            h = h % 12 + (12 if marker == "pm" else 0)
        signals.add(f"time:{h:02}:{m:02}")
        applied.add("clock_time")
        return " "

    residual = _TIME.sub(clock, residual)
    residual = _TIME_ZH.sub(clock, residual)

    def ordinal(match: re.Match[str]) -> str:
        n = int(match.group(1) or match.group(2))
        suffix = match.group(3)
        expected = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        if suffix and suffix.lower() != expected:
            signals.add("invalid_ordinal")
        signals.add(f"ordinal:{n}")
        applied.add("ordinal")
        return " "

    residual = _ORDINAL.sub(ordinal, residual)

    def month_only(match: re.Match[str]) -> str:
        raw = match.group().rstrip(".").lower()
        signals.add(f"month:{_MONTHS[raw]}")
        applied.add("month_name")
        return " "

    residual = re.sub(rf"\b{_MONTH}\b", month_only, residual, flags=re.I)

    def chinese_month(match: re.Match[str]) -> str:
        signals.add(f"month:{int(match.group(1))}")
        applied.add("month_name")
        return " "

    residual = re.sub(r"(\d{1,2})月", chinese_month, residual)

    def year_only(match: re.Match[str]) -> str:
        signals.add(f"year:{match.group(1)}")
        return " "

    residual = re.sub(
        r"(?<![a-zA-Z\d])((?:19|20)\d{2})(?:年|(?![a-zA-Z\d\u3400-\u9fff]))",
        year_only,
        residual,
    )
    return NormalizedFacts(residual, frozenset(signals), tuple(sorted(applied)))


def event_conflicts(claim: str, quote: str) -> tuple[str, ...]:
    """Conservative guards for explicitly different events/modality/polarity."""
    reasons: list[str] = []
    if (
        _PLANNED.search(quote)
        and not _PLANNED.search(claim)
        and (not _ACTUAL_LAUNCH.search(quote) or re.search(r"计划|planned|scheduled", quote, re.I))
    ):
        reasons.append("planned_does_not_prove_actual")
    if _OPENING.search(claim) and not _OPENING.search(quote) and _COMPLETE.search(quote):
        reasons.append("completion_does_not_prove_opening")
    if _NEGATION.search(quote) and not _NEGATION.search(claim):
        reasons.append("negated_quote_does_not_prove_positive_claim")
    return tuple(reasons)
