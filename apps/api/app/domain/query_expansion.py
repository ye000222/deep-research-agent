"""Intent-scoped retrieval vocabulary; never an evidence acceptance rule."""

import re

from app.domain.passage_relevance import events


def query_intent(question: str) -> str:
    if re.search(r"何时|日期|哪年|哪天|时间|\bwhen\b|\bdate\b|\byear\b", question, re.I):
        return "factual_date_event"
    patterns = (
        ("policy_regulation", r"政策|法规|法律|条例|regulation|legislation|policy"),
        ("performance_benchmark", r"性能|准确率|基准|benchmark|accuracy|performance"),
        ("application_example", r"应用|案例|落地|application|case study|deployment example"),
        ("definition", r"定义|含义|是什么|definition|\bdefine\b|what is"),
        ("entity_status", r"状态|是否|宣布|宣告|status|operational|has .+ (opened|launched)"),
    )
    return next(
        (kind for kind, pattern in patterns if re.search(pattern, question, re.I)),
        "entity_status" if events(question) else "technical_background",
    )


def intent_gap_hints(question: str, *, rotation: int = 0) -> list[str]:
    """Every variant keeps the full task anchor, including its precise event.

    Rotate authority/genre, NOT event identity: opening is not launch or
    commissioning. A generic date task never inherits product/catalog terms.
    """
    kind = query_intent(question)
    vocab = {
        "factual_date_event": (
            "日期 官方 公告 原始报道",
            "date official announcement press release",
            "日期 主管机构 原始来源",
        ),
        "entity_status": (
            "官方 状态 公告",
            "official status authority statement",
            "主管机构 当前状态",
        ),
        "definition": ("官方 定义 术语", "official definition terminology", "标准 定义 原文"),
        "technical_background": (
            "技术原理 官方技术文档",
            "technical documentation research paper",
            "技术标准 原始论文",
        ),
        "application_example": (
            "官方 应用 案例",
            "official application example case study",
            "部署 实例 技术报告",
        ),
        "policy_regulation": (
            "法规 正式文本 主管机关",
            "official regulation legal text",
            "政策 公告 原文",
        ),
        "performance_benchmark": (
            "公开基准 数据集 测量方法",
            "benchmark performance measurement methodology",
            "评测 原始结果 数据",
        ),
    }[kind]
    anchor = " ".join(question.split())[:220]
    return [f"{anchor} {vocab[(i + rotation) % len(vocab)]}" for i in range(3)]
