"""Deterministic offline recall & input-savings replay for the P2 source selector.

Replays fixed synthetic webpages (no provider/network) and compares the
「first-14k-characters baseline」 against the relevant-block selector on:

  - key-evidence recall (does the selected text cover the ground-truth quotes)
  - input scale savings (characters/tokens) versus the baseline window

This is the scheme's layer-A deterministic replay: it validates selection
scheduling and coverage, NOT real-internet or real-model accuracy/429/latency.
"""

# ruff: noqa: RUF001 -- Chinese replay fixtures intentionally use full-width punctuation.

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from app.context.source_selector import select_relevant_blocks

_OUT = Path("evals/reports/replay_recall_savings.json")
_HEADER = (
    "导航链接 广告 面包屑 Cookie 提示 侧栏推荐 相关文章 页脚版权 "
    "Secondary nav Service links More readings Popular now Trending tags Legal links"
)
_UNRELATED = [
    "这位作者长期关注供应链数字化，多篇文章被行业媒体转载。",
    "读者可以在评论区留言，也可以邮件订阅每周摘要。",
    "公司成立于十年前，早期以咨询业务为主，后来转向软件产品。",
    "本文仅代表个人观点，不构成任何投资建议，请结合自身判断。",
    "页面使用响应式布局，在移动端与桌面端均可正常阅读。",
    "更多相关主题将在后续栏目陆续发布，敬请期待后续内容。",
    "该平台提供开放接口，第三方可以申请接入并集成到自身系统。",
    "团队分布在不同时区，采用异步协作方式推进日常迭代。",
]


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _nav_blocks(count: int, rng: random.Random) -> str:
    return "\n\n".join(
        f"{_HEADER} {_UNRELATED[rng.randrange(len(_UNRELATED))]}".strip() for _ in range(count)
    )


def _build_case(rng: random.Random, evidence_sentence: str) -> str:
    """Compose a long synthetic page; evidence lands beyond the 14k prefix."""
    prefix = _nav_blocks(160, rng)
    tail = (
        "关键数据来自官方年度报告。\n"
        "以下是经过交叉核对的实证结论：\n"
        f"{evidence_sentence}\n"
        "该结论在后续章节的对比中再次得到验证。\n"
    )
    return prefix + "\n\n" + tail


def _evidence_sentences() -> list[str]:
    return [
        "工业视觉检测中结构光方案在透明玻璃上的误检率明显高于漫反射光源。",
        "高光谱相机对果蔬表面早期缺陷的识别灵敏度达到百分之九十六。",
        "单目测距精度主要受基线与目标纹理影响，在弱纹理场景误差约百分之八。",
        "该市场二零二四年全球出货量达到一千二百万台，较上年增长一成。",
        "跨段比较显示，深度学习方法在遮挡场景下的鲁棒性优于传统特征匹配。",
    ]


def _coverage(text: str, sentences: list[str]) -> float:
    norm_text = _norm(text)
    hit = sum(1 for s in sentences if _norm(s) in norm_text)
    return hit / max(1, len(sentences))


def main() -> int:
    rng = random.Random(20260910)
    baseline_window = 14_000
    cases: list[dict[str, object]] = []
    sentences = _evidence_sentences()
    for index in range(30):
        sentence = sentences[index % len(sentences)]
        page = _build_case(rng, sentence)
        baseline_text = page[:baseline_window]

        selected = select_relevant_blocks(
            page,
            ("结构光 高光谱 精度 出货量 遮挡 检测",),
            limit=64,
            expand_neighbors=1,
        )
        candidate_text = "\n".join(w.content for w in selected)

        baseline_recall = _coverage(baseline_text, [sentence])
        candidate_recall = _coverage(candidate_text, [sentence])
        savings = max(0.0, (baseline_window - len(candidate_text)) / max(1, baseline_window))
        cases.append(
            {
                "case_id": f"replay_{index + 1:02d}",
                "evidence_present_in_prefix": baseline_recall > 0,
                "baseline_recall": baseline_recall,
                "candidate_recall": candidate_recall,
                "baseline_input_chars": min(baseline_window, len(page)),
                "candidate_input_chars": len(candidate_text),
                "input_savings_pct": round(savings * 100, 1),
                "candidate_covers_evidence": candidate_recall > 0,
            }
        )

    total = len(cases)
    hit_evidence = sum(1 for c in cases if c["candidate_covers_evidence"])
    baseline_hit = sum(1 for c in cases if c["baseline_recall"] > 0)
    avg_savings = sum(float(c["input_savings_pct"]) for c in cases) / total
    report = {
        "mode": "replay (deterministic, fixed pages, no provider/network)",
        "policy": "relevant_block_selector vs first-14k-prefix baseline",
        "case_count": total,
        "key_evidence_present_beyond_prefix": total - baseline_hit,
        "key_evidence_recovered_by_selector": hit_evidence,
        "baseline_evidence_hit_cases": baseline_hit,
        "candidate_evidence_hit_cases": hit_evidence,
        "avg_input_savings_pct": round(avg_savings, 1),
        "cases": cases,
        "caveat": (
            "Deterministic recall/savings only. NOT real-model accuracy, 429 rate, "
            "or real-internet p95; those require real Provider + network + cost approval."
        ),
    }
    _OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "cases": total,
            "evidence_beyond_prefix": total - baseline_hit,
            "recovered_by_selector": hit_evidence,
            "baseline_hit_cases": baseline_hit,
            "candidate_hit_cases": hit_evidence,
            "avg_input_savings_pct": round(avg_savings, 1),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
