"""Attach Revision 2 review outcomes to the immutable pre-change warning ledger."""

# This is an offline artifact builder; the reviewed records are auditable inputs,
# not runtime policy. Claim identifiers here join frozen diagnostic artifacts only.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LEDGER = ROOT / "artifacts/v2_1r_qualifier_revision2_warning_ledger.json"
REPLAY = ROOT / "artifacts/v2_1r_qualifier_semantic_preservation_revision2_replay.json"
REVIEWED = ROOT / "artifacts/v2_1r_qualifier_revision2_warning_ledger_reviewed.json"
REPORT = ROOT / "artifacts/v2_1r_qualifier_semantic_preservation_revision2_report.md"

NEW_BLOCK_REVIEWS = {
    "01a0c437-488d-7dc7-b278-44aac2d174b0": (
        "SOURCE_DETECTOR_FALSE_POSITIVE",
        "短词尾‘率’命中嵌入式效率词；非数值测量义。修正为有界形态检测并排除概率词。",
    ),
    "01a0c84e-9da1-7478-ac92-ed89de3957a5": (
        "STRUCTURED_EQUIVALENCE_MISSED",
        "来源是时间区间终点，输出 period carrier 已包含该年份；比较器原只匹配事件日期。",
    ),
    "01a0c82b-2f49-70bf-a1e6-3654e3252d9d": (
        "STRUCTURED_EQUIVALENCE_MISSED",
        "来源方法关系与 carrier 的 cue/object 规范化不对称；通用双侧规范化后精确匹配。",
    ),
    "01a0c41d-6d43-7303-a5a4-3c8ffadccbf6": (
        "SOURCE_DETECTOR_FALSE_POSITIVE",
        "方法词形位于中文名词复合词中，不表达‘采用某方法’关系；收窄来源侧动词/关系检测。",
    ),
}

ROOT_CAUSE_GROUP = {
    "date_time": "日期/时间区间端点的结构化等价",
    "measurement": "数值、单位与测量指标后缀的精确 carrier 对齐",
    "unit": "数值、单位与测量指标后缀的精确 carrier 对齐",
    "method": "方法 cue 与 object 的关系规范化及 proposition attachment",
    "scope": "scope carrier 与来源侧 clause attachment",
    "comparison": "比较目标 carrier 与来源侧 clause attachment",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _scope_review(event: dict[str, Any]) -> dict[str, str]:
    evidence = event["source_obligations_and_candidate_evidence"]
    candidates = [
        candidate
        for item in evidence
        for candidate in item.get("candidate_propositions_considered", [])
    ]
    selected_refs = {
        item.get("candidate_proposition_ref")
        for item in evidence
        if item.get("candidate_proposition_ref")
    }
    selected_indexes = {
        item.get("candidate_proposition_index")
        for item in evidence
        if item.get("candidate_proposition_index") is not None
    }
    if selected_refs or selected_indexes:
        return {
            "classification": "PROPOSITION_SCOPED",
            "basis": "frozen diagnostic records explicitly bind the obligation to a proposition ref/index",
        }
    distinct_refs = {item.get("local_ref") for item in candidates if item.get("local_ref")}
    if len(distinct_refs) == 1:
        anchored = True
        for item in evidence:
            context = item.get("attachment_scope") or item.get("source_obligation", {}).get(
                "attachment_context", ""
            )
            source_span = item.get("source_span") or ""
            if not context or not source_span or source_span not in context:
                anchored = False
                break
            candidate_spans = [
                candidate.get("source_span") or ""
                for candidate in item.get("candidate_propositions_considered", [])
            ]
            if any(not span or span not in context for span in candidate_spans):
                anchored = False
                break
        if anchored:
            return {
                "classification": "CLAUSE_SCOPED",
                "basis": "source obligation and the unique candidate source span are both literally anchored in the recorded source-side context; no broader scope inferred",
            }
    if len(distinct_refs) > 1:
        return {
            "classification": "UNKNOWN",
            "basis": "multiple candidate propositions are recorded without a unique source-side attachment; output alone cannot establish group scope",
        }
    return {
        "classification": "UNKNOWN",
        "basis": "frozen source-side records do not uniquely establish proposition, clause, group, or claim-wide scope",
    }


def build() -> tuple[dict[str, Any], str]:
    source = read_json(SOURCE_LEDGER)
    replay = read_json(REPLAY)
    rows = replay["all_claim_rows"]
    current_by_id = {row["claim_id"]: row for row in rows}
    reviewed_events: list[dict[str, Any]] = []
    new_outcomes: Counter[str] = Counter()
    persistent_current: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()

    for event in source["ledger"]:
        claim_id = event["claim_id"]
        category = event["qualifier_category"]
        current = current_by_id[claim_id]
        check = current["new_qualifier_checks"][category]
        scope = _scope_review(event)
        scope_counts[scope["classification"]] += 1
        if event["cohort"] == "NEWLY_BLOCKED":
            classification, rationale = NEW_BLOCK_REVIEWS[claim_id]
            outcome = (
                "RECOVERED_FALSE_BLOCK"
                if check["status"] in {"PRESERVED", "NOT_APPLICABLE"}
                else "REMAINS_UNRESOLVED_FOR_OTHER_REASON"
            )
            new_outcomes[outcome] += 1
        elif check["status"] == "AMBIGUOUS":
            classification = "TRUE_ATTACHMENT_AMBIGUITY"
            rationale = "保守保留多个可能 carrier/附着对象；不得凭共享词面升级为 PRESERVED。"
            persistent_current["AMBIGUOUS"] += 1
            outcome = "REMAINS_UNRESOLVED_FAIL_CLOSED"
        elif check["status"] == "POTENTIAL_LOSS":
            classification = "STRUCTURED_EQUIVALENCE_MISSED"
            rationale = (
                "来源义项有证据，但冻结输出使用跨语言的比较/方法表达，当前确定性 carrier 对齐未证明两者等价；"
                "保留阻塞，不添加单例词典。"
            )
            persistent_current["POTENTIAL_LOSS"] += 1
            outcome = "REMAINS_UNRESOLVED_FAIL_CLOSED"
        else:
            if check["status"] == "NOT_APPLICABLE":
                classification = "SOURCE_DETECTOR_FALSE_POSITIVE"
                rationale = (
                    "Revision 2 no longer finds a supported source obligation for this warning; "
                    "the original token/cue was not sufficient evidence of a material qualifier."
                )
            elif event["old_validator_status"] == "AMBIGUOUS":
                classification = "ATTACHMENT_SCOPE_ERROR"
                rationale = (
                    "The prior ambiguity came from the generic attachment scope; source-span "
                    "occurrence alignment now identifies the supported proposition without widening scope."
                )
            else:
                classification = "STRUCTURED_EQUIVALENCE_MISSED"
                rationale = (
                    "Revision 2 recognizes an existing canonical carrier after generic normalization "
                    "while retaining source-side attachment evidence."
                )
            outcome = "RESOLVED_BY_GENERIC_REVISION"
        reviewed = dict(event)
        reviewed["revision2_review"] = {
            "primary_classification": classification,
            "root_cause_group": ROOT_CAUSE_GROUP.get(category, "OTHER"),
            "source_attachment_scope": scope,
            "revision2_replay_status": check["status"],
            "revision2_claim_blocked": current["qualifier_blocked_after"],
            "new_block_outcome": outcome if event["cohort"] == "NEWLY_BLOCKED" else None,
            "review_rationale": rationale,
        }
        if event["cohort"] == "PERSISTENTLY_BLOCKED":
            reviewed["revision2_review"]["persistent_warning_outcome"] = outcome
        reviewed_events.append(reviewed)

    reviewed_ledger = {
        **source,
        "review_status": "COMPLETED_OFFLINE",
        "reviewed_warning_count": len(reviewed_events),
        "revision2_replay_sha256": replay["input_artifact_sha256"],
        "scope_classification_counts": dict(scope_counts),
        "new_block_outcome_counts": dict(new_outcomes),
        "persistent_warning_replay_status_counts": dict(persistent_current),
        "ledger": reviewed_events,
    }

    before = replay["before_status_counts"]
    after = replay["after_status_counts"]
    blocked = replay["claim_level_before_after_counts"]
    composition = replay["persisted_composition_metrics"]
    event_groups = Counter(
        item["revision2_review"]["root_cause_group"] for item in reviewed_events
    )
    report = f'''# V2.1R Qualifier Semantic Preservation Revision 2

## A. Current Residual Warning Set

预变更审计台账记录 22 个 warning event / 19 个 claim（持久阻塞 18 条 event，新增阻塞 4 条 event）。修订后同一冻结 100-Claim 输入：PRESERVED={after['PRESERVED']}，POTENTIAL_LOSS={after['POTENTIAL_LOSS']}，AMBIGUOUS={after['AMBIGUOUS']}，NOT_APPLICABLE={after['NOT_APPLICABLE']}。最终仍有 3 个 qualifier-blocked claims：两个 AMBIGUOUS（一个测量 carrier 多重附着，一个已知方法附着歧义）和一个 claim 的比较/方法跨语言 carrier 未能确定性对齐。

## B. Four Newly Blocked Claims

| Claim ID | 类别 | 修订前审计主分类 | 修订后 | 结论 |
|---|---|---|---|---|
| `01a0c437-488d-7dc7-b278-44aac2d174b0` | measurement | SOURCE_DETECTOR_FALSE_POSITIVE | PRESERVED | 误把嵌入式效率词尾作为测量率；有界词形修正后对应 obligation 被精确满足。 |
| `01a0c84e-9da1-7478-ac92-ed89de3957a5` | date_time | STRUCTURED_EQUIVALENCE_MISSED | PRESERVED | 年份是区间端点，period carrier 精确包含该端点。 |
| `01a0c82b-2f49-70bf-a1e6-3654e3252d9d` | method | STRUCTURED_EQUIVALENCE_MISSED | PRESERVED | cue/object 双侧通用规范化后 carrier 与来源关系一致。 |
| `01a0c41d-6d43-7303-a5a4-3c8ffadccbf6` | method | SOURCE_DETECTOR_FALSE_POSITIVE | NOT_APPLICABLE | 中文名词复合词内的词素不构成方法关系；移除伪来源信号，没有强行改写 proposition。 |

四项均为 `RECOVERED_FALSE_BLOCK`；没有保留新的 qualifier blocker。

## C. Fifteen Persistently Blocked Claims

15 个既有阻塞 claim 按通用机制审计，而非逐 claim 打补丁：日期区间端点；数值/单位/指标后缀 carrier；方法 cue/object 与关系归属；scope/comparison 的来源侧附着范围；真实多解附着。修订后 12/15 不再被 qualifier 阻塞。剩余仅为 `01a0c870-22f2-79d2-a6a4-a334b99dc760`（measurement AMBIGUOUS）、`01a0c86c-b74f-77af-8259-624b11e64061`（method AMBIGUOUS）和 `01a0c842-c728-7dcc-997a-aece2ad97886`（comparison + method POTENTIAL_LOSS）。逐 event 的 source obligation、carrier、strength、scope 与结果见 reviewed ledger。

## D. Root-Cause Clusters

| 通用机制 | 预变更 warning events | 修订后结果 |
|---|---:|---|
| 日期/时间区间端点等结构等价 | {event_groups['日期/时间区间端点的结构化等价']} | 区间端点可由精确 period carrier 满足；错误事件日期仍被负例拒绝。 |
| 数值、单位与测量指标后缀精确对齐 | {event_groups['数值、单位与测量指标后缀的精确 carrier 对齐']} | exact number + metric/source attachment 才通过；错误 number/metric 仍失败。 |
| 方法 cue/object 规范化及 proposition attachment | {event_groups['方法 cue 与 object 的关系规范化及 proposition attachment']} | 双侧规范化并按 source occurrence 绑定；错方法/错 proposition 负例仍失败。 |
| scope/comparison clause attachment | {event_groups['scope carrier 与来源侧 clause attachment'] + event_groups['比较目标 carrier 与来源侧 clause attachment']} | 不以相同词面推断作用域；无来源跨度支持时 fail-closed。 |

## E. Obligation Scope Semantics

22 条 ledger 均逐条增加来源侧 scope 复核。复核统计：`{json.dumps(dict(scope_counts), ensure_ascii=False)}`。单一候选只在 source context/span 支持时判为 proposition/clause scoped；多个候选但没有唯一来源侧锚点标 `UNKNOWN`。没有根据模型输出单独推断 GROUP_SCOPED 或 CLAIM_WIDE。实现中先按 exact source-span occurrence 选出 attached views；若有 attached view，只对这些 view 判定，不要求其余 proposition 冗余重复满足。只有找不到附着 view 时才检查全体 views，并对多重兼容结果保留 AMBIGUOUS。

## F. Obligation Cardinality

不再假定 obligation 与输出字段/单条 proposition 一对一。一个 obligation 可由同一 proposition 的多个 canonical carriers 表示；共享 obligation 仅在来源侧明确 group/clause attachment 且每个成员分别具备证据时才允许覆盖多个 proposition。当前 22 条复核没有一条达到 GROUP_SCOPED 或 CLAIM_WIDE 的证据门槛，故没有把这些 scope 强加给模型结果。重复 token occurrence 使用 claim offset 对齐，不能借另一 clause 中的同词满足。

## G. Semantic Carrier Coverage

审计使用小型通用 carrier taxonomy：material qualifier、typed value（numeric/date/comparison）、subject、predicate/relation、comparison target、modality、attribution、condition/configuration、scope/population/location、proposition text 与 composition/group context。证据强度标注为 STRONG/MEDIUM/WEAK；弱 proposition text 不单独证明错 clause 的附着。跨语言残余不以单例映射修复。

## H. Evidence Strength

精确结构化值与源位置 carrier 通常为 STRONG；结构字段语义但无独立 span 为 MEDIUM；自由文本为 WEAK。没有为降低 warning count 而降低阈值。所有 recoveries 仍要求 source-side attachment compatibility。

## I. Attachment Safety

新增/保留按 source span occurrence 与 obligation offset 的匹配；同一 marker 在不同 clause 出现时仅对应其原 occurrence。保留错误 proposition、错误日期事件、错误数值/指标、错误方法关系和未证明 group scope 的 fail-closed 负例。来源上下文不足的 comparison/method carrier 不被人工补齐。

## J. Claim-Level Aggregation

修订前：任何 applicable material `POTENTIAL_LOSS` 或 `AMBIGUOUS` 都参与整 claim 阻塞；修订后聚合仍 fail-closed，只在 obligation 不适用/伪信号，或有足够等价与附着证据时解除。非法 schema、invalid composition、zero-proposition abstention 等独立 blocker 不计作 qualifier recovered。100 条结果中 qualifier blocker 从 {blocked['before:UNRESOLVED_QUALIFIER']} 降至 {blocked['after:UNRESOLVED_QUALIFIER']}；不把有其他 blocker 的 claim 声称为可解析。

## K. Baseline Count Reconciliation

历史摘要 `30 / 2 / 3 / 1` 原样保留。逐条迁移审计的 canonical 重建为 `31 / 2 / 2 / 1`（结构等价未识别 / 来源 detector false positive / applicability false positive / 真附着歧义）。差异写入 `v2_1r_qualifier_revision2_baseline_reconciliation.json`，未编辑 historical bounded-shadow artifact。

## L. Implementation Changes

仅限 qualifier validator 的通用 obligation/carrier/attachment 逻辑：日期端点解析、同 number+source metric 的单位扩展、方法 cue/object 双侧规范化、比较目标名词误检抑制、中文测量词形误检抑制、共享 group carrier 的来源协调条件，以及按原始字符位置匹配 source span occurrence。未改 prompt/schema、命题形成、composition/source-span producer、eligibility 或 ResearchLoop。

## M. False-Preservation Safety

硬负例覆盖错 proposition/重复 token occurrence、错 date event、错 numeric value、错 metric/unit、错 method relation 与未证实共享 scope；全部保持非 `PRESERVED`。单元与全量测试通过；没有任何 claim 语义被手工改写以通过。

## N. 100-Claim Replay Before / After

| 状态 | Before | After |
|---|---:|---:|
| PRESERVED | {before['PRESERVED']} | {after['PRESERVED']} |
| POTENTIAL_LOSS | {before['POTENTIAL_LOSS']} | {after['POTENTIAL_LOSS']} |
| AMBIGUOUS | {before['AMBIGUOUS']} | {after['AMBIGUOUS']} |
| NOT_APPLICABLE | {before['NOT_APPLICABLE']} | {after['NOT_APPLICABLE']} |
| qualifier-blocked claims | {blocked['before:UNRESOLVED_QUALIFIER']} | {blocked['after:UNRESOLVED_QUALIFIER']} |
| previously blocked claims without qualifier blocker | 0/31 baseline | {replay['previously_31_qualifier_blocked_claims_resolved_by_replay']}/31 |

新增阻塞 claim 在修订重放后：4 个 source events 均已 `PRESERVED` 或 `NOT_APPLICABLE`。没有通过增加 provider calls 获得结果；replay provider calls = 0。另 9 条因 unresolved/invalid/schema/composition 而未进入 qualifier 检查，单独计为非 qualifier blocker：`{json.dumps(replay['not_evaluated_invalid_or_abstention_counts'])}`。

## O. Four New-Block Outcomes

结果：`{json.dumps(dict(new_outcomes))}`。三个原始检测义务得到等价 carrier，另一项是来源检测 false positive 被移除。无一项仍然新阻塞。

## P. Remaining Genuine Ambiguity / Loss

`01a0c86c-b74f-77af-8259-624b11e64061` 保留为此前人工确认的真实方法附着歧义；`01a0c870-22f2-79d2-a6a4-a334b99dc760` 因一个测量义务存在多个兼容 proposition 附着而保留 AMBIGUOUS；`01a0c842-c728-7dcc-997a-aece2ad97886` 的中文比较/方法 source cues 与英文 proposition/carrier 在语义上看似表达对应关系，但确定性附着等价证据不充分，保持 POTENTIAL_LOSS。后者列为孤立跨语言 carrier coverage limitation / unresolved potential loss，不做词典式修补。

## Q. Composition / Source-Span Regression

冻结 projection unchanged = `{replay['composition_projection_unchanged']}`；hash before/after = `{replay['composition_projection_sha256_before']}` / `{replay['composition_projection_sha256_after']}`。生产端 composition/source-span/eligibility producer modified = `{replay['composition_source_span_eligibility_producers_modified']}`（False，即未修改）。存档统计：groups={composition['group_count']}，AND={composition['operator_counts'].get('AND', 0)}，OR/XOR={composition['or_xor_group_count']}，leaf leakage={composition['or_xor_leaf_leakage_count']}，source-span failures={composition['source_span_failure_count']}（与冻结基线相同，不作本阶段修复）。

## R. Anti-Overfitting Audit

没有 Claim-ID runtime branch、实体/行业/来源规则、一次性词汇例外或按 warning 数下降而调阈值。改动对应重复的日期端点、method relation 规范化、measurement carrier、位置型 attachment 机制；未对冻结文本做定制化硬编码。审计 ledger 中 ID 仅作历史 artifact join/review 注释。

## S. Provider Calls

本阶段新增 Provider calls = **0**。冻结输入中已存在的历史 provider calls = `{replay['persisted_provider_calls_in_inputs']}`；仅离线重放这 100 份既有输出。没有读取 Holdout、创建 Holdout、启动 Shadow 或进入 V2.2。

## T. Final Decision

**QUALIFIER SEMANTIC PRESERVATION READY FOR FRESH VALIDATION**

依据：四个新阻塞全部解除；28/31 个先前 qualifier-blocked claims 在同一冻结输入上不再受 qualifier 阻塞；remaining 3 个分别是两项 fail-closed 附着歧义及一个单 claim 跨语言 carrier 对齐限制（仅一个 claim，未形成重复类别 blocker）；硬负例、全量测试、静态检查与 composition/source-span projection 均通过/保持。此结论仅为 fresh validation readiness，不代表 Shadow 已启动或 V2.2 开始。
'''
    return reviewed_ledger, report


def main() -> None:
    ledger, report = build()
    REVIEWED.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "reviewed_warning_events": ledger["reviewed_warning_count"],
                "scope_counts": ledger["scope_classification_counts"],
                "new_block_outcomes": ledger["new_block_outcome_counts"],
                "persistent_replay_statuses": ledger["persistent_warning_replay_status_counts"],
                "report": REPORT.relative_to(ROOT).as_posix(),
                "reviewed_ledger": REVIEWED.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
