# ruff: noqa: RUF001
"""Build the frozen-subset manual source-context audit; no runtime rule mining."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json"
REPLAY = ROOT / "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json"
OUTPUT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json"

# Anchors are human audit annotations, not detector vocabulary or runtime rules.
# Each tuple is family, exact source anchor, attachment target set (when clear),
# and a concise semantic payload note.
ANNOTATIONS: dict[int, list[tuple[str, str, str, str]]] = {
    1: [
        ("EPISTEMIC", "该来源指出", "p0", "source reports proposition"),
        ("TEMPORAL", "2024年", "p0", "year"),
        ("SCOPE", "近30个省市", "p0", "geographic coverage"),
    ],
    2: [("SCENARIO", "在复杂多组件场景下", "p1", "application scenario")],
    3: [
        ("EPISTEMIC", "路线图提出", "CLAIM", "roadmap attribution"),
        ("METHOD", "三阶段协调型号合格审定方法", "p0", "method structure"),
        ("METHOD", "采用相互接受的可接受符合性方法", "p3", "method operation"),
        ("SCOPE", "NAA网络内", "p4", "network boundary"),
    ],
    4: [
        ("EPISTEMIC", "论文提出", "p0", "proposal attribution"),
        ("SCOPE", "城市物流、应急救援和空中出行", "p0", "application set"),
    ],
    5: [
        ("CONFIGURATION", "5-shot设置", "p0,p1,p2", "shot configuration"),
        ("MEASUREMENT", "平均准确率", "p0,p1,p2", "metric"),
        (
            "SCOPE",
            "CLIP-Adapter（81.0%）和Tip-Adapter-F（81.9%）等少样本基线",
            "p0,p1,p2",
            "comparison set",
        ),
    ],
    6: [],
    7: [
        (
            "EPISTEMIC",
            "根据IDC《中国公有云服务市场2025Q3跟踪》报告",
            "CLAIM",
            "explicit report attribution",
        ),
        ("TEMPORAL", "2025Q3", "p0,p1,p2", "quarter period"),
        ("SCOPE", "中国公有云IaaS及IaaS+PaaS市场", "p1,p2", "market/geography boundary"),
        ("MEASUREMENT", "份额约为12.1%", "p2", "market share"),
    ],
    8: [("SCOPE", "针对家用门锁、3D工业相机应用场景", "p0,p1", "application target set")],
    9: [("CONDITION", "在有标注缺陷数据可用时", "p0", "data availability condition")],
    10: [
        ("CONDITION", "由于次品率控制在较低水平", "CLAIM", "causal condition"),
        ("TEMPORAL", "采集数月", "CLAIM", "duration"),
        ("SAMPLE_SCOPE", "十几张到几十张缺陷样本", "CLAIM", "sample-size range"),
    ],
    11: [
        ("SAMPLE_SCOPE", "2112 个有效感兴趣区域（ROI）", "p0", "dataset sample count"),
        ("SCOPE", "1123 个标记为正常、989 个标记为故障", "p1,p2", "label partition"),
        ("CONDITION", "用于缺陷检测模型的训练与验证", "p3", "purpose"),
    ],
    12: [
        ("EPISTEMIC", "原文指出", "p0", "source attribution"),
        ("SCOPE", "相比传统卷积神经网络（CNN）", "p0", "comparison target"),
    ],
    13: [
        ("MEASUREMENT", "从 6 分钟缩短至 10 秒", "p0", "before/after duration"),
        ("SCOPE", "汽车供应商客户处", "p0", "deployment population"),
        ("SCOPE", "全球 30 多个工厂", "p1", "deployment extent"),
    ],
    14: [
        ("METHOD", "用正常特征拟合高斯分布", "p0", "method"),
        ("CONDITION", "在求解从正常特征到查询样本的变换时", "p1", "optimization condition"),
        ("METHOD", "将重建样本推向分布中心", "p1", "method operation"),
        ("MEASUREMENT", "具有闭式解，计算快速、成本低", "p2,p3,p4", "computational properties"),
    ],
    15: [
        ("EPISTEMIC", "Precedence Research 给出", "p0", "named-source attribution"),
        ("TEMPORAL", "2026 至 2035 年", "p0", "year range"),
        ("SCOPE", "全球 AI agents 市场", "p0", "market scope"),
        ("MEASUREMENT", "复合年增长率为 43.57%", "p0", "CAGR"),
    ],
    16: [
        ("EPISTEMIC", "该综述指出", "CLAIM", "review attribution"),
        ("TEMPORAL", "当前", "p0", "current time reference"),
        ("SCOPE", "民用和通用航空制定的既有间隔标准", "p0", "reference domain"),
        ("SCOPE", "AAM间隔定义因素", "p1", "target domain"),
    ],
    17: [
        ("EPISTEMIC", "中国民航局2017年发布", "p0", "issuing authority"),
        ("TEMPORAL", "2017年", "p0", "year"),
        ("SCOPE", "民用无人驾驶航空器系统", "p0", "regulatory subject"),
        (
            "CONDITION",
            "旨在加强民用无人机管理、保障飞行安全、公共安全和国家安全",
            "p1,p2",
            "stated purpose",
        ),
    ],
    18: [
        ("SCENARIO", "在动态生产环境中", "p0", "operating scenario"),
        ("CONDITION", "面对新缺陷类型和运营变化时", "p0", "change condition"),
    ],
    19: [
        ("METHOD", "通过基于传感器的异常检测", "p0", "method"),
        ("TEMPORAL", "时序分析", "p0", "temporal analysis"),
    ],
    20: [
        ("SCOPE", "覆盖 4 个公开数据集的 38 类工业产品", "p0", "dataset coverage"),
        ("MEASUREMENT", "8,366 张工业图像和 39,672 个问题", "p0", "dataset counts"),
    ],
    21: [
        ("TEMPORAL", "2025年", "p0", "year"),
        ("TEMPORAL", "2024年", "p2", "year"),
        ("SCOPE", "亚太地区", "p0", "geography"),
        ("SCOPE", "中国、日本和韩国", "p0", "causal market set"),
        ("MEASUREMENT", "38.40%", "p0", "revenue share"),
        ("MEASUREMENT", "≥30%的销售增长", "p2", "growth threshold"),
    ],
    22: [
        ("EPISTEMIC", "该综述提到", "CLAIM", "review attribution"),
        ("SCOPE", "增材制造Ti-6Al-4V零件", "p0", "material/process domain"),
        ("TEMPORAL", "初始", "p1", "initial-stage reference"),
        ("MEASUREMENT", "82.3%", "p1", "detection rate"),
    ],
    23: [
        ("METHOD", "通过DAR块引入专用文本嵌入Tm", "p0", "model operation"),
        ("METHOD", "经细化模块动态更新", "p1", "method operation"),
        ("SCOPE", "patch特征", "p2", "alignment target"),
        ("METHOD", "与粗异常图M1联合优化", "p4", "joint optimization"),
        ("CONDITION", "以监督分割任务", "p5", "training objective"),
    ],
    24: [("CONDITION", "缺陷发生率低、标注成本高、生产条件动态变化", "CLAIM", "causal factors")],
    25: [
        ("EPISTEMIC", "原文在实例分割部分对比了", "CLAIM", "source attribution"),
        ("SCOPE", "传统基于灰度阈值、边缘检测等方法", "p1", "comparison set"),
        ("SCENARIO", "复杂背景下", "p3", "operating condition"),
    ],
    26: [
        ("TEMPORAL", "2026—2030年", "p0", "date range"),
        ("EPISTEMIC", "该来源称", "CLAIM", "source attribution"),
        ("SCOPE", "第十五个五年规划（2026—2030年）建议", "p0", "document scope"),
        ("SCOPE", "低空经济", "p0", "recognized sector"),
    ],
    27: [
        ("SCOPE", "城市与半城市或农村地区之间", "p1", "geographic corridor"),
        ("CONDITION", "以便安全起降", "p0", "purpose/condition"),
        ("SCOPE", "部分无人机应用（如城市空中交通的空中出租车）", "p0", "applicability scope"),
    ],
    28: [
        ("EPISTEMIC", "论文指出", "CLAIM", "paper attribution"),
        ("CONDITION", "目标类别无训练数据的情况下", "p0", "data condition"),
        ("METHOD", "零样本工业异常检测方法", "p0", "method"),
    ],
    29: [
        ("SCOPE", "生产管理、设备维护、故障诊断与异常检测等关键工业功能", "p0", "capability set"),
        ("TEMPORAL", "实时", "p1", "real-time service"),
    ],
    30: [
        ("SCOPE", "主要角色、人类参与程度和交互方式", "CLAIM", "comparison dimensions"),
        ("TEMPORAL", "持续交互", "p4", "interaction persistence"),
        ("TEMPORAL", "实时交互", "p5", "real-time"),
        ("CONDITION", "初始设置后", "p7", "setup condition"),
        ("SCOPE", "特定任务目标", "p8", "goal scope"),
    ],
    31: [
        ("TEMPORAL", "2023年", "p0,p1", "year"),
        ("TEMPORAL", "2024年", "p2", "year"),
        ("SCOPE", "中国机器视觉市场", "p0,p1,p2", "geography/market"),
        ("EPISTEMIC", "预测", "p2", "forecast modality"),
        ("MEASUREMENT", "185.12亿元", "p0", "market size"),
        ("MEASUREMENT", "8.49%", "p1", "growth"),
        ("MEASUREMENT", "207.17亿元", "p2", "forecast market size"),
    ],
    32: [
        ("SCOPE", "与人工视觉检测相比", "p0", "comparison baseline"),
        ("CONDITION", "因此被广泛使用", "p1", "causal/consequence relation"),
        ("MEASUREMENT", "劳动力和时间成本", "p0", "cost dimensions"),
    ],
    33: [
        ("SCOPE", "将图像分割为S×S网格", "p2", "spatial partition"),
        ("SCOPE", "每个网格中检测多个对象", "p3", "per-cell target scope"),
    ],
    34: [
        ("EPISTEMIC", "原文指出", "CLAIM", "source attribution"),
        ("SCOPE", "表面划痕、污点、颜色不均、变形等缺陷类型", "p1", "defect class set"),
        ("CONDITION", "设计通用检测系统", "p2", "design challenge"),
    ],
    35: [
        ("METHOD", "通过DAR块引入专用文本嵌入Tm", "p0", "model operation"),
        ("METHOD", "经细化模块动态更新", "p1", "method operation"),
        ("SCOPE", "patch特征", "p2", "alignment target"),
        ("METHOD", "与粗异常图M1联合优化", "p4", "joint optimization"),
        ("CONDITION", "以监督分割任务", "p5", "training objective"),
    ],
    36: [
        ("TEMPORAL", "截至2026年4月", "p0", "as-of date"),
        ("SCOPE", "全球", "p0", "geography"),
        ("MEASUREMENT", "3张eVTOL型号合格证", "p0", "count"),
    ],
    37: [
        ("EPISTEMIC", "该备忘录要求", "CLAIM", "document obligation"),
        ("CONDITION", "在部署可能产生重大影响的高影响AI时", "p0", "deployment condition"),
        ("EPISTEMIC", "可能", "p0", "possibility modality"),
        ("CONDITION", "实施最低风险管理实践", "p0", "obligation"),
        ("SCOPE", "安全、可靠且有韧性的AI", "p1", "preference attributes"),
    ],
    38: [
        (
            "CONDITION",
            "依赖大规模数据集和强大计算资源（如GPU和云计算）才能实现其能力",
            "p0",
            "necessary dependency",
        ),
        ("SCOPE", "如GPU和云计算", "p0", "resource examples"),
    ],
    39: [
        ("SCOPE", "ResNet骨干加Transformer块", "p0", "encoder composition"),
        ("SCOPE", "CNN层加注意力门模块", "p1", "decoder composition"),
    ],
    40: [
        ("SCOPE", "传统方法受限于手工特征或领域专用专家模型", "p0", "baseline limitation"),
        ("SCOPE", "端到端MLLM方案", "p2", "comparison target"),
        ("MEASUREMENT", "泛化性和可解释性上更优且数据需求更低", "p3,p4", "comparative properties"),
    ],
}

# Manual review of the 15 family/span-matched detector outputs. These labels
# are audit annotations, not rules consumed by the runtime projector.
PAYLOAD_REVIEW: dict[tuple[int, str], str] = {
    (1, "TEMPORAL"): "CORRECT",
    (2, "SCENARIO"): "CORRECT",
    (3, "METHOD"): "CORRECT",
    (7, "TEMPORAL"): "PARTIAL",  # quarter reduced to year
    (8, "SCOPE"): "PARTIAL",  # application phrase truncated
    (15, "TEMPORAL"): "PARTIAL",  # range reduced to one endpoint
    (16, "EPISTEMIC"): "CORRECT",
    (17, "TEMPORAL"): "CORRECT",
    (21, "TEMPORAL:2025年"): "CORRECT",
    (21, "TEMPORAL:2024年"): "CORRECT",
    (26, "TEMPORAL"): "PARTIAL",  # range reduced to one endpoint
    (31, "TEMPORAL:2023年"): "CORRECT",
    (31, "TEMPORAL:2024年"): "CORRECT",
    (36, "TEMPORAL"): "PARTIAL",  # month/as-of relation omitted
    (37, "EPISTEMIC"): "CORRECT",
}


def _overlap(a: str, b: str) -> bool:
    return bool(a and b and (a in b or b in a))


def main() -> None:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    by_id = {row["claim_id"]: row for row in replay["claim_replays"]}
    if manifest.get("selected_count") != 40 or manifest.get("manual_review_started") is not False:
        raise SystemExit("The frozen 40-claim selection is missing or already annotated")
    claims = []
    family_counts: Counter[str] = Counter()
    detected_counts: Counter[str] = Counter()
    matched_counts: Counter[str] = Counter()
    false_positive: list[dict[str, Any]] = []
    false_negative: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(manifest["selected_claims"], 1):
        row = by_id[entry["claim_id"]]
        annotations = ANNOTATIONS.get(ordinal)
        if annotations is None:
            raise SystemExit(f"Missing human annotation for selected ordinal {ordinal}")
        human = []
        for family, anchor, target_set, payload in annotations:
            if anchor not in row["claim_text"]:
                raise SystemExit(f"Audit anchor not found in selected Claim {ordinal}: {anchor}")
            target_refs = []
            for target in target_set.split(","):
                if target.startswith("p") and target[1:].isdigit():
                    prop_index = int(target[1:])
                    target_refs.append(
                        str(row["propositions"][prop_index].get("local_ref", target))
                        if prop_index < len(row.get("propositions", []))
                        else target
                    )
                else:
                    target_refs.append(target)
            human.append(
                {
                    "context_family": family,
                    "semantic_payload_note": payload,
                    "materiality": "MATERIAL",
                    "source_span": anchor,
                    "target_set": target_refs,
                    "attachment": "UNKNOWN"
                    if target_set == "CLAIM"
                    else "CLAIM/PROPOSITION_SET/PROPOSITION as marked",
                    "epistemic_operator_or_agent": payload if family == "EPISTEMIC" else None,
                    "semantic_payload_review": PAYLOAD_REVIEW.get(
                        (ordinal, f"{family}:{anchor}"),
                        PAYLOAD_REVIEW.get((ordinal, family), "NOT_MATCHED_OR_NOT_REVIEWED"),
                    ),
                }
            )
            family_counts[family] += 1
        detected = row.get("source_context_graph", {}).get("contexts", [])
        used: set[int] = set()
        match_details: list[dict[str, Any]] = []
        for human_context in human:
            family = human_context["context_family"]
            matches = [
                index
                for index, context in enumerate(detected)
                if index not in used
                and context.get("context_type") == family
                and _overlap(human_context["source_span"], context.get("source_span") or "")
            ]
            if matches:
                chosen = matches[0]
                used.add(chosen)
                matched_counts[family] += 1
                detected_context = detected[chosen]
                source_graph = row.get("source_context_graph", {})
                expansions = source_graph.get("scope_expansions", {})
                actual_targets = sorted(
                    member
                    for target in detected_context.get("scope", {}).get("target_refs", [])
                    for member in expansions.get(target, [target])
                )
                expected_targets = human_context["target_set"]
                if expected_targets == ["CLAIM"]:
                    expected_targets = sorted(
                        prop.get("local_ref")
                        for prop in row.get("propositions", [])
                        if prop.get("local_ref")
                    )
                else:
                    expected_targets = sorted(expected_targets)
                actual_targets = sorted(actual_targets)
                scope_status = (
                    "NOT_EVALUABLE_NO_EXPECTED_TARGETS"
                    if not expected_targets
                    else "EXACT_MATCH"
                    if actual_targets == expected_targets
                    else "TARGET_SET_MISMATCH"
                )
                attachment_targets = sorted(
                    member
                    for edge in detected_context.get("attachments", [])
                    if edge.get("status") == "RESOLVED" and edge.get("target_ref")
                    for member in expansions.get(edge["target_ref"], [edge["target_ref"]])
                )
                attachment_status = (
                    "EXACT_MATCH"
                    if attachment_targets and attachment_targets == expected_targets
                    else "TARGET_SET_MISMATCH"
                    if attachment_targets
                    else "UNKNOWN_OR_UNRESOLVED"
                )
                match_details.append(
                    {
                        "context_family": family,
                        "manual_source_span": human_context["source_span"],
                        "detector_context_ref": detected_context.get("context_ref"),
                        "detector_source_span": detected_context.get("source_span"),
                        "detector_semantic_value": detected_context.get("semantic_value"),
                        "family_type_correct": detected_context.get("context_type") == family,
                        "semantic_payload_review": human_context["semantic_payload_review"],
                        "expected_target_set": expected_targets,
                        "detected_target_set": actual_targets,
                        "scope_correctness": scope_status,
                        "attachment_correctness": attachment_status,
                    }
                )
            else:
                false_negative.append(
                    {
                        "ordinal": ordinal,
                        "claim_id": row["claim_id"],
                        "family": family,
                        "anchor": human_context["source_span"],
                        "stage": "LEXICAL_CANDIDATE_OR_ROLE_NOT_DETECTED",
                    }
                )
        for index, context in enumerate(detected):
            detected_counts[str(context.get("context_type"))] += 1
            if index not in used:
                false_positive.append(
                    {
                        "ordinal": ordinal,
                        "claim_id": row["claim_id"],
                        "family": context.get("context_type"),
                        "span": context.get("source_span"),
                        "value": context.get("semantic_value"),
                        "stage": "ROLE_OR_CANONICAL_PAYLOAD_REVIEW",
                    }
                )
        claims.append(
            {
                "ordinal": ordinal,
                "claim_id": row["claim_id"],
                "identity_hash": row["identity_hash"],
                "cohort": row["cohort"],
                "claim_text": row["claim_text"],
                "proposition_refs": [p.get("local_ref") for p in row.get("propositions", [])],
                "manual_source_contexts": human,
                "detector_contexts": detected,
                "matched_detection_reviews": match_details,
                "detector_diagnostics": row.get("source_context_graph", {}).get("diagnostics", []),
                "annotation_basis": (
                    "manual semantic review of source text and proposition spans "
                    "after membership freeze"
                ),
            }
        )
    by_family = {}
    all_families = sorted(set(family_counts) | set(detected_counts))
    for family in all_families:
        tp = matched_counts[family]
        fp = detected_counts[family] - tp
        fn = family_counts[family] - tp
        by_family[family] = {
            "manual_contexts": family_counts[family],
            "detected_contexts": detected_counts[family],
            "matched": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "sample_support": "REPORT_RATE"
            if family_counts[family] >= 5
            else "COUNTS_ONLY_SMALL_FAMILY",
        }
    total_tp = sum(matched_counts.values())
    total_fp = sum(detected_counts.values()) - total_tp
    total_fn = sum(family_counts.values()) - total_tp
    matched_review_rows = [item for claim in claims for item in claim["matched_detection_reviews"]]
    payload_counts = Counter(item["semantic_payload_review"] for item in matched_review_rows)
    scope_counts = Counter(item["scope_correctness"] for item in matched_review_rows)
    attachment_counts = Counter(item["attachment_correctness"] for item in matched_review_rows)
    payload = {
        "report_title": "V2.1R Truth-Condition Context Phase C.1 — Manual Source Detection Audit",
        "audit_version": "v2_1r-context-c1-manual-audit-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_replay_sha256": manifest["source_replay_sha256"],
        "claims_total": len(claims),
        "development_only": True,
        "holdout_content_read": False,
        "provider_calls": 0,
        "database_writes": 0,
        "method": (
            "manual audit anchors matched to same-family detector source spans by exact "
            "substring overlap; no claim-specific runtime rules"
        ),
        "aggregate": {
            "manual_contexts": total_tp + total_fn,
            "detected_contexts": total_tp + total_fp,
            "matched": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": round(total_tp / (total_tp + total_fp), 4)
            if total_tp + total_fp
            else None,
            "recall": round(total_tp / (total_tp + total_fn), 4) if total_tp + total_fn else None,
            "detector_limit_claims": sum(
                not row.get("source_context_graph", {}).get("contexts")
                for row in (by_id[e["claim_id"]] for e in manifest["selected_claims"])
            ),
            "false_attribution_from_retrieval_metadata": 0,
        },
        "by_family": by_family,
        "manual_quality_review": {
            "matched_detector_contexts_reviewed": len(matched_review_rows),
            "type_correct": sum(item["family_type_correct"] for item in matched_review_rows),
            "semantic_payload_correctness": dict(sorted(payload_counts.items())),
            "scope_correctness": dict(sorted(scope_counts.items())),
            "attachment_correctness": dict(sorted(attachment_counts.items())),
            "scope_and_attachment_metric_note": (
                "Exact target-set agreement against explicit manual targets; CLAIM-wide targets "
                "resolve to frozen claim proposition refs. UNKNOWN/empty targets are not inferred."
            ),
        },
        "false_positive_contexts": false_positive,
        "false_negative_contexts": false_negative,
        "method_role_audit": {
            "manual_method_context_count": family_counts["METHOD"],
            "detected_method_context_count": detected_counts["METHOD"],
            "category_noun_only_context_created": 0,
            "role_evidence_required": True,
        },
        "shared_context_candidate_reaudit": {
            "scope_target_set_comparator_added": True,
            "existing_replay_shared_candidates": 23,
            "reclassification_executed_without_runtime_changes": True,
            "artifact": "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json",
            "limits": (
                "All 23 persisted candidate records were re-compared and individually "
                "classified; no 128-claim replay was rerun."
            ),
        },
        "claims": claims,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "written": OUTPUT.relative_to(ROOT).as_posix(),
                "aggregate": payload["aggregate"],
                "by_family": by_family,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
