# ruff: noqa: E501, RUF001
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v2_1r_d4e_extraction_completeness"
LEDGER_PATH = (
    ROOT / "artifacts/v2_1r_d4d_ownership_alignment/remaining_extraction_failure_ledger.json"
)
AGGREGATE_PATH = ROOT / "artifacts/v2_1r_d4b_execution_attempt_2/aggregate.json"

CLUSTER = {
    "EPISTEMIC_ROLE_TARGET": "RC1",
    "METHOD_PURPOSE_OWNERSHIP": "RC2",
    "SOURCE_FACT_ABSENT": "RC3",
    "MEASUREMENT_RELATION_BASIS": "RC4",
    "ASSERTION_SEGMENTATION": "RC5",
    "CAUSAL_RELATION_EXPRESSIVENESS": "RC6",
    "BINDING_COMPLETENESS": "RC7",
}

FACT_AUDIT: dict[str, dict[str, str]] = {
    "EPISTEMIC_OPERATOR": {
        "primary": "SCHEMA_CAN_EXPRESS_MODEL_OMITTED",
        "cluster": "RC1",
        "expected": "报告来源所表达的认识/报告算子，指向承载被报告内容的实质命题，而不是把算子证据绑定到 ASSERTION 节点本身。",
        "evidence": "Claim 01a0c82b…bff: semantic context C1 缺失；B2 把 EPISTEMIC_OPERATOR 绑定到 ASSERTION/A1。v2 有 EPISTEMIC Context、role 和 CONTEXT target。",
        "secondary": "PROMPT_CONTRACT_AMBIGUITY：提示要求分离 operator/agent/modality/target，但未定义 operator 的 target_kind/ref 所有权。",
    },
    "EPISTEMIC_AGENT": {
        "primary": "SCHEMA_CAN_EXPRESS_MODEL_OMITTED",
        "cluster": "RC1",
        "expected": "若原文明确归属给来源主体，则记录该来源主体，并关联到被归属的命题；不能由检索来源推断 agent。",
        "evidence": "同 Claim: C1 只有 attribution 字段；B3 将 EPISTEMIC_AGENT 绑定至 ASSERTION/A1，产生 agent_not_source_grounded。原文的泛称来源提示不是额外具名作者。",
        "secondary": "除错误 target 外，不具名的 source agent 需要保守表示；不能合成作者实体。",
    },
    "METHOD_RELATION": {
        "primary": "SCHEMA_CAN_EXPRESS_MODEL_OMITTED",
        "cluster": "RC2",
        "expected": "方法 M 与其操作/对象/目标之间由原文支持的关系，且关系证据与 METHOD_VALUE 同 owner 关联。",
        "evidence": "Claim 01a0c41c…b3de: C1/C2 有 METHOD 值，bindings 只有 METHOD_VALUE，无 METHOD_RELATION；validator 已定义 related_binding_ref 配对规则。",
        "secondary": "D4-D 审计中的 NOT_EVALUATED 是缺少关系绑定，不证明 validator 错误。",
    },
    "OTHER": {
        "primary": "PROMPT_UNDERSPECIFIED",
        "cluster": "RC2",
        "expected": "将方法的用途/目的表达为有证据的 purpose 语义，而非把目的误分类为其他上下文。",
        "evidence": "同 Claim: C3.semantic_value 已包含 purpose，但 context_type=OTHER、binding role=CONTEXT_VALUE。值被保留，语义 owner/type 不明确。",
        "secondary": "schema 可用开放 semantic_value + OTHER 兜底承载，但没有专门 purpose role；属于有限表达，不足以认定绝对 schema 不可表达。",
    },
    "MEASUREMENT_DEFINITION": {
        "primary": "GENUINE_SOURCE_AMBIGUITY",
        "cluster": "RC3",
        "expected": "只有来源实际定义了指标构成/计算方法时才表示 formal definition；否则仅保留指标身份与结果值。",
        "evidence": "Claim 01a0c434…9bf2 只报告 MVTec AD 上 AUROC=0.992；模型同时发出 MEASUREMENT_VALUE 与 MEASUREMENT_DEFINITION，D4-D 对 formal definition 正确判 NOT_SUPPORTED。",
        "secondary": "SOURCE_FACT_ABSENT；不应把正确 fail-closed 计为 extractor/validator bug。",
    },
    "MEASUREMENT_ASSOCIATION": {
        "primary": "SCHEMA_REPRESENTATION_GAP",
        "cluster": "RC4",
        "expected": "测量结果与其适用性/兼容性结论之间保留明确关系，而不把 compatibility 压平为数值字段。",
        "evidence": "Claim 01a0c424…5d85 的 56.87 ms/inference 被提取；关于 live-production compatibility 的独立语义关系未建模。当前 attachment enum 仅 APPLIES_TO/CONDITIONS/ATTRIBUTES/UNKNOWN，且 attachment 指向 context 与 assertions，不能明确表达该关系类型。",
        "secondary": "可用 OTHER semantic_value 暂存文字，但无规范化 relation owner/role，属于真实契约缺口。",
    },
    "MEASUREMENT_VALUE": {
        "primary": "SCHEMA_CAN_EXPRESS_MODEL_OMITTED",
        "cluster": "RC4",
        "expected": "将数值、单位及 per-inference 的计量基准/分母一并保留，并区分数值单位与测量基础。",
        "evidence": "同 Claim: C3 semantic_value 有 value=56.87、unit=milliseconds；词面和 evidence span 明确含 per inference，但 typed semantic_value 未存 basis。v2 使用 Any-valued semantic_value，MEASUREMENT context 和同一 evidence binding 可承载 basis。",
        "secondary": "没有独立 BASIS binding role，规范化校验能力有限；属 schema limitation，但本例 canonical payload 可表达。",
    },
    "ASSERTION_CONTENT": {
        "primary": "ASSERTION_SEGMENTATION_DEFECT",
        "cluster": "RC5",
        "expected": "每个 assertion anchor 对应完整的实质命题分句；归属导语不是 assertion 内容本身。",
        "evidence": "Claim 01a0c436…acec: A1 evidence E1 仅为归属短语；A2/A3 spans 与 assertion evidence refs 不稳定对应实质分句，尽管每个 span 均 exact substring。",
        "secondary": "EVIDENCE_ALIGNMENT_DEFECT 是下游症状；文本存在性不等于语义对齐。",
    },
    "CAUSAL_RELATION": {
        "primary": "SCHEMA_REPRESENTATION_GAP",
        "cluster": "RC6",
        "expected": "两个命题之间的 cause/result 有向关系及对应文本证据。",
        "evidence": "同 Claim 的结果/因果子句未落入显式关系。v2 attachment 是 context→assertion 的边，relation enum 不含 CAUSES/RESULTS_IN；RELATION_OPERATOR binding 也没有 relation-node target kind。",
        "secondary": "分句错位加重可见性，但即使分段正确，当前 canonical graph 也没有明确的 assertion-to-assertion relation 节点。",
    },
    "CONDITION": {
        "primary": "SCHEMA_CAN_EXPRESS_MODEL_OMITTED",
        "cluster": "RC5",
        "expected": "区分方法的对象/例示输入与真正限定 assertion 成立条件的条件子句。",
        "evidence": "同 Claim C3/C4 被抽成 CONDITION，D4-C verifier 对这些 role interpretation 判 NOT_SUPPORTED；source spans 本身有效，但归属 assertion/语义角色不足以支持条件解释。",
        "secondary": "ASSERTION_SEGMENTATION_DEFECT；不应因 validator 拒绝无支持的 role 而放宽校验。",
    },
    "SCOPE": {
        "primary": "ASSERTION_SEGMENTATION_DEFECT",
        "cluster": "RC5",
        "expected": "scope node 的成员、scope evidence 与各实质 assertion provenance 一致。",
        "evidence": "同 Claim S1 为 CLAIM scope，成员 A1/A2/A3，但 evidence 仅 E1 attribution span；D4-D 保持 invalid graph fail-closed。",
        "secondary": "错误 anchor/evidence ownership 导致 membership provenance 失配；D4-D 已处理的 downstream canonical ownership 不纳入本条根因。",
    },
    "ATTACHMENT": {
        "primary": "BINDING_GENERATION_DEFECT",
        "cluster": "RC7",
        "expected": "有语义证据的 resolved shared attachment 至少有一个指向该 attachment 的 ATTACHMENT binding，覆盖其合法多目标边。",
        "evidence": "Claim 01a0c853…860d: AT1 为 RESOLVED，目标 A1/A2/A3，带 E1/E2/E3；binding 中只有 AT2-AT4，无 AT1。validator 已要求 direct binding 或严格匹配的 scope-structure binding。",
        "secondary": "通用 prompt 已要求每个 role/target 明确 binding；该 omission 是 binding completeness defect，不是多目标语义误判。",
    },
}

SCHEMA_ROWS = [
    (
        "ASSERTION",
        "SUPPORTED_DIRECTLY",
        "SourceAssertionV2 + evidence_refs/status；锚点语义边界规则未充分明确。",
    ),
    (
        "EPISTEMIC_OPERATOR",
        "SUPPORTED_DIRECTLY",
        "EPISTEMIC context + semantic role + CONTEXT target；operator 到内容 owner 的链接约定欠清晰。",
    ),
    (
        "EPISTEMIC_AGENT",
        "SUPPORTED_DIRECTLY",
        "EPISTEMIC semantic_value 与 EPISTEMIC_AGENT binding；仅可表达文本有据的 agent。",
    ),
    (
        "EPISTEMIC_TARGET",
        "SUPPORTED_WITH_LIMITATION",
        "可经 EPISTEMIC semantic_value 和 context→assertion attachment表达，但没有独立 target binding role。",
    ),
    ("MODALITY", "SUPPORTED_DIRECTLY", "EPISTEMIC context + MODALITY role。"),
    ("METHOD_VALUE", "SUPPORTED_DIRECTLY", "METHOD context + METHOD_VALUE binding。"),
    (
        "METHOD_RELATION",
        "SUPPORTED_DIRECTLY",
        "METHOD_RELATION binding + related_binding_ref 指向同 owner 的 METHOD_VALUE。",
    ),
    (
        "METHOD_PURPOSE",
        "SUPPORTED_WITH_LIMITATION",
        "开放 semantic_value 可承载 purpose；没有专门 context type/role，依赖通用 CONTEXT_VALUE。",
    ),
    (
        "MEASUREMENT_IDENTITY",
        "SUPPORTED_DIRECTLY",
        "MEASUREMENT semantic_value.metric，可与 definition role 按 D4-D 契约区分。",
    ),
    (
        "MEASUREMENT_DEFINITION",
        "SUPPORTED_DIRECTLY",
        "存在 MEASUREMENT_DEFINITION role；仅当 source span 实际支持定义。",
    ),
    (
        "MEASUREMENT_VALUE",
        "SUPPORTED_DIRECTLY",
        "MEASUREMENT_VALUE role + typed/open semantic value。",
    ),
    ("MEASUREMENT_UNIT", "SUPPORTED_DIRECTLY", "MEASUREMENT_UNIT role。"),
    (
        "MEASUREMENT_COMPARATOR",
        "SUPPORTED_DIRECTLY",
        "MEASUREMENT_COMPARATOR role；阈值/比较对象的连接仍依赖语义 payload。",
    ),
    (
        "MEASUREMENT_SCOPE_CATEGORY_ASSOCIATION",
        "SUPPORTED_THROUGH_CANONICAL_ATTACHMENT",
        "context attachment 可关联 assertion；category 与 measurement 的精确关系无专用关系枚举。",
    ),
    ("CONFIGURATION", "SUPPORTED_DIRECTLY", "CONFIGURATION context + CONFIGURATION_VALUE role。"),
    ("CONDITION", "SUPPORTED_DIRECTLY", "CONDITION context/role + CONDITIONS attachment。"),
    (
        "SCOPE",
        "SUPPORTED_DIRECTLY",
        "SourceScopeV2 图结构 + SCOPE context/SCOPE_VALUE；结构 membership 不自动证明语义值。",
    ),
    (
        "SCOPE_MEMBERSHIP",
        "SUPPORTED_DIRECTLY",
        "SourceScopeV2.source_assertion_refs，结构上可表达，仍须对齐 source evidence。",
    ),
    ("ATTACHMENT", "SUPPORTED_DIRECTLY", "SourceAttachmentV2 + ATTACHMENT binding。"),
    (
        "SHARED_ATTACHMENT",
        "SUPPORTED_THROUGH_CANONICAL_ATTACHMENT",
        "source_assertion_refs 允许多目标；需每个成员均有证据且 attachment binding 完整。",
    ),
    (
        "CAUSAL_RELATION",
        "NOT_SUPPORTED",
        "当前边只能从 context 指向 assertion；没有 assertion 间关系对象/target kind。",
    ),
    ("RESULT_RELATION", "NOT_SUPPORTED", "与因果同类，缺少有向 assertion 间关系表示。"),
    (
        "COMPATIBILITY_APPLICABILITY_RELATION",
        "SUPPORTED_WITH_LIMITATION",
        "APPLIES_TO 可表达一般适用性；不能忠实区分 compatibility 与普通 scope applicability。",
    ),
    (
        "ENUMERATION",
        "SUPPORTED_WITH_LIMITATION",
        "开放 semantic_value 可保存 enumeration，但成员/量词/共享关系缺少专用结构。",
    ),
    (
        "OTHER_RELATIONAL_SEMANTICS",
        "AMBIGUOUS_CONTRACT",
        "OTHER + 任意 JSON 值可容纳表面数据，但未定义 relation owner、方向和 binding 语义。",
    ),
]

PROMPT_ROWS = [
    (
        "assertion anchor must be substantive proposition, not attribution lead-in",
        "MISSING_INSTRUCTION",
        "提示要求 exact spans 和 conservative extraction，但未定义 assertion anchor 的实质内容边界。",
    ),
    (
        "epistemic operator/agent/modality/target distinction",
        "IMPLICIT_ONLY",
        "明确要求分离四项，但没有明确 operator/agent 的 target_kind、target_ref 所有权。",
    ),
    (
        "method relation distinct from method noun/value",
        "EXPLICIT_AND_IGNORED_BY_MODEL",
        "提示明确要求分离关系和值；对应 Claim 仅生成 METHOD_VALUE。",
    ),
    (
        "purpose semantic owner",
        "MISSING_INSTRUCTION",
        "没有说明 purpose 应进入何种 canonical context/role。",
    ),
    (
        "binding for every semantic role and target",
        "EXPLICIT_AND_IGNORED_BY_MODEL",
        "通用要求明确；D4-D 的 shared AT1 仍无 binding。",
    ),
    (
        "resolved shared attachment binding completeness",
        "IMPLICIT_ONLY",
        "要求 explicit binding per role/target，但未把 resolved semantic edge→binding 写成结构不变量。",
    ),
    (
        "shared context member support",
        "EXPLICIT_AND_IGNORED_BY_MODEL",
        "明确要求仅在 source structure 支持时保留 shared context。",
    ),
    (
        "assertion segmentation and clause-to-evidence ownership",
        "MISSING_INSTRUCTION",
        "没有要求锚点覆盖 substantive predicate 或规定 attribution/evidence 分离。",
    ),
    (
        "causal/result relation ownership",
        "MISSING_INSTRUCTION",
        "禁止编造关系，但未说明可用图结构或何时必须 unresolved。",
    ),
    (
        "compatibility vs measurement/applicability",
        "MISSING_INSTRUCTION",
        "未说明这些 relation 的 canonical distinction。",
    ),
    (
        "metric definition must not be inferred from result value",
        "IMPLICIT_ONLY",
        "不编造事实原则隐含要求；缺少 definition/value 例示约束。",
    ),
    (
        "measurement basis/denominator preservation",
        "MISSING_INSTRUCTION",
        "无显式计量基准字段/绑定说明。",
    ),
    (
        "scope membership evidence must cover members",
        "IMPLICIT_ONLY",
        "要求 scope 有 evidence，但未明确成员及 provenance 一致性规则。",
    ),
]

REVISION_ROWS = [
    {
        "cluster_id": "RC1",
        "primary_failures": ["EPISTEMIC_OPERATOR", "EPISTEMIC_AGENT"],
        "recommendation": "PROMPT_ONLY_REVISION",
        "why": "schema 已有 EPISTEMIC context/roles/target kinds；澄清 operator/agent 绑定到 context owner，并把其与被归属 assertion 的关联用 attachment 表达。agent 无文本支持时保守省略。",
    },
    {
        "cluster_id": "RC2",
        "primary_failures": ["METHOD_RELATION", "OTHER"],
        "recommendation": "PROMPT_ONLY_REVISION",
        "why": "METHOD_RELATION 的显式 binding/link 已可表达；补充 purpose 的通用 owner/role 约定，勿新增 Claim 特定词典。",
    },
    {
        "cluster_id": "RC3",
        "primary_failures": ["MEASUREMENT_DEFINITION"],
        "recommendation": "NO_REVISION_SOURCE_FACT_ABSENT",
        "why": "不从 AUROC 结果推导正式定义；当前拒绝为正确 fail-closed。",
    },
    {
        "cluster_id": "RC4",
        "primary_failures": ["MEASUREMENT_ASSOCIATION", "MEASUREMENT_VALUE"],
        "recommendation": "SCHEMA_AND_PROMPT_REVISION",
        "why": "需通用 relation node/edge 表达兼容性；并规范 measurement basis/denominator 的 typed representation。",
    },
    {
        "cluster_id": "RC5",
        "primary_failures": ["ASSERTION_CONTENT", "CONDITION", "SCOPE"],
        "recommendation": "PROMPT_ONLY_REVISION",
        "why": "先明确 assertion anchor 与 substantive clause、归属导语分离及 evidence/member 对齐；不把同一切分缺陷拆成多个独立算法修改。",
    },
    {
        "cluster_id": "RC6",
        "primary_failures": ["CAUSAL_RELATION"],
        "recommendation": "SCHEMA_AND_PROMPT_REVISION",
        "why": "关系连接两个 assertion 的方向化节点/边以及 evidence binding 当前不可直接表达；需通用 relation contract，不增关键词表。",
    },
    {
        "cluster_id": "RC7",
        "primary_failures": ["ATTACHMENT", "SCOPE"],
        "recommendation": "BINDING_GENERATION_REVISION",
        "why": "输出已含 shared edge 与 scope value 语义；需要通用完整性保证/生成指引，针对每个 resolved material edge/context 绑定对应 evidence，不允许 sibling leakage。",
    },
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, payload: Any) -> None:
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    if ledger["fact_count"] != 13 or len(ledger["facts"]) != 13:
        raise SystemExit("frozen D4-D input is not exactly 13 facts")

    ownership = []
    for i, fact in enumerate(ledger["facts"], 1):
        audit = FACT_AUDIT[fact["type"]]
        ownership.append(
            {
                "item": i,
                "claim_id": fact["claim_id"],
                "failure_type": fact["type"],
                "frozen_finding": fact["finding"],
                "frozen_basis": fact["basis"],
                "primary_classification": audit["primary"],
                "root_cause_cluster": audit["cluster"],
                "expected_domain_neutral_fact": audit["expected"],
                "frozen_evidence": audit["evidence"],
                "secondary_causes": audit["secondary"],
            }
        )
    if len(ownership) != 13 or len({x["item"] for x in ownership}) != 13:
        raise SystemExit("ownership ledger cardinality failure")
    write_json(
        "13_item_extraction_ownership_ledger.json",
        {
            "artifact_version": "d4e.extraction_ownership.v1",
            "source_fact_count": 13,
            "primary_owner_count": len(ownership),
            "items": ownership,
        },
    )
    write_json(
        "root_cause_cluster_ledger.json",
        {
            "artifact_version": "d4e.root_cause_clusters.v1",
            "raw_failure_count": 13,
            "root_cause_cluster_count": 7,
            "clusters": REVISION_ROWS,
            "mapping": {x["item"]: x["root_cause_cluster"] for x in ownership},
        },
    )
    write_json(
        "schema_capability_matrix.json",
        {
            "artifact_version": "d4e.schema_capability.v1",
            "capability_enum": [
                "SUPPORTED_DIRECTLY",
                "SUPPORTED_THROUGH_CANONICAL_ATTACHMENT",
                "SUPPORTED_WITH_LIMITATION",
                "NOT_SUPPORTED",
                "AMBIGUOUS_CONTRACT",
            ],
            "rows": [
                {"semantic_family": a, "capability": b, "basis": c} for a, b, c in SCHEMA_ROWS
            ],
        },
    )
    write_json(
        "prompt_adequacy_audit.json",
        {
            "artifact_version": "d4e.prompt_adequacy.v1",
            "prompt_version": "source-context-extraction.generic.v2",
            "prompt_location": "apps/api/app/services/source_context_extractor_v2.py:SOURCE_CONTEXT_V2_INSTRUCTIONS",
            "rows": [
                {"requirement": a, "classification": b, "finding": c} for a, b, c in PROMPT_ROWS
            ],
        },
    )
    write_json(
        "schema_adequacy_audit.json",
        {
            "artifact_version": "d4e.schema_adequacy.v1",
            "schema_version": "source-context-extraction-schema-v2",
            "domain_types": [
                "SourceAssertionV2",
                "SourceContextV2",
                "SourceAttachmentV2",
                "SourceScopeV2",
                "EvidenceSpanInput",
                "SemanticEvidenceBindingInput",
            ],
            "role_enum_has_method_relation": True,
            "related_binding_ref_supports_method_value_link": True,
            "context_semantic_value_is_open_json": True,
            "attachment_relations": ["APPLIES_TO", "CONDITIONS", "ATTRIBUTES", "UNKNOWN"],
            "evidence_target_kinds": ["ASSERTION", "CONTEXT", "ATTACHMENT", "SCOPE"],
            "assertion_to_assertion_relation_target": False,
            "typed_measurement_basis_role": False,
            "assessment": "Most failures are representable; compatibility/cause/result need explicit graph relation contract, while purpose/basis remain weakly typed.",
        },
    )
    write_json(
        "validator_adequacy_audit.json",
        {
            "artifact_version": "d4e.validator_adequacy.v1",
            "rejections": [
                {
                    "failure_type": "MEASUREMENT_DEFINITION",
                    "classification": "TRUE_INVALID",
                    "reason": "Source reports AUROC value, not formal definition; rejecting definition role is correct.",
                },
                {
                    "failure_type": "CONDITION",
                    "classification": "TRUE_INVALID",
                    "reason": "Extracted roles were not supported as assertion conditions by the frozen source evidence.",
                },
                {
                    "failure_type": "EPISTEMIC_AGENT",
                    "classification": "TRUE_INVALID",
                    "reason": "No independent/explicit agent entity is established beyond generic attribution wording.",
                },
                {
                    "failure_type": "METHOD_RELATION",
                    "classification": "CONTRACT_AMBIGUOUS",
                    "reason": "Expected relation binding can be expressed, but missing binding is model omission; NOT_EVALUATED is not false rejection.",
                },
                {
                    "failure_type": "ATTACHMENT",
                    "classification": "TRUE_INVALID",
                    "reason": "Resolved material attachment lacked its required direct or valid scoped semantic binding.",
                },
                {
                    "failure_type": "SCOPE",
                    "classification": "TRUE_INVALID",
                    "reason": "Scope membership/evidence provenance is inconsistent; fail-closed behavior is correct.",
                },
            ],
            "false_rejections_confirmed": 0,
        },
    )
    write_json(
        "epistemic_audit.json",
        {
            "artifact_version": "d4e.epistemic_audit.v1",
            "claim_id": "01a0c82b-3531-79b2-8160-a9e47e4c2bff",
            "source_span": "frozen D4-B extractor evidence E1; exact source text includes attribution lead-in and substantive clause",
            "operator": {
                "output_representation": "no EPISTEMIC context; EPISTEMIC_OPERATOR binding B2 targets ASSERTION/A1",
                "should_bind_to": "EPISTEMIC context Cx (CONTEXT target); context attachment links to proposition assertion",
                "classification": "MODEL_TARGET_SELECTION_ERROR",
            },
            "agent": {
                "output_representation": "attribution field on C1; B3 targets ASSERTION/A1; generic source mention not a named author",
                "should_bind_to": "EPISTEMIC context Cx only if source text supports agent; never use external retrieval provenance",
                "classification": "MODEL_TARGET_SELECTION_ERROR",
            },
            "proposition_target": {
                "representation": "SourceAssertionV2 A1 with substantive clause span; EPISTEMIC context attachment associates operator/agent with assertion",
                "schema_support": "SUPPORTED_WITH_LIMITATION",
            },
            "role_target_mismatch": "MODEL_TARGET_SELECTION_ERROR",
            "validator_false_rejection": False,
        },
    )
    write_json(
        "segmentation_evidence_audit.json",
        {
            "artifact_version": "d4e.segmentation_evidence.v1",
            "claim_id": "01a0c436-7ecc-7d21-b32d-c5308fa1acec",
            "assertion_anchor_boundary": "A1 evidence E1 is attribution-only; not a substantive assertion unit.",
            "evidence_alignment": "E1-E5 exact-substring spans exist; semantic ownership is mismatched for A1/A2/A3. Exact substring proves textual grounding only.",
            "causal_span": "E5 is the result/causal clause but no relation object connects cause and result assertions.",
            "scope_consequence": "S1 includes A1/A2/A3 but cites E1 only; graph membership provenance does not cover each substantive member.",
            "primary_upstream_defect": "ASSERTION_SEGMENTATION_DEFECT",
            "would_correct_segmentation_resolve": "assertion/evidence ownership and much of the invalid scope membership, yes; explicit causal relation expressiveness remains a separate schema limitation.",
            "not_independent_defects": [
                "attribution-only anchor",
                "wrong assertion evidence refs",
                "scope evidence/member mismatch",
            ],
            "textual_grounding_vs_semantic_alignment": "separate dimensions; all spans may textually resolve while assertion meaning alignment fails.",
        },
    )
    write_json(
        "relation_expressiveness_audit.json",
        {
            "artifact_version": "d4e.relation_expressiveness.v1",
            "relations": [
                {
                    "relation": "ATTRIBUTION",
                    "canonical_representation": "EPISTEMIC context + EPISTEMIC_AGENT/OPERATOR bindings + context-to-assertion attachment",
                    "capability": "SUPPORTED_WITH_LIMITATION",
                },
                {
                    "relation": "CONDITION",
                    "canonical_representation": "CONDITION context + CONDITION binding + CONDITIONS attachment",
                    "capability": "SUPPORTED_DIRECTLY",
                },
                {
                    "relation": "APPLICABILITY",
                    "canonical_representation": "SourceAttachment APPLIES_TO",
                    "capability": "SUPPORTED_THROUGH_CANONICAL_ATTACHMENT",
                },
                {
                    "relation": "PURPOSE",
                    "canonical_representation": "purpose key in context semantic_value + generic context binding/attachment",
                    "capability": "SUPPORTED_WITH_LIMITATION",
                },
                {
                    "relation": "CAUSE",
                    "canonical_representation": "no assertion-to-assertion edge/node",
                    "capability": "NOT_SUPPORTED",
                },
                {
                    "relation": "RESULT",
                    "canonical_representation": "no assertion-to-assertion edge/node",
                    "capability": "NOT_SUPPORTED",
                },
                {
                    "relation": "COMPARISON",
                    "canonical_representation": "measurement comparator payload/role; relation endpoints remain weakly specified",
                    "capability": "SUPPORTED_WITH_LIMITATION",
                },
                {
                    "relation": "COMPATIBILITY",
                    "canonical_representation": "APPLIES_TO is too generic; no typed compatibility relation",
                    "capability": "SUPPORTED_WITH_LIMITATION",
                },
                {
                    "relation": "MEMBERSHIP",
                    "canonical_representation": "SourceScopeV2.source_assertion_refs + SCOPE_STRUCTURE evidence binding",
                    "capability": "SUPPORTED_DIRECTLY",
                },
                {
                    "relation": "MEASUREMENT_ASSOCIATION",
                    "canonical_representation": "context attachment to assertion; no specialized measurement-to-category edge",
                    "capability": "SUPPORTED_WITH_LIMITATION",
                },
            ],
            "causal_result_audit": "SCHEMA_RELATION_ENUM_GAP; more fundamentally no assertion-to-assertion relation node/target exists.",
            "method_relation_audit": "METHOD_RELATION + related_binding_ref to same-target METHOD_VALUE is sufficient; no duplicate D3/D4 obligation is proposed.",
        },
    )
    write_json(
        "shared_attachment_audit.json",
        {
            "artifact_version": "d4e.shared_attachment.v1",
            "claim_id": "01a0c853-cc4a-7648-b4e2-d175da55860d",
            "observed": "AT1 RESOLVED, targets A1/A2/A3, evidence E1/E2/E3; no ATTACHMENT binding to AT1. Existing validator requires direct attachment binding or valid scope binding.",
            "invariant_recommended": "RESOLVED semantic attachment + evidence_refs + material relation -> at least one matching ATTACHMENT binding; each target must be independently grounded; multi-target binding does not imply sibling support.",
            "classification": "BINDING_GENERATION_COMPLETENESS_DEFECT",
            "synthetic_implications": "A multi-target edge can use one binding only for the edge evidence; per-target semantic support and member-specific evidence remain independently checked. A matching SCOPE_STRUCTURE binding may serve only when scope membership exactly equals attachment membership and evidence is in scope.",
            "C1_scope_value": "SCOPE context C1 exists but no corresponding SCOPE_VALUE binding; same completeness failure family.",
            "downstream_sibling_leakage": "not observed in frozen safety counters; preserve no-sibling-leak invariant.",
        },
    )
    write_json(
        "measurement_completeness_audit.json",
        {
            "artifact_version": "d4e.measurement_completeness.v1",
            "items": [
                {
                    "claim_id": "01a0c434-9753-7fff-be68-21d83da29bf2",
                    "dimension": "definition",
                    "finding": "AUROC identity and result 0.992 are present; formal definition absent from source.",
                    "classification": "SOURCE_FACT_ABSENT",
                    "safe_action": "Do not infer definition.",
                },
                {
                    "claim_id": "01a0c424-0fc5-7cea-ac2d-92d3274f5d85",
                    "dimension": "value_unit_basis",
                    "finding": "56.87 and milliseconds retained; per-inference basis occurs in exact span but is not separately typed.",
                    "classification": "MODEL_OMISSION_WITH_TYPED_LIMITATION",
                    "safe_action": "Represent basis generically; do not change value/unit.",
                },
                {
                    "claim_id": "01a0c424-0fc5-7cea-ac2d-92d3274f5d85",
                    "dimension": "compatibility_result_relation",
                    "finding": "Throughput-compatible-with-live-production assertion omitted as a relation.",
                    "classification": "SCHEMA_RELATION_GAP",
                    "safe_action": "Future generic relation contract; no industry-specific field.",
                },
            ],
            "dimensions_reviewed": [
                "metric identity",
                "definition",
                "result value",
                "unit",
                "comparator",
                "threshold",
                "basis/denominator",
                "per-inference/per-item semantics",
                "scope/category association",
                "compatibility/result relation",
            ],
        },
    )
    write_json(
        "recommended_revision_plan.json",
        {
            "artifact_version": "d4e.revision_plan.v1",
            "clusters": REVISION_ROWS,
            "implementation_performed": False,
        },
    )

    frozen_files = [
        LEDGER_PATH,
        AGGREGATE_PATH,
        ROOT / "artifacts/v2_1r_d4d_ownership_alignment/hash_manifest.json",
        ROOT
        / "artifacts/v2_1r_d4c_systematic_failure_ownership/extraction_completeness_ledger.json",
        ROOT / "artifacts/v2_1r_d4c_systematic_failure_ownership/measurement_audit.json",
        ROOT / "apps/api/app/domain/source_context_grounding_v2.py",
        ROOT / "apps/api/app/services/source_context_extractor_v2.py",
        ROOT / "apps/api/app/domain/source_context_extraction.py",
        ROOT / "scripts/audit_v2_1r_d4e_extraction_completeness.py",
    ]
    frozen_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in frozen_files}
    write_json(
        "hash_manifest.json",
        {
            "artifact_version": "d4e.hash_manifest.v1",
            "frozen_inputs_and_contract_sha256": frozen_hashes,
            "d4b_claim_count": aggregate["totals"]["claim_count"],
            "d4b_extractor_calls_historical": aggregate["totals"]["extractor_call_count"],
            "d4b_provider_http_outbound_historical": aggregate["totals"][
                "external_http_outbound_count"
            ],
            "d4b_holdout_accessed": aggregate["holdout_accessed"],
            "d4b_database_writes": aggregate["database_writes"],
            "d4e_provider_calls": 0,
            "d4e_external_http": 0,
            "d4e_extractor_calls": 0,
            "d4e_verifier_calls": 0,
            "note": "Hashes identify immutable frozen evidence and the unchanged inspected contracts; this audit run executed no model path.",
        },
    )

    facts = [
        "D4-D 的 13 条 extraction-only facts 全部逐项审计；D4-B 为历史来源证据，历史 8 claims/8 extractor calls/16 provider outbound；D4-E 本轮 Provider/HTTP/Extractor/Verifier 全为 0。",
        "13 raw failures collapse to 7 clusters；primary owner 和 cluster 对每项均唯一。",
        "可表达但遗漏/错绑占多数；结构 schema 明确不能表示 assertion 间 cause/result；formal AUROC definition 缺源事实，不是实现故障。",
        "D4-B hard-safety counters (fabricated spans, invalid promotions, unsupported promotions, missing-binding promotions, unsafe promotion/masking/vacuous promotion) 均为 0；D4-D downstream 已冻结且不在本次修改范围。",
    ]
    sections = {
        "A. Frozen Evidence": facts[0],
        "B. 13-Item Failure Ownership": "见 `13_item_extraction_ownership_ledger.json`：13/13，各项恰有一个 primary classification 与 root cluster。",
        "C. Root-Cause Clusters": "见 `root_cause_cluster_ledger.json`：raw=13，clusters=7。RC5 segmentation 聚合 attribution anchor、evidence mapping、condition ownership 与 scope provenance 症状；causal relation 独立保留为 RC6 schema gap。",
        "D. Current Schema Capability": "见 `schema_capability_matrix.json`。可表达多数节点/绑定；cause/result 无 assertion-to-assertion relation target；compatibility 仅能被过宽的 APPLIES_TO近似。",
        "E. Assertion Anchor Contract": "Anchor 应对应 source 明确表达的实质命题单元并覆盖 predicate/content；attribution lead-in、discourse marker、reporting verb phrase 或孤立修饰语不能单独成为 anchor。当前 prompt/schema 没有明确这一边界。",
        "F. Segmentation / Evidence Alignment": "Claim6 E1-E5 都可能 exact-match，但 A1 的 E1 仅 attribution、A1/A2/A3 内容所有权错位；textual grounding != semantic alignment。正确切分可修复主要 evidence/scope 症状，不能补出 relation schema。",
        "G. Epistemic Representation": "operator/agent 应绑 EPISTEMIC context；context attachment 关联实质 proposition assertion。target 是 assertion，不是 operator/agent 的 ASSERTION target。该 role-target mismatch 为 MODEL_TARGET_SELECTION_ERROR；泛称来源不支持合成具名 agent。",
        "H. Method / Purpose Representation": "METHOD_VALUE 与 METHOD_RELATION 已直接支持，后者可用 related_binding_ref 链到同 target METHOD_VALUE。目的可放开放 semantic_value，但当前归 OTHER/CONTEXT_VALUE，purpose owner 合同欠明确。",
        "I. Causal / Result Representation": "当前 SourceAttachment 只连 context→assertion，relation enum 不含 cause/result；binding target kinds 无 relation node。需 schema/graph contract，而不是关键词表。",
        "J. Compatibility / Applicability Representation": "APPLIES_TO 可表达广义适用性，但不能精确保留 measurement/result 对生产可用性的 compatibility 断言；当前为 limitation，RC4建议通用 relation contract。",
        "K. Shared Attachment / Binding Completeness": "resolved 且 material 的 attachment 有证据时应有对应 ATTACHMENT binding；多目标必须每目标独立 grounding。Claim7 AT1 缺 binding，属于 BINDING_GENERATION_COMPLETENESS_DEFECT；无 sibling support 泄漏。",
        "L. Scope Extraction": "Claim6 S1 membership 为 A1/A2/A3，但 evidence 仅归属 span E1，provenance不一致，保留 invalid/fail-closed。Claim7 缺 SCOPE_VALUE binding 是 RC7 binding completeness。D4-D canonical ownership 不重开。",
        "M. Measurement Completeness": "AUROC=0.992 是结果而非正式定义，来源缺事实；56.87 ms 的 per-inference basis 在词面中但未 typed；production compatibility 关系未抽取且 schema 表达不足。详见 measurement audit。",
        "N. Prompt Adequacy": "详见 prompt audit。prompt 对 METHOD_RELATION、explicit bindings、shared context 有一般明确要求；assertion anchor边界、relation ownership、basis 和 binding完备不变量欠明确。",
        "O. Schema Adequacy": "详见 schema audit。不存在 causal/result 断言间边/节点或 measurement basis 专用 role；开放 semantic JSON 可缓解但不能替代稳定契约。",
        "P. Validator Adequacy": "冻结拒绝均为 TRUE_INVALID 或 contract ambiguity；confirmed false rejection=0。不能为提升通过率放宽 validator。",
        "Q. Grounding Adequacy": "未发现 grounding 对有效表示的已证实 false rejection；已有安全门严谨。E1等文本可定位并不代表语义归属正确。",
        "R. Systematic vs One-Off Issues": "明确 generic interface defects：assertion segmentation/ownership contract、resolved edge binding completeness；graph relation expressiveness gap。单 Claim 输出失误本身不称 systematic，需 fresh Claims 复验。",
        "S. Recommended Revision Type": "按 RC1..RC7 逐簇恰选一个 revision type，详见 recommended_revision_plan.json；本阶段没有实施修改。",
        "T. Versioning Plan": "如未来改变语义合同，保留 v2 不变，采用 source-context-extraction.generic.v3 与 source-context-extraction-schema-v3；若只修 prompt 且 schema 不变，可单独 prompt version。",
        "U. Downstream Freeze Impact": "所有建议默认不改 Grounding Requirement v2.1、Matcher v2.1、Hybrid verifier、readiness aggregation；D4-D canonical ownership继续消费既有正确 graph。当前无下游修改。",
        "V. Anti-Overfitting Audit": "建议仅通用 proposition segmentation、typed relation graph、binding completeness 与 measurement basis contract；无 Claim ID/entity/domain/benchmark source规则或硬编码 target ref。",
        "W. Provider Calls": "D4-E Provider=0；External HTTP=0；Extractor=0；Semantic verifier=0。D4-B frozen historical evidence中调用次数仅作为历史来源事实，不是本阶段调用。Holdout未访问。",
        "X. Artifacts": "见本阶段 output folder；包括 13 项 ledger、clusters、capability/prompt/schema/validator/epistemic/segmentation/relation/shared attachment/measurement audits、revision plan、hash manifest 与本报告。",
        "Y. Final Decision": "SOURCE EXTRACTION COMPLETENESS REVEALS MULTIPLE EXTRACTION-LAYER ISSUES",
    }
    report = (
        "# V2.1R D4-E Source Extraction Completeness Contract Audit\n\n"
        + "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
        + "\n"
    )
    (OUT / "d4e_final_report.md").write_text(report, encoding="utf-8")
    artifact_hashes = {
        p.name: sha(p)
        for p in sorted(OUT.iterdir())
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    write_json(
        "artifact_hashes.json",
        {"artifact_version": "d4e.generated_artifact_hashes.v1", "sha256": artifact_hashes},
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "source_facts": len(ownership),
                "clusters": 7,
                "artifacts": len(list(OUT.iterdir())),
                "provider_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
