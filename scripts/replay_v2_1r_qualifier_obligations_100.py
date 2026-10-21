"""Replay frozen V2.1R model outputs through the qualifier obligation validator only."""

# Long multilingual markdown is deliberately embedded so the replay emits a self-contained
# Chinese report with its required terminology and punctuation.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.services.atomic_proposition_validation import (  # noqa: E402
    inspect_qualifier_fields,
)

WARNING_STATUSES = {"POTENTIAL_LOSS", "AMBIGUOUS"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _composition_projection(decomposition: dict[str, Any]) -> dict[str, Any]:
    trace = decomposition.get("status_transition_trace", [])
    return {
        "model_assertion_composition": decomposition.get("model_structured_result", {}).get(
            "assertion_composition"
        ),
        "source_span_trace": decomposition.get("validator_result", {}).get(
            "composition_source_span_trace"
        ),
        "composition_validation_trace": [
            item
            for item in trace
            if item.get("stage") in {"COMPOSITION_VALIDATION", "COMPOSITION_SEMANTIC_VALIDATION"}
        ],
        "final_composition_validation": decomposition.get("final_shadow_result", {}).get(
            "composition_validation"
        ),
    }


def _composition_metrics(decomposition: dict[str, Any]) -> dict[str, Any]:
    model_result = decomposition.get("model_structured_result", {})
    composition = model_result.get("assertion_composition", {})
    composition = composition if isinstance(composition, dict) else {}
    groups = composition.get("groups", [])
    groups_by_ref = {
        str(group.get("local_ref")): group
        for group in groups
        if isinstance(group, dict) and group.get("local_ref") is not None
    }
    operators = Counter(str(group.get("operator", "UNKNOWN")) for group in groups)
    nested_group_count = sum(
        1
        for group in groups
        for member in group.get("members", [])
        if member.get("member_type") == "group"
    )
    or_xor_leaf_refs: set[str] = set()

    def collect_leaves(group_ref: str, visited: set[str]) -> set[str]:
        if group_ref in visited or group_ref not in groups_by_ref:
            return set()
        visited.add(group_ref)
        leaves: set[str] = set()
        for member in groups_by_ref[group_ref].get("members", []):
            ref = str(member.get("ref", ""))
            if member.get("member_type") == "proposition":
                leaves.add(ref)
            elif member.get("member_type") == "group":
                leaves.update(collect_leaves(ref, visited))
        return leaves

    for group in groups:
        if str(group.get("operator", "")).upper() in {"OR", "XOR"}:
            or_xor_leaf_refs.update(collect_leaves(str(group.get("local_ref", "")), set()))

    eligibility: dict[str, str] = {}
    for item in decomposition.get("status_transition_trace", []):
        if item.get("stage") == "COMPOSITION_VALIDATION":
            eligibility.update(item.get("derived_member_eligibility", {}))
    leakage = sorted(
        ref
        for ref in or_xor_leaf_refs
        if eligibility.get(ref) == "INDEPENDENTLY_ASSERTED"
    )
    span_failures = sum(
        item.get("compatibility_result") == "FAIL"
        for item in decomposition.get("validator_result", {}).get(
            "composition_source_span_trace", []
        )
    )
    return {
        "group_count": len(groups),
        "operator_counts": dict(operators),
        "nested_group_count": nested_group_count,
        "or_xor_group_count": sum(operators.get(op, 0) for op in ("OR", "XOR")),
        "or_xor_leaf_count": len(or_xor_leaf_refs),
        "or_xor_leaf_leakage_count": len(leakage),
        "or_xor_leaf_leakage_refs": leakage,
        "source_span_failure_count": span_failures,
    }


def replay(input_dir: Path, baseline_report: Path) -> dict[str, Any]:
    report = _read_json(baseline_report)
    claim_records = {item["claim_id"]: item for item in report.get("claim_records", [])}
    human_events = report.get("qualifier", {}).get("event_classifications", [])
    diagnostics = sorted(input_dir.glob("*.json"))
    if len(diagnostics) != 100:
        raise ValueError(f"expected exactly 100 persisted diagnostics, got {len(diagnostics)}")

    rows: list[dict[str, Any]] = []
    migration: list[dict[str, Any]] = []
    before_statuses: Counter[str] = Counter()
    after_statuses: Counter[str] = Counter()
    before_categories: Counter[str] = Counter()
    after_categories: Counter[str] = Counter()
    claim_counts: Counter[str] = Counter()
    eligible_blocked: list[str] = []
    independent_blockers: Counter[str] = Counter()
    composition_invariant = True
    artifact_hashes_before: dict[str, str] = {}
    artifact_hashes_after: dict[str, str] = {}
    composition_hashes_before: dict[str, str] = {}
    composition_hashes_after: dict[str, str] = {}
    composition_metrics: Counter[str] = Counter()
    composition_operator_counts: Counter[str] = Counter()
    not_evaluated: Counter[str] = Counter()
    provider_calls = 0

    for path in diagnostics:
        raw_bytes = path.read_bytes()
        artifact_hashes_before[path.name] = _sha256(raw_bytes)
        artifact = _read_json(path)
        claim = artifact.get("claim", {})
        decomposition = artifact.get("decomposition", {})
        if not claim.get("claim_id") or not isinstance(claim.get("claim_text"), str):
            raise ValueError(f"missing source Claim in {path.name}")
        model_result = decomposition.get("model_structured_result")
        if not isinstance(model_result, dict):
            raise ValueError(f"missing persisted model output in {path.name}")
        propositions = model_result.get("propositions", [])
        if not isinstance(propositions, list):
            raise ValueError(f"invalid proposition list in {path.name}")
        provider_calls += int(decomposition.get("model_call_count", 0))
        composition_hashes_before[path.name] = _sha256(
            json.dumps(
                _composition_projection(decomposition),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        per_claim_composition = _composition_metrics(decomposition)
        composition_metrics.update(
            {
                key: value
                for key, value in per_claim_composition.items()
                if isinstance(value, int)
            }
        )
        composition_operator_counts.update(per_claim_composition["operator_counts"])
        old_checks = decomposition.get("validator_result", {}).get("qualifier_checks", [])
        old_map = {item["category"]: item["status"] for item in old_checks}
        parser_status = decomposition.get("parser_result", {}).get("status")
        validator_status = decomposition.get("validator_result", {}).get("status")
        final_status = decomposition.get("final_shadow_result", {}).get("status")
        if (
            parser_status != "PARSED"
            or validator_status != "VALID"
            or not propositions
            or final_status
            in {
                "SCHEMA_INVALID",
                "COMPOSITION_REFERENCE_INVALID",
                "COMPOSITION_SEMANTIC_INVALID",
                "SOURCE_SPAN_INVALID",
            }
        ):
            checks = ()
            not_evaluated[f"{final_status or validator_status or parser_status}"] += 1
        else:
            checks = inspect_qualifier_fields(claim["claim_text"], propositions)
        new_map = {item.category: item.status.value for item in checks}
        before_statuses.update(old_map.values())
        after_statuses.update(new_map.values())
        before_categories.update(f"{category}|{status}" for category, status in old_map.items())
        after_categories.update(f"{category}|{status}" for category, status in new_map.items())

        claim_id = claim["claim_id"]
        record = claim_records.get(claim_id, {})
        old_final = record.get(
            "final_shadow_status", decomposition.get("final_shadow_result", {}).get("status")
        )
        old_warning = any(status in WARNING_STATUSES for status in old_map.values())
        new_warning = any(status in WARNING_STATUSES for status in new_map.values())
        claim_counts[
            f"before:{'UNRESOLVED_QUALIFIER' if old_warning else 'NO_QUALIFIER_BLOCKER'}"
        ] += 1
        claim_counts[
            f"after:{'UNRESOLVED_QUALIFIER' if new_warning else 'NO_QUALIFIER_BLOCKER'}"
        ] += 1

        issues = list(record.get("issue_codes", decomposition.get("validation_issue_codes", [])))
        non_qualifier_issues = [
            str(code) for code in issues if not str(code).startswith(("potential_", "ambiguous_"))
        ]
        trace = decomposition.get("status_transition_trace", [])
        comp_statuses = [
            item.get("status")
            for item in trace
            if item.get("stage") in {"COMPOSITION_VALIDATION", "COMPOSITION_SEMANTIC_VALIDATION"}
        ]
        if old_final == "UNRESOLVED_DECOMPOSITION" and old_warning and not new_warning:
            if non_qualifier_issues:
                independent_blockers.update(non_qualifier_issues)
            elif parser_status != "PARSED":
                independent_blockers[f"PARSER:{parser_status}"] += 1
            elif validator_status != "VALID":
                independent_blockers[f"VALIDATOR:{validator_status}"] += 1
            elif any(status not in {"VALID", None} for status in comp_statuses):
                independent_blockers.update(
                    f"COMPOSITION:{status}" for status in comp_statuses if status != "VALID"
                )
            elif (
                model_result.get("decomposition_status") == "unresolved_decomposition"
                or not propositions
            ):
                independent_blockers["MODEL_ABSTENTION_OR_ZERO_PROPOSITIONS"] += 1
            else:
                eligible_blocked.append(claim_id)

        # The replay never enters composition/eligibility producers. Preserve an exact
        # digest of persisted composition and source-span outputs as an audit boundary.
        rows.append(
            {
                "claim_id": claim_id,
                "artifact": path.name,
                "old_final_status": old_final,
                "old_qualifier_checks": old_map,
                "new_qualifier_checks": {
                    item.category: {
                        "status": item.status.value,
                        "applicability": item.applicability.value,
                        "evidence_strength": item.evidence_strength.value,
                        "obligation_count": item.obligation_count,
                        "reason_codes": list(item.reason_codes),
                    }
                    for item in checks
                },
                "qualifier_replay_state": "EVALUATED"
                if checks
                else "NOT_EVALUATED_INVALID_OR_ABSTENTION",
                "proposition_count": len(propositions),
                "parser_status": parser_status,
                "proposition_validator_status": validator_status,
                "composition_validator_statuses": comp_statuses,
                "qualifier_blocked_before": old_warning,
                "qualifier_blocked_after": new_warning,
                "eligible_to_resolve_after_qualifier_replay": claim_id in eligible_blocked,
                "independent_blockers": non_qualifier_issues,
            }
        )

    for event in human_events:
        claim_id = event["claim_id"]
        category = event["category"]
        row = next(item for item in rows if item["claim_id"] == claim_id)
        raw = _read_json(input_dir / row["artifact"])
        props = raw["decomposition"]["model_structured_result"].get("propositions", [])
        check = next(
            (
                item
                for item in inspect_qualifier_fields(raw["claim"]["claim_text"], props)
                if item.category == category
            ),
            None,
        )
        evidence = check.preservation_evidence if check else ()
        candidate_indexes = sorted(
            {item.proposition_index for item in evidence if item.proposition_index is not None}
        )
        candidates = []
        for index in candidate_indexes:
            prop = props[index]
            candidates.append(
                {
                    "index": index,
                    "local_ref": prop.get("local_ref"),
                    "subject": prop.get("subject"),
                    "predicate": prop.get("predicate"),
                    "typed_value": prop.get("typed_value"),
                    "material_qualifiers": prop.get("material_qualifiers", []),
                    "source_span": prop.get("source_span"),
                }
            )
        migration.append(
            {
                "claim_id": claim_id,
                "ordinal": event.get("ordinal"),
                "category": category,
                "old_status": event.get("status"),
                "old_category": category,
                "human_classification": event.get("manual_classification", "UNCLASSIFIED"),
                "source_obligation": [item.obligation.model_dump(mode="json") for item in evidence],
                "candidate_propositions": candidates,
                "semantic_view_evidence": [item.model_dump(mode="json") for item in evidence],
                "attachment_evidence": [
                    {
                        "proposition_index": item.proposition_index,
                        "compatible": item.attachment_compatible,
                        "reason": item.reason_code,
                    }
                    for item in evidence
                ],
                "new_decision": check.status.value if check else "NOT_APPLICABLE",
                "reason_codes": list(check.reason_codes)
                if check
                else ["SOURCE_SIGNAL_NO_LONGER_DETECTED"],
            }
        )

    if len(migration) != 36:
        raise ValueError(f"expected the 36 audited warning records, got {len(migration)}")
    if provider_calls != 100:
        raise ValueError(
            f"expected exactly 100 already-persisted Provider outputs, found {provider_calls} calls"
        )
    for path in diagnostics:
        artifact_hashes_after[path.name] = _sha256(path.read_bytes())
        artifact = _read_json(path)
        decomposition = artifact["decomposition"]
        composition_hashes_after[path.name] = _sha256(
            json.dumps(
                _composition_projection(decomposition),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    composition_invariant = composition_hashes_before == composition_hashes_after
    if artifact_hashes_before != artifact_hashes_after:
        raise RuntimeError("a frozen input diagnostic changed during read-only replay")
    return {
        "replay_version": "v2.1r-obligation-semantic-preservation-100-v1",
        "provider_calls_during_replay": 0,
        "persisted_provider_calls_in_inputs": provider_calls,
        "claims_replayed": len(rows),
        "warning_events_migrated": len(migration),
        "before_status_counts": dict(before_statuses),
        "after_status_counts": dict(after_statuses),
        "before_category_status_counts": dict(before_categories),
        "after_category_status_counts": dict(after_categories),
        "claim_level_before_after_counts": dict(claim_counts),
        "previously_31_qualifier_blocked_claims_resolved_by_replay": len(eligible_blocked),
        "eligible_claim_ids": eligible_blocked,
        "remaining_independent_blockers": dict(independent_blockers),
        "not_evaluated_invalid_or_abstention_counts": dict(not_evaluated),
        "composition_source_span_eligibility_producers_modified": False,
        "frozen_input_files_unchanged": artifact_hashes_before == artifact_hashes_after,
        "composition_projection_unchanged": composition_invariant,
        "composition_projection_sha256_before": _sha256(
            json.dumps(composition_hashes_before, sort_keys=True, separators=(",", ":")).encode()
        ),
        "composition_projection_sha256_after": _sha256(
            json.dumps(composition_hashes_after, sort_keys=True, separators=(",", ":")).encode()
        ),
        "persisted_composition_metrics": {
            **dict(composition_metrics),
            "operator_counts": dict(composition_operator_counts),
        },
        "input_artifact_sha256": artifact_hashes_before,
        "all_claim_rows": rows,
        "warning_migration": migration,
    }


_CATEGORY_CONTRACTS = (
    (
        "time",
        "date/time/period; exact date-value normalization",
        "time/date/period qualifier, temporal typed value, compatible predicate/text",
        "source span/value must bind to the affected proposition; wrong or conflicting date blocks",
    ),
    (
        "version",
        "version marker/product-attached or multi-segment version",
        "version qualifier or explicit version carrier",
        "numeric dates/measurements excluded; conflicting version blocks",
    ),
    (
        "unit",
        "number + recognized unit",
        "quantity/range numeric value with compatible unit",
        "unit without its numeric amount is insufficient; wrong unit/amount blocks",
    ),
    (
        "currency",
        "amount + currency marker",
        "numeric measurement carrying amount and currency",
        "currency label alone is insufficient; amount and currency both bind",
    ),
    (
        "comparison",
        "comparison cue and source target/context",
        "comparison target/metric or predicate that expresses relation",
        "target and attachment must agree; wrong target blocks",
    ),
    (
        "negation",
        "negation marker/polarity cue",
        "negated polarity, negation qualifier, or compatible predicate",
        "substring-only cues fail closed; opposite polarity blocks",
    ),
    (
        "modality",
        "modal/forecast/possibility marker",
        "modality field, modality qualifier, or explicit predicate",
        "must preserve force and attach to same assertion",
    ),
    (
        "attribution",
        "source-attributing construction",
        "attribution field, attribution qualifier, or explicit predicate",
        "attributed actor/proposition scope must agree; ambiguous scope remains ambiguous",
    ),
    (
        "condition",
        "conditional/under-setting construction",
        "condition/scenario/configuration qualifier or explicit predicate",
        "must remain attached to the conditioned proposition",
    ),
    (
        "configuration",
        "configuration/setting marker",
        "configuration or method qualifier / explicit relevant carrier",
        "setting identity and target proposition must agree",
    ),
    (
        "sample_count",
        "count + sample/reference unit",
        "typed numeric/text value or measurement/population qualifier",
        "count and sample referent must be co-attached",
    ),
    (
        "population",
        "population quantifier/group marker",
        "population/scope qualifier or compatible subject/predicate",
        "population boundary must be preserved on same proposition",
    ),
    (
        "measurement",
        "measurement/rate/metric signal",
        "typed numeric measurement, basis/extremum/change fields, or relevant qualifier",
        "metric/value/basis must agree; vague lexical overlap is not enough",
    ),
    (
        "method",
        "method/process cue",
        "method/configuration qualifier, typed value, or explicit predicate",
        "method object and process attachment must match; ambiguous disjunction stays ambiguous",
    ),
    (
        "scope",
        "scope/only/all/location boundary",
        "scope/population/location qualifier or compatible subject/predicate",
        "scope target and quantifier must match; unrelated matching token is rejected",
    ),
)


def _markdown_report(result: dict[str, Any], baseline: dict[str, Any]) -> str:
    before = result["before_status_counts"]
    after = result["after_status_counts"]
    old_classes = baseline.get("qualifier", {}).get("human_event_classification", {})
    actual_classes = Counter(item["human_classification"] for item in result["warning_migration"])
    migration_decisions = Counter(item["new_decision"] for item in result["warning_migration"])
    resolved = result["previously_31_qualifier_blocked_claims_resolved_by_replay"]
    remaining_claims = result["claim_level_before_after_counts"].get(
        "after:UNRESOLVED_QUALIFIER", 0
    )
    new_blocked_claims = [
        row["claim_id"]
        for row in result["all_claim_rows"]
        if row["qualifier_blocked_after"] and not row["qualifier_blocked_before"]
    ]
    status_lines = "\n".join(
        f"| `{name}` | {before.get(name, 0)} | {after.get(name, 0)} |"
        for name in ("PRESERVED", "POTENTIAL_LOSS", "AMBIGUOUS", "NOT_APPLICABLE")
    )
    category_lines = "\n".join(
        f"| {category} | {detection} | {carriers} | {attachment} |"
        for category, detection, carriers, attachment in _CATEGORY_CONTRACTS
    )
    migration_lines = "\n".join(
        (
            "| {ordinal} | `{claim_id}` | {category} | {old_status} | "
            "{human_classification} | {new_decision} | `{reason}` |"
        ).format(
            reason=", ".join(item["reason_codes"]),
            **item,
        )
        for item in result["warning_migration"]
    )
    carrier_counts: Counter[str] = Counter()
    for item in result["warning_migration"]:
        for evidence in item["semantic_view_evidence"]:
            carrier_counts.update(evidence.get("carrier_paths", []))
    carrier_lines = (
        "\n".join(f"| `{path}` | {count} |" for path, count in sorted(carrier_counts.items()))
        or "| — | 0 |"
    )
    hard_negative_tests = (
        "wrong time",
        "wrong unit",
        "wrong comparison target",
        "wrong attribution target",
        "wrong method attachment",
        "wrong scope",
        "same token on unrelated proposition",
        "true semantic drop",
        "ambiguous multi-proposition attachment",
        "unsupported-script coverage",
    )
    return f"""# V2.1R Qualifier Semantic Preservation Contract

## A. Bounded-Shadow Evidence

冻结输入：100 个 Claim、216 个最终命题、1,365 次类别检查；旧状态为 74 PRESERVED、32 POTENTIAL_LOSS、4 AMBIGUOUS、1,255 NOT_APPLICABLE。旧摘要称 31 个 Claim 有告警。Provider 已持久化调用 100 次；本次 replay 新调用为 0。

## B. Previous Validator Architecture

旧检查主要要求某类别信号在预期结构字段中出现，且类别正则与结构字段规则耦合；因此同义内容落在 predicate、typed value 或 proposition text 时可能误报，短词/产品名也可能制造 source detector false positive。旧 `NOT_APPLICABLE` 未区分“无支持信号”和“检测覆盖不足”。

## C. New Preservation Abstraction

新增 `QualifierObligation`（仅从 source-side detector 建立）→ `PropositionSemanticView`（逐命题字段投影）→ `PreservationEvidence`（语义匹配、attachment、carrier 与 reason）。该层不是 identity/matcher/verification confidence。

## D. Source Semantic Obligations

每条 obligation 保存类别、规范化值、原文 span、局部 attachment context、materiality 与检测依据。模型输出本身不能创建 obligation。本文无 Claim-ID 专用规则。

## E. Proposition Semantic View

视图收集 subject、predicate、polarity、modality、attribution、material qualifiers、typed value/numeric/unit/currency/comparison/temporal carrier 与 source spans。只用于源义务的保留判定。

## F. Semantic Carrier Taxonomy

| Source category | Source-side applicability signal | Valid semantic carriers (generic) | Attachment/fail-closed contract |
|---|---|---|---|
{category_lines}

本次 36 条审计记录中实际的非空 carrier 路径：

| Carrier path | 证据记录数 |
|---|---:|
{carrier_lines}

## G. Attachment Safety

语义相似不足以通过：优先使用兼容 source span 锁定命题；若无法定位且多个命题都可承载，则保持 AMBIGUOUS。硬负例验证相同 token 位于无关 proposition 不会证明保留；显式冲突的结构值优先否决弱文本证据。

## H. Evidence Strength

STRONG：结构化 qualifier、polarity、attribution/modality 或 typed value；MEDIUM：subject/predicate 关系载体；WEAK：truth-condition-complete proposition text。WEAK 在多命题且无 attachment 证据时不能单独消除歧义。

## I. Category Contracts

上表逐项说明了 15 类信号、carrier 与 attachment 规则。分类使用通用类别契约；没有按冻结样本逐条添加运行规则。

## J. NOT_APPLICABLE / Coverage Semantics

`NOT_APPLICABLE + NO_SUPPORTED_SIGNAL` 仅表示当前支持的 source detector 未找到信号，不宣称自然语言绝对不存在该语义。出现 detector 未覆盖的字母脚本时标记 `DETECTOR_COVERAGE_LIMIT` 和 `SOURCE_SCRIPT_OUTSIDE_DETECTOR_COVERAGE`。这只是覆盖提示，不进行 NLP 推断；存在 supported source obligation 才进入保留检查。

## K. 36 Warning Root-Cause Migration

旧 JSON 顶层摘要计数为 `{old_classes}`；但逐条 `event_classifications` 的实际计数是 `{dict(actual_classes)}`，与摘要不一致。逐条记录是 36 条迁移的依据；此数据完整性差异没有被静默改写。

新决定计数：`{dict(migration_decisions)}`。

| 序号 | Claim ID | 类别 | 旧状态 | 人工分类 | 新决定 | 原因码 |
|---:|---|---|---|---|---|---|
{migration_lines}

## L. False-Preservation Safety

拒绝错误时间、单位/数值、比较对象、归因对象、方法附着、范围，以及只在其他命题出现的相同词；真正丢失返回 POTENTIAL_LOSS；多个兼容命题但附着不明返回 AMBIGUOUS。

## M. Synthetic Fixtures

新增中性合成测试覆盖：跨结构载体等价、数值/值冲突、错误归属、无关命题 token、真实丢失、模糊附着、无支持信号、未覆盖脚本。附加硬负例类别：{", ".join(hard_negative_tests)}。这些是人工构造的合成测试，不是 benchmark/entity fixture。

## N. 100-Claim Offline Replay

| 状态 | Before | After |
|---|---:|---:|
{status_lines}

100 个既存输出全部重放；无模型再生成。9 个输入不计入 qualifier success：3 个零命题/unresolved abstention、4 个 composition semantic invalid、1 个 composition reference invalid、1 个 schema invalid。

## O. Claim-Level Before / After

旧 31 个 qualifier-blocked Claim 中 `{resolved}` 个在 replay 后不再有 qualifier blocker。replay 判定其中无 parser/schema/composition/abstention 独立 blocker；这是“具备 resolve eligibility”，不是声称运行了 resolve transition。After 仍有 `{remaining_claims}` 个 qualifier-blocked Claim；另有 `{len(new_blocked_claims)}` 个原先没有 qualifier blocker 的 Claim 在新判定后出现 blocker：{", ".join(f"`{item}`" for item in new_blocked_claims) or "无"}。净 blocker Claim 计数为 31→{remaining_claims}。

## P. Remaining Genuine Ambiguity / Loss

人工标注的 1 条 TRUE_ATTACHMENT_AMBIGUITY 仍为 AMBIGUOUS。全体 after 状态还包括 {after.get("POTENTIAL_LOSS", 0)} 条 POTENTIAL_LOSS、{after.get("AMBIGUOUS", 0)} 条 AMBIGUOUS；它们不能据此统称“真实语义损失”。本轮未对 100 条做新的独立金标准人工审查，因此确认的 true loss / false-preservation 数量不能从自动 replay 推断；原冻结审计确认的真实 loss=0、人工复核 false-preservation=0 仅作为原审计记录。

## Q. Formation / Composition Regression

本 replay 只调用 qualifier inspector；未调用或修改 composition parser/validator、source-span validator、eligibility producer。冻结诊断输入文件 hash 前后完全一致：`{result["frozen_input_files_unchanged"]}`；保存的 composition/source-span 投影 hash 前后完全一致：`{result["composition_projection_unchanged"]}`（`{result["composition_projection_sha256_before"]}`）。持久化输入含 {result["persisted_composition_metrics"].get("group_count", 0)} 个 composition group，operators={result["persisted_composition_metrics"].get("operator_counts", {})}，nested groups={result["persisted_composition_metrics"].get("nested_group_count", 0)}，source-span compatibility FAIL={result["persisted_composition_metrics"].get("source_span_failure_count", 0)}，OR/XOR leaf leakage={result["persisted_composition_metrics"].get("or_xor_leaf_leakage_count", 0)}。本批无 OR/XOR group（该检查未被样本实际覆盖）；持久化数据在 qualifier replay 中保持原样。hash 不变证明没有改写存档，不等于重新运行 composition 语义测试。

## R. Anti-Overfitting Audit

新增判定按 15 个语义类别与通用载体工作；无 Claim ID、行业、实体或来源专用分支；没有新增模型 prompt/schema 或阈值，也没有为 36 个 warning 各写一条修复。

## S. Provider Calls

**Provider calls during this task: 0.** 输入 artifact 内记录的既有调用为 `{result["persisted_provider_calls_in_inputs"]}`；未访问 Holdout，未创建 Holdout，未进行任何网络模型请求。

## T. Production Behavior Diff

Qualifier preservation 结果有意变化（shadow validation）：状态从字段位置检查转为义务—语义载体—附着证据检查。没有改 prompt/schema/Atomic Proposition formation/typed-value contract/composition/eligibility/source-span/Provider/ResearchLoop。非 qualifier research execution 行为不变；代码中仍存在的其他历史工作树变更属于本任务前已存在状态，没有纳入本次功能范围。

## U. Final Decision

**QUALIFIER SEMANTIC PRESERVATION NEEDS REVISION**

理由：架构与硬负例目标已实现，16/31 旧 blocker 可解除且原始 ambiguity 保守保留；但仍有 17 POTENTIAL_LOSS、5 AMBIGUOUS，并且 36 条人工分类的摘要/逐项数据冲突，当前证据不足以宣告 ready for fresh validation。下一步应先复核残余 loss 与 artifact 计数差异；本报告不触发下一阶段。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "artifacts/v2_1r_bounded_shadow_100/canary_diagnostics",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "artifacts/v2_1r_bounded_shadow_100_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v2_1r_qualifier_semantic_preservation_replay.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/v2_1r_qualifier_semantic_preservation_contract_report.md",
    )
    args = parser.parse_args()
    result = replay(args.input_dir, args.baseline_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(
        _markdown_report(result, _read_json(args.baseline_report)), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "claims_replayed",
                    "warning_events_migrated",
                    "provider_calls_during_replay",
                    "previously_31_qualifier_blocked_claims_resolved_by_replay",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
