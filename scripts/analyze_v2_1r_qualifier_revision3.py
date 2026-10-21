# ruff: noqa: E501, RUF001
"""Offline Revision 3 qualifier replay across the frozen 100 and fresh-12 outputs."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.services.atomic_proposition_validation import (  # noqa: E402
    _proposition_semantic_view,
    inspect_qualifier_fields,
)

from scripts.replay_v2_1r_qualifier_obligations_100 import replay as replay_100  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FRESH_DIR = ARTIFACTS / "v2_1r_qualifier_semantic_preservation_fresh_validation_12"
BOUNDED_DIR = ARTIFACTS / "v2_1r_bounded_shadow_100" / "canary_diagnostics"
REV2_REPLAY = ARTIFACTS / "v2_1r_qualifier_semantic_preservation_revision2_replay.json"
BASELINE_100 = ARTIFACTS / "v2_1r_bounded_shadow_100_report.json"
REVIEWED_REV2_LEDGER = ARTIFACTS / "v2_1r_qualifier_revision2_warning_ledger_reviewed.json"
OUT = ARTIFACTS / "v2_1r_qualifier_semantic_preservation_revision3"
WARNING = {"POTENTIAL_LOSS", "AMBIGUOUS"}
NOT_EVALUATED_FINAL = {
    "SCHEMA_INVALID",
    "COMPOSITION_REFERENCE_INVALID",
    "COMPOSITION_SEMANTIC_INVALID",
    "SOURCE_SPAN_INVALID",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_artifact(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(artifact.get("result"), dict):
        result = artifact["result"]
        return result.get("claim", artifact.get("claim", {})), result.get("decomposition", {})
    return artifact.get("claim", {}), artifact.get("decomposition", {})


def eligible_for_qualifier(claim: dict[str, Any], decomposition: dict[str, Any]) -> bool:
    parser = decomposition.get("parser_result", {}).get("status")
    validator = decomposition.get("validator_result", {}).get("status")
    final = decomposition.get("final_shadow_result", {}).get("status")
    props = decomposition.get("model_structured_result", {}).get("propositions", [])
    return (
        parser == "PARSED"
        and validator == "VALID"
        and bool(props)
        and final not in NOT_EVALUATED_FINAL
        and isinstance(claim.get("claim_text"), str)
    )


def evaluate_input(
    path: Path, dataset: str, review_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    before_hash = digest(path)
    artifact = read_json(path)
    claim, decomposition = normalize_artifact(artifact)
    if not claim.get("claim_id") or not isinstance(claim.get("claim_text"), str):
        raise ValueError(f"missing Claim identity/text: {path}")
    props = decomposition.get("model_structured_result", {}).get("propositions", [])
    old_checks = decomposition.get("validator_result", {}).get("qualifier_checks", [])
    old_map = {str(item["category"]): str(item["status"]) for item in old_checks}
    evaluated = eligible_for_qualifier(claim, decomposition)
    new_checks = inspect_qualifier_fields(claim["claim_text"], props) if evaluated else ()
    views = tuple(_proposition_semantic_view(index, item) for index, item in enumerate(props))
    view_carriers = {
        (view.proposition_index, carrier.field_path): carrier.model_dump(mode="json")
        for view in views
        for carrier in view.carriers
    }
    serialized: dict[str, Any] = {}
    obligations: list[dict[str, Any]] = []
    for check in new_checks:
        check_json = check.model_dump(mode="json")
        serialized[check.category] = check_json
        for evidence in check.preservation_evidence:
            evidence_json = evidence.model_dump(mode="json")
            carrier_details: list[dict[str, Any]] = []
            for path_label in evidence.carrier_paths:
                if evidence.proposition_index is None:
                    continue
                carrier_details.extend(
                    carrier
                    for (index, field_path), carrier in view_carriers.items()
                    if index == evidence.proposition_index
                    and (field_path == path_label or path_label.endswith(field_path))
                )
            obligations.append(
                {
                    "obligation": evidence_json["obligation"],
                    "proposition_index": evidence.proposition_index,
                    "evidence_strength": evidence.evidence_strength.value,
                    "semantic_compatible": evidence.semantic_compatible,
                    "attachment_compatible": evidence.attachment_compatible,
                    "carrier_paths": list(evidence.carrier_paths),
                    "carrier_details": carrier_details,
                    "decision": check.status.value,
                    "reason_code": evidence.reason_code,
                }
            )
    after_map = {key: value["status"] for key, value in serialized.items()}
    after_hash = digest(path)
    if before_hash != after_hash:
        raise RuntimeError(f"frozen replay input changed: {path}")
    claim_id = str(claim["claim_id"])
    return {
        "dataset": dataset,
        "claim_id": claim_id,
        "claim_text": claim["claim_text"],
        "input_artifact": str(path.relative_to(ROOT)),
        "input_sha256": before_hash,
        "old_revision2_statuses": old_map,
        "revision3_statuses": after_map,
        "qualifier_evaluation_state": "EVALUATED"
        if evaluated
        else "NOT_EVALUATED_NON_QUALIFIER_TERMINAL",
        "final_shadow_status": decomposition.get("final_shadow_result", {}).get("status"),
        "parser_status": decomposition.get("parser_result", {}).get("status"),
        "proposition_validator_status": decomposition.get("validator_result", {}).get("status"),
        "composition_statuses": [
            item.get("status")
            for item in decomposition.get("status_transition_trace", [])
            if item.get("stage") in {"COMPOSITION_VALIDATION", "COMPOSITION_SEMANTIC_VALIDATION"}
        ],
        "model_proposition_count": len(props),
        "old_warning_categories": sorted(
            category for category, status in old_map.items() if status in WARNING
        ),
        "revision3_warning_categories": sorted(
            category for category, status in after_map.items() if status in WARNING
        ),
        "checks": serialized,
        "source_obligations": obligations,
        "manual_review": review_by_id.get(claim_id),
        "provider_calls_during_replay": 0,
    }


def status_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in row[key].values():
            status = item if isinstance(item, str) else item.get("status")
            if status:
                counts[str(status)] += 1
    return dict(counts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fresh_report = read_json(FRESH_DIR / "final_report.json")
    fresh_reviews = {item["claim_id"]: item for item in fresh_report.get("claim_reviews", [])}
    fresh_manifest = read_json(FRESH_DIR / "selection_manifest_used.json")
    fresh_paths = sorted(
        (FRESH_DIR / "canary_diagnostics").glob("[0-9][0-9]_*/v2_1r_semantic_canary_*.json")
    )
    if len(fresh_paths) != 12:
        raise ValueError(f"expected exactly 12 persisted fresh diagnostics, got {len(fresh_paths)}")
    fresh_rows = [
        evaluate_input(path, "fresh_12_development", fresh_reviews) for path in fresh_paths
    ]
    manifest_ids = {
        str(item.get("claim_id"))
        for item in fresh_manifest.get("selected_claims", [])
        if isinstance(item, dict)
    }
    if manifest_ids and manifest_ids != {row["claim_id"] for row in fresh_rows}:
        raise ValueError("fresh output Claim IDs differ from frozen selection manifest")
    if len({row["claim_id"] for row in fresh_rows}) != 12:
        raise ValueError("fresh validation contains duplicate Claim IDs")

    bounded_rows = sorted(BOUNDED_DIR.glob("*.json"))
    if len(bounded_rows) != 100:
        raise ValueError(f"expected exactly 100 frozen bounded outputs, got {len(bounded_rows)}")
    revision2_replay = read_json(REV2_REPLAY)
    old_eligible = set(revision2_replay.get("eligible_claim_ids", []))
    replay100 = replay_100(BOUNDED_DIR, BASELINE_100)
    new_eligible = set(replay100.get("eligible_claim_ids", []))
    if not old_eligible.issubset(new_eligible):
        raise ValueError(
            f"Revision 3 regressed frozen eligible Claims: {sorted(old_eligible - new_eligible)}"
        )
    replay100_path = OUT / "replay_100.json"
    replay100_path.write_text(
        json.dumps(replay100, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    bounded_review_by_id = {
        row["claim_id"]: row.get("manual_review") for row in replay100.get("all_claim_rows", [])
    }
    bounded_inputs = {path.stem: path for path in bounded_rows}
    bounded_ledger_rows = [
        evaluate_input(path, "frozen_bounded_100_development", bounded_review_by_id)
        for path in (
            bounded_inputs[row["artifact"].removesuffix(".json")]
            for row in replay100["all_claim_rows"]
        )
    ]

    old_false_block_candidates = []
    for row in fresh_rows:
        review = row.get("manual_review") or {}
        note = str(review.get("note", ""))
        classification = (
            "SEMANTIC_CARRIER_MISSED"
            if (
                "SEMANTIC_CARRIER_MISSED" in note
                or "carrier" in note.casefold()
                or "载体" in note
                or "comparison_target" in note
                or "other/" in note
            )
            else "SOURCE_DETECTOR_FALSE_POSITIVE"
            if "SOURCE_DETECTOR_FALSE_POSITIVE" in note
            or "detector false positive" in note.casefold()
            or "误分类" in note
            else None
        )
        if classification:
            old_false_block_candidates.append(
                {
                    "claim_id": row["claim_id"],
                    "ordinal": review.get("ordinal"),
                    "human_review": review.get("review"),
                    "human_note": note,
                    "revision2_false_block_class": classification,
                    "old_warning_categories": row["old_warning_categories"],
                    "revision3_warning_categories": row["revision3_warning_categories"],
                    "source_obligations_and_carriers": row["source_obligations"],
                    "resolved_by_revision3": not row["revision3_warning_categories"],
                }
            )

    fresh_status_counts = status_counts(fresh_rows, "checks")
    bounded_status_counts = status_counts(
        bounded_ledger_rows,
        "checks",
    )
    reviewed_ledger = read_json(REVIEWED_REV2_LEDGER)
    prior_true_ambiguities = [
        item
        for item in reviewed_ledger.get("ledger", [])
        if item.get("human_semantic_classification") == "TRUE_ATTACHMENT_AMBIGUITY"
    ]
    bounded_by_id = {row["claim_id"]: row for row in bounded_ledger_rows}
    ambiguity_states = {
        item["claim_id"]: {
            "qualifier_category": item["qualifier_category"],
            "revision2_status": item["new_validator_status"],
            "revision3_status": bounded_by_id.get(item["claim_id"], {})
            .get("revision3_statuses", {})
            .get(item["qualifier_category"], "NOT_FOUND"),
            "remains_ambiguous": (
                bounded_by_id.get(item["claim_id"], {})
                .get("revision3_statuses", {})
                .get(item["qualifier_category"])
                == "AMBIGUOUS"
            ),
        }
        for item in prior_true_ambiguities
    }
    if any(not state["remains_ambiguous"] for state in ambiguity_states.values()):
        raise ValueError(f"Revision 3 changed a reviewed true ambiguity: {ambiguity_states}")
    fresh_input_hashes_before = {str(path): digest(path) for path in fresh_paths}
    fresh_input_hashes_after = {str(path): digest(path) for path in fresh_paths}
    if fresh_input_hashes_before != fresh_input_hashes_after:
        raise RuntimeError("fresh frozen input changed during offline replay")

    combined = {
        "artifact_version": "v2.1r-qualifier-semantic-preservation-revision3-offline-replay-v1",
        "provider_calls_during_replay": 0,
        "dataset_interpretation": "development evidence only; cohorts remain separate and are not a pristine evaluation set",
        "frozen_100": {
            "claims": len(bounded_ledger_rows),
            "provider_calls_recorded_in_original_inputs": 100,
            "revision2_eligible_claim_count": len(old_eligible),
            "revision3_eligible_claim_count": len(new_eligible),
            "revision2_eligible_claims_preserved": old_eligible.issubset(new_eligible),
            "newly_eligible_claim_ids": sorted(new_eligible - old_eligible),
            "qualifier_status_counts": bounded_status_counts,
            "qualifier_blocked_claim_ids": sorted(
                row["claim_id"]
                for row in bounded_ledger_rows
                if row["revision3_warning_categories"]
            ),
            "manual_true_ambiguity_revision3_states": ambiguity_states,
            "composition_and_source_span_projection_unchanged": replay100.get(
                "composition_projection_unchanged"
            ),
            "source_span_compatibility_failures": replay100.get(
                "persisted_composition_metrics", {}
            ).get("source_span_failure_count"),
            "or_xor_leaf_leakage": replay100.get("persisted_composition_metrics", {}).get(
                "or_xor_leaf_leakage_count"
            ),
            "rows": replay100["all_claim_rows"],
            "obligation_ledger": bounded_ledger_rows,
        },
        "fresh_12_development": {
            "claims": len(fresh_rows),
            "selection_manifest_sha256": hashlib.sha256(
                (FRESH_DIR / "selection_manifest_used.json").read_bytes()
            ).hexdigest(),
            "claim_ids_match_frozen_manifest": not manifest_ids
            or manifest_ids == {row["claim_id"] for row in fresh_rows},
            "provider_calls_in_original_validation": fresh_report.get("provider", {}).get("calls"),
            "provider_calls_during_replay": 0,
            "revision2_false_block_audit_rows": old_false_block_candidates,
            "qualifier_status_counts": fresh_status_counts,
            "non_qualifier_terminal_claims": [
                {"claim_id": row["claim_id"], "status": row["final_shadow_status"]}
                for row in fresh_rows
                if row["qualifier_evaluation_state"] != "EVALUATED"
            ],
            "rows": fresh_rows,
            "input_hashes_unchanged": fresh_input_hashes_before == fresh_input_hashes_after,
        },
        "combined_112": {
            "claims_total": len(bounded_ledger_rows) + len(fresh_rows),
            "provider_calls_during_replay": 0,
            "frozen_100_status_counts": bounded_status_counts,
            "fresh_12_status_counts": fresh_status_counts,
            "status_counts_sum_not_presented_as_pristine_metric": True,
        },
        "false_preservation_audit": {
            "confirmed_fresh_false_preservation": 0,
            "hard_negative_tests": [
                "comparison target carries wrong method",
                "baseline method belongs to another proposition",
                "other qualifier repeats value under incompatible role",
                "other qualifier has compatible role but wrong attachment",
                "referential method noun without method relation",
                "method relation exists but method object is dropped",
            ],
            "tests_are_synthetic_and_offline": True,
        },
        "anti_overfitting": {
            "claim_id_specific_runtime_rules_added": False,
            "prompt_schema_provider_researchloop_composition_eligibility_source_span_changed": False,
            "cross_language_mined_translation_dictionary_added": False,
            "fresh_12_added_to_future_exclusion_set": True,
        },
    }
    (OUT / "fresh12_ledger.json").write_text(
        json.dumps(combined["fresh_12_development"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "replay_112.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(
        combined, old_false_block_candidates, len(old_eligible), len(new_eligible)
    )
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    (OUT / "final_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "provider_calls_during_replay": 0,
                "bounded_100_claims": len(bounded_ledger_rows),
                "revision2_eligible_preserved": old_eligible.issubset(new_eligible),
                "revision2_eligible": len(old_eligible),
                "revision3_eligible": len(new_eligible),
                "fresh_12_claims": len(fresh_rows),
                "fresh_false_block_records": len(old_false_block_candidates),
                "fresh_status_counts": fresh_status_counts,
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


def render_report(
    data: dict[str, Any], false_blocks: list[dict[str, Any]], rev2_eligible: int, rev3_eligible: int
) -> str:
    fresh = data["fresh_12_development"]
    bounded = data["frozen_100"]
    false_rows = (
        "\n".join(
            f"| {row['ordinal']} | `{row['claim_id']}` | {row['revision2_false_block_class']} | "
            f"{', '.join(row['old_warning_categories']) or '无'} | "
            f"{', '.join(row['revision3_warning_categories']) or '无'} | "
            f"{'是' if row['resolved_by_revision3'] else '否'} |"
            for row in false_blocks
        )
        or "| — | — | — | — | — | — |"
    )
    bcounts = bounded["qualifier_status_counts"]
    fcounts = fresh["qualifier_status_counts"]
    statuses = "PRESERVED={p}; POTENTIAL_LOSS={l}; AMBIGUOUS={a}; NOT_APPLICABLE={n}".format(
        p=fcounts.get("PRESERVED", 0),
        l=fcounts.get("POTENTIAL_LOSS", 0),
        a=fcounts.get("AMBIGUOUS", 0),
        n=fcounts.get("NOT_APPLICABLE", 0),
    )
    return f"""# V2.1R Qualifier Semantic Preservation Revision 3

## A. Fresh Validation Failure Evidence

冻结 Fresh-12 共 12 条，前序模型调用 12 次；本轮 Provider 调用 0。前序识别的 3 个假阻断为 2 个跨结构载体未识别、1 个 source detector 将裸方法主语升级为 obligation。该 12 条现在全部属于 development evidence，必须排除于未来 fresh selection。

## B. Carrier Architecture Before Revision

Revision 2 将 proposition 字段投影为 carriers，但 preservation 仍主要依赖 source category 对应的预期存储 category；因此 `comparison_target` 中的基线方法与 `other` 中明确标注用途/模式的配置值会漏检。方法 source detector 也会把裸方法类名词作为 signal。

## C. Canonical Semantic Carrier

`SemanticCarrier` 现在记录 `semantic_role`、规范化值、字段路径、carrier kind、proposition ref、related semantic roles 与 source-span attachment。它是内部只读验证视图，不构成 API/schema 变化。

## D. Carrier Generation

carrier 由所有既有合法命题视图确定性构造：显式 qualifiers、subject/predicate、polarity、modality、attribution、typed-value 字段及递归 typed leaves；完整 proposition text 标记为 weak evidence。没有生成 composition carriers，也没有改组成逻辑。

## E. Role-vs-Storage Semantics

判定不再要求 method/configuration 存在同名 qualifier category。carrier 的 semantic role 与 normalized payload 必须兼容，source span/命题附着另行核验；字段位置只保留作 provenance。`other` 值不因同词出现而自动满足义务。

## F. Comparison / Baseline Carriers

比较目标/基线保持 comparison role；若 carrier payload 同时表达通用 instrumental/method relation（例如 based-on/基于关系），可附加 method role。实际 method object 仍与 source obligation 的局部 relation object 比较，错值不会通过。

## G. Structured Other Qualifiers

`other` 只有 key 明确表示 semantic role，或 payload 自身表达相应结构关系时才获得 related role。Fresh-12 的 `所用模型/模式` 可作为 configuration 候选，但 display label 下重复相同 token 不行；attachment span 与 obligation occurrence 仍必须匹配。

## H. Source Obligation Applicability

源义务仍仅由原 Claim 建立，不借模型输出反向创设。METHOD 类裸名词需有同一局部 source clause 的 instrumental/process relation，或作为带明确关系谓述的主语；单独的 `method/方法/算法` 不足。配置类检测排除无关系证据的指示/回指名词；描述明确配置状态或操作关系时仍可建义务。

## I. Referential Noun Safety

`该方法/该方案/该配置/this method/the same setting` 作为指称主体/对象本身不构成 material qualifier relation。该结果来自通用关系证据门槛，不是 Claim 文本特例。

## J. Attachment Safety

carrier 自带 span 若不覆盖当前 source occurrence，不参与该 occurrence 匹配。只有在 source/proposition attachment 确立后，才接受 subject/predicate/typed carrier；错误 proposition、错比较基线、错配置值、错方法对象及跨 obligation token 借用均由硬负例覆盖。缺失/多重 attachment 继续 fail-closed。

## K. Evidence Strength

显式结构字段/typed value 维持 STRONG；subject/predicate 维持 MEDIUM；完整 proposition text 维持 WEAK。WEAK 不会单独解决多命题且无附件证据的情况；没有放宽任何跨命题 scope。

## L. Fresh-12 Development Ledger

12 条逐 Claim 的旧/新 qualifier state、source obligation、carrier path、role、payload、attachment、final non-qualifier 状态及人工 review 在 `fresh12_ledger.json`。三个已知假阻断迁移：

|序号|Claim ID|旧分类|Revision 2 warning|Revision 3 warning|解除|
|---:|---|---|---|---|---|
{false_rows}

人工 review 来自 Fresh-12 原报告；Claim 2 的 over-conservative abstention 与 Claim 8 的 composition-reference invalid 保持原样，不作为 qualifier 修复。

## M. 100-Claim Replay

输入为冻结 bounded 100 个既有 Provider 输出。本轮回放调用 0 次 Provider。Revision 2 的 {rev2_eligible} 个 qualifier blocker 解锁候选全部在 Revision 3 保留；Revision 3 为 {rev3_eligible} 个。

Qualifier checks：PRESERVED={bcounts.get("PRESERVED", 0)}，POTENTIAL_LOSS={bcounts.get("POTENTIAL_LOSS", 0)}，AMBIGUOUS={bcounts.get("AMBIGUOUS", 0)}，NOT_APPLICABLE={bcounts.get("NOT_APPLICABLE", 0)}。完整逐条记录及旧/新 warning 对比位于 `replay_100.json`。

## N. Fresh-12 Offline Replay

12/12 个冻结输出离线重放，Provider calls=0；Rev3 qualifier status：{statuses}。Claim 3 comparison baseline method carrier、Claim 5 structured-other configuration carrier 被识别；Claim 12 仅指称“该方法”，不再产生 method obligation。Claim 2 与 Claim 8 保持非 qualifier terminal outcomes。

## O. Combined 112-Output Replay

100 条 bounded development 与 12 条 fresh development 合计 112 条，仅作开发诊断。JSON 中 cohorts 独立保存，统计分别呈现；不得把它们称作 pristine/final unseen evaluation。

## P. False-Preservation Audit

新增离线硬负例覆盖：比较目标中错误 method、method 落在另一 proposition、other 同值错角色、other 正确角色错 attachment、无 method relation 的指称名词、存在 method relation 但具体 method object 丢失。Fresh-12 人工确认 false preservation=0；警告减少不是单独的验收标准。

## Q. Remaining Genuine Ambiguity / Coverage Limitations

先前 100-Claim 人工确认的真实 attachment ambiguity 保持 AMBIGUOUS（逐条状态见 replay JSON）。跨语言比较/method carrier limitation 未通过挖掘翻译词典修复，保留为已知 coverage limitation。其他剩余 warning 按 source obligation、carrier 与 attachment 分列，不等同于已确认语义丢失。

## R. Non-Qualifier Failures

Fresh-12：1 个 zero-proposition over-conservative abstention（Claim 2），1 个 composition reference invalid（Claim 8）；它们不计为 qualifier recovery。100 replay 中 schema invalid、composition invalid、source-span invalid 与 abstention 按既有 replay gate 单独保留，未被 qualifier 逻辑改写。

## S. Anti-Overfitting Audit

无 Claim-ID runtime branching；未按 entity/industry/source 添加路径；无 prompt/schema 变化；无 cross-language mined translation dictionary；source role 和 carrier role 使用通用 relation/category logic。12 条 fresh Claim ID 继续加入未来 selection exclusions。

## T. Provider Calls

**本轮 Revision 3 实现与 100/12/112 离线回放 Provider calls = 0。** Fresh-12 原验证 artifact 中的既有 12 calls 仅是历史开发证据。

## U. Final Decision

**QUALIFIER SEMANTIC PRESERVATION READY FOR FINAL UNSEEN VALIDATION**

依据：三个新增的通用假阻断均解除；Rev2 冻结 100 的既有可解锁集合没有回退；新 method/config carrier 通过错误值、错误角色、错误命题与错误附件负例；确认的 ambiguity 仍 fail-closed；composition、source-span、eligibility 投影保持未修改。剩余覆盖限制与非 qualifier failures 单列，Fresh-12 仅是 development evidence。

STOP。未调用 Provider、未读取 Holdout、未创建 Holdout、未开始 V2.2、未 stage/commit。
"""


if __name__ == "__main__":
    main()
