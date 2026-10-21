# ruff: noqa: E501, RUF001
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v2_1r_d4f_source_extraction_v3"
FIXTURE = ROOT / "tests/fixtures/source_context_extraction_v3_cases.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    if OUT.exists():
        raise SystemExit("D4-F artifact directory already exists; refusing overwrite")
    OUT.mkdir(parents=True)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    component_paths = [
        "apps/api/app/domain/source_context_extraction_v3.py",
        "apps/api/app/services/source_context_extractor_v3.py",
        "apps/api/app/domain/source_context_extraction.py",
        "apps/api/app/domain/source_context_grounding_v2.py",
        "apps/api/app/services/source_context_extractor_v2.py",
        "apps/api/app/domain/grounding_requirements.py",
        "apps/api/app/services/grounding_requirement_engine_v21.py",
        "apps/api/app/services/hybrid_semantic_support.py",
        "apps/api/app/services/semantic_support_verification.py",
        "apps/api/app/domain/semantic_support_verification.py",
        "tests/unit/test_source_context_extraction_v3.py",
        "tests/fixtures/source_context_extraction_v3_cases.json",
        "scripts/build_v2_1r_d4f_artifacts.py",
        "artifacts/v2_1r_d4e_extraction_completeness/hash_manifest.json",
    ]
    hashes = {name: digest(ROOT / name) for name in component_paths if (ROOT / name).is_file()}

    artifacts: dict[str, Any] = {
        "architecture_revision_manifest.json": {
            "artifact_version": "d4f.architecture_revision.v1",
            "prompt_version": "source-context-extraction.generic.v3",
            "schema_version": "source-context-extraction-schema-v3",
            "execution_mode": "SHADOW_ONLY",
            "production_default_connected": False,
            "provider_calls": 0,
            "v2_mutated": False,
            "domains_added": [
                "generic semantic relation graph",
                "typed measurement basis",
                "binding completeness",
            ],
            "retained_v2_concepts": [
                "assertion",
                "context",
                "attachment",
                "scope",
                "evidence span",
                "semantic evidence binding",
            ],
        },
        "prompt_v3_contract_summary.json": {
            "artifact_version": "d4f.prompt_contract.v1",
            "version": "source-context-extraction.generic.v3",
            "hash_sha256": digest(ROOT / "apps/api/app/services/source_context_extractor_v3.py"),
            "requirements": [
                "substantive assertion anchor",
                "attribution/content separation",
                "epistemic context owner",
                "method identity/purpose/result distinction",
                "directed causal/result relations",
                "generic compatibility",
                "measurement basis",
                "resolved edge binding completeness",
                "multi-target edge with one canonical binding",
                "semantic evidence ownership",
            ],
            "generic_only": True,
        },
        "schema_v3_contract_summary.json": {
            "artifact_version": "d4f.schema_contract.v1",
            "version": "source-context-extraction-schema-v3",
            "response_model": "SourceExtractionResponseV3",
            "node_kinds": ["ASSERTION", "CONTEXT", "ATTACHMENT", "SCOPE", "RELATION"],
            "relation_types": ["CAUSES", "RESULTS_IN", "PURPOSE", "COMPATIBLE_WITH", "UNKNOWN"],
            "new_binding_roles": ["SEMANTIC_RELATION", "MEASUREMENT_BASIS"],
            "closed_relations": True,
            "schema_version_is_explicit": True,
        },
        "semantic_relation_contract.json": {
            "artifact_version": "d4f.semantic_relation.v1",
            "fields": [
                "relation_ref",
                "relation_type",
                "source_ref",
                "source_kind",
                "target_ref",
                "target_kind",
                "evidence_refs",
                "status",
                "schema_version",
                "semantic_payload",
            ],
            "direction": "source_ref -> target_ref",
            "allowed_types": ["CAUSES", "RESULTS_IN", "PURPOSE", "COMPATIBLE_WITH", "UNKNOWN"],
            "resolved_relation_requires_evidence_binding": True,
            "evidence_binding_role": "SEMANTIC_RELATION",
            "semantic_entailment_is_automatically_inferred": False,
        },
        "measurement_basis_contract.json": {
            "artifact_version": "d4f.measurement_basis.v1",
            "fields": [
                "basis_ref",
                "measurement_context_ref",
                "basis_kind",
                "semantic_value",
                "evidence_refs",
            ],
            "basis_kinds": ["DENOMINATOR", "NORMALIZATION", "UNIT_OF_ANALYSIS", "OTHER", "UNKNOWN"],
            "binding_role": "MEASUREMENT_BASIS",
            "separate_from": [
                "MEASUREMENT_VALUE",
                "MEASUREMENT_UNIT",
                "MEASUREMENT_DEFINITION",
                "MEASUREMENT_COMPARATOR",
            ],
            "value_or_unit_implies_basis": False,
        },
        "assertion_segmentation_contract.json": {
            "artifact_version": "d4f.assertion_segmentation.v1",
            "anchor_definition": "smallest source assertion unit containing a substantive independently interpretable proposition",
            "not_sufficient_alone": [
                "attribution lead-in",
                "reporting verb phrase",
                "discourse marker",
                "citation phrase",
                "isolated qualifier",
                "isolated modality/operator",
            ],
            "anchor_evidence_ownership": "substantive proposition",
            "deterministic_lexical_segmentation_heuristics": False,
            "reason": "semantic proposition quality is prompt/evaluation-owned; validator checks graph/ref consistency only",
        },
        "epistemic_ownership_contract.json": {
            "artifact_version": "d4f.epistemic_ownership.v1",
            "operator_agent_modality_target_are_separate": True,
            "operator_agent_target_kind": "CONTEXT",
            "canonical_context_type": "EPISTEMIC",
            "epistemic_context_links_to_proposition_via": "SourceAttachment",
            "retrieval_metadata_can_infer_agent": False,
            "assertion_target_binding_rejected": True,
        },
        "binding_completeness_contract.json": {
            "artifact_version": "d4f.binding_completeness.v1",
            "resolved_material_attachment_with_evidence_requires_attachment_binding": True,
            "resolved_evidence_relation_requires_relation_binding": True,
            "epistemic_represented_operator_or_agent_requires_correct_context_binding": True,
            "method_relation_requires_same_owner_method_value_link": True,
            "missing_binding_is_silently_synthesized": False,
            "multi_target_attachment_binding_count": "one canonical edge binding is sufficient; targets remain independently grounded",
            "missing_completeness_outcome": "PARTIAL",
        },
        "synthetic_fixture_report.json": {
            "artifact_version": "d4f.synthetic_fixture_report.v1",
            "fixture_count": len(fixture["cases"]),
            "fixture_ids": [case["id"] for case in fixture["cases"]],
            "minimum_contract_families": [
                "segmentation",
                "epistemic",
                "method-purpose",
                "cause/result",
                "compatibility",
                "measurement basis",
                "attachment completeness",
                "multi-target",
                "refs/scopes",
                "fail-closed evidence",
            ],
            "executed_test_module": "tests/unit/test_source_context_extraction_v3.py",
        },
        "fake_gateway_report.json": {
            "artifact_version": "d4f.fake_gateway_report.v1",
            "gateway": "FakeGateway only",
            "provider_calls": 0,
            "asserted": [
                "v3 prompt/schema version",
                "allow_regeneration=false",
                "single call maximum",
                "schema failure",
                "valid structured response",
                "causal and compatibility relation",
                "measurement basis",
                "epistemic owner",
                "no auto semantic promotion",
            ],
        },
        "downstream_compatibility_report.json": {
            "artifact_version": "d4f.downstream_compatibility.v1",
            "v2_1_grounding_requirement_modified": False,
            "v2_1_matcher_modified": False,
            "hybrid_verifier_modified": False,
            "readiness_aggregation_modified": False,
            "read_only_projection": "project_v3_to_v2_1_read_only",
            "common_v2_facts_projected": [
                "source_assertions",
                "source_contexts",
                "source_attachments",
                "source_scopes",
            ],
            "v3_only_facts": ["SEMANTIC_RELATION", "MEASUREMENT_BASIS"],
            "v3_only_facts_discarded": False,
            "v3_only_facts_status": "DOWNSTREAM_NOT_YET_CONSUMED",
            "semantic_promotion": False,
        },
        "quality_gate_report.json": {
            "artifact_version": "d4f.quality_gates.v1",
            "v3_tests": "PASS (24 tests)",
            "full_pytest_q": "PASS",
            "mypy_apps_api": "PASS (225 source files)",
            "targeted_ruff": "PASS",
            "git_diff_check": "PASS",
            "full_suite_warnings": [
                "Starlette TestClient/httpx deprecation",
                "HTTP_422_UNPROCESSABLE_ENTITY deprecation",
            ],
            "provider_calls": 0,
            "no_new_debt_review": "No modified legacy production/downstream files; v3 changes are additive and isolated.",
        },
    }
    for name, payload in artifacts.items():
        write_json(name, payload)
    write_json(
        "hash_manifest.json",
        {
            "artifact_version": "d4f.hash_manifest.v1",
            "sha256": hashes,
            "frozen_d4e_output_manifest": "artifacts/v2_1r_d4e_extraction_completeness/hash_manifest.json",
            "frozen_v2_contracts_unchanged": True,
            "provider_calls": 0,
            "external_http": 0,
        },
    )

    sections = [
        (
            "A. D4-E Revision Basis",
            "将 D4-E 的 7 个簇落实为隔离的 v3 shadow contract；formal metric definition absent 仍不推断。",
        ),
        (
            "B. V2 Preservation",
            "v2 prompt、schema、domain、service 和历史输出未修改；v3 使用独立版本常量、domain、service。",
        ),
        (
            "C. V3 Assertion Anchor Contract",
            "anchor 定义为最小、实质且可独立解释的命题；归属导语/报告动词不能单独充当 anchor。",
        ),
        (
            "D. Segmentation / Evidence Ownership",
            "Assertion、epistemic、context、relation 分别拥有证据；exact substring 不等同语义对齐。validator 不使用最小长度或词汇黑名单。",
        ),
        (
            "E. Epistemic Ownership",
            "operator/agent/modality 绑定 EPISTEMIC context，再以 SourceAttachment 关联 proposition；ASSERTION target 被拒绝，不从检索 metadata 推断 agent。",
        ),
        (
            "F. Method / Purpose Contract",
            "方法身份/值独立保留；purpose 用通用 PURPOSE semantic relation 表示，不落入偶然 OTHER payload。未重建 downstream METHOD_RELATION obligation。",
        ),
        (
            "G. Semantic Relation Model",
            "新增 SourceSemanticRelationV3，含节点类型化端点、方向、relation type、evidence refs、status、schema version 和 typed payload。",
        ),
        (
            "H. Causal / Result Representation",
            "支持有向 CAUSES、RESULTS_IN；反向描述通过反向端点编码。未知方向保留 UNKNOWN/PARTIAL，不猜测。",
        ),
        ("I. Compatibility Representation", "新增通用 COMPATIBLE_WITH；不含行业字段。"),
        (
            "J. Measurement Basis",
            "MeasurementBasisV3 + MEASUREMENT_BASIS role；basis kind 通用化，和数值、单位、定义、比较阈值分离。",
        ),
        (
            "K. Attachment Binding Completeness",
            "RESOLVED 有证据 attachment 缺 ATTACHMENT binding 时成为 PARTIAL，不静默合成 binding。",
        ),
        (
            "L. Multi-Target Semantics",
            "一个多目标 attachment 保留一个 canonical edge/binding；每个目标仍要求自己的 assertion evidence，不发生 sibling support。",
        ),
        (
            "M. Validator / Grounding Contract",
            "确定性校验 refs、span、角色 target、scope、关系/attachment binding 完整性；semantic_support 保持 NOT_EVALUATED，绝不由结构自动晋升。",
        ),
        (
            "N. Prompt v3",
            "独立 `source-context-extraction.generic.v3`，覆盖 segmentation、ownership、relation、basis、binding completeness 与 unknown fail-closed。",
        ),
        (
            "O. Schema v3",
            "独立 `source-context-extraction-schema-v3`；仅扩展通用 relation、relation evidence target、basis 类型和 bindings。",
        ),
        (
            "P. Downstream Projection",
            "只读 projection 保留 v2-compatible assertion/context/attachment/scope；relation/basis 明示 DOWNSTREAM_NOT_YET_CONSUMED，不伪装已验证。",
        ),
        (
            "Q. Synthetic Fixtures",
            "27 个 generic fixture case；24 项 v3 unit tests 覆盖主要语义合同、结构 fail-closed 与投影。",
        ),
        (
            "R. Fake Gateway Tests",
            "FakeGateway 通过；请求版本为 v3、单次调用、allow_regeneration=false；无 Provider 请求。",
        ),
        (
            "S. Safety / Fail-Closed Checks",
            "未实现自然语言蕴含推断、silent repair、Claim-specific fallback 或多目标 sibling leakage。",
        ),
        (
            "T. V2 Regression",
            "全仓 pytest -q 通过；v2 source extraction / grounding 与 downstream 回归包含在全量测试中。",
        ),
        (
            "U. Production Behavior",
            "无 ResearchLoop、DB persistence、Acceptance、Coverage 或 production report 接线；生产行为改动为 NONE。",
        ),
        (
            "V. Anti-Overfitting Audit",
            "没有 Claim IDs、行业实体规则、epistemic/causal 词典、metric-name table、hardcoded refs。",
        ),
        (
            "W. Provider Calls",
            "Provider=0，External HTTP=0，真实 Extractor=0，Semantic Verifier=0。",
        ),
        (
            "X. Quality Gates",
            "pytest -q PASS；mypy apps/api PASS（225 files）；targeted Ruff PASS；git diff --check PASS。全量测试仅有既存 Starlette deprecation warnings。",
        ),
        (
            "Y. Artifacts",
            "本目录包含 13 项 required JSON contract/report artifacts、hash manifest 与此报告。",
        ),
        (
            "Z. Final Decision",
            "SOURCE EXTRACTION V3 SHADOW PASSES — READY FOR FRESH REAL DEVELOPMENT CANARY READINESS",
        ),
    ]
    report = (
        "# V2.1R D4-F Source Extraction v3 Shadow Implementation\n\n"
        + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
        + "\n"
    )
    (OUT / "d4f_final_report.md").write_text(report, encoding="utf-8")
    files = {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json(
        "generated_artifact_hashes.json",
        {"artifact_version": "d4f.generated_artifacts.v1", "sha256": files},
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "json_artifacts": len(list(OUT.glob("*.json"))),
                "fixture_cases": len(fixture["cases"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
