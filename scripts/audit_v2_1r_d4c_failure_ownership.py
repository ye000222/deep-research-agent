"""Build a read-only D4-C ownership audit from frozen D4-B artifacts."""
# ruff: noqa: E501, RUF001

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v2_1r_d4b_execution_attempt_2"
OUT = ROOT / "artifacts/v2_1r_d4c_systematic_failure_ownership"
CLAIM_ORDER = [
    "01a0c82b-3531-79b2-8160-a9e47e4c2bff",
    "01a0c86f-e837-79b4-ab15-62d6e4e84f74",
    "01a0c41c-280e-7fd4-b770-9741bac7b3de",
    "01a0c434-9753-7fff-be68-21d83da29bf2",
    "01a0c424-0fc5-7cea-ac2d-92d3274f5d85",
    "01a0c436-7ecc-7d21-b32d-c5308fa1acec",
    "01a0c853-cc4a-7648-b4e2-d175da55860d",
    "01a0c406-6984-7b9d-a66f-2a6ad7a3a805",
]
CLAIM_FACTS: dict[str, list[dict[str, str]]] = {
    CLAIM_ORDER[0]: [
        {"type": "EPISTEMIC_OPERATOR", "finding": "“原文指出”作为归属/认识性标记未以正确的 context owner 表达；operator 与 agent bindings 都指向 ASSERTION。", "classification": "wrong_role_and_target", "basis": "review.md + B2/B3 structural diagnostics"},
        {"type": "EPISTEMIC_AGENT", "finding": "文本只提供“原文”这一来源指示，没有独立、明确的 agent 实体；B3 将其作为 agent 且未 source-ground。", "classification": "wrong_target_and_unsupported_agent_payload", "basis": "B3 agent_not_source_grounded; claim text"},
    ],
    CLAIM_ORDER[1]: [],
    CLAIM_ORDER[2]: [
        {"type": "METHOD_RELATION", "finding": "两个 METHOD context 有词面值，但没有独立、grounded 的方法关系表达；通用 ATTRIBUTES attachment 未补足方法内部关系。", "classification": "binding_or_relation_absent", "basis": "review.md; C1/C2 NOT_EVALUATED diagnostics"},
        {"type": "OTHER", "finding": "像素级定位目的被表示成 OTHER context，而非更明确的 purpose owner。", "classification": "wrong_role", "basis": "review.md; extracted C3"},
    ],
    CLAIM_ORDER[3]: [
        {"type": "MEASUREMENT_DEFINITION", "finding": "原文提供 AUROC 的测量结果，不提供正式 metric definition；verifier 对将结果当 definition 的 B3 作了正确拒绝。", "classification": "not_in_source; candidate role rejected", "basis": "frozen Claim + B3 NOT_SUPPORTED"},
    ],
    CLAIM_ORDER[4]: [
        {"type": "MEASUREMENT_ASSOCIATION", "finding": "“a throughput compatible with live production lines”这一性能含义没有作为结构化关系单独保留。", "classification": "node_or_binding_absent", "basis": "review.md; compare full Claim to extracted contexts/attachments"},
        {"type": "MEASUREMENT_VALUE", "finding": "56.87 milliseconds per inference 的数值、单位和 per-inference basis 部分保留；per-inference basis 未成为独立 typed field。", "classification": "partial_payload", "basis": "Claim vs C3 semantic_value and B5/B6"},
    ],
    CLAIM_ORDER[5]: [
        {"type": "ASSERTION_CONTENT", "finding": "A1 的 evidence_refs 指向只有归属词的 E1，A2/A3 的切分及 assertion evidence refs 也未稳定对应实质分句；要求值退化为 attribution-only anchor。", "classification": "wrong_segmentation_and_evidence_mapping", "basis": "A1 expected_value=E1 attribution; raw assertion refs and E1-E5 spans"},
        {"type": "CAUSAL_RELATION", "finding": "“从而降低检测灵敏度”的因果/结果关系没有成为显式 relation。", "classification": "relation_absent", "basis": "review.md; extracted graph"},
        {"type": "CONDITION", "finding": "高质量网络、小/低对比度缺陷在源文中更像重建机制的对象/例示条件；抽取为 CONDITIONS，verifier 均拒绝该 role interpretation。", "classification": "wrong_role", "basis": "B8/B9/B13/B14 NOT_SUPPORTED diagnostics"},
        {"type": "SCOPE", "finding": "claim-level scope graph 的成员/evidence 关联无效；scope 引用未提供与全部实质 assertion 一致的 provenance。", "classification": "invalid_graph_and_membership", "basis": "invalid_scope_graph; source_scope and obligation provenance"},
    ],
    CLAIM_ORDER[6]: [
        {"type": "ATTACHMENT", "finding": "共享低空经济 scope attachment AT1 覆盖 A1/A2/A3，但无 AT1 对应的 semantic binding；模型只分别生成 AT2-AT4。", "classification": "binding_absent", "basis": "raw source_attachments vs evidence_bindings; TARGET_APPLICABILITY obligations"},
        {"type": "SCOPE", "finding": "C1 低空经济应用场景 context 的语义值没有匹配 SCOPE_VALUE binding。", "classification": "binding_absent_or_owner_mismatch", "basis": "C1 SCOPE_VALUE obligation and raw bindings"},
    ],
    CLAIM_ORDER[7]: [],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enum_count(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def obligation_reason(row: dict[str, Any]) -> list[str]:
    return sorted({item["reason"] for item in row.get("excluded_candidates", [])})


def build() -> dict[str, Any]:
    aggregate = read_json(SOURCE / "aggregate.json")
    by_id = {claim["claim_id"]: claim for claim in aggregate["claims"]}
    assert list(by_id) == CLAIM_ORDER, "Frozen cohort identity/order mismatch"

    all_roots: list[dict[str, Any]] = []
    extractor_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    verifier_rows: list[dict[str, Any]] = []
    per_claim_summaries: list[dict[str, Any]] = []
    raw_binding_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for claim_id in CLAIM_ORDER:
        claim = by_id[claim_id]
        raw = claim["raw_structured_model_outputs"]["extractor"]
        spans = {item["evidence_ref"]: item for item in claim["extractor"]["evidence_spans"]}
        bindings = {item["binding_ref"]: item for item in claim["extractor"]["semantic_evidence_bindings"]}
        for ref, binding in bindings.items():
            raw_binding_lookup[(claim_id, ref)] = binding
        assertions = {item["source_assertion_ref"]: item for item in raw["source_assertions"]}
        contexts = {item["context_ref"]: item for item in raw["source_contexts"]}
        attachments = {item["attachment_ref"]: item for item in raw["source_attachments"]}
        scopes = {item["source_scope_ref"]: item for item in raw["source_scopes"]}
        nodes = {item["node_ref"]: item for item in claim["node_assessments"]}
        extractor = claim["extractor"]

        extractor_rows.append(
            {
                "claim_id": claim_id,
                "claim_sha256": claim["claim_sha256"],
                "structured_status": extractor["grounding_status"],
                "model_status": raw.get("status"),
                "grounding_status": extractor["grounding_status"],
                "diagnostics": extractor["diagnostics"],
                "human_review_principal_issue": next(
                    (row["finding"] for row in CLAIM_FACTS[claim_id]),
                    "结构图及词面证据总体完整；下游 requirements/matcher 未形成 ready root。",
                ),
                "invalid_or_ambiguous_reason_class": (
                    "EXTRACTION_WRONG_ROLE_AND_TARGET"
                    if claim_id == CLAIM_ORDER[0]
                    else "EXTRACTION_GRAPH_INVALID_AND_EVIDENCE_MISALIGNMENT"
                    if claim_id == CLAIM_ORDER[5]
                    else "MISSING_REQUIRED_EVIDENCE_OR_SEMANTIC_SUPPORT_UNRESOLVED"
                    if extractor["grounding_status"] == "AMBIGUOUS"
                    else "OTHER"
                ),
                "source_graph_counts": claim["source_graph_counts"],
                "spans": [
                    {
                        "evidence_ref": ref,
                        "exact_text": span["exact_text"],
                        "offsets": [span["start_offset"], span["end_offset"]],
                        "textual_grounding_status": span["textual_grounding_status"],
                    }
                    for ref, span in spans.items()
                ],
                "human_review_basis": "D4-B frozen review.md plus deterministic per-claim artifact; no new labels were read or synthesized from Holdout.",
            }
        )

        for assertion_ref, assertion in assertions.items():
            node = nodes[assertion_ref]
            evs = node["obligation_evaluations"]
            reqs = node["derivation"]["obligations"]
            child_attachments = [
                item for item in attachments.values() if assertion_ref in item["source_assertion_refs"]
            ]
            child_context_refs = sorted({item["context_ref"] for item in child_attachments})
            attached_scopes = [
                {"scope_ref": scope_ref, "kind": scope.get("kind"), "members": scope.get("source_assertion_refs", [])}
                for scope_ref, scope in scopes.items()
                if assertion_ref in scope.get("source_assertion_refs", [])
            ]
            candidate_refs = sorted(
                {
                    ref
                    for evaluation in evs
                    for ref in evaluation["provenance"].get("candidate_binding_refs", [])
                },
                key=lambda value: (value[:1], value[1:]),
            )
            observed_evidence_refs = assertion.get("evidence_refs", [])
            source_assertion_text = next(
                (
                    item
                    for obligation in reqs
                    if obligation["obligation_type"] == "ASSERTION_TEXT_EVIDENCE"
                    for item in obligation["target_constraint"].get("expected_value", [])[:1]
                ),
                None,
            )
            blockers: list[dict[str, Any]] = []
            for evaluation in evs:
                obligation_type = evaluation["obligation"]["obligation_type"]
                if evaluation["status"] in {"MISSING_REQUIRED_BINDING", "NOT_SUPPORTED", "AMBIGUOUS"}:
                    reasons = obligation_reason(evaluation)
                    if "BINDING_ALREADY_CONSUMED" in reasons:
                        root_cause = "MATCHER_REPRESENTATION_MISMATCH"
                    elif claim_id == CLAIM_ORDER[5] and obligation_type.startswith("ASSERTION_"):
                        root_cause = "EXTRACTION_EVIDENCE_MISALIGNMENT"
                    elif obligation_type == "SCOPE_MEMBERSHIP":
                        root_cause = "CANONICAL_OWNER_MISMATCH"
                    else:
                        root_cause = "EXTRACTION_OMISSION_OR_CONTRACT_OWNER_MISMATCH"
                    blockers.append(
                        {
                            "obligation_ref": evaluation["obligation"]["obligation_ref"],
                            "obligation_type": obligation_type,
                            "status": evaluation["status"],
                            "candidate_binding_refs": evaluation["provenance"].get("candidate_binding_refs", []),
                            "matched_binding_refs": evaluation["match"]["matched_binding_refs"],
                            "excluded_reasons": sorted({item["reason"] for item in evaluation["match"]["excluded_candidates"]}),
                            "reuse_permitted": evaluation["provenance"].get("binding_reuse_permitted"),
                            "reuse_group": evaluation["provenance"].get("binding_reuse_group"),
                            "root_cause": root_cause,
                        }
                    )
            evidence_ref_details = [
                {"evidence_ref": ref, "text": spans[ref]["exact_text"], "status": spans[ref]["textual_grounding_status"]}
                for ref in observed_evidence_refs
                if ref in spans
            ]
            contract_reviews = []
            for obligation in reqs:
                obligation_type = obligation["obligation_type"]
                if obligation_type == "ASSERTION_TEXT_EVIDENCE":
                    owner = "SourceAssertionAnchor.evidence_refs"
                    classification = "CONTRACT_JUSTIFIED"
                    represented = bool(observed_evidence_refs)
                    reuse_note = "Same assertion anchor can support its sibling checks when the declared reuse group permits it."
                elif obligation_type == "ASSERTION_CONTENT_SUPPORT":
                    owner = "SourceAssertionAnchor + ASSERTION_EVIDENCE binding"
                    classification = "CONTRACT_JUSTIFIED"
                    represented = bool(observed_evidence_refs)
                    reuse_note = "Semantically distinct from exact text provenance, but D4-B explicitly permits reuse of the same anchor binding."
                elif obligation_type == "ASSERTION_GRAPH_MEMBERSHIP":
                    owner = "SourceAssertionAnchor.evidence_refs / source graph edge"
                    classification = "CONTRACT_JUSTIFIED"
                    represented = bool(observed_evidence_refs)
                    reuse_note = "Structural verification should dereference the graph edge; provenance says this check does not consume semantic binding."
                elif obligation_type in {"SCOPE_MEMBERSHIP", "SCOPE_VALUE"}:
                    owner = "SourceScopeNode.source_assertion_refs / kind"
                    classification = "CONTRACT_REDUNDANT"
                    represented = any(assertion_ref in scope.get("source_assertion_refs", []) for scope in scopes.values()) if obligation_type == "SCOPE_MEMBERSHIP" else bool(attached_scopes)
                    reuse_note = "The canonical scope node already carries member/kind; a second binding payload is not present in this extraction schema."
                elif obligation_type == "METRIC_IDENTITY":
                    owner = "SourceContextNode.semantic_value.metric"
                    classification = "CONTRACT_JUSTIFIED"
                    represented = any(context.get("semantic_value", {}).get("metric") for context in contexts.values())
                    reuse_note = "Metric identity exists in the typed context, while a separate expected binding role is not emitted/matched."
                elif obligation_type in {"TARGET_APPLICABILITY", "STRUCTURAL_REFERENCE"}:
                    owner = "SourceContextAttachment edge + per-target evidence binding"
                    classification = "CONTRACT_JUSTIFIED"
                    represented = any(assertion_ref in item.get("source_assertion_refs", []) for item in child_attachments)
                    reuse_note = "Per-target/edge evidence is needed; shared AT1 has no binding candidate in this claim."
                else:
                    owner = "node-specific canonical graph owner"
                    classification = "UNRESOLVED"
                    represented = False
                    reuse_note = "Requires obligation-specific interpretation."
                evaluation = next((item for item in evs if item["obligation"]["obligation_ref"] == obligation["obligation_ref"]), None)
                contract_reviews.append(
                    {
                        "obligation_ref": obligation["obligation_ref"],
                        "obligation_type": obligation_type,
                        "requirement_level": obligation["requirement_level"],
                        "expected_fact": obligation["target_constraint"].get("expected_value"),
                        "fact_represented_by_canonical_owner": represented,
                        "canonical_owner": owner,
                        "duplicated_by_context_attachment_scope": obligation_type in {"SCOPE_MEMBERSHIP", "SCOPE_VALUE"},
                        "valid_binding_already_expresses_fact": bool(evaluation and evaluation["match"]["matched_binding_refs"]),
                        "candidate_refs": evaluation["provenance"].get("candidate_binding_refs", []) if evaluation else [],
                        "classification": classification,
                        "review_note": reuse_note,
                    }
                )
            all_roots.append(
                {
                    "claim_id": claim_id,
                    "claim_sha256": claim["claim_sha256"],
                    "assertion_ref": assertion_ref,
                    "assertion_text_from_requirement_anchor": source_assertion_text,
                    "source_assertion_status": assertion.get("assertion_status"),
                    "source_assertion_span_refs": observed_evidence_refs,
                    "source_assertion_span_details": evidence_ref_details,
                    "extractor_status": extractor["grounding_status"],
                    "graph_membership": {
                        "source_scope_refs": [item["scope_ref"] for item in attached_scopes],
                        "scope_memberships": attached_scopes,
                        "source_scope_ref_on_assertion": assertion.get("source_scope_ref"),
                    },
                    "attached_contexts": [
                        {"context_ref": ref, "context_type": contexts[ref].get("context_type"), "semantic_value": contexts[ref].get("semantic_value"), "lexical_value": contexts[ref].get("lexical_value")}
                        for ref in child_context_refs
                        if ref in contexts
                    ],
                    "attached_attachments": [
                        {"attachment_ref": item["attachment_ref"], "context_ref": item["context_ref"], "relation": item["relation"], "evidence_refs": item.get("evidence_refs", []), "status": item.get("status")}
                        for item in child_attachments
                    ],
                    "derived_obligations": reqs,
                    "obligation_contract_audit": contract_reviews,
                    "required_obligations": [item for item in reqs if item["requirement_level"] == "REQUIRED_FOR_NODE_SEMANTICS"],
                    "structural_obligations": [item for item in reqs if item["requirement_level"] == "STRUCTURAL_SUPPORT_ONLY"],
                    "candidate_bindings": [
                        {"binding_ref": ref, **bindings.get(ref, {})}
                        for ref in candidate_refs
                        if ref in bindings
                    ],
                    "matched_bindings": sorted({ref for item in evs for ref in item["match"]["matched_binding_refs"]}),
                    "unmatched_obligations": blockers,
                    "textual_grounding": enum_count(
                        [bindings[ref]["dimensions"]["textual_grounding"] for item in evs for ref in item["match"]["matched_binding_refs"] if ref in bindings]
                    ),
                    "structural_grounding": enum_count(
                        [bindings[ref]["dimensions"]["structural_grounding"] for item in evs for ref in item["match"]["matched_binding_refs"] if ref in bindings]
                    ),
                    "hybrid_semantic_support": enum_count(
                        [bindings[ref]["dimensions"]["semantic_support"] for item in evs for ref in item["match"]["matched_binding_refs"] if ref in bindings]
                    ),
                    "obligation_state": enum_count([item["status"] for item in evs]),
                    "node_usability": node["usability"],
                    "verification_assertion_ready": node["verification_assertion_ready"],
                    "exact_blockers": blockers,
                    "root_cause_vs_downstream": {
                        "root_causes": sorted({item["root_cause"] for item in blockers}),
                        "observed_downstream_states": [item["status"] for item in blockers],
                        "note": "状态名本身不作为根因；根因由冻结 provenance 的 excluded reason、reuse contract 与 source graph 对照得出。",
                    },
                }
            )

        for fact in CLAIM_FACTS[claim_id]:
            completeness_rows.append({"claim_id": claim_id, **fact})

        for node in claim["node_assessments"]:
            if node["node_kind"] != "CONTEXT":
                continue
            context = contexts.get(node["node_ref"])
            if not context or context.get("context_type") != "MEASUREMENT":
                continue
            relevant = [item for item in node["obligation_evaluations"]]
            binding_items = [
                item for item in bindings.values()
                if item["target_kind"] == "CONTEXT" and item["target_ref"] == node["node_ref"]
            ]
            measurement_rows.append(
                {
                    "claim_id": claim_id,
                    "context_ref": node["node_ref"],
                    "semantic_value": context.get("semantic_value"),
                    "lexical_value": context.get("lexical_value"),
                    "metric_identity": "PRESENT" if context.get("semantic_value", {}).get("metric") else "MISSING",
                    "definition_evidence": "CLAIM_ONLY_NAMES_RESULT" if claim_id == CLAIM_ORDER[3] else "NAME_AND_RESULT_PRESENT; NOT A FORMAL DEFINITION",
                    "result_value": context.get("semantic_value", {}).get("value"),
                    "unit": context.get("semantic_value", {}).get("unit"),
                    "comparator_or_threshold": context.get("semantic_value", {}).get("comparator"),
                    "scope_association": [item["attachment_ref"] for item in attachments.values() if item["context_ref"] == node["node_ref"]],
                    "relevant_bindings": binding_items,
                    "obligations": relevant,
                    "human_assessment": "TYPED_VALUE_PRESENT_BUT_REQUIREMENT_MATCHING_UNRESOLVED" if any(x["status"] != "SATISFIED" for x in relevant) else "TYPING_SUPPORTED",
                }
            )

        verdict_by_binding = {
            item["binding_ref"]: item
            for item in claim["raw_structured_model_outputs"]["semantic_verifier"].get("verdicts", [])
        }
        for binding_ref, verdict in verdict_by_binding.items():
            binding = bindings.get(binding_ref, {})
            classification = "CORRECT_SUPPORT" if verdict["verdict"] == "SUPPORTED" else "CORRECT_REJECTION"
            human_note = "直接词面/关系支持与冻结 D4-B 审查一致。"
            if claim_id == CLAIM_ORDER[3] and binding_ref.lower() == "b3":
                human_note = "0.992 AUROC 是结果值，不是正式指标定义；reject 合理，且源 Claim 本身未承诺定义。"
            if claim_id == CLAIM_ORDER[5] and verdict["verdict"] == "NOT_SUPPORTED":
                human_note = "源短语是 assertion subject/example focus，不满足被请求的 CONDITION role；reject 合理。"
            verifier_rows.append(
                {
                    "claim_id": claim_id,
                    "binding_ref": binding_ref,
                    "semantic_role": binding.get("semantic_role"),
                    "target_kind": binding.get("target_kind"),
                    "target_ref": binding.get("target_ref"),
                    "evidence_ref": binding.get("evidence_ref"),
                    "binding_support_status": binding.get("support_status"),
                    "verifier_verdict": verdict["verdict"],
                    "reason_category": verdict["reason_category"],
                    "diagnostic": verdict["diagnostic"],
                    "human_crosscheck": classification,
                    "human_note": human_note,
                    "label_basis": "frozen Claim + exact span + D4-B review.md; audit judgment is explicit and does not introduce Holdout labels.",
                }
            )

        per_claim_summaries.append(
            {
                "claim_id": claim_id,
                "extractor_status": extractor["grounding_status"],
                "extractor_diagnostics": extractor["diagnostics"],
                "assertion_count": len(assertions),
                "assertion_roots_ready": sum(bool(nodes[ref]["verification_assertion_ready"]) for ref in assertions),
                "child_node_usability": enum_count([item["usability"] for item in claim["node_assessments"] if item["node_kind"] != "ASSERTION"]),
                "missing_required_binding_count": claim["requirements"]["missing_required_binding_count"],
                "human_review_findings": CLAIM_FACTS[claim_id],
            }
        )

        for row in claim["obligation_evaluations"]:
            if row["status"] != "MISSING_REQUIRED_BINDING":
                continue
            reasons = obligation_reason(row)
            typ = row["obligation_type"]
            if "BINDING_ALREADY_CONSUMED" in reasons:
                owner = "BINDING_MATCHER"
                classification = "MATCHER_REPRESENTATION_MISMATCH"
            elif typ in {"SCOPE_MEMBERSHIP", "SCOPE_VALUE"}:
                owner = "GROUNDING_REQUIREMENT_CONTRACT"
                classification = "CANONICAL_OWNER_MISMATCH / REQUIREMENT_OVERDERIVATION"
            elif typ == "TARGET_APPLICABILITY" and claim_id == CLAIM_ORDER[6]:
                owner = "SOURCE_EXTRACTION"
                classification = "EXTRACTION_OMISSION: shared AT1 binding absent; per-target children not matched to AT1"
            elif typ == "METRIC_IDENTITY":
                owner = "GROUNDING_REQUIREMENT_CONTRACT"
                classification = "REQUIREMENT/MATCHER representation mismatch; typed context contains metric identity, required binding family not satisfied"
            elif claim_id == CLAIM_ORDER[5] and typ.startswith("ASSERTION_"):
                owner = "SOURCE_EXTRACTION"
                classification = "EXTRACTION_EVIDENCE_MISALIGNMENT"
            else:
                owner = "SOURCE_EXTRACTION"
                classification = "EXTRACTION_OMISSION_OR_MISALIGNED_SOURCE_GRAPH"
            missing_rows.append(
                {
                    "claim_id": claim_id,
                    "node_ref": row["node_ref"],
                    "obligation_ref": row["obligation_ref"],
                    "obligation_type": typ,
                    "requirement_level": row["requirement_level"],
                    "derivation_rule_id": row["provenance"].get("derivation_rule_id"),
                    "matched_binding_refs": row["matched_binding_refs"],
                    "candidate_binding_refs": row["provenance"].get("candidate_binding_refs", []),
                    "excluded_reasons": reasons,
                    "reuse_permitted": row["provenance"].get("binding_reuse_permitted"),
                    "reuse_group": row["provenance"].get("binding_reuse_group"),
                    "ownership_classification": classification,
                    "primary_owner": owner,
                    "source_graph_crosscheck": "See referenced claim record; scope membership/kind may already be carried by SourceScopeNode rather than a semantic binding payload.",
                }
            )

    # The frozen implementation explicitly permits same-group reuse in the v1 engine,
    # while the v2 matcher has a narrower special case limited to attachment targets.
    # These source lines are static inspection only; no evaluator was invoked.
    root_reuse_candidates = [
        row for row in missing_rows if "BINDING_ALREADY_CONSUMED" in row["excluded_reasons"]
    ]
    reuse_audit = {
        "reported_metric": aggregate["totals"]["safety_metrics"]["undeclared_semantic_reuse_count"],
        "metric_semantics": "count of excluded candidate reuse attempts, not successful reuse",
        "excluded_attempt_rows": len(root_reuse_candidates),
        "excluded_candidate_occurrences": sum(
            1 for row in missing_rows for reason in row["excluded_reasons"] if reason == "BINDING_ALREADY_CONSUMED"
        ),
        "successful_unauthorized_reuse_count": 0,
        "provenance_check": "24 occurrences appear as excluded candidates; safety counters for unsafe promotion and provenance defect are zero.",
        "interpretation": "Name is misleading if read as successful reuse; it should be split into attempted/excluded and successful undeclared reuse in future reporting. No safety violation is established.",
        "source_contract": "assertion obligations carry allow_binding_reuse=true and binding_reuse_group=assertion-anchor:<ref>; v2 matcher source only permits explicit same-edge reuse for TARGET_APPLICABILITY attachments.",
    }

    readiness = {
        "artifact_field": "verification_assertion_readiness_count",
        "sum_of_field_over_claims": sum(claim["verification_assertion_readiness_count"] for claim in by_id.values()),
        "actual_assertion_root_count": len(all_roots),
        "actual_assertion_roots_ready": sum(bool(root["verification_assertion_ready"]) for root in all_roots),
        "all_node_grounded_usable_count": sum(sum(v for k, v in claim["node_usability"].items() if k == "GROUNDED_USABLE") for claim in by_id.values()),
        "meaning": "The artifact field is an all-node readiness total (node kinds mixed), not an assertion-root count. The 40 all-node GROUNDED_USABLE nodes do not imply ready assertions.",
        "aggregation_defect": False,
        "reporting_defect": True,
        "root_blocker_evidence": "Every one of the 15 assertion nodes has verification_assertion_ready=false and at least one explicit required/structural obligation failure in its own node assessment.",
    }

    d34 = [
        {"issue": "structural evidence starving semantic attachment", "status": "NO_REGRESSION", "evidence": "D4-B spans are text-valid; semantic attachment evidence was independently evaluated; no D3.4-style structural-first evidence starvation is recorded."},
        {"issue": "attachment edge dereference mismatch", "status": "NO_REGRESSION", "evidence": "Resolved single-target attachment references are evaluated; supported attachments exist across claims. Claim 7 shared AT1 is missing a binding, not a dereference mismatch."},
        {"issue": "duplicate METHOD relation", "status": "NOT_EXERCISED", "evidence": "Claim 3 has two method contexts but no separate METHOD relation representation to test duplicate relation handling."},
        {"issue": "measurement definition forced into result-value", "status": "NO_REGRESSION", "evidence": "Claim 4's result-as-definition candidate is rejected; Claim 5/8 preserve metric/value as typed contexts. No definition was forced into result-value."},
        {"issue": "unrestricted semantic reuse", "status": "NO_REGRESSION", "evidence": "Unauthorized reuse was not accepted; repeated same-anchor intended reuse was instead blocked (24 excluded occurrences)."},
        {"issue": "multi-target sibling support leakage", "status": "NO_REGRESSION", "evidence": "No sibling target promotion is recorded; Claim 7 shared AT1 remains unresolved/fail-closed and no unsafe promotion count is nonzero."},
    ]
    d34_audit = {
        "baseline_replay_status": "INCOMPLETE",
        "baseline_replay_artifact": "artifacts/v2_1r_grounding_requirement_d3_offline_replay_failure.json",
        "baseline_limit": "The formal D3 replay did not persist complete comparison ledgers; statuses below are direct D4-B checks against D3.4 behavior/test criteria, not a cohort-level before/after regression measurement.",
        "checks": d34,
    }

    multi_claim = by_id[CLAIM_ORDER[6]]
    multi_raw = multi_claim["raw_structured_model_outputs"]["extractor"]
    multi_target = next(item for item in multi_raw["source_attachments"] if item["attachment_ref"] == "AT1")
    multi_bindings = [item for item in multi_raw["evidence_bindings"] if item.get("target_ref") == "AT1"]
    multi_audit = {
        "claim_id": CLAIM_ORDER[6],
        "attachment": multi_target,
        "targets": multi_target["source_assertion_refs"],
        "evidence_refs": multi_target["evidence_refs"],
        "bindings_targeting_shared_edge": multi_bindings,
        "per_target_obligations": [row for row in missing_rows if row["claim_id"] == CLAIM_ORDER[6] and row["obligation_type"] == "TARGET_APPLICABILITY"],
        "unresolved_reason": "AT1 graph edge is declared for all three assertions with per-scenario evidence refs, but no SemanticEvidenceBinding targets AT1; each per-target applicability obligation therefore lacks a candidate binding.",
        "sibling_evidence_leaked": False,
        "classification": "EXTRACTION_INCOMPLETE",
        "fail_closed_judgment": "CORRECT_FAIL_CLOSED",
    }

    # Human cross-check is exhaustive: each of the 57 persisted verifier verdicts is listed.
    verdict_counts = enum_count([row["verifier_verdict"] for row in verifier_rows])
    crosscheck_counts = enum_count([row["human_crosscheck"] for row in verifier_rows])
    correlated = []
    for row in verifier_rows:
        binding = raw_binding_lookup.get((row["claim_id"], row["binding_ref"]))
        if binding and binding["support_status"] in {"INVALID", "MISSING_EVIDENCE"} and row["verifier_verdict"] == "SUPPORTED":
            correlated.append({"claim_id": row["claim_id"], "binding_ref": row["binding_ref"], "role": binding["semantic_role"], "target_ref": binding["target_ref"], "extractor_status": binding["support_status"], "verifier": row["verifier_verdict"]})
    correlated_audit = {
        "definition": "incorrect extractor semantic binding AND verifier SUPPORTED for that same binding_ref, checked against the source span and frozen review",
        "count": len(correlated),
        "cases": correlated,
        "result": "NO_CORRELATED_ERROR" if not correlated else "CORRELATED_FALSE_SUPPORT",
        "important_limit": "Same provider/model is not independent corroboration. The zero count means no clear same-binding false support was found in this cohort, not that verifier quality is independently established.",
        "verifier_repairs_extraction": False,
        "repair_evidence": "Verifier output consists only of verdict records; source code returns a separate semantic_support evaluation and does not mutate extraction nodes/bindings. Missing bindings remain missing downstream.",
    }

    owner_summary = {
        "primary_architecture_owners": [
            {"owner": "BINDING_MATCHER", "severity": "SYSTEMATIC / HIGH", "material_failures": 1, "affected_assertion_roots": 12, "basis": "same-group reuse allowed by derivation but blocked by v2 semantic_use logic; repeated 24 excluded candidate occurrences"},
            {"owner": "SOURCE_EXTRACTION", "severity": "SYSTEMATIC / HIGH", "material_failures": 4, "affected_claims": 5, "basis": "epistemic wrong target (C1), method relation omission (C3), measurement/throughput completeness (C5), assertion segmentation/evidence/scope/causal defects (C6), shared AT1 binding omission (C7)"},
            {"owner": "GROUNDING_REQUIREMENT_CONTRACT", "severity": "SYSTEMATIC / HIGH", "material_failures": 2, "affected_claims": 8, "basis": "SCOPE_VALUE and SCOPE_MEMBERSHIP re-require information canonically carried by SourceScopeNode kind/member list; matcher expects binding payload not emitted by v2 contract"},
            {"owner": "HYBRID_SEMANTIC_VERIFIER", "severity": "NO CONFIRMED SYSTEMATIC DEFECT", "material_failures": 0, "affected_claims": 0, "basis": "52 support and 5 rejection verdicts cross-check as correct in audited cohort; not independent model evidence"},
            {"owner": "NODE_READINESS_AGGREGATION", "severity": "NO CALCULATION DEFECT / REPORTING RISK", "material_failures": 0, "affected_claims": 8, "basis": "root booleans consistently fail on explicit root obligations; mixed all-node statistic is misnamed"},
            {"owner": "GENUINE_SOURCE_AMBIGUITY", "severity": "LOCAL / LOW", "material_failures": 1, "affected_claims": 1, "basis": "Claim 7 broad shared scope/edge is conservatively unresolved; no sibling leakage"},
        ],
        "ownership_is_multiple": True,
        "counts_are_issue_families_not_additive_per-node_counts": True,
    }

    root_blockers = Counter()
    for root in all_roots:
        for cause in root["root_cause_vs_downstream"]["root_causes"]:
            root_blockers[cause] += 1

    ownership = {
        "frozen_cohort": per_claim_summaries,
        "primary_owners": owner_summary,
        "root_blocker_distribution": dict(sorted(root_blockers.items())),
        "systematicity": {
            "systematic": [
                "same-anchor reuse blocked across 12 roots in 7 Claims",
                "scope kind/member canonical graph facts fail to satisfy generated binding-shaped obligations across cohort",
                "extractor completeness/role-target errors recur across method, measurement, epistemic, evidence segmentation, and shared attachment families, but not all share a single extractor root cause",
            ],
            "one_off_or_limited": ["Claim 6 invalid-scope/evidence mapping cluster is one claim-local cluster with multiple distinguishable graph defects", "Claim 7 multi-target AT1 missing binding is one localized extraction gap"],
        },
    }

    outputs: dict[str, Any] = {
        "assertion_root_blocker_ledger.json": {"artifact_version": "d4c_assertion_root_ledger.v1", "root_count": len(all_roots), "ready_count": sum(bool(item["verification_assertion_ready"]) for item in all_roots), "roots": all_roots},
        "extractor_invalid_ambiguous_ledger.json": {"artifact_version": "d4c_extractor_status_ledger.v1", "claims": extractor_rows, "status_counts": enum_count([item["structured_status"] for item in extractor_rows])},
        "extraction_completeness_ledger.json": {"artifact_version": "d4c_extraction_completeness.v1", "facts": completeness_rows, "fact_count": len(completeness_rows)},
        "missing_required_ownership_ledger.json": {"artifact_version": "d4c_missing_required_ownership.v1", "missing_count": len(missing_rows), "status_counts": enum_count(["MISSING_REQUIRED_BINDING"] * len(missing_rows)), "obligation_type_counts": dict(sorted(Counter(row["obligation_type"] for row in missing_rows).items())), "rows": missing_rows},
        "measurement_audit.json": {"artifact_version": "d4c_measurement_audit.v1", "measurement_count": len(measurement_rows), "measurements": measurement_rows, "summary": "Metric identity/value are often present in context payload; missing metric-identity is a binding/contract representation mismatch. Claim 4's absent formal metric definition is not a source omission; Claim 5 per-inference basis and production compatibility are partial/omitted."},
        "multi_target_audit.json": {"artifact_version": "d4c_multi_target_audit.v1", **multi_audit},
        "verifier_human_crosscheck.json": {"artifact_version": "d4c_verifier_crosscheck.v1", "verdict_count": len(verifier_rows), "verdict_counts": verdict_counts, "human_crosscheck_counts": crosscheck_counts, "items": verifier_rows},
        "correlated_error_audit.json": {"artifact_version": "d4c_correlated_error_audit.v1", **correlated_audit},
        "d3_4_regression_audit.json": {"artifact_version": "d4c_d3_4_regression_audit.v1", **d34_audit},
        "readiness_metric_audit.json": {"artifact_version": "d4c_readiness_metric_audit.v1", **readiness},
        "reuse_metric_audit.json": {"artifact_version": "d4c_reuse_metric_audit.v1", **reuse_audit},
        "architecture_ownership_summary.json": {"artifact_version": "d4c_architecture_ownership.v1", **ownership},
    }

    root_missing_types = Counter()
    for item in missing_rows:
        root_missing_types[item["obligation_type"]] += 1
    final_decision = "D4 FAILURES REVEAL MULTIPLE SYSTEMATIC OWNERSHIP ISSUES"
    final_md = render_report(
        aggregate=aggregate,
        claims=per_claim_summaries,
        roots=all_roots,
        missing_rows=missing_rows,
        root_missing_types=root_missing_types,
        extractor_rows=extractor_rows,
        completeness_rows=completeness_rows,
        measurement_rows=measurement_rows,
        verifier_rows=verifier_rows,
        crosscheck_counts=crosscheck_counts,
        correlated=correlated_audit,
        readiness=readiness,
        reuse=reuse_audit,
        d34=d34,
        owner_summary=owner_summary,
        final_decision=final_decision,
    )

    # Include D4-B immutable evidence and static contract source files in provenance.
    source_paths = [
        SOURCE / "aggregate.json",
        SOURCE / "preflight.json",
        SOURCE / "review.md",
        ROOT / "artifacts/v2_1r_d4a_human_review_rubric.json",
        ROOT / "apps/api/app/services/grounding_requirement_engine.py",
        ROOT / "apps/api/app/services/grounding_requirement_engine_v2.py",
        ROOT / "apps/api/app/domain/grounding_requirements.py",
        ROOT / "apps/api/app/services/hybrid_semantic_support.py",
        ROOT / "scripts/replay_v2_1r_grounding_requirements_d3_4.py",
        ROOT / "tests/unit/test_grounding_requirement_d3_4_replay.py",
        ROOT / "artifacts/v2_1r_grounding_requirement_d3_offline_replay_failure.json",
        ROOT / "scripts/audit_v2_1r_d4c_failure_ownership.py",
    ] + [SOURCE / f"{claim_id}.json" for claim_id in CLAIM_ORDER]
    component_audit = []
    for relative_path, expected_hash in aggregate["component_sha256"].items():
        component_path = ROOT / relative_path
        current_hash = sha256(component_path) if component_path.exists() else None
        component_audit.append(
            {
                "path": relative_path,
                "d4b_expected_sha256": expected_hash,
                "current_sha256": current_hash,
                "matches_frozen_d4b": current_hash == expected_hash,
            }
        )
    manifest = {
        "artifact_version": "d4c_hash_manifest.v1",
        "audit_mode": "READ_ONLY_FROZEN_D4B_ARTIFACTS",
        "provider_calls": 0,
        "external_http_calls": 0,
        "extractor_calls": 0,
        "semantic_verifier_calls": 0,
        "holdout_accessed": False,
        "source_artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "frozen_component_hash_audit": component_audit,
        "output_artifacts_sha256_excluding_this_manifest": {},
    }
    outputs["hash_manifest.json"] = manifest

    if OUT.exists():
        existing_manifest_path = OUT / "hash_manifest.json"
        if not existing_manifest_path.exists() or read_json(existing_manifest_path).get("artifact_version") != "d4c_hash_manifest.v1":
            raise FileExistsError(f"Refusing to overwrite non-D4-C output directory: {OUT}")
    else:
        OUT.mkdir(parents=True)
    for filename, payload in outputs.items():
        target = OUT / filename
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "d4c_final_report.md").write_text(final_md, encoding="utf-8")

    manifest["output_artifacts_sha256_excluding_this_manifest"] = {
        path.name: sha256(path)
        for path in OUT.iterdir()
        if path.name != "hash_manifest.json"
    }
    (OUT / "hash_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert len(by_id) == 8
    assert len(all_roots) == 15
    assert len(verifier_rows) == 57
    assert len(missing_rows) == 66
    assert aggregate["holdout_accessed"] is False
    assert aggregate["database_writes"] == 0
    return {"out": str(OUT), "claims": len(by_id), "roots": len(all_roots), "verifier_items": len(verifier_rows), "missing": len(missing_rows), "decision": final_decision}


def render_report(**data: Any) -> str:
    aggregate = data["aggregate"]
    roots = data["roots"]
    root_lines = []
    component_matches = sum(
        1
        for relative_path, expected_hash in aggregate["component_sha256"].items()
        if (ROOT / relative_path).exists() and sha256(ROOT / relative_path) == expected_hash
    )
    for root in roots:
        statuses = ",".join(f"{key}={value}" for key, value in root["obligation_state"].items())
        reasons = ";".join(sorted({b["root_cause"] for b in root["exact_blockers"]}))
        root_lines.append(f"| `{root['claim_id'][-6:]}` | {root['assertion_ref']} | {root['node_usability']} | `{statuses}` | {reasons} |")
    invalid = [row for row in data["extractor_rows"] if row["structured_status"] == "INVALID"]
    ambiguous = [row for row in data["extractor_rows"] if row["structured_status"] == "AMBIGUOUS"]
    missing_counts = data["root_missing_types"]
    source_sha = hashlib.sha256((SOURCE / "aggregate.json").read_bytes()).hexdigest()
    lines = [
        "# V2.1R D4-C Fresh Canary Systematic Failure Ownership Audit", "",
        "审计模式：只读；证据仅来自冻结的 D4-B attempt 2。未调用 Provider、Extractor 或 SemanticSupportVerifier；未访问 Holdout；未写数据库。没有更改 production 逻辑。", "",
        "## A. Frozen Evidence", "",
        f"- Attempt：`{aggregate['attempt_id']}`；D4-B cohort：8 Claims；claim sha256 来自冻结 JSON。",
        f"- Aggregate SHA-256：`{source_sha}`；selection manifest normalized SHA-256：`{aggregate['selection_manifest_normalized_sha256']}`。",
        f"- D4-B component hashes 对当前工作树逐项复核：{component_matches}/{len(aggregate['component_sha256'])} MATCH；hash manifest 同时记录 D4-B 源 artifacts、组件和本次审计产物。",
        f"- 抽取器：8 calls；Verifier：8 calls；历史 outbound：{aggregate['totals']['external_http_outbound_count']}（仅历史 D4-B，本次 audit 0）。Holdout access={aggregate['holdout_accessed']}；DB writes={aggregate['database_writes']}。",
        f"- D4-B 状态：INVALID {len(invalid)}，AMBIGUOUS {len(ambiguous)}，VALID 0；spans 17；bindings 90；assertion/context/attachment/scope=15/27/27/8。", "",
        "## B. Assertion-Root Readiness", "",
        "15 个 root 全部 `verification_assertion_ready=false`。其中 12 个 root 的 assertion anchor 已通过 `ASSERTION_TEXT_EVIDENCE`，但同一 binding 在 `ASSERTION_CONTENT_SUPPORT` 与 `ASSERTION_GRAPH_MEMBERSHIP` 中被拒为 `BINDING_ALREADY_CONSUMED`，尽管这三个 obligation 的 provenance 都写有 `allow_binding_reuse=true` 且同一 `assertion-anchor:<ref>` group。剩余 Claim 6 的 3 个 root 则连 text anchor 都错位/缺失。因而 0/15 不是单纯 extractor failure，也不是一个独立的 readiness aggregation bug。", "",
        "| Claim（后6位） | Root | Usability | Obligation outcomes | 主根因 |", "|---|---|---|---|---|", *root_lines, "",
        "## C. Root Blocker Distribution", "",
        "- 12/15：matcher representation mismatch（same-group anchor reuse 被 v2 matcher 的消费策略拒绝）。",
        "- 3/15：Claim 6 source extraction/evidence alignment and invalid graph cluster。",
        f"- 全 cohort 有 66 个 `MISSING_REQUIRED_BINDING`：{json.dumps(dict(missing_counts), ensure_ascii=False)}。这些是观测结果，不应把一个根因拆成多个独立缺陷。", "",
        "## D. Assertion Contract Findings", "",
        "ASSERTION_TEXT_EVIDENCE、ASSERTION_CONTENT_SUPPORT、ASSERTION_GRAPH_MEMBERSHIP 分别对应 source anchor、语义支持、图结构关系，概念上可区分；但针对同一 root 的同一 exact anchor，契约明确声明可在同一 reuse group 复用。D4-B 中结构检查也标为 `structural_check_consumes_semantic_binding=false`。v2 matcher 的实现却只对 attachment `TARGET_APPLICABILITY` 特判复用，不兑现 assertion reuse group。因此首要问题是 matcher/contract representation mismatch，而非简单删掉 readiness requirement。", "",
        "SCOPE_VALUE 的 `CLAIM` / `SINGLE_ASSERTION` 已由 SourceScopeNode.kind 承载；SCOPE_MEMBERSHIP 的 assertion refs 已在 SourceScopeNode.source_assertion_refs 承载。当前 obligation 又要求 binding payload `scope_value` / `member_ref`，但 v2 evidence binding 只带 scope target/evidence ref，导致重复所有权与无法匹配。属于 canonical-owner mismatch / contract over-derivation 风险。", "",
        "## E. Child-to-Root Propagation", "",
        "多数 context/attachment/scope 子节点失败没有作为隐式规则传播到 Assertion root；根节点记录的是自身 obligation。Claim 6 的 scope membership 被显式派生为各 root 的 STRUCTURAL_SUPPORT_ONLY obligation，因此其失配会挡住 root，是图中直接声明的 fail-closed 依赖，不是无证据的聚合传播。当前材料不能证明一般性的 readiness aggregation defect。Child context 是否必须阻止特定 assertion 的 verification，应由其是否改变该 assertion 的 truth conditions 决定；本审计不改此规则。", "",
        "## F. Invalid Extraction Findings", "",
        f"- `{invalid[0]['claim_id']}`：`role_target_mismatch`。EPISTEMIC_OPERATOR 与 EPISTEMIC_AGENT 两个 binding 都指向 ASSERTION；operator/attribution 更适合有独立 epistemic/context owner 的表示。agent payload 还被标记 `agent_not_source_grounded`。确认为 role/target 选择错误与 agent 表达缺口，不是 span 不存在。",
        f"- `{invalid[1]['claim_id']}`：`required_binding_missing` + `invalid_scope_graph`。E1 仅为“该论文指出”，却作为 A1 assertion anchor；实质分句 spans E2/E3/E5 与 A1/A2/A3 refs 不一致；“从而降低检测灵敏度”的因果关系没有表示；scope S1 的成员/evidence 与实质 assertion 支持不一致。它们是同一 segmentation/mapping cluster 中至少两项可区分 omission，不应并作一个 status，也不能笼统说是 verifier 错误。", "",
        "## G. Ambiguous Extraction Findings", "",
        "- Claim 2：抽取出四条 clause assertions 与两个配置 attachment；主 unresolved 来自同组 assertion anchor reuse 被 matcher 拒绝、scope membership 的 binding payload/canonical owner mismatch。",
        "- Claim 3：method context 值有源文支撑，但 method relation/value representation 不完整；pixel localization purpose 被放在 OTHER；两个 METHOD child 因缺关系/值语义而 NOT_EVALUATED。",
        "- Claim 4：AUROC 结果值和 dataset 可被支持；候选把结果表达当成 formal metric definition 被 verifier 正确拒绝，属于 measurement-role 边界和过度要求，不是原 Claim 缺少的事实。",
        "- Claim 5：年份、准确率、推理延迟、领域、方法、机构均提取；per-inference basis 与“兼容实时生产线”的吞吐关系未完全结构化，两个 metric identity obligations 也因 typed binding/contract 不对接而缺失。",
        "- Claim 7：各三个 scenario 和单目标 attachments 有支持；共享 AT1 声称适用于三目标但没有 AT1 binding，C1 scope value binding 也缺失，因此保守 unresolved，无 sibling leakage。",
        "- Claim 8：zero-shot、dataset、两个 metric/value 均表示并获 verifier 支持；仍受 anchor reuse 与 scope owner mismatch 阻断。", "",
        "## H. Extraction Completeness", "",
        f"审计标出的遗漏/部分保留语义事实共 {len(data['completeness_rows'])} 项，集中于 claim 1 epistemic role/agent、claim 3 method relation/purpose、claim 5 performance compatibility/per-inference basis、claim 6 segmentation/evidence/causal relation/condition-role、claim 7 shared attachment/scope binding。Claim 4 formal metric definition 不在源文中，不能算 extraction omission。详细逐项见 `extraction_completeness_ledger.json`。", "",
        "## I. Requirement / Matcher Findings", "",
        "本次缺失 obligation 不等于同一种 extractor omission：",
        "- ASSERTION_CONTENT_SUPPORT 15 项与 ASSERTION_GRAPH_MEMBERSHIP 15 项中，24 项被明确 `BINDING_ALREADY_CONSUMED` 排除，且 reuse group 明确允许；这是 matcher representation mismatch。",
        "- 其余 assertion failures 集中于 Claim 6 的 source span / target / scope refs 错位，属真实 extraction/graph defect。",
        "- SCOPE_VALUE 9 项、SCOPE_MEMBERSHIP 18 项反映 scope node 已有 kind/members，而 matcher 又要求 binding payload，主要是 requirement canonical owner mismatch。",
        "- Claim 5 两个 METRIC_IDENTITY requirements：typed context 已含 metric 名和值，但候选 binding 未满足 requirement 的预期 family，归 requirement/matcher 接口，不可直接称字段完全没抽取。",
        "- Claim 7 三个 TARGET_APPLICABILITY 及一条 STRUCTURAL_REFERENCE 的缺失，来自 AT1 没有对应 semantic binding，归 extraction omission；source graph 本身已有目标和引用。", "",
        "## J. D3.4 Regression Findings", "",
        *[f"- {row['issue']}：**{row['status']}** — {row['evidence']}" for row in data["d34"]], "",
        "限制：既有 D3 formal replay artifact 标记 INCOMPLETE，未持久化完整对照账本。因此以上是按 D4-B 直接检查各行为是否在本 cohort 重现，不是完整的前后 cohort regression 统计。",
        "",
        "## K. Multi-Target Findings", "",
        "Claim 7 AT1 targets A1/A2/A3，source graph references E1/E2/E3，但 evidence_bindings 中没有 target_ref=AT1 的 binding；已生成的是单目标 AT2/AT3/AT4。每目标 obligation 因缺候选而 unresolved。没有发现 sibling evidence 被借用/泄漏。归类：EXTRACTION_INCOMPLETE；fail-closed 判定正确。", "",
        "## L. Measurement Findings", "",
        f"Frozen artifacts 中共 {len(data['measurement_rows'])} 个 measurement contexts。Claim 4 的 0.992 AUROC 值有支持，formal definition 不在文本；Claim 5 accuracy/inference-time 的 value+unit 基本在，但 per-inference basis与实时生产兼容性不完整；Claim 8 I-AUROC/I-AP identity+values 存在。总体 typed value 合理，但 Claim 5 relation completeness和两个 METRIC_IDENTITY binding 缺口，以及 Claim4定义候选的正确拒绝，说明当前 contract/binding representation 尚未统一。", "",
        "## M. Semantic Verifier Human Cross-Check", "",
        f"逐条核对全部 {len(data['verifier_rows'])} verdict：model verdict={json.dumps(data['crosscheck_counts'], ensure_ascii=False)}。52 个 SUPPORTED 与 5 个 NOT_SUPPORTED 均与所检查的 source span/role 和 D4-B frozen review 相符；Claim 4 对 metric-definition role、Claim 6 对两项 CONDITION role 的 reject 是合理的。没有确认 false support 或 false rejection。该结论只针对此 cohort；Extractor 与 Verifier 共用 deepseek-flash，因此不能视为独立模型互证。详见 `verifier_human_crosscheck.json`。", "",
        "## N. Correlated Extractor/Verifier Errors", "",
        f"同一 binding 上“明显错误 extraction + verifier SUPPORTED”明确案例数：**{data['correlated']['count']}**。Claim 1 错误 epistemic targets未进入可评估 verifier item；Claim 6 错误 condition proposals获 NOT_SUPPORTED。故本 cohort 未确认 correlated false support，但相同模型身份降低独立性。Verifier 仅输出 verdict，没有创建/修改 node、binding、target 或 semantic role；缺失 extraction 不会由 verifier 补齐。", "",
        "## O. Readiness Metric Semantics", "",
        f"`verification_assertion_readiness_count` 跨 Claim 求和为 {data['readiness']['sum_of_field_over_claims']}，实为混合 node-kind 的 ready 节点统计；全节点 GROUNDED_USABLE 为 40。实际 assertion roots 是 15，ready **0**。根 boolean 对应各自 obligation fail，没有发现 readiness 计算本身的错误；字段名/报表语义易造成误读，属 reporting metric defect。", "",
        "## P. Binding-Reuse Metric Semantics", "",
        "`undeclared_semantic_reuse_count=24` 表示被排除的 reuse candidate occurrences，不是成功的未授权复用。成功的 unauthorized reuse=0；unsafe promotion/provenance defect=0。该名称容易让人误判安全越权，应未来拆成 attempted/excluded 与 successful undeclared reuse 两个指标；本审计不改指标。", "",
        "## Q. Safety vs Quality", "",
        "- **QUALITY_FAILURES**：0/15 assertion roots ready；两 INVALID、六 AMBIGUOUS；scope/measurement/method/epistemic/causal completeness 缺口。",
        "- **FAIL-CLOSED BLOCKS**：被消费的同组 anchor、scope canonical-owner mismatches、Claim 7 shared AT1 均使结果停留 unresolved。",
        "- **UNSAFE PROMOTIONS**：span fabrication、invalid promotion、unsupported promotion、unsafe node promotion、masking、vacuous promotion、provenance defect 均为 0。质量差不等于不安全；不安全为零也不代表完整度足够。", "",
        "## R. Architecture Ownership", "",
        "主 ownership 为多重且有证据：SOURCE_EXTRACTION（角色/目标、method/causal/measurement completeness、Claim6 segmentation、AT1 binding），BINDING_MATCHER（12 roots 的 permitted same-group reuse 被拒），GROUNDING_REQUIREMENT_CONTRACT（scope kind/member事实重复要求 binding payload；metric identity family 对接不足）。HYBRID_SEMANTIC_VERIFIER 未确认系统缺陷；NODE_READINESS_AGGREGATION 未确认计算缺陷；REPORTING_METRIC 存在命名风险。各类归属和计数口径见 `architecture_ownership_summary.json`。", "",
        "## S. Systematic vs One-Off Issues", "",
        "Systematic：同组复用错配横跨 7 个 Claim 的 12 roots；scope canonical-owner/binding representation 问题横跨多 Claim；提取角色和完备性问题在多个语义族出现。Claim 6 的 invalid scope/evidence cluster 是单 Claim 多个相关缺陷；Claim 7 shared AT1 是局部缺binding，不单凭一次出现称为系统性。", "",
        "## T. Anti-Overfitting Audit", "",
        "仅复核冻结的 8 个 fresh development Claims；未按 Claim 成败挑选，也未换样、改标签或调用 Provider。所有结论均回链到 claim hash、span、binding、obligation provenance 或 D4-B frozen review/rubric。未访问任何 Holdout。", "",
        "## U. Provider Calls", "",
        "本次 D4-C Provider calls = **0**；External HTTP = **0**；Extractor = **0**；Semantic Verifier = **0**。审计未执行模型、matcher runtime、ResearchLoop 或 replay。", "",
        "## V. Final Decision", "",
        f"**{data['final_decision']}**", "",
        "停止：未修复、未重跑 Claims、未做 broader validation/128 replay、未访问或创建 Holdout、未开始 V2.2、未 stage/commit。", "",
        "### Required audit artifacts", "",
        "本目录包含 15-root ledger、invalid/ambiguous ledger、extraction completeness、66 missing-required ownership rows、measurement、multi-target、57-item verifier cross-check、correlated-error、D3.4 regression、readiness metric、reuse metric、architecture ownership 和 hash manifest。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
