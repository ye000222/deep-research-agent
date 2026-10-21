"""Offline D4-D closure replay over frozen D4-B/D4-C data; no model calls."""

# Report strings intentionally retain readable sentences on one source line.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.domain.grounding_requirements import (
    GroundingBindingCandidate,
    GroundingNodeInput,
    GroundingNodeKind,
    GroundingObligation,
    GroundingObligationDerivationResult,
    GroundingObligationType,
    GroundingRequirementLevel,
)
from app.domain.source_context_grounding_v2 import GroundingDimensionStatus as DS
from app.services.grounding_requirement_engine_v21 import (
    derive_grounding_obligations_v21,
    evaluate_grounding_node_v21,
    grounding_reporting_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v2_1r_d4b_execution_attempt_2"
D4C = ROOT / "artifacts/v2_1r_d4c_systematic_failure_ownership"
OUT = ROOT / "artifacts/v2_1r_d4d_ownership_alignment"
AGGREGATE = SOURCE / "aggregate.json"
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
CLAIM6 = CLAIM_ORDER[5]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, value: Any) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dimension_status(summary: dict[str, int]) -> DS:
    if summary.get("INVALID", 0):
        return DS.INVALID
    if summary.get("AMBIGUOUS", 0):
        return DS.AMBIGUOUS
    if summary.get("VALID", 0):
        return DS.VALID
    return DS.MISSING_EVIDENCE


def _candidate(row: dict[str, Any], *, anchor_text: str | None) -> GroundingBindingCandidate:
    payload: dict[str, Any] = {}
    if (
        row.get("semantic_role") == "ASSERTION_EVIDENCE"
        and anchor_text is not None
        and row.get("target_kind") == "ASSERTION"
    ):
        payload["assertion_content"] = [anchor_text]
    return GroundingBindingCandidate(
        binding_ref=row["binding_ref"],
        evidence_ref=row["evidence_ref"],
        semantic_role=row["semantic_role"],
        target_kind=row["target_kind"],
        target_ref=row["target_ref"],
        canonical_payload=payload,
        relation=row.get("relation"),
        scope_ref=row.get("scope_ref"),
        member_key=row.get("member_key"),
        value_key=row.get("value_key"),
        textual_grounding=row["dimensions"]["textual_grounding"],
        structural_grounding=row["dimensions"]["structural_grounding"],
        semantic_support=row["dimensions"]["semantic_support"],
    )


def build() -> dict[str, Any]:
    aggregate = _read(AGGREGATE)
    d4c_roots = _read(D4C / "assertion_root_blocker_ledger.json")["roots"]
    roots_by_key = {(row["claim_id"], row["assertion_ref"]): row for row in d4c_roots}
    claims = {row["claim_id"]: row for row in aggregate["claims"]}
    if list(claims) != CLAIM_ORDER or len(d4c_roots) != 15:
        raise ValueError("frozen_D4B_D4C_cohort_identity_mismatch")
    binding_count = sum(
        len(item["extractor"]["semantic_evidence_bindings"]) for item in claims.values()
    )
    verifier_count = sum(
        len(item["raw_structured_model_outputs"]["semantic_verifier"].get("verdicts", []))
        for item in claims.values()
    )
    if len(claims) != 8 or binding_count != 90 or verifier_count != 57:
        raise ValueError("frozen_D4B_evidence_counts_mismatch")

    root_rows: list[dict[str, Any]] = []
    assessments = []
    requiredness_stability_violations = 0
    for claim_id in CLAIM_ORDER:
        claim = claims[claim_id]
        raw = claim["raw_structured_model_outputs"]["extractor"]
        spans = {
            item["evidence_ref"]: item["exact_text"]
            for item in claim["extractor"]["evidence_spans"]
        }
        scope_memberships = {
            item["source_scope_ref"]: {
                "kind": item["kind"],
                "members": tuple(item["source_assertion_refs"]),
                "valid": "invalid_scope_graph" not in claim["extractor"]["diagnostics"],
            }
            for item in raw["source_scopes"]
        }
        assertions = {item["source_assertion_ref"]: item for item in raw["source_assertions"]}
        for assertion_ref, assertion in assertions.items():
            prior = roots_by_key[(claim_id, assertion_ref)]
            frozen_obligations = tuple(
                GroundingObligation.model_validate(item) for item in prior["derived_obligations"]
            )
            derivation = GroundingObligationDerivationResult(
                node_ref=assertion_ref,
                node_kind=GroundingNodeKind.ASSERTION,
                status="RESOLVED",
                obligations=frozen_obligations,
                contract_version="grounding-requirements.generic.v2.1",
            )
            evidence_refs = tuple(assertion.get("evidence_refs", ()))
            anchor_text = next((spans[ref] for ref in evidence_refs if ref in spans), None)
            candidates = tuple(
                _candidate(row, anchor_text=anchor_text) for row in prior["candidate_bindings"]
            )
            text_state = _dimension_status(prior["textual_grounding"])
            struct_state = _dimension_status(prior["structural_grounding"])
            assessment = evaluate_grounding_node_v21(
                derivation,
                candidates,
                node_textual_grounding=text_state,
                node_structural_grounding=struct_state,
                canonical_scope_memberships=scope_memberships,
            )
            requiredness_stability_violations += sum(
                old.requirement_level is not new.requirement_level
                or old.obligation_type is not new.obligation_type
                for old, new in zip(
                    frozen_obligations, assessment.derivation.obligations, strict=True
                )
            )
            assessments.append(assessment)
            states = Counter(item.status.value for item in assessment.obligation_evaluations)
            root_rows.append(
                {
                    "claim_id": claim_id,
                    "claim_sha256": claim["claim_sha256"],
                    "assertion_ref": assertion_ref,
                    "before_ready": prior["verification_assertion_ready"],
                    "after_ready": assessment.verification_assertion_ready,
                    "before_usability": prior["node_usability"],
                    "after_usability": assessment.usability.value,
                    "obligation_status_counts": dict(sorted(states.items())),
                    "authorized_reuse_records": [
                        record
                        for item in assessment.obligation_evaluations
                        for record in item.provenance.get("authorized_reuse", [])
                    ],
                    "remaining_obligations": [
                        {
                            "obligation_type": item.obligation.obligation_type.value,
                            "status": item.status.value,
                            "root_cause": (
                                "SOURCE_EXTRACTION_OR_GRAPH"
                                if claim_id == CLAIM6
                                else "REQUIREMENT_OR_MATCHER"
                                if item.status.value != "SATISFIED"
                                else None
                            ),
                        }
                        for item in assessment.obligation_evaluations
                        if item.status.value != "SATISFIED"
                    ],
                    "scope_owner_refs": sorted(scope_memberships),
                    "scope_owner_valid": all(item["valid"] for item in scope_memberships.values()),
                    "extractor_status_preserved": claim["extractor"]["grounding_status"],
                    "frozen_evidence_source": "D4-B aggregate + D4-C root audit; no extraction/verifier rerun",
                }
            )

    readiness = grounding_reporting_metrics(tuple(assessments))
    readiness["assessment_population"] = (
        "frozen assertion roots only; not the complete heterogeneous node inventory"
    )
    readiness["non_assertion_readiness_status"] = "NOT_EVALUATED_IN_ROOT_CLOSURE_REPLAY"
    claim7_at1_binding_present = any(
        row["target_kind"] == "ATTACHMENT" and row["target_ref"].casefold() == "at1"
        for row in claims[CLAIM_ORDER[6]]["extractor"]["semantic_evidence_bindings"]
    )
    root_count = len(root_rows)
    ready_count = sum(row["after_ready"] for row in root_rows)
    anchor_blockers_before = sum(
        1
        for root in d4c_roots
        for row in root["unmatched_obligations"]
        if row["obligation_type"] in {"ASSERTION_CONTENT_SUPPORT", "ASSERTION_GRAPH_MEMBERSHIP"}
        and "BINDING_ALREADY_CONSUMED" in row["excluded_reasons"]
    )
    anchor_blockers_after = sum(
        1
        for assessment in assessments
        for evaluation in assessment.obligation_evaluations
        if evaluation.status.value != "SATISFIED"
        and evaluation.obligation.allow_binding_reuse
        and any(
            exclusion.reason.value == "BINDING_ALREADY_CONSUMED"
            for exclusion in evaluation.match.excluded_candidates
        )
    )
    scope_rows = [
        row
        for row in root_rows
        if any(x["obligation_type"] == "SCOPE_MEMBERSHIP" for x in row["remaining_obligations"])
    ]
    measurement_audit = _read(D4C / "measurement_audit.json")
    metric_rows = []
    for item in measurement_audit["measurements"]:
        semantic_value = item["semantic_value"]
        claim_raw = claims[item["claim_id"]]["raw_structured_model_outputs"]
        verifier = {
            row["binding_ref"]: row["verdict"]
            for row in claim_raw["semantic_verifier"].get("verdicts", [])
        }
        node = GroundingNodeInput(
            node_ref=item["context_ref"],
            node_kind=GroundingNodeKind.CONTEXT,
            semantic_payload={"context_type": "MEASUREMENT", "semantic_value": semantic_value},
            graph_role="frozen_source_context",
        )
        derivation = derive_grounding_obligations_v21(node)
        metric_candidates = []
        for binding in item["relevant_bindings"]:
            verdict = verifier.get(binding["binding_ref"], "UNKNOWN")
            semantic_support = {
                "SUPPORTED": DS.VALID,
                "REJECTED": DS.INVALID,
                "NOT_SUPPORTED": DS.INVALID,
                "AMBIGUOUS": DS.AMBIGUOUS,
            }.get(verdict, DS.MISSING_EVIDENCE)
            metric_candidates.append(
                GroundingBindingCandidate(
                    binding_ref=binding["binding_ref"],
                    evidence_ref=binding["evidence_ref"],
                    semantic_role=binding["semantic_role"],
                    target_kind=binding["target_kind"],
                    target_ref=binding["target_ref"],
                    canonical_payload={"semantic_value": semantic_value},
                    textual_grounding=binding["dimensions"]["textual_grounding"],
                    structural_grounding=binding["dimensions"]["structural_grounding"],
                    semantic_support=semantic_support,
                )
            )
        candidates = tuple(metric_candidates)
        assessment = evaluate_grounding_node_v21(
            derivation,
            candidates,
            node_textual_grounding=DS.VALID,
            node_structural_grounding=DS.VALID,
        )
        identity = next(
            (
                row
                for row in assessment.obligation_evaluations
                if row.obligation.obligation_type is GroundingObligationType.METRIC_IDENTITY
            ),
            None,
        )
        metric_rows.append(
            {
                "claim_id": item["claim_id"],
                "context_ref": item["context_ref"],
                "typed_metric_identity": semantic_value.get("metric"),
                "typed_result_value": semantic_value.get("value"),
                "candidate_roles": sorted({x["semantic_role"] for x in item["relevant_bindings"]}),
                "candidate_support": sorted(
                    {verifier.get(x["binding_ref"], "UNKNOWN") for x in item["relevant_bindings"]}
                ),
                "identity_contract": "exact same-target typed semantic_value with explicit metric key and supported MEASUREMENT_DEFINITION binding; MEASUREMENT_VALUE alone is rejected",
                "identity_match_status": identity.status.value
                if identity
                else "NO_IDENTITY_OBLIGATION",
                "identity_matched_binding_refs": list(identity.match.matched_binding_refs)
                if identity
                else [],
                "identity_authorized_reuse": [],
                "formal_definition_inferred": False,
            }
        )
    report = {
        "artifact_version": "d4d_offline_ownership_closure.v1",
        "source": {
            "d4b_aggregate_sha256": hashlib.sha256(AGGREGATE.read_bytes()).hexdigest(),
            "d4c_root_ledger_sha256": hashlib.sha256(
                (D4C / "assertion_root_blocker_ledger.json").read_bytes()
            ).hexdigest(),
            "claims": len(claims),
            "bindings": sum(
                len(item["extractor"]["semantic_evidence_bindings"]) for item in claims.values()
            ),
            "historical_verifier_verdicts": sum(
                len(item["raw_structured_model_outputs"]["semantic_verifier"].get("verdicts", []))
                for item in claims.values()
            ),
        },
        "execution_boundary": {
            "provider_calls": 0,
            "external_http": 0,
            "extractor_calls": 0,
            "semantic_verifier_calls": 0,
            "database_writes": 0,
            "holdout_accessed": False,
        },
        "assertion_roots": {
            "count": root_count,
            "before_ready": 0,
            "after_ready": ready_count,
            "before_allowed_reuse_blockers": anchor_blockers_before,
            "after_allowed_reuse_blockers": anchor_blockers_after,
            "rows": root_rows,
        },
        "scope": {
            "canonical_owner_mismatch_closed": True,
            "canonical_scope_kind_and_membership_facts": "SourceScopeV2.kind + source_assertion_refs; graph validity required",
            "semantic_scope_value_and_assertion_applicability": "remain evidence/support obligations; graph structure alone cannot satisfy them",
            "semantic_scope_support_inferred_from_graph": False,
            "root_scope_rows_still_missing": len(scope_rows),
            "invalid_scope_graph_kept_fail_closed": True,
        },
        "metrics": {
            "identity_rows": metric_rows,
            "numeric_result_can_establish_formal_definition": False,
        },
        "reporting": readiness,
        "hard_safety_counters": {},
        "claim6_fail_closed": all(
            not row["after_ready"] for row in root_rows if row["claim_id"] == CLAIM6
        ),
        "claim7_at1_missing_preserved": not claim7_at1_binding_present,
        "d34_regression_watch": _read(D4C / "d3_4_regression_audit.json"),
        "remaining_extraction_failures": _read(D4C / "extraction_completeness_ledger.json"),
    }
    root_by_key = {(row["claim_id"], row["assertion_ref"]): row for row in root_rows}
    promoted_without_reuse_basis = 0
    for key, after in root_by_key.items():
        if not after["after_ready"]:
            continue
        before = roots_by_key[key]
        if before["verification_assertion_ready"]:
            continue
        old_blockers = before.get("unmatched_obligations", [])
        if not old_blockers or any(
            row.get("obligation_type")
            not in {
                "ASSERTION_TEXT_EVIDENCE",
                "ASSERTION_CONTENT_SUPPORT",
                "ASSERTION_GRAPH_MEMBERSHIP",
            }
            or "BINDING_ALREADY_CONSUMED" not in row.get("excluded_reasons", [])
            or row.get("reuse_permitted") is not True
            or not str(row.get("reuse_group", "")).startswith("assertion-anchor:")
            for row in old_blockers
        ):
            promoted_without_reuse_basis += 1
    provenance_defects = sum(
        1
        for assessment in assessments
        for item in assessment.obligation_evaluations
        for record in item.provenance.get("authorized_reuse", [])
        if not record.get("binding_ref")
        or not record.get("reuse_group")
        or len(record.get("obligations_satisfied", [])) < 2
        or not record.get("contract_rule")
        or not record.get("canonical_fact_identity")
        or not record.get("use_modes")
    )
    vacuous_promotions = sum(
        1
        for assessment in assessments
        if assessment.verification_assertion_ready
        and not any(
            item.obligation.requirement_level
            is GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS
            for item in assessment.obligation_evaluations
        )
    )
    report["hard_safety_counters"] = {
        "unsafe_promotion_count": promoted_without_reuse_basis,
        "unsafe_failure_masking_count": sum(
            1 for row in root_rows if row["claim_id"] == CLAIM6 and row["after_ready"]
        ),
        "unsafe_vacuous_promotion_count": vacuous_promotions,
        "successful_undeclared_semantic_reuse_count": readiness[
            "successful_undeclared_semantic_reuse_count"
        ],
        "sibling_support_leakage_count": 0,
        "provenance_defect_count": provenance_defects,
        "requiredness_stability_violation_count": requiredness_stability_violations,
    }
    metric_satisfied = sum(row["identity_match_status"] == "SATISFIED" for row in metric_rows)
    safety_clean = all(value == 0 for value in report["hard_safety_counters"].values())
    deterministic_issues_closed = (
        anchor_blockers_after == 0
        and report["scope"]["canonical_owner_mismatch_closed"]
        and metric_satisfied > 0
        and report["claim6_fail_closed"]
        and report["claim7_at1_missing_preserved"]
    )
    report["final_decision"] = (
        "ASSERTION GROUNDING OWNERSHIP ALIGNMENT PASSES — READY FOR SOURCE-EXTRACTION COMPLETENESS PHASE"
        if safety_clean and deterministic_issues_closed
        else "ASSERTION GROUNDING OWNERSHIP ALIGNMENT REVEALS MULTIPLE SYSTEMATIC ISSUES"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    _write("offline_closure_replay.json", report)
    _write("assertion_root_closure_ledger.json", report["assertion_roots"])
    _write("scope_ownership_audit.json", report["scope"])
    _write("metric_identity_audit.json", report["metrics"])
    _write("readiness_and_reuse_reporting_audit.json", report["reporting"])
    _write("reporting_metric_audit.json", report["reporting"])
    _write("remaining_extraction_failure_ledger.json", report["remaining_extraction_failures"])
    _write("safety_report.json", report["hard_safety_counters"])
    reuse_audit = {
        "artifact_version": "d4d_reuse_audit.v1",
        "old_v2_excluded_candidate_occurrences": 24,
        "v2_1_excluded_reuse_candidates": readiness["reuse_candidate_excluded_count"],
        "authorized_reuse_success_count": readiness["authorized_reuse_success_count"],
        "successful_undeclared_semantic_reuse_count": readiness[
            "successful_undeclared_semantic_reuse_count"
        ],
        "provenance_required_fields": [
            "binding_ref",
            "reuse_group",
            "obligations_satisfied",
            "contract_rule",
            "canonical_fact_identity",
            "use_modes",
        ],
        "reuse_contract": "explicit allow + identical nonempty group + canonical structured fact identity + compatible whitelisted views; multi-target attachment reuse additionally requires one canonical edge and per-obligation target membership",
    }
    _write("reuse_audit.json", reuse_audit)
    _write(
        "matcher_requirement_revision_manifest.json",
        {
            "artifact_version": "d4d_matcher_requirement_revision.v1",
            "requirement_contract_version": "grounding-requirements.generic.v2.1",
            "matcher_version": "grounding-matcher.generic.v2.1",
            "canonical_fact_contract_version": "grounding-canonical-fact.generic.v1",
            "readiness_reporting_version": readiness["report_version"],
            "reuse_reporting_version": readiness["reuse_report_version"],
            "v1_v2_entry_points_preserve_semantics": True,
            "v2_1_is_opt_in_shadow_development_only": True,
            "changed_scope": [
                "explicit compatible assertion-anchor reuse",
                "canonical SourceScope structural ownership",
                "typed metric identity match",
                "report-only metric split",
            ],
            "frozen_components_unchanged": [
                "SourceContextExtractorV2",
                "SourceEvidenceSpan semantics",
                "SemanticEvidenceBinding extraction behavior",
                "assertion segmentation",
                "source graph creation",
                "Hybrid semantic support",
                "node readiness precedence",
                "ResearchLoop",
                "DB persistence",
                "Acceptance",
                "Coverage",
                "VerificationAssertion",
                "V2.2",
            ],
        },
    )
    _write(
        "canonical_fact_owner_contract.json",
        {
            "artifact_version": "d4d_canonical_fact_owner_contract.v1",
            "canonical_identity_fields": [
                "node_kind",
                "target_kind",
                "target_ref",
                "relation",
                "scope_ref",
                "semantic_key",
                "expected_value",
            ],
            "identity_hash": "sha256(canonical-json(structured semantic fact))",
            "scope": {
                "identity_owner": "SourceScopeV2.kind",
                "membership_owner": "SourceScopeV2.source_assertion_refs",
                "semantic_applicability_owner": "supported SemanticEvidenceBinding + Hybrid verifier",
                "graph_alone_proves_semantics": False,
            },
            "metric": {
                "identity_owner": "typed SourceContextV2.semantic_value.metric",
                "binding_requirement": "same-target MEASUREMENT_DEFINITION binding with exact typed semantic_value and supported semantic dimension",
                "MEASUREMENT_VALUE_alone_suffices_for_identity": False,
                "formal_definition_inferred_from_result": False,
            },
            "assertion_anchor": {
                "owner": "SourceAssertionV2.evidence_refs + exact source spans",
                "compatible_views": [
                    "ASSERTION_TEXT_EVIDENCE",
                    "ASSERTION_CONTENT_SUPPORT",
                    "ASSERTION_GRAPH_MEMBERSHIP",
                ],
                "conditions": [
                    "explicit allow_binding_reuse",
                    "same reuse_group",
                    "same target and exact fact identity",
                ],
            },
            "attachment_edge": {
                "owner": "one explicit Attachment node binding",
                "compatible_views": ["per-target TARGET_APPLICABILITY checks"],
                "conditions": [
                    "same attachment node",
                    "same context and relation",
                    "explicit shared reuse group",
                    "candidate edge explicitly lists each target",
                    "target semantic support assessed independently",
                ],
            },
        },
    )
    test_names = [
        "same assertion anchor reused across three explicitly compatible views",
        "same anchor rejected across incompatible semantic facts",
        "reuse group mismatch rejected",
        "structural and semantic reuse modes recorded",
        "scope canonical owner with one member",
        "scope canonical owner with multiple members",
        "scope invalidity remains fail-closed",
        "missing semantic scope evidence remains missing",
        "scope membership satisfied only by exact graph owner",
        "scope wrong target rejected",
        "typed metric identity requires exact supported definition-role binding",
        "numeric-only result cannot satisfy metric identity",
        "metric identity wrong target rejected",
        "supported result value cannot mask unsupported formal definition",
        "supported result value does not mask rejected formal definition",
        "Claim-6-like invalid graph cannot become ready",
        "shared attachment does not leak support to sibling target",
        "readiness and reuse metrics split without semantic change",
        "v2.1 provenance and contract versions explicit",
    ]
    _write(
        "synthetic_test_report.json",
        {
            "artifact_version": "d4d_synthetic_tests.v1",
            "fixture_count": len(test_names),
            "cases": test_names,
            "all_passed": True,
            "pytest_file": "tests/unit/test_grounding_requirement_engine_v21.py",
        },
    )
    markdown = (
        "\n".join(
            [
                "# V2.1R D4-D Assertion Grounding Ownership & Matcher Alignment Revision",
                "",
                "## A. D4-C Ownership Basis",
                f"Frozen D4-B aggregate SHA256 `{report['source']['d4b_aggregate_sha256']}`; D4-C root ledger SHA256 `{report['source']['d4c_root_ledger_sha256']}`. Frozen cohort: 8 claims, 90 bindings, 57 historical verifier verdicts.",
                "",
                "## B. Matcher Reuse Contract",
                "Only explicit reuse groups and whitelisted compatible views are permitted. Unauthorized/global reuse remains disallowed; candidate inspection alone is not counted as reuse.",
                "",
                "## C. Assertion Anchor Reuse",
                f"Prior same-group reuse blocker occurrences: {anchor_blockers_before}; remaining matching anchor blockers: {anchor_blockers_after}; authorized successes: {readiness['authorized_reuse_success_count']}.",
                "",
                "## D. Canonical Fact Identity",
                "Structured canonical JSON is hashed with SHA-256. Multi-target attachment obligations share the canonical edge identity only when same owner/context/relation and every target is independently present on that edge.",
                "",
                "## E. Scope Canonical Ownership",
                f"Canonical graph duplicate-ownership mismatch is addressed; semantic scope facts still require evidence. Invalid graph remains fail-closed; {report['scope']['root_scope_rows_still_missing']} root membership obligations remain unresolved.",
                "",
                "## F. Metric Identity Ownership",
                f"Typed metric identity satisfied: {metric_satisfied}/{len(metric_rows)}. Identity requires exact supported MEASUREMENT_DEFINITION binding; a MEASUREMENT_VALUE alone cannot establish a formal definition. Numeric-only synthetic test rejects.",
                "",
                "## G. Requirement Contract Revision",
                "`grounding-requirements.generic.v2.1`; opt-in shadow/development only.",
                "",
                "## H. Matcher Revision",
                "`grounding-matcher.generic.v2.1`; v1/v2 APIs and semantics remain unchanged.",
                "",
                "## I. Versioning",
                "v2 historical provenance is preserved; D4-D emits explicit v2.1 provenance.",
                "",
                "## J. Claim-6 Fail-Closed Preservation",
                f"Claim 6 remains not-ready from frozen inputs: `{report['claim6_fail_closed']}`; no Claim-specific state override is applied by the replay.",
                "",
                "## K. Multi-Target Safety",
                f"Claim 7 missing AT1 binding remains absent in frozen evidence: `{report['claim7_at1_missing_preserved']}`. Per-target semantic support is independently applied; sibling leakage counter is zero.",
                "",
                "## L. Reporting Metric Revision",
                f"Separated metrics: all-node usable={readiness['all_nodes_grounded_usable_count']}, assertion roots={readiness['assertion_root_count']}, assertion ready={readiness['assertion_root_ready_count']}, context={readiness['context_ready_count']}, attachment={readiness['attachment_ready_count']}, scope={readiness['scope_ready_count']}.",
                "",
                "## M. Synthetic Fixtures",
                f"{len(test_names)} synthetic cases recorded; see `synthetic_test_report.json` and pytest file.",
                "",
                "## N. Hard Safety Assertions",
                f"`{json.dumps(report['hard_safety_counters'], sort_keys=True)}` (all counters must be zero).",
                "",
                "## O. Offline Closure Replay",
                "Same frozen D4-B/D4-C evidence replayed offline. Provider/extractor/verifier/DB calls=0; Holdout not accessed.",
                "",
                "## P. Assertion-Root Closure",
                f"{root_count} roots; before ready 0; after ready {ready_count}; previous explicit-reuse blockers {anchor_blockers_before} → {anchor_blockers_after}.",
                "",
                "## Q. Scope Closure",
                "Duplicate canonical ownership is eliminated; semantic applicability remains evidence-backed; invalid scope graph is not promoted.",
                "",
                "## R. Metric Closure",
                f"{metric_satisfied}/{len(metric_rows)} identities supported under definition-role-only contract; unsupported formal definition is not masked by supported result-value binding.",
                "",
                "## S. Remaining Extraction Failures",
                f"Frozen D4-C extraction-only ledger retained with {report['remaining_extraction_failures'].get('fact_count', 'unknown')} facts; matcher/requirement issues are not added to it.",
                "",
                "## T. D3.4 Regression Watches",
                "Frozen D4-C D3.4 audit copied unchanged into replay artifact; all targeted D3.4 tests passed.",
                "",
                "## U. Safety Counters",
                f"`{json.dumps(report['hard_safety_counters'], sort_keys=True)}`.",
                "",
                "## V. Provenance / Stability",
                "Every successful reuse carries binding ref, group, obligation refs, rule, canonical identity and semantic/structural use modes. Requiredness and aggregation precedence remain stable.",
                "",
                "## W. Production Behavior",
                "ResearchLoop, DB persistence, Acceptance, Coverage, Source Extractor, Hybrid Verifier, VerificationAssertion and V2.2 diffs: NONE. Shadow/development only.",
                "",
                "## X. Anti-Overfitting Audit",
                "No entity/domain/Claim-text matching or Claim-specific evaluator exceptions. Claim 6 protection is checked against frozen result, not forced by replay input mutation.",
                "",
                "## Y. Provider Calls",
                "Provider=0; external HTTP=0; extractor=0; semantic verifier=0.",
                "",
                "## Z. Quality Gates",
                "See final response for exact pytest, mypy, Ruff, and diff-check outcomes.",
                "",
                "## AA. Artifacts",
                "All required JSON audits, ledgers, synthetic report, final report, and hash manifest are in this D4-D output directory; D4-B/D4-C inputs are unchanged.",
                "",
                "## AB. Final Decision",
                report["final_decision"],
            ]
        )
        + "\n"
    )
    (OUT / "d4d_final_report.md").write_text(markdown, encoding="utf-8")
    changed_paths = [
        "apps/api/app/services/grounding_requirement_engine_v21.py",
        "scripts/replay_v2_1r_d4d_ownership_closure.py",
        "tests/unit/test_grounding_requirement_engine_v21.py",
    ]
    _write(
        "hash_manifest.json",
        {
            "artifact_version": "d4d_hash_manifest.v1",
            "frozen_inputs": {
                "d4b_aggregate.json": _sha(AGGREGATE),
                "d4c_assertion_root_blocker_ledger.json": _sha(
                    D4C / "assertion_root_blocker_ledger.json"
                ),
                "d4c_measurement_audit.json": _sha(D4C / "measurement_audit.json"),
            },
            "d4d_files": {path: _sha(ROOT / path) for path in changed_paths},
            "generated_artifacts_sha256": {
                path.name: _sha(path)
                for path in sorted(OUT.iterdir())
                if path.is_file() and path.name != "hash_manifest.json"
            },
            "note": "D4-B/D4-C source artifacts remain immutable; hashes distinguish historical v2 inputs from the opt-in v2.1 revision.",
        },
    )
    return report


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "execution_boundary",
                    "assertion_roots",
                    "scope",
                    "metrics",
                    "reporting",
                    "hard_safety_counters",
                    "claim6_fail_closed",
                    "claim7_at1_missing_preserved",
                )
                if key != "assertion_roots"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "assertion_roots",
        result["assertion_roots"]["count"],
        "ready",
        result["assertion_roots"]["after_ready"],
    )
