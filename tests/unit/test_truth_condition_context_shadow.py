# ruff: noqa: RUF001
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.truth_condition_context import (
    AttachmentStatus,
    ContextFamilyCoverage,
    ContextPresenceConclusion,
    ContextScope,
    ContextScopeKind,
    ContextType,
    ContextValidationOutcome,
    TruthConditionContext,
    adapt_legacy_qualifier_inputs,
    build_verification_assertion_views,
    canonical_context_value,
    compare_context_graphs,
    normalize_semantic_value,
    project_output_context_graph,
    project_source_context_graph,
)

FIXTURE = Path("tests/fixtures/truth_condition_context_shadow_v1.json")


def _cases() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["synthetic_only"] is True
    return data["cases"]


def _case(case_id: str) -> dict[str, Any]:
    return next(item for item in _cases() if item["case_id"] == case_id)


def test_domain_scope_and_context_identity_keep_occurrences_distinct() -> None:
    scope = ContextScope(kind=ContextScopeKind.PROPOSITION_SET, target_refs=("p0", "p1"))
    first = TruthConditionContext(
        context_ref="sc0",
        semantic_signature="same-diagnostic-signature",
        context_type=ContextType.TEMPORAL,
        semantic_value={"year": 2025},
        source_span="2025",
        materiality="MATERIAL",
        scope=scope,
        origin="SOURCE",
    )
    second = first.model_copy(update={"context_ref": "sc1", "source_start": 10})
    assert first.context_ref != second.context_ref
    assert first.semantic_signature == second.semantic_signature
    assert first.scope.target_refs == ("p0", "p1")


def test_source_coverage_is_explicit_by_family_and_absence_is_not_inferred() -> None:
    graph = project_source_context_graph("Atlas reached eight units.", [{"local_ref": "p0"}])
    assert graph.context_presence is ContextPresenceConclusion.NO_CONTEXT_DETECTED
    assert graph.source_detection_complete is False
    assert (
        graph.source_detection_coverage["TEMPORAL"] is ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT
    )
    assert (
        graph.source_detection_coverage["EPISTEMIC"] is ContextFamilyCoverage.PARTIAL_DETECTOR_LIMIT
    )
    result = compare_context_graphs(graph, project_output_context_graph([{"local_ref": "p0"}]))
    assert result[0].outcome is ContextValidationOutcome.DETECTOR_LIMIT
    assert result[0].source_presence_conclusion is ContextPresenceConclusion.DETECTOR_CANNOT_DECIDE
    assert result[0].reason_codes == ("no_context_detected_but_detector_cannot_decide_absence",)


def test_no_context_exists_requires_all_evaluated_families_in_bounded_contract() -> None:
    graph = project_source_context_graph("Atlas reached eight units.", [{"local_ref": "p0"}])
    supported = graph.model_copy(
        update={
            "source_detection_complete": True,
            "source_detection_coverage": {
                key: ContextFamilyCoverage.SUPPORTED_FOR_CONTRACT
                for key in graph.source_detection_coverage
            },
        }
    )
    result = compare_context_graphs(supported, project_output_context_graph([{"local_ref": "p0"}]))
    assert result[0].outcome is ContextValidationOutcome.NOT_APPLICABLE
    assert result[0].source_presence_conclusion is ContextPresenceConclusion.NO_CONTEXT_EXISTS
    assert result[0].reason_codes == ("bounded_contract_confirms_no_material_context",)


def test_temporal_canonicalization_handles_supported_dates_ranges_and_periods() -> None:
    claim = "Between 2025-01-02 and 2025-02-03, Atlas reached 8 units."
    graph = project_source_context_graph(
        claim, [{"local_ref": "p0", "source_span": "Atlas reached 8 units"}]
    )
    # Range separators are deliberately limited to the detector contract.
    context = graph.contexts[0]
    period_a = context.model_copy(
        update={"semantic_value": {"period": "in Q1 2025"}, "lexical_value": "in Q1 2025"}
    )
    period_b = context.model_copy(
        update={"semantic_value": {"period": "Q1 2025"}, "lexical_value": "Q1 2025"}
    )
    assert canonical_context_value(period_a) == canonical_context_value(period_b)
    iso_date = context.model_copy(update={"semantic_value": {"date": "2025-01-02"}})
    assert canonical_context_value(iso_date) == {
        "family": "TEMPORAL",
        "kind": "DATE",
        "value": "2025-01-02",
    }
    iso_range = context.model_copy(
        update={"semantic_value": {"start_date": "2025-01-02", "end_date": "2025-02-03"}}
    )
    assert canonical_context_value(iso_range)["kind"] == "RANGE"
    zh_dates = project_source_context_graph(
        "2025年12月31日发布，统计区间为2026—2030年。",
        [{"local_ref": "p0", "source_span": "发布"}],
    )
    zh_date = next(item for item in zh_dates.contexts if item.source_span == "2025年12月31日")
    zh_range = next(item for item in zh_dates.contexts if item.source_span == "2026—2030年")
    assert canonical_context_value(zh_date) == {
        "family": "TEMPORAL",
        "kind": "DATE",
        "year": 2025,
        "month": 12,
        "day": 31,
    }
    assert canonical_context_value(zh_range) == {
        "family": "TEMPORAL",
        "kind": "YEAR_RANGE",
        "start": 2026,
        "end": 2030,
    }
    quarter_graph = project_source_context_graph(
        "IDC report 2025Q3 says market share is 12%.",
        [{"local_ref": "p0", "source_span": "market share is 12%"}],
    )
    quarter = next(
        item for item in quarter_graph.contexts if item.context_type is ContextType.TEMPORAL
    )
    assert canonical_context_value(quarter) == {
        "family": "TEMPORAL",
        "kind": "NAMED_PERIOD",
        "period": "Q3",
        "year": 2025,
    }


def test_product_identifier_year_is_not_treated_as_temporal_context() -> None:
    graph = project_source_context_graph(
        "The X2000 vision system is designed for industrial inspection.",
        [
            {
                "local_ref": "p0",
                "source_span": "The X2000 vision system is designed for industrial inspection.",
            }
        ],
    )
    assert not any(item.context_type is ContextType.TEMPORAL for item in graph.contexts)


def test_epistemic_canonicalization_preserves_attribution_but_not_plain_assertion() -> None:
    source = project_source_context_graph(
        "Atlas Research reports that Atlas reached 8 units.",
        [{"local_ref": "p0", "source_span": "Atlas reached 8 units"}],
    )
    attributed = project_output_context_graph(
        [{"local_ref": "p0", "attribution": "Atlas Research"}]
    )
    unqualified = project_output_context_graph([{"local_ref": "p0"}])
    assert (
        compare_context_graphs(source, attributed)[0].outcome is ContextValidationOutcome.PRESERVED
    )
    assert (
        compare_context_graphs(source, unqualified)[0].outcome is ContextValidationOutcome.MISSING
    )


def test_canonical_payload_ignores_output_field_wrapper_and_keeps_family() -> None:
    claim = "Under scenario adaptive calibration, Atlas reached 8 units."
    source = project_source_context_graph(claim, [{"local_ref": "p0", "source_span": claim}])
    output = project_output_context_graph(
        [
            {
                "local_ref": "p0",
                "material_qualifiers": [
                    {"category": "scenario", "key": "scenario", "value": "adaptive calibration"}
                ],
            }
        ]
    )
    scenario = next(item for item in source.contexts if item.context_type is ContextType.SCENARIO)
    assert canonical_context_value(scenario)["payload"] == "adaptive calibration"
    assert compare_context_graphs(source, output)[0].outcome is ContextValidationOutcome.PRESERVED


def test_scope_target_set_comparison_distinguishes_under_and_overexpansion() -> None:
    source_claim = (
        "In both statements, within region North, Atlas reached 8 units and Boreal reached 9 units."
    )
    props = [
        {"local_ref": "p0", "source_span": "Atlas reached 8 units"},
        {"local_ref": "p1", "source_span": "Boreal reached 9 units"},
    ]
    source = project_source_context_graph(source_claim, props)
    context = next(item for item in source.contexts if item.context_type is ContextType.SCOPE)
    source = source.model_copy(
        update={
            "contexts": (
                context.model_copy(
                    update={
                        "scope": context.scope.model_copy(
                            update={
                                "kind": ContextScopeKind.PROPOSITION_SET,
                                "target_refs": ("p0", "p1"),
                            }
                        ),
                        "attachments": (
                            context.attachments[0].model_copy(update={"target_ref": "p0"}),
                        ),
                    }
                ),
            ),
        }
    )

    def output_for(refs: tuple[str, ...]):
        return project_output_context_graph(
            [
                {
                    "local_ref": ref,
                    "material_qualifiers": [
                        {"category": "scope", "key": "location", "value": "region North"}
                    ],
                }
                for ref in refs
            ]
        )

    assert (
        compare_context_graphs(source, output_for(("p0",)))[0].outcome
        is ContextValidationOutcome.SCOPE_UNDEREXPANSION
    )
    assert (
        compare_context_graphs(source, output_for(("p0", "p1")))[0].outcome
        is ContextValidationOutcome.PRESERVED
    )
    reverse_source = source.model_copy(
        update={
            "contexts": (
                context.model_copy(
                    update={
                        "scope": context.scope.model_copy(
                            update={"kind": ContextScopeKind.PROPOSITION, "target_refs": ("p0",)}
                        ),
                        "attachments": (
                            context.attachments[0].model_copy(update={"target_ref": "p0"}),
                        ),
                    }
                ),
            )
        }
    )
    assert (
        compare_context_graphs(reverse_source, output_for(("p0", "p1")))[0].outcome
        is ContextValidationOutcome.SCOPE_OVEREXPANSION
    )


def test_source_projection_fixtures_a_to_k_and_n() -> None:
    for case in _cases():
        if case["case_id"] in {
            "L_numeric_canonical_equivalence",
            "M_same_value_wrong_attachment",
            "O_nested_composition_shared_context",
        }:
            continue
        graph = project_source_context_graph(
            case["claim"],
            case["propositions"],
            composition_groups=case.get("composition_groups"),
        )
        types = [context.context_type.value for context in graph.contexts]
        if "expected_types" in case:
            assert types == case["expected_types"], case["case_id"]
        if "expected_scope" in case:
            assert graph.contexts[0].scope.kind.value == case["expected_scope"], case["case_id"]
        if "expected_targets" in case:
            assert graph.contexts[0].scope.target_refs == tuple(case["expected_targets"])
        if "expected_targets_by_context" in case:
            assert [context.scope.target_refs for context in graph.contexts] == [
                tuple(targets) for targets in case["expected_targets_by_context"]
            ], case["case_id"]
        if case["case_id"] == "D_unknown_attachment":
            assert graph.contexts[0].scope.kind is ContextScopeKind.UNKNOWN
            assert graph.contexts[0].attachments[0].status is AttachmentStatus.UNKNOWN
        if case.get("expected_no_method_context"):
            assert ContextType.METHOD not in [context.context_type for context in graph.contexts]
            assert "candidate_rejected_no_semantic_relation" in graph.diagnostics


def test_shared_attribution_is_one_epistemic_context_with_multiple_targets() -> None:
    case = _case("F_shared_attribution")
    graph = project_source_context_graph(case["claim"], case["propositions"])
    assert len(graph.contexts) == 1
    node = graph.contexts[0]
    assert node.context_type is ContextType.EPISTEMIC
    assert node.semantic_value["operator"] == "REPORTS"
    assert node.scope.kind is ContextScopeKind.CLAUSE
    assert node.scope.target_refs == ("p0", "p1")
    assert len(node.attachments) == 2


def test_explicit_claim_scope_and_nested_group_scope_are_supported() -> None:
    claim_case = {
        "claim": "Throughout the claim, in 2025, Atlas reached 8 units; Boreal reached 9 units.",
        "propositions": [
            {"local_ref": "p0", "source_span": "Atlas reached 8 units"},
            {"local_ref": "p1", "source_span": "Boreal reached 9 units"},
        ],
    }
    claim_graph = project_source_context_graph(claim_case["claim"], claim_case["propositions"])
    assert claim_graph.contexts[0].scope.kind is ContextScopeKind.CLAIM
    assert claim_graph.contexts[0].scope.target_refs == ("p0", "p1")

    case = _case("O_nested_composition_shared_context")
    graph = project_source_context_graph(
        case["claim"], case["propositions"], composition_groups=case["composition_groups"]
    )
    assert graph.contexts[0].scope.kind is ContextScopeKind.ASSERTION_GROUP
    assert graph.contexts[0].scope.target_refs == ("g0",)
    assert graph.scope_expansions["g0"] == ("p0", "p1")


def test_source_graph_does_not_infer_attribution_from_retrieval_provenance() -> None:
    claim = "Atlas reached 8 units."
    propositions = [
        {
            "local_ref": "p0",
            "source_span": claim,
            "evidence_source_domain": "research.example",
            "source_owner_key": "owner-1",
        }
    ]
    graph = project_source_context_graph(claim, propositions)
    projected = project_output_context_graph(propositions)
    assert not graph.contexts
    assert not projected.contexts
    assert (
        compare_context_graphs(graph, projected)[0].outcome
        is ContextValidationOutcome.DETECTOR_LIMIT
    )
    complete_no_signal = graph.model_copy(update={"source_detection_complete": True})
    assert (
        compare_context_graphs(complete_no_signal, projected)[0].outcome
        is ContextValidationOutcome.NOT_APPLICABLE
    )


def test_explicit_proposition_set_scope_is_not_attachment_ambiguity() -> None:
    graph = project_source_context_graph(
        "In both statements, during 2025, Atlas reached 8 units and Boreal reached 9 units.",
        [
            {"local_ref": "p0", "source_span": "Atlas reached 8 units"},
            {"local_ref": "p1", "source_span": "Boreal reached 9 units"},
        ],
    )
    temporal = next(
        context for context in graph.contexts if context.context_type is ContextType.TEMPORAL
    )
    assert temporal.scope.kind is ContextScopeKind.PROPOSITION_SET
    assert temporal.scope.target_refs == ("p0", "p1")
    assert all(edge.status is AttachmentStatus.RESOLVED for edge in temporal.attachments)


def test_numeric_exact_normalization_is_typed_and_narrow() -> None:
    pair = _case("L_numeric_canonical_equivalence")["normalization_pair"]
    assert normalize_semantic_value(pair[0]) == normalize_semantic_value(pair[1])
    assert normalize_semantic_value({"value": 38.4, "unit": "%"}) != normalize_semantic_value(
        {"value": 38.4, "unit": "fraction"}
    )
    assert normalize_semantic_value({"value": 10, "unit": "USD"}) != normalize_semantic_value(
        {"value": 10, "unit": "EUR"}
    )


def test_output_projection_uses_only_v3_present_fields_and_compares_shared_context() -> None:
    case = _case("B_shared_context_two_propositions")
    source = project_source_context_graph(case["claim"], case["propositions"])
    output = project_output_context_graph(
        [
            {
                "local_ref": "p0",
                "material_qualifiers": [
                    {"category": "date", "key": "year", "value": "2025", "source_span": "2025"}
                ],
            },
            {
                "local_ref": "p1",
                "material_qualifiers": [
                    {"category": "date", "key": "year", "value": 2025, "source_span": "2025"}
                ],
            },
        ]
    )
    result = compare_context_graphs(source, output)
    assert result[0].outcome is ContextValidationOutcome.PRESERVED
    assert len(result[0].output_context_refs) == 2


def test_same_context_value_with_wrong_attachment_is_scope_mismatch() -> None:
    case = _case("M_same_value_wrong_attachment")
    source = project_source_context_graph(case["claim"], case["source_propositions"])
    output = project_output_context_graph(case["output_propositions"])
    result = compare_context_graphs(source, output)
    assert result[0].outcome is ContextValidationOutcome.SCOPE_MISMATCH


def test_material_unknown_attachment_survives_verification_view() -> None:
    graph = project_source_context_graph("In 2025, Atlas reached 8 units.", [{"local_ref": "p0"}])
    view = build_verification_assertion_views(
        [
            {
                "local_ref": "p0",
                "subject": "Atlas",
                "predicate": "reached",
                "typed_value": {
                    "kind": "quantity",
                    "numeric": {"operator": "exact", "value": 8, "unit": "units"},
                },
                "polarity": "asserted",
            }
        ],
        graph,
    )[0]
    assert view.material_context_unknown is True
    assert view.unknown_context_details == ("sc0",)
    assert view.composition_eligibility == "UNKNOWN"


def test_verification_view_is_derived_and_eligibility_is_composition_input() -> None:
    props = [
        {
            "local_ref": "p0",
            "subject": "Atlas",
            "predicate": "reached",
            "typed_value": {
                "kind": "quantity",
                "numeric": {"operator": "exact", "value": 38.40, "unit": "%"},
            },
            "polarity": "asserted",
        }
    ]
    graph = project_output_context_graph(props)
    view = build_verification_assertion_views(
        props, graph, eligibility_by_ref={"p0": "INDEPENDENTLY_ASSERTED"}
    )[0]
    assert view.proposition_core["typed_value"] == normalize_semantic_value(props[0]["typed_value"])
    assert view.composition_eligibility == "INDEPENDENTLY_ASSERTED"


def test_verification_view_exposes_detection_completeness_separately_from_eligibility() -> None:
    props = [{"local_ref": "p0", "source_span": "Atlas reached 8 units.", "polarity": "asserted"}]
    source = project_source_context_graph("Atlas reached 8 units.", props)
    output = project_output_context_graph(props)
    view = build_verification_assertion_views(
        props, output, source_graph=source, eligibility_by_ref={"p0": "INDEPENDENTLY_ASSERTED"}
    )[0]
    assert view.source_context_completeness == "DETECTOR_LIMIT"
    assert view.source_context_presence == "NO_CONTEXT_DETECTED"
    assert "CONDITION" in view.detector_limited_context_families
    assert view.composition_eligibility == "INDEPENDENTLY_ASSERTED"


def test_legacy_adapter_is_diagnostic_only_and_has_no_competing_truth_state() -> None:
    adapter = adapt_legacy_qualifier_inputs(
        [{"category": "date", "normalized_value": "2025", "source_text": "2025"}],
        [
            {
                "proposition_index": 0,
                "proposition_ref": "p0",
                "carriers": [{"category": "date", "normalized_value": "2025", "value": "2025"}],
            }
        ],
    )
    assert adapter.semantic_preservation_obligation_status == "ADAPT"
    assert adapter.proposition_semantic_view_status == "ADAPT"
    assert adapter.semantic_carrier_status == "DEPRECATE"
    assert adapter.diagnostic_only is True
    assert adapter.source_graph.contexts[0].scope.kind is ContextScopeKind.UNKNOWN
    assert adapter.output_graph.contexts[0].scope.target_refs == ("p0",)
