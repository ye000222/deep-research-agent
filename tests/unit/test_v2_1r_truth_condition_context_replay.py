from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_v2_1r_truth_condition_context as replay  # noqa: E402


def test_normalizes_v3_and_legacy_proposition_field_names_without_losing_context() -> None:
    proposition = replay.normalize_proposition(
        {
            "proposition_id": "p3",
            "original_text": "Atlas reached 38.40% in 2025",
            "subject": "Atlas",
            "predicate": "reached",
            "value": {"numeric": {"value": 38.40, "unit": "%"}},
            "qualifiers": [
                {"category": "date", "key": "year", "value": "2025", "source_span": "2025"}
            ],
            "decomposition": {"source_span": "Atlas reached 38.40%"},
        },
        0,
    )
    assert proposition["local_ref"] == "p3"
    assert proposition["source_span"] == "Atlas reached 38.40%"
    assert proposition["typed_value"] == {"numeric": {"value": 38.40, "unit": "%"}}
    assert proposition["material_qualifiers"] == [
        {"category": "date", "key": "year", "value": "2025", "source_span": "2025"}
    ]


def test_context_supplement_does_not_overwrite_composition_eligibility() -> None:
    proposition = {
        "local_ref": "p0",
        "subject": "Atlas",
        "predicate": "reached",
        "typed_value": {"kind": "quantity", "value": 9, "unit": "units"},
        "polarity": "POSITIVE",
        "material_qualifiers": [],
    }
    graph = replay.tcc.project_output_context_graph([proposition])
    view = replay.tcc.build_verification_assertion_views(
        [proposition], graph, eligibility_by_ref={"p0": "INDEPENDENTLY_ASSERTED"}
    )[0]
    source_graph = replay.tcc.project_source_context_graph(
        "Atlas reached 9 units.",
        [{"local_ref": "p0", "source_span": "Atlas reached 9 units"}],
    )
    comparison = replay.tcc.compare_context_graphs(source_graph, graph)
    supplemental = replay.supplemental_view(view, source_graph, graph, comparison)
    assert supplemental["composition_eligibility"] == "INDEPENDENTLY_ASSERTED"
    assert supplemental["context_completeness"] == "DETECTOR_LIMIT"
    assert supplemental["material_context_unknown"] is True
    assert supplemental["epistemic_context_signatures"] == []


def test_source_and_output_projection_is_deterministic_and_reports_context_loss() -> None:
    row = {
        "cohort": "fixture",
        "claim_id": "fixture-1",
        "claim_text": "In 2025, Atlas reached 8 units.",
        "identity_hash": None,
        "propositions": [
            {
                "local_ref": "p0",
                "source_span": "Atlas reached 8 units",
                "subject": "Atlas",
                "predicate": "reached",
                "typed_value": {"kind": "quantity", "value": 8, "unit": "units"},
                "polarity": "POSITIVE",
                "material_qualifiers": [],
            }
        ],
        "composition_groups": [],
        "eligibility_by_ref": {"p0": "INDEPENDENTLY_ASSERTED"},
        "metadata": {},
        "raw_record": {"propositions": []},
    }
    result = replay.claim_replay(row)
    assert result["context_outcome_counts"].get("MISSING") == 1
    assert result["composition_eligibility_counts_independent_of_context"] == {
        "INDEPENDENTLY_ASSERTED": 1
    }


def test_historical_cohort_loader_enforces_exactly_128_disjoint_development_claims() -> None:
    cohorts = replay.load_cohorts()
    assert {name: len(rows) for name, rows in cohorts.items()} == {
        "bounded_shadow": 100,
        "fresh_validation": 12,
        "final_unseen": 16,
    }
    claim_ids = [row["claim_id"] for rows in cohorts.values() for row in rows]
    assert len(claim_ids) == 128
    assert len(set(claim_ids)) == 128
    assert all(row["claim_text"] for rows in cohorts.values() for row in rows)
