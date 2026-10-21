from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from app.domain.atomic_proposition import (
    AtomicProposition,
    ClaimPropositionLink,
    DecompositionMetadata,
)

FIXTURE = Path("tests/fixtures/logical_connectives_v2_1r.json")


def test_neutral_connective_fixture_documents_existing_contract_limit() -> None:
    fixture = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert fixture["synthetic_only"] is True
    assert len(fixture["cases"]) == 7
    assert {case["relation"] for case in fixture["cases"]} == {
        "AND",
        "OR",
        "XOR",
        "BECAUSE",
        "IF_THEN",
        "WHILE",
        "NEGATED_ALTERNATIVES",
    }

    connective_fields = {"connective", "group_id", "logical_relation"}
    assert not connective_fields & AtomicProposition.model_fields.keys()
    assert not connective_fields & DecompositionMetadata.model_fields.keys()
    assert set(ClaimPropositionLink.model_fields) == {
        "claim_id",
        "proposition_id",
        "relation",
        "proposition_index",
        "claim_span",
        "decomposition_status",
    }
    assert fixture["contract_audit"]["separate_asserted_propositions_safely_encode_or"] is False
