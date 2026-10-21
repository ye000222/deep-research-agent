"""Offline comparison of frozen D4-J results with the isolated v3.3 contract."""

# The embedded replay report includes complete audit caveats in prose.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.source_context_extraction_v3_3 import (
    MeasurementIdentityTypeV33,
    SourceExtractionResponseV33,
)
from app.services.source_context_extractor_v3_3 import (
    SOURCE_CONTEXT_V33_INSTRUCTIONS,
    evaluate_measurement_admissibility_v33,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/v2_1r_d4j_b"
V32_REPLAY = ROOT / "artifacts/v2_1r_d4m/d4j_offline_replay.json"
OUTPUT = ROOT / "artifacts/v2_1r_d4p"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_v31_to_v33(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap legacy names without inferring missing type or synthesizing spans."""
    adapted = dict(payload)
    adapted["schema_version"] = "source-context-extraction-schema-v3.3"
    contexts: list[dict[str, Any]] = []
    for original in payload.get("source_contexts", []):
        context = dict(original)
        semantic = dict(context.get("semantic_value") or {})
        old_identity = semantic.get("measurement_identity")
        if context.get("context_type") == "MEASUREMENT" and isinstance(old_identity, str):
            semantic["measurement_identity"] = {
                "name": old_identity,
                "identity_type": MeasurementIdentityTypeV33.UNKNOWN.value,
                "source_span": None,
            }
        context["semantic_value"] = semantic
        contexts.append(context)
    adapted["source_contexts"] = contexts
    adapted["semantic_relations"] = [
        {**relation, "schema_version": "source-context-extraction-schema-v3.3"}
        for relation in payload.get("semantic_relations", [])
    ]
    return adapted


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "source-context-extraction-schema-v3.3.json").write_text(
        json.dumps(SourceExtractionResponseV33.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "source-context-extraction.generic.v3.3.md").write_text(
        SOURCE_CONTEXT_V33_INSTRUCTIONS + "\n", encoding="utf-8"
    )
    v32 = _read(V32_REPLAY)
    v32_by_artifact = {row["artifact"]: row for row in v32["claims"]}
    claim_rows: list[dict[str, Any]] = []
    parser_pass = 0
    parser_fail = 0
    measurement_contexts = 0
    admissibility_counts = {"ADMISSIBLE": 0, "NOT_MEASUREMENT": 0, "UNKNOWN": 0}

    for path in sorted(INPUT.glob("claim_*.json")):
        frozen = _read(path)
        claim = frozen["claim"]
        raw = frozen["raw_output"]["structured_response"]["model_structured_object"]
        adapted = _legacy_v31_to_v33(raw)
        row: dict[str, Any] = {
            "artifact": path.name,
            "claim_id": claim["claim_id"],
            "claim_sha256": claim["claim_sha256"],
            "v3_2": v32_by_artifact.get(path.name, {}),
            "legacy_adapter": {
                "used": True,
                "identity_name_preserved": True,
                "identity_type_inferred": False,
                "source_span_synthesized": False,
                "missing_v3_3_identity_type": "UNKNOWN",
            },
        }
        try:
            response = SourceExtractionResponseV33.model_validate(adapted)
        except Exception as exc:  # diagnostics contain only Pydantic error types
            parser_fail += 1
            error_types = sorted(
                {str(item.get("type", "invalid")) for item in exc.errors()}
            ) if hasattr(exc, "errors") else [type(exc).__name__]
            row.update(parser_status="INVALID", parser_error_types=error_types[:12])
            claim_rows.append(row)
            continue

        parser_pass += 1
        decisions = evaluate_measurement_admissibility_v33(response, claim["claim_text"])
        statuses = {ref: decision.value for ref, decision in decisions.items()}
        for status in statuses.values():
            admissibility_counts[status] += 1
        count = len(decisions)
        measurement_contexts += count
        row.update(
            parser_status="PASS",
            v3_3_measurement_context_count=count,
            v3_3_admissibility_by_context=statuses,
        )
        claim_rows.append(row)

    summary = {
        "artifact_version": "v2_1r_d4p_offline_replay.v1",
        "input_manifest": "artifacts/v2_1r_d4j_b/raw_freeze_manifest.json",
        "comparison_artifact": "artifacts/v2_1r_d4m/d4j_offline_replay.json",
        "method": "Offline deterministic schema adaptation and v3.3 structural/source-reference admissibility evaluation. Legacy names are wrapped as UNKNOWN; no semantic type or source span is inferred.",
        "claims_total": len(claim_rows),
        "v3_3_parser_pass": parser_pass,
        "v3_3_parser_fail": parser_fail,
        "v3_3_measurement_contexts": measurement_contexts,
        "v3_3_admissibility_counts": admissibility_counts,
        "v3_2_parser_pass": v32["parser_pass_count"],
        "v3_2_measurement_contexts": v32["measurement_context_count"],
        "v3_2_admissibility_pass_claims": v32["measurement_admissibility_pass_claims"],
        "comparison": {
            "v3_2": "Frozen D4-M replay accepted structural anchors; five claims passed Measurement admissibility, including the known qualitative false-positive structure.",
            "v3_3": "All legacy identity strings lack explicit v3.3 identity_type and nested source_span. The adapter preserves names, assigns UNKNOWN, synthesizes no span, and therefore fails closed as UNKNOWN rather than promoting them.",
            "semantic_classification_from_text": False,
        },
        "provider_calls": 0,
        "external_http": 0,
        "db_writes": 0,
        "holdout_access": 0,
        "claims": claim_rows,
    }
    (OUTPUT / "offline_replay.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = (
        "# D4-P Offline Replay\n\n"
        f"- Claims: {len(claim_rows)}\n"
        f"- v3.3 parser: {parser_pass} PASS / {parser_fail} FAIL\n"
        f"- v3.3 Measurement contexts: {measurement_contexts}\n"
        f"- v3.3 admissibility: {admissibility_counts}\n"
        f"- Frozen v3.2 comparison: parser {v32['parser_pass_count']}/"
        f"{v32['claims_total']} PASS; {v32['measurement_admissibility_pass_claims']} claims "
        f"passed Measurement admissibility across {v32['measurement_context_count']} contexts.\n\n"
        "The D4-J legacy adapter preserves the identity string as `name`, leaves its type as "
        "`UNKNOWN`, and does not synthesize a nested source span. Thus existing D4-J outputs "
        "parse under the v3.3 envelope, but their Measurement contexts fail closed as UNKNOWN "
        "until an explicit v3.3 identity classification and source-span reference exist. "
        "This is an offline format comparison, not model re-extraction or semantic inference.\n"
    )
    (OUTPUT / "offline_replay.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT),
                "claims": len(claim_rows),
                "parser_pass": parser_pass,
                "parser_fail": parser_fail,
                "measurement_contexts": measurement_contexts,
                "admissibility": admissibility_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
