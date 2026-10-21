"""Offline replay for the frozen eight-Claim V2.1R qualifier revision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.services.atomic_proposition_validation import (  # noqa: E402, I001
    QualifierCheckStatus,
    inspect_qualifier_fields,
)


HUMAN_REVIEW: dict[str, str] = {
    "01a0c82e-da26-7f87-9ab5-df0c606b3c90": "MINOR_OTHER_CONTRAST_FRAMING_ONLY",
}


def _status_map(checks: list[dict[str, Any]]) -> dict[str, str]:
    return {item["category"]: item["status"] for item in checks}


def _candidate_indexes(
    category: str, detected: str, propositions: list[dict[str, Any]]
) -> list[int]:
    needle = detected.casefold()
    candidates: list[int] = []
    for index, proposition in enumerate(propositions):
        values = [
            proposition.get("subject", ""),
            proposition.get("predicate", ""),
            proposition.get("proposition_text", ""),
        ]
        typed_value = proposition.get("typed_value")
        if isinstance(typed_value, dict):
            values.extend(str(value) for value in typed_value.values() if isinstance(value, str))
            numeric = typed_value.get("numeric")
            if isinstance(numeric, dict):
                values.extend(str(value) for value in numeric.values() if isinstance(value, str))
        values.extend(
            str(field.get(key, ""))
            for field in proposition.get("material_qualifiers", [])
            if isinstance(field, dict)
            for key in ("category", "key", "value", "source_span")
        )
        if any(needle in value.casefold() for value in values if value):
            candidates.append(index)
    return candidates


def _reason_code(category: str, detected: str, after: str, claim_text: str) -> str:
    if category == "unit" and "年" in detected:
        return "NORMALIZATION_FAILURE"
    if category in {"method", "sample_count"} and detected in {"采用", "一份"}:
        return "SOURCE_DETECTOR_FALSE_POSITIVE"
    if category == "negation" and (
        "无法" in claim_text or "不能" in claim_text or "cannot" in claim_text.casefold()
    ):
        return "STRUCTURED_FIELD_NOT_CONSIDERED"
    if category == "negation" and detected in {"无", "不"}:
        return "SOURCE_DETECTOR_FALSE_POSITIVE"
    if category in {"negation", "measurement"} and after == QualifierCheckStatus.PRESERVED.value:
        return "STRUCTURED_FIELD_NOT_CONSIDERED"
    return "INSUFFICIENT_EVIDENCE"


def replay(input_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        result = artifact["result"]
        claim = artifact["claim"]
        propositions = result["model_structured_result"]["propositions"]
        before_checks = result["validator_result"]["qualifier_checks"]
        before = _status_map(before_checks)
        after_checks = inspect_qualifier_fields(claim["claim_text"], propositions)
        after = {item.category: item.status.value for item in after_checks}
        warnings: list[dict[str, Any]] = []
        for check in before_checks:
            if check["status"] not in {
                QualifierCheckStatus.POTENTIAL_LOSS.value,
                QualifierCheckStatus.AMBIGUOUS.value,
            }:
                continue
            category = check["category"]
            detected = check.get("detected_text") or ""
            warnings.append(
                {
                    "category": category,
                    "source_span": detected,
                    "candidate_proposition_indexes": _candidate_indexes(
                        category, detected, propositions
                    ),
                    "candidate_propositions": propositions,
                    "structured_fields": [
                        {
                            "proposition_index": index,
                            "subject": proposition.get("subject"),
                            "predicate": proposition.get("predicate"),
                            "polarity": proposition.get("polarity"),
                            "modality": proposition.get("modality"),
                            "attribution": proposition.get("attribution"),
                            "typed_value": proposition.get("typed_value"),
                            "material_qualifiers": proposition.get("material_qualifiers", []),
                        }
                        for index, proposition in enumerate(propositions)
                    ],
                    "before_decision": check["status"],
                    "after_decision": after.get(category),
                    "reason_code": _reason_code(
                        category, detected, after.get(category, ""), claim["claim_text"]
                    ),
                    "human_review": HUMAN_REVIEW.get(
                        claim["claim_id"], "NO_HUMAN_CONFIRMED_MATERIAL_QUALIFIER_LOSS"
                    ),
                }
            )
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "stratum": claim["stratum"],
                "claim_text": claim["claim_text"],
                "before": before,
                "after": after,
                "before_unresolved": any(
                    status
                    in {
                        QualifierCheckStatus.POTENTIAL_LOSS.value,
                        QualifierCheckStatus.AMBIGUOUS.value,
                    }
                    for status in before.values()
                ),
                "after_unresolved": any(
                    status
                    in {
                        QualifierCheckStatus.POTENTIAL_LOSS.value,
                        QualifierCheckStatus.AMBIGUOUS.value,
                    }
                    for status in after.values()
                ),
                "warnings": warnings,
                "provider_calls": 0,
            }
        )
    return {
        "replay_version": "v2.1r-qualifier-equivalence-offline-v1",
        "provider_calls": 0,
        "claims_replayed": len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: result[key] for key in ("claims_replayed", "provider_calls")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
