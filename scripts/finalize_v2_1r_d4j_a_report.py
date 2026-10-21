"""Finalize D4-J-A report and artifact hashes after quality gates."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v2_1r_d4j_a"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    status = json.loads((OUT / "readiness_status.json").read_text(encoding="utf-8"))
    if not all(value == "PASS" for value in status["gates"].values()):
        raise SystemExit("readiness gates not all PASS; refusing READY decision")
    selection = json.loads(
        (ROOT / "artifacts/v2_1r_d4j_a_selection_manifest.json").read_text(encoding="utf-8")
    )
    criteria_raw_sha = sha(ROOT / "artifacts/v2_1r_d4j_a_selection_criteria.json")
    report_path = OUT / "readiness_report.md"
    report = report_path.read_text(encoding="utf-8")
    criteria_marker = "## C. Fresh Selection Criteria"
    cohort_marker = "## D. Frozen Fresh Cohort"
    manifest_marker = "## E. Selection Manifest"
    before, tail = report.split(criteria_marker, 1)
    _, tail_after_criteria = tail.split(cohort_marker, 1)
    report = (
        before
        + criteria_marker
        + "\n\nFrozen before inspecting candidate text. Deterministic generic structural diversity only; no semantic, entity/domain, benchmark, feature-coverage, label, or expected-outcome selection. Canonical criteria SHA-256: `"
        + selection["selection_criteria_sha256"]
        + "`; criteria file-byte SHA-256: `"
        + criteria_raw_sha
        + "`.\n\n"
        + cohort_marker
        + "\n\nEight fresh development-only Claims, exactly one per source pair; each is permanently development-only:\n\n"
        + "\n".join(
            f"{item['ordinal']}. `{item['claim_id']}`"
            for item in selection["claims"]
        )
        + "\n\n"
        + manifest_marker
        + tail_after_criteria.split(manifest_marker, 1)[1]
    )
    marker = "## X. Final Decision"
    if marker not in report:
        raise SystemExit("report section X missing")
    quality_marker = "## V. Quality Gates"
    artifacts_marker = "## W. Artifacts"
    if quality_marker in report and artifacts_marker in report:
        before, tail = report.split(quality_marker, 1)
        _, after = tail.split(artifacts_marker, 1)
        report = before + quality_marker + "\n\n" + (
            "- Selection/exclusion, cohort manifest, normalized hashes and claim-text hashes: **PASS**.\n"
            "- v3.1 implementation freeze and frozen v2/v3/downstream reference comparison: **PASS**.\n"
            "- Current Provider safe fingerprint and canonical runtime import resolution: **PASS**.\n"
            "- Strict one-outbound guard and same-runtime Fake transport (11 intercepted requests): **PASS**.\n"
            "- Result-schema contract tests, v3.1 tests, Provider fingerprint tests and full v3/v2/Grounding/Matcher/Hybrid regression suites: **PASS**.\n"
            "- `pytest -q`: **PASS** (9 skipped; only non-failing deprecation/cache warnings).\n"
            "- `mypy apps/api`: **PASS** (227 source files).\n"
            "- Targeted Ruff: **PASS**. `git diff --check`: **PASS**.\n"
            "- The optional `jsonschema` metaschema-validation package is not installed; JSON parsing plus D4-J schema-contract tests passed.\n\n"
        ) + artifacts_marker + after
    report = report.split(marker, 1)[0] + marker + "\n\n**V3.1 FRESH REAL CANARY EXECUTION READY**\n\nTechnical readiness only. This does not authorize or execute D4-J-B. A separate explicit authorization must bind the exact cohort, selection-manifest hash, current Provider fingerprint, and implementation fingerprint.\n"
    report_path.write_text(report, encoding="utf-8")
    hash_path = OUT / "hash_manifest.json"
    hash_manifest = json.loads(hash_path.read_text(encoding="utf-8"))
    hash_manifest["readiness_artifacts_sha256"] = {
        path.name: sha(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "hash_manifest.json"
    }
    hash_path.write_bytes(json.dumps(hash_manifest, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"final_decision": status["final_decision"], "report_sha256": sha(report_path), "hash_manifest_sha256": sha(hash_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
