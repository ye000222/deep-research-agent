from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prospective_acquisition.source_integrity import (
    ACQUISITION_RUNTIME_SOURCE_FILES,
    FINGERPRINT_CONTRACT_VERSION,
    AcquisitionRuntimeFingerprintError,
    fingerprint_acquisition_runtime,
    require_attested_runtime_match,
    validate_runtime_fingerprint_snapshot,
)


def _write_source_set(checkout_root: Path, api_root: Path) -> None:
    for relative in ACQUISITION_RUNTIME_SOURCE_FILES:
        is_api_file = relative.startswith("apps/api/")
        root = api_root if is_api_file else checkout_root
        suffix = relative.removeprefix("apps/api/") if is_api_file else relative
        path = root / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen source: {relative}\n", encoding="utf-8")


class AcquisitionRuntimeFingerprintTests(unittest.TestCase):
    def test_historical_and_current_fingerprints_have_distinct_contracts(self) -> None:
        historical = {
            "generation_contract": "path-c-generation-metadata.v1",
            "implementation_sha256": {"generation.py": "a" * 64},
        }
        self.assertNotEqual(FINGERPRINT_CONTRACT_VERSION, historical["generation_contract"])
        self.assertNotIn("implementation_sha256", {
            "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
            "runtime_source_sha256": {},
            "current_acquisition_runtime_fingerprint": "b" * 64,
        })

    def test_fingerprint_is_deterministic_and_has_exact_allowlist(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            checkout, api = root / "checkout", root / "api"
            _write_source_set(checkout, api)

            first = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)
            second = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)

            self.assertEqual(first, second)
            self.assertEqual(
                first["runtime_source_file_count"], len(ACQUISITION_RUNTIME_SOURCE_FILES)
            )
            self.assertEqual(
                set(first["runtime_source_sha256"]), set(ACQUISITION_RUNTIME_SOURCE_FILES)
            )
            validate_runtime_fingerprint_snapshot(first)

    def test_source_change_changes_fingerprint_and_attestation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            checkout, api = root / "checkout", root / "api"
            _write_source_set(checkout, api)
            attested = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)
            changed = checkout / "scripts/prospective_acquisition/execution_lane.py"
            changed.write_text("changed after host attestation\n", encoding="utf-8")
            current = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)

            self.assertNotEqual(
                attested["current_acquisition_runtime_fingerprint"],
                current["current_acquisition_runtime_fingerprint"],
            )
            attested["status"] = "PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS"
            with self.assertRaisesRegex(
                AcquisitionRuntimeFingerprintError,
                "ACQUISITION_RUNTIME_FINGERPRINT_MISMATCH",
            ):
                require_attested_runtime_match(attested, current)

    def test_non_allowlisted_document_change_does_not_change_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            checkout, api = root / "checkout", root / "api"
            _write_source_set(checkout, api)
            first = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)
            docs = checkout / "docs/notes.md"
            docs.parent.mkdir(parents=True)
            docs.write_text("documentation change", encoding="utf-8")
            second = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)

            self.assertEqual(
                first["current_acquisition_runtime_fingerprint"],
                second["current_acquisition_runtime_fingerprint"],
            )

    def test_attested_snapshot_must_have_exact_keys_and_valid_aggregate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            checkout, api = root / "checkout", root / "api"
            _write_source_set(checkout, api)
            snapshot = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)
            extra_key_snapshot = json.loads(json.dumps(snapshot))
            extra_key_snapshot["runtime_source_sha256"]["tests/not-runtime.py"] = "0" * 64

            with self.assertRaisesRegex(
                AcquisitionRuntimeFingerprintError,
                "ACQUISITION_RUNTIME_ATTESTATION_INVALID",
            ):
                validate_runtime_fingerprint_snapshot(extra_key_snapshot)

    def test_equal_attested_and_current_fingerprint_passes(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            checkout, api = root / "checkout", root / "api"
            _write_source_set(checkout, api)
            snapshot = fingerprint_acquisition_runtime(checkout_root=checkout, api_root=api)

            attested = {
                **snapshot,
                "status": "PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS",
            }
            require_attested_runtime_match(attested, snapshot)


if __name__ == "__main__":
    unittest.main()
