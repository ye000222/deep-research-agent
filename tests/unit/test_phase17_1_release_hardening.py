"""Phase 17.1 release-hardening invariants."""

from __future__ import annotations

from app.core.config import Settings

from scripts.analyze_phase17_1_rc import (
    EXPECTED_FLAGS,
    RC_VERSION,
    REFERENCE_CONFIG_ID,
    stamped_identity,
)


def test_rc_reference_defaults_are_frozen() -> None:
    settings = Settings(_env_file=None)
    assert settings.evidence_aware_context_enabled is False
    assert settings.independent_source_targeting_enabled is True
    assert settings.evidence_input_quality_enabled is False


def test_run_provenance_contains_rc_identity_and_flags() -> None:
    identity = stamped_identity(
        {
            "source_revision": "candidate-build",
            "benchmark": {
                "reference_config_id": REFERENCE_CONFIG_ID,
                "benchmark_id": "v1-dev-model-comparison",
                "benchmark_version": "1.0.0",
                "metric_definition_version": "coverage-v1",
                "tier": "standard",
                "feature_flags": EXPECTED_FLAGS,
            },
        }
    )
    assert identity["reference_config_id"] == REFERENCE_CONFIG_ID
    assert identity["feature_flags"] == EXPECTED_FLAGS
    assert identity["benchmark_id"] == "v1-dev-model-comparison"


def test_legacy_absent_input_quality_flag_is_auditable_false() -> None:
    identity = stamped_identity(
        {
            "benchmark": {
                "reference_config_id": REFERENCE_CONFIG_ID,
                "feature_flags": {
                    "evidence_aware_context_enabled": False,
                    "independent_source_targeting_enabled": True,
                },
            }
        }
    )
    assert identity["feature_flags"] == EXPECTED_FLAGS


def test_release_metadata_is_explicit() -> None:
    assert RC_VERSION == "v1.0.0-rc.1"
    assert REFERENCE_CONFIG_ID == "v1-pre-rc-reference-1"
