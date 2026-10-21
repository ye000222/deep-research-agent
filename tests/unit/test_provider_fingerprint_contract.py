from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from scripts.provider_fingerprint_contract import (
    build_safe_provider_payload,
    safe_provider_fingerprint,
)


@pytest.fixture
def profile() -> dict[str, Any]:
    return {
        "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
        "adapter": "openai_compatible_chat",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1/",
        "auth_type": "bearer",
        "profile_version": 23,
        "credential": {"id": "cred-version-id", "version": 4, "present": True},
        "non_secret_settings": {"temperature": 0.0, "timeout_seconds": 30},
        "updated_at": "dynamic-and-not-in-contract",
        "container_id": "dynamic-and-not-in-contract",
    }


def test_same_safe_payload_produces_same_fingerprint(profile: dict[str, Any]) -> None:
    assert safe_provider_fingerprint(profile)[1] == safe_provider_fingerprint(profile)[1]


def test_mapping_insertion_order_does_not_change_fingerprint(profile: dict[str, Any]) -> None:
    reversed_profile = dict(reversed(list(profile.items())))
    reversed_profile["credential"] = dict(reversed(list(profile["credential"].items())))
    assert safe_provider_fingerprint(profile)[1] == safe_provider_fingerprint(reversed_profile)[1]


def test_semantic_provider_change_changes_fingerprint(profile: dict[str, Any]) -> None:
    changed = deepcopy(profile)
    changed["model"] = "different-model"
    assert safe_provider_fingerprint(profile)[1] != safe_provider_fingerprint(changed)[1]


def test_dynamic_non_contract_fields_do_not_change_fingerprint(profile: dict[str, Any]) -> None:
    changed = {**profile, "updated_at": "later", "container_id": "new-container"}
    assert safe_provider_fingerprint(profile)[1] == safe_provider_fingerprint(changed)[1]


def test_optional_credential_id_none_and_omitted_are_equivalent(
    profile: dict[str, Any],
) -> None:
    explicit = deepcopy(profile)
    explicit["credential"] = {"id": None, "version": 4, "present": True}
    omitted = deepcopy(profile)
    omitted["credential"] = {"version": 4, "present": True}
    assert safe_provider_fingerprint(explicit)[1] == safe_provider_fingerprint(omitted)[1]
    assert build_safe_provider_payload(omitted)["credential"]["id"] is None


def test_endpoint_trailing_slash_and_default_port_are_normalized(
    profile: dict[str, Any],
) -> None:
    normalized = deepcopy(profile)
    normalized["base_url"] = "HTTPS://API.DEEPSEEK.COM:443/v1"
    assert safe_provider_fingerprint(profile)[1] == safe_provider_fingerprint(normalized)[1]


def test_secrets_and_unapproved_settings_never_enter_payload(
    profile: dict[str, Any],
) -> None:
    with_secret = {
        **profile,
        "api_key": "must-not-be-captured",
        "authorization": "must-not-be-captured",
        "non_secret_settings": {
            "temperature": 0.0,
            "extra_headers": {"Authorization": "must-not-be-captured"},
            "api_key": "must-not-be-captured",
        },
    }
    payload = build_safe_provider_payload(with_secret)
    rendered = repr(payload)
    assert "must-not-be-captured" not in rendered
    assert "api_key" not in rendered
    assert "extra_headers" not in rendered


def test_credential_safe_metadata_is_part_of_fingerprint(profile: dict[str, Any]) -> None:
    changed = deepcopy(profile)
    changed["credential"] = {"id": "other-credential-id", "version": 4, "present": True}
    assert safe_provider_fingerprint(profile)[1] != safe_provider_fingerprint(changed)[1]


def test_endpoint_credentials_query_and_fragment_are_rejected(profile: dict[str, Any]) -> None:
    for endpoint in (
        "https://user:pass@api.deepseek.com/v1",
        "https://api.deepseek.com/v1?token=x",
        "https://api.deepseek.com/v1#fragment",
    ):
        changed = {**profile, "base_url": endpoint}
        with pytest.raises(ValueError, match="provider_endpoint_not_safe"):
            build_safe_provider_payload(changed)
