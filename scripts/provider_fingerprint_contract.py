"""Secret-free, versioned Provider identity fingerprinting for readiness audits only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

FINGERPRINT_CONTRACT_VERSION = "provider-safe-config-fingerprint.v1"
_SAFE_SETTING_KEYS = frozenset(
    {
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "timeout_seconds",
        "temperature",
        "max_output_tokens",
        "structured_output_mode",
        "supports_json_mode",
        "reasoning_enabled",
    }
)


def _normalize_endpoint(raw_url: str) -> dict[str, str | int | None]:
    parsed = urlsplit(raw_url.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("provider_endpoint_not_safe_to_fingerprint")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        port = None
    path = parsed.path.rstrip("/")
    return {"scheme": scheme, "host": host, "port": port, "path": path}


def _safe_settings(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key in sorted(_SAFE_SETTING_KEYS):
        value = raw.get(key)
        if isinstance(value, bool | int | float | str) or value is None:
            safe[key] = value
    return safe


def build_safe_provider_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Project only explicitly safe identity/config fields; never accept secret material."""
    credential = profile.get("credential")
    if not isinstance(credential, Mapping):
        credential = {}
    profile_id = profile.get("profile_id")
    adapter = profile.get("adapter")
    model = profile.get("model")
    profile_version = profile.get("profile_version")
    credential_version = credential.get("version")
    present = credential.get("present")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("provider_identity_required")
    if not isinstance(adapter, str) or not adapter.strip():
        raise ValueError("provider_identity_required")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("provider_identity_required")
    if not isinstance(profile_version, int) or isinstance(profile_version, bool):
        raise ValueError("profile_version_required")
    if not isinstance(credential_version, int) or isinstance(credential_version, bool):
        raise ValueError("credential_version_required")
    if not isinstance(present, bool):
        raise ValueError("credential_presence_required")
    credential_id = credential.get("id")
    if credential_id is not None and not isinstance(credential_id, str):
        raise ValueError("credential_id_must_be_safe_string_or_null")

    endpoint = profile.get("base_url")
    if not isinstance(endpoint, str):
        raise ValueError("provider_endpoint_required")

    auth_type = profile.get("auth_type")
    if not isinstance(auth_type, str) or not auth_type.strip():
        raise ValueError("provider_auth_type_required")

    return {
        "contract_version": FINGERPRINT_CONTRACT_VERSION,
        "profile_id": profile_id.strip().lower(),
        "adapter": adapter.strip(),
        "model": model.strip(),
        "endpoint": _normalize_endpoint(endpoint),
        "auth_type": auth_type.strip().lower(),
        "profile_version": profile_version,
        "credential": {
            # Missing and explicit null are intentionally equivalent under v1.
            "id": credential_id.strip().lower() if credential_id else None,
            "version": credential_version,
            "present": present,
        },
        "safe_settings": _safe_settings(profile.get("non_secret_settings")),
    }


def canonical_safe_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def safe_provider_fingerprint(profile: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = build_safe_provider_payload(profile)
    digest = hashlib.sha256(canonical_safe_payload_bytes(payload)).hexdigest()
    return payload, digest
