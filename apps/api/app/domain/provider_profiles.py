"""Provider profile domain records returned without plaintext credentials."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.domain.providers import AdapterType


@dataclass(frozen=True, slots=True)
class ProviderProfileView:
    profile_id: UUID
    name: str
    adapter_type: AdapterType
    base_url: str
    endpoint_host: str
    model: str
    status: str
    version: int
    is_default: bool
    credential_version_id: UUID
    credential_version: int
    credential_last_four: str
    credential_fingerprint: str
    created_at: datetime
    updated_at: datetime
