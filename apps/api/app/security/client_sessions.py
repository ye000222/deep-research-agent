"""Signed, opaque browser-session identifiers for single-user V1 isolation."""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from uuid import UUID, uuid4

from fastapi import Response


@dataclass(frozen=True, slots=True)
class ClientSession:
    client_id: UUID
    owner_hash: str
    is_new: bool


class ClientSessionManager:
    def __init__(self, signing_key: bytes, *, cookie_name: str) -> None:
        self._key = hmac.new(signing_key, b"client-session-signing-v1", hashlib.sha256).digest()
        self.cookie_name = cookie_name

    def resolve(self, token: str | None) -> ClientSession:
        client_id = self._verify(token) if token else None
        is_new = client_id is None
        client_id = client_id or uuid4()
        owner_hash = hmac.new(
            self._key,
            b"provider-profile-owner-v1:" + client_id.bytes,
            hashlib.sha256,
        ).hexdigest()
        return ClientSession(client_id=client_id, owner_hash=owner_hash, is_new=is_new)

    def issue_cookie(self, response: Response, session: ClientSession, *, secure: bool) -> None:
        if not session.is_new:
            return
        payload = session.client_id.hex
        signature = hmac.new(self._key, payload.encode("ascii"), hashlib.sha256).digest()
        token = f"{payload}.{_urlsafe(signature)}"
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=31_536_000,
            httponly=True,
            secure=secure,
            samesite="lax",
            path="/",
        )

    def _verify(self, token: str) -> UUID | None:
        try:
            payload, supplied = token.split(".", maxsplit=1)
            client_id = UUID(hex=payload)
            expected = hmac.new(self._key, payload.encode("ascii"), hashlib.sha256).digest()
            supplied_bytes = _urlsafe_decode(supplied)
        except (ValueError, TypeError):
            return None
        return client_id if hmac.compare_digest(expected, supplied_bytes) else None


def _urlsafe(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
