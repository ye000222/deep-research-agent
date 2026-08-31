"""Authenticated encryption for persisted provider credentials."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class EncryptedSecret:
    ciphertext: bytes
    nonce: bytes
    key_version: int
    aad_version: int
    hmac_fingerprint: str
    last_four: str


class SecretCipher:
    """AES-256-GCM cipher whose key never enters domain state or persistence."""

    KEY_BYTES = 32
    NONCE_BYTES = 12

    def __init__(self, master_key: bytes, *, key_version: int = 1) -> None:
        if len(master_key) != self.KEY_BYTES:
            raise ValueError("secret master key must be exactly 32 bytes")
        self._master_key = master_key
        self._aead = AESGCM(master_key)
        self.key_version = key_version

    def encrypt(
        self,
        secret: SecretStr,
        *,
        credential_id: UUID,
        adapter_type: str,
        credential_version: int,
    ) -> EncryptedSecret:
        plaintext = secret.get_secret_value().encode("utf-8")
        if not plaintext:
            raise ValueError("provider API key may not be empty")
        nonce = os.urandom(self.NONCE_BYTES)
        aad = self._aad(credential_id, adapter_type, credential_version)
        ciphertext = self._aead.encrypt(nonce, plaintext, aad)
        fingerprint = hmac.new(
            self._master_key,
            b"provider-key-fingerprint-v1:" + plaintext,
            hashlib.sha256,
        ).hexdigest()
        return EncryptedSecret(
            ciphertext=ciphertext,
            nonce=nonce,
            key_version=self.key_version,
            aad_version=1,
            hmac_fingerprint=fingerprint,
            last_four=plaintext[-4:].decode("utf-8", errors="replace"),
        )

    def decrypt(
        self,
        encrypted: EncryptedSecret,
        *,
        credential_id: UUID,
        adapter_type: str,
        credential_version: int,
    ) -> SecretStr:
        if encrypted.key_version != self.key_version:
            raise ValueError("credential key version is not available")
        aad = self._aad(credential_id, adapter_type, credential_version)
        plaintext = self._aead.decrypt(encrypted.nonce, encrypted.ciphertext, aad)
        return SecretStr(plaintext.decode("utf-8"))

    @staticmethod
    def _aad(credential_id: UUID, adapter_type: str, credential_version: int) -> bytes:
        return (
            f"deep-research:credential:v1:{credential_id}:{adapter_type}:{credential_version}"
        ).encode()


def load_or_create_master_key(settings: Settings) -> bytes:
    """Load a configured key or create a durable development-only key file."""

    configured = settings.secret_master_key_base64
    if configured is not None:
        return _decode_key(configured.get_secret_value())

    if settings.app_env.lower() not in {"development", "test"}:
        raise ValueError("SECRET_MASTER_KEY_BASE64 is required outside development")

    path = settings.secret_master_key_file
    try:
        return _decode_key(path.read_text(encoding="ascii").strip())
    except FileNotFoundError:
        return _create_development_key(path)


def _create_development_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(SecretCipher.KEY_BYTES)
    encoded = base64.b64encode(key)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _decode_key(path.read_text(encoding="ascii").strip())
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
    return key


def _decode_key(value: str) -> bytes:
    try:
        key = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError("secret master key must be valid Base64") from exc
    if len(key) != SecretCipher.KEY_BYTES:
        raise ValueError("secret master key must decode to exactly 32 bytes")
    return key
