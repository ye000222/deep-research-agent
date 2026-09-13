from uuid import uuid4

import pytest
from app.security.secrets import SecretCipher, SecretDecryptionError
from pydantic import SecretStr


def test_secret_cipher_round_trip_and_metadata() -> None:
    cipher = SecretCipher(b"k" * 32)
    credential_id = uuid4()

    encrypted = cipher.encrypt(
        SecretStr("sk-example-123456"),
        credential_id=credential_id,
        adapter_type="openai_responses",
        credential_version=1,
    )
    decrypted = cipher.decrypt(
        encrypted,
        credential_id=credential_id,
        adapter_type="openai_responses",
        credential_version=1,
    )

    assert decrypted.get_secret_value() == "sk-example-123456"
    assert encrypted.last_four == "3456"
    assert len(encrypted.nonce) == 12
    assert b"sk-example" not in encrypted.ciphertext


def test_secret_cipher_rejects_wrong_aad() -> None:
    cipher = SecretCipher(b"k" * 32)
    encrypted = cipher.encrypt(
        SecretStr("secret-value"),
        credential_id=uuid4(),
        adapter_type="openai_responses",
        credential_version=1,
    )

    with pytest.raises(SecretDecryptionError) as raised:
        cipher.decrypt(
            encrypted,
            credential_id=uuid4(),
            adapter_type="openai_responses",
            credential_version=1,
        )
    assert raised.value.detail_code == "CREDENTIAL_AUTHENTICATION_FAILED"


def test_secret_cipher_classifies_unavailable_key_version() -> None:
    credential_id = uuid4()
    old_cipher = SecretCipher(b"k" * 32, key_version=1)
    encrypted = old_cipher.encrypt(
        SecretStr("secret-value"),
        credential_id=credential_id,
        adapter_type="openai_responses",
        credential_version=1,
    )

    rotated_cipher = SecretCipher(b"n" * 32, key_version=2)
    with pytest.raises(SecretDecryptionError) as raised:
        rotated_cipher.decrypt(
            encrypted,
            credential_id=credential_id,
            adapter_type="openai_responses",
            credential_version=1,
        )

    assert raised.value.detail_code == "KEY_VERSION_UNAVAILABLE"
