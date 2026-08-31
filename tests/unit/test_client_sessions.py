from app.security.client_sessions import ClientSessionManager
from fastapi import Response


def test_signed_session_cookie_survives_refresh() -> None:
    manager = ClientSessionManager(b"s" * 32, cookie_name="dr_session")
    first = manager.resolve(None)
    response = Response()
    manager.issue_cookie(response, first, secure=False)

    cookie_header = response.headers["set-cookie"]
    token = cookie_header.split("dr_session=", 1)[1].split(";", 1)[0]
    restored = manager.resolve(token)

    assert restored.is_new is False
    assert restored.client_id == first.client_id
    assert restored.owner_hash == first.owner_hash


def test_tampered_session_gets_new_identity() -> None:
    manager = ClientSessionManager(b"s" * 32, cookie_name="dr_session")
    restored = manager.resolve("0" * 32 + ".invalid")

    assert restored.is_new is True
