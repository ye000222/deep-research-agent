from app.domain.identifiers import uuid7


def test_uuid7_sets_version_and_rfc_variant() -> None:
    generated = uuid7()

    assert generated.version == 7
    assert generated.variant == "specified in RFC 4122"
