from app.domain.source_policy import normalize_source_url, source_owner_key


def test_normalize_source_url_collapses_transport_tracking_and_order() -> None:
    assert normalize_source_url(
        "HTTP://Example.com:80/path/?utm_source=x&b=2&a=1#section"
    ) == "https://example.com/path?a=1&b=2"


def test_normalize_source_url_does_not_merge_non_default_https_port() -> None:
    assert normalize_source_url("https://example.com:80/path") == "https://example.com:80/path"


def test_normalize_source_url_is_stable_for_trailing_root_slash() -> None:
    assert normalize_source_url("https://example.com") == "https://example.com/"


def test_source_owner_preserves_registrable_domain_for_chinese_public_suffix() -> None:
    assert source_owner_key("https://www.aas.net.cn/article/1") == "aas.net.cn"
