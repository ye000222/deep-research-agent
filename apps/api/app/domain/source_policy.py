"""Shared policy for stable, attributable Web research candidates."""

from __future__ import annotations

from collections.abc import Collection
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

LOW_VALUE_SOURCE_DOMAINS = frozenset(
    {
        "11467.com",
        "accounts.google.com",
        "ai.so.com",
        "baike.baidu.com",
        "bilibili.com",
        "b2b168.com",
        "csdn.net",
        "docin.com",
        "douyin.com",
        "mail.google.com",
        "max.book118.com",
        "renrendoc.com",
        "smzdm.com",
        "wenku.baidu.com",
        "wenku.so.com",
        "youtu.be",
        "youtube.com",
        "zhihu.com",
    }
)

_TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
    }
)


def normalize_source_url(url: str) -> str:
    """Return a stable identity URL for page-level deduplication.

    HTTP and HTTPS identify the same public page for research-budget purposes.
    Fragments and common tracking parameters likewise do not create a new
    source. Query parameters that may select real content are retained and
    sorted so provider-specific ordering cannot bypass the identity check.
    """

    value = url.strip()
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if not hostname:
        return value.casefold()
    original_scheme = parsed.scheme.casefold()
    scheme = "https" if original_scheme in {"http", "https"} else original_scheme
    try:
        port = parsed.port
    except ValueError:
        return value.casefold()
    # Only the scheme's own default port is interchangeable. Treating
    # ``https://host:80`` as ``https://host`` could merge two endpoints.
    default_port = (original_scheme == "https" and port == 443) or (
        original_scheme == "http" and port == 80
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def matches_domain(hostname: str, domains: Collection[str]) -> bool:
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def is_stable_read_url(url: str) -> bool:
    """Return whether a URL can reasonably yield directly attributable text."""

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname or matches_domain(hostname, LOW_VALUE_SOURCE_DOMAINS):
        return False
    path = parsed.path.lower().rstrip("/")
    if path.endswith(("/login", "/signin", "/sign-in")):
        return False
    return not (matches_domain(hostname, {"google.com"}) and path == "/search")


def source_owner_key(url: str) -> str:
    """Return a publisher-level owner key, not a URL/subdomain key.

    Independent-source checks must not treat ``docs.example.com`` and
    ``www.example.com`` as two publishers.  This is intentionally a small,
    deterministic registrable-domain approximation; it avoids adding a DNS
    dependency to the evidence path while handling the common multi-label
    public suffixes used by the supported search providers.
    """

    hostname = (urlsplit(url).hostname or "unknown").lower().rstrip(".")
    if hostname in {"", "unknown"}:
        return "unknown"
    labels = [label for label in hostname.split(".") if label]
    if len(labels) <= 2:
        return hostname[:255]
    multi_label_suffixes = {
        "ac.cn",
        "com.au",
        "com.br",
        "com.cn",
        "co.jp",
        "co.uk",
        "edu.cn",
        "gov.cn",
        "net.cn",
        "org.cn",
    }
    suffix = ".".join(labels[-2:])
    keep = 3 if suffix in multi_label_suffixes else 2
    return ".".join(labels[-keep:])[:255]
