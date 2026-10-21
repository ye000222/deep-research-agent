"""Bounded, read-only readiness probe for the host-side disposable PostgreSQL URL."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

MAX_ATTEMPTS = 8
CONNECT_TIMEOUT_SECONDS = 2
RETRY_INTERVAL_SECONDS = 0.5
EXPECTED_DATABASE = "d4r_lane_attestation"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}


def _endpoint_metadata(database_url: str) -> dict[str, Any]:
    url = make_url(database_url)
    if (
        url.drivername != "postgresql+psycopg"
        or url.host not in LOOPBACK_HOSTS
        or url.port is None
        or url.port in {5432, 55432}
        or url.database != EXPECTED_DATABASE
        or not url.username
        or not url.password
    ):
        raise ValueError("HOST_POSTGRES_ENDPOINT_CONTRACT_INVALID")
    return {
        "host": url.host,
        "port": url.port,
        "database": url.database,
    }


def _classify_exception(exc: BaseException) -> dict[str, str | None]:
    original = getattr(exc, "orig", None) or exc
    dbapi_type = f"{type(original).__module__}.{type(original).__name__}"
    raw_text = str(original).lower()
    sqlstate_value = getattr(original, "sqlstate", None) or getattr(
        original, "pgcode", None
    )
    sqlstate = (
        str(sqlstate_value).upper()
        if isinstance(sqlstate_value, str) and re.fullmatch(r"[0-9A-Z]{5}", sqlstate_value, re.I)
        else None
    )

    if any(
        marker in raw_text
        for marker in (
            "could not translate host name",
            "name or service not known",
            "getaddrinfo failed",
            "nodename nor servname",
        )
    ):
        category = "HOST_NOT_RESOLVED"
    elif "connection refused" in raw_text:
        category = "CONNECTION_REFUSED"
    elif any(
        marker in raw_text
        for marker in (
            "connection timeout",
            "connection timed out",
            "timeout expired",
            "connect timeout",
        )
    ):
        category = "CONNECTION_TIMEOUT"
    elif (
        any(
            marker in raw_text
            for marker in (
                "password authentication failed",
                "authentication failed",
                "no pg_hba.conf entry",
            )
        )
        or sqlstate == "28P01"
    ):
        category = "AUTHENTICATION_FAILED"
    elif ("does not exist" in raw_text and "database" in raw_text) or sqlstate == "3D000":
        category = "DATABASE_NOT_FOUND"
    elif sqlstate == "42703" or type(original).__name__ == "UndefinedColumn":
        category = "UNDEFINED_COLUMN"
    elif sqlstate == "42P01" or type(original).__name__ == "UndefinedTable":
        category = "MISSING_SCHEMA"
    elif "invalid input value for enum" in raw_text or sqlstate == "22P02":
        category = "INVALID_ENUM_VALUE"
    else:
        category = "UNKNOWN_DATABASE_ERROR"
    return {
        "dbapi_exception_type": dbapi_type,
        "sqlstate": sqlstate,
        "root_cause_category": category,
    }


def _connect_and_select_one(database_url: str, timeout_seconds: int) -> None:
    engine = create_engine(
        database_url,
        connect_args={"connect_timeout": timeout_seconds},
        poolclass=NullPool,
    )
    try:
        with engine.connect() as connection:
            if connection.execute(text("SELECT 1")).scalar_one() != 1:
                raise RuntimeError("HOST_POSTGRES_READINESS_QUERY_FAILED")
    finally:
        engine.dispose()


def run_readiness_probe(
    database_url: str,
    *,
    connect_probe: Callable[[str, int], None] = _connect_and_select_one,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Probe only the disposable loopback endpoint with a bounded SELECT 1."""
    started = monotonic()
    try:
        endpoint = _endpoint_metadata(database_url)
    except Exception:
        return {
            "status": "HOST_POSTGRES_ENDPOINT_UNREACHABLE",
            "host": None,
            "port": None,
            "database": None,
            "attempt_count": 0,
            "elapsed_ms": max(0, round((monotonic() - started) * 1000)),
            "dbapi_exception_type": None,
            "root_cause_category": "ENDPOINT_CONTRACT_INVALID",
        }

    last_failure: dict[str, str | None] = {
        "dbapi_exception_type": None,
        "sqlstate": None,
        "root_cause_category": "UNKNOWN_DATABASE_ERROR",
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            connect_probe(database_url, CONNECT_TIMEOUT_SECONDS)
            return {
                "status": "HOST_POSTGRES_READINESS_PASS",
                **endpoint,
                "attempt_count": attempt,
                "elapsed_ms": max(0, round((monotonic() - started) * 1000)),
                "dbapi_exception_type": None,
                "root_cause_category": None,
            }
        except Exception as exc:
            last_failure = _classify_exception(exc)
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_INTERVAL_SECONDS)

    return {
        "status": "HOST_POSTGRES_ENDPOINT_UNREACHABLE",
        **endpoint,
        "attempt_count": MAX_ATTEMPTS,
        "elapsed_ms": max(0, round((monotonic() - started) * 1000)),
        **last_failure,
    }


def main() -> int:
    result = run_readiness_probe(os.environ.get("TEST_DATABASE_URL", ""))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "HOST_POSTGRES_READINESS_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
