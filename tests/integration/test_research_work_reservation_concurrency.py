"""Opt-in PostgreSQL validation for durable WorkAttempt ownership.

The shared integration harness hard-restricts writes to the explicitly
configured localhost:55432/d4ra6_test database and cleans only its own run and
profile rows. This module never falls back to DATABASE_URL.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.integration.test_prospective_acquisition_receipts import (
    _run_postgres_concurrent_reservation_and_execution_ownership_fencing,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1"
    or not os.getenv("TEST_DATABASE_URL"),
    reason=(
        "requires RUN_POSTGRES_INTEGRATION=1 and explicit TEST_DATABASE_URL; "
        "never falls back to DATABASE_URL"
    ),
)


def test_postgres_concurrent_reservation_and_execution_ownership_fencing(
    tmp_path: Path,
) -> None:
    _run_postgres_concurrent_reservation_and_execution_ownership_fencing(tmp_path)
