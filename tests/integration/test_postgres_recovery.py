import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)


def test_postgres_recovery_smoke_script_contract():
    """The Docker smoke script is the executable integration contract."""
    assert os.path.exists("scripts/verify_postgres_recovery.py")
