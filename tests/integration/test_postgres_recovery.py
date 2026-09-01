import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)


def test_postgres_recovery_smoke_script_contract():
    """Execute the recovery contract against the CI PostgreSQL service."""
    assert os.path.exists("scripts/verify_postgres_recovery.py")
    assert os.getenv("DATABASE_URL")
    completed = subprocess.run(
        [sys.executable, "scripts/verify_postgres_recovery.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
