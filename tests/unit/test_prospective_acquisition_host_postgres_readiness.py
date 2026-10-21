from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prospective_acquisition.host_postgres_readiness import (
    MAX_ATTEMPTS,
    run_readiness_probe,
)

ROOT = Path(__file__).resolve().parents[2]


def _disposable_url(port: int = 49123) -> str:
    return (
        "postgresql+psycopg://d4r_lane:synthetic-password@"
        f"127.0.0.1:{port}/d4r_lane_attestation"
    )


def test_host_readiness_retries_boundedly_then_passes_without_exposing_credentials() -> None:
    attempts = 0
    sleeps: list[float] = []
    clock = iter((1.0, 1.1, 1.2, 1.3, 1.4))

    def connect_probe(_url: str, timeout: int) -> None:
        nonlocal attempts
        attempts += 1
        assert timeout == 2
        if attempts < 3:
            raise TimeoutError("connection timeout expired; password=synthetic-password")

    result = run_readiness_probe(
        _disposable_url(),
        connect_probe=connect_probe,
        sleep=sleeps.append,
        monotonic=lambda: next(clock),
    )

    assert result["status"] == "HOST_POSTGRES_READINESS_PASS"
    assert result["host"] == "127.0.0.1"
    assert result["port"] == 49123
    assert result["database"] == "d4r_lane_attestation"
    assert result["attempt_count"] == 3
    assert sleeps == [0.5, 0.5]
    assert "synthetic-password" not in json.dumps(result)


def test_host_readiness_timeout_fails_closed_at_fixed_attempt_limit() -> None:
    attempts = 0
    sleeps: list[float] = []

    def connect_probe(_url: str, _timeout: int) -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("connection timeout expired")

    result = run_readiness_probe(
        _disposable_url(), connect_probe=connect_probe, sleep=sleeps.append
    )

    assert result["status"] == "HOST_POSTGRES_ENDPOINT_UNREACHABLE"
    assert attempts == MAX_ATTEMPTS
    assert result["attempt_count"] == MAX_ATTEMPTS
    assert result["root_cause_category"] == "CONNECTION_TIMEOUT"
    assert len(sleeps) == MAX_ATTEMPTS - 1
    assert "synthetic-password" not in json.dumps(result)


@pytest.mark.parametrize(
    ("url", "category"),
    [
        (
            "postgresql+psycopg://u:p@postgres:5432/d4r_lane_attestation",
            "ENDPOINT_CONTRACT_INVALID",
        ),
        (
            "postgresql+psycopg://u:p@127.0.0.1:5432/d4r_lane_attestation",
            "ENDPOINT_CONTRACT_INVALID",
        ),
        ("postgresql+psycopg://u:p@127.0.0.1:49123/production", "ENDPOINT_CONTRACT_INVALID"),
    ],
)
def test_host_readiness_rejects_non_disposable_or_container_endpoint(
    url: str, category: str
) -> None:
    result = run_readiness_probe(url)

    assert result["status"] == "HOST_POSTGRES_ENDPOINT_UNREACHABLE"
    assert result["attempt_count"] == 0
    assert result["root_cause_category"] == category
    assert url not in json.dumps(result)


def test_operator_runs_host_readiness_with_pytest_python_and_same_endpoint_before_pytest() -> None:
    script = ROOT / "scripts" / "operator_verify_prospective_acquisition_lane.ps1"
    source = script.read_text(encoding="utf-8")
    readiness = source.split("$readinessScript =", maxsplit=1)[1].split(
        "$pytestArguments =", maxsplit=1
    )[0]

    assert "$env:TEST_DATABASE_URL = $hostDatabaseUrl" in source
    assert "Invoke-Native $pythonPath @($readinessScript)" in readiness
    assert "HOST_POSTGRES_READINESS_PASS" in readiness
    assert "$hostPostgresReadiness.port -ne $port" in readiness
    assert "throw 'HOST_POSTGRES_ENDPOINT_UNREACHABLE'" in readiness
    assert source.index("throw 'HOST_POSTGRES_ENDPOINT_UNREACHABLE'") < source.index(
        "$pytest = Invoke-Native $pythonPath $pytestArguments"
    )
    assert '"@postgres:5432/d4r_lane_attestation"' not in readiness


def test_operator_published_port_and_host_url_use_the_same_dynamic_value() -> None:
    script = ROOT / "scripts" / "operator_verify_prospective_acquisition_lane.ps1"
    source = script.read_text(encoding="utf-8")

    assert "'--publish', \"127.0.0.1:${port}:5432\"" in source
    assert '$hostDatabaseUrl = "postgresql+psycopg://d4r_lane:$password@127.0.0.1:$port/' in source
    assert "$env:TEST_DATABASE_URL = $hostDatabaseUrl" in source


def test_readiness_failure_artifact_contains_only_safe_endpoint_and_error_fields(
    tmp_path: Path,
) -> None:
    import shutil
    import subprocess

    module = ROOT / "scripts" / "prospective_acquisition" / "host_command_diagnostics.ps1"
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    module_path = str(module).replace("'", "''")
    root = str(tmp_path).replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
. '{module_path}'
$readiness = [pscustomobject]@{{
  status='HOST_POSTGRES_ENDPOINT_UNREACHABLE'; host='127.0.0.1'; port=49123;
  database='d4r_lane_attestation'; attempt_count=8; elapsed_ms=12000;
  dbapi_exception_type='psycopg.OperationalError'; sqlstate=$null;
  root_cause_category='CONNECTION_TIMEOUT'; password='do-not-store'
}}
$path = Write-HostPostgresReadinessFailureArtifact '{root}' 'run-abc' $readiness
Write-Output $path
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    artifact_path = Path(result.stdout.strip())
    artifact = artifact_path.read_text(encoding="utf-8")
    assert "CONNECTION_TIMEOUT" in artifact
    assert "127.0.0.1" in artifact
    assert "do-not-store" not in artifact
    assert "password" not in artifact.lower()
    assert "postgresql+psycopg://" not in artifact
