from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "prospective_acquisition" / "host_command_diagnostics.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def test_host_command_failure_diagnostic_preserves_cause_and_redacts_secrets() -> None:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    module_path = str(MODULE).replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
. '{module_path}'
$diagnostic = New-HostCommandFailureDiagnostic `
  -RunId 'run-123' -FailureGate 'API_IMAGE_ATTESTATION_COMMAND_FAILED' `
  -Executable 'docker.exe' `
  -Arguments @(
    'run', '--pull=never', '--env',
    'DATABASE_URL=postgresql://user:db-secret@postgres/db',
    '--workdir', '/checkout'
  ) `
  -ExitCode 17 -InvocationException $null `
  -Stdout 'safe output' -Stderr 'ModuleNotFoundError: package_missing; password=supersecret' `
  -WorkingDirectory 'D:\\agentProject'
$diagnostic | ConvertTo-Json -Depth 10 -Compress
"""
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    diagnostic = json.loads(result.stdout.strip())
    assert diagnostic["failure_gate"] == "API_IMAGE_ATTESTATION_COMMAND_FAILED"
    assert diagnostic["exit_code"] == 17
    assert "ModuleNotFoundError: package_missing" in diagnostic["stderr"]
    assert "supersecret" not in result.stdout
    assert "db-secret" not in result.stdout
    assert diagnostic["command"]["container_working_directory"] == "/checkout"
    safe_database_url = "DATABASE_URL=postgresql://user:[REDACTED]@postgres/db"
    assert safe_database_url in diagnostic["command"]["environment"]
    assert diagnostic["command"]["host_working_directory"] == r"D:\agentProject"


def test_targeted_test_failure_diagnostic_records_safe_node_and_exit_without_task_text() -> None:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    module_path = str(MODULE).replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
. '{module_path}'
$safeStdout = 'FAILED tests/integration/test_lane.py::test_isolation - AssertionError'
$safeStdout += "`n1 failed, 8 passed"
$diagnostic = New-TargetedHostTestFailureDiagnostic `
  -RunId 'host-run-456' -Executable 'python.exe' `
  -Arguments @('-m','pytest','-q','tests/integration/test_lane.py') `
  -ExitCode 1 `
  -Stdout $safeStdout `
  -Stderr "AssertionError: task text must never persist; password=do-not-leak" `
  -WorkingDirectory 'D:\\agentProject'
$diagnostic | ConvertTo-Json -Depth 10 -Compress
"""
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    diagnostic = json.loads(result.stdout.strip())
    assert diagnostic["diagnostic_version"] == "prospective-acquisition-targeted-test-failure.v1"
    assert diagnostic["exit_code"] == 1
    assert diagnostic["failed_test_nodeid"] == "tests/integration/test_lane.py::test_isolation"
    assert "AssertionError" in diagnostic["safe_stderr"]
    assert "task text" not in result.stdout
    assert "do-not-leak" not in result.stdout
    assert diagnostic["raw_output_preserved"] is False
    assert diagnostic["task_text_preserved"] is False


def test_targeted_test_diagnostic_does_not_claim_a_node_when_summary_is_unparseable() -> None:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    module_path = str(MODULE).replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
. '{module_path}'
$diagnostic = New-TargetedHostTestFailureDiagnostic `
  -RunId 'host-run-789' -Executable 'python.exe' -Arguments @('-m','pytest') `
  -ExitCode 2 -Stdout 'collection failed' -Stderr 'ImportError: unavailable' `
  -WorkingDirectory 'D:\\agentProject'
$diagnostic | ConvertTo-Json -Depth 10 -Compress
"""
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    diagnostic = json.loads(result.stdout.strip())
    assert diagnostic["failed_test_nodeid"] is None
    assert diagnostic["exit_code"] == 2
    assert diagnostic["safe_stdout"] == "[No safe pytest summary lines were parseable]"


def test_operator_writes_targeted_test_artifact_only_for_nonzero_exit() -> None:
    script = ROOT / "scripts" / "operator_verify_prospective_acquisition_lane.ps1"
    source = script.read_text(encoding="utf-8")
    failure_branch = source.split("if ($pytest.ExitCode -ne 0) {", maxsplit=1)[1].split(
        "\n    }", maxsplit=1
    )[0]

    assert "Write-TargetedHostTestFailureArtifact" in failure_branch
    assert "throw 'TARGETED_HOST_INTEGRATION_TESTS_FAILED'" in failure_branch
    pass_result = (
        "$result = [ordered]@{\n"
        "        status = 'PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS'"
    )
    assert source.index("if ($pytest.ExitCode -ne 0) {") < source.index(pass_result)


def test_disposable_database_urls_are_separate_for_container_and_host() -> None:
    operator = ROOT / "scripts" / "operator_verify_prospective_acquisition_lane.ps1"
    integration = ROOT / "tests" / "integration" / "test_prospective_acquisition_lane_host.py"
    source = operator.read_text(encoding="utf-8")
    test_source = integration.read_text(encoding="utf-8")

    assert "$containerDatabaseUrl = 'postgresql+psycopg://d4r_lane:" in source
    assert "+ $password + '@postgres:5432/d4r_lane_attestation'" in source
    assert '$hostDatabaseUrl = "postgresql+psycopg://d4r_lane:' in source
    assert "$password@127.0.0.1:$port/d4r_lane_attestation\"" in source
    assert "'--publish', \"127.0.0.1:${port}:5432\"" in source
    assert "$env:TEST_DATABASE_URL = $hostDatabaseUrl" in source
    assert "@127.0.0.1:5432" not in source
    assert "parsed.hostname not in {\"127.0.0.1\", \"localhost\"}" in test_source
    assert "raw_url = os.environ.get(\"TEST_DATABASE_URL\", \"\")" in test_source
    pytest_wiring = source.split("$pytestArguments = @(", maxsplit=1)[1].split(")", maxsplit=1)[0]
    assert "DATABASE_URL" not in pytest_wiring


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("could not translate host name \"postgres\"", "HOST_NOT_RESOLVED"),
        ("connection refused", "CONNECTION_REFUSED"),
        (
            "password authentication failed for user d4r_lane SQLSTATE=28P01",
            "AUTHENTICATION_FAILED",
        ),
        ("database d4r_lane_attestation does not exist SQLSTATE=3D000", "DATABASE_NOT_FOUND"),
        ("psycopg.errors.UndefinedColumn SQLSTATE=42703", "UNDEFINED_COLUMN"),
        ("invalid input value for enum run_kind", "INVALID_ENUM_VALUE"),
    ],
)
def test_operational_error_classification_is_safe(text: str, expected: str) -> None:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    module_path = str(MODULE).replace("'", "''")
    escaped_text = text.replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
. '{module_path}'
$result = Get-SafeDatabaseFailureMetadata -Text @('{escaped_text}')
$result | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout.strip())
    assert metadata["root_cause_category"] == expected
    assert text not in result.stdout
    assert "password authentication failed" not in result.stdout


def test_operational_error_metadata_exposes_only_safe_dbapi_type_and_sqlstate() -> None:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    module_path = str(MODULE).replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
. '{module_path}'
$failure = 'sqlalchemy.exc.OperationalError (psycopg.errors.UndefinedColumn) '
$failure += 'password=private SQLSTATE=42703 task_text=private'
$result = Get-SafeDatabaseFailureMetadata -Text @($failure)
$result | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout.strip())
    assert metadata == {
        "dbapi_exception_type": "psycopg.errors.UndefinedColumn",
        "sqlstate": "42703",
        "root_cause_category": "UNDEFINED_COLUMN",
    }
    assert "private" not in result.stdout
