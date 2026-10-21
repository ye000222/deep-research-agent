"""Exact Windows launch boundary, harmless Python fixtures; never Docker."""

from __future__ import annotations

import ast
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPERATOR = ROOT / "scripts/operator_verify_canonical_runtime.ps1"
PROBE = ROOT / "scripts/prospective_acquisition/canonical_runtime_probe.py"


def test_extracted_probe_retains_schema_and_identity_gates() -> None:
    source = PROBE.read_text(encoding="utf-8")
    ast.parse(source)
    for gate in (
        "CURRENT_CHECKOUT_IDENTITY", "IMPORT_PROBE", "SCHEMA_RUNTIME_PROBE",
        "DB_READ_PROBE", "FROZEN_CONTEXT", "FINAL_ATTESTATION_READY",
    ):
        assert gate in source
    assert 'probe_schema_runtime(pathlib.Path("/opt/artifacts/' in source
    schema_position = source.index('_stage = "SCHEMA_RUNTIME_PROBE"')
    assert schema_position < source.index("async def _resolve_provider")
    shell = OPERATOR.read_text(encoding="utf-8")
    assert "'scripts/prospective_acquisition/canonical_runtime_probe.py'" in shell
    assert "'/opt/scripts/prospective_acquisition/canonical_runtime_probe.py'" in shell
    assert "'--entrypoint','python','api',$probeArgument" in shell
    assert "$probeArgument = $probe" not in shell
    assert "invocation_failure = $probeInvocationFailure" in shell


def test_probe_safe_import_missing_identity_fails_before_external_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPERATOR_EXPECTED_HASHES_B64", "e30=")  # synthetic empty hash map
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit) as failed:
        runpy.run_path(str(PROBE), run_name="__main__")
    assert failed.value.code == 21
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "FAILED"
    assert report["safe_message"] == "CURRENT_CHECKOUT_IDENTITY_FAILED"
    assert report["provider_generation_calls"] == report["provider_http_count"] == 0
    assert report["db_writes"] == 0
    assert "SCHEMA_RUNTIME_PROBE_STARTED" not in captured.err


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific CreateProcess boundary")
def test_same_native_launcher_overflow_and_short_script_path(tmp_path: Path) -> None:
    fixture = tmp_path / "harmless probe with spaces.py"
    fixture.write_text('print("SYNTHETIC_LOCAL_PROBE_OK")\n', encoding="utf-8")
    shell = tmp_path / "local launch proof.ps1"
    shell.write_text(r'''
param([string]$OperatorPath, [string]$PythonPath, [string]$FixturePath, [string]$RepoRoot)
$ErrorActionPreference = 'Stop'
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $OperatorPath, [ref]$null, [ref]$null)
foreach ($name in @('ConvertTo-WindowsArgument','Invoke-Native','Get-NativeInvocationFailure')) {
    $fn = $ast.Find({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    . ([scriptblock]::Create($fn.Extent.Text))
}
$longFailure = $null
try {
    $unexpected = Invoke-Native $PythonPath @('-c', ('pass #' + ('x' * 34000)))
    throw 'EXPECTED_LONG_ARGUMENT_START_FAILURE'
} catch { $longFailure = Get-NativeInvocationFailure $_.Exception }
if ($longFailure.native_error_code -ne 206 -or $longFailure.phase -ne 'PROCESS_START') {
    throw 'WRONG_FAILURE_BOUNDARY'
}
$short = Invoke-Native $PythonPath @($FixturePath)
if ($short.ExitCode -ne 0 -or $short.Stdout.Trim() -ne 'SYNTHETIC_LOCAL_PROBE_OK') {
    throw 'SHORT_FILE_INVOCATION_FAILED'
}
$repoRoot = $RepoRoot
$ephemeralName = 'synthetic-no-container-created'
$expectedExecutionContextId = '8862f286-e707-461e-876f-ec57207802e8'
$expectedHashB64 = 'x' * 5000
$probeArgument = '/opt/scripts/prospective_acquisition/canonical_runtime_probe.py'
$argsAst = $ast.Find({ param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $n.Left.Extent.Text -eq '$runArgs'
}, $true)
$argsOnly = & ([scriptblock]::Create($argsAst.Right.Extent.Text))
$length = (($argsOnly | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' ').Length
if ($length -ge 8192 -or $argsOnly -contains '-c') { throw 'LARGE_INLINE_SOURCE_REGRESSION' }
[pscustomobject]@{
    native_code=$longFailure.native_error_code; phase=$longFailure.phase
    short_exit=$short.ExitCode; new_argument_characters=$length
    docker_calls=0; provider_calls=0
} | ConvertTo-Json -Compress
''', encoding="utf-8")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(shell),
         "-OperatorPath", str(OPERATOR), "-PythonPath", sys.executable,
         "-FixturePath", str(fixture), "-RepoRoot", str(ROOT)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    safe = json.loads(result.stdout)
    assert safe["native_code"] == 206 and safe["phase"] == "PROCESS_START"
    assert safe["short_exit"] == 0
    assert safe["new_argument_characters"] < 8192
    assert safe["docker_calls"] == safe["provider_calls"] == 0
