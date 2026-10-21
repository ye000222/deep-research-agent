[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'prospective_acquisition\host_command_diagnostics.ps1')
$composeFile = Join-Path $repoRoot 'docker-compose.yml'
$schemaRoot = Join-Path $repoRoot 'artifacts\v2_1r_d4r_a51'
$runId = [Guid]::NewGuid().ToString('N')
$networkName = "d4r-pa-attest-net-$runId"
$postgresName = "d4r-pa-attest-pg-$runId"
$createdNetwork = $false
$createdPostgres = $false
$cleanupFailed = $false
$result = $null
$lastCommandDiagnosticPath = $null
$priorRunGate = $env:RUN_PROSPECTIVE_LANE_HOST_ATTESTATION
$priorDisposableGate = $env:PROSPECTIVE_LANE_DISPOSABLE_DB
$priorDatabaseUrl = $env:TEST_DATABASE_URL
$priorNoBytecode = $env:PYTHONDONTWRITEBYTECODE

function ConvertTo-NativeArgument([string]$Value) {
    if ($Value.Length -eq 0) { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append([char]34)
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]92) { $slashes++; continue }
        if ($character -eq [char]34) {
            [void]$builder.Append(('\' * (2 * $slashes + 1)))
            [void]$builder.Append([char]34)
        } else {
            if ($slashes -gt 0) { [void]$builder.Append(('\' * $slashes)) }
            [void]$builder.Append($character)
        }
        $slashes = 0
    }
    if ($slashes -gt 0) { [void]$builder.Append(('\' * (2 * $slashes))) }
    [void]$builder.Append([char]34)
    return $builder.ToString()
}

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [int]$TimeoutMs = 120000) {
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $FilePath
    $start.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw 'NATIVE_PROCESS_START_FAILED' }
    } catch {
        $process.Dispose()
        throw
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit($TimeoutMs)
    if (-not $completed) {
        try { $process.Kill() } catch { }
        $process.WaitForExit()
        $process.Dispose()
        throw 'NATIVE_COMMAND_TIMEOUT'
    }
    $output = [pscustomobject]@{
        ExitCode = $process.ExitCode
        Stdout = $stdoutTask.Result
        Stderr = $stderrTask.Result
    }
    $process.Dispose()
    return $output
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments, [string]$FailureCode, [int]$TimeoutMs = 120000) {
    try {
        $invocation = Invoke-Native $FilePath $Arguments $TimeoutMs
    } catch {
        if ($FailureCode -eq 'API_IMAGE_ATTESTATION_COMMAND_FAILED') {
            $diagnostic = New-HostCommandFailureDiagnostic `
                -RunId $runId -FailureGate $FailureCode -Executable $FilePath -Arguments $Arguments `
                -ExitCode $null -InvocationException $_.Exception.Message -Stdout $null -Stderr $null `
                -WorkingDirectory (Get-Location).Path
            $script:lastCommandDiagnosticPath = Write-HostCommandFailureArtifact $repoRoot $diagnostic
        }
        throw $FailureCode
    }
    if ($invocation.ExitCode -ne 0) {
        if ($FailureCode -eq 'API_IMAGE_ATTESTATION_COMMAND_FAILED') {
            $diagnostic = New-HostCommandFailureDiagnostic `
                -RunId $runId -FailureGate $FailureCode -Executable $FilePath -Arguments $Arguments `
                -ExitCode $invocation.ExitCode -InvocationException $null -Stdout $invocation.Stdout `
                -Stderr $invocation.Stderr -WorkingDirectory (Get-Location).Path
            $script:lastCommandDiagnosticPath = Write-HostCommandFailureArtifact $repoRoot $diagnostic
        }
        throw $FailureCode
    }
    return $invocation.Stdout
}

function Get-FreeLoopbackPort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try { return [int]$listener.LocalEndpoint.Port }
    finally { $listener.Stop() }
}

function Get-ApiMount([object]$Service) {
    foreach ($mount in @($Service.volumes)) {
        if ($mount.target -eq '/app/artifacts' -and $mount.type -eq 'volume') { return $mount }
    }
    throw 'CANONICAL_ARTIFACT_VOLUME_NOT_DECLARED'
}

function Invoke-ApiImage([string]$Image, [string]$Network, [string]$DatabaseUrl, [string[]]$Command) {
    $arguments = @(
        'run', '--rm', '--pull=never', '--network', $Network,
        '--workdir', '/checkout',
        '--volume', "${repoRoot}\apps\api:/checkout/apps/api:ro",
        '--volume', "${repoRoot}\scripts:/checkout/scripts:ro",
        '--volume', "${repoRoot}\alembic.ini:/checkout/alembic.ini:ro",
        '--volume', "${canonicalVolume}:/checkout/artifacts:ro",
        '--volume', "${schemaRoot}:/opt/artifacts/v2_1r_d4r_a51:ro",
        '--env', 'PYTHONPATH=/checkout/apps/api:/checkout',
        '--env', 'APP_ENV=test',
        '--env', "DATABASE_URL=$DatabaseUrl",
        '--env', 'ARTIFACT_ROOT=/checkout/artifacts',
        $Image
    ) + $Command
    return Invoke-Checked $dockerPath $arguments 'API_IMAGE_ATTESTATION_COMMAND_FAILED' 300000
}

try {
    if (-not (Test-Path -LiteralPath $composeFile)) { throw 'COMPOSE_FILE_MISSING' }
    if (-not (Test-Path -LiteralPath $schemaRoot -PathType Container)) { throw 'FROZEN_TASK_SCHEMA_SOURCE_MISSING' }
    $dockerCommand = Get-Command docker -ErrorAction Stop
    $dockerPath = $dockerCommand.Source

    $pythonCandidates = @(
        (Join-Path $repoRoot '.venv\Scripts\python.exe'),
        (Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
    $pythonPath = $null
    foreach ($candidate in $pythonCandidates) {
        $probe = Invoke-Native $candidate @('-c', 'import pytest, sqlalchemy, psycopg, fastapi, alembic') 20000
        if ($probe.ExitCode -eq 0) { $pythonPath = $candidate; break }
    }
    if (-not $pythonPath) { throw 'HOST_PYTHON_TEST_DEPENDENCIES_UNAVAILABLE' }

    $composeOutput = Invoke-Checked $dockerPath @('compose', '--project-directory', $repoRoot, '-f', $composeFile, 'config', '--format', 'json') 'COMPOSE_CONFIG_UNAVAILABLE'
    try { $compose = $composeOutput | ConvertFrom-Json -ErrorAction Stop }
    catch { throw 'COMPOSE_CONFIG_INVALID' }
    $apiService = $compose.services.api
    $postgresService = $compose.services.postgres
    if (-not $apiService.image -or -not $postgresService.image) { throw 'COMPOSE_IMAGE_IDENTITY_MISSING' }
    $apiMount = Get-ApiMount $apiService
    $volumeDefinition = $compose.volumes.($apiMount.source)
    $canonicalVolume = if ($volumeDefinition.name) { [string]$volumeDefinition.name } else { "$($compose.name)_$($apiMount.source)" }

    [void](Invoke-Checked $dockerPath @('image', 'inspect', '--format', '{{.Id}}', [string]$apiService.image) 'API_IMAGE_NOT_PRESENT_LOCALLY')
    [void](Invoke-Checked $dockerPath @('image', 'inspect', '--format', '{{.Id}}', [string]$postgresService.image) 'POSTGRES_IMAGE_NOT_PRESENT_LOCALLY')
    $volumeInfoText = Invoke-Checked $dockerPath @('volume', 'inspect', $canonicalVolume) 'CANONICAL_ARTIFACT_VOLUME_NOT_PRESENT'
    $volumeInfo = @($volumeInfoText | ConvertFrom-Json)
    if ($volumeInfo.Count -ne 1 -or $volumeInfo[0].Name -ne $canonicalVolume) { throw 'CANONICAL_ARTIFACT_VOLUME_IDENTITY_MISMATCH' }
    $volumeLabels = $volumeInfo[0].Labels
    if (
        $volumeLabels.'com.docker.compose.project' -ne [string]$compose.name -or
        $volumeLabels.'com.docker.compose.volume' -ne [string]$apiMount.source
    ) { throw 'CANONICAL_ARTIFACT_VOLUME_LABEL_MISMATCH' }

    $port = Get-FreeLoopbackPort
    $randomBytes = New-Object byte[] 24
    $randomGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $randomGenerator.GetBytes($randomBytes) }
    finally { $randomGenerator.Dispose() }
    $password = [BitConverter]::ToString($randomBytes).Replace('-', '').ToLowerInvariant()
    $hostDatabaseUrl = "postgresql+psycopg://d4r_lane:$password@127.0.0.1:$port/d4r_lane_attestation"
    $containerDatabaseUrl = 'postgresql+psycopg://d4r_lane:' + $password + '@postgres:5432/d4r_lane_attestation'

    [void](Invoke-Checked $dockerPath @('network', 'create', '--internal', $networkName) 'DISPOSABLE_NETWORK_CREATE_FAILED')
    $createdNetwork = $true
    $createdPostgres = $true
    [void](Invoke-Checked $dockerPath @(
        'run', '--pull=never', '--detach', '--name', $postgresName,
        '--network', $networkName, '--network-alias', 'postgres',
        '--publish', "127.0.0.1:${port}:5432",
        '--env', 'POSTGRES_USER=d4r_lane',
        '--env', "POSTGRES_PASSWORD=$password",
        '--env', 'POSTGRES_DB=d4r_lane_attestation',
        [string]$postgresService.image
    ) 'DISPOSABLE_POSTGRES_START_FAILED')

    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $probe = Invoke-Native $dockerPath @('exec', '--env', "PGPASSWORD=$password", $postgresName, 'pg_isready', '-U', 'd4r_lane', '-d', 'd4r_lane_attestation') 10000
        if ($probe.ExitCode -eq 0) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'DISPOSABLE_POSTGRES_NOT_READY' }

    [void](Invoke-ApiImage ([string]$apiService.image) $networkName $containerDatabaseUrl @('python', '-m', 'alembic', '-c', 'alembic.ini', 'upgrade', 'head'))
    $currentRevision = (Invoke-ApiImage ([string]$apiService.image) $networkName $containerDatabaseUrl @('python', '-m', 'alembic', '-c', 'alembic.ini', 'current')).Trim()
    if ($currentRevision -notmatch '20261013_0026') { throw 'EXPECTED_ALEMBIC_HEAD_NOT_REACHED' }

    $manifestText = Invoke-ApiImage ([string]$apiService.image) $networkName $containerDatabaseUrl @(
        'python', '/checkout/scripts/prospective_acquisition/host_attestation.py',
        '--artifact-root', '/checkout/artifacts',
        '--checkout-root', '/checkout',
        '--api-root', '/checkout/apps/api',
        '--schema-path', '/opt/artifacts/v2_1r_d4r_a51/task_manifest_schema.json'
    )
    $manifestLine = @($manifestText -split "`r?`n" | Where-Object { $_.Trim().StartsWith('{') } | Select-Object -Last 1)
    if ($manifestLine.Count -ne 1) { throw 'MANIFEST_ATTESTATION_OUTPUT_MISSING' }
    $manifestAttestation = $manifestLine[0] | ConvertFrom-Json -ErrorAction Stop
    if ($manifestAttestation.status -ne 'PASS') { throw ([string]$manifestAttestation.safe_error_code) }

    $env:RUN_PROSPECTIVE_LANE_HOST_ATTESTATION = '1'
    $env:PROSPECTIVE_LANE_DISPOSABLE_DB = '1'
    $env:TEST_DATABASE_URL = $hostDatabaseUrl
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $readinessScript = Join-Path $repoRoot 'scripts\prospective_acquisition\host_postgres_readiness.py'
    $readiness = Invoke-Native $pythonPath @($readinessScript) 30000
    $readinessLine = @(
        $readiness.Stdout -split "`r?`n" |
            Where-Object { $_.Trim().StartsWith('{') } |
            Select-Object -Last 1
    )
    $hostPostgresReadiness = $null
    if ($readinessLine.Count -eq 1) {
        try { $hostPostgresReadiness = $readinessLine[0] | ConvertFrom-Json -ErrorAction Stop }
        catch { $hostPostgresReadiness = $null }
    }
    if (
        $readiness.ExitCode -ne 0 -or
        $null -eq $hostPostgresReadiness -or
        $hostPostgresReadiness.status -ne 'HOST_POSTGRES_READINESS_PASS' -or
        $hostPostgresReadiness.host -ne '127.0.0.1' -or
        $hostPostgresReadiness.port -ne $port -or
        $hostPostgresReadiness.database -ne 'd4r_lane_attestation'
    ) {
        $safeReadiness = if ($null -ne $hostPostgresReadiness) {
            $hostPostgresReadiness
        } else {
            [ordered]@{
                status = 'HOST_POSTGRES_ENDPOINT_UNREACHABLE'
                host = '127.0.0.1'
                port = $port
                database = 'd4r_lane_attestation'
                attempt_count = 0
                elapsed_ms = $null
                dbapi_exception_type = $null
                sqlstate = $null
                root_cause_category = 'READINESS_PROBE_OUTPUT_INVALID'
            }
        }
        $script:lastCommandDiagnosticPath = Write-HostPostgresReadinessFailureArtifact `
            $repoRoot $runId $safeReadiness
        throw 'HOST_POSTGRES_ENDPOINT_UNREACHABLE'
    }
    $pytestArguments = @(
        '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/integration/test_prospective_acquisition_lane_host.py',
        'tests/unit/test_prospective_acquisition_execution_lane.py',
        'tests/unit/test_prospective_acquisition_control.py',
        'tests/unit/test_bounded_prospective_acquisition_profile.py'
    )
    $pytest = Invoke-Native $pythonPath $pytestArguments 600000
    if ($pytest.ExitCode -ne 0) {
        $diagnostic = New-TargetedHostTestFailureDiagnostic `
            -RunId $runId -Executable $pythonPath -Arguments $pytestArguments `
            -ExitCode $pytest.ExitCode -Stdout $pytest.Stdout -Stderr $pytest.Stderr `
            -WorkingDirectory $repoRoot
        $script:lastCommandDiagnosticPath = Write-TargetedHostTestFailureArtifact $repoRoot $diagnostic
        throw 'TARGETED_HOST_INTEGRATION_TESTS_FAILED'
    }

    $result = [ordered]@{
        status = 'PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS'
        host_postgres_readiness = $hostPostgresReadiness
        alembic_head = '20261013_0026'
        run_kind_schema_valid = $true
        normal_run_outbox_valid = $true
        acquisition_run_outbox_count = 0
        dispatcher_isolation_valid = $true
        worker_isolation_valid = $true
        reconciler_isolation_valid = $true
        internal_get_valid = $true
        ordinary_list_filter_valid = $true
        manifest_sha_valid = $true
        task_count = 40
        task_identity_count = 40
        task_identity_unique = $true
        task_identity_set_sha256 = $manifestAttestation.task_identity_set_sha256
        fingerprint_contract_version = $manifestAttestation.fingerprint_contract_version
        runtime_source_file_count = $manifestAttestation.runtime_source_file_count
        runtime_source_sha256 = $manifestAttestation.runtime_source_sha256
        current_acquisition_runtime_fingerprint = $manifestAttestation.current_acquisition_runtime_fingerprint
        ledger_valid = $true
        receipt_passthrough_valid = $true
        preflight_zero_effect_valid = $true
        provider = 0
        search = 0
        reader = 0
        external_http = 0
        holdout = 0
    }
} catch {
    $message = [string]$_.Exception.Message
    if ($message -match '^[A-Z0-9_]+$') { $safeFailure = $message } else { $safeFailure = 'HOST_ATTESTATION_FAILED' }
    $result = [ordered]@{
        status = 'PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_BLOCKED'
        first_failed_gate = $safeFailure
        provider = 0
        search = 0
        reader = 0
        external_http = 0
        holdout = 0
        command_failure_diagnostic = $script:lastCommandDiagnosticPath
    }
    Write-Output ($result | ConvertTo-Json -Compress -Depth 8)
    exit 2
} finally {
    if ($null -eq $priorRunGate) { Remove-Item Env:RUN_PROSPECTIVE_LANE_HOST_ATTESTATION -ErrorAction SilentlyContinue } else { $env:RUN_PROSPECTIVE_LANE_HOST_ATTESTATION = $priorRunGate }
    if ($null -eq $priorDisposableGate) { Remove-Item Env:PROSPECTIVE_LANE_DISPOSABLE_DB -ErrorAction SilentlyContinue } else { $env:PROSPECTIVE_LANE_DISPOSABLE_DB = $priorDisposableGate }
    if ($null -eq $priorDatabaseUrl) { Remove-Item Env:TEST_DATABASE_URL -ErrorAction SilentlyContinue } else { $env:TEST_DATABASE_URL = $priorDatabaseUrl }
    if ($null -eq $priorNoBytecode) { Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue } else { $env:PYTHONDONTWRITEBYTECODE = $priorNoBytecode }
    if ($createdPostgres) {
        $cleanup = Invoke-Native $dockerPath @('rm', '--force', $postgresName) 30000
        if ($cleanup.ExitCode -ne 0) { $cleanupFailed = $true; Write-Warning 'DISPOSABLE_POSTGRES_CLEANUP_FAILED' }
    }
    if ($createdNetwork) {
        $cleanup = Invoke-Native $dockerPath @('network', 'rm', $networkName) 30000
        if ($cleanup.ExitCode -ne 0) { $cleanupFailed = $true; Write-Warning 'DISPOSABLE_NETWORK_CLEANUP_FAILED' }
    }
}

if ($cleanupFailed -and $result.status -eq 'PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS') {
    $result.status = 'PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_BLOCKED'
    $result.first_failed_gate = 'DISPOSABLE_RESOURCE_CLEANUP_FAILED'
}
if ($result.status -eq 'PROSPECTIVE_ACQUISITION_LANE_HOST_ATTESTATION_PASS') {
    $attestationDirectory = Join-Path $repoRoot 'artifacts\v2_1r_d4r_execution_lane_host'
    [void](New-Item -ItemType Directory -Path $attestationDirectory -Force)
    $attestationPath = Join-Path $attestationDirectory "host_attestation_$runId.json"
    $result['attestation_artifact'] = "artifacts/v2_1r_d4r_execution_lane_host/host_attestation_$runId.json"
    $json = $result | ConvertTo-Json -Compress -Depth 16
    [System.IO.File]::WriteAllText(
        $attestationPath,
        $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}
Write-Output ($result | ConvertTo-Json -Compress -Depth 16)
