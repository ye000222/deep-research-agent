[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$expectedRoot = 'D:\agentProject'
if ($repoRoot.TrimEnd('\') -ine $expectedRoot) {
    throw 'PATH_C_OPERATOR_WRONG_REPOSITORY_ROOT'
}
if ((Get-Location).Path.TrimEnd('\') -ine $expectedRoot) {
    throw 'Run this operator entry from D:\agentProject.'
}

$verifier = Join-Path $repoRoot 'scripts\operator_verify_canonical_runtime.ps1'
if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) {
    throw 'CANONICAL_RUNTIME_VERIFIER_MISSING'
}

# The established host-side verifier checks Docker, the current checkout,
# read-only database access, the frozen ExecutionContext and Provider binding.
# Its nonzero result is terminal; this wrapper never retries it.
$verificationOutput = @(& $verifier 2>&1)
$verificationExit = $LASTEXITCODE
foreach ($line in $verificationOutput) { Write-Output $line }
if ($verificationExit -ne 0) {
    throw "CANONICAL_RUNTIME_ATTESTATION_FAILED_EXIT_$verificationExit"
}

$completionLine = @($verificationOutput | ForEach-Object { [string]$_ } |
    Where-Object { $_ -match '^Canonical runtime attestation complete: ' } |
    Select-Object -Last 1)
if ($completionLine.Count -ne 1) {
    throw 'CANONICAL_RUNTIME_ATTESTATION_PATH_MISSING'
}
$attestationDirectory = $completionLine[0].Substring('Canonical runtime attestation complete: '.Length).Trim()
if (-not (Test-Path -LiteralPath $attestationDirectory -PathType Container)) {
    throw 'CANONICAL_RUNTIME_ATTESTATION_DIRECTORY_MISSING'
}

function Read-RequiredJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw 'CANONICAL_ATTESTATION_ARTIFACT_MISSING' }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -ErrorAction Stop }
    catch { throw 'CANONICAL_ATTESTATION_ARTIFACT_INVALID_JSON' }
}

$runtime = Read-RequiredJson (Join-Path $attestationDirectory 'canonical_runtime_attestation.json')
$provider = Read-RequiredJson (Join-Path $attestationDirectory 'provider_readiness_attestation.json')
$probe = Read-RequiredJson (Join-Path $attestationDirectory 'operator_probe_execution.json')
$dockerHost = Read-RequiredJson (Join-Path $attestationDirectory 'docker_host_attestation.json')
$hashManifest = Read-RequiredJson (Join-Path $attestationDirectory 'hash_manifest.json')
$manifestFiles = $hashManifest.files
if (-not $manifestFiles) { throw 'CANONICAL_ATTESTATION_HASH_MANIFEST_INVALID' }
foreach ($property in $manifestFiles.PSObject.Properties) {
    $manifestPath = if ($property.Name -like 'scripts/*') {
        Join-Path $repoRoot $property.Name
    } else {
        Join-Path $attestationDirectory $property.Name
    }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw 'CANONICAL_ATTESTATION_HASH_TARGET_MISSING'
    }
    $actualHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$property.Value).ToLowerInvariant()) {
        throw 'CANONICAL_ATTESTATION_HASH_MISMATCH'
    }
}
$runtimeState = if ($runtime.PSObject.Properties['status']) { [string]$runtime.status } else { '' }
$providerState = if ($provider.PSObject.Properties['status']) { [string]$provider.status } else { '' }
if ($runtimeState -ne 'PASS' -or $providerState -ne 'RESOLVED' -or $probe.status -ne 'PASS') {
    throw 'CANONICAL_RUNTIME_OR_PROVIDER_ATTESTATION_NOT_PASS'
}
$dockerCommandsPass = (
    $dockerHost.commands.docker_version.exit_code -eq 0 -and
    $dockerHost.commands.docker_context_show.exit_code -eq 0 -and
    $dockerHost.commands.docker_ps.exit_code -eq 0 -and
    $dockerHost.commands.docker_compose_config.exit_code -eq 0 -and
    $dockerHost.ephemeral_container.compose_run_exit_code -eq 0 -and
    $dockerHost.ephemeral_container.removed_by_operator_script -eq $true
)
if (-not $dockerCommandsPass) { throw 'DOCKER_HOST_ATTESTATION_NOT_PASS' }
$expectedContextId = '8862f286-e707-461e-876f-ec57207802e8'
$expectedFingerprint = '32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2'
if ($runtime.execution_context_id -ne $expectedContextId -or
    $provider.execution_context_id -ne $expectedContextId -or
    $provider.fingerprint_contract -ne 'provider-safe-config-fingerprint.v1' -or
    @($provider.profiles).Count -ne 1 -or
    $provider.profiles[0].safe_identity.profile_id -ne '49a9ae5c-b062-4617-805f-dc587b706ce1' -or
    $provider.profiles[0].provider_safe_config_fingerprint -ne $expectedFingerprint) {
    throw 'CANONICAL_EXECUTION_OR_PROVIDER_BINDING_MISMATCH'
}
if ($probe.provider_generation_calls -ne 0 -or $probe.provider_http_count -ne 0 -or
    $probe.db_writes -ne 0 -or $probe.holdout_access -ne 0) {
    throw 'CANONICAL_RUNTIME_SAFETY_ACCOUNTING_INVALID'
}

$docker = Get-Command docker -ErrorAction Stop | Select-Object -First 1
$acquisitionScriptsMount = (Join-Path $repoRoot 'scripts\prospective_acquisition').Replace('\', '/') + ':/opt/scripts/prospective_acquisition:ro'
$scriptsPackageInitMount = (Join-Path $repoRoot 'scripts\__init__.py').Replace('\', '/') + ':/opt/scripts/__init__.py:ro'
$amendmentTestsMount = (Join-Path $repoRoot 'tests\unit').Replace('\', '/') + ':/opt/tests/unit:ro'
$a51Mount = (Join-Path $repoRoot 'artifacts\v2_1r_d4r_a51').Replace('\', '/') + ':/opt/artifacts/v2_1r_d4r_a51:ro'
$amendmentMount = (Join-Path $repoRoot 'artifacts\v2_1r_d4r_path_c_seed_amendment').Replace('\', '/') + ':/opt/artifacts/v2_1r_d4r_path_c_seed_amendment:ro'
$pythonPath = '/app/apps/api:/opt'
$modelHttpPolicyMount = (Join-Path $repoRoot 'apps\api\app\llm\model_http_policy.py').Replace('\', '/') + ':/app/apps/api/app/llm/model_http_policy.py:ro'
$dockerArgs = @(
    'compose', 'run', '--rm', '--no-deps',
    '-v', $acquisitionScriptsMount,
    '-v', $scriptsPackageInitMount,
    '-v', $amendmentTestsMount,
    '-v', $a51Mount,
    '-v', $amendmentMount,
    '-v', $modelHttpPolicyMount,
    '-e', "PYTHONPATH=$pythonPath",
    'api', 'python', '-m', 'scripts.prospective_acquisition.path_c_generation_entry'
)
& $docker.Source @dockerArgs
$generationExit = $LASTEXITCODE
if ($generationExit -ne 0) {
    throw "PATH_C_GENERATION_ENTRY_FAILED_EXIT_$generationExit"
}
