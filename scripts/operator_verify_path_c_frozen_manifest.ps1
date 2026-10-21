# Read-only Attempt 006 manifest verification. Never starts Compose services.
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$docker = (Get-Command docker -ErrorAction Stop).Source

# Compose config is local parsing only; do not print the expanded configuration.
$configLines = @(& $docker compose config --format json)
if ($LASTEXITCODE -ne 0) { throw 'COMPOSE_CONFIG_FAILED' }
$config = ($configLines -join "`n") | ConvertFrom-Json
$artifactMounts = @($config.services.api.volumes | Where-Object {
    $_.type -eq 'volume' -and $_.target -eq '/app/artifacts'
})
if ($artifactMounts.Count -ne 1) { throw 'CANONICAL_ARTIFACT_VOLUME_AMBIGUOUS_OR_MISSING' }
$volumeProperty = $config.volumes.PSObject.Properties[[string]$artifactMounts[0].source]
if ($null -eq $volumeProperty -or -not $volumeProperty.Value.name) {
    throw 'CANONICAL_ARTIFACT_VOLUME_NAME_UNRESOLVED'
}
$volumeName = [string]$volumeProperty.Value.name
$image = [string]$config.services.api.image
if (-not $image) { throw 'CANONICAL_API_IMAGE_MISSING' }

# Exact read-only source mounts: scripts, frozen schema, and API modules needed
# to validate the generation implementation fingerprint. Durable volume is RO.
$scriptRoot = Join-Path $repoRoot 'scripts'
$schemaRoot = Join-Path $repoRoot 'artifacts/v2_1r_d4r_a51'
$apiRoot = Join-Path $repoRoot 'apps/api'
foreach ($path in @($scriptRoot, $schemaRoot, $apiRoot)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw 'VERIFIER_SOURCE_MOUNT_MISSING'
    }
}

& $docker image inspect $image --format '{{.Id}}' *> $null
if ($LASTEXITCODE -ne 0) { throw 'LOCAL_API_IMAGE_UNAVAILABLE_NO_PULL' }

$dockerArgs = @(
    'run', '--rm', '--pull=never', '--network=none', '--read-only',
    '--mount', "type=volume,src=$volumeName,dst=/app/artifacts,readonly",
    '--mount', "type=bind,src=$scriptRoot,dst=/opt/scripts,readonly",
    '--mount', "type=bind,src=$schemaRoot,dst=/opt/artifacts/v2_1r_d4r_a51,readonly",
    '--mount', "type=bind,src=$apiRoot,dst=/app/apps/api,readonly",
    '--workdir', '/opt',
    '-e', 'PYTHONPATH=/app/apps/api:/opt',
    '-e', 'PYTHONDONTWRITEBYTECODE=1',
    '--entrypoint', 'python', $image,
    '-m', 'scripts.prospective_acquisition.verify_frozen_manifest',
    '--artifact-root', '/app/artifacts',
    '--schema-path', '/opt/artifacts/v2_1r_d4r_a51/task_manifest_schema.json',
    '--checkout-root', '/opt',
    '--api-root', '/app/apps/api'
)
& $docker @dockerArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
