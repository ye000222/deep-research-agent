[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$composeFile = Join-Path $repoRoot 'docker-compose.yml'
$docker = (Get-Command docker -ErrorAction Stop).Source

$composeText = & $docker compose --project-directory $repoRoot -f $composeFile config --format json
if ($LASTEXITCODE -ne 0) { throw 'CANONICAL_COMPOSE_CONFIG_FAILED' }
$compose = $composeText | ConvertFrom-Json -ErrorAction Stop
$artifactMounts = @($compose.services.api.volumes | Where-Object {
    $_.target -eq '/app/artifacts' -and $_.type -eq 'volume'
})
if ($artifactMounts.Count -ne 1) { throw 'CANONICAL_ARTIFACT_VOLUME_NOT_UNIQUE' }
$sourceName = [string]$artifactMounts[0].source
$volumeDefinition = $compose.volumes.PSObject.Properties[$sourceName].Value
if ($null -eq $volumeDefinition) { throw 'CANONICAL_ARTIFACT_VOLUME_UNDEFINED' }
$volumeName = if ($volumeDefinition.name) { [string]$volumeDefinition.name } else { "$($compose.name)_$sourceName" }
$volumeLabelsJson = & $docker volume inspect --format '{{json .Labels}}' $volumeName
if ($LASTEXITCODE -ne 0) { throw 'CANONICAL_ARTIFACT_VOLUME_NOT_PRESENT' }
$volumeLabels = $volumeLabelsJson | ConvertFrom-Json -ErrorAction Stop
if (
    $volumeLabels.'com.docker.compose.project' -ne [string]$compose.name -or
    $volumeLabels.'com.docker.compose.volume' -ne $sourceName
) { throw 'CANONICAL_ARTIFACT_VOLUME_IDENTITY_MISMATCH' }

$image = [string]$compose.services.api.image
if (-not $image) { throw 'CANONICAL_API_IMAGE_IDENTITY_MISSING' }
$probe = @'
import hashlib, json, pathlib, sys
root = pathlib.Path('/artifacts/v2_1r_d4r_path_c_generation/v2_1r_d4r_prospective_population_001__v2_1r_d4r_prospective_population_001_batch_001/attempt_006')
metadata_path = root / 'generation_metadata.json'
manifest_path = root / 'hash_manifest.json'
state_path = root / 'generation_state.json'
try:
    metadata_bytes = metadata_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    manifest = json.loads(manifest_bytes)
    state = json.loads(state_path.read_bytes())
except Exception:
    raise SystemExit('ATTEMPT_006_FROZEN_METADATA_UNREADABLE')
metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
if (manifest.get('files', {}).get('generation_metadata.json') != metadata_sha
        or state.get('metadata_sha256') != metadata_sha
        or state.get('hash_manifest_sha256') != manifest_sha):
    raise SystemExit('ATTEMPT_006_FROZEN_METADATA_INTEGRITY_MISMATCH')
hashes = metadata.get('implementation_sha256')
if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(path, str) or not isinstance(sha, str) or len(sha) != 64
        or any(char not in '0123456789abcdef' for char in sha) for path, sha in hashes.items()):
    raise SystemExit('ATTEMPT_006_GENERATION_HASH_RECORD_INVALID')
print(json.dumps({'status': 'PASS', 'generation_implementation_sha256': hashes}, sort_keys=True))
'@
$arguments = @(
    'run', '--rm', '--pull=never', '--network', 'none', '--read-only',
    '--volume', "${volumeName}:/artifacts:ro",
    '--env', 'PYTHONDONTWRITEBYTECODE=1',
    $image, 'python', '-c', $probe
)
$output = & $docker @arguments
if ($LASTEXITCODE -ne 0) { throw 'ATTEMPT_006_GENERATION_HASH_PROBE_FAILED' }
Write-Output $output
