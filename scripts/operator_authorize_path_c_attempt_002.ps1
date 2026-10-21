# Authorization only. Never invoke PATH-C generation or retry.
[CmdletBinding()]
param(
    [ValidateSet(2, 3, 4, 5, 6)]
    [int]$ApprovedAttempt = 2
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$docker = (Get-Command docker -ErrorAction Stop).Source
$raw = @(& $docker compose config --format json)
if ($LASTEXITCODE -ne 0) { throw 'COMPOSE_CONFIG_FAILED' }
$config = ($raw -join "`n") | ConvertFrom-Json
$mounts = @($config.services.api.volumes | Where-Object { $_.type -eq 'volume' -and $_.target -eq '/app/artifacts' })
if ($mounts.Count -ne 1) { throw 'CANONICAL_ARTIFACT_VOLUME_AMBIGUOUS_OR_MISSING' }
$definition = $config.volumes.PSObject.Properties[[string]$mounts[0].source]
if ($null -eq $definition -or -not $definition.Value.PSObject.Properties['name']) { throw 'CANONICAL_VOLUME_NAME_MISSING' }
$volume = [string]$definition.Value.name
& $docker volume inspect $volume --format '{{.Name}}'
if ($LASTEXITCODE -ne 0) { throw 'CANONICAL_VOLUME_UNAVAILABLE' }
$image = [string]$config.services.api.image
if (-not $image) { throw 'CANONICAL_API_IMAGE_MISSING' }
& $docker image inspect $image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'LOCAL_IMAGE_UNAVAILABLE_NO_PULL' }
$argsList = @('run', '--rm', '--pull=never', '--network=none', '--read-only',
    '--mount', "type=volume,src=$volume,dst=/app/artifacts,volume-nocopy",
    '--mount', "type=bind,src=$repoRoot,dst=/opt,readonly",
    '-e', 'PYTHONPATH=/opt/apps/api:/opt', '-e', 'PYTHONDONTWRITEBYTECODE=1',
    '--entrypoint', 'python', $image, '-m', 'scripts.prospective_acquisition.path_c_authorize_002',
    '--approved-attempt', [string]$ApprovedAttempt)
& $docker @argsList
$authorizationExit = $LASTEXITCODE
if ($authorizationExit -ne 0) { exit $authorizationExit }
exit 0
