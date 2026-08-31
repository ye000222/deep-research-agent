$daemonPath = "C:\ProgramData\Docker\config\daemon.json"
Write-Host "daemon.json path: $daemonPath"
if (Test-Path $daemonPath) {
    Write-Host "--- current content ---"
    Get-Content $daemonPath
} else {
    Write-Host "(file not exists)"
}
Write-Host "--- docker desktop CLI ---"
docker desktop --help 2>&1 | Select-Object -First 5
