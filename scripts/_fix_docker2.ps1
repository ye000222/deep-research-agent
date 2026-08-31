$ErrorActionPreference = "Stop"

$settingsPath = "$env:APPDATA\Docker\settings-store.json"
$backupPath = "$env:APPDATA\Docker\settings-store.json.bak"

# 1. 备份
Copy-Item $settingsPath $backupPath -Force
Write-Host "backup: $backupPath"

# 2. 注入 Manual proxy 配置（保留原有字段）
$settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
$settings | Add-Member -NotePropertyName "ProxyHttpMode" -NotePropertyValue "manual" -Force
$settings | Add-Member -NotePropertyName "OverrideProxyHttp" -NotePropertyValue "http://127.0.0.1:7897" -Force
$settings | Add-Member -NotePropertyName "OverrideProxyHttps" -NotePropertyValue "http://127.0.0.1:7897" -Force
$settings | Add-Member -NotePropertyName "OverrideProxyExclude" -NotePropertyValue "localhost,127.0.0.1" -Force
$settings | ConvertTo-Json -Depth 5 | Set-Content $settingsPath -Encoding UTF8
Write-Host "settings-store.json updated:"
Get-Content $settingsPath

# 3. 重启 Docker Desktop
Write-Host "Restarting Docker Desktop ..."
docker desktop restart
Write-Host "Restart command finished."

# 4. 等待引擎就绪（最多 240 秒）
$waited = 0
while ($true) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    docker info 2>&1 | Out-Null
    $ok = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $previous
    if ($ok) { break }
    Start-Sleep -Seconds 5
    $waited += 5
    if ($waited -ge 240) {
        Write-Host "TIMEOUT waiting for docker engine" -ForegroundColor Red
        exit 1
    }
}
Write-Host "Docker engine ready after ${waited}s"

# 5. 验证代理
docker info --format "HTTP Proxy: {{.HTTPProxy}} / HTTPS Proxy: {{.HTTPSProxy}} / No Proxy: {{.NoProxy}}"
