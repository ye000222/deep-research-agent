$ErrorActionPreference = "Stop"

# 1. 写入 daemon.json（代理配置，指向本机 Clash 代理 7897）
$dir = "C:\ProgramData\Docker\config"
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$config = @{
    proxies = @{
        "http-proxy"  = "http://127.0.0.1:7897"
        "https-proxy" = "http://127.0.0.1:7897"
        "no-proxy"    = "localhost,127.0.0.1"
    }
} | ConvertTo-Json
[System.IO.File]::WriteAllText("$dir\daemon.json", $config, (New-Object System.Text.UTF8Encoding $false))
Write-Host "daemon.json written:"
Get-Content "$dir\daemon.json"

# 2. 重启 Docker Desktop 使代理配置生效
Write-Host "Restarting Docker Desktop ..."
docker desktop restart
Write-Host "Restart command finished."

# 3. 等待引擎就绪（最多 240 秒）
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

# 4. 验证代理生效
docker info --format "HTTP Proxy: {{.HTTPProxy}} / HTTPS Proxy: {{.HTTPSProxy}} / No Proxy: {{.NoProxy}}"
