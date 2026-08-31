Write-Host "=== 1. Docker registry mirrors ==="
docker info --format "Mirrors: {{.RegistryConfig.Mirrors}}"

Write-Host "=== 2. hosts 文件中 docker/registry 相关条目 ==="
Select-String -Path "C:\Windows\System32\drivers\etc\hosts" -Pattern "docker|registry|auth" -ErrorAction SilentlyContinue
if (-not $?) { Write-Host "(hosts 无相关条目或读取失败)" }

Write-Host "=== 3. 解析 auth.docker.io / registry-1.docker.io 的 IP ==="
Resolve-DnsName auth.docker.io -ErrorAction SilentlyContinue | Select-Object Name, IPAddress, Type | Format-Table -AutoSize
Resolve-DnsName registry-1.docker.io -ErrorAction SilentlyContinue | Select-Object Name, IPAddress, Type | Format-Table -AutoSize

Write-Host "=== 4. 系统代理设置 ==="
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" | Select-Object ProxyEnable, ProxyServer, AutoConfigURL | Format-List
