<#
.SYNOPSIS
    停止 DeepResearch Agent 全部服务

.DESCRIPTION
    默认保留数据卷（PostgreSQL / Redis / Artifacts），下次启动数据仍在。
    如需连数据一起删除，请加 -Data 参数（会清空所有数据，慎用）。

.PARAMETER Data
    停止并删除全部数据卷（不可恢复）

.EXAMPLE
    .\scripts\stop.ps1
    .\scripts\stop.ps1 -Data
#>

[CmdletBinding()]
param(
    [switch]$Data
)

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) {
    Write-Host "无法定位项目根目录 $Root" -ForegroundColor Red
    exit 1
}
Set-Location $Root

if ($Data) {
    Write-Host "正在停止服务并删除全部数据卷（不可恢复）..." -ForegroundColor Yellow
    docker compose down -v
} else {
    Write-Host "正在停止服务（保留数据卷）..." -ForegroundColor Cyan
    docker compose down
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "已停止。" -ForegroundColor Green
} else {
    Write-Host "停止失败，请检查 Docker 状态。" -ForegroundColor Red
}
