<#
.SYNOPSIS
    查看 DeepResearch Agent 服务运行状态

.EXAMPLE
    .\scripts\status.ps1
#>

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
docker compose ps
