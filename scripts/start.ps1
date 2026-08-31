<#
.SYNOPSIS
    DeepResearch Agent 一键启动（Docker Compose 全栈模式）

.DESCRIPTION
    自动完成以下步骤：
    1. 检查/生成 .env 环境变量文件
    2. 检查 Docker 引擎，未运行时自动尝试启动 Docker Desktop
    3. 构建并启动全部服务（postgres / redis / searxng / api / worker / dispatcher / web）
    4. 等待 API 与 Web 就绪
    5. 初始化 LangGraph Checkpoint（幂等，可重复执行）
    6. 自动打开浏览器访问 http://localhost:5174

.PARAMETER NoBrowser
    启动完成后不自动打开浏览器

.PARAMETER NoBuild
    跳过镜像构建，仅使用已有镜像启动（docker compose up -d）

.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -NoBrowser -NoBuild
#>

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

# 控制台按 UTF-8 输出，避免中文提示乱码（需配合 start.bat 中的 chcp 65001）
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 项目根目录 = 脚本所在目录的上一级
$Root = Split-Path -Parent $PSScriptRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Http([string]$Url, [int]$TimeoutSeconds = 5) {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSeconds -NoProxy
        } else {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSeconds -UseBasicParsing
        }
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-DockerEngine {
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & docker info 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Invoke-DockerCompose {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & docker compose @Arguments 2>&1
        $code = $LASTEXITCODE
        $output | ForEach-Object { Write-Host $_ }
        return $code
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Wait-DockerEngine([int]$MaxSeconds = 120) {
    $waited = 0
    while (-not (Test-DockerEngine)) {
        Start-Sleep -Seconds 3
        $waited += 3
        if ($waited -ge $MaxSeconds) {
            return $false
        }
    }
    return $true
}

if (-not (Test-Path $Root)) {
    Write-Host "无法定位项目根目录 $Root" -ForegroundColor Red
    exit 1
}
Set-Location $Root

# ---------------------------------------------------------------- 1. .env
Write-Step "检查环境变量文件 (.env)"
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile) -or (Get-Item $envFile).Length -eq 0) {
    Copy-Item (Join-Path $Root ".env.example") $envFile
    Write-Host "已从 .env.example 生成 .env" -ForegroundColor Green
} else {
    Write-Host ".env 已存在，跳过生成" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- 2. Docker
Write-Step "检查 Docker 环境"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 docker 命令，请先安装 Docker Desktop：" -ForegroundColor Red
    Write-Host "  https://www.docker.com/products/docker-desktop/" -ForegroundColor Red
    exit 1
}

if (-not (Test-DockerEngine)) {
    Write-Host "Docker 引擎未运行，尝试自动启动 Docker Desktop ..." -ForegroundColor Yellow
    $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerDesktop) {
        Start-Process $dockerDesktop
    } else {
        Write-Host "未找到 Docker Desktop，请手动启动后重新运行本脚本。" -ForegroundColor Red
        exit 1
    }
    if (-not (Wait-DockerEngine)) {
        Write-Host "等待 Docker 引擎就绪超时（120 秒），请检查 Docker Desktop 后重试。" -ForegroundColor Red
        exit 1
    }
    Write-Host "Docker 引擎已就绪" -ForegroundColor Green
} else {
    Write-Host "Docker 引擎运行中" -ForegroundColor Green
}

# ---------------------------------------------------------------- 3. Compose
Write-Step "构建并启动全部服务（首次构建可能需要数分钟）"
if ($NoBuild) {
    $code = Invoke-DockerCompose up -d
} else {
    $code = Invoke-DockerCompose up -d --build
}
if ($code -ne 0) {
    Write-Host "docker compose 启动失败，可查看日志：docker compose logs -f api" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 4. API
Write-Step "等待 API 就绪 (http://localhost:8000/healthz)"
$apiReady = $false
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Http "http://localhost:8000/healthz") { $apiReady = $true; break }
    Start-Sleep -Seconds 2
}
if ($apiReady) {
    Write-Host "API 已就绪" -ForegroundColor Green
} else {
    Write-Host "API 未在预期时间内就绪，可查看日志：docker compose logs -f api" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 5. Checkpoint
Write-Step "初始化 LangGraph Checkpoint（幂等操作）"
$code = Invoke-DockerCompose run --rm api python -m app.cli.setup_checkpoints
if ($code -ne 0) {
    Write-Host "Checkpoint 初始化未完成，可稍后手动执行：" -ForegroundColor Yellow
    Write-Host "  docker compose run --rm api python -m app.cli.setup_checkpoints" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 6. Web
Write-Step "等待 Web 就绪 (http://localhost:5174)"
$webReady = $false
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Http "http://localhost:5174") { $webReady = $true; break }
    Start-Sleep -Seconds 2
}
if ($webReady) {
    Write-Host "Web 已就绪" -ForegroundColor Green
} else {
    Write-Host "Web 未在预期时间内就绪，可查看日志：docker compose logs -f web" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 汇总
Write-Step "启动完成"
Write-Host "  Web       : http://localhost:5174" -ForegroundColor White
Write-Host "  API       : http://localhost:8000" -ForegroundColor White
Write-Host "  API 文档  : http://localhost:8000/docs" -ForegroundColor White
Write-Host "  SearXNG   : http://localhost:8081" -ForegroundColor White
Write-Host ""
Write-Host "  常用命令:" -ForegroundColor DarkGray
Write-Host "    查看状态 : docker compose ps" -ForegroundColor DarkGray
Write-Host "    查看日志 : docker compose logs -f api" -ForegroundColor DarkGray
Write-Host "    停止     : .\scripts\stop.ps1" -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Process "http://localhost:5174"
}
