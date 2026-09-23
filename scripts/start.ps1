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

.PARAMETER DockerContext
    可选的 Docker context 名称。用于连接 WSL2、远程 Linux 或其他独立 Docker 引擎，
    不需要 Docker Desktop；等价于设置 DOCKER_CONTEXT。

.PARAMETER NoAutoStartDockerDesktop
    Docker 引擎不可用时不尝试启动 Docker Desktop，而是直接输出替代引擎诊断。

.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -NoBrowser -NoBuild
#>

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoBuild,
    [string]$DockerContext,
    [switch]$NoAutoStartDockerDesktop
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
        & docker compose @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        $code = $LASTEXITCODE
        return $code
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Get-SourceRevision {
    $revision = (& git -C $Root rev-parse --verify HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $revision -notmatch "^[0-9a-f]{40}$") {
        throw "Unable to determine the Git commit for SOURCE_REVISION."
    }

    $worktreeState = @(& git -C $Root status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine whether the source worktree is clean."
    }
    if ($worktreeState.Count -gt 0) {
        return "$revision-dirty"
    }

    return $revision
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

# Compose 会读取 DOCKER_CONTEXT/DOCKER_HOST。显式参数优先，但不覆盖用户已经
# 配置好的远程引擎连接；这样命令行 Docker、WSL2 和远程 Docker 均可使用。
if ($DockerContext) {
    $env:DOCKER_CONTEXT = $DockerContext
}
$env:SOURCE_REVISION = Get-SourceRevision
Write-Host "Source revision: $env:SOURCE_REVISION" -ForegroundColor DarkGray

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
    $hasExplicitEngine = [bool]($env:DOCKER_HOST -or $env:DOCKER_CONTEXT -or $NoAutoStartDockerDesktop)
    if ($hasExplicitEngine) {
        Write-Host "Docker CLI 未连接到可用引擎。当前连接：" -ForegroundColor Red
        if ($env:DOCKER_CONTEXT) { Write-Host "  context: $($env:DOCKER_CONTEXT)" -ForegroundColor Red }
        if ($env:DOCKER_HOST) { Write-Host "  DOCKER_HOST: $($env:DOCKER_HOST)" -ForegroundColor Red }
        Write-Host "请先启动该 context 对应的 Docker 引擎，或运行 docker context ls 选择可用 context。" -ForegroundColor Yellow
        Write-Host "示例：.\scripts\start.ps1 -DockerContext my-linux-engine -NoBrowser" -ForegroundColor Yellow
        exit 1
    }

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
    $code = Invoke-DockerCompose -Arguments @(
        "up", "-d", "--force-recreate", "searxng", "api", "worker", "dispatcher", "beat", "web"
    )
} else {
    $code = Invoke-DockerCompose -Arguments @(
        "up", "-d", "--build", "--force-recreate", "searxng", "api", "worker", "dispatcher", "beat", "web"
    )
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
$code = Invoke-DockerCompose -Arguments @(
    "run", "--rm", "api", "python", "-m", "app.cli.setup_checkpoints"
)
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

# ---------------------------------------------------------------- 7. Revision
Write-Step "核对 API、Worker 与 Web 源码版本"
$revisionOk = $apiReady -and $webReady
if ($revisionOk) {
    try {
        $meta = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/meta" -TimeoutSec 5
        $revisionOk = $meta.source_revision -eq $env:SOURCE_REVISION
        foreach ($service in @("worker", "dispatcher", "beat")) {
            $serviceRevision = (& docker compose exec -T $service printenv SOURCE_REVISION 2>$null).Trim()
            $revisionOk = $revisionOk -and ($serviceRevision -eq $env:SOURCE_REVISION)
        }
        $webRevision = (& docker compose exec -T web printenv VITE_SOURCE_REVISION 2>$null).Trim()
        $revisionOk = $revisionOk -and ($webRevision -eq $env:SOURCE_REVISION)
    } catch {
        $revisionOk = $false
    }
}
if (-not $revisionOk) {
    Write-Host "运行态版本核对失败；拒绝把当前服务视为最新版本。" -ForegroundColor Red
    exit 1
}
Write-Host "运行态版本一致：$env:SOURCE_REVISION" -ForegroundColor Green

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
