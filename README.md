# DeepResearch Agent

DeepResearch Agent 是一个面向长周期研究的 Evidence-Aware & Memory-Aware 深度研究系统。它不把“搜索网页并总结”作为创新点，而是围绕三个可验证核心运行：Evidence Graph、Coverage/Information-Gain Research Loop、Context Budget Manager。系统持续回答当前知道什么、还缺什么、为什么继续研究以及哪些证据支持结论。

V1 的产品定位是“证据驱动、自适应上下文管理 Research Agent”：Research Plan 展示研究维度覆盖度和缺口，Agent Activity 展示上下文预算、工具决策和信息增益，Evidence 展示 Claim、来源、支持/反驳关系及冲突。当前 StateGraph、细粒度 Evidence Graph、Context Manifest、PostgreSQL Hybrid Retrieval、Working/Episodic/Semantic Memory 和 Evaluation Snapshot 均已接入 V1。

完整设计见 [DeepResearch_Agent_V1_开发计划.txt](./DeepResearch_Agent_V1_开发计划.txt)。

## 当前进度

V1 的自主 Research Loop 与证据驱动报告主链路已经可以端到端运行：

- FastAPI、React/Vite、PostgreSQL 18、Redis、SearXNG 和 Docker Compose
- Known / Unknown / Next Action 状态契约与不可变 Reducer
- 多模型 Provider 的统一请求、结果、能力与凭据版本契约
- OpenAI Responses、Anthropic Messages、Google Gemini 和 OpenAI Compatible 四类结构化 Adapter
- 已保存配置可切换 API 协议，服务端自动用新 AAD 重加密同一 API Key
- Provider Profile 的创建、读取、更新、密钥轮换和软删除 API
- API Key 的 AES-256-GCM 服务端加密持久化
- HttpOnly 签名浏览器会话；刷新后恢复模型配置和密钥末四位
- Research Run 的幂等创建、查询、取消和恢复
- PostgreSQL Agent Event 事实源、严格递增 Run Seq 与 SSE Last-Event-ID 重放
- Research Run、首条 Event 和 Dispatch Outbox 的同事务提交
- 独立 Outbox Dispatcher、Celery Worker、晚确认与单并发预取
- PostgreSQL Worker Lease、重复投递跳过和失败边界事件
- 前端按 Last-Event-ID 增量重放，并在刷新后恢复最近任务
- 真实 Planner 模型调用、5–8 个子问题 Schema 校验、版本化计划表和用量快照
- SearXNG Web Search Tool；搜索摘要仅存候选元数据，禁止直接成为 Evidence
- PublicWebReader：SSRF / DNS / Redirect / Port / Content-Type / 2 MB 大小边界
- Trafilatura 正文提取、30,000 字符上限、SHA-256 内容哈希与本地 Artifact
- Evidence Extractor：最小 Context、网页提示注入隔离、严格 Schema 与逐字引文校验
- 来源可靠性 × 相关性 × 模型置信度的确定性 Evidence Score
- PostgreSQL Gap、Tool Call、Search Query/Result、Source、Evidence 长期记忆表
- Evaluator 多轮质量快照：覆盖度、来源质量、来源独立性、有效证据、冲突与引用数
- 基于质量停止条件与不可变资源预算的自主补缺循环
- Evidence API 与前端证据卡片，可查看接受/拒绝原因并直接打开原始来源
- Evidence-only Report Writer：按问题选择有限证据卡片，避免将完整网页塞回上下文
- 稳定引用注册表：正文引用绑定 Evidence、来源 URL、内容哈希与访问时间
- 确定性报告校验：引用完整度、数字引用有效率、限制项与降级状态
- 报告 API 与前端报告阅读器，正文引用可直接跳转原始来源
- 用户主动结束研究、失败恢复，以及预算耗尽时的可验证降级报告
- Alembic Migration、真实 PostgreSQL 集成测试和 GitHub Actions CI

当前 Worker 会调用用户明确选择的 Provider Adapter 生成并持久化研究计划，依据当前 Known / Unknown / Next Action 状态选择工具，自主执行多轮搜索、网页安全读取、证据抽取、质量评估和动态补缺。满足质量阈值或达到资源上限后进入 Writer；Writer 只接收经过验证的有限 Evidence Cards，并通过稳定引用和确定性校验生成报告。预算不足时会生成明确标注限制的证据报告；没有有效证据时会失败，而不会伪造结论、来源或质量指标。

V1 已进入 Release Gate 阶段；以下命令生成静态、集成、Web 与 Golden Eval 的可审计
JSON 报告，但不能单独证明真实 V1 收口：

```powershell
$env:RUN_POSTGRES_INTEGRATION = "1"
$env:DATABASE_URL = "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research"
$env:TEST_DATABASE_URL = $env:DATABASE_URL
$env:CHECKPOINT_DATABASE_URI = "postgresql://deep_research:deep_research@localhost:5432/deep_research_checkpoint"
python scripts/release_gate.py --skip-compose --report-path artifacts/release_gate.json
```

该 Gate 会执行 Ruff、MyPy、全量测试、零跳过 PostgreSQL 集成测试、前端构建、13 条
Golden Eval 和高置信度 Secret 扫描。只有增加 `--v1-closeout`，并由数据库证明同一配置、
可识别代码版本下最新连续三次真实 Run 全部达到质量门，才允许宣称 V1 收口通过：

```powershell
$env:V1_ACCEPTANCE_OWNER_HASH = "<当前验收用户 owner hash>"
python scripts/release_gate.py --v1-closeout --v1-source-revision "<提交版本>" `
  --report-path artifacts/release_gate.json
```

真实三连验收会调用已配置的模型与搜索服务，应在得到用户明确授权后另行启动；Gate 本身
只读取已经完成的运行，不会创建任务或调用外部服务。

## 技术栈

- Python 3.12、FastAPI、Pydantic v2、SQLAlchemy 2、Alembic
- LangGraph、官方 PostgreSQL Checkpointer
- PostgreSQL 18、Redis、Celery
- React 19、TypeScript、Vite
- SearXNG、httpx、Trafilatura
- cryptography / AES-256-GCM

## 快速启动

### 环境要求与冻结配置

- Docker Engine（支持 Docker Compose v2）以及可用的容器镜像构建环境；
- 首次启动需要网络访问以获取镜像/依赖；
- Windows 一键启动需要 PowerShell；手动启动可使用支持 Docker Compose 的操作系统。

V1 RC 使用 `v1.0.0-rc.1` 与冻结配置 `v1-pre-rc-reference-1`：
`EVIDENCE_AWARE_CONTEXT_ENABLED=false`、
`INDEPENDENT_SOURCE_TARGETING_ENABLED=true`、
`EVIDENCE_INPUT_QUALITY_ENABLED=false`。这些开关由 Compose 显式传给 API 和 Worker；
不要在发布验证中通过个人 shell 环境变量覆盖它们。

### 一键启动（Windows）

双击项目根目录的 `start.bat`，或在 PowerShell 中运行：

```powershell
.\scripts\start.ps1
```

脚本会自动完成：生成 `.env`（如缺失）→ 检查并启动 Docker → 构建并启动全部服务 → 等待 API/Web 就绪 → 初始化 LangGraph Checkpoint（幂等）→ 打开浏览器。

常用参数与配套脚本：

```powershell
.\scripts\start.ps1 -NoBrowser   # 不自动打开浏览器
.\scripts\start.ps1 -NoBuild     # 跳过镜像构建，仅启动已有镜像
.\scripts\start.ps1 -DockerContext my-linux-engine -NoBrowser  # 使用独立 Docker 引擎，不依赖 Docker Desktop
.\scripts\stop.ps1               # 停止全部服务（保留数据）
.\scripts\stop.ps1 -Data         # 停止并删除全部数据卷（不可恢复）
.\scripts\status.ps1             # 查看服务运行状态
```

命令行 Docker 本身只是客户端，仍需要一个可访问的 Docker 引擎。项目不再把 Docker Desktop
作为唯一引擎：可以先设置 `DOCKER_HOST`，或创建/选择一个连接到 WSL2、远程 Linux、Podman
兼容引擎的 Docker context，再启动项目：

```powershell
docker context ls
$env:DOCKER_CONTEXT = "my-linux-engine"
.\scripts\start.ps1 -NoBrowser -NoAutoStartDockerDesktop
```

当指定了 `-DockerContext`、`DOCKER_CONTEXT` 或 `DOCKER_HOST` 时，启动脚本不会尝试拉起
Docker Desktop；如果该引擎不可用，会直接提示当前连接信息和诊断命令。若未指定替代引擎，
脚本仍保留原有的 Docker Desktop 自动启动行为。

若 PowerShell 执行策略禁止运行脚本，可先执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，或直接使用 `start.bat`。

### 手动启动

复制环境变量：

```powershell
Copy-Item .env.example .env
```

启动完整本地环境：

```powershell
docker compose up -d --build
```

Compose 中的一次性 `migrate` 服务会先执行 Alembic，再启动 API。首次安装后，另行初始化官方 LangGraph Checkpoint 数据库：

```powershell
docker compose run --rm api python -m app.cli.setup_checkpoints
```

默认地址：

- Web：http://localhost:5174
- API：http://localhost:8000
- API 文档：http://localhost:8000/docs
- SearXNG：http://localhost:8081

### 创建 Run、查看状态与获取报告

先在 Web 页面创建 Provider Profile 并保存凭据。API 使用浏览器建立的 HttpOnly
客户端会话；不要把 API Key 放进 URL、命令历史或日志。创建 Run 的接口为
`POST /api/v1/research-runs`，请求包含 `query`、`saved_profile_version_id` 和
`budget_tier`（`quick`、`standard` 或 `deep`），并提供唯一的 `Idempotency-Key`。
响应中的 `run_id`、`status_url` 和 `event_url` 可用于后续查询/订阅。

- 存活检查：`GET /healthz`；依赖就绪检查：`GET /readyz`。
- Run 状态：`GET /api/v1/research-runs/{run_id}`；事件：`GET .../{run_id}/events`。
- 报告：`GET /api/v1/research-runs/{run_id}/report`；验证状态：
  `GET /api/v1/research-runs/{run_id}/verification`。

没有 Accepted Evidence 时，系统会以 `REPORT_NO_ACCEPTED_EVIDENCE` 终止，不会生成无依据报告。

## 本地开发

后端：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:DATABASE_URL = "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research"
alembic upgrade head
uvicorn app.main:app --app-dir apps/api --reload
```

前端：

```powershell
corepack enable
pnpm install
pnpm --filter @deep-research/web dev
```

质量检查：

```powershell
pytest -q
ruff check apps/api tests
mypy apps/api tests
pnpm --filter @deep-research/web build
docker compose config --quiet
```

Benchmark 定义位于 `evals/benchmarks/`。静态/回归质量门可通过
`python scripts/release_gate.py` 运行；该命令不会代替真实 qualification，
也不会自动启动 Benchmark Run。当前已知任务族限制见
[`artifacts/v1_known_issues.md`](./artifacts/v1_known_issues.md)。

真实 PostgreSQL 集成测试：

```powershell
$env:RUN_POSTGRES_INTEGRATION = "1"
$env:TEST_DATABASE_URL = "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research"
pytest tests/integration/test_provider_profile_persistence.py -q
```

## 凭据安全边界

- API Key 只通过 HTTPS POST/PATCH 请求体进入后端，不进入 URL。
- 数据库不保存明文，只保存 AES-GCM ciphertext、nonce、版本、HMAC 指纹和末四位。
- API 响应、SSE、Agent State、Prompt、日志、Redis 和 Dispatch Outbox 不包含明文密钥。
- 浏览器不使用 localStorage/sessionStorage 保存密钥，只持有 HttpOnly 签名会话 Cookie。
- 开发环境主密钥位于被 Git 忽略的 `artifacts/.secrets`；生产必须注入 `SECRET_MASTER_KEY_BASE64`。
- V1 不执行模型生成的任意 Python 或 Shell。

## 关键架构文档

- [PostgreSQL 边界](./docs/adr/0001-postgresql-boundaries.md)
- [BYOK Provider 与凭据隔离](./docs/adr/0002-byok-provider-boundary.md)
- [P0/P1 基础架构](./docs/architecture/p0-p1-foundation.md)
- [V1 方向与创新点](./docs/architecture/0003-v1-focus-and-innovation.md)

## V1 Release Gate

在提交 GitHub 前运行静态、测试与构建门禁：

~~~powershell
python scripts/release_gate.py
~~~

该命令的报告范围是 `static_release_checks`，不能据此宣称 V1 已完成。真实收口必须在同一
可识别代码版本和同一配置下完成连续三次 Run，再运行：

~~~powershell
$env:RUN_POSTGRES_INTEGRATION = "1"
$env:V1_ACCEPTANCE_OWNER_HASH = "<验收用户 owner hash>"
python scripts/release_gate.py --v1-closeout --v1-source-revision "<提交版本>" `
  --report-path artifacts/release_gate.json
~~~

真实门禁会核验三次运行均进入并完成报告阶段，且总体覆盖率 >=85%、P1 覆盖率 >=80%、
交叉验证 >=70%、关键缺口为 0；任一运行是 `completed_with_limitations` 都会失败。
`--v1-closeout` 禁止搭配任何 `--skip-*` 参数，并要求提供非开发态的候选代码版本。

所有 API Key 只通过本地 `.env` 或页面输入使用；`.env`、运行时 Artifact、Secret 和构建产物均由 `.gitignore` 排除，不应提交到 GitHub。
