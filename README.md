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

V1 已进入严格 Release Gate 阶段；本地可用以下命令生成可审计 JSON 报告（严格模式要求 PostgreSQL 集成开关）：

```powershell
$env:RUN_POSTGRES_INTEGRATION = "1"
$env:DATABASE_URL = "postgresql+psycopg://deep_research:deep_research@localhost:5432/deep_research"
$env:TEST_DATABASE_URL = $env:DATABASE_URL
$env:CHECKPOINT_DATABASE_URI = "postgresql://deep_research:deep_research@localhost:5432/deep_research_checkpoint"
python scripts/release_gate.py --skip-compose --report-path artifacts/release_gate.json
```

该 Gate 会执行 Ruff、MyPy、全量测试、零跳过 PostgreSQL 集成测试、前端构建、13 条 Golden Eval 和高置信度 Secret 扫描；后续版本可继续增强来源冲突归因、语义向量检索、PDF/多模态来源和 Multi-Agent 协作。

## 技术栈

- Python 3.12、FastAPI、Pydantic v2、SQLAlchemy 2、Alembic
- LangGraph、官方 PostgreSQL Checkpointer
- PostgreSQL 18、Redis、Celery
- React 19、TypeScript、Vite
- SearXNG、httpx、Trafilatura
- cryptography / AES-256-GCM

## 快速启动

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
.\scripts\stop.ps1               # 停止全部服务（保留数据）
.\scripts\stop.ps1 -Data         # 停止并删除全部数据卷（不可恢复）
.\scripts\status.ps1             # 查看服务运行状态
```

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

在提交 GitHub 前运行：

~~~powershell
python scripts/release_gate.py
~~~

脚本检查必需文件、PostgreSQL 迁移头、Compose 配置和 MySQL 专属依赖残留。完整验证还应执行：

~~~powershell
pytest -q
pnpm --dir apps/web build
docker compose run --rm --no-deps api python /app/scripts/verify_postgres_recovery.py
~~~

所有 API Key 只通过本地 `.env` 或页面输入使用；`.env`、运行时 Artifact、Secret 和构建产物均由 `.gitignore` 排除，不应提交到 GitHub。
