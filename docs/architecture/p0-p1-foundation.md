# P0/P1 工程基础验收

本里程碑冻结系统边界并提供可运行骨架，不包含伪造的研究结果。

## 已实现

- FastAPI 应用工厂、liveness、dependency readiness、Request ID。
- React/Vite 研究工作台与 BYOK 输入边界。
- Known / Gap / NextAction、预算、质量快照与 CAS StatePatch。
- Provider、Tool Calling、Evaluation 的稳定领域契约。
- PostgreSQL 业务库与独立 Checkpoint 库、Redis、SearXNG 本地拓扑。
- Checkpoint 显式初始化命令与首批契约测试。

## 验收命令

```bash
pip install -e ".[dev]"
pytest
ruff check apps/api tests
mypy apps/api/app
corepack enable
pnpm install
pnpm --filter @deep-research/web build
docker compose config
```

下一阶段实现 Alembic 业务表、Repository、事件 Outbox、Credential Vault 与 Provider Gateway。
