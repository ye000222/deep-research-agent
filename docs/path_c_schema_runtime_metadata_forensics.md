# PATH-C Canonical Schema Runtime Metadata Failure Forensics

## A. 首个失败子检查

权威 run：
`artifacts/v2_1r_d4r_a6_operator/run_20261013T003128497Z_d9f2f042`。

首个失败是 runtime package metadata 判断：
`schema_runtime.runtime_dependency_declared = false`。失败阶段为
`SCHEMA_RUNTIME_PROBE_COMPLETED`。这是 CASE B：**ATTESTATION_PROBE**。

证据文件：

- `canonical_runtime_attestation.json`：总体 `BLOCKED_SCHEMA_RUNTIME_DEPENDENCY_OR_VALIDATION_FAILED`，镜像 ID、失败 stage、probe 汇总。
- `operator_probe_execution.json`：ephemeral command 的 exit code、envelope 和安全 stdout/stderr。
- `operator_probe_stage_report.json`：Python identity、模块导入、schema runtime 子结果及 provider 状态。
- `provider_readiness_attestation.json`：Provider resolution `NOT_REACHED`。
- `docker_host_attestation.json`：host Docker prechecks 和零调用计数。
- `hash_manifest.json`：上述 artifacts 的文件哈希及 manifest digest。

这些 JSON 均可解析；五份结果文件哈希和 manifest digest 匹配。

## B. 精确异常 / 判定

没有抛出 exception。probe 以 exit code **21** 结束，返回安全码
`SCHEMA_RUNTIME_NOT_READY`。

首个失败语句在 `scripts/prospective_acquisition/schema_runtime_probe.py` 的
`probe_schema_runtime()`：原逻辑以 `r.lower().startswith("jsonschema[format]")`
识别依赖。Wheel metadata 将 PEP 508 requirement 规范化为
`jsonschema[format]<5,>=4.23`；该字符串不以 `jsonschema[format]>=` 开头。
运行时依赖实际存在，但这个字符串前缀条件得到 false。

安全实际值：metadata flag `false`；预期值：`true`。独立 build log 也显示
规范化依赖 `jsonschema[format]<5,>=4.23` 和成功安装 `jsonschema-4.26.0`。

## C–E. Schema runtime 子检查

| 子检查 | 状态 | Artifact 证据 |
|---|---|---|
| A. runtime metadata 含非 dev jsonschema dependency | FAIL（探针判定错误） | `schema_runtime.runtime_dependency_declared=false`；规范化 metadata 顺序导致旧前缀判断失败 |
| B. `import jsonschema` | PASS | `jsonschema_version=4.26.0`；validator 已初始化 |
| C. 安装版本满足 `>=4.23,<5` | PASS | 已记录版本 `4.26.0`，符合该范围 |
| D. FormatChecker import / initialization | PASS | `validator_initialized=true`；validator factory 在返回前导入并构造 `FormatChecker()` |
| E. Validator class selection | PASS | `validator_class=Draft202012Validator` |
| F. 对 frozen schema 初始化 validator | PASS | `validator_initialized=true`；schema SHA 与冻结 SHA 相同 |
| G. synthetic valid 40-task 文档被接受 | PASS | `valid_document_passed=true` |
| H. synthetic invalid 文档被拒绝 | PASS | `invalid_document_rejected=true` |
| I. Provider / HTTP counters 为零 | PASS | envelope 和 host artifacts 均记 generation=0、HTTP=0；DB writes=0 |

## 运行路径证据

`operator_probe_execution.json` 记录 ephemeral command exit code **21**，
stdout 有 probe envelope，stderr 的 stage marker 顺序为：

```text
PROBE_PROCESS_STARTED
PYTHON_ENV_READY (Python 3.12.14)
CURRENT_CHECKOUT_IDENTITY
IMPORT_PROBE_COMPLETED (28 passed, 0 failed)
SCHEMA_RUNTIME_PROBE_STARTED
SCHEMA_RUNTIME_PROBE_COMPLETED (FAIL)
```

`canonical_runtime_attestation.json` 的 image ID 是
`sha256:836a3e23486f256d9fcc8d3862de3105e080aaabc70b167fea99673e6b7a2dc7`。
Schema probe 在 Provider resolution 之前失败；Provider 状态 `NOT_REACHED`。
这是 metadata assertion failure，不是旧的 Windows command-length 或 mount/path
failure，也不是 jsonschema import、schema 构造或 fixture 失败。

## F. 分类

**ATTESTATION_PROBE**。没有证据支持 `RUNTIME_PACKAGING`、`SCHEMA_VALIDATION`、
`SYNTHETIC_FIXTURE` 或 `PATH/MOUNT` 缺陷。

## G. 最小修复

`schema_runtime_probe.py` 现以 `packaging.requirements.Requirement` 解析 PEP 508
metadata，比较规范化 package name、`format` extra、`SpecifierSet` 和 marker；
因此会忽略仅在 `extra == "dev"` 时启用的依赖。另显式核验实际安装版本符合
`>=4.23,<5`，并把此结果纳入 probe PASS 条件。

Fail-closed、冻结 schema、fixture、Acceptance 和生成授权均未放宽；没有动态 pip
install 或 fallback。

## H. Deterministic validation

- schema runtime / metadata parsing / unsupported version / fail-closed tests：通过。
- frozen schema validation 与 canonical invocation tests：通过。
- 定向 pytest：**24 passed**。
- targeted Ruff：PASS。
- targeted mypy：PASS，3 个文件。
- PowerShell parser 与 `-SelfTest`：PASS。
- JSON / historical artifact hash / manifest digest：PASS。
- `git diff --check`：PASS；仅既有 pyproject.toml LF/CRLF warning。

Docker、Provider、External HTTP、DB writes、Holdout access 均为 0。本轮未更改
canonical run artifacts，未 stage/commit；Attempt 001–004 保持不动，未授权或运行
Attempt 005。

## I. Host re-attestation

修复后需要重新执行只读 canonical host attestation，以在镜像 runtime path 中确认
metadata gate 完整通过。唯一命令：

```powershell
Set-Location D:\agentProject; powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\operator_verify_canonical_runtime.ps1
```

## 最终判定

CANONICAL SCHEMA RUNTIME DEFECT IDENTIFIED AND CLOSED — HOST RE-ATTESTATION REQUIRED
