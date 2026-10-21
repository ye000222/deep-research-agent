# PATH-C Canonical Runtime Attestation Failure Forensics

## 范围与结论

只分析已有 host build log、冻结 attestation artifacts 与本地脚本；修复已复现的
Windows 调用边界缺陷。没有运行 Docker、重建镜像、调用 Provider 或生成任务。

历史证据目录：
`artifacts/v2_1r_d4r_a6_operator/run_20261012T153719022Z_2c7ebe80`。
六份 JSON 均可解析，五份结果文件的 SHA-256 与历史 manifest 一致，manifest 自身
排除 self-hash 后的摘要也匹配。历史 artifacts 没有改写。

## A. 失败命令 / gate

`operator_probe_execution.json`：`status=FAILED`、
`failure_category=EPHEMERAL_CONTAINER_COMMAND_FAILED`、
`failed_stage=COMPOSE_RUN_INVOCATION`。耗时 29 ms，没有 Python envelope、stage markers
或 import results。历史 canonical artifact 保留了旧的
`RUNNING_EPHEMERAL_CURRENT_SOURCE_PROBE` 状态，不能解释为仍在运行或 PASS。

旧命令的完整语义结构如下；占位符对应当时的完整编码内容，不是另一次执行：

```text
docker compose run --no-deps --name d4ra6-operator-61f9878007f3
  -e OPERATOR_EXPECTED_HASHES_B64=<4532-character base64 hash map>
  -e OPERATOR_EXECUTION_CONTEXT_ID=8862f286-e707-461e-876f-ec57207802e8
  -v D:\agentProject\scripts\prospective_acquisition:/opt/scripts/prospective_acquisition:ro
  -v D:\agentProject\scripts\provider_fingerprint_contract.py:/opt/scripts/provider_fingerprint_contract.py:ro
  -v D:\agentProject\artifacts\v2_1r_d4r_a51:/opt/artifacts/v2_1r_d4r_a51:ro
  --entrypoint python api -c <27737-character Python coordinator>
```

继承 API service 的 `/app/apps/api` 只读源码挂载及 `/app/artifacts` volume。
镜像声明 `WORKDIR /app`；这是计划值，没有此次进程内 cwd 证明。
环境假设沿用 API Settings、只读 DB probe 与 frozen execution context；没有独立
Provider selector，也没有生成调用。

## B. 精确退出码

历史 `compose_run.exit_code=null`，不是 1，也不是 Python probe 的 21。
历史仅保留 `MethodInvocationException`；native process invocation 没有返回退出码。
顶层 PowerShell 失败不能冒充 Docker/Python 的退出码。

## C. stdout / stderr 与根因

历史 `stdout_safe=""`、`stderr_safe=""`，两个 present 标志均为 false，
`timed_out=false`。历史 inner Win32 error code 未保留。

使用原 `ConvertTo-WindowsArgument` 重建旧参数长度：Python source 27,737 字符，
转义后的 source argument 28,833 字符，总 arguments 33,827 字符。
arguments 本身已超过 Windows 32,767 字符命令行上限，尚未算 executable。

使用同一 `Invoke-Native`，仅向本地 Python 传入无害超长 `pass` fixture，独立复现
`MethodInvocationException -> Win32Exception`、native error 206、
`PROCESS_START` 失败。没有执行 Docker。

**分类：Case E，host ephemeral invocation 的 argv 长度缺陷。**
206 是离线复现取得的值，不是补写进旧 artifacts 的历史值。没有证据支持
jsonschema 缺失、schema validation 失败或 Docker pipe permission failure。
历史 cleanup 记录不能证明容器曾被成功创建。

## D. 是否到达 rebuilt image 内的 jsonschema import

没有到达。没有任何 Python 启动标记或 schema probe 输出。
不能把总体 attestation failure 归因于 jsonschema import。

## E. schema_runtime 子检查

| 子检查 | 本次历史 attestation 状态 |
|---|---|
| 安装 metadata 的 non-dev jsonschema dependency 检查 | NOT_REACHED |
| jsonschema import | NOT_REACHED |
| validator initialization | NOT_REACHED |
| synthetic valid 40-task document accepted | NOT_REACHED |
| synthetic invalid document rejected | NOT_REACHED |
| Provider generation / HTTP counters 为 0 | PASS（外层 artifacts 记录；不是 probe 成功证明） |

Schema probe 未开始，也未完成。本地 deterministic schema tests 不替代 rebuilt image
中的 runtime attestation。

## F. Packaging 验证边界

host build log 证明 runtime dependency 安装 `jsonschema 4.26.0`，build 成功。
artifact image ID 与 rebuild log 的 image manifest list 一致：
`sha256:836a3e23486f256d9fcc8d3862de3105e080aaabc70b167fea99673e6b7a2dc7`。

因此 dependency declaration / build installation 已验证；rebuilt image 内 import、
validator 与 synthetic probes 尚未验证。没有依据重开 dependency-declaration 缺陷，
也没有依据宣称 canonical schema runtime 已 PASS。

## G. 最小修复

1. `scripts/prospective_acquisition/canonical_runtime_probe.py`：抽出原内嵌 Python
   coordinator。保留 schema/checkout/import/DB/frozen-context/Provider 检查顺序及
   fail-closed 行为；仅移除两个无用 import，没有修改生成逻辑。
2. `scripts/operator_verify_canonical_runtime.ps1`：使用已有只读脚本挂载直接执行
   `/opt/scripts/prospective_acquisition/canonical_runtime_probe.py`，不再传入
   `python -c` source；新脚本加入 current-checkout hash 验证。
3. Process.Start 失败只保留安全异常类型、native code、phase 与参数字符数；
   不保存完整 arguments、exception message、credential 或 env。未来失败 artifact
   明确记录 FAIL，不再留下 RUNNING 状态。
4. `tests/unit/test_canonical_runtime_probe_invocation.py`：真实本地 Windows launcher
   边界复现、带空格文件路径成功执行、实际 Docker 参数构造的短长度断言，以及
   缺失 identity 的 safe-import fail-closed 检查。不执行 Docker。
5. `tests/unit/test_path_c_schema_runtime_dependency.py`：适配 coordinator 的新位置，
   继续断言 schema probe 在 Provider resolution 前执行且 host gate 不被跳过。

当前实际参数重建结果：32 个 runtime files；hash env 4,704 字符；
完整 arguments **5,226** 字符；没有 `-c`。不需再次重建镜像即可读取只读挂载脚本。
没有动态 pip install、schema fallback、validation bypass 或授权修改。

## 离线质量门

PowerShell syntax、`-SelfTest`：PASS。
Python AST / synthetic safe-import：由新增测试覆盖。
targeted Ruff：两份测试及 schema_runtime_probe 全规则 PASS；抽出的历史动态
coordinator 维持原风格，E9/F PASS。
mypy：两份测试及 schema_runtime_probe，3 个文件 PASS；不是全仓类型检查。
动态 coordinator 保留原内嵌脚本类型风格，以 AST / safe-import / gate tests 验证。
git diff --check：PASS，仅既有 pyproject.toml LF/CRLF warning。
JSON / artifact hash / manifest digest validation：PASS。

首次沙箱测试被 temp directory 权限与 asyncio 本地 socketpair 初始化阻塞，未进入
真实模型调用。未修改项目或测试以绕过；完整 deterministic regression 在获准的
沙箱外本地测试环境执行：全部 `test_path_c_*.py`、新增 launcher tests 与
JSON Schema unit tests，**207 passed / 3 skipped，51.66s**。
三个 skip 为可选 Brotli codec 不存在；不是 schema/invocation gate 的 skip。

## H. 是否需要 host re-attestation

需要。此轮只证明 invocation 缺陷修复，不能代替真实 canonical runtime gate。
仅重新执行只读 attestation；不得调用 generation entry，也不得授权 Attempt 005。

## I. 唯一 host 命令

```powershell
Set-Location D:\agentProject; powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\operator_verify_canonical_runtime.ps1
```

## 安全与历史

Docker=0；Provider generation=0；External HTTP=0；DB writes=0；Holdout access=0。
Attempt 001–004 未改动，Attempt 004 不复位；未创建或授权 Attempt 005。
没有 stage/commit，Git index SHA-256 保持
`2ff622a209ef7ea3e60e12e496b057cd7d0fe59488561bfe2bb1cc02c03f5787`。
没有修改 Production / Research semantics、task schema、Provider config 或生成授权。

## 最终判定

CANONICAL SCHEMA RUNTIME ATTESTATION DEFECT CLOSED — HOST RE-ATTESTATION REQUIRED
