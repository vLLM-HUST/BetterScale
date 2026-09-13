# 固定 donor 环境下的启动与回退

## 交付边界

维护实现位于 `src/strengthen_dsv4/patches/`，通过 vLLM 原生 `worker_cls` 接入。
不改子模块、不改 `site-packages`、不用 `sitecustomize`，也不依赖 prototype 目录。
安装 Python 包只是可选的命令注册；**不要用 pip 升级 donor 或重编算子来启动本交付。**

当前固定：DSV4 Flash43层/4096 hidden/256 routed experts、W8A8、BF16、TP8/EP/DP1/PP1、
DSACP、DCP1/PCP1、K5、4 active seats、4128 graph budget、原生 async scheduler。
输入／输出总长度上限15104；默认12GiB KV/rank。首次遇到新 draft bank 会发生 capture，
冷启动成本不应混入 warm steady-state 性能。
4128是捕获容量；原生配置对齐后，实际单波调度 token 上限为4112。

## 启动

```bash
# 先激活或指定已经准备好的 donor 环境。
export STRENGTHEN_PYTHON=/path/to/donor-env/bin/python
./bin/strengthen-dsv4 check
./bin/strengthen-dsv4 plan --model /models/DeepSeek-V4-Flash

# 共享实验机：先按本机已有协议持有 ~/tp8.lock，检查八卡健康且无外来任务。
# 启动器不是调度器，不会抢占或终止别人的进程。
./bin/strengthen-dsv4 serve \
  --model /models/DeepSeek-V4-Flash \
  --profile optimized \
  --artifacts runs/serve-optimized-001
```

`plan` 不导入 vLLM、不触碰 NPU；`check` 不加载模型。`serve` 在导入引擎前检查
版本与11份相关私有 API 源码，再由 workers 检查模型与并行配置。
出现不匹配时拒绝启动，不提供“静默忽略”的开关。

Shell 入口默认 source `/usr/local/Ascend/cann-9.0.1/set_env.sh`；非默认安装路径用
`STRENGTHEN_CANN_ENV` 指定。CANN 和硬件仍需在已验收环境中；源码检查不是整个
二进制工具链的认证。入口从当前 donor distribution 定位 Ascend 自定义算子库，
不会复制或重新编译它们。

默认只监听 `127.0.0.1:8000`，模型名 `dsv4`。可显式设置 `--host`、`--port`、
`--served-model-name`；对外暴露时另行设置鉴权、TLS和网络访问控制，不要把公网
监听误当成已经具备生产安全部署。每次使用新的 artifacts 目录，避免覆盖证据。

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","prompt":"Hello","max_tokens":16,"temperature":0}'
```

也可在 donor venv 中 `python -m pip install --no-deps --no-build-isolation .`，
使用 `strengthen-dsv4` 命令；这需要已有 setuptools>=68，且调用前须先配置 CANN。
从仓库运行 `bin/strengthen-dsv4` 不需要这一步。

## 回退与停止

给前台服务正常发送 SIGTERM/Ctrl-C，等待其 workers 退出、八卡释放，再启动：

```bash
./bin/strengthen-dsv4 serve --model /models/DeepSeek-V4-Flash \
  --profile baseline --artifacts runs/serve-baseline-001
```

Baseline 只带 K5/TP8 LCM 启动兼容修复；target 为原生 FULL_DECODE_ONLY，draft 为
原生 eager，不启用其余优化。与 optimized 保持 seats、预算、KV 和模型长度一致。
不热切换已有进程，不需要逆向修改 donor 文件。对照测试使用同一 host、输入、
采样设置，且单独 warmup；不要把 profiler/shadow 时间当成性能。

## 运维证据

- `launch.json`：配置、版本、固定源码校验与受控环境变量。
- `ready-rank0.json` … `ready-rank7.json`：八卡确认启用的补丁集和 KV 容量。
- `draft-graph-*.json`、`split-draft-*.json`：private banks、capture、replay、fallback。
- `cross-step-rank*.json`：准入、保守 bounds、延后回调计数；不是用户请求正文。
- `shutdown-rank*.json`：仅在 native worker shutdown hook 被调用时保存最终状态。
  本轮 HTTP 服务正常终止也未必调用此 hook；文件缺失不能作为释放失败的判断。

Bank receipts 在 capture／首次 replay 等节点保存，不保证是退出时的完整累计计数。
以进程退出和 NPU admission/release 检查确认资源释放，不依赖 shutdown receipt。

Native 混合 KV 的 token capacity 是给定 max context / prefill budget 下的容量估计，
不是简单的平坦 token slots。启动门槛要求能容纳4个满长请求；减小 `--kv-gib` 可能
触发门槛。扩大席位、换 K、启用 DP 或 prefix caching 都属于新的验收范围。

在线没有全 KV shadow。CPU 合约测试：

```bash
PYTHONPATH=src "$STRENGTHEN_PYTHON" -m unittest discover -s tests
```

HTTP 验收脚本 `tests/serve_acceptance.py` 必须放在现有 NPU 租约／admission supervisor
下运行。它调用上述真实启动入口，完成32条保留的 OpenCompass 原始 tokenized 输入
和一个 streaming 请求；评分复用固定的 OpenCompass evaluator。测试数据不进入包。
