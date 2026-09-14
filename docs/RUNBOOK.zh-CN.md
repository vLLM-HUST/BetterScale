# 使用原生 vLLM 命令安装与回退

## 唯一入口

在**已经准备好的 vLLM-Ascend 环境**中安装本包，给原来的启动命令加一个参数：

```bash
python -m pip install --no-deps vllm-betterscale==0.3.0
vllm serve /models/DeepSeek-V4-Flash <原有的原生参数> \
  --worker-cls betterscale.worker.Worker
```

`<原有的原生参数>` 是说明占位符，不是 shell 中直接执行的文本。
`betterscale.worker.Worker` 与旧 `strengthen_dsv4.worker.Worker` 是同一个类，
没有额外包装或执行路径。若安装过旧 `strengthen-dsv4` 发行包，先停服卸载旧包，
避免两个发行包共同拥有同一份实现文件。
已有 wheel 时直接 `python -m pip install --no-deps /path/to/vllm_betterscale-0.3.0-py3-none-any.whl`。
源码安装需要已有 setuptools>=77.0.3；包没有 donor 依赖安装/升级动作。

**没有另一套 serve/check/plan CLI，没有私有 profile，没有必须配置的目录。**
包不会 source CANN、寻找 Python、修改 `LD_LIBRARY_PATH`、设置 HCCL/allocator
环境变量或更改监听地址、模型长度、KV 预算。CANN 和 Ascend 算子加载由用户已有的
vLLM-Ascend 环境负责；环境缺失应在部署阶段解决，不由补丁偷偷补救。

## 补丁如何接入

1. vLLM 按 `--worker-cls` 创建我们的 `NPUWorker` 子类。
2. 初始化前检查固定 donor 版本与相关私有 API 源码，再分别安装 compat_lcm 和 target_full。
3. 原生 warmup 完成后，安装 draft graph、stable receipt cut、ordered replay、CPU QLI。
4. 功能模块各自拥有 hook 和 `install`，不相互 import，不在 import 时安装；
   worker 只选择组合与时机。`split_draft` 用 `DraftGraphRunner` 分流，metadata 整理直接内联在同文件，
   仅单图捕获/replay 保留在 `_graph.py`。
5. 继续原生服务；日志中每个 worker 输出 `strengthen-dsv4 rank=... READY patches=...`。

不修改 donor 源文件或 installed packages，不添加 scheduler，不带入未采用的全波次 worker-retirement 扩展。DP8 的稳定 decode 双槽 producer
是另行验收的较小组合，见下节。graph 仍在首次遇到合法 shape 时捕获，首次 capture 不等于 warm replay。

## 当前支持范围，不是参数预设

配置检查只读，不覆写用户选项。原有 TP 路线仍要求：

- 固定 pins：vLLM0.25.1、vLLM-Ascend0.25.1rc1、torch-npu2.10.0.post2；
  相关私有 API 的源码必须匹配 `pins.json`。版本一致不保证源码一致。
- DSV4 Flash43层/4096 hidden/256 experts，真实 Ascend W8A8；TP8/EP、DP1/PP1、
  DSACP、DCP1/PCP1；K5、原生 eager DSpark 配置、standard rejection。
- 四个 active seats、4128配置预算、max_model_len<=15104、target FULL、
  原生 scheduler、prefix caching 关闭。原生对齐后实际波次上限可能为4112。

四席位和预算仍是当前实现/验收边界，不是本次新增的通用能力承诺。
KV 大小完全采用用户原生设置；不再强制能同时容纳四条满长请求，是否足够由
原生引擎及用户负载决定。更大席位或其他 graph 配置需要单独验收；DP8 的单独范围见下节。

下面只是旧验收配置的**原生参数示例**，不是包里硬编码的启动预设。
环境变量继续采用你自己的、已能运行 donor 的配置：

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 \
  --max-num-seqs 4 --max-num-batched-tokens 4128 --max-model-len 15104 \
  --kv-cache-memory-bytes 12884901888 --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[24,4128],"max_cudagraph_capture_size":4128}' \
  --additional-config '{"enable_dsa_cp":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

启动前遵守实验机租约和资源检查。公网部署的鉴权/TLS/访问控制依旧使用原生运维方式。

## 回退

停止服务，等待 workers 退出，再回到你原来的 native worker 和启动命令。
无需回滚 donor 文件或卸载本包；不要在活跃 worker 内热切换补丁。

注意 pinned donor 的 K5/TP8+sequence-parallel 某些配置需要 LCM 启动修复。
删除 worker 参数也会删除这个修复，所以不保证同一组参数能作为未打补丁对照启动。
历史 baseline 是专门隔离过的实验控制，不再作为交付包的第二个入口。做性能 A/B
必须重新核对配置可运行性、工作量、warmup 和环境，而不是只比较两次总耗时。

## 验证与历史证据

新版不创建 launch/bank/shutdown JSON，不要求写工作目录。运行日志用于确认安装，
原生健康接口用于检查服务；进程退出和 NPU 检查用于确认资源释放。

```bash
PYTHONPATH=src python -m unittest discover -s tests
curl --fail http://127.0.0.1:8000/health
```

`tests/serve_acceptance.py` 是仓库内的实验工具，不随包作为启动入口发布。
它接收 `--output`、`--requests`、`--port`，以及 `--` 后完整的原生服务命令；
命令中的端口/served-model-name 应与测试参数一致。先在外部配置环境和取得租约。

`docs/acceptance.json` 记录的是旧 CLI 的32题 HTTP 验收，保留其原始身份；
不能把它叫作新版入口重新跑过的真权重结果。本轮入口改造的 CPU/安装验证与
历史 graph/KV/模型质量证据分开报告，既不抹去旧结果，也不新增性能收益声明。


## DP8 稳定 decode continuation（独立于 TP8 组合）

同一个 `strengthen_dsv4.worker.Worker` 根据用户给出的并行配置选择模块，不增加
CLI、私有环境变量或启动后的激活 RPC。DP8 路线是 TP1 native DSA + EP8，
每 rank 两席位（全局16）、本地配置 token 预算1026、context上限16384、K5、
FULL target、prefix caching关闭；DSACP必须关闭，DSpark仍使用原生 eager。
不把 TP-only 的 split-draft 偷塞进 DP，也不恢复无稳定收益的 worker-retirement。

在你原本能正常启动的 DP8 donor 环境中，以下是本轮验收形状的原生参数示例：

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 \
  --max-num-seqs 2 --max-num-batched-tokens 1026 --max-model-len 16384 \
  --kv-cache-memory-bytes 8589934592 --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,132,264,516,1026],"max_cudagraph_capture_size":1026}' \
  --additional-config '{"enable_dsa_cp":false,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}'
```

KV预算由用户管理，8GiB是测试配置，不是补丁自行保留的份额。
输入/输出所有权、hook位置、fallback和证据口径见
[`async_decode/README.md`](../src/strengthen_dsv4/patches/async_decode/README.md)。
离线原生 LLM/Worker 验收不等于重新做过 DP HTTP 流量验收；服务路由与 admission
仍由原生 vLLM 决定，不能把 matched-cycle 收益直接写成在线吞吐收益。
