# 让 Qwen 的动态 mixed 请求走 FULL graph，而不是枚举请求组合

这里讲当前 **Qwen27 BF16、TP2、无 MTP** 路径。入口仍是
`betterscale.worker.Worker`；原生配置选择这套覆盖。原生 scheduler、请求入退场、
KV 页分配、sampling 与结果 D2H 都保留，不是另起一个执行器。

## 原来一轮请求怎么走

先以固定 donor 的 [NPUModelRunner](../../../../upstream/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
为底图。下面省略无关分支，箭头表示调用关系，不表示所有 GPU 工作同步完成：

```text
原生 scheduler 给出本轮请求与 token 配额
  → execute_model
      → _prepare_inputs                         准备 token、位置、slot mapping 等
      → _determine_batch_execution_and_padding  选执行模式、padding 和图描述符
      → _build_attention_metadata               为各 attention group 构造 metadata
      → _model_forward
          → 原生模型 / ACLGraphWrapper           eager 执行或捕获、重放图
          → 原生 FULL attention 参数更新路径     按图内任务更新本轮 attention 参数
  → 原生 sample_tokens / 输出回收                不在本补丁捕获范围内
```

在当前非 ENPU donor 分支，`_model_forward` 的 Python 调用顺序是先 `run_model()`，
后 `_update_full_graph_params_if_needed()`。图内事件和 update stream 协议协调
设备实际执行；不能把它画成“所有 attention 参数先更新完，模型才提交”，也不能
把 Python 返回顺序当成 GPU 完成顺序。FIA 部分见[对应调用边界](../qwen_fia/README.md)。

模型内部又有一层原生调用链。本检查点是 48 层 GDN、16 层 full attention；
[GDN 原生实现](../../../../upstream/vllm-ascend/vllm_ascend/ops/gdn.py)的外层负责：

```text
输入投影：QKV、gate 等
  → GDN core（卷积、归一化 / gate、recurrent 或 chunk 计算、状态更新）
  → gated norm
  → 输出投影
```

原生 prefill 分支会按请求索引取出 recurrent state，转置成 chunk 算子的布局，
执行后再转回并写回状态池；混合输入还会拆分 decode/prefill 部分并拼合结果。
原生 builder 则提供这些分支需要的请求长度、状态索引和 chunk 信息。

## 原来的限制在哪里

请求数、各请求长度和 prefill/decode 组成都会变。若沿用需要 host 分支和临时
状态整理的计算路径，仅增加几个 graph bucket，并不会自动得到能服务任意合法
混合分区的图。枚举“几个 decode 加几个多长的 prefill”也会让图目录不断膨胀。

我们的改变是：图按 **token 容量 × metadata bank** 捕获；请求分区与状态位置
变成图读取的 metadata 内容。算子直接在持久 K-V 状态池上工作，不再为每波
prefill gather 一份 recurrent state、转置后计算、再 scatter 回去。
这不等于消除所有中间张量布局转换；`execution.py` 中 Q/K/W/U 的整理仍存在。

## 我们 hook 在哪里，原生还负责什么

安装入口在 [__init__.py](./__init__.py)，顺序是 metadata → execution → publication。
[Qwen 装配](../../models/qwen.py)在原生 Worker 初始化前安装 GDN，初始化后安装 FIA，
模型加载后整理不可变卷积权重。以下覆盖只对已通过准入的进程安装：

| 原生接缝 | 我们接入的实现 | 保留的原生行为 |
|---|---|---|
| `AscendGDNAttentionMetadataBuilder.build` / `build_for_cudagraph_capture` | [metadata.py](./metadata.py)：按真实请求长度、状态槽位生成容量稳定的 metadata；声明本路径的 graph 支持 | scheduler 的请求与配额、缓存页管理；不是伪造所有模型都支持 FULL |
| donor Qwen GDN 类的 `_forward_core` | [execution.py](./execution.py)：卷积 → 融合预处理 → owned recurrent 或 chunk H/O，统一 K-V 状态解释 | 外层输入投影、gated norm、输出投影；不是替换整个模型 |
| `CudagraphDispatcher._create_padded_batch_descriptor` / `initialize_cudagraph_keys` | [publication.py](./publication.py)结合 [graphs.py](./graphs.py)：容量描述符加 bank 身份，并生成另一 bank 的 keys | 原生 dispatcher、原生 capture/replay 框架 |
| `NPUModelRunner._determine_batch_execution_and_padding` | 选当前 bank，prefill 容量向上取整，再调用原生方法 | 原生执行模式和 padding 计算的其余部分 |
| `NPUModelRunner._build_attention_metadata` | 给 GDN builder 当前 frame 与 CPU block table；原生 build 返回后统一发布 GDN slab | 原生 attention-group 遍历及普通 attention metadata 构造 |
| `NPUModelRunner._warmup_and_capture` / `_dummy_run` | 标记捕获 bank / dummy 范围，退出时清除标记 | 原生 warmup、模型执行与图捕获主体 |
| `NPUModelRunner._model_forward` | publication 拥有唯一覆盖入口；显式调用 FIA wave 回调，并圈定 GDN 图资源与 consumed fence | 原生模型 forward、原生输入、输出及后续 sampler |

`execution.install()` 先导入 donor 自己的 Qwen patch，再覆盖 `_forward_core`，
避免原生模块稍后导入时把我们的实现盖回去。这仍是进程级类方法覆盖，不是修改
安装目录源码，也不是已经获得了支持任意模型混装的局部插件接口。

## 一波 mixed 请求具体怎么经过这些接缝

1. 原生 runner 准备当前请求；我们选择足够大的 token 容量和当前 bank。
2. 在原生 metadata 构造调用外，publication 取得该 `(capacity, bank)` 的 frame。
   GDN builder 填入 query 边界、状态槽位、cold/warm 标志和 chunk 索引。
3. 所有 GDN group 的 metadata 合成一块 pinned slab，在 ingress stream 上 H2D；
   compute stream 等上传事件。FIA 另有自己的 slab，复用 ingress stream，
   每 wave 运行一次 native planner，并发布本轮 attention 参数。
4. 在 `_model_forward` 内按 **FIA wave 准备 → 当前 bank 的 GDN 图资源 →
   原生 forward → GDN consumed 标记 → FIA release** 的顺序运行。
   命中图时重放图，图内 GDN/FIA 读取稳定地址上的新 metadata。
5. 原生 sampler、输出对象和 D2H 继续收尾。我们不接管请求完成和 KV 页退休。

图捕获之外也必须使用相同的 owned K-V GDN；不能因为这轮没命中图，就把已经
写成 K-V 的状态交给原生 V-K 实现。APC align 的缓存页与状态复制仍由原生处理，
我们用相同的 block-table 列选择规则找到本轮可写目标，不修改共享前缀快照。

## 两个 bank 不是两份模型进度

| 存储 | 谁写 / 谁读 | 重用条件 |
|---|---|---|
| GDN/FIA pinned metadata slab | host / ingress DMA | host 重写前确认该 slab 上次 uploaded 完成 |
| GDN/FIA device metadata bank | ingress / graph | ingress 覆盖前等该 bank 的 consumed；compute 使用前等 uploaded |
| bank 对应 graph 与图参数 | 原生 capture / 串行 replay | descriptor 区分 bank；图资源在 forward 范围内选择并恢复 |
| GDN recurrent state、conv state、attention KV | 原生缓存管理 + 有序模型计算 | 一套请求状态池，按请求槽位更新；不复制成两个独立进度 |
| 临时 H/V、workspace、输出中间量 | 算子 / 后续消费者 | 交给原生 graph pool 分配和复用，不为两 bank 常驻复制 scratch |
| sampler 最终结果与 D2H | 原生 sampler / 原生输出回收 | 继续遵守 donor 原有输出所有权，不另造输出双槽 |

所以这里的 overlap 能力来自明确的读写依赖，不是仅加 `non_blocking=True`。
当前仍是串行 compute stream；不是全链路 N+2 scheduler，也没有捕获 sampling。

## 准入、部署和证据

统一入口后的[新验收记录](../../../../docs/evidence/worker-unification.json)：
88 项 CPU 测试；hw3 TP2 的 68 个 rank-step、8,772 项 hidden/cache 对照全过，
max_abs 为 0；冷/热 APC 和八路共享前缀通过。FULL/NONE 两侧都使用 owned K-V
数值实现，不是对原生 V-K 路径的位级等价证明，也不是新吞吐成绩。

下面保留具体容量、原生库构建与历史实验边界，不能混作同一轮测量。

## Contract

- Fresh process and newly allocated state pools; no reinterpretation of a live
  native V-K pool. Both chunk/mixed and one-token decode read/write K-V FP32.
- Qwen27 BF16, TP2/DP1/PP1, qk8/v24 local heads, K/V128, convolution width4.
  Eight seats, token budget2048, FULL capacities1/2/4/8 for decode and
  16/32/64/128/256/512/1024/1536/2048 for prefill/mixed.
- No MTP, cache transfer, LoRA or context parallelism. APC uses native `align`
  mode; `all` mode is unsupported. Uncaptured model
  execution uses the same owned K-V operators, NEVER native V-K GDN fallback.
  To revert, restart with the native vLLM-Ascend Worker (without BetterScale)
  and a fresh pool. The old Qwen import aliases are not a native fallback.
- `TASK_QUEUE_ENABLE=0` is mandatory for the generated raw ACL launcher. This
  is distinct from vLLM's asynchronous request scheduler, which stays enabled.
- `BETTERSCALE_GDN_LIBRARY` must name the qualified Ascend910B2 library identified
  by `native.json`. The content check rejects old ABI / owned-init-OFF binaries.
  `BETTERSCALE_GDN_HOST_LIBRARY` is also required and content-checked: it names
  the framework host adapter, not a second kernel implementation.
  Library source/build instructions are in the repository's
  `prototypes/qwen38-serving/ascendc_gdn/README.md`; pool mode MUST use
  `-DBS_GDN_OWNED_INIT=ON`. Native binaries are not committed to Git.

## Ownership and capacity

Graphs are keyed by token capacity and alternating bank (26 entries), not request-length partitions. Metadata has
up to eight positive-length requests followed by empty rows and a permanent
empty sentinel. Logical cu endpoints, request slots/cold flags and chunk-index
rows change before replay. Unused chunk tasks target the sentinel. A one-token
prefill tail is not classified as decode merely because its length is one.

The native scheduler/allocator owns exclusive active state slots. Whole-row
Mamba state copies are layout-opaque; convolution caches retain native layout.
The owned core receives real convolution endpoints and does not extend a real
request into padded token space. H/O physical head/chunk strides use capacity,
not the current logical token/chunk total.

Metadata banks own stable device buffers and immutable H/O tiling PODs,
initialized before capture. They do **not** own workspace/H/V/output arenas.
The native `host.cpp` operator queries the qualified workspace size (~22MiB,
including the reserved system prefix) and allocates invocation-local tensors
through the framework allocator. During capture these allocations use the
runner's existing shared graph pool. H/V/workspace die after their consuming
launches; output survives its downstream consumer. Captured addresses are
retained/reused by the graph allocator, not by manual cross-bank aliasing.
Submission remains the pinned single-runner serial stream protocol; concurrent
replay on multiple compute streams is unsupported. Metadata's upload/consume
fences and persistent state ownership are unchanged.

All GDN groups publish one packed pinned slab per wave on a separate ingress
stream. Uploaded events protect host-slab reuse; consumed events protect the
old device reader before a bank is overwritten. Compute waits uploaded on device.
CPU block-table selection and CPU sequence/query lengths produce slots, cold flags,
all dtype variants and chunk tables without per-field GPU copies or Sub/Gt ops.
APC-off selects column0; APC `align` selects `max((seq_len-1)//block_size,0)`
per request, matching the donor. Native preprocessing copies a cached/previous
whole state into that destination before forward; shared snapshots are not
modified in place. Empty large-block triangular-solve tasks
skip the recurrence on device; the active arithmetic and donor merge stay intact.

FIA uses the sibling `qwen_fia` wave-shared native planner and banked metadata
publication; per-layer native task updates are bypassed only inside this owned
FULL path. See its README for the qualified native-library/preload boundary. Native model input publication, sampling, D2H, scheduling and
KV retirement are unchanged. This is not a full N+2 executor or sampling capture.
The raw-ACL TASK_QUEUE_ENABLE=0 restriction remains. Pinned H2D safety must not
be weakened to only non_blocking=True without both reuse fences.

## Deployment

After sourcing CANN and activating the pinned donor environment, expose this
checkout's `src` on `PYTHONPATH`. Set `QWEN_MODEL_PATH`,
`BETTERSCALE_GDN_LIBRARY`, `BETTERSCALE_GDN_HOST_LIBRARY`,
`BETTERSCALE_FIA_LIBRARY`, `ASCEND_RT_VISIBLE_DEVICES` (an admitted pair), and a
dedicated `VLLM_CACHE_ROOT`, then invoke the colocated `serve.sh`. It binds only
127.0.0.1:32181 by default (`SERVING_PORT` overrides the port), uses6GiB KV,
and does not enable diagnostic RPCs or profile hooks. The launcher deliberately
does not claim or acquire shared-host leases: use the host's admission protocol.

This launcher sets `HCCL_OP_EXPANSION_MODE=AIV` before starting vLLM, enabling
native device-side HCCL collectives for this Qwen mixed configuration. Direct
`MixedWorker` deployments should export the same variable before process startup;
changing it after communicator creation or graph capture is not supported here.
The DSV4 and native-Qwen launch paths are unchanged. TP2 BF16 10/20KiB isolated
graph tests show substantially lower allreduce latency, including alternating
banks and rank-skew correctness checks. The subsequent matched APC-on SWE
study uses AIV on both native and candidate; it does not isolate AIV's
whole-model performance contribution. It does not install a custom
communicator or change the existing PG stream protocol.

### Native host adapter build

`host.cpp` and `build_host.py` ship as mod source. Build against the unchanged
pinned Torch/Torch-NPU environment (CPU-only compilation):

```bash
python -m betterscale.patches.qwen_gdn.build_host /absolute/path/to/host.cpp /absolute/path/to/libbs_gdn_host.so
```

The resulting artifact must match the qualified digest in `native.json`;
different toolchains/RPATHs can produce different binaries and need explicit
qualification, not disabling the gate. No runtime JIT build, global donor-op
replacement or new GE path is involved. The old raw non-pool entry is retained
only for standalone historical operator probes; MixedWorker cannot silently
fall back to bank-owned scratch.

## Service qualification (2026-09-17)

`elastic-service9` uses the production MixedWorker execution with diagnostic
FULL-versus-NONE shadows, a 1GiB KV pool and 32 output tokens. Ten single-request
lengths (1,7,17,129,512,513,1024,1536,2048,2051) and C4/C8 cohorts pass.
Twenty-two steps on each of two ranks check valid hidden output and all128 cache
tensors: 5,676 comparisons, all max_abs0. Actual changing mixed partitions include
[1,1,512,513,17] and [1,1,1,1,18], plus transitions to pure decode. C8 denotes
submitted concurrency, not eight active requests in every observed step.

The standalone `elastic-core5` covers up to eight active requests, changing slots,
finite and NaN padding. All12 graph/NONE cases are exact and every initial H tile
matches its warm seed or cold zero. That extra invariant matters: two execution
modes can share the same bug. This qualification is not a model-quality benchmark
or a guarantee about long-generation drift versus native recurrent arithmetic.

Use only the cold-fill-DMA-fenced library in `native.json` (build6, kernel source
4e21bb1). Earlier build4 passed four-request operator checks but has a cold/warm
buffer-reuse race exposed by five/eight-request mixtures; it is not service safe.
See the repository operator README for the reproducer and synchronization fix.


## End-to-end comparison

hw3 `elastic-ab1`: same cards6/7, TP2/noMTP/APCoff,6GiB KV,8 seats,2048 token
budget,64 generated tokens; warmup then two cohorts/case. Native server followed
by candidate (AB, not ABBA). Native queue1 versus owned queue0 is part of the
service comparison. No profiler active. Pooled output tokens/s:

| workload | native | MixedWorker | change |
| --- | ---: | ---: | ---: |
| C1,512 prompt |26.655|28.783|+7.98%|
| C1,1024 prompt |26.600|27.411|+3.05%|
| C1,2048 prompt |25.686|25.179|-1.97%|
| C4,mixed lengths |64.513|70.546|+9.35%|
| C8,mixed lengths |96.149|104.167|+8.34%|

C4 mean TTFT1104→895ms, C8 1889→1590ms. Single-request mean output gaps get
slightly worse (~31.8–32.1→32.5–33.0ms); prefill savings do not offset this at
C1/2048. Keep this as an opt-in configuration, not a universal faster default.
Two warmed samples establish bounded behavior, not statistical certainty.
Compact receipts and all round metrics: `docs/evidence/qwen-mixed-full.json`
in the repository. This source integration has not been published to PyPI.


### Dual-bank SWE acceptance

`dualbank-service2` repeats the full shadow envelope above with alternating
banks:5,676 comparisons max_abs0 plus independent host/device checks of every
GDN metadata field. `dualbank-core1` retains the12 core/initial-H checks. Capture
uses26 graphs and5.25GiB per rank in the diagnostic service (previous13-graph
path about2.9GiB). This historical scratch duplication is removed by the
framework-pool follow-up below; it is not an intrinsic double-graph cost.

`dualbank-swe1` (source7115858), same hw3 cards6/7, sequential ABBA, two full
8-session/78-call SWE cohorts per concurrency and arm,6GiB KV, no MTP/APC:

| concurrency | native tokens/s | dual-bank MixedWorker | change |
| --- | ---: | ---: | ---: |
| 4 |70.021|75.092|+7.24%|
| 8 |96.774|104.409|+7.89%|

Mean TTFT1389→1205ms /1534→1349ms; mean TPOT48.01→44.96ms /
63.05→58.03ms. Six-step separate profiles show steady model-to-model gaps
about1.49ms, down from the prior candidate's4.8ms and near fresh native1.52ms.
These are selected <=8K original-history traces, fixed recorded output budgets,
no tool latency or accuracy claim; two matched repeats are not confidence bounds.
The old C1 fixed-prompt regression above was not remeasured or claimed fixed.
See `docs/evidence/qwen-dualbank.json` and
`prototypes/qwen38-serving/DUALBANK-RESULTS.zh-CN.md` for provenance and timelines.

### Framework-owned scratch follow-up

`graph-scratch-core2`: all12 graph/NONE output, convolution, whole-state-pool
and independent initial-H cases pass exactly. The H oracle retains the actual
captured H tensor; reading the next eager invocation's H is not a valid graph
check. `graph-scratch-service1`: same26-graph TP2 shadow envelope, all comparisons
pass and server exits0. Logged capture delta falls **5.25→0.88GiB per rank**
(4.37GiB reclaimed), measured with the same1GiB diagnostic KV setting. This
quantity is capture-time device-memory delta, not a separate descriptor-only
allocation statistic. Persistent K-V state and both metadata banks remain.


`graph-scratch-swe1`, runtime e5460c0, two candidate-only repetitions on hw3
cards6/7, same original-history SWE fixture/settings and6GiB KV, no profiler:
C4 **75.049tok/s**, C8 **104.440tok/s**. Relative to the retained dual-bank
candidate these are−0.058%/+0.029% (effectively unchanged in this observation).
Retained native controls give+7.18%/+7.92%, **not a fresh paired comparison**.
All78calls/20,648 outputs per cohort match their prescribed prompt/output budgets;
both services exit0 and cards are reclaimed. CPU77tests and wheel source-content
checks pass. See `docs/evidence/qwen-graph-pool.json`. This remains an opt-in
source configuration; no PyPI publication is implied.

### GDN inter-projection fusion

MixedWorker now fuses convolution-output unpacking, Q/K normalization, V packing
and gating into one layout-aware Triton kernel. BF16 normalized Q/K and beta
rounding boundaries are preserved. The mixed path shares head-major beta/g
between KKT, WY and H/O instead of repeating three gate-layout conversions.
The native recurrence library, state layout, graph pool and metadata bank fences
are unchanged. No new service flag is required within this opt-in Worker.

`gdn-fusion-core1` compares with the frozen pre-fusion core:12 mixed and12 decode
cases, changing slots/partitions/inputs and poisoned padding, all output and
whole-state comparisons exact. `gdn-fusion-service1` repeats the26-graph TP2
shadow envelope:5,676 comparisons across22 steps/rank, all max_abs0; logged
capture delta0.84GiB/rank with1GiB diagnostic KV. CPU77tests pass.
See `docs/evidence/qwen-gdn-fusion.json` for bounded microbenchmarks and evidence;
these isolated kernel savings are not a service-throughput claim.

`gdn-fusion-swe1` (runtime d803579), two warmed same-pair candidate-only SWE
cohorts on hw3 6/7, unchanged fixture/settings:

| concurrency | previous candidate | fused candidate | increment |
| --- | ---: | ---: | ---: |
| 4 |75.049tok/s|78.527tok/s|+4.63%|
| 8 |104.440tok/s|108.429tok/s|+3.82%|

Against retained native70.021/96.774tok/s these are+12.15%/+12.04%; controls
were **not rerun**, so this is not a fresh paired comparison. Mean TPOT improves
45.00→42.93ms /57.95→55.48ms. All78calls/20,648 outputs per cohort obey the
fixture budgets. Two service exits and admission are0; cards reclaimed.

A separate six-step/rank capture, processed by TraceLoom c2a6920, recovers all
six exact graph bodies (14,593 members/rank). Every replay replaces48ConcatD +
96QK norm +48gating launches with48preprocessing launches. The1536-capacity mixed
body has336 rather than480transposes. The retained and new profiles have the
same observed partitions/capacities/bank sequence; rank0 GDN conv-to-out-projection
spans sum2.89→1.45ms for decode1,3.66→2.01ms for decode2,68.34→63.38ms for mixed.
These are diagnostic spans, not additive HTTP savings or proof of causality for
other kernel changes. Step gaps remain about1.3–1.5ms: this optimization removes
in-graph work, not another scheduling gap. Both native Perfetto exports preserve
exact member and internal structure geometry. No new NPU run is needed to view
or re-export them.

## Align-mode prefix caching qualification (September 18)

`apc-align-service3` on hw3 TP2 cards6/7, BF16, no MTP, AIV, 1GiB KV:
68 graph/NONE steps, 8,772 hidden/cache comparisons, maximum absolute error0.
Three cold/reused pairs (1537/2051/3073 prompt tokens) reused1536/1536/3072
tokens and produced identical greedy eight-token output. Eight concurrent
branches sharing a cached prefix each reused1536 tokens and matched their
independent cold output. Ten short/long prompt lengths plus C4/C8 service
cohorts also passed. This is a bounded correctness witness, not bitwise
equivalence for arbitrary batching, an accuracy result, or an APC speedup claim.

The native state/page layout sets cache blocks to1536 tokens. A513-token prompt
cannot witness reuse; `apc-align-service2` passed its state checks but was rejected
for that incorrect probe expectation. Keep these receipts separate. Reproduce
with the existing admitted elastic probe and `ELASTIC_APC=1 ELASTIC_SHADOW=1`;
The capsule's `elastic_probe.py` owns the hit/branch checks. The production launcher now enables
APC align. These are the pre-consolidation measurements; the unified no-MTP
entry selects the same owned state protocol for both APC align and APC off.
