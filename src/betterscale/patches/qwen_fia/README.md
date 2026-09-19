# Wave-shared FIA for Qwen27 mixed FULL

This is the head256 adaptation of `prototypes/owned-wave/fia-plan`, enabled only
by the owned Qwen route of `betterscale.worker.Worker`. No attention arithmetic is replaced.
Native NONE/eager attention remains the numerical oracle. DSV4 and native Qwen MTP2 compositions do not install this patch.

## 原生 FULL FIA 怎么走，我们改了哪一段

先看 [原生 attention backend](../../../../upstream/vllm-ascend/vllm_ascend/attention/attention_v1.py)：
模型里的 full-attention 层最终进入 `AscendAttentionBackendImpl.forward_fused_infer_attention`。
捕获时，它走 native FULL FIA 路径，记录 kernel、任务句柄与图内同步事件；
后续 replay 仍需要为变化的请求长度等信息更新图内任务参数。

这部分由 [runner](../../../../upstream/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
的 `_update_full_graph_params_if_needed` 接入原生 update 路径，backend 中有
`graph_task_update_begin/end` 和对应 native attention 调用。当前非 ENPU 分支
是在提交 `run_model()` 后调用 update；不能把 host 提交顺序误解成 GPU 同步完成。
原生 `ACLGraphWrapper.__call__` 还有保护旧更新协议的 FULL replay host 等待。

问题不是 FIA 的数学不能捕获，而是 **每层都通过 host 更新本轮图任务**。
我们保留 native tiling 决策，把它变成一次 wave 规划，结果通过稳定地址的
device metadata 交给图内 kernel：

| 接缝 | 覆盖后的行为 | 没有改掉的部分 |
|---|---|---|
| GDN publication 持有的 `Runner._model_forward` | 显式调用 [wave.py](./wave.py) 安装的 `_betterscale_fia_forward` 回调；每波准备 planner/frame，退出时释放读者依赖 | publication 仍是唯一 runner forward 覆盖入口；原生模型 forward 保留 |
| `AscendAttentionBackendImpl.forward_fused_infer_attention` | owned FULL 内使用 native planner 提取的 kernel/tiling，通过固定 metadata 地址捕获数值 launch | attention 数学不变；未进入 owned 条件时调用保存的原生方法 |
| `Runner._update_full_graph_params_if_needed` | 仅 `_fia_wave_active` 时跳过逐层 native task update | 其他调用仍走原生方法 |
| `ACLGraphWrapper.__call__` | owned FULL 范围内临时选择 donor 的 caller-ordered replay 分支，退出恢复；启动时 prime 新 bank | 原生图捕获、`NPUGraph.replay()` 和实际 `aclmdlRIExecuteAsync` 提交仍在 |

`static_plan.cpp` / `host_metadata.cpp` 是 native 调用边界适配层，不是另写了一套
attention 算法。必须在 Python 启动前 preload，使 native planner 的数值 launch
能被拦截；晚些时候 `dlopen` 不能补救。planner 仍在 host 上每波运行一次，
不是声称所有 tiling 都搬到了 device。GDN 与 FIA 的整体调用链、双 bank 所有权
见 [GDN README](../qwen_gdn/README.md)。

## Pinned boundary

CANN9.0.1 / Ascend910B, BF16, TP-local Q12/KV2 heads, head256, causal TND,
paged128, no sinks or sliding window, up to eight real requests plus a padding
row and2048 query tokens. The admitted native function,24-block non-FD grid,
2528-byte tiling and descriptor relocation layout are checked, not guessed.
An unexpected variant fails closed. The donor/runtime pins still apply.

Each wave runs the native host planner **once**, suppressing its numerical
launch, and publishes one packed pinned slab: native tiling, query endpoints,
KV lengths and block table. Every full-attention layer reads this bank's device
metadata inside its captured kernel. No per-layer task-group update, FIA host
launch, or GE parameter injection is needed during replay. A padding request
may have positive query capacity but zero KV length; it must not be rejected or
extend a real request. Native host planning is retained, not replaced with an
invented FD/tiling policy.

Frames are per `(capacity, bank)`. Upload completion protects pinned-host reuse;
consumption events protect device overwrite. The existing ingress stream
publishes and compute waits on device. Layer scratch is invocation-local Torch
allocation from the existing graph pool, not duplicated persistent bank scratch.
A single independent Q/output planning fixture is retained per runner; KV fixture
references borrow the model's existing persistent pool. The planner uses a
transient128MiB workspace allocation; its intercepted launch never reads it.
This is a serial single-runner protocol, not concurrent multistream replay.

## Build and deploy

Source the pinned CANN environment, expose this package, then CPU-build:

```bash
python -m betterscale.patches.qwen_fia.build /absolute/path/libbs_fia.so
export BETTERSCALE_FIA_LIBRARY=/absolute/path/libbs_fia.so
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
export TASK_QUEUE_ENABLE=0
```

Set preload **before starting Python**. `qwen_gdn/serve.sh` does this from
`BETTERSCALE_FIA_LIBRARY`. The native digest and process-global symbol identity
must match `native.json`; a late dlopen is not sufficient. Do not install the
interceptor into CANN or globally preload unrelated services. Native binaries
are not in Git. A different toolchain binary requires requalification, not
removal of the gate. Run `test_native.cpp` against CANN headers with `-ldl` for
CPU ABI rejection and bounded persistent-plan refresh checks.

The captured FIA numerical launch and ordinary graph replay are still necessary.
This does not capture sampling or replace the native request scheduler.

## Replay ordering and startup

The pinned ACLGraph wrapper's `enable_enpu` field is scoped only around owned
FULL calls to select its existing caller-ordered replay branch. This does NOT
enable the runner's ENPU path. It removes the CPU stream barrier that existed
for native attention task updates; serial compute ordering and both metadata
reuse fences remain. The wrapper flag is restored even on exceptions; NONE,
other workers and other graph modes keep their original behavior.

Each newly captured bank is replayed and drained once during startup, before
requests are admitted. The gate requires `_owned_capture_bank` and an empty
request pool; it is not live-request state rollback or speculative execution.
Only disposable startup state is touched. Ordinary cold-request initialization
still owns real state/cache rows. This follows LiveInference's principle of
priming unpublished graphs: measured first-use runtime setup is paid at startup,
not hidden in a later request's first encounter with that capacity/bank.

`NPUGraph.replay()` / `aclmdlRIExecuteAsync` remains the actual graph submission.
The cold call is not a steady per-step7ms cost. The retained observations and
qualification capsules are indexed in the repository's Qwen-serving knowledge
scenario; profiler spans must not be presented as fresh HTTP throughput gains.
