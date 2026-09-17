# Owned Qwen GDN service integration

Opt-in entry: `betterscale.qwen_worker.MixedWorker`. This is independent of the
DSV4 Worker and the existing `betterscale.qwen_worker.Worker`. The hw3 TP2 service passes the bounded whole-model qualification below; this
is not a claim of arbitrary model or scheduling compatibility.

## Contract

- Fresh process and newly allocated state pools; no reinterpretation of a live
  native V-K pool. Both chunk/mixed and one-token decode read/write K-V FP32.
- Qwen27 BF16, TP2/DP1/PP1, qk8/v24 local heads, K/V128, convolution width4.
  Eight seats, token budget2048, FULL capacities1/2/4/8 for decode and
  16/32/64/128/256/512/1024/1536/2048 for prefill/mixed.
- No MTP, APC, cache transfer, LoRA or context parallelism. Uncaptured model
  execution uses the same owned K-V operators, NEVER native V-K GDN fallback.
  To revert, restart with the original Worker and a fresh pool.
- `TASK_QUEUE_ENABLE=0` is mandatory for the generated raw ACL launcher. This
  is distinct from vLLM's asynchronous request scheduler, which stays enabled.
- `BETTERSCALE_GDN_LIBRARY` must name the qualified Ascend910B2 library identified
  by `native.json`. The content check rejects old ABI / owned-init-OFF binaries.
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

Metadata banks own stable device buffers and one H/O scratch engine per
capacity/bank/group, shared serially across their layer group. They initialize device PODs
before capture. Workspace/output lifetimes cover every graph that references
them. Submission remains the pinned single-runner serial stream protocol;
concurrent model threads/runners sharing these resources are unsupported.

All GDN groups publish one packed pinned slab per wave on a separate ingress
stream. Uploaded events protect host-slab reuse; consumed events protect the
old device reader before a bank is overwritten. Compute waits uploaded on device.
CPU block-table column0 and CPU sequence/query lengths produce slots, cold flags,
all dtype variants and chunk tables without per-field GPU copies or Sub/Gt ops.
Only mamba_cache_mode=none is supported. Empty large-block triangular-solve tasks
skip the recurrence on device; the active arithmetic and donor merge stay intact.

The native FIA parameter-update path has separate resources per bank/capacity
and is retained. Native model input publication, sampling, D2H, scheduling and
KV retirement are unchanged. This is not a full N+2 executor or sampling capture.
The raw-ACL TASK_QUEUE_ENABLE=0 restriction remains. Pinned H2D safety must not
be weakened to only non_blocking=True without both reuse fences.

## Deployment

After sourcing CANN and activating the pinned donor environment, expose this
checkout's `src` on `PYTHONPATH`. Set `QWEN_MODEL_PATH`,
`BETTERSCALE_GDN_LIBRARY`, `ASCEND_RT_VISIBLE_DEVICES` (an admitted pair), and a
dedicated `VLLM_CACHE_ROOT`, then invoke the colocated `serve.sh`. It binds only
127.0.0.1:32181 by default (`SERVING_PORT` overrides the port), uses6GiB KV,
and does not enable diagnostic RPCs or profile hooks. The launcher deliberately
does not claim or acquire shared-host leases: use the host's admission protocol.

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
path about2.9GiB): the lower host gap trades extra graph/scratch memory.

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
