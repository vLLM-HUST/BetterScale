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

Graphs are keyed by token capacity, not request-length partitions. Metadata has
up to eight positive-length requests followed by empty rows and a permanent
empty sentinel. Logical cu endpoints, request slots/cold flags and chunk-index
rows change before replay. Unused chunk tasks target the sentinel. A one-token
prefill tail is not classified as decode merely because its length is one.

The native scheduler/allocator owns exclusive active state slots. Whole-row
Mamba state copies are layout-opaque; convolution caches retain native layout.
The owned core receives real convolution endpoints and does not extend a real
request into padded token space. H/O physical head/chunk strides use capacity,
not the current logical token/chunk total.

Metadata builders own stable device buffers and one H/O scratch engine per
capacity, shared serially across their layer group. They initialize device PODs
before capture. Workspace/output lifetimes cover every graph that references
them. Submission remains the pinned single-runner serial stream protocol;
concurrent model threads/runners sharing these resources are unsupported.

Metadata publication currently uses blocking pageable-source copies plus ordered
device copies. It makes no pinned-slab lifetime or H2D-overlap claim. The existing
native FIA parameter-update path is retained; this change does not claim to
remove every attention update or capture sampling into the model graph.

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
