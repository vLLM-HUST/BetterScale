# Owned Qwen GDN service integration

Opt-in entry: `betterscale.qwen_worker.MixedWorker`. This is independent of the
DSV4 Worker and the existing `betterscale.qwen_worker.Worker`. Integration is
under hardware qualification; do not infer whole-service qualification from the
operator microbenchmarks.

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
