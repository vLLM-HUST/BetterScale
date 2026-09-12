# DSV4 FULL prefill/mixed: source audit and dispatch evidence

2026-09-12; vLLM 752a3a5 + Ascend 9bf964c. Source paths below are relative to
`upstream/vllm-ascend/` unless marked vLLM. No runtime changes or NPU launches.

## What the trace establishes

Native per-rank DBs, not distributed clock fits, show intermittent starvation
of the selected compute/communication task coverage. Rank 3 has 632.6ms of
interior gaps >=50us in 3.660s; ranks 1/6 have 708.2/592.8ms; rank 0 51.3ms.

Median linked launch-END to device-task-START: rank0 59.55ms, rank1 2.04ms,
rank2 58.63ms, rank3 3.93ms, rank4 38.70ms, rank5 17.78ms, rank6 10.02ms,
rank7 34.90ms. This measures submission lead, not pure queue residence: device
work/dependencies/collectives can also contribute to it. The pattern supports
unequal ability to stay ahead of device execution, without identifying why.

All ranks issue 5,508 aclrtStreamWaitEvent calls. Rank3 has ONE >1ms call
(14.276ms); every other rank's maximum is <0.05ms. Rank1 is gappier but has
no long wait_event. Therefore the 14ms call is a local stall, not the general
explanation. Its API connection links to a TASK of type848 on stream38, lasting
20ns, near the API return. The provider has no event handle/producer relation;
source attribution to a specific shared-expert event would be guesswork.

The HcPost case is stronger evidence of asynchronous dispatch lag: frontend
scope completes about 1.02ms into a gap, execution-thread launch starts at
5.46ms. Frontend and launch threads differ. This is not simply Python doing
nothing. Other gaps have tasks submitted well before they execute.

Unresolved: CPU descheduling/affinity or contention, runtime dispatch queue
behavior, and profiling perturbation. The profile has no CPU sched-switch
trace. Need a minimal-overhead paired measurement to separate these. Do not
promise all 632.6ms is removable, sum ranks, or blame all event dependencies.

## Existing pieces worth retaining

- `attention/context_parallel/dsa_cp.py:655` builds unified prefill+decode
  request metadata. CP splits token rows, maintains complete KV per rank,
  and restores TP-head layout with fixed-size AllToAll (`:1634`). This is NOT
  a need to introduce variable-size EP dispatch.
- Builder already caches local token metadata across KV groups (`:597`),
  local RoPE (`:693`), CPU views (`:706`), SAS per compression ratio (`:931`),
  and QLI (`:995`). Metadata is already shared by layers in the same group
  (`worker/model_runner_v1.py:2993`). Do not claim per-layer SAS/QLI rebuilding.
- Persistent `req_sas_metadata` / `req_qli_metadata` buffers exist. DSA
  `update_graph_params` is explicitly a no-op (`attention/dsa_v1.py:1565`),
  unlike attention implementations that patch per-layer graph task parameters.
- On this A2 TP8 configuration, MoE selector returns ALLGATHER
  (`ascend_forward_context.py:289`). Routing keeps expert counts on device
  (`ops/fused_moe/token_dispatcher.py:355`), then grouped GEMM/unpermute.
  The ALLTOALL-v CPU split-count route is a different branch, not a required
  blocker for this experiment. Attention layout AllToAll still exists.
- Existing ACLGraph wrapper already provides capture pools, replay, stable
  output handling, and stream capture. Shared-expert overlap already works
  for decode; preserve it rather than serializing MoE to make capture pass.

## Gaps and intended treatment

### 1. Capability and dispatch gates (observed)

DSACP advertises UNIFORM_BATCH (`dsa_cp.py:267`). vLLM
`vllm/config/compilation.py:1337` downgrades FULL mixed to FULL_AND_PIECEWISE
or FULL_DECODE_ONLY when support is not ALWAYS. Merely setting FULL is not
sufficient. The explicit builder `build_for_graph_capture` rejects prefill
(`dsa_cp.py:1025`), but current runner bypasses that helper for DSA and calls
`build` directly (`model_runner_v1.py:2958`); do not misidentify the unused
helper as the actual runtime stop.

Plan: opt-in support only after capture/replay validation; retain old fallback.
Runtime and dummy capture must select the same prefill/mixed branch. Current
uniform decode dummy inputs alone cannot establish this.

### 2. Stable metadata shapes, values and addresses (observed + design)

Python values affect captured execution: `has_prefill`, real token slice,
local token bounds, real request count, and compressed metadata capacity.
`_num_compressor_metadata_rows` is min(T, floor(T/ratio)+R) (`:589`).
`compressor_metadata` allocates from that scalar; `num_reqs_actual` is also a
host scalar op attribute (`csrc/torch_binding.cpp:946`). Changing tensor values
alone cannot change captured scalar attributes or allocation shapes.

Use a small set of fixed (global-token capacity, request capacity, mode) buckets;
TP-local token capacity follows global/TP. Runtime values go into stable
query offsets, sequence lengths, positions, block tables and slot maps. Need
explicit safe inactive/padded rows: no KV corruption, no accidental expert
contributions, no stale outputs. Host request-count scalar must either be
bucket-fixed with demonstrated zero-length masking or gain a device-valued
contract. Do not silently reuse a captured real count for another batch.

The vLLM dispatcher already creates padded token/request descriptors
(`vllm/v1/cudagraph_dispatcher.py:134`); use these rather than a second scheduler.
Audit RoPE proxy materialization and every returned metadata pointer across
back-to-back replay. Persistent base buffers alone do not prove slice/clone
addresses seen by the graph stay current.

### 3. Central preparation is partly present, partly missing

QLI metadata uses device `seq_lens_q.max().item()` / `seq_lens.max().item()`
(`dsa_cp.py:999`) although CPU local maxima are already computed at :724.
These two reads can potentially use those CPU mirrors, with validity checked
under speculative correction/async scheduling. Other `.item()` calls on CPU
are not D2H synchronization. Preparation may remain outside the captured
model graph: the objective is once per wave, not pretending host disappears.

Compressor mapping metadata is still generated at forward consumers (`:1150`,
`:1537`, `:1680`), independent of hidden activations but dependent on state
mapping, RoPE family and compression ratio. The existing
`compressor_metadata_out` binding (`csrc/torch_binding.cpp:976`) can fill fixed
buffers; no new allocation API is required. A candidate is to prepare it once
per truly identical state-layout group and reuse across layers. Do NOT share
solely by compression ratio if block tables or state banks differ. A capture
may instead keep these ops inside the graph initially; lifting them is an
independent optimization, not a mandatory first capture condition.

### 4. Event ordering and mutable-buffer lifetime

`ops/fused_moe/fused_moe.py:728-824` overlaps shared/routed expert execution via
record/wait events and joins shared stream. Retain correct dependencies inside
capture; don't remove waits just because the API trace looks expensive.
The ACLGraph replay wrapper (`compilation/acl_graph.py:253`) synchronizes the
current stream before FULL replay unless its special exemption applies, to
protect update/replay ordering. Thus FULL support alone does not yield LiveInfer-
style continuous host/device pipelining. First preserve this safe boundary;
only later replace it with proven buffer ownership/event sequencing if useful.
Never overwrite metadata for wave N+1 while graph N still consumes it.

### 5. Shape-dependent memory and mixed semantics

Need actual capture/reserved-memory measurements at useful 4K/8K buckets,
including warmups and coexistence with decode graphs and KV. A full graph can
retain different intermediates; no memory saving is established from static
inspection. Dynamic expert routing must fit existing bounded allocations for
skewed routes as well as dummy balanced routes. DSV4 mixed batches also retain
correct speculative masks, positions and state transitions. Sampling / draft
can initially remain outside target-model capture; do not promise an entire
serving iteration in one graph.

## Smallest useful implementation sequence (not yet executed)

1. Fixed-shape FULL target-prefill dummy probe using existing ALLGATHER MoE
   and CP attention, one modest token bucket, matching eager reference.
   Keep real metadata preparation outside capture and existing replay barrier.
2. Replay same capture with changed lengths, prefix offsets, routing, block
   tables and request count; include empty local-query slices, C4/C128 boundary
   crossings, padded seats and back-to-back mixed/verify traffic. Check outputs
   AND cache/state mutations. Fail safely to old path on unsupported shapes.
3. Stabilize metadata capacity/scalars as needed; reuse CPU maxima and existing
   metadata-out bindings. Only then advertise extended capability and connect
   dispatcher. No blanket ALWAYS claim ahead of tests.
4. Measure unprofiled eager vs FULL on identical model states, then bounded
   profiles. Compare wall time, launch density, collective waiting, graph pool
   reservation and end-to-end TTFT/output-gap distributions. An overlap toggle
   is diagnostic only; keep the best verified shared-expert overlap in production.

This is a tractable scoped extension, not a one-line mode flip and not a new
engine. There is sufficient source support to justify the first probe, but no
FULL-prefill success or speedup has yet been measured.
