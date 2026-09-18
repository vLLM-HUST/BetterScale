# Borrow the fused W8A8 expert pipeline, not just its name

Read when optimizing Qwen38 server ACTIVATE/up GEMM or proposing to embed
GroupedMatmulSwigluQuant in the persistent server. Investigation: 2026-09-18.
The useful distinction is **group-to-core scheduling, intra-core GEMM scheduling,
and Cube-to-Vector readiness**. They are three different mechanisms.

## Source identities and navigation

- vllm-ascend checkout `/workspace/strengthen-dsv4/upstream/vllm-ascend`, commit
  `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`.
  Start at `csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/`:
  `grouped_matmul_swiglu_quant.cpp` selects branches; corresponding `.h` owns
  key0; `grouped_matmul_swiglu_quant_split_ws.h` owns key1. Host tiling is
  in sibling `op_host/grouped_matmul_swiglu_quant_tiling.{cpp,h}`.
- Installed local CANN **9.0.1** source root:
  `/usr/local/Ascend/cann-9.0.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`.
  Its `grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_spilit_fusion.h`
  (spelling intentional) contains `GroupedMatmulDequantSwigluQuantFusion`.
  V2 `.cpp` selects it at key3. This is the more interesting W8A8 C/V pipeline.
- Matmul internals are under CANN
  `aarch64-linux/asc/impl/adv_api/detail/matmul/scheduler/base/`:
  `scheduler_mdl.h`, `scheduler_mdl_common.h`, `scheduler_mdl_base.h`.
  These are explicitly **internal, unstable, not customization interfaces**.
- Our `prototypes/attention-client/qwen38/quant_gmm.cpp` selects CATLASS;
  `device-service/build_actual_gmm.sh` defaults its headers to CANN
  `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/common/catlass/include`.
  Inspect that installed `catlass/gemm/kernel/grouped_matmul_slice_m.hpp`,
  not a nearby LiveInfer mirror whose ending barriers may differ.

These are source observations, not proof of the installed binary's selected
branch. Native v1 and v2 and donor's custom NZ tensor-list operator are distinct.
A profiler name alone does not establish tiling key or compilation provenance.
Upstream files carry Huawei CANN Open Software License Agreement v2.0; do not
silently relicense copied implementation as our Apache code.

## 1. Across experts: carry the tile cursor forward

In v1 `CubeProcess`, each expert produces a grid
`ceil(actual_M / singleM) * ceil(N / singleN)`. Cores consume every core-count-th
block. The next expert starts at the previous grid's remainder modulo core count,
not at core0. Empty groups consume no work, but weight offsets still advance.
`MNBlockIdxCompute` is plain M/N quotient/remainder here; its threshold argument
is not evidence of an active fancy swizzle.

Example: 24 cores, two experts with 10 blocks each. Resetting per expert gives
cores0–9 two blocks and cores10–23 none. Carrying the cursor assigns the second
expert to cores10–19. There is no need for a host task per expert or a barrier
between experts in this assignment loop.

**We already have this property.** Installed CATLASS `GroupedMatmulSliceM` keeps
`startCoreIdx = (startCoreIdx + coreLoops) % coreNum`, reads device cumulative
ends, and uses the actual expert M. Our wrapper selects
`GemmIdentityBlockSwizzle<9,1>` and `MmadAtlasA2PreloadAsync<1,2,2,2,1,false,true>`.
Do not claim that swapping to the native op newly adds actual-count GEMM or
cross-expert tail balancing. Both implementations also disable weight L2 cache
hint when the expert has only one M block.

## 2. Inside a Cube: the MDL scheduler is a memory/compute pipeline

The upstream v1 static MatmulImpl config uses base128x256x128,
stepKa=stepKb=4 and depthA1=depthB1=8. These are this source's choices, not
universally optimal dimensions for our K2560/N1280 workload.

Read `ScheduleOnce -> MoveNext -> ReduceK`:

- M/N iterators choose the next output tile and preserve valid L1 cached data.
- `CopyIn` obtains A/B in L1; `DoPreloadLoad` can issue asynchronous next-M,
  next-N or next-K loads **when its compile-time policy and shape guards allow**.
- `ReduceKOneIter` versus `ReduceKMultiIter` handles whether K fits the L1 tile.
- `ComputeKDB` allocates L0 buffers, loads A2/B2, queues readiness, issues MMAD,
  then retires buffers. Other policy branches support M/N double buffering.
- `DoPreloadAWait` and buffer-cache invalidation enforce reuse lifetimes.

This is not the server's request queue and not a magic global work-stealing
scheduler. Do not infer every optional preload mode is active in a given binary.
Use supported Matmul/CATLASS interfaces rather than depending on MDL internals.
Our CATLASS GEMM already has asynchronous preloading; replacing its matrix loop
alone does not address the observed long vector/coordinator intervals.

## 3. Between Cube and Vector: version matters enormously

### v1 key0: full intermediate, one bulk handoff

Cube writes INT32 `[actual rows,N]` into an allocation sized by input capacity;
AIV waits at `SyncAll<false>()` before consuming it. Hence “fused” does **not**
mean no HBM intermediate, nor per-tile C/V overlap.

Vector partitions contiguous row ranges across 2x Cube cores, processes UB-sized
row batches, and retains channel scales across rows of the same expert. Key0
allocates double queue buffers. It batches DMA rather than repeating a complete
channel-scale fetch for every token. `UpdateVecConfig` resets TPipe: do not paste
it into a persistent worker sharing pipe/event ownership.

### v1 key1: bounded double HBM workspace

Host tiling uses a64MiB user-workspace budget. For W8A8,
`mLimit = floor(32MiB / (4*N))`; key1 is chosen when capacity M exceeds2*mLimit.
Cube and Vector alternate two INT32 windows. Cube can fill window1 while Vector
consumes window0; the extra join protects reuse from chunk2 onward. Expert ranges
are clipped to each chunk, including an expert crossing the chunk boundary.
This is a real C/V pipeline, but chunk-level, not arbitrary token-ready service.
Do not confuse it with `grouped_matmul_swiglu_pipeline.h`: that named pipeline is
an **A8W4 MSD** branch (key2), not the W8A8 key0 path.

### Installed v2 key3: incremental complete-row regions

`GroupedMatmulDequantSwigluQuantFusion` flattens all expert M/N blocks into a
single logical stream. Each Cube advances by `cubeBlockDim`. Device group ends
map blocks back to expert weights and real tails. `cvTimes` and `rsvBlockNum`
account for the N tiles: Vector needs complete row regions, not an arbitrary
single N tile, before applying SwiGLU and rowwise dynamic quantization.

- Cube publishes flag0x8 on **PIPE_FIX**, after its relevant output writes.
- Vector waits for0x8 plus `SyncAll<true>()` before reading the released region.
  Even inactive Vector lanes participate in this synchronization.
- Cube can advance while Vector dequantizes/activates/quantizes earlier regions.
- Every14 publication groups, Cube waits for reverse flag0x9; Vector publishes
  it on PIPE_MTE2. This bounds producer lead; it is **not** a512/14-row buffer
  allocation or a server generation protocol.
- Tail Cube lanes call `FinalizeCubeSync` to supply remaining notifications;
  omitting tail accounting can deadlock a port.
- INT32 output still lives in HBM. Channel scales are cached by group ID;
  activation work is batched over `ubFactorDimx` rows.

This source establishes an available mechanism, **not that our v1 probe used
V2**, nor that the runtime will select key3 for every shape. It also does not
turn this up+activation kernel into a whole up/down persistent expert service.

## What to borrow into our server

Current Qwen38 config runs whole up -> vector ACTIVATE -> whole down -> SEND.
A coordinator and16 movers use a separate persistent Vector launch; Cube has24
blocks. Native mixed kernels assume their own Cube:Vector launch, barriers,
TPipe and event IDs. They cannot be called as a harmless nested Cube function.

Best-supported opportunities, in order:

1. Reuse channel scales and batch several contiguous expert rows per UB/DMA
   operation in ACTIVATE. Preserve gate/value order, clipping, rounding and
   per-token scale semantics. Our scale ABI is8floats/row; native output is1.
2. Transfer the complete-row-region readiness idea: up-region-ready -> vector
   quantization -> down-input-ready. Pair every buffer's reuse with consumer
   completion. Keep task generation/layer/source retirement outside this loop.
3. Consider the supported native mixed op as a bounded compute backend if its
   graph launch/queue integration cost is acceptable; measure before adopting.

Do not blame GEMM without a matching control. Existing coordinator intervals at
4096 routed rows show up95.66us, activation381.60us, down84.97us, send563.61us and
fetch-end-to-pack-start562.80us (see `server-phase-result.json` beside
`prototypes/attention-client/qwen38/PREFILL-PROFILE.md`). Those include task
handoff/join and are not pure kernel durations. The single-coordinator route/map
construction remains an independent bottleneck, untouched by fused SwiGLU.

## Bounded native gate, not a performance claim

hw0,910B2,CANN9.0.1,torch2.10.0+cpu/torch_npu2.10.0.post2; dummy NZ INT8 weights,
171groups,K2560,N1280. Active groups include0,1,169,170 and empty groups.
Reproducer: `prototypes/attention-client/qwen38/probe_native_swiglu.py`.
[Result](native-swiglu-result.json) retains10 samples per arm.

| Allocated rows | Actual routed rows | Median native v1 replay |
|---|---:|---:|
|4096|4096|161.28us|
|4096|3072|131.16us|
|51200|4096|169.48us|
|51200|3072|147.38us|

Changed input and device group ends under the same FULL graph: active output
and scales exactly match eager native execution. Two sampled rows/expert against
independent dequantized SwiGLU have maximum relativeL2 0.0211; this is a loose
quantization sanity gate, **not** bitwise equivalence to our current server,
real-weight model quality, nor full output validation. The native third return
is not consumed by this probe.

No server fetch/pack/down/send, concurrent client, or persistent competition is
included. The different capacity timings are bounded observations, not a
universal capacity-independent guarantee. V2 performance/selection and a fair
same-resource fused-vs-current ACTIVATE comparison remain unmeasured. Probe
exited0 and released its device/lease; no production default was changed.

## First transfer into our implementation

The opt-in `build.py --batch-activate` now implements opportunity1 above in
`prototypes/attention-client/qwen38/quant_batch.hpp`: four rows, contiguous
worker ranges, expert-boundary clipping, cached scales, existing64KiB UB.
Exact old/new quantized outputs and scales passed the standalone dynamic FULL
gate. Real layer0 A1+E3 online leaf time fell about8%; A2+E3 sampled independent
oracle also passed. See the **Four-row ACTIVATE** section in `PREFILL-PROFILE.md`
and its `batch-activate-*-result.json` receipts for conditions and boundaries.
This does not yet implement opportunity2's incremental up/activate/down pipeline;
keep that distinct from the now-measured batched ACTIVATE improvement.
