# BF16 FFN fusion and native FULL graph investigation

The subsequent local single-card BF16 fusion trial is complete: see
[ffn/README.md](ffn/README.md). Persistent tiled fusion bounds relay memory but
remains2.2–2.3x slower than native whole FFN; it was NOT integrated into serving.
The original AscendC adaptation below remains a design, not that implementation.

## Resource boundary

Fletcher's September15 steering: single-card work runs on the **local host**;
leave hw3's whole-eight-card window available. The already-submitted hw3 FULL
configuration probe finished and released its card before this steering was
processed. No new hw3 NPU work was submitted; subsequent analysis is local CPU.

## Actual reusable kernel, not merely its public dtype guard

Pinned Ascend source: `upstream/vllm-ascend` at9bf964c. Inspect
`csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/`:

- `grouped_matmul_swiglu_quant.h:79–86,119,181–183`: INT8 A/B, INT32 GEMM
  intermediate in **GM**, `IterateAll(mmOutGM)`; Vector reloads it in
  `customDataCopyIn` around368, casts, applies scales, then SwiGLU and quantization.
  This is NOT a zero-intermediate-HBM implementation.
- `grouped_matmul_swiglu_quant_split_ws.h:134–136,203–213`: two GM intermediate
  buffers sized by `mLimit`, selected modulo2. Cube waits before reuse, publishes
  completion, Vector consumes, with `SyncAll` coordination. This bounds workspace
  and pipelines chunks; it does NOT eliminate their writes/reads.
- `grouped_matmul_swiglu_quant.cpp:23–35` chooses INT32 GM output matmul type;
  the op definition admits quantized weights/output. Removing quantization means
  changing GEMM types, tiling/layout and Vector epilogue, not only skipping Quant().
- `grouped_matmul_swiglu_pipeline.h` is the A8W4 MSD pipeline, not the simpler
  A8W8 split-workspace route. Do not inherit its unnecessary stages for dense BF16.

### Minimal candidate (design, NOT implemented or measured)

Keep one dense expert and the already-combined gate/up projection. Use BF16
A/B, BF16 intermediate GM ring, FP32 Vector arithmetic as required, BF16 SwiGLU
output; remove routing, dequant scales, max-reduction/quantization and scale output.
Retain native down GEMM consuming the complete `[T,I]` output. Do not trade away
GEMM efficiency merely to put both matmuls under one launch.

Preserve the original BF16 gate/up rounding boundary before activation. Consuming
FP32 accumulators directly changes that boundary and requires a distinct numerical
qualification. Weight layout/tiling must be recalculated for BF16, including NZ
alignment; INT8 tiling is not a drop-in BF16 recipe. Gate/up matching columns must
be delivered together to Vector. Either keep native half-concatenated weight
layout and gather paired tiles, or prepack paired columns once; do not repack on
每次 forward. Completion/slot-reuse fences must protect every chunk, including
tails; static workspace addresses and bounded loops must survive FULL capture.

At Qwen I=13824,T=4096, full BF16 gate/up `[T,2I]` is216MiB. A two-slot
256-row BF16 ring is27MiB (FP32 would be54MiB). The final `[T,I]` is still108MiB.
These are intermediate sizes, NOT total graph-pool savings. A ring reduces peak
and may improve locality/overlap; total intermediate traffic is not automatically
reduced, and L2 residency is not guaranteed. Native large-GEMM throughput is the
control to beat. Compare actual Qwen shapes on local single-card dummy microbench
before integrating anything into serving.

### Why LocalTensor alone does not prove on-chip handoff

Installed CANN9.0.1:
`aarch64-linux/asc/impl/adv_api/detail/matmul/utils/matmul_utils.h:24–38`
selects USE_SSBUF only for architecture3510 (unless disabled); the220-compatible
route is USE_WORKSPACE. In
`aarch64-linux/asc/include/adv_api/matmul/matmul_client.h:1264–1325`,
`GetTensorC(LocalTensor)` asynchronously reads `cacheWorkspaceAddr` via CopyToUB;
the synchronous branch also exchanges a GM address then copies into UB. Thus the
standard A2 mixed-kernel client path is not zero-GM merely because its API accepts
LocalTensor. Other lower-level routes remain unqualified, not declared impossible.

The official GetTensorC API documents UB output and BF16 support:
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0639.html
The implementation above is stronger evidence for this installed machine's path.

### Existing all-FFN operator

Installed `torch_npu/_op_plugin_docs.py:7238–7277` and upstream
https://gitee.com/ascend/cann-ops-adv/blob/master/docs/FFN.md
restrict SwiGLU in `npu_ffn` to non-expert FP16 high-performance mode. BF16 support
for plain silu is NOT support for gated SwiGLU. This only rules out direct use,
not adapting a fused kernel. Do not silently change model dtype or add quantization.

Measured native SwiGLU task totals: C1 prefill12.9825ms/48layers, seven C1 decode
steps1.7138ms/336layers; C8 prefill/mixed11.3442ms/144layers, seven decode steps
3.2455ms/336layers. These are not a fusion speedup ceiling: fusion can also alter
GEMM stores, pipeline overlap and GEMM efficiency. They do show why removing only
the activation launch cannot produce a large decode gain.

## Native FULL configuration check (completed, no model patch)

Qwen's AscendAttentionMetadataBuilder already returns AttentionCGSupport.ALWAYS
(`attention/attention_v1.py:252`). `full_graph_fia` and `update_graph_params`
provide capture and parameter-update paths. FULL_AND_PIECEWISE retains PIECEWISE
for mixed inputs; increasing buckets alone does not turn it into FULL.

The bounded candidate changed only compilation config to:
```json
{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,512,1024,2048,4096]}
```
The same real BF16 model completed both native cohorts with32 generated tokens
per request and separately profiled8-token windows. This is execution/count
acceptance, NOT output/KV equivalence or model quality qualification.

Unprofiled cohort seconds, two repeats:
- C1:1.36419 /1.35127 (prior1.36003 /1.36565), essentially unchanged.
- C8:1.53076 /1.52809 (prior1.46455 /1.48153), about3.8% slower by mean.
These sequential, short cohort samples are not a stable release-speedup claim.

Profiles confirm eight/ten replay calls rather than prior seven/seven. Prefill
is now inside captured models. The C8 large-forward QKV shapes are512,1024,4096,
versus baseline512,1025,2563: scheduler composition and bucket padding differ.
Therefore the graph configuration works, but coarse buckets do not automatically
improve cohort throughput. Do not attribute the whole slowdown to graph overhead.

Evidence:
`/workspace/strengthen-dsv4/runs/qwen14b-native-20260915/hw3-full-single0/`.
Frozen source/launch, native DBs, measurements and complete TraceLoom37323af
`measurements/{c1-4k,c8-512}/analysis/*.json.gz` exports are retained. Card reclaimed.
No production configuration was changed. Further single-card tests use local.
