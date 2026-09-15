# Native grouped-matmul/SwiGLU quantization cuts (read-only analysis)

Fletcher stopped implementation after the Triton experiment. This note inspects
native AscendC only; no operator edits or NPU jobs accompany it. Source pin:
`upstream/vllm-ascend`9bf964c. Paths below are under
`csrc/gmm/grouped_matmul_swiglu_quant/`.

## Exact arithmetic boundaries

`op_kernel/grouped_matmul_swiglu_quant_split_ws.h`:

1. `MMCompute`320–346 computes INT8 A/B -> INT32 GM through native Matmul
   `IterateAll`. This is the GEMM engine, not an output quantizer.
2. `customDataCopyIn`453–487 loads the INT32 block, casts to FP32, loads a
   per-token scale through scalar GetValue, multiplies it with S/V event ordering.
3. `UpdateChannelScale`494–522 maintains expert/channel scales. `Dequant`535–548
   multiplies channel scale. Together2/3 restore the GEMM result to float.
4. `Swiglu`553–578 calls native FP32 SwiGLU on the two half-row views, optionally
   clamps, then copies the half-width result back into the former input buffer.
5. `Quant`583–617 computes abs, row-wide ReduceMax, V->S handoff, derives the
   row scale, stores it, S->V handoff, rescales the row and casts FP32 to INT8.
6. `customDataCopyOut`624–637 writes both quantized data and per-row scale.

Full-precision A/B removes2/3 and replaces5/6 with output casting/store. Removing
ONLY5/6 can keep INT8 GEMM but must retain2/3; that would isolate output-quant
cost, not implement the requested BF16 model. Do not conflate these two cuts.
BF16 output is twice the bytes of INT8; lower Vector work is not guaranteed to
win whole-kernel latency. BF16 GEMM has its own input bandwidth and tiling costs.
Compare eventually to native BF16, not to INT8 arithmetic throughput.

## The stronger opportunity: remove the whole-row dependency

SwiGLU itself is channel-pair-local. Dynamic per-token output quantization is
not: its single scale requires max over all I output channels. The inspected
kernel consequently distributes full rows to Vector cores, loads paired half
rows, and keeps full-row buffers. Once row quantization is removed, Vector can
consume matching `(gate[j:j+b], up[j:j+b])` tiles without waiting for every
channel to contribute to a maximum. This is a structural opportunity for more
independent Vector work and bounded UB, not a measured speedup.

For Qwen2.5-14B, I=13824 and fused gate/up N=27648. The original
`op_host/grouped_matmul_swiglu_quant_tiling.cpp:82–104` budgets full-row input,
double-buffer channel scales, quantized output and reduction scratch. It logs
that N should not exceed10240; its approximate one-row threshold is
`(UB-72)/19`. At N27648, roughly19N alone is513KiB, beyond the relevant UB
budget. Even deleting scales/quantization does not automatically make every
remaining full-row FP32 buffer fit. The tiling must change, not just the schema.
The source reports a limit; do not claim every unsupported width safely rejects.

A plausible native adaptation keeps Cube's efficient scheduling and uses bounded
Vector row/channel tiles over the native half-concatenated output layout. With
whole-slab readiness retained initially, gate/up pairs can be read from two
locations in the slab; no weight repack is inherently required. If later publishing
per-N-tile readiness, both half-tiles must be complete before activation and no
producer may overwrite either while a reader is active. That is a separate
protocol change, not a free consequence of deleting ReduceMax.

The SwiGLU scratch is not wholly quant-only: `reduceWorkspace` also hosts the
activation output. Remove only the reduction-specific tail/scratch, or reorganize
activation destination so its copy-back can be replaced by direct BF16 output
packing. Preserve exact gate/up order, clamp setting (Qwen: disabled) and BF16
rounding at the gate/up boundary; bypassing that rounding changes the program.

## What actually pipelines today

Common `grouped_matmul_swiglu_quant.h` computes the complete assigned GEMM then
uses SyncAll before Vector consumption. A fused task name does not establish
intra-chunk Cube/Vector overlap.

`grouped_matmul_swiglu_quant_split_ws.h:203–294` instead owns two GM slots:
Cube publishes a completed slab; Vector consumes it; Cube can advance to the
other slot, and later reuse waits until the consumer is done. Important edges:
- Cube completion publication at245, matched Vector wait at260;
- before reuse, Cube waits at213 when loop index>=2;
- Vector's release at290 when future reuse exists.

These SyncAll calls include data/ownership edges. Do NOT delete them as presumed
quantization barriers. The per-token V/S handoffs INSIDE Quant disappear with
row quantization; that is a different synchronization scope.

The host selects split workspace only if `M > 2*mLimit` (tiling.cpp244), where
for A8W8 `mLimit=floor(64MiB /2 /4 /N)` (219–223). Ordinary smaller cases use the
common branch. For example N4096 gives mLimit2048 and the split threshold M>4096.
No measured donor launch's tiling key was recovered in this static inquiry;
do not assert its observed overlap came from the split variant without that key.

The default native Cube configuration in `grouped_matmul_swiglu_quant_utils.h`
uses single-core/basic M128,N256 and basic K128. It calls native Matmul across
full K; these constants do not mean a Triton-style Python-level K loop. Preserve
the native framework, but recalculate BF16/NZ alignment, buffer depths and tile
resource limits instead of reusing INT8 constants blindly.

## Decision, not implementation

The justified route remains native BF16 GEMM + stripped Vector epilogue,
with bounded gate/up channel tiles, retained two-slot ownership, and native down
GEMM initially unchanged. The benefits to investigate are reduced scale traffic,
removed row max/scalar synchronization, lower UB pressure and earlier independent
SwiGLU consumption. Main risks are BF16 GEMM efficiency, twice-sized final output,
paired-tile readiness, too-small slabs reducing weight reuse, and a new Vector
schedule that destroys rather than improves overlap. No performance estimate or
implementation acceptance follows from this source analysis.
