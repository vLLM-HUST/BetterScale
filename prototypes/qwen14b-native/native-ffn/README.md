# Native BF16 fork of grouped_matmul_swiglu_quant

Independent branch `lumi/qwen-native-ffn`; immutable upstream backup in `upstream/`.
No Triton, no donor installation changes, no HW3 single-card work. CANN9.0.1,
local910B2. `kernel.cpp` adapts native MatmulImpl static-MDL GEMM and the original
split-workspace two-slot producer/consumer SyncAll protocol. Integer routing,
scales and quantization are removed. Vector owns paired channel tiles; native
BF16 down GEMM remains separate. Preserve file-level CANN license and attribution.

## Contract before acceptance

Fixed H5120,I13824, dense BF16 X[M,H], NZ BF16 B[H,2I] representing original
half-concatenated gate/up weights, BF16 Y[M,I]. M1..4096, slab128..4096 in128-row
multiples. Two BF16 scratch slots, each slab*2I elements. No aliases; X/B immutable;
Y/scratch caller-owned and live until completion. Full graph retains all addresses.
Shape/launch attrs are static per capture; input content may change between replay.
No physical state, shared memory, multiple ranks or expert routing involved.

Cube:24 cores, basic/singleM128,N256, fullK5120 through original native MatmulImpl;
BF16-specific K-step2/depth4 initial L1 configuration, not the INT8 depth8.
Vector:48 lanes,8 rows ×256/512 paired channels, <=96KiB explicit queues/scratch.
Each result tile has one writer. All row tails are masked via valid DMA row count;
I is divisible by both channel tiles. All DMA byte/stride fields fit32 bits;
GM offsets use64 bits. Output/slab state cannot be reused before consumer release.
Slab publication and reuse barriers are preserved exactly in role/order from the
native two-slot loop. Vector output carries BF16 gate/up rounding, native FP32
SwiGLU and BF16 final rounding. No clamp or quant scale.

First gate: compile isolated shared library, dummy differential against
`npu_swiglu(mm(X,original_weight.T))`, guard/input checks, >2-slot lifetime cases,
changed-input FULL replay. Then crossed device-event native-vs-fused whole-FFN
measurements and allocated-peak accounting. Compilation alone is not acceptance.
The ctypes launcher is prototype-only, not a public Torch/FakeTensor schema.

## Local dummy acceptance, 2026-09-15

Immutable native backup and first fork: commit `7c51854`. This is an isolated
research branch, not an installed donor or product-default change.

`probe.py` passed rows128/257/769, both Vector channel widths, guard checks,
immutable inputs and three changed-input FULL replays. The four-slab case checks
reuse beyond both scratch slots. Native differential tolerance is max_abs<=0.0625
and RMS<=0.005; observed eager max<=0.03125, RMS<=0.000040. This is not model
quality acceptance or a bitwise contract.

`bench.py` crosses ND native / NZ native / fused whole FFN, six alternating
trials of ten FULL replays. At4096 rows, slab256/VC512: ND6.095ms,
NZ5.845ms, fused9.001ms. At512 rows: ND1.046ms, NZ0.885ms, fused1.279ms.
Down GEMM is native in all cases. The fork has not beaten the native baseline.
Increasing slab256->2048 or VC256->512 did not materially fix that gap.
Slab256 at4096rows used195MiB combined capture-allocation delta plus explicit
output/scratch; this is NOT total reserved graph-pool memory.

`diagnose.py` removes Vector arithmetic but retains all producer/consumer
barriers (`vc=0`, diagnostic only: Y is not written). It checks the final scratch
slab against native GEMM, full-output correctness, changed-input FULL replay,
and scratch guards. Both M128/N256 and M256/N128 builds passed at4096rows,
slabs256/1024. Within-case crossed medians (ms):

| Cube M/N | slab | native NZ GEMM | fused gate/up+SwiGLU | Cube+protocol only |
|---|---:|---:|---:|---:|
|128/256|256|3.615|6.903|6.577|
|128/256|1024|3.632|7.284|6.237|
|256/128|256|3.617|6.294|6.092|
|256/128|1024|3.634|6.251|6.035|

The M256/N128 full fused result is ~8.8% faster than the initial fork atslab256,
not faster than native GEMM+SwiGLU. Variant cohorts are sequential; inputs are
scaled between cases and native timing stays stable. Cube-only includes slab
protocol overhead: it is not a pure hardware Cube-utilization measurement.
Inference: BF16 GEMM scheduling/tiling is the dominant remaining gap, not output
quantization or Vector channel arithmetic. Do not blindly inherit INT8 tiling.

Build with `BUILD_DIR=/absolute/nonhidden/path bash build.sh`; optional
`NATIVE_CM=256 NATIVE_CN=128` selects the tested alternate Cube configuration.
Default128/256 remains the tail-tested configuration. The alternate has only
4096-row diagnostic acceptance so far. Use the existing local selected-card
lease/admission supervisor; no HW3 single-card runs. `probe.py`, `bench.py` and
`diagnose.py` take `--output`; the first two take `--library`, the diagnostic
accepts `--base` and `--m256` libraries. Sources run against torch-npu2.10.0.post2,
CANN9.0.1, local910B2; pinned native Ascend9bf964c. No Triton.

Compact crossed samples: `RESULTS.json`. Full frozen source capsules, binary
identities, admission/release receipts and logs are under
`/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/`:
`smoke-v3-local1`, `bench-v3-local1`, `diagnose-v4-local1`, and matching
`build-v3`, `build-v4-base`, `build-v4-m256`. All jobs released their local card.

## Pair-local follow-up

See [MICROPIPELINE.md](MICROPIPELINE.md): the goal is SwiGLU micro-pipelining,
not limited workspace alone. NATIVE_PAIR=1 removes slab-wide SyncAll in favor of
paired Cube/Vector credits; measured4K whole FFN7.225->6.528ms, still below native
performance. All switches remain opt-in research controls.

## Best measured paired variant

[PANEL.md](PANEL.md) records native-inspired L2 panel scheduling. NATIVE_PANEL=1
with256x128x64 paired two-slot micro-pipeline improves4K full FFN6.297->5.758ms,
vs matched native NZ5.917ms. L2 hit rises71.7->94.1%; still leaf-only evidence.
