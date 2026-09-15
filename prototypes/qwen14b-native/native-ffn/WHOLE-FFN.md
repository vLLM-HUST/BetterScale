# Whole FFN fusion: compiled, correct, not adopted

Fletcher authorized this prototype and then explicitly allowed single-card hw3
execution on2026-09-15 because local cards were busy. This was a task-specific
exception to the earlier local-only single-card preference. No8-card jobs,
real weights or other processes were used/stopped. Remote0 was admitted under
/home/jingyuan/tp8.lock, same health/ownership supervisor as local.

## Implementation and resource lifetime

`whole_ffn.cpp` includes the accepted gate/up+SwiGLU source unchanged and adds
one MIX_AIC_1_2 entry point. Each row chunk completes all SwiGLU channels, then
executes full-K13824 down GEMMs, output H5120. Chunks512/1024/4096 are runtime
attrs frozen per graph; rows1..4096. Down uses native MatmulImpl128x256x64,
L1 step4/depth8, two2560-channel weight panels with diagonal tile assignment.
Both weight matrices are BF16 NZ. No split-K partial sums/atomics/quantization.

A Gate phase has its own TPipe/MatmulImpl scope, drains paired credits, End,
PIPE_ALL and a cross-core completion barrier. Its pipe is destroyed before the
Down phase creates another, reusing the on-chip resource pools rather than
allocating both GEMMs simultaneously. Down completes and synchronizes before
the next chunk. This is row-block PHASED execution: it does not claim overlapping
gate GEMM for chunkB with down GEMM for chunkA on the same Cube cores. Inside
Gate the existing Cube/SwiGLU micro-pipeline remains operational.

Caller owns persistent X, NZ gate/up weights, NZ down weights, BF16 Z[M,I],
scratch (existing slab256 allocation sufficient for the6MiB paired ring), and
BF16 Y[M,H], no aliases. Z still lands in HBM. No memory-peak savings claimed.
Graph addresses remain stable, inputs may change per replay. All row tails mask
actual M. Down tile assignment bijection was checked for1..32 row tiles.

Build with BUILD_SOURCE=whole_ffn.cpp, NATIVE_CM=256 NATIVE_CN=128 NATIVE_CK=64
NATIVE_V3=1 NATIVE_PAIR=1 NATIVE_PANEL=1, default2slots/full-buffer off. Same
library exports both old launch_native_ffn and new launch_whole_ffn. Standalone
prototype only, not a Torch schema or installed package.

## Acceptance

Local first attempt started after a window appeared but failed in the harness:
flat Z was passed to torch.mm before the whole fused kernel ran. Corrected Z/Y
views, retained failed artifact, and migrated at Fletcher's instruction. No local
queue or job remained. Binary identity matched across SSH transfer.

hw3 CANN9.0.1, torch2.10.0+cpu/torch-npu2.10.0.post2, physical0.
Rows128/257/769/1025/512/4096, chunks512/1024/4096, three changed-input FULL
replays, output differential (max_abs<=.0625,RMS<=.005) and Z/scratch/Y guards
passed. First v11 had4 timing controls; v12 added the missing fair control:
accepted panel fusion + native NZ down. v12 reused the same tested binary and
reran only512/4096 numerical/performance checks. Input weight immutability was
not separately snapshot-tested in the new harness; kernel reads weight buffers.
This is not true-weight model quality or serving acceptance.

Six alternating-order trials, ten FULL replays per graph, all controls on hw3:

|M|chunk|native allND|native bothNZ|panel+NDdown|panel+NZdown|whole fused|
|---:|---:|---:|---:|---:|---:|---:|
|512|512|1.039|0.824|0.893|0.846|0.837|
|512|1024|1.040|0.825|0.892|0.847|0.838|
|512|4096|1.038|0.824|0.892|0.847|0.838|
|4096|512|6.033|5.996|5.835|5.711|6.555|
|4096|1024|6.029|5.978|5.873|5.707|6.000|
|4096|4096|6.021|5.996|5.860|5.784|5.975|

Units ms, whole FFN including down, excluding residual add. Weight conversion
happens before capture, not per-step. Do not compare these host-specific times
as regressions/improvements against the earlier local7 hardware cohort.

Decision: keep down separate.1024-chunk whole fusion is~5.1% slower than the fair
panel+NZdown control;512chunks much worse. At512rows tiny whole-fusion advantage
vs split panel is~1%, but native bothNZ is still faster. The simplest observed
benefit is preformatting down weight to NZ, not adding whole-fusion complexity.
That finding is a prototype configuration result, not a product integration.
No claim that whole fusion is universally impossible; this phased design does
not justify adoption. Need a genuinely different scheduling hypothesis before
repeating, not a wider blind chunk sweep.

## Reproduction/evidence

Local run root `/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/`:
`build-v11-whole`, failed`whole-v11-local7`, `whole-v11-hw3-source`,
`whole-v11-hw3`, `whole-v12-hw3-source`, `whole-v12-hw3`.
Remote root `/workspace/my-ascend-workspace/runs/qwen-native-ffn/`:
`20260915-whole-v11-hw3`, `20260915-whole-v12-hw3` contain exact source, binary,
admission, outputs and release receipts. Source capsules freeze runtime/helper
paths; `whole_probe.py --library ... --output ...` runs full matrix,
`--performance-only` selects512/4096 after an unchanged binary's smoke passed.
All task NPUs released. Existing source branch remains isolated; no main/package
or production-default change.
