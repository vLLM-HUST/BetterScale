# Pair-local Cube/SwiGLU pipeline

Purpose: expose SwiGLU micro-pipelining, not primarily bound workspace. The
original slab-wide SyncAll is a control, not a desired ownership boundary.

## MatMulV3 backend ablation (v6)

NATIVE_V3=1 switches off inherited VecND2NZ config and uses Iterate/GetTensorC,
as the installed related MatMulV3 base kernel does. It does not establish exact
identity with the dispatched native torch.mm backend. NATIVE_SWIZZLE=1 borrows
its diagonal tile assignment; a CPU bijection check covered1..16 M tiles and
108/216 N tiles. Both builds passed complete smoke/FULL probes.

Matched K64 M256/N128 atslab256: old5.386ms vsV3-call5.381ms; slab1024:
old5.334ms vsV3-call5.343ms. No meaningful gain. Adding diagonal assignment
at1024 worsened to6.102ms. Do not enable that policy by default. The run's legacy
CLI labels k128-base/k128-m256 actually identify v5 K64 m256/m128; RESULTS.json
normalizes them. Frozen artifact labels are preserved, not rewritten.

## Pair-local prototype (v7)

Build flags: NATIVE_CM=256 NATIVE_CN=128 NATIVE_CK=64 NATIVE_V3=1
NATIVE_PAIR=1, NATIVE_SWIZZLE=0. Existing default switches preserve original
control. Each Cube owns successive paired gate/up channel tiles, with two GM
slots; its two Vector lanes divide rows and consume that same tile. Gate/up
remain two native GEMMs at the current128-channel granularity. No new weight
format and no quantization. FP32 native SwiGLU keeps BF16 GEMM rounding.

After both FIX outputs, Cube publishes CrossCoreSetFlag mode2/PIPE_FIX, flag8.
Both paired Vector lanes wait, consume, and return mode2/PIPE_MTE3 credit flag9.
Cube waits only before reusing its third and subsequent slots and drains the
last outstanding credits at exit. Other physical cores need not rendezvous.
Final credit uses the conservative output-store completion boundary. Per-vector
PIPE_ALL inside its local work loop remains; no claim of zero synchronization.

Fine-grained mode2 signaling has a native910B precedent in pinned Ascend9bf964c:
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_spilit_fusion.h`.
Our pair ownership/addressing is different; this is not a verbatim port of that
algorithm. Existing CANN attribution remains. The v2 source itself is unmodified.

Scratch actual requirement:24cores *2slots *256rows *256columns *2bytes=6MiB.
The benchmark intentionally kept the older, larger scratch allocation to isolate
execution changes; no measured memory-saving claim. Existing supported slab
allocations cover this requirement; the slab argument no longer sets pair-loop
granularity. VC256/512 both fit CN128. Pair scratch layout differs from slab
layout: do NOT run diagnose.py's final-slab oracle on a paired binary. Use
probe.py and whole-FFN bench.py. The vc0 diagnostic writes paired scratch only.

## Observed local7 dummy result

All rows128/257/769, two VC sizes, guards/immutable inputs, three changed-input
FULL replays passed. Whole FFN tests512/4096 use native down in every variant,
six alternating ND/NZ/candidate trials of ten FULL replays. Full FFN correctness
also passed. Model quality/production integration are not tested.

| rows | slab control FFN | pair FFN | native ND (pair cohort) | native NZ (pair cohort) |
|---|---:|---:|---:|---:|
|512|1.040ms|0.977ms|1.038ms|0.882ms|
|4096|7.225ms|6.528ms|6.090ms|5.997ms|

Pair improves4K FFN9.7% vs slab, but remains slower than native controls. Native
NZ was5.861ms in the earlier slab cohort; use matched cohort samples rather than
assuming all external variation disappeared. Pair scheduling changes both
synchronization and tile ordering; this speedup cannot be attributed solely to
barrier latency. The micro-pipeline is operational, not a claim that all overlap
or backend performance opportunities have been captured.

Artifacts under `/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/`:
`build-v6-sw0`, `build-v6-sw1`, `diagnose-v6-local7`, `build-v7-pair`,
`paired-v7-source`, `paired-v7-local7`. Frozen source, binary identities, six
samples and release receipt retained. Test bench capsule limits slab to256 and
VC256 instead of repeating identical paired scheduling for irrelevant slab sizes.
No installed package or main default changed; local7 released after completion.
