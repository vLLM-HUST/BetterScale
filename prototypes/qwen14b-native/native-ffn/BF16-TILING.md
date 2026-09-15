# BF16 native tiling investigation (2026-09-15)

## Concrete finding, not yet a timing result

Our fork changed INT8 operands to BF16 but retained baseK128. Static native
GetL0ADb/GetL0BDb automatically disables double buffering if one operand tile
exceeds half of its64KiB L0 buffer. AtM128/N256/K128, B occupies64KiB, so B
loses double buffering. AtM256/N128/K128, A loses it instead. Fixing L1 step/depth
alone did not fix L0. This is a directly derivable configuration defect relative
to the intended two-sided pipeline, not proof of how much latency it explains.

| L0 tile M/N/K | A bytes | B bytes | A/B double buffering |
|---|---:|---:|---|
|128/256/128|32768|65536|yes/no|
|256/128/128|65536|32768|no/yes|
|128/256/64|16384|32768|yes/yes|
|256/128/64|32768|16384|yes/yes|

FP32 L0C occupies128KiB for all four, so this does not enable L0C pingpong.
The smallest controlled next experiment is baseK64 with stepKa/Kb4 and
depthA1/B1=8. That keeps L1 K extent256, two buffers, and total L1 operand
footprint384KiB unchanged while changing the L0 pipeline. Test both M/N
orientations against current K128 and native NZ on one local card. Keep the
slab/Vector protocol unchanged to avoid confounding changes. No speedup claimed.

## Existing native designs to borrow

Installed CANN9.0.1 ships readable MatMulV3-related source at:
`/usr/local/Ascend/cann-9.0.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/mat_mul_v3/`.
It is Mc2MatMulV3 source, NOT established as the exact dispatched torch.mm binary.

- `op_host/op_tiling/matmul_v3_base_tiling.cpp:640`:
  OptimizeBasicKernelStepK explicitly recognizes BF16/FP16 M/N128/256 or256/128,
  baseK64,24cores. Its eligible basic-kernel branch reduces stepK8 to4 for
  selected shapes. Qwen4096x27648x5120 meets the numerical size/alignment checks,
  but runtime dispatch/tiling conditions have not been recovered, so do not
  assert that the measured baseline selected it.
- `op_kernel/mat_mul_base_kernel.h:143`: L2 M/N blocking, alternating N traversal,
  staggered core indexing and per-block Iterate/GetTensorC. Our linear traversal
  lacks that locality policy. Crucially native ALSO loops over output blocks;
  the mere presence of our tile loop does not prove excessive launch overhead.
- Installed `aarch64-linux/asc/impl/adv_api/tiling/matmul/matmul_constant_tiling_utils.h:300`
  contains the exact automatic L0 DB size tests. `asc/include/adv_api/matmul/constant_tiling.h`
  calls them and fixes dbL0C=1 in this static-tiling route.
- Ascend CATLASS optimization guidance documents A2 L1[128,256,256],
  L0[128,256,64], pingpong and optimized preload/shuffleK policies:
  https://gitee.com/ascend/catlass/blob/master/docs/catlass_optimize_guidance.md
  Its displayed storage example is FP16; transfer the two-byte storage arithmetic
  to BF16, not an unmeasured performance guarantee. No CATLASS code was imported.

Qwen H5120 and gate/upN27648 are aligned to256; the large4096-row wave has no
M/N/K tail at these blocks. Small/tail batches are a separate acceptance case.
Weight layout NZ already helps our native control; keep that stronger control.
Next priority after L0 is measured locality/pipeline behavior, not blindly larger
M/N, splitK accumulation, new quantization, or a replacement kernel framework.

## Controlled K64 result

Local physical7, dummy BF16, same six crossed trials/ten FULL replays as v4.
Both K64 orientations passed the complete rows128/257/769 guard/immutability/
changed-input FULL probe, followed by4096-row diagnostic acceptance. The initial
physical1 run was interrupted by a foreign job after its first smoke passed;
no timings from that run are used. All four timing variants below ran on7.

| M/N | slab | K128 fused ms | K64 fused ms | K64 Cube+protocol ms |
|---|---:|---:|---:|---:|
|128/256|256|7.002|6.879|6.567|
|128/256|1024|6.996|5.489|4.583|
|256/128|256|6.254|5.269|4.971|
|256/128|1024|6.189|5.262|4.910|

Native NZ GEMM control3.597..3.659ms across cohorts. M256/N128/slab256 improves
15.8% over matched K128. The benefit depends on slab/orientation; this is not a
universal K64 speedup. Cube-only keeps protocol barriers and cannot be
subtracted to infer independently overlapping Vector duration. End-to-end FFN
with down was NOT remeasured in v5; no serving speedup claim. Fastest fusion is
still slower than native pure GEMM, so the dominant gap is not closed.

Reproduce `NATIVE_CK=64` with `NATIVE_CM`/`NATIVE_CN`; L1 step/depth derive from
CK to preserve K extent256/two buffers. Default K128 deliberately remains the
historical control, not a recommended tuned production setting. Artifacts:
`/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/build-v5-{base,m256}`
and `diagnose-v5-local7` (frozen suite, crossed results, admission/release). The
run capsule includes original build source; the tracked follow-up changes only
its explanatory comment. Use the recorded binary identities for exact repeats.
