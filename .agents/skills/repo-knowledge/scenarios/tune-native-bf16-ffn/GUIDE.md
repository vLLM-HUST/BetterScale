# Tune native BF16 dense FFN

Use before repeating Qwen2.5-14B native AscendC fusion experiments. Read the
bounded results and reproducible leaf in
[the native fork](../../../../../prototypes/qwen14b-native/native-ffn/README.md)
and the quantization boundary analysis in
[NATIVE-QUANT-CUT](../../../../../prototypes/qwen14b-native/NATIVE-QUANT-CUT.md).

The immutable upstream subtree is a backup, not an edit target. Adapted source
retains its CANN file-level license. Build the small isolated ASC shared library,
not the installed donor package. No Triton; single-card work is local, not HW3.

Observed: removing quantization and bounding GM slabs passed dummy FULL replay,
but did not beat native BF16 FFN. Native NZ is a stronger control than native ND.
A Cube-only diagnostic preserving synchronization isolates a large residual
GEMM/protocol cost; M256/N128 improves the fork but does not close it. Use these
controls before a new sweep. Allocated capture delta plus external scratch is
not reserved-pool peak. Model quality and serving integration remain untested.

Follow-up source inspection found that inherited baseK128 disables one BF16
L0 operand double buffer. Read [BF16 tiling](../../../../../prototypes/qwen14b-native/native-ffn/BF16-TILING.md)
before further sweeps; K64 with preserved L1 extent is the next controlled test,
not an already measured improvement.

The K64 controlled test is now recorded in BF16-TILING: both tail/FULL probes
passed; M256/N128/slab256 fused stage improved15.8% vs matched K128 on local7.
The earlier "next test" paragraph is historical; do not repeat it without a new
hypothesis. The fusion remains slower than native NZ pure GEMM.

[Pair-local micro-pipeline](../../../../../prototypes/qwen14b-native/native-ffn/MICROPIPELINE.md)
records MatMulV3 call/swizzle ablations and the successful per-Cube paired-slot
protocol. Objective is SwiGLU overlap, not bounded scratch alone. Pair scratch
layout invalidates diagnose.py's slab oracle; use the full-output smoke/FFN tests.

Buffer-depth ablation is complete in MICROPIPELINE:8 slots and fully unique
per-tile storage both passed dummy FULL but were slower than2 slots. Do not
repeat buffer expansion on the assumption Cube must be credit-starved. Distinguish
net timing from direct wait attribution; scratch allocation differs by build.

[Per-core PMU](../../../../../prototypes/qwen14b-native/native-ffn/PMU.md)
now measures pair credit wait~13us, Vector ready wait~3.32ms, and worse AIC
L2 read-hit/MTE2 duration than native. No dominant credit-wait evidence.
TimelineDetail failed while real-board PMU CSVs succeeded; preserve that partial
status. Kernel-prefix filtering works; tested MSTX wrappers did not select kernels.
