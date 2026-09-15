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
