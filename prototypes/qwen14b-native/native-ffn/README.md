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
