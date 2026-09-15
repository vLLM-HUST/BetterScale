# BF16 gate/up + SwiGLU leaf experiment

Contract: local910B2 only, BF16 contiguous input[M,H], frozen packed weight[H,2I],
mutable output[M,I], disjoint storage, 1<=M<=4096,H<=5120,I<=13824. No quantization,
no routing, no distributed work. Native reference is SwiGLU(mm(X,W.T)); down uses
unchanged native matmul. Preserve BF16 gate/up rounding before FP32 activation.
Tail loads/stores are masked. Each program owns disjoint BM-by-BN output tiles;
compiler owns bounded Cube/Vector relay and synchronization. Capture specializes
M,H,I and tile; input data may change without shape/address changes.

First cheap test uses Triton-Ascend tiled dot + fused Vector epilogue, NOT a port
of the integer MoE kernel and NOT a claim of zero HBM traffic. This discriminates
numerical semantics, compiler relay cost and GEMM throughput before investing in
a hand-written AscendC BF16 double-buffer implementation. Inspect generated code
and memory before claiming the compiler achieved our intended bounded workspace.
Weight packing is outside measurement/capture; it does not change values and can
replace native storage in an integration. The leaf fixture retains both layouts
for differential comparison; its resident weight footprint is not production.

Reference mechanism: Ascend/triton-ascend programming guide cv_fusion_operator.md
and tutorial03-matrix-multiplication.py. Existing compiler/toolchain only; no
runtime modification. No serving integration or quality claim from dummy inputs.

## Outcome: useful workspace proof, **not a speedup**

2026-09-15, local physical1/logical0; all owned jobs reclaimed. Dummy BF16
Qwen shapes, no model checkpoint load. `results.json` retains crossed samples,
errors, capture deltas, exact generated binary identities and failed variants.
Frozen source and admission/release receipts:
`/workspace/strengthen-dsv4/runs/qwen14b-ffn-20260915/`.

| Rows | Native whole FFN | Best measured fused whole FFN | Ratio |
| --- | --- | --- | --- |
| 512 | 1.039ms | 2.264ms | 2.18x slower |
| 4096 | 6.062ms | 14.022ms | 2.31x slower |

Best tested: 24 persistent programs, M-fast traversal to reuse weights,
BM64/BN64/BK512. Initial one-program-per-tile version was slower and reserved a
relay per logical program: at4096 rows its candidate capture delta reached432MiB
instead of native340MiB. A fused source expression did not imply bounded memory!

Persistent programs reduce the compiler-reported per-program relay to32768bytes
×24 =0.75MiB. At4096 rows, measured extra allocated peak DURING whole-FFN capture
is60.001MiB; candidate also uses a preallocated108MiB SwiGLU output, for an
accounted168.001MiB, versus native340.002MiB capture delta. At512 it is
5.000+13.5=18.500MiB versus56.502MiB. These are leaf allocated-peak comparisons,
NOT graph-pool total reservation or serving savings. Both weight layouts remain
resident only for the oracle; packing and those persistent weights are excluded.
The result supports bounded relay reuse, not zero physical HBM traffic.

Changing-input FULL replay and input immutability/output guards passed the dummy
thresholds. Max error at the gate/SwiGLU output can reach0.0625 on rare elements;
RMS is much smaller. This is tolerance-qualified BF16 behavior, not exact output
identity. An early65x256 /272-intermediate case tests row/column tails; latest
M-fast variant is measured only at512/4096 rows with actual Qwen dimensions.
No OpenCompass, real-weight or serving qualification was performed.

BM128 persistent variants exceed192KiB UB. Increasing BK from512 to1024 exceeds
512KiB CBUF (compiler requires768KiB); BK2048 was not reached. Do not keep sweeping
that resource-invalid region. Explicit K tiling, compiler relay and GEMM schedule
remain costs. M-fast ordering and larger valid BK materially improve latency but
still lose to native; this does NOT prove a tuned AscendC BF16 fused kernel must
lose. It establishes that this bounded Triton implementation is not deployable
as a performance improvement. Further investment would need a different GEMM
pipeline, not more claims from the same fusion wrapper.

Reproduce through the frozen local admission launcher, not a direct unleased
NPU invocation. `probe.py --output PATH` expects the pinned donor Python/CANN
environment and an already selected local card. It takes no8-card resources.
