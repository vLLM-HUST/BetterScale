# Qwen3.8 remote routed-expert integration

Active integration; **not yet end-to-end qualified**. Reuse the owned LiveInfer
Qwen4Exp root at820103bf, including host PLE, HC, QSA pair and GDN State. The
checkpoint is the pinned shared-directory W8A8+BF16 snapshot documented in
`../qwen-next/priority/model-readiness.md`. Published defaults remain untouched.

## Execution/placement plan

First one TP2 attention group plus four expert owners (six devices); later two
independent TP2 groups plus E4. QSA cannot be treated as the older TP1 Next
client. Each attention group retains native shared expert and request State.
Only its leader publishes routed work; shared TP work runs before collect and
the completed routed result is broadcast inside that attention pair. Generation,
layer, class, shared-completion promotion and all-owner retirement remain the
existing server contracts. This is a plan, not a passed transport gate.

Target experts are W8A8_DYNAMIC; MTP layer48 experts remain fused BF16. Do not
silently dequantize the whole checkpoint to BF16 and call it W8A8. QSA target
q/k/v/o and index_qk projections are also quantized, so swapping only the remote
MLP loader is insufficient. GDN/HC/shared/PLE remain on their native path.

## First passed math gates (September17)

- `runs/qwen38-quant-20260917T022823Z`: native npu_dynamic_quant/quant_matmul,
  real target experts(layer0/id0, layer3/id511, layer47/id257), real BF16 MTP
  expert0; rows1/32/128. Integer-accumulation/dequant oracle max relative L2
  0.0000750523. Changed-input graph replay matches eager exactly. No serving claim.
- `quant_gmm.cpp` is a thin INT8→INT32 instantiation of installed CANN CATLASS,
  not a rewritten GEMM. Actual device group ends admit empty/interior-zero groups
  and a live prefix shorter than capacity. Eight exact integer cases at
  up2560×1280/down640×2560 pass; inactive tails retain canaries. Build closure
  `runs/qwen38-quant-gmm-build-20260917T023033Z`; source and compile log retained.
  This qualifies the matrix primitive, not persistent-server integration.

`weights.py` only reads selected expert tensors and rejects unsupported nonzero
weight offsets. Current gates read actual checkpoint scales; no dummy route or
unrelated model checkpoint substitutes for the new geometry.

## Remaining integration gates

1. Compose quantize/dequantize/SwiGLU with actual-count Cube work, preserving
   paired source packing and readiness/return ordering; keep BF16 MTP branch.
2. Add native W8A8 QSA projection loading to the reused attention root; route
   experts externally without first allocating local routed weights.
3. Six-role dummy/real one-layer shadow, then full48 target, real host PLE,
   recurrent continuation and captured decode. MTP is a separate following gate.
4. Independent numerical/quality evidence before online performance claims.

No other LiveInfer worktree or installed donor runtime has been modified.
