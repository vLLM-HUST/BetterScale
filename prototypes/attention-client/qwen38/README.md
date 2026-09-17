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

## September17 integration checkpoint

The composition gates above have now advanced:

- Native input DynamicQuant + actual-count INT8 GMM + fused SwiGLU quantization
  passes the real-weight chain (`quant-chain-result.json`). A custom input
  quantizer differed by one integer level and was rejected; ingress uses native
  DynamicQuant, publishing INT8 rows and FP32 per-row scales.
- Persistent mixed target/MTP catalog gate passes three cases, including two
  sources at the same and different layers (`server-leaf-result.json`). This
  remains a one-device protocol/math gate, not a performance result.
- Full TP2 attention **meta** construction retains 60 quantized QSA projections
  and zero routed-expert parameters. Each rank owns 5,479,377,280 parameter bytes;
  this excludes State, host PLE, allocator and graph workspaces.
- Five-device real layer0 gate passes: one client and four independent owners,
  each loading its 128 experts; rows1/4/32, changed-input outer graph replay,
  source scale publication, collect, weighted unpermute, promotion and EOF drain.
  Maximum relative L2 vs the integer/FP32/BF16 reference is3.271887e-6.
  `wire-result.json` is the compact receipt. This does **not** qualify full-model
  output, MTP across devices, large prefill or concurrent attention sources.

`catalog.py` groups checkpoint reads by shard within each layer and keeps only
NZ weights resident. `attention.py` replaces the immutable MoE binding, not a
process-global installed donor. `model_client.py` is the full-root integration
runner under development; no full48 success claim yet.

The first INT8 server deliberately uses whole up/down readiness and complete-owner
collect. Earlier BF16 fine-grained prefix/return optimizations are not assumed to
work with the new mixed-dtype scratch ABI. Large-prefill capacity is still a
separate missing gate: this wire admits at most32 rows per source.

`ipc_acl.py` is a disconnected copy of the owned LiveInfer IPC helper at file
revision1dc65a99. That helper postdates the pinned Qwen38 branch; importing it
from that old branch was an invalid assumption. Keep this dependency explicit.
The private native runtime overlay combines Python source820103bf with the
September7 mapped-QSA wheel; its receipt lives in `runs/qwen38-native-runtime-20260917`.
Preserve CANN's PYTHONPATH when adding the overlay (otherwise TBE import fails).

### Exact PLE metadata repair

The downloaded quantized snapshot has BF16 `layer_multipliers`,
`ngram_heads_offsets` and `ngram_heads_vocab_sizes`. These are semantic integer
buffers, not activations; BF16 rounding destroys the PLE hash/index contract.
The colocated original checkpoint retains int64. Text configs match; converting
all three original buffers to BF16 reproduces the quantized snapshot exactly.
Sampled unquantized router and embedding rows also match.

`ple_metadata.py` restores the three original buffers without changing either
snapshot, with explicit dtype/shape/cast checks. **Results belong to a repaired
checkpoint**, not the untouched Eco-Tech export. The TP2 real attention load then
passed (48target layers,60quantized projections, actual PLE ownership, no routed
weights): allocated9,865,748,480bytes/rank, reserved12,002,000,896bytes/rank including
the4GiB State budget. This is a construction gate, not full forward quality.

Full target runner: `bash prototypes/attention-client/qwen38/run_model.sh 1,3,4,5,6,7`.
Use `... 1,3 --construct-only` for the already-passed loading gate. `--decode-graph`
is the following eager-vs-replay State-shadow gate and is not yet qualified.
Run capsules snapshot Python and admit only their selected devices. Do not
manually bypass an occupied card or overwrite a capsule's source/binary closure.

### Persistent-kernel lifetime is a launch contract

The inherited `device-service/launch.cpp` explicitly supplied
`ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US=10000000` (10seconds). That per-launch
microbenchmark setting remained present despite the process-level1200s setter.
Full model runs consequently lost the resident servers while the client was
still doing cold work; `neural_collect` timeout was downstream, not evidence of
GDN arithmetic failure. A diagnostic run completed all48 prefill layers and
agreed on token7824 before the next phase exposed another boundary.

Qwen38 now owns `launch.cpp` with a1200s launch budget and ABI receipt v2.
Both Engine and client reject the old short-lifetime closure. The supervisor
still owns finite startup/execution deadlines and fail-stop cleanup. This does
not establish an indefinitely resident production server; long-running service
must account for device task lifetime explicitly.

The target-only GDN adapter normalizes the old metadata's always-present,
K=0-clamped acceptance selector to `None` at the ordinary-decode backend boundary.
Candidate handling for K>0 remains unchanged. Earlier first-token gates alone
could not expose this target-only continuation gap.
