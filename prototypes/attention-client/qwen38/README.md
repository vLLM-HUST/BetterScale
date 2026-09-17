# Qwen3.8 remote routed-expert integration

**Basic real-weight end-to-end and FULL decode gates passed.** This is not yet
a production, quality or throughput qualification. Reuse the owned LiveInfer
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
existing server contracts. The target-only topology now passes the gates below;
two independent TP2 groups remain unqualified.

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

1. Add the separately math-qualified BF16 MTP layer to full-model serving.
2. Qualify multiple attention sources and large-prefill batching; the first
   Qwen38 source window still admits only1..32 rows.
3. Establish independent numerical/language-quality evidence and online
   throughput/latency under comparable workloads. Short agreed outputs alone
   are not quality or performance evidence.
4. Account explicitly for device-task lifetime in long-running serving.

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

## Full target continuation passed

`full-target-result.json`: TP2 attention plus E4, all48 real target layers,
real original-integer-repaired PLE, one3-token prefill and three1-token decode
calls. No per-layer diagnostic synchronization. Both attention ranks agree on
`[7824,11,6326,9703]`; each expert owner completes192 generations and drains
normally. Each owner retains30,293,360,640bytes of routed weights/scales.
The full async run passes after replacing the short per-launch deadline.

This is a real end-to-end **mechanism gate**, not a language-quality benchmark,
throughput gain, multi-source campaign, large-prefill test, or MTP qualification.
The following FULL decode gate has since passed, as recorded below.

### PLE convolution graph dispatch

The target-only capture exposed PLE `nn.Conv1d` dispatching through legacy
aclop Conv2D. `GraphPLEConv` scopes `ALLOW_INTERNAL_FORMAT=disable` to that leaf,
restoring the prior setting immediately; it does not change INT8 NZ weights.
Real PLE weights at1/3/32query rows match the previous eager convolution exactly,
and changed-input replay is exact (`ple-conv-result.json`). The installed
`torch.npu.config` has a setter but no getter for this field; preserve its option
via the same `_npu_getOption` mechanism used by the installed HF32 accessors.

Retain the **whole captured ForwardContext**, including active/query-length
inputs, until graph reset. Keeping only token ids, positions and sequence lengths
leaves other external graph inputs eligible for allocator reuse. This is a
capture-lifetime requirement, not a new scheduler.


## FULL decode and same-State shadow passed

`full-graph-result.json` records the final six-device gate (physical2,3,4,5,6,7):

- One TP2 attention group, four independent E4 owners, all48 real target layers,
  repaired integer PLE metadata and native W8A8 projection/expert weights.
- One3-token prefill, then three changed-input1-token FULL graph decode replays.
  An additional eager shadow starts from the identical post-prefill State.
- Hidden-output relative L2 is **0.0 on both attention ranks** in that shadow.
  Generated ids remain `[7824,11,6326,9703]`, matching the preceding eager run.
- Each expert owner completes exactly240 calls (including the extra shadow),
  then drains and releases its channel. All six processes exit successfully.
- Per-layer diagnostics are disabled. Graph inputs retain the entire captured
  metadata frame until graph reset; the earlier partial-retention attempt is
  rejected evidence, not a passed run.

Reproducer: `bash prototypes/attention-client/qwen38/run_model.sh 2,3,4,5,6,7 --decode-graph`.
The private native closure and repaired original checkpoint dependency described
above are required. Published Worker defaults and other model lanes are unchanged.

## Two independent attention sources

`run_model.sh 0,1,2,3,4,5,6,7 --sources 2 --decode-graph` starts two
independent TP2 attention groups on cards0..3 and the shared E4 on cards4..7.
Each group uses a separate HCCL rendezvous and one publishing leader. The
server keys registrations by source ID, not accept order; both windows must
be discovered before clients export their input allocation. Each source has
its own generation and output allocation. Both sources must drain before
server shutdown; a fast source cannot reclaim another source's buffers.

The existing two-source kernel and weight catalog are unchanged. This first
fixture uses the same short prompt on each source; it is not a large-prefill
or online arrival experiment. The two sources progress independently through
layers. Same-layer co-batching is opportunistic, never forced by a per-layer
host barrier. FULL graph capture/shadow work remains included in wave1 wall
time and must not be reported as steady-state latency.

After a successful run, `analyze_concurrency.py <capsule>/roles` validates the
four clients and owner completion counts. Each server wave consumes one or
two source descriptors, so `sum(completed_counts) - waves` counts full-run
paired waves. The64-record rolling trace checks sampled pairs' layer identity;
it does not establish a full-run distribution of per-layer behavior.

September17: Python/shell checks pass. The first eight-card attempt was stopped
by admission supervision after foreign work appeared on cards0/1; no performance
or dual-source correctness qualification is claimed from that attempt. The
second admission also reached real weight loading, then stopped for a new
foreign owner. Both capsules are rejected; no job remains queued. The previously
passed six-card single-source result remains a separate qualification.


### hw0 qualification: two sources sharing the same E4 pool

After direct pinned ModelScope download and exact PLE metadata validation, the
full48 W8A8 target passed on **two independent TP2 groups plus E4**. All four
attention ranks have zero same-State eager/FULL-graph hidden relative L2. The
65-token output sequences match across both sources and both single-source
controls. No MTP, broad language quality, or large-prefill claim follows.

Cold-start/capture skew made the initial four-wave dual-source gate observe
**zero** co-batching. The next fixture rendezvous **once** after capture, then
runs63 continuous decode replays/source without per-step or per-layer barriers.
The two sources have independent State but the same short prompt, a favorable
routing-locality case rather than a diverse workload.

| Configuration | Cards | Steady output tokens/s | Per-source step median |
| --- | ---: | ---: | ---: |
| One TP2 source + E4, first control | 6 | 19.41 | 43.45 ms |
| One TP2 source + E4, repeat | 6 | 20.53 | 42.13 ms |
| Two TP2 sources + the same E4 | 8 | 42.28 aggregate | 46.21–46.45 ms |

**Do not present the raw >2x ratio as a clean scaling gain.** The two controls
both stall at wave52 (493ms and401ms); the later GC probe below explains why. All pauses
are retained, not silently excluded. The more conservative observation is that
serving a second source increases typical step cost by about7–10%, while the
expert pool stays at four devices. This is a short closed-loop decode test,
not equal-card native-vs-separated serving or SLO-qualified online throughput.

The dual-source servers each complete `[3168,3168]` layer calls, in5642–5647
compute waves: **689–694 paired waves per server**, about22% of source calls
co-batched over the full run (including warm-up). The64-entry ring tail happens
to contain no pairs, so it supplies no sampled paired-layer verification here;
the exact aggregate count comes from completed calls minus compute waves.

[Compact comparison](hw0-concurrency-result.json) records all three timing runs
and capsule identities. `compare_sources.py` recomputes the timing without
removing outliers; `analyze_concurrency.py` computes owner co-batching counts.
Full role receipts are retained in local `runs/qwen38-hw0-results-20260917/`
and the corresponding hw0 capsules. All admitted jobs exited0 and released
owned devices; the download and campaign waiters finished.


### Long-pause cause and controlled comparison

The instrumented single-source run120404Z again pauses at wave52:425ms total.
Its generation2 Python GC lasts**379.9ms**, entirely inside replay submission to
validity readback, collecting**zero** objects. The paired TP rank waits too.
With `--defer-steady-gc`, wave52 becomes42.1ms and max steady latency stays below
45ms. [GC evidence](hw0-gc-pause-result.json) records the intervals and causal
control. This is host cyclic-GC interference, not400ms of expert GEMM.

This option collects before the once-only steady-start rendezvous, disables
cyclic GC for at most96 steps, then restores it and collects after measurement.
Reference counting stays active. It is a **bounded experimental control**, not
an adopted production GC policy or permission to disable GC indefinitely.
`--observe-pauses` records GC/host phase times only when explicitly selected.

The [same-GC-condition comparison](hw0-gc-controlled-comparison.json) is the
cleaner result (63 post-capture decode steps/source, unchanged output IDs):

| Same E4 pool | Cards | Aggregate output tokens/s | Per-source step median |
| --- | ---: | ---: | ---: |
| One TP2 attention source | 6 | 23.24 | 42.09 ms |
| Two TP2 attention sources | 8 | 44.38 | 43.66–44.14 ms |

That is**1.91x aggregate output rate**, for two additional attention cards and
about**4–5% higher typical per-source latency**. No steady-window GC occurred;
all same-State graph errors remain0. This supports the pool's ability to serve
concurrent attention sources, not an equal-card advantage over colocated vLLM.

Each owner in this controlled dual run completed3168 calls/source but only48
paired waves over the full run, versus689–694 in the earlier dual run. Pairing
is sensitive to relative layer phase. **Do not attribute all throughput scaling
to co-batching, or claim a stable co-batch rate from these short runs.**

Reproduce either source count with the corresponding6/8 admitted cards:

```bash
bash prototypes/attention-client/qwen38/run_model.sh 0,1,2,3,4,5,6,7 \
  --sources 2 --decode-graph --decode-steps 64 --align-steady-start \
  --observe-pauses --defer-steady-gc
```

The single-source control retains attention cards0/1 and expert cards4..7:
use devices `0,1,4,5,6,7` and `--sources 1`. The hw0 capsules are120738Z
(single) and121027Z(dual), both under `runs/qwen38-model-20260917T...`.
