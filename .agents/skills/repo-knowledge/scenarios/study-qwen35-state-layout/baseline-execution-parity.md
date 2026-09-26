# Restore BetterScale baseline execution around owned State

Enter when replacing the direct weight loader, interpreting a numerical
mismatch after registering native arithmetic, or qualifying async submission.
**Fletcher's latest correction (September26): apart from State, behavior must
remain the qualified BetterScale baseline's behavior.** Bare native Ascend is
not the reference for reimplementing individual numerical leaves. Reusing only
NPUWorker initialization does not satisfy that requirement when the model,
MTP protocol, sampler and scheduler are independently rewritten.

## Authoritative baseline and substitution boundary

The measured control is frozen in workspace
`runs/betterscale-mtp-small-fish/20260924T160000Z-qualification/`:
`swe/optimized35b-c16/plan.json`, `launch.sh`, `candidate/` (including `package`
and `runtime-source`) and `controller3/memory_worker.py`. Its C16 point is
368.9277778 output tokens/s/chip. The observer Worker derives from that capsule's
BetterScale Worker, not bare NPUWorker. The plan/source, not a similarly named
current checkout or the native-reference capsule, defines the baseline.

| Responsibility | Qualified baseline | Alternative live vertical |
|---|---|---|
| Scheduling | `apc_boundary.BoundaryScheduler(AsyncScheduler)` + EngineCore batch queue | local `Scheduler`, one kind/width family per wave; newly added pending events do not recreate the baseline queue |
| Model arithmetic | actual target/draft model forward and installed BetterScale leaves | `numerics.decoder/attention/gdn` manually compose layers |
| GDN preprocessing | fused `preprocess` preserves BF16 Q/K **and beta** boundaries | separate gating/splits plus internally FP32-normalized short recurrence |
| Prefill/mixed |4096-token budget, mixed real requests, qualified padded FULL keys |1024 budget, one-request chunk graphs, forced initial short step, binary short-wave batches |
| MTP | donor proposer with BetterScale merged draft/banked device feedback | host generator submits two draft steps, verifies, then repairs accepted hidden seeds |
| Sampling/API | vLLM request/generation/sampling path; capsule draft-only reduced greedy flag | separate greedy-only service/protocol |
| State | hybrid shared allocator and Mamba/APC boundary machinery | approved exclusive resident GDN/candidate lanes plus pooled regular-attention pages |

Only the last row's storage/ownership and the necessary graph-generation lifetime
substitution are the desired reform. Preserve baseline compute, MTP, sampling,
prefill/mixed policy and asynchronous scheduling; adapt their State addressing
and resident-lease lifecycle instead of treating the standalone vertical as the
new serving baseline. Hot-seat affinity/eviction and shared-page capacity remain
accepted requirements; they do not authorize changes to numerical execution.

The standalone path was useful for State/graph qualification but is not a
State-only port. Stop benchmarking or patch-by-patch numerical reconciliation of
it as if it were the final route. The uncommitted native-init/async exploration
is preserved in `20260926-baseline-boundary-audit/`; none of it is a qualified
production repair. No hardware job remains from this audit.

## Historical native-init exploration (not the final route)
The native Worker may initialize the device, register operators, load target/MTP
weights and run post-load transforms. It must not allocate request caches,
profile native capacity, capture native graphs or execute requests. Keep it alive
until the live root and all pending invocations are closed.

## Numerical evidence before performance claims

Capsules below are under workspace `runs/qwen35-state-lanes/`. They freeze the
working candidate based on `f82a447`; the code/launch scripts in each capsule,
not current mutable source, define the tested artifact. This is ongoing repair,
not a new accepted 35B serving or performance result.

- `20260926-native-init-small1` and `20260926-native-async-small1` pass the
  0.8B TP1 chunked target/MTP and hot/cold gates using actual
  `AscendGemmaRMSNorm`. The latter executes pending-event submission/readback.
- `20260926-native-async-tp2-1` reaches 16808 shared pages /30 graphs at
  262144 context, E16/R20, but fails the first raw12 token comparison. Do not
  call it a passed long-context gate or run a leaderboard on that evidence.
- `20260926-native-init-diagnostic1`: 35B TP2 E1/R2,256 context,4 pages,
  no bulk prefill. Native graph/MTP, native eager target-only and explicit-FP32
  router eager all match the old raw12 reference.
- `20260926-native-prefill-diagnostic1`: enabling four-token chunks changes
  the raw12 sequence; disabling chunk use on that same root restores it.
- `20260926-native-chunk-capture2`: direct chunk graph and eager results
  have exactly equal per-layer hidden/residual tensors. Adding stream barriers
  does not change them. Their next two short steps also agree, but differ from
  all-single-token prefill. This does not establish graph or async corruption.
- `20260926-native-chunk-state1`: first-layer convolution history after the
  four-token chunk equals serial exactly. Recurrent max difference is0.0277833
  against reference magnitude9.28629; later layers accumulate differences.
  Neither explicitly adding the FP32 router nor native fused QKV projection
  alone restores the raw first choice. This is not a proof of State misaddressing.
- `20260926-short-chunk-leaf1`: captured4/16/64-token chunks, QK8/V16 and
  QK16/V16, pass the independent FP32 recurrence with BF16-normalized operands;
  maximum final-State error is below9.8e-5. Neighbor candidates remain exact.

## Do not mistake a tie for a layout oracle

The existing independent native raw reference has exactly equal log probabilities
for IDs198 and220 at the first output (`20260925-native-reference2`). Its
Transformers CPU comparison also has a narrow margin; see `owned-execution.md`.
Full autoregressive divergence after that choice cannot by itself prove State
corruption. Keep independent meaningful chat/continuation gates and compare
teacher-forced numerical contracts rather than forcing one marginal token.

Two source-level contract corrections matter:

1. The earlier claim that native Qwen always gets FP32 router weights was too
   broad. Pinned Ascend linear post-load does so only with
   `precast_fp32_weight=True`; native Qwen reports false, no `weight_fp32`, and
   external BF16 routing. The old direct-loader FP32 intervention was a
   separately qualified policy, not a faithful description of native bootstrap.
2. Pinned `vllm_ascend/ops/gdn.py` applies `l2norm_fwd` to Q/K before native
   prefill, decode **and speculative verification**. Normalized tensors have
   activation dtype. The live short-wave kernel had instead retained FP32 Q/K
   normalization internally, unlike its BF16-normalized chunk path. Restore
   this numerical boundary only as an isolated diagnostic, not a substitute
   for the actual BetterScale fused preprocess. `20260926-native-normalized-short1`
   makes serial/chunk raw12 agree with each other but not the old native reference;
   `matches_native=false`. Its ad-hoc chat EOS configuration also continued past
   the reference terminator. The normalization-only source edit was reverted
   after Fletcher reasserted the baseline boundary; no acceptance is claimed.

The async candidate owns cloned device hidden outputs and one pinned host logit
readback per batch until a recorded completion event, then retires its invocation.
CPU fixtures check submit-all-buckets-before-retirement, cancellation draining and
HTTP arrivals while a device event is pending. These are not evidence of full
native scheduling overlap or throughput parity; a fresh timeline and C16 run
remain required after numerical qualification.

## Owned baseline port and native State seams (September26, in progress)

The model-specific closure now lives in `betterscale.models.qwen35`, composing
closed GDN/FIA leaves rather than adding cross-imports among generic patch
packages. `betterscale.qwen35_worker.Worker` aliases the original Worker and
installs the baseline's early FULL-key policy. The initial CPU suite passed207
checks; serving remains a separate gate. The generic dense geometry defaults
stay unchanged;35B explicitly supplies GDN QK8/V16,16 requests plus sentinel,
FIA Q8/KV1 and4096 query capacity.

Audit **both directions** of a capsule migration. The frozen donor runner itself
imports top-level `device_apc` twice (execute-model ingress/preparation). Moving
only the helper imports does not close the dependency. `runtime-imports.patch`
redirects those two calls to the owned namespace and the Qwen35 source pin binds
that changed runner. Do not install `sys.modules` aliases or retain the old
capsule on PYTHONPATH. `20260926-baseline-port1` was deliberately cancelled during
weight loading after discovering this seam; it is not numerical evidence.
`20260926-baseline-port2` PASS:24 exact retrievals (cold/warm through262080
and16 concurrent),32 mixed FULL graphs,98 drafted/98 accepted, server exit0;
selected0/1 released IDLE. The capsule uses the previously qualified generalized
GDN host adapter at explicit QK8/V16 geometry; the device GDN binary is unchanged.
`numerical-origin.json` records15 numerical/protocol leaves with identical AST
apart from import namespaces. This is baseline-port functional acceptance, not
a live-State or new performance result.

Native integration findings, not an implemented/qualified State adapter:

- Keep the regular-attention KV manager's allocation and preemption policy.
  Remove GDN from that **capacity** domain, not model execution. Worker-only GDN
  metadata groups and published resident/candidate addresses need explicit
  handling; native group IDs must not accidentally refer to nonexistent token
  block tables. FA specs must lose the old hybrid page padding.
- `initialize_kv_cache_tensors` is the allocation/binding seam. Native
  `bind_kv_cache` publishes the given object directly to each consumer; FA can
  consume the separate K,V tensors already declared by `QwenStateRoot`.
- CPU experiment `20260926-baseline-boundary-audit/partial-page-handoff.py`
  exercised the actual pinned core KV manager: a257-token resident can retain
  three128-token pages and transfer them to a new request without allocating
  more pages. Pin/free references were balanced and the pool returned to its
  original free count. But native `num_cached_block` becomes3 although only2
  pages are complete. A resident adapter must explicitly own partial-tail hash
  bookkeeping; the experiment does **not** prove a full prefix-cache protocol.
- Hot-seat identity still includes MTP's +1 lookahead token. It cannot advertise
  a cursor shorter than the actual recurrent State, or equate visible text/EOS
  truncation with the raw accepted device frontier. Preserve async completion
  fences before reusing a departed seat, just as native deferred block freeing
  protects pages. Stop/abort with outstanding frames requires a separate gate.
