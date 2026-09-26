# Restore BetterScale baseline execution around owned State

Enter when replacing the direct weight loader, interpreting a numerical
mismatch after registering native arithmetic, or qualifying async submission.
**Fletcher's latest correction (September26): apart from State, behavior must
remain the qualified BetterScale baseline's behavior.** Bare native Ascend is
not the reference for reimplementing individual numerical leaves. Reusing only
NPUWorker initialization does not satisfy that requirement when the model,
MTP protocol, sampler and scheduler are independently rewritten.

## Current implementation scope — Fletcher, September26

Use the baseline's per-seat GDN short history and MTP2 candidate selection/update
protocol. One convolution window and three recurrent candidates per resident;
CPU async queue depth must not multiply these declarations. **Historical/aligned
checkpoints are out of scope for this cut.** Do not add terminal rollback banks
or a checkpoint pool to guarantee every CPU-visible stop prefix is reusable.
A retained hot seat may advertise only its actual represented frontier after
all queued writers retire; incompatible earlier prefixes are misses.

The rejected two-bank worktree is archived in the workspace audit capsule as
`paused-two-bank-worktree.tar.gz`. Its arithmetic-address extension and bank
copies have been removed from current source; baseline numerical leaves are
again identical to committed namespace port5b976bd. Single-set CPU suite passes
219 tests. Hardware functional acceptance is recorded below; a fresh matched
timeline/performance comparison remains required.

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

Native integration findings used by the State adapter:

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

### Single-set State adapter and qualification

Baseline namespace port is committed as `5b976bd`. The following working cut is
separate: `state_backend` substitutes declarations/binding and removes GDN from
the scheduler's page capacity; `state_address/state_slots` publish independent
conv/candidate addresses without changing arithmetic; `seat_scheduler` subclasses
the real AsyncScheduler and uses its existing deferred-free fences. It does not
replace the queue, model, proposer or sampler. The root currently retains native
graph capture, so this is **not yet final live graph-lifecycle integration**.

The actual core's deferred-free feature is disabled for ordinary connector-free
serving; merely finding its implementation did not prove baseline enabled it.
The State adapter explicitly enables that existing fence for resident retirement.
The CPU fixture `tests/native_qwen35_state_fixture.py` exercises its ownership
hooks with the actual KV manager:257-token/3-page warm transfer, balanced native
references, and delayed hot publication after a queued final writer. The fixture now also exercises active preemption, same-request-ID restart on a
different seat while the old writer remains fenced, abort and balanced page
reclamation. NPU evidence is separate, below.

**Accepted semantic boundary:** raw model progress may extend past CPU EOS or
length trimming under asynchronous scheduling. Retain the exact physical frontier,
not an earlier visible prefix. Fletcher explicitly excluded historical checkpoints
and terminal rollback from this cut. `Frontier.checkpoint()` names an exact CPU
identity, not a saved historical numerical bank. The finish-order audit below
explains why some visible request-end prefixes must miss.

The State adapter's current repository CPU suite passes219 tests, including the
pinned native page-manager fixture and eviction-before-new-page-allocation. Run
`pytest tests`, not unbounded repository discovery: vendored upstream suites
require separate dependency/accelerator environments and are not this contract.


### Observed async terminal boundary; rejected two-bank proposal

`20260926-baseline-finish3` has scheduler-process evidence:11 requests,
68 schedule/output events, normal server exit and selected-device release.
With23-token prompts and greedy limits1/2/3/4/5/8/16 (serial, then C4),5 requests
have a raw output frame after the CPU terminal decision;6 finish inside an
accepted group. This is length-stop evidence, not EOS/abort coverage.
`finish1` failed before HTTP on tokenizer BatchEncoding serialization;
`finish2` served HTTP but its Worker-only observer never reached the scheduler.
Neither is finish-order evidence.

The rejected two-bank proposal confused that distinction with a requirement to
retain arbitrary terminal prefixes. `20260926-native-state1` was cancelled during
startup, no requests ran, and selected0/1 were released IDLE. Its files are
archived, its source changes removed. Its3.5615GiB fixed-State /1,970,176-token
estimate is not current capacity or a required async penalty. Do not revive it
from the old audit's `two-wave-capacity.json`.

### Hardware functional observations — September26

All capsules below are under workspace `runs/qwen35-state-lanes/`. They freeze
the single-set adapter atop5b976bd with no changes to baseline numerical leaves.
E16/R20,35B BF16/TP2/MTP2, native AsyncScheduler batch queue and32 mixed FULL
graphs remain enabled. Native graph capture is deliberately retained; these
results do not establish final live graph-lifecycle ownership or performance.

- `20260926-native-state2`:24/24 exact retrievals, cold/repeated-cold through
  262080 prompt tokens plus16 concurrent. Earlier prompt-only repeats correctly
  miss: no historical checkpoints. State budget per rank26,038,239,232 bytes;
  fixed resident State1,912,095,840; attention24,126,143,392;
  16,720 physical128-token pages =2,140,160 token positions. No capacity search
  and no resident-count × max-context allocation. Server/launcher exit0,
  selected0/1 released IDLE.
- `20260926-native-state-hot1`:29 requests. A8193-token prompt generates one
  token, unrelated B uses a different empty seat, then A's actual continuation
  hits8193 cached tokens and matches an independent cold oracle's output IDs.
  Wrong MTP lookahead misses.24 unique-salt cold incarnations crossR20 and
  produce identical results after overwrite/epoch clearing. Exit0 and selected
  devices released. This is exact-frontier affinity, not arbitrary terminal
  rollback or historical checkpoint support.
- `20260926-native-state-pressure1`: all16 requests with32769-token prompts and
  768 forced outputs return correct initial retrievals and requested lengths,
  but **zero actual preemptions**, so the preemption gate fails. Admission and
  natural completions keep active pages under the reduced401408-token pool.
  Do not label a small-budget run as preemption coverage without actual events.
- `20260926-native-state-pressure2`: same source/budget,4096 forced outputs.
  PASS with3 real native preemptions. The scheduler-process observer verifies
  each releases seat ownership and invalidates the old frontier/pending-hot
  identity, including a final-writer fence beyond the processed step. All16
  responses have correct initial retrieval and4096 outputs; clean server and
  launcher exit0, selected0/1 released IDLE. Forced post-EOS text is not a
  quality oracle or token-for-token numerical parity check.

### Baseline does not keep two full candidate generations

Pinned core `MambaBase.get_kv_cache_spec` sets speculative blocks to MTP depth2.
`MambaManager.allocate_new_blocks` first allocates1 running +2 speculative rows,
reuses the speculative rows when moving forward, and allocates at most one new
row on an align-boundary transition. Its align sizing bound is2+2=4 rows, not
2*(1+2)=6. Boundary overlap retains the old source until migration is safe;
ordinary iterations within one logical block overwrite the same three candidate
rows. The live numerical convolution is one extended window, not one full conv
copy per candidate. Allocated hybrid blocks still contain their padded conv/SSM
regions; do not confuse physical block bytes with meaningful numerical contents.

CPU `native-gdn-retention.py` uses the actual pinned MambaManager/BlockPool with
B2048/MTP2/align. An uncached single-request sequence observes3,3,3,4,4,4,4,3
referenced non-null rows, with balanced final free. This is an allocator-level
observation, not a recorded NPU State dump or a cached multi-request simulation.
Separately, hashed aligned checkpoints can remain in the pool at refcount0 until
eviction; there is no fixed count of all historical cached rows.

`device_apc.wait_for_previous` orders the previous State postprocess before the
next execution's preparation. CPU queue depth2 therefore does **not** imply two
concurrent unordered GDN writers or require two State banks. Baseline APC retains
aligned checkpoints, and does not promise a GDN snapshot at every CPU-visible
terminal token. Reusing a hot seat's actual frontier and guaranteeing every
visible terminal prefix are different contracts. The finish3 observation proves
that distinction matters; it does not authorize increasing State capacity to
satisfy the stronger contract. Fletcher resolved the scope above: keep one baseline candidate set, and do not
guarantee arbitrary visible-terminal reuse this cut.
