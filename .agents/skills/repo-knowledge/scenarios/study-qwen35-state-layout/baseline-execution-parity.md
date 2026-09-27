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
copies have been removed from current source; active-row numerical leaves retain the committed namespace port5b976bd
behavior; the later negative-slot output correction is described below. The initial single-set CPU suite passed
219 tests; final write-budget source passes229. Functional and matched
timeline/performance acceptance are recorded below.

## Packaged entry and donor closure — September27

Source9386e3b closes delivery of the qualified35B route. The old public
`serve-qwen --runtime live` dispatch still selected the standalone executor even
though State/numerical source already lived under`src/betterscale`; it now selects
`models.qwen35.launch` and the original Worker/AsyncScheduler route. The small-model
standalone research entry remains separate. See the packaged model README for the
complete command; do not reuse an older standalone launch when validating State-only
performance.

Only four pinned Ascend files differ from pristine9bf964cb: runner APC ordering,
GDN strided gates, proposer draft greedy selection, and vocabulary gather policy.
`runtime.patch` contains the complete adaptation (not merely the older two-import
`runtime-imports.patch`). The CPU-only`models.qwen35.runtime` helper copies a built
installed donor into a new isolated directory, validates original/adapted identities,
applies the exact patch when needed, then checks all pinned sources and required
native payload presence. Unknown/mixed inputs and existing outputs are rejected;
no shared runtime is mutated. The Worker still checks core source/version pins.

The source delivery capsule`20260927-src-closure` reproduces all477 measured donor
Python files byte-for-byte starting from the installed pristine donor.231 owned CPU
tests pass. A fresh sdist-to-wheel build includes patch, manifest, scripts, State
source and all four qualified native artifacts; strict Twine checks and a clean
`--no-deps --target` install pass. Platform wheels place Python files under
`.data/purelib`; archive checks must not assume root-level`betterscale/` members.
Build's pip-isolated dependency fetch hit a certificate error; using its supported
`--installer uv` succeeded without disabling TLS verification. No PyPI upload or
shared installation change is authorized/implied. Actual installed-entry hardware
acceptance passes48 requests at BF16/TP2/MTP2/E16/R20/262144 and the full24.25GiB
State budget. All16 exact terminal hot hits, original-code retrievals and output-ID
comparisons with independent cold continuations pass; new-user deltas include5406
tokens across the4096 prefill budget. The public launcher invokes the actual Worker
without experiment observer imports. Server/launcher exit0; selected0/1 released
IDLE. Admission waited for foreign work instead of displacing it. Seven donor native
libraries also match the measured runtime directly. Compact receipt:
`docs/evidence/qwen35-resident-package.json`. This is delivery acceptance, separately
from the immutablea8abd05 timing result.

## Current serving entry

Use the real Worker alias`betterscale.qwen35_worker.Worker`, not the earlier
standalone service loop. For resident State select both:

```text
--additional-config '{"enable_cpu_binding":false,"using_live_runtime":true}'
--scheduler-cls betterscale.models.qwen35.seat_scheduler.LiveStateScheduler
```

The control omits`using_live_runtime` and selects
`betterscale.models.qwen35.apc_boundary.BoundaryScheduler`. Both retain the same
FULL keys, query4096, max-seqs16, native MTP2 and async flags; use a qualified
capsule's complete command rather than these two fragments alone. Source pins
include the two donor runner imports redirected by`runtime-imports.patch`;
ordinary unpatched donor installation is not this qualified runtime. Never
relax pins or put an old capsule on PYTHONPATH to make the import pass.

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

### Paired native-runtime profiles (September26)

`20260926-state-profile-{baseline,state}1` use identical6f81f81 Python source,
changing only the optional State flag/scheduler. Both pass exact retrieval,
C16 decode and heterogeneous mixed dispatch, clean exit and selected0/1 release.
`20260926-state-profile-comparison.json` records the comparison; per-arm
`profile-validation.json` binds both rank owners, raw identities and DB integrity.
Pinned TraceLoom isab8b5131191c6d5aeee2dd8566c34411f49ceab0.

Each rank has12 exact C16 decode waves (48 scheduled tokens),24 graph launches.
Target/draft graph member symbol counts are identical between arms. New-State
median target envelope/control ratios are0.99666/0.99968 on ranks0/1; draft
0.99783/0.99765. Between-wave graph gaps did not grow in this bounded capture.
This supports restored baseline execution, not production throughput equivalence;
a matched900-second C16 SWE comparison remains a separate gate.

**Do not misattribute a reconstruction limit to missing device execution.**
The mixed raw TASK/API window and12 FULL dispatches include the heterogeneous
arrival. The pinned analyzer's `exact_periodic_suffix` classifies leading
non-periodic launches as `unrecognized_leading_context` (baseline8 of24 launches).
Its16 recognized suffix launches must not be aligned to dispatch zero. The old
whole-window `trace_metrics.py` count assertion rejects this case correctly;
exact body/cost comparison here is restricted to complete decode coverage.
Preserve mixed Perfetto/raw evidence and typed unknown regions instead of
rerunning hardware merely to manufacture a periodic profile.

### Performance follow-up: protect the current length frontier

`20260926-native-state-swe-c16` (6f81f81) passes the900-second C16 protocol,
542 total requests,0 failures and selected-device release, but delivers only
165.9389 output tokens/s/chip. Of517 continuation requests, only16 hit resident
State; cached prompt tokens are3.14%. The older small-fish reference hits992/1034
continuations and93.69% of prompt tokens. Exact decode graph cost above did not
regress, so repeated prefill is the concrete next performance hypothesis—not a
reason to rewrite model arithmetic. Fresh same-pair `20260926-owned-baseline-swe-c16` passes900 seconds/0 failures
at369.1728 output tokens/s/chip, decode P90 55.3413 and TTFT P95 0.8494s.
Its clean exit/release agrees with the historical throughput reference; this
rules out namespace migration as the large regression seen in the first State run.

The write-budget implementation qualified in`20260926-write-budget1` does **not**
add a numerical bank or preserve historical checkpoints. The request's known
max-output limit bounds its right to advance the current resident State. Device
postprocess selects the terminal candidate from the existing three; an already
queued later frame publishes the baseline negative padding addresses for GDN,
so it cannot overwrite that candidate or its convolution window. Only one extra
int32 per resident is declared (80 bytes total/rank); sampler raw acceptance,
FA execution, scheduler queue and active-row recurrence remain unchanged.
EOS/stop-string rollback is still unsupported. This is not permission to infer
an arbitrary earlier frontier from a newer matrix.

Padding must produce defined finite activations: the old recurrence returned
without writing output for negative slots, because padded outputs were unused.
A retired request is still physically present in the queued model frame, so the
experiment makes that branch emit zero without touching State. Otherwise garbage
can reach MoE/draft sampling even though the CPU will discard the frame. The
leaf gate reuses the independent24-wave convolution/recurrence oracle, adding a
nonempty negative-slot row: it passes exact zero output, unchanged padding State
and active-row recurrence (maximum State error1.1176e-8).

A second admission detail matters: a matching length-frozen resident can still
await the native final-writer fence when the next HTTP turn arrives. Wait for
that one resident, rather than immediately assigning a cold empty seat. Keep
normal async scheduling and page-reference fencing; do not publish or reuse
State before retirement. CPU fixtures cover this wait/hit transition. The full CPU suite passes229 checks, followed by the expanded real-page-manager
fence fixture. `write-budget1` passes101 HTTP requests:29 original hot/turnover
cases,16 concurrent starts with budgets1/2/3/4/5/8/16/20, all16 exact terminal
hits plus16 independent cold continuations,8 cold/repeated-cold long-context
checks through262080, and16 pressure requests. Its6GiB State budget triggers2
actual native preemptions with whole-seat invalidation. Server/launcher exit0,
selected0/1 released IDLE. Pressure's post-EOS forced text is not a quality oracle.
Final-source timeline and900-second performance gates subsequently passed below.

The old adapter bypassed native prefix-cache statistic recording in its custom
`get_computed_blocks`. Therefore a server metric saying0% did not prove zero
hits; use per-response cached-token usage from retained requests. The experiment
also restores the existing native stats recorder for the new hit domain.

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


### Final-source matched acceptance — September26

Source`a8abd05`, same selected0/1 devices, BF16/TP2/MTP2, E16/R20,
262144 context,4096 query budget, native mixed FULL keys and AsyncScheduler.
The final fixed resident State is1,912,095,920 bytes/rank; the same
26,038,239,232-byte total budget leaves16,720 shared128-token FA pages
(2,140,160 nominal token positions). No second State bank, context reduction,
capacity search or synchronous fallback was used.

`20260926-state-profile-state2` passes both-rank raw/owner/DB validation and
clean teardown. `20260926-write-budget-profile-comparison.json` compares its
12-wave exact C16 decode against`state-profile-baseline1`: identical target/draft
member symbol counts; target median envelope ratios0.99334/0.99095,
draft0.99773/0.99623. Between-wave gaps are0.435/0.434ms. The mixed reconstruction
limitation above still applies; this is not an exact whole-mixed cost claim.

`20260926-write-budget-swe-c16` passes the actual pinned`swe-prefix-reuse`
29136f1f481ebab8566014a05b7ed53bcf79dc84 C16/900-second protocol with0 failures,
1194 completions in-window +16 drained, clean server/launcher exit and selected
cards released IDLE. Same workload/seed, model and runtime pins as the fresh
`owned-baseline-swe-c16` control; only State entry/scheduler and the qualified
write-lifetime change differ. This is one predeclared run per arm, not a
best-of-repeat uncertainty estimate.

| Metric | Fresh baseline | Resident State |
|---|---:|---:|
| Output tokens/s/chip |369.1728|414.8011|
| Decode tokens/s P90 |55.3413|59.4792|
| TTFT P95 (s) |0.8494|0.4623|
| Continuation requests with cache hits |997/1039|1170/1170|
| Prompt-token cache fraction |93.692%|97.516%|
| Uncached prompt tokens, including drain |1,929,434|905,142|

Throughput is12.36% higher in this bounded comparison, passing the no-more-than5%
regression working target; TTFT P95 is45.58% lower. Full-concurrency fraction is
99.495%; no preemptions occurred in this full-budget timing run (use the separate
reduced-pool gate for preemption evidence). Cache counters include all requests,
while throughput uses only the900-second window. Different completed workload
prefixes are inherent to this fixed-time exact-token continuation protocol.
`20260926-write-budget-swe-comparison.json` retains the derived comparison.

Interpretation: avoiding repeated prefill, not faster decode arithmetic, explains
the observed recovery. Do not generalize this one C16 point to C32, arbitrary
EOS/stop rollback, or final live graph-lifecycle ownership. Those remain outside
this accepted State-only cut. Capsule installs were isolated; no shared installed
runtime or PyPI release was changed.


`20260926-write-budget-hot-delta1` closes the joint continuation gate on the
same final source:48 requests,16 concurrent forced terminal budgets1..20,
then actual generated token IDs + a new native-chat user segment with
37/254/1822/5406 added tokens. All16 resumes hit the exact represented frontier;
each retrieves the original access code and has exactly the same output IDs as
its independent cold oracle. The5406-token delta crosses the4096 chunk budget,
so this covers hot GDN State resuming bulk/chunked prefill, not just a one-token
resume. Server/launcher exit0, selected0/1 released IDLE. The capsule's126 Python
files matcha8abd05; no extra observer was required.
