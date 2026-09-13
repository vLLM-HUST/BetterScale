# Native DSV4 decode ping-pong integration (work in progress)

Scope: native DP8TP1 / TP8DP1, EP8, K5, stable text decode. Keep the existing
scheduler, prefill route, release pins, State ownership and eager/kept draft
choice. No N+2 scheduler rewrite. Experimental branch only; deployment defaults
remain unchanged.

## Current checkpoint (September13, runs097–100)

- Real TP8 and DP8 pass exact producer/sampler, target output and full-KV
  qualification with strict HCCL; performance uses normal HCCL instead.
- TP8,16 requests/96 target queries: matched cycles are68.44/68.68ms native
  versus57.01/56.75ms with the explicit producer plus captured metadata, two
  repeats (16.7–17.4% shorter). Endpoint ping-pong alone has no stable win.
- At the same8192 admission/8GiB KV configuration, reserved memory is50.178GiB,
  only28MiB above the retained endpoint control. Shared auxiliary scratch pools
  avoid requiring one private pool per shape/bank; live outputs remain owned.
- DP8 performance and retained32-item quality are pending an eight-card window:
  run100 lost usable memory on card6 during startup and measured no serving.
- Detailed boundaries, reproducible flags and failure observations follow below.
  This is not yet the packaged TP8 profile's split-draft integration or a complete
  three-stream serving implementation.

## Transfer the invariant, not the old engine

LiveInfer's source reference is `AscendDSV4WaveExecutor` in
`src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py` in the retained
`/root/my-ascend-workspace/livemodule-main-integration` checkout. It binds two
independent metadata/argument trees to `device_wave_0/1`, uses ingress/compute/
egress events and retains ONE numerical continuation State. The host holds a
two-wave ledger. Graph-read completion and output-copy completion are distinct
reuse gates.

Native DSV4 already has the device feedback operation:
`update_num_computed_tokens_for_batch_change` computes previous GPU progress +
actual accepted count for continuing speculative rows. **Do not double-buffer
that tensor** or replace it with optimistic CPU progress. CPU lengths used for
metadata tiling and GPU lengths used for addresses are separate contracts.

## Bounded implementation stages

1. `--pingpong`: capture two PRIVATE decode call packets and two native graph
   entries per bounded small descriptor. Preserve storage aliasing and slices
   across metadata fields, including heterogeneous views. Scalars use the
   original native capture exemplar; never capture bounds from the first short
   live request. Prefill remains the prior native/FULL path. Target/draft/State
   operations stay on the native device stream. This first stage copies native
   metadata into private banks (D2D), NOT overlapped H2D.
2. `--pingpong-sources`: rotate pinned CPU ingress sources, preserving their
   logical contents and NumPy aliases, and wait for the same bank's prior input
   preparation event. Device progress, sampled/draft feedback and D2H accepted
   count destinations remain single-copy. Both dummy and actual waves use the
   same scope. Native GPU staging destinations / device ordering stay unchanged.
3. `--pingpong-continuous`: reuse the already tested stable-verification receipt
   cut, extended explicitly to DSA and the admitted seat limit. Authorize using
   conservative CPU bounds, keep device progress exact, submit target before
   consuming the previous CPU bookkeeping receipt. Retain the current source
   DMA completion fence before mutating its CPU logical view. Extend the shadow
   oracle to rebuild exact CPU metadata for its eager reference.
4. `--shadow-decode`: manually construct owned pinned ingress on the host,
   publish it on an independent H2D stream, then replay the native device
   preparation arithmetic. Stable K5 only; keep numerical progress single-copy.
5. `--shadow-metadata`: capture downstream native device metadata construction
   with conservative tiling bounds and exact device lengths. Reject unbanked
   host transfers/readbacks. This permits skipping the CURRENT input-DMA fence
   for the admitted budget-only late callback; original receipt waits remain.

These flags are experimental and compose in that order. None is enabled by
`strengthen-dsv4 serve`; do not promote a partial bridge as the complete
LiveInfer three-stream protocol. Stages4–5 own the separate H2D lane and were
qualified independently of the earlier endpoint bridge. Numerical shadow-check
flags are separate from this serving protocol and excluded from timing.

## Evidence / unresolved gates

- 43 CPU tests pass, including independent bank/storage-view identity, repeated
  tensor aliases, rejection before partial refresh, same-bank-only host waits,
  coherent logical CPU state on rotation, and fail-closed generation poisoning.
- Run068: local card6 was admitted, then a foreign process arrived during
  initialization. The launcher stopped only this run; no model evidence.
- Run069: both native decode graph banks captured, but the runtime stream guard
  incorrectly retained the temporary startup-capture stream. Explicit post-start
  RPC binds the serving stream instead. The failed run is not acceptance.
- Run070: source-slot setup rejected the existence of an accepted-count event.
  Native K5 allocates it even without a proven active D2H write. It is now kept
  untouched as a feedback event, with its original wait, rather than asserting
  absence or removing it. This setup failure is not a graph-state failure.
- Run071: graph/source slots pass40 same-program checks, exact output/KV;
  two decode banks each replay19 times,52 total actual/dummy source scopes.
- Run072: adding the stable receipt cut passes40 checks;30 independently rebuild
  exact CPU metadata,35 late callbacks and36 bound bypasses. Single card only.
- Run071 also exposed an ownership mistake in the first packet inventory:
  copying the immutable full RoPE tables cost2.0GiB across four small banks.
  `get_full_cos_and_sin_dsa` returns precomputed model constants, unlike mutable
  position-selected runtime RoPE. Full compression RoPE and Hadamard are now
  explicitly shared with address/layout guards; mutable runtime RoPE stays private.
- Run073 DP8: idle ranks inherited the previous actual wave's `skipped` oracle
  flag and rebuilt dummy metadata after its source slot map was invalidated.
  Outputs matched, but the independent write map no longer matched. Reset the
  per-wave admission/reference flags at entry to the common source scope (used
  by BOTH actual and dummy waves). A CPU gate protects that reset. Run074 checks
  this diagnosis rather than loosening the KV oracle.
- Run074 DP8: all eight ranks pass40 checks, exact output and whole-KV bytes;
  immutable-table sharing reduces private packet backing to1,127,872bytes/rank.
- Run075 TP8: the packet layout guard rejected a shorter leading metadata view.
  Native DSACP FULL retains its capture view over the same backing. The bridge
  now copies that entire backing and permits only leading metadata-view changes,
  retaining dtype/stride/offset/backing-extent guards (never argument reshaping).
- Run076 TP8: all eight ranks pass48 checks against the unchanged native graph;
  output difference0, whole-KV bytes exact,43 exact-CPU-metadata checks/rank.
  Private packet backing878,776bytes/rank. TP uses strict HCCL,4 dummy seats and
  native K5 alignment. This is incremental graph-vs-graph, NOT a new native
  graph/eager equivalence claim; earlier native graph/eager uncertainty remains.
- Still required: matching real-weight control/candidate steady windows and
  compressed diagnostic timelines,
  memory delta, real retained OpenCompass quality before default promotion.

Original capsules live on hw3 under
`/workspace/my-ascend-workspace/runs/strengthen-dsv4-full-mixed/pingpong-*/`.
Never include profile/shadow time in the performance comparison. DP and TP must
match global active requests and actual target query rows; retain acceptance
and output-throughput accounting separately from step time.

## Isolate the benefit before promotion

`--pingpong-study` runs two within-engine repetitions at16 global requests,
96 target query rows, K5, eager native draft. Four policies share the weights,
State allocation, prefill route and reserved graph pool:

- `native`: original native target replay/fences and receipt handling.
- `cut`: previously qualified ordered replay + late receipt, plus CPU QLI
  metadata maxima for TP. No banked packets/sources in the executing path.
- `sources`: cut plus two pinned CPU source slots.
- `pair`: sources plus two private graph packets and alternating replay.

Warm each policy separately. Compare the first10 fully occupied FULL cycles,
not total cohort time alone; keep every request's actual output/wave count.
`cut -> pair` is the incremental banking effect, NOT `native -> pair`.
The retained native reference graph exists only under study/shadow flags.

Run077 exposed a real-weight host cost hidden by the four-layer fixture: TP8
`pair` took336ms/cycle versus about71ms native. The refresh walker treated
metadata shared across layers as a tree, repeating validation/temporary-view
creation. Run078 was stopped by its verified owned launcher during startup;
there is no DP performance evidence from it. The walker now visits each
(source,destination) identity pair once; a CPU test protects both DAG reuse and
alias-split rejection. Runs079/080 retest this specific diagnosis. No slowdown
is dismissed as noise, and the banking bridge remains opt-in.

### Stop per-layer host copies at the graph boundary

Runs079/080 remove the repeated DAG walk but still find about10ms of pair
submission overhead: DP pair71.3/72.8ms versus native62.0/61.7ms; TP pair78.4/
79.3ms versus native67.7/68.2ms (two within-engine repeats, first10 occupied
cycles). This rejects the runtime packet-refresh bridge as a performance patch.

`--pingpong-captured-copy` instead records the fixed native-input-backing copies
inside each target bank's graph. It relies on the SAME persistent capture input
addresses as the untouched native FULL graph, holds those source allocations,
and removes the metadata walk/per-copy host dispatch from replay altogether.
It still does not create an independent ingress stream. It is an explicit
alternative to runtime refresh, not a relaxed numerical acceptance threshold.

Run081 TP8 passes48 checks/rank and run082 DP8 passes40 checks/rank after this
change, with exact outputs/KV. Runs083/084 measure the real-weight effect.
The observation now attaches actual query/request counts and dummy status at
native mode selection; padded cross-DP buckets are never actual work counts.
Older early occupied windows use their retained real-schedule ordinals; requests
can arrive one tick apart, so take a common later window only when every rank
still has all16 global requests active. Do not assume rank-local window starts
are equal or include native drain dummy waves in the throughput denominator.

## Fletcher's correction: enter at shadow construction, not after preparation

The captured-copy bridge is still an endpoint adapter. It does NOT implement
LiveInfer's host shadow protocol, even if its copies are captured. Stop layering
more endpoint machinery onto it; retain it as a bounded negative/control result.

Read the retained LiveInfer checkout's:
- `src/livemodule/runtime/invocation.py:134`: `shadow_replay` runs the recorded
  explicit construction program and publishes its ingress before PREPARED.
- `src/livemodule/runtime/shadow.py:GraphIngressEffectTrace`: graph-owned stable
  host sources and device destinations, ordered H2D effect matching and staging.
- `src/livemodule/arch/ascend/request_parallel/dsv4/speculative.py:752`:
  `_construct_wave_routes` makes CPU owner geometry into ingress; actual lengths,
  addresses, validity, rotary selection and descriptors remain device work.
- `src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py:186`:
  ingress waits the SAME bank's graph-done, compute waits ingress-ready and that
  bank's old copy-done, then egress runs separately and retains host source leases.

This is NOT the same as our numerical same-state `*_shadow.py` oracle. Do not
confuse the name with validation, and do not move actual speculative progress
to host merely because some metadata are host-constructed.

### Donor seam to change

Pinned MRV1 `_prepare_inputs` (814) interleaves CPU projection, H2D, and real
State-dependent GPU operations. Its larger `synchronize_input_prep` scope (1773)
also includes state bookkeeping, DP shape agreement and metadata building.
Wrapping the resulting call tree cannot extract that producer dependency.

For admitted stable K5, separate:

| Responsibility | Producer / lifetime |
| --- | --- |
| Request identity, scheduled width, block leases, previous-row mapping, CPU tiling bounds | Host construction; snapshot into bank-owned pinned ingress, not mutable runner bookkeeping arrays |
| Actual accepted counts, sampled/draft IDs, computed-token progress, KV | Single continuation State on device; preserve native arithmetic and ordering |
| Exact positions, lengths, slot mapping, SAS/QLI descriptors, selected RoPE | Device derivation ordered after previous feedback and before the target; not speculative host guesses |
| Output receipt and CPU accounting | Separate retained egress/source lifetime; do not recycle pages or mutate in-flight H2D sources |

The minimal credible integration is producer-level host snapshot + explicit
publication and device derivation, with per-bank reuse events. Preserve native
DP collective order/shape agreement and prefill/turnover fallback. DP and TP
share the protocol but NOT the same metadata builder or collective schedule.
A generic copy of every field, a FakeTensor walk through the whole model on each
step, and an extra end-of-preparation packet clone are not substitutes.

Open implementation hinge: identify the exact H2D input set and separate the
late CPU bookkeeping destinations from DMA source slots. In particular native
`num_computed_tokens_cpu_tensor.to(device)` is a host budget input to the GPU
correction kernel, not the numerical progress State. `num_computed_tokens` on
GPU must remain single-copy. The existing current-input-DMA host fence can only
be removed after this ownership separation, not just after adding a ready event.

### Closed endpoint-adapter measurements (083/084)

Both layouts use16 actual requests,96 target rows,K5,8GiB KV/rank and eager native
draft. Two within-engine repetitions; numbers are occupied cycle medians in ms:

| Layout | native | receipt cut | + host source slots | + captured-copy graph pair |
| --- | --- | --- | --- | --- |
| TP8DP1 | 69.31 / 69.16 | 67.35 / 68.60 | 67.57 / 69.19 | 69.23 / 69.27 |
| DP8TP1 | 62.25 / 60.57 | 62.54 / 61.63 | 62.81 / 62.63 | 62.55 / 62.94 |

Capture removes the adapter's large host-launch tax but does NOT establish an
incremental speedup. Do not promote this bridge or use cohort-time variation as
a banking win. Packaged defaults and main remain untouched. These measurements
are not a test of the producer-level LiveInfer shadow protocol.

### Producer host-projection audit

`--producer-shadow-audit` is one diagnostic stable `_prepare_inputs` call, not a
new serving path. It uses the same host-real/device-fake separation as LiveInfer
to inventory inputs and find explicit device action boundaries. Do NOT treat
FakeTensor's disabled fallback as protection against arbitrary direct launchers.

Run085 entered native block-table `compute_slot_mapping`, which directly calls a
Triton kernel outside Torch dispatch, and ended with507035 / invalid MPU address.
The owned process was reclaimed; release records show no residual NPU processes.
No successful projection or correctness claim comes from this run. Source shows
this direct launcher consumes fake positions. Run086 explicitly excludes that
pure device-write action from host projection and preserves its persistent
output contract; this is a changed boundary, not a retry of unguarded fake mode.
The audit checks progress/input/position/length/slot-map device buffers unchanged
and never modifies production defaults. Further native direct launchers, if any,
need explicit boundaries before they can enter host shadow.

Run086 completes the input-preparation host projection with the explicit
slot-mapping boundary:115 virtualized Torch operations, no observed readbacks,
and unchanged progress/input/position/length/slot-map device buffers. Its21-copy
H2D inventory is incomplete: it omitted ten `aten.to.dtype_layout` occurrences
(the detector now recognizes that overload). Do not call21 the complete ingress
count. This diagnostic is discovery, not the serving implementation.

Fletcher explicitly chose manual construction of the actual shadow protocol.
`decode_shadow.py` now implements a bounded producer instead of replaying that
FakeTensor audit at runtime:
- explicit CPU K5 geometry and mutable resource/budget snapshots;
- two pinned ingress/source and device-input banks, their own H2D stream;
- H2D-source reuse waits old upload, device bank reuse waits old consumption;
- native accepted-count arithmetic and direct slot-mapping kernel in a real
  captured device-preparation program, leaving progress single-copy;
- current native sampled/draft feedback handed to the preparation program on the
  compute stream, never guessed from CPU budgets;
- prefills, turnover, masked partial queries and hybrid feedback remain native.

This first producer boundary covers `_prepare_inputs`, NOT the entire downstream
DSA/DSACP metadata builder or a complete three-stream donor runtime. Existing
late-receipt/current-input-DMA guards remain until the remaining H2D owners are
separated. Run087 checks the new producer's device fields and complete sampler
metadata against original preparation from identical State. No new performance
claim precedes that gate.

Run087 caught an exact sampler **dtype** contract error (target logits indices
were int64, native uses in-place int32 arithmetic). No tolerance relaxation:
geometry now emits the native int32 result. Run088 passes12 original-preparation
checks, including initial capture, both banks, and stable2-request to1-request
shapes. Run089 additionally captures the device metadata program and passes38
original-native target output/KV checks (all output differences0, KV bytes exact)
and12 input/sampler checks. Both are single-card dummy gates, not throughput or
TP/DP qualification.

`decode_metadata.py` refuses Torch-visible host/device transfers in the captured
metadata stage; reviewed direct native Triton metadata operations execute with
REAL device tensors. Tiling maxima use the admitted model limit, while exact
lengths and positions stay dynamic on device. Cache identity includes actual and
padded geometry plus CPU source-carrier identity; target metadata is cached but
DSpark receives a fresh common envelope and refreshed per-group views. Native
prefill/dummy/turnover and original exact-bound oracle builders remain separate.

The next scoped fence removal follows the actual callback writes: native
`correct_spec_decode_token_counts` changes request state and CPU computed-token
budgets only. The explicit producer already snapshots that budget into owned
pinned ingress, so its admitted waves can retire the old receipt without waiting
for CURRENT native input DMA. The real receipt wait itself remains in the native
callback. Native/fallback waves retain the current-DMA guard. This change is
CPU-contract-tested and enters the next multirank gate; it is not yet a measured
continuous-replay gain.

TP2 run090 exposed a missing activation, not a new QLI algorithm gap: the native
DSACP builder still extracted NPU scalars because this new entry had not enabled
the already-qualified CPU-QLI hook. Capture rejected that readback (107027).
The metadata entry now explicitly activates CPU-QLI on TP layouts; DeviceOnly
also rejects `_local_scalar_dense` before a backend readback. CPU tests cover
local host/device work, H2D, D2H, cross-device copy and scalar extraction.

Run091 TP2 dummy passes12 input/sampler checks and28 original-native target
output/KV checks on EACH rank (all output differences0, KV bytes exact). Both
ranks execute22 captured metadata replays across8 CPU-carrier/shape entries and
avoid22 current-input-DMA fences. Run092 DP2 dummy passes12 input/sampler checks
and48 target output/KV checks per rank, including independent DP progress and
native dummy EP drain. These do not establish eight-rank real-weight throughput.

Real-weight qualification uses `HCCL_DETERMINISTIC=strict` to separate scheduling
from collective arithmetic variation; performance runs use normal HCCL. Run093
was stopped during initialization to correct that missing qualification setting,
with owned launcher cleanup, and provides no result. Protocol settings are now
retained per DP rank. `--decode-only` bounds the large-real-model investigation
to the entrusted decode path instead of repeating unrelated long-prefill cohorts.
The same-engine producer study separates native, previous receipt cut, endpoint
pair, explicit input producer, and captured metadata; optional profiles are
collected in separate warmed native/candidate cohorts, never timed as throughput.

Run094 closes real TP8 strict-HCCL qualification:8 ranks ×48 target/KV checks
(all output differences0, full KV bytes exact) and8×12 producer/sampler checks.
It uses192-token admission and3GiB KV/rank; it is NOT throughput evidence.

Run095 did NOT complete the performance study. It failed with207001 in
`AclrtReserveMemAddress` during the second metadata warmup (native shared-expert
quant allocation). Shape-specific preparation and metadata captures each created
private pools; multiplying expandable-segment address reservations is the current
suspect, not a proven KV capacity failure. The next implementation shares ONE
scratch pool per program across serialized shapes/banks, retaining all returned
tensors; preparation and metadata have separate pools. Policy-boundary receipts
now retain graph counts and allocated/reserved HBM to observe growth.

There is also a measurement-design correction:192-token admission fragmented
prefill and the16-request cohort, leaving no qualifying11-forward full-occupancy
window. Do not turn its cohort times into a speedup/slowdown claim. Restore the
previous run083's8192-token admission budget for TP steady-decode measurements;
the actual measured target remains96 rows/16 requests. The failed run's receipts
are retained locally with `FAILED_NOT_THROUGHPUT_EVIDENCE`, not forged completion.

Run096 stops at native KV-capacity validation before serving/capture qualification:
at8192 admission/16384 context the pinned hybrid cache requires4.59GiB even for
one max-length request, so the previous3GiB oracle budget is invalid. This is a
separate admission constraint, NOT a recurrence of207001 and not a shared-pool
result. Run097 uses5GiB: run094 measured43.82GiB resident/49.72GiB shadow peak at
3GiB KV; the extra2GiB KV plus two extra2GiB snapshots predicts about55.72GiB peak.
The performance control retains its previously qualified8GiB KV budget without
oracle snapshots. Do not reuse a small-prefill KV minimum after changing admission.

Run097 passes with the shared auxiliary pools, real TP8/strict HCCL,8192
admission and5GiB KV: each rank has12 exact producer/sampler checks and48
original-native target checks (output max difference0 and whole KV bytes exact),
including the16-request/96-row shape. Final allocated46.43GiB/reserved47.21GiB;
oracle peak56.74GiB includes full backing snapshots, not serving demand.
Eight preparation banks and eight metadata entries were exercised per rank.
This is a passing bounded configuration, NOT proof that private pools caused
run095's address-reservation error (admission/KV budgets also differ).
The target pair already uses the native wrapper's shared graph pool. New
auxiliary preparation and metadata programs now each have one shared scratch
pool; their live outputs/ingress are retained independently. Ping-pong does not
require duplicating model activation workspace. Different allocation layouts and
live graph outputs can still add memory; capture count alone is not an HBM bill.
Run098 restores the qualified performance configuration8192/8GiB without the
numerical oracle; its policy-boundary receipts will measure actual growth.

Run098 completes real TP8 performance and both native/candidate eight-rank raw
profiles, with normal HCCL and no numeric oracle. Matched16-request/96-query-row
cycle medians (ms, two repeats) are: native68.44/68.68, prior receipt-cut
68.17/68.71, endpoint pair68.18/68.34, explicit producer64.40/64.55, and captured
metadata57.01/56.75. Thus the complete producer+metadata path shortens these
matched cycles16.7–17.4%; endpoint banking alone still does not. Target-event
spans include queued preparation/waits and are NOT isolated model compute.
Whole-cohort durations still vary with acceptance/drain/cold shape admission;
no workload-throughput guarantee follows from these step measurements.

Final allocated49.376GiB, reserved50.178GiB, peak49.700GiB/rank. Reserved memory
is28MiB above run083 with the same8192/8GiB configuration. In-engine policy
receipts range50.051–50.178GiB reserved while prep banks grow6→11 and metadata
entries6→10. All policies retain the same graphs, so this is not a comparison
against an unpatched engine's memory. It supports bounded extra memory for this
shared-pool implementation, not proof of run095's failure cause. The raw profile
parse is offline; daemon parse warnings are expected, not dropped rank evidence.

Run099 additionally passes real DP8/strict HCCL at1026 admission and3GiB KV:
12 producer/sampler and48 original-native target/KV checks per rank, all output
max differences0 and full KV bytes exact. This includes independent rank
progress and native dummy EP service. Reserved53.717–53.719GiB; the58.469GiB
peak includes oracle backing snapshots. This is correctness, not DP performance.

Run100 cannot provide a performance or quality result. All eight devices passed
initial admission, but rank6 then reported only32.49/60.96GiB free during native
startup, below the0.92 utilization check. npu-smi showed33352MiB on device6 with
no listed process. Do not attribute this to graph scratch or invent ownership;
model capture never began. The owned launcher was interrupted, reclaimed its
workers, and wrote release.txt; no foreign process was killed. Failed native DP
clients can leave sibling clients blocked at initialization: the outer launcher
owns process-group reclamation, so do not wait for a Python traceback alone to
release the lease. DP performance/quality still requires an uncontended window.

Run098's two complete eight-rank profiles are now parsed and exported through
frozen TraceLoom37323af, with native rank/device identity checks. Local roots are
`runs/098-tp8-shared-pool-study/engine/profiledecode-{native,metadata}`;
user-facing files are `analysis/target-draft-tp8-end-aligned.json.gz` within each.
Both pass the unchanged50us holdout gate using eager small-control collective
identities. Fits are candidate-only display alignment, not physical clock
calibration. Official profiler DBs, derived rank DBs, rejected/accepted protocol
boundaries and raw timestamps remain available; no raw JSON need be opened.
