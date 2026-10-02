# Integrate incremental State with two-host online PD

Enter for the follow-on to `dual-host-pd.md`. Fletcher clarified on2026-10-02:
`codex/qwen35-incremental-cache` is the ready local cache implementation to
integrate, not an already completed cross-host connector. Do not wait for a
nonexistent newer branch or repeat naive whole-checkpoint profiling.

## Resume after the October2 online campaign

Current branch codex/pd-incremental-online carries concurrency/DMA work after
57affb0; hw86 is the active development checkout. The latest bounded State gate is the dual-native83-lane DRAM roundtrip; see
**Shared DRAM registration performance boundary** below. The last model gate
remains online-dma-v1 gate2; neither is a new workload throughput claim. Enter the **Device-bandwidth**
and **DRAM fast-path correction** sections near the end for the current frontier.
The active direction is now **Accepted correction: private rank-owned DRAM**,
not the earlier shared-memory prototype. Model/kernel capsules remain
online-v2-P/D. MTP is explicitly deferred.
The older online-stream-v6 and concurrent-v1 campaigns remain comparison
evidence, not the new DRAM-first connector. Both model services are stopped;
use fresh output paths on restart, never stale DRAM manifests.

## Accepted execution boundary

State pipeline and decode cadence are decoupled. The decode server continues
its normal native DP4/TP2/EP8 rotation. Requests enter through an asynchronous
warm admission path only after their own State is ready. Transfers never require
a fabricated model/dummy round or global DP idle. A matching producer fence,
page/seat ownership and TP completion quorum remain necessary for each request.
Fletcher explicitly corrected the empty-round framing; it is a risk of blindly
reusing the old DP1 dispatch seam, not a desired architecture.

## Current integration candidate and bounded qualification

Branch `codex/pd-incremental-online`, first frozen candidate69159a9. Cache code
is selectively adopted from2dac92c: CacheActions/PageResidency, policy, worker,
completion inbox and PageStateStore/host backend. E36 capacity plumbing and its
kernel/capture changes are deliberately not adopted. Existing E16/R20, donor
pins, CANN9.1 and target-only PD numerical baseline remain unchanged.

Opt-in `state_cache_control_rpc` dispatches TP-local `state_cache_actions`,
which enqueues copies and returns without a model forward or EP collective.
UTILITY and socket WAKEUP handling progresses this path independently of a
model step. Scheduler mutations and rank receipts stay on the owning Core
thread. A pending local transfer must not put an EP participant to sleep while
native `engines_running` is true. The original DP1 scheduling path is retained
when this option is false. CPU gates exercise both paths.

`page_transport.py` composes PageStateStore DMA with an immutable object sink.
One object is staged at a time; no growing-session blob crosses the control RPC.
Target FA page versions reuse sealed identities, tails are generation-private,
and GDN/conv plus selectors are boundary snapshots. Source/receiver replicas
must acknowledge before controller ownership publication. No per-P-instance
durable host mirror is retained: all instances use their node shared Store.
The prototype copies all target recurrent candidate rows, not just the selected
matrix; MTP execution/bootstrap remains outside its qualification.

`online_objects.py` provides the two explicitly addressed private HTTP/Mooncake
replicas.8MiB content chunks avoid the64MiB client staging limit. HEAD requires
all constituent chunks; immutable logical-version collisions fail closed.
Store eviction is not a warm hit. Recovery/HA and cache-pressure fallback are
not qualified by this first candidate.

`online_actor.py` multiplexes tagged local RPC replies so a generation does not
hold the control pipe until it finishes. `online_coordinator.py` retains the
SQLite single-writer directory, fixed D owner and caller-cancellation fence.
Per-owner seats and full-request page reservations bound D admission; frontend
metadata is separate from worker data movement. No global DP drain is called.

## Hardware gates and the event-lifetime correction

The first69159a9 hardware gate failed closed at exact post-H2D readback.
The integration incorrectly called PageStateStore's private batch-copy backend
for a standalone restore: that backend relies on its outer batch fence and
does not create a per-object event. This was not an accepted numerical drift.
Commit cd21eeb uses ordinary TorchHostStateBackend for standalone restore/audit,
waits its real event, then releases staging/publishes completion. A delayed-event
CPU regression protects both restore and audit. Source D2H retains the correctly
fenced PageStateStore batch path.

Frozen online-v2-source/packages (cd21eeb) passed all16-card byte-audited gates:
four-owner cold/warm1K, four-owner32K, eight-session4K, eight-session32K with
bounded P pipelining, and four-owner261952-token cold/warm plus one exact262144
context boundary. The last gate completed in243.762s; this is audit-on diagnostic
wall time, not production latency. Each gate checks output counts, ownership,
no pending D transfers and post-H2D object byte equality. It does not assert
greedy IDs must be invariant across segmentation or batching.

Artifacts are under /workspace/betterscale-pd-runtime on hw86:
hw86-online-v2-probe2, hw86-online-v2-32k, hw86-online-v2-online8,
hw86-online-v2-pipeline and hw86-online-v2-256k. P logs reside on hw81.
The earlier v2-probe failure was a helper's off-by-one cached-token expectation;
the corrected expectation is prompt length, with unique session IDs per campaign.

The32K gate measured warm D load of one missing FA page plus boundary State,
116572172 DMA bytes/rank, versus first load431144972 bytes/rank. These counts
exclude audit copies and are not network-throughput measurements.

P admission now permits up to four request lifetimes per TP2 instance, allowing
compute while another request backs up. Returning sessions prefer their former
P instance only when a permit is immediately available, retaining weak pages
without imposing affinity waiting. Each owner's incremental transactions remain
serialized, independently of compute. Neither optimization is a new model loop.

56 CPU tests passed (lifecycle/control/object transport/admission/affinity).
Runtime pins, kernels, E16/R20, context262144 and24.25GiB State/rank are preserved.
Both node Stores use512GiB DRAM. Correctness services were stopped with all NPUs
released; native resource_tracker reported four shared-memory cleanup warnings
per host. Audit-off performance services use the same frozen v2 source/packages.
online_workload.py is a bounded synthetic mixed4K/32K staggered closed-loop
diagnostic, not an official SWE benchmark. Actor arrival intervals must not be
reported as device step timings. Sixteen-card profiling and boundary evidence follow below.


## Online slowdown: output return versus D execution — 2026-10-02

Fletcher correctly required separating device cadence from token arrival.
Instrumented source cde4366 adds opt-in host receipts only: worker call boundaries,
Core output enqueue, output-thread dequeue, and native request-ID/actor arrivals.
No device synchronization or inference rounds are added by those receipts.
Native input processing appends a suffix to request IDs; analysis requires one
unambiguous match, not blind positional joining.

The synthetic16-session, two-turn4K/32K workload (512 outputs/turn,0.3s stagger)
passed lifecycle/output-count gates with audit off:
- hw86-online-perf1-baseline:432.400s,37.89 output tokens/s;
  actor one-token interval median17.239ms, P95=357.262ms.
- hw86-online-perf1-profiled:333.768s,49.09 tokens/s. Despite its artifact name,
  this is an UNPROFILED repeat: dynamic msprof attachment rejected all PIDs,
  including one verified single-worker PID. Do not describe it as a capture.
- hw86-online-perf2-observed:361.764s, with observer receipts and two bounded
  native torch_npu/CANN captures. Do not use this as profiler-free performance.
These are diagnostic all-turn timings, not official SWE scores or steady-state
D-only throughput. Final turn completion includes D backup/publication; the
prototype's generation RPC returns after the whole generation, not streamed
HTTP token delivery.

### Observed output boundary

Across32 completed turns,16341 actor arrivals matched Core enqueue receipts.
Each turn's Core-to-actor P95 was at most2.029ms. Representative slow-turn
enqueue P95=450.186ms and actor interval P95=450.499ms agree; the hundreds-of-ms
gap was already upstream of the output queue. Missing tail receipts from
buffered process shutdown are not invented. This rules out that return segment
as the dominant delay in this run, not every possible production frontend.

### Native device and source evidence

All16 NPUs have two raw captures. The first D window has ZERO graph replay
calls on all8 cards. Devices4/5 (DP2) have no real-attention slot-mapping anchor;
other active ranks show median device cycles about450–471ms. For device0:
37 interior cycles, median454.527ms; non-HCCL compute interval union median
11.350ms; HCCL interval union median369.550ms. HCCL includes synchronization
waiting, not just link bytes. Category medians are not additive, and uncovered
time is not proven host overhead.

A later device0 window has479 aclmdlRIExecuteAsync calls. Its476 fast cycles
have median23.386ms under profiling, versus16 slow cycles at472.359ms. Both
classes have approximately12ms non-HCCL compute coverage: the large difference
is not extra arithmetic. All four P instances execute real graph work in this
later window (one or two replays per TP rank), confirming overlap-capable
P/D operation rather than a D-only profile.

The source mechanism is explicit:
1. ep6_state_entry.idle_target_only forces CUDAGraphMode.NONE so idle EP
   participants do target collectives without writing live resident State or
   executing mismatched MTP collectives.
2. The pinned Ascend _sync_metadata_across_dp reduces graph modes using
   _post_process_cudagraph_mode, the minimum across DP.
3. Therefore one idle owner downgrades the whole D EP machine to eager.
   Native graph dispatch re-dispatches using that synchronized mode.

A host-receipt cross-check (50ms nearest-start matching, not an exact collective
ID join) found159/167 device0 host calls over100ms accompanied by fewer than
four active owners;1317 fast calls had all four. The static mechanism plus
native eager/replay contrast establish a concrete online occupancy-dependent
degradation path. A controlled safe-idle-graph intervention has NOT yet been
implemented or A/B qualified, so do not claim all residual latency explained.

**Next owned performance boundary:** design a graph-compatible target-only idle
participant that cannot mutate a resident session and preserves the exact EP
collective sequence. Do not simply delete the NONE safety override, fabricate
real requests, or couple State completion to model cadence. Recheck skew,
empty-owner transitions, resident bytes/leases, and loaded-request admission.

### Other measured latency and profiling reuse

First baseline phase wait medians: P store8.085s, D load2.670s, D store7.256s,
warm P load1.516s. These include shared-backend queueing, not pure PCIe or network
transfer time. The object service serializes operations per node and currently
holds its lock during PUT body reception; this is a separate, visible transfer
critical path, not a proven explanation for decode graph-mode degradation.
P generate RPC median0.238s (maximum5.032s); D generate median38.214s.
The final reply waits for State backup; streaming output delivery and durable
handoff completion need distinct latency metrics.

Native profiler start/stop is an opt-in private control RPC, using upstream
AsyncLLM/core/worker profiling. A daemon worker cannot run the automatic parser;
raw capture succeeds and must be exported OFFLINE with msprof --export=on
--type=db --output=PROF_PATH. Both captures retain raw data for every rank.
Exported DBs cover all first-window D ranks, later D TP0/TP1 of DP0, and all P
ranks/windows. The remaining later D DB exports were intentionally omitted:
raw files remain, and repeating expensive exports would not change this
diagnosis. Use analyze_online_device.py for any additional bounded DB, and
analyze_online_boundaries.py for same-host output joins. Never identify host
dispatch time as device cadence or sum overlapping stream durations.

Artifacts: hw86-online-perf2-observed/{boundary-summary.json,
device-summary-dp0.json,device-summary-first-all8.json}; raw/native profiles,
timing receipts and launcher logs under the same runtime root on both hosts.
All16 NPUs were released after capture (cleanup receipts8 idle/host).
The observer/control CPU suite passed8 tests. Source, exact gates and profile
evidence are backed up locally outside Documents; no model weights are copied.

## Idle graph participation and streamed replies — 2026-10-02

The first D8-only idle-graph qualification uses source6694588 and the unchanged
online-v2-D numerical capsule. Native wave/graph-bucket selection remains in
charge; no request or model round is manufactured to advance I/O. Idle GDN
publication bypasses resident address/selector normalization, then uses
negative verify slots. FA clears the consumer slot mapping (not merely its
native source) and reads the allocator-reserved zero null page. A peer-prefill
bucket uses empty cold prefill rows and routes outputs to one defined-zero
negative-slot verification token. It is not safe to assign a negative prefill
State slot to the raw chunk kernel.

The v1 gate failed before workload because importing Worker through the probe
client preceded plugin initialization. Keep diagnostic worker entry separate.
The v2 gate deliberately rejected a synchronized peer-prefill bucket; v3 added
the empty-row path, rather than silently allowing resident0 writes.

hw86-idle-graph-v3 exited0: all owners parked1K/16; owner0 alone produced128;
all owners returned32. The first32 tokens of owner0 match exactly between
skewed and all-active phases. Idle guard compares all target GDN tensors and
four control tensors, plus first64 kernel pages of each target FA leaf.
The pinned DPLB utility broadcasts/checks every engine but returns only engine0's
receipt (which has zero guarded tensors). Therefore v3's summary is not a
per-rank census; the subsequent probe explicitly gathers eight receipts and
requires six nonempty guards. Do not mistake its first-engine-only response
for either a six-rank evidence list or a non-broadcast RPC.

All8 exported native DBs contain160 graph replays. Device0's126 steady
slot-mapping cycles: median19.582ms, P9521.533ms, P9923.809ms; one142.082ms
prefill/transition cycle is not hidden. Device1 median19.540ms. Idle devices
have no real-request FA anchors, as expected. This is a short profiled1K skew
gate, not a matched production throughput claim. Raw/profile/timing/summary
artifacts are hw86-idle-graph-v3* under the runtime root.

fd8ea9b adds bounded token progress through actor/node, exact-token SSE ingress,
and separate generation-output and checkpoint-commit futures. A completed
response does not await D backup; a same-session following turn waits for its
previous commit, while a genuinely overlapping generation is rejected.
Disconnect/slow-reader delivery does not stop native State work. Thirteen
CPU stream/coordinator/object/timing tests pass; the unmodified SWE client also
accepts the frontend against a CPU fixture. Hardware streaming remains a
separate gate. Full active-generation sealed-page backup is NOT established
by this change: current checkpoint commands still require a retired frontier.

Real workload source is swe-prefix-reuse695dd8b. hw86's isolated
/workspace/swe-workloads contains512 accepted complete trajectories,
24227 turns, max context230362, source revisionfb0c0dccc7a5cce79b3f6de891848acdede36685.
The initializer verifies pinned metadata and source digest; its receipt records
527 examined rows and exclusions. HF mirror was used without implicit tokens
after direct endpoint failures. The official TraceLab v0.0.2 raw digest was
verified in the local workload checkout; its prepared empirical profile was
copied, not approximated from histogram bins. Plan arrival-gate-0.05-160.json
uses8 sessions, constant0.05 sessions/s,160s, codex gaps, seed20261002.
The profile's first-output gaps are synthetic completion-relative waits, not
measured user think time. Large datasets and profiles stay outside Git.

## Real online replay and object-service CPU bottleneck — 2026-10-02

fd8ea9b (online-stream-v1-source), byte audit ON: the4-owner1K/64-output
cold/warm lifecycle gate passed in34.313s. The official SWE arrival gate
(rate0.05,160s,8 sessions) is valid:32 requests,0 failures,0 missed due,
27 completed inside window plus5 drained. Window output46.30625 tokens/s
across16 chips, TTFT P953.360s. This low-offer audited run is qualification,
not a saturated throughput claim.

2018d7b (online-stream-v2-source) moves PUT body receipt outside Store locking,
with4 admitted bodies, audit OFF. Frozen rate0.2/300s plan: valid,60 new
sessions,142 requests,0 failures/missed due;93 completed in window,49 drained;
wall443.661s, window output63.07667 tokens/s total, TTFT P9546.648s.
This is an overloaded diagnostic point, not acceptable-latency goodput.
An artifact SCP overlapped the early window; do not claim a clean A/B speedup.
Partial/drain phase receipts showed P-store median8.15s and D-store7.15s,
versus P-load1.58s and D-load2.26s. Device request supply was sparse while
many controller requests queued. A10s late-drain NIC sample is NOT a link
ceiling (the two host samples were sequential).

probe_online_objects.py isolates CPU Store/HTTP/network from model/DMA.
Against the live v2 nodes,80MiB random opaque objects:
local PUT0.498s/GET0.367s, peer PUT0.512s/GET0.410s.
Four concurrent objects: local PUT180.5MiB/s, peer166.9MiB/s;
GET262.4/271.4MiB/s. Each object has unique content to avoid dedup hits.
They remain bounded LRU probe entries, not session checkpoints.

The standalone object-cpu-stage-probe used a separate1GiB Mooncake fixture
on explicit loopback55051 (closed afterward). Three80MiB direct Objects writes:
0.229/0.234/0.234s, Store calls only0.068/0.076/0.065s.
Reads0.239/0.223/0.218s, Store calls0.062/0.070/0.066s.
cProfile totals: SHA2560.655s of1.334s, byte joins0.129s, write slicing/glue0.140s.
This establishes unnecessary CPU serialization in the prototype, not a
physical network limit or a Mooncake native throughput ceiling.

548396a (online-stream-v3-source) prepares hashes/chunks outside the Store
lock, serializes collision check plus commit, snapshots immutable bytes under
the lock then assembles/verifies outside. Full-object SHA256 still validates
all bytes/order; redundant per-chunk rehash on GET is removed. Four bounded
download permits cover actual response sending, not only response creation.
Object CPU tests8 pass including concurrent HTTP byte equality, conflicting
versions, post-snapshot Store mutation, and failed publication. The wider
preceding stream/idle/controller suite passed22 before the final HTTP fixture.
The v3 real workload comparison is pending; do not infer a throughput gain
from the design or CPU fixtures.

Runtime evidence: stream-v2-object-probe.json, object-cpu-stage-probe/,
hw86-online-stream-v{1,2}-{front,timing}, and /workspace/swe-workloads/
stream-v1-gate and stream-v2-rate0.2. v3 services and same rate0.2 plan are
launched by start-online-stream-v3.py; model/kernel capsules remain unchanged.
The95MiB accepted source pool, empirical profile, plans and v1/v2 replay
artifacts are archived outside Git. Node object service currently shares
lifetime with the model host; CPU endpoint edits require expensive model
restarts. This is iteration friction, not a reason to hot-patch live workers.

## Lossless resident-wire candidate and audit switching — 2026-10-02

The v3 rate0.2/300s replay is valid:60 sessions,173 requests,0 failures/missed
due,131 completed in window and42 drained; wall400.817s. Output98.65 tokens/s
total across16 chips, TTFT P9529.359s. This improves the preceding diagnostic
point but remains queue-heavy; neither point establishes acceptable-latency
production goodput or a repeated A/B effect size.

The same opaque80MiB object probe on v3 shows four-way local/peer PUT201.8/
227.8MiB/s, GET515.0/527.1MiB/s. Single local/peer PUT0.538/0.490s,
GET0.279/0.332s. Removing CPU work from Store locking helps concurrent reads
especially; do not claim the memory/network hardware limit from this wrapper.

A sampled real committed resident object in that v3 replay is95,608,332 bytes
(63 lanes: target GDN and controls). Zstd level1 gives30,167,980 bytes in
0.117s, decode0.143s, exact bytes; zlib1 takes1.679s to encode, so is rejected.
A20,974,006-byte dense page shrinks only to16,540,219 with zstd, costing about
0.059s each direction; dense compression is not selected. These are bounded
single-object observations, not a universal compression ratio. Original State
bytes were not archived; immutable object keys and conditions are recorded in
stream-v3-lossless-probe.json. In this target-only baseline many candidate
lanes are zero; do not extrapolate this gain to active MTP.

Use the established python-zstandard binding rather than importing a large
Arrow dependency or maintaining custom C bindings. Pinned0.25.0, BSD-3-Clause,
installed ONLY in /workspace/pd-kv-layout-results/store-cpu-venv on both hosts.
The cp312 manylinux2014_aarch64 wheel is5,063,012 bytes; its PyPI SHA256 is
6dffecc361d079bb48d7caef5d673c88c8988d3d33fb74ab95b7ee6da42652ea.
The wheel and receipt are backed up locally; model Python/pins/kernels were
not replaced. requirements-wire.txt is an optional prototype dependency list.
Primary references: [release](https://pypi.org/project/zstandard/0.25.0/) and
[decoder contracts](https://python-zstandard.readthedocs.io/en/latest/decompressor.html).

online_codec.ResidentWireSink wraps the existing sink without changing State
lanes, checkpoint/DMA fences, or ObjectStateTransport. Only objects>=64MiB that
shrink to<=60% are encoded; dense pages remain raw. Standard Zstd frames carry
content size and checksum. Each call has a separate codec context (not shared
across threads). Before decompression, validate embedded content/window size,
no dictionary, checksum, and wire budget; reject trailing frames/bytes.
max_output_size alone is not the known-content-size allocation boundary.
CPU tests cover exact compressed/raw roundtrips, corruption/truncation,
oversized/unknown frames, trailing data, and opaque replica repair. The actual
Python binding reproduces the real resident sample exactly:30,167,984 bytes
with checksum, encode0.124s/decode0.126s (stream-v3-resident-wire-codec.json).

Set BETTERSCALE_PD_COMPRESS_RESIDENT=1 for the optional path. Its namespace is
qwen35-target-state-v3-zstd-resident/tp2/headN, distinct from raw producers.
Node readiness advertises state_wire; Coordinator rejects a P/D mismatch.
Reported CacheActions transfer_bytes remain logical DMA bytes, not compressed
network bytes. This optimization reduces CPU Store/wire work, not PCIe copies.

d79dd4c is frozen as online-stream-v4-source. Both nodes launch with compression
and byte audit ON. start-online-stream-v4.py first runs the4-owner1K/64 cold/
warm gate. Only after it passes does a private audit RPC disable verification.
The setter rejects any in-flight worker I/O, never launches a model wave, and
collects all16 rank receipts (previous=True,enabled=False) before starting the
unaudited rate0.2 run. This avoids reloading16 models just to switch an audit.
The hardware/compressed workload gate passed (12.607s), then all16 audit
receipts confirmed previous=True/enabled=False with no in-flight State I/O.

The v4 frontend also retains output-ready and checkpoint-committed boundaries.
D-stream-return measures actor-yield to coordinator callback only when D and
frontend share the explicit bind address; no cross-host monotonic comparison.
It is return-path latency, not device cadence. Actor arrivals remain in committed
turn receipts for joining the native Core/worker observer where useful.


## Compressed online baseline — 2026-10-02

The d79dd4c v4 rate0.2/300s replay is valid:60 sessions,278 requests,
0 failures or missed due requests,243 completed in window and35 drained;
wall404.603s. Total16-chip output261.91 tokens/s (16.3694/chip),
TTFT P955.548s. This is target-only, not active-MTP production throughput.
No bulk artifact copy overlapped this window. Short completion-relative plans
visit different subsequent turns when the system is faster: do not treat the
request counts or these single runs as a paired fixed-request benchmark.

The278 completed turns independently show output-ready precedes checkpoint
commit: post-output commit delay median3.193s/P9510.812s/max17.438s, excluded
from response completion. Controller State wait RPC medians/P95s: P-store
2.152/3.286s, P-load0.469/0.739s, D-store1.842/2.801s, D-load0.556/1.003s.
These are await durations, not isolated network times or device cadence.
There is still per-owner maintenance queueing beyond those operation waits.

Return-path jitter remains observable: across per-request actor-yield to
coordinator callback P95 values, median194.8ms/P95669.5ms; largest observed
single chunk1253.1ms. The late drained request's submillisecond return path
is not representative of concurrent load. This metric excludes native Core
compute and does not establish that decode cadence stalled. Keep native
profile and return-path analysis separate. Evidence is hw86-online-stream-v4-
front/control.jsonl, gate/summary.json, audit-disabled.json and the immutable
/workspace/swe-workloads/stream-v4-rate0.2 artifacts. The next0.5/300s point
and a separate bounded16-card real-workload profile are not yet qualified.


## Offered-load knee and real16-card profile — 2026-10-02

The same v4 system at0.5 new sessions/s for300s remains protocol-valid:
150 sessions,433 requests,314 in-window,119 drained,0 failed/missed,
wall417.479s. Total output230.55 tokens/s; TTFT P9547.692s. This offered
load is queue-heavy, not higher useful throughput. Do not continue blindly
to rate1.0. The earlier0.2 point is a useful lower-load reference, not a
saturation optimum.

TTFT is the first token emitted by P, before P-to-D handoff. It does not
describe that handoff's pause. Across all successful drained requests, v4
rate0.2 E2E median/P9511.852/43.354s; per-request average inter-token time
(after first token, including handoff) median/P9554.897/175.841ms. At0.5
those are48.262/101.483s and136.447/313.970ms. Output lengths differ and
these are per-request distributions, not token-weighted device cadence.

A separate90s/rate0.5 real replay captures all16 NPUs after30s, for12s;
all five actor start/stop receipts succeed, eight raw profiles per host.
Its throughput is diagnostic only. Receipt: online-stream-v4-profile-
receipt.json; raw hw86/hw81-online-stream-v4-profile; workload stream-v4-
profile. Native DP0/device0 has375 graph replays. Only105 real attention
slot-map anchors exist: a6.59s gap between those anchors contains dummy
participation and is NOT a6.59s decode step.

analyze_online_device now additionally correlates CANN replay connection IDs
to device TASK envelopes, including dummy waves, rather than calling host
API intervals device cadence. All375 launches have two correlated device
records. Device0 replay start intervals median22.821ms/P9572.696ms,
envelope median15.654ms, intervening gap median6.887ms. Two intervals exceed
100ms, max1004ms; globally idle demand can contribute, so gaps are not proved
host overhead. P0/device0 has only two replay/prefill events in this window:
P is sparsely supplied, not demonstrated compute-saturated. Raw remaining
ranks are retained; do not present one exported D rank as all-rank statistics.

c608372 batches up to64 already-ready Actor pipe messages per executor
wakeup, preserving per-tag order, bounded delivery and fail-closed behavior.
CPU tests cover burst bounds, interleaved tags, prior replies before EOF,
error quarantine, and output-before-backup. Node arrival/send timestamps
split the same-host D return path into actor-to-node, node queue and network/
coordinator phases. This candidate is frozen as online-stream-v5-source;
the audited gate and identical rate0.2 comparison are pending. No numerical,
State ownership, model pins or kernel changes accompany this output-path cut.


## Bounded return-pipe drain qualified — 2026-10-02

c608372 v5 passes the audited4-owner cold/warm gate in11.995s, then disables
audits with all16 receipts. Same rate0.2/300s replay: valid,60 sessions,
284 requests,250 in-window/34 drained,0 failed/missed; wall406.674s.
Total268.103 tokens/s, TTFT P955.027s. The small throughput change versus
v4 is not a repeated A/B estimate; the return-path improvement is much larger.

Across284 per-request return-path P95 values, median19.040ms/P9528.259ms
(max single chunk64.592ms), versus v4 median194.806ms/P95669.503ms
(max1253.082ms). Stage-P95 distributions (median/P95 across requests):
actor-to-node14.551/22.172ms, node queue4.367/9.876ms,
node-to-coordinator0.999/1.448ms. The measured delay was largely before the
node HTTP send, not the network RTT. Device cadence is a separate metric.
Evidence: stream-v5-return-summary.json and v5 frontend/workload receipts.

An idle-system CPU-only replica fanout probe uses fresh32MiB random objects,
three alternating-order repeats per setting, exact GET equality and both
HEAD receipts. Serial versus concurrent two-replica PUT median0.405/0.199s
for one object; four concurrent objects1.017/0.697s. Source remains opaque;
no NPU/State copy or active model workload overlaps this microprobe. It is
not end-to-end throughput. replica-fanout-probe.json and its driver retain
the setup. A bounded two-thread sink candidate now overlaps only immutable
replica PUTs, joining BOTH validated receipts even on one failure; it does
not relax checkpoint publication quorum. CPU success/failure tests prove
the caller cannot return while the other replica is still in flight.
Hardware and full-load qualification of this next cut remain pending.


DP1/device2's completed v4 export corroborates the all-wave cadence:
376 correlated graph replays,336 real slot-map anchors, device-envelope
start interval median22.704ms/P9573.968ms; five intervals>100ms, max1028ms.
Use the final online-stream-v4-dp1-device2-summary.json. An msprof final-name
DB appears BEFORE export completes and is incrementally populated: an early
read falsely showed zero slot-map anchors. Wait for the exporter to exit
successfully before analysis OR archiving; do not infer missing kernels from
a still-written DB. The first hw86 archive was retained with an .incomplete
suffix and was not delivered. Its replacement was created only after export
completion. Final v3-v5 archives are backed up locally (hw86~895MB,
hw81~11MB, SWE~0.95MB) and their full tar member streams were read successfully.


## Final measured envelope — 2026-10-02

1850d72 (v6, concurrent two-replica PUTs) passes the4-owner byte-audited cold/
warm gate in10.670s, then all16 audit-disable receipts. Numerical kernels,
State byte geometry, writer fences, graph policy and TP publication quorum
are unchanged. Compression is lossless; this remains target-only.

Unprofiled real SWE prefix replay,300s independent session-arrival windows:

| New sessions/s | Sessions | Requests | In-window/drained | Total output tokens/s | TTFT P95 |
| --- | --- | --- | --- | --- | --- |
|0.2|60|296|264/32|282.847|4.303s|
|0.3|90|361|307/54|278.430|14.273s|

Both runs are valid,0 failures and0 missed due requests; drain wall times
392.847/423.542s. Client connection queue is zero. No profiler, bulk SCP or
offline DB exporter overlaps either replay. Returning sessions use actual
generated token IDs. The window stops new turns then drains admitted ones:
these are NOT completed whole long trajectories (completed_sessions=0).
The differing continuation paths mean these are diagnostic offered-load
points, not matched fixed-request A/B trials or a formal SLO goodput claim.

Rate0.2 prompt median/P954846/14580 tokens, maximum23434; output median/P95
131/1605. Rate0.3 prompt median/P953458/12791, maximum23434; output125/1234.
Thus this campaign does NOT measure average100K-context production throughput,
even though the pool includes longer trajectories and earlier separate gates
cover262144. Do not extrapolate the measured compression ratio to active MTP.

Rate0.2 E2E median/P959.693/43.761s; mean inter-token time per request
(including P-to-D handoff) median/P9546.812/121.400ms. Rate0.3:
20.456/59.206s and82.830/258.374ms. TTFT is the first P token, not first D
token. In these points,0.2 has the better measured latency/throughput balance;
neither a saturation optimum nor a user-selected SLA is established.

Across657 unprofiled turns, per-request return-P95 median19.323ms/P9527.807ms,
max single chunk79.324ms. Output-ready still precedes commit: post-output
commit median3.116s/P9513.482s across the two loads, not billed to response
completion. State wait RPC median/P95: P-store1.809/2.997s,
D-store1.546/2.508s, P-load0.500/1.000s, D-load0.582/0.870s.
These waits exclude some controller queueing and are not isolated wire times.

A separate90s/rate0.5 diagnostic replay captures all16 ranks for12s after60s.
It passes, with all five actor start/stop receipts and8 raw profiles/host.
Export selection uses sender activity, not cherry-picked low device latency:
DP1 had371 sender events (DP0/2/3:100/371/269). Its completed device2 DB
passes SQLite quick_check, has378 graph replays and378 real attention anchors.
Device-correlated replay interval median24.122ms/P9570.425ms/P9998.095ms;
four intervals>100ms, max943.984ms. Graph envelope median17.605ms and
inter-envelope gap median6.683ms. Short attention-anchored cycles have
noncommunication coverage median11.834ms, communication (including waits)
4.350ms and uncovered6.569ms. Do not sum separate quantiles or label all
uncovered time CPU overhead; dispatch, globally idle demand, copies and
communication need their own attribution. This is the latest real-load
profile, not the earlier19.6ms isolated1K skew gate.

Remaining frontier: core CacheActions intentionally admits ONE incremental
transaction per attention owner, and coordinator maintenance holds that owner
through its completion. Streaming output does not remove this State admission/
handoff bottleneck. Removing it is a page-lifetime/publication protocol change,
not deleting a mutex: pending producers, shared-page borrowing, cancellation,
drop/eviction and TP failure quorum need new qualification. Active-generation
sealed-page backup and MTP bootstrap also remain outside this campaign.
No optimal-engine, HA/recovery or long100K capacity claim follows from this cut.

Evidence: online-stream-v6-{performance-summary,state-wait-summary,
device-summary,profile-receipt,export-selection}.json; hw86/hw81-online-
stream-v6*; /workspace/swe-workloads/stream-v6-{rate0.2,rate0.3,profile}.
Final code, launchers, accepted pool/plans and measured artifacts are preserved
outside the Documents repository. Stop after the model/State workload drains;
both hosts' v6-cleanup receipts confirm8 idle NPUs.


## Concurrent State handoff gate — October2, in progress

Fletcher selected removal of owner-wide State serialization and restoration of
MTP as the next two gaps. Do not interpret the earlier282.8tokens/s target-only
point as the16-card compute ceiling.

The isolated `online-concurrent-v1-{P,D}-package` capsules overlay only the
native cache seams on online-v2. Native arithmetic, pins, kernel payloads and
target-only execution are unchanged. `online-concurrent-v1-source` freezes the
matching prototype. Admission permits20 pending operations/owner, but preserves
per-seat writer/I/O fences, shared-object conflicts, host-byte admission and TP
quorum. Staging uses exclusive bounded lanes; an uncertain DMA quarantines its
lane and fails queued waiters, rather than recycling its storage. Coordinator
maintenance locks end after admission/dispatch, not after DMA/replica ACK.
P admission now reserves actual pages and at most16 execution seats/instance;
it no longer imposes an unrelated four-request cap.

`hw86-online-concurrent-v1-gate/summary.json`:16 independent sessions, two
turns,1024-token initial prompts and128 then8 output tokens;31seconds rounded.
Both P's four instances and D's four owners observed peak_pending=4. All
post-H2D object bytes matched, all TP completions retired, no pending I/O or
owned seats remained at the terminal assertions. This is concurrency/lifetime
and transfer-byte evidence, not a full model-quality or saturated throughput
claim. CPU concurrency suite58passed; the following MTP idle-wrapper tests
bring the focused suite to64passed, but do not qualify MTP hardware.

The next MTP capsule is explicitly experimental. Target-only wire omits the
drafter's one FA layer. Restoring the native full-prefix drafter without either
hydrating that layer or giving it an explicit restricted-context protocol is
invalid; it would read uninitialized draft history. The conservative first
gate restores P-side draft-prefix computation and transfers that FA layer in a
distinct `mtp-prefix-` wire namespace, then uses native MTP2 on D. It is not
the desired eventual P-no-drafter/D-local-bootstrap optimization. Do not label
this design as already qualified or silently substitute approximate draft
history for the baseline. No active-generation sealed-page backup claim.


Concurrent-v1's matched arrival-plan0.3/300s replay is valid:365 requests,
309 in-window/56 drained,0 failures/missed due,284.2533total tokens/s,
TTFT P9510.4044s, wall411.807s. Earlier serialized v6 at the same arrival plan
was278.430tokens/s and14.273s; this small throughput difference is not evidence
of a saturated speedup. Worker receipts across the gate/replay (not an isolated
steady-state/device window) show mean live target-only decode rows2.02–2.45 per
D group, far below16. Core cache admission/describe P95 stays below123ms,
whereas wait P95 reaches7.403s on P and6.450s on D. These waits mix transfer and
service queueing; they are not measured PCIe or wire times.

`hw86-online-concurrent-v1-quality3` passes24 real-chat code retrievals and8
exact warm/cold continuations. Both turns use the known code's token count as
the terminal budget, then append a valid new chat turn. Preserve earlier
fixtures:quality1 passed a Transformers BatchEncoding rather than IDs and
failed before generation;quality2 forced32 output tokens after the chat EOS,
creating malformed/partially repeated text in later prompts. Its post-EOS
divergences and truncated-code answers (also present in the independent cold
oracle) are not a State parity pass/failure oracle. Do not relabel them as
proven rounding. Render the template with tokenize=False then encode explicitly;
do not trim a represented recurrent frontier backwards just to repair a fixture.

Both concurrent-v1 services/frontend exit0 and all16 NPUs are released before
the next capsule. MTP-v1 fails before weight loading: native Worker.__init__
does not yet own model_runner. Set the experimental runtime flag only after
compile_or_warm_up_model, when the runner exists; v2 changes that seam. Neither
v1 startup failure nor v2 preparation is MTP acceptance.

A separate CPU DRAM fixture on explicit loopback55051,
`object-concurrent-native2`, compares8 unique32MiB objects/concurrency level.
Prehashed commit throughput C1/2/4/8:1103/1026/1049/836MiB/s;
detached chunk reads1149/1400/1695/1697MiB/s with exact reconstructed bytes.
This bounded observation does not prove shared-client thread safety for all
operations or production protocol correctness. It gives no reason to remove
the Store lock blindly: write scaling is absent. Earlier native1 reused payloads
across levels, so later writes were dedup hits and its write speedup is invalid.
The next useful localization is whole object encode/hash/HTTP/queue service,
not extrapolation from that deduplicated microbenchmark.

### Device-bandwidth qualification (2026-10-02; MTP deferred)

Fletcher requires approximately **20 GB/s per direction simultaneously**, not a
20 GB/s aggregate. Decimal GB/s below. Earlier ~1.7 GiB/s object reads measured
CPU Store throughput, not PCIe. No model services run during these probes.

Pinned independent DMA rings (128 MiB x 4, 8 GiB/direction/sample, one warmup,
three samples, exact all-byte checks), CANN 9.1.0 / torch_npu 2.10.0.post4:

- Card 0, NUMA 6: H2D 26.08, D2H 28.29, duplex 25.61/direction.
- Eight simultaneous cards, nearest-node binding: H2D 25.83–25.94 and D2H
  28.28–28.48, but duplex **15.57–18.09/card/direction**.
- Pair 0+2 sharing NUMA 6: duplex 15.67/16.52; pair 0+1 on separate nodes:
  25.61/25.36. Moving card 2 CPU/memory to adjacent same-socket node 7 yields
  19.47/19.03. This implicates shared locality resources, not a proven hardware
  ceiling or uniquely identified memory-controller limit.
- Eight-card CPU/memory binding `[6,0,7,1,4,2,5,3]` (device order 0..7), instead
  of `[6,0,6,0,4,2,4,2]`, yields duplex
  `[19.74,19.34,19.33,18.90,19.68,19.57,19.16,19.01]`, all exact. Both CPU and
  memory binding changed; this does not isolate memory alone. A candidate
  placement, not yet qualified with compute or production allocator ownership.

Artifacts under /workspace/betterscale-pd-runtime: hw86-duplex-dma-numa6.json,
hw86-duplex-all8/, hw86-duplex-same-node/, hw86-duplex-different-node/,
hw86-duplex-split-memory/, hw86-duplex-all8-split-memory/. The cohort uses
per-repeat barriers; source duplex_dma_probe.py. Do not extrapolate single-card
peak to the full host.

Actual TorchHostStateBackend synthetic TP2 geometry: 83 target/control lanes,
116,572,172 useful bytes/transfer, four disjoint seats/direction, 32 transfers;
allocation, selection, completion and release included; no codec/Store/model.
State_dma_probe.py uses distinct logical-block values and exact restore checks.
One 2048-token allocator block has 16 contiguous kernel pages; arbitrary
128-token fragmentation is a separate worst-case, not current allocation.

- Ordinary copies: D2H 22.73, H2D 24.24, duplex 13.80.
- First probe-only batched adapter: 24.37 / 23.67 / 19.54.
- Owned injected adapter with validation: 23.59 / 22.96 / 18.90, exact,
  hw86-state-dma-batch-owned.json. Host submit/wait is not device-event timing.

Reuse pinned native _C_ascend.swap_blocks_batch. The optional submitter in
models/qwen35/state_dma.py keeps completion/events/quarantine in host_state and
page_state. PageStateStore's event-free inner backend must remain inside its
batch event; standalone restore/audit still owns a real event. Online flag
BETTERSCALE_PD_BATCH_DMA=1 is opt-in, **not online-qualified yet**. Focused CPU
address, injection/audit and existing state-protocol suite: 54 passed.

Native Store pointer probe (64 MiB x 4 registered pinned buffers, unique keys,
local CPU/TCP DRAM, no HTTP/hash/codec/device DMA): PUT 1.779, GET 6.124 GB/s;
four Python threads 1.769 / 6.037, no fix. Exact bytes; artifacts
hw86-store-pointer-numa6/ and hw86-store-pointer-parallel-numa6/.
Installed non-CUDA 0.3.13.post1 is a wheel, not a local fork build. Store remains
unresolved; never remove integrity/lifetime fences merely for a faster number.

### DRAM fast-path correction and new primitive (2026-10-02)

Fletcher explicitly rejected treating HTTP object-service concurrency as the
solution: device State lands in DRAM first; incremental peer replication and
future disk demotion are background work. Required State readiness still gates
its own consumer. Do not make token output or native decode rotation wait for
object-service/disk acknowledgement. Do not replace D2H by P recomputation on
the current evidence. Existing tests use DRAM only; there is no disk I/O to blame.
MTP remains deferred.

The audited online-dma-v1 gate2 passed 16 sessions/all owners, 31.487s, with
batch DMA and unchanged wire path. Gate1 was a launcher context mismatch
(missing BETTERSCALE_PD_CONTEXT=262144), before model admission, not corruption.
Per-rank host phase medians in gate2: D load get .9755s, decode .1624s,
H2D submit .0052s/wait .0047s; D store submit .0098s/wait .0063s, encode .2599s,
put 3.4613s. Device events were not used for these host durations. D object's
PUT upload-admission queue p50 1.1629s; Store commit p50 .0367s. Thus widening
semaphores is not a substitute for removing the indirect payload path.

The subsequent online-supply-v2 HTTP-concurrency candidate was **aborted during
startup after Fletcher's correction**, not qualified. Its source archive is
historical only. The concurrency-knob edit was removed from working source.
D TERM released devices but left startup actor/node hung; after 120s, verified
owned remnants were killed. All16 cards returned idle. NUMA staging and codec
work remain optional/unqualified online rather than discarded.

Cold Store PUT diagnostics: each256MiB batch incurs ~65,536 minor faults,
1.756GB/s. Prefill1.5GiB of test-owned objects then remove them, using the same
2GiB segment: ~0 faults, PUT4.960GB/s; GET6.194GB/s remains similar. This isolates
first-touch cost but does not solve the remaining copy path. Source fork
c992ba75 transfer_task.cpp has one MemcpyWorkerPool worker; do not assume the
installed wheel is binary-identical without verifying its build identity.

The wire codec now joins memoryviews once rather than materializing every lane
and rejoining full payloads. Decode copies directly from the wire buffer into
owned host storage, no bytes-slice/bytearray copies. 95,607,470-byte CPU resident
fixture, exact wire parity: encode190.74→63.92ms, decode46.94→21.44ms. This remains
CPU/allocation-limited and is only an interim codec improvement, not the desired
DRAM direct path. Seven dtype/ownership tests plus focused tests pass.

state_numa.py binds State completion threads only (not compute threads or the
shared Store segment), with an explicit physical-device mapping and visibility
translation. hw81 PCI map matches hw86. Linux set_mempolicy maxnode is padded to
the mask word width; node+1 returned EINVAL in this environment. A child
numactl --show confirmed bind policy/CPU144–167/memory node6. Do not claim this
places existing cached allocations or provides the final cache arena design.

shared_dram_probe.py proves the necessary shared-memory primitive on hw81:
16MiB memfd MAP_SHARED, NUMA6 first-touch, aclrtHostRegister succeeds, and native
Ascend batch DMA performs exact H2D/D2H. A second independent process on card1
maps the same FD, registers it, reads the bytes through H2D, writes a new pattern
through D2H, and the parent sees exactly that pattern in its existing mapping.
Both unregister successfully. Registration took6.60/9.07ms, so it must be
amortized with reused registered regions, not done per request. This is no
throughput, eviction, distributed lease, or production qualification.
Artifacts hw81-shared-dram-capability.json and hw81-shared-dram-cross-process*
under runtime; Mooncake reference source uses the same HostRegister primitive.

Borrow before writing another LRU: mooncake-hust c992ba75 already contains
LocalHotCache (local_hot_cache.h/.cpp), memfd-backed shared segments, refcounted
hot-cache acquire/release and stale-fill generation guards. DummyClient
get_buffer can return a shared hot-cache pointer with RAII release; ordinary
RealClient get_buffer allocates and copies. The public mount_segment(path,...)
can mount MAP_SHARED file-backed global segments. Put-session APIs reserve and
publish native Store objects but do not yet expose a qualified externally-DMA-
written buffer lease. Assess these seams; never bypass eviction/lease fencing
by dereferencing a raw replica address or copying an expired descriptor.


### Native shared-cache seam and split completion qualification

The installed CPU wheel is mooncake-transfer-engine-non-cuda0.3.13.post1.
Reference fork source c992ba75 is not established as binary-identical.
store_shared_hot_probe.py, hw86-store-shared-hot2: first16MiB get_buffer
acquisition16.24ms via writable fallback; next three182/168/169us point into
the same read-only shared hot-cache memfd, all bytes exact. These are lease
acquisition times, NOT transfer throughput. Retain each BufferHandle through
DMA and drop all handles before closing DummyClient/native service.
BufferHandle ptr()/size() are methods; BufferLease ptr/size are properties.
The first probe accidentally retained a handle past shutdown and aborted;
this was a probe ownership bug, not corruption of the successful read.

BufferPool(dummy) fails in the installed wheel: “requires a store configured
with a local buffer”, even after setup_dummy(64MiB,...). Reference BufferPool
accesses the base client_buffer_allocator_ directly; DummyClient overrides
allocate_client_buffer through RPC but does not populate that allocator.
Its prewarm() is a no-op in the reference source. Do not assume a ready
pre-touched/reused shared staging pool from this API. The failed hot3 experiment
is preserved outside Git; the probe source now exercises only its qualified
read seam. Native hot cache also creates a second copy over Store backing,
not automatically the desired single full node-wide LRU.

CacheActions now has an opt-in two_phase_store protocol, disabled by default
and NOT wired into model workers/online launch. A staged receipt means the
rank's entire selected snapshot has safely left the device and has owned DRAM
lifetime independent of the device. TP staged quorum releases device I/O and
extra backup pins; committed quorum separately publishes the remotely usable
checkpoint. Final ACK must not inspect or mutate a seat reused after staging.
Per-rank commit may arrive before the other rank stages, but never before its
own staged receipt. Cancellation and post-stage replication failure do not
double-free or touch a new seat incarnation. No network ACK or disk durability
is implied by staged. 43 cache-actions/incremental CPU tests pass, including
quorum, reused-seat, duplicate/stale/error receipts and cancelled backup pins.
This is the scheduler half only; do not enable until the connector and rank
worker provide owned bounded DRAM leases and both completion receipts.

Read-only shared mapping is a concrete NPU compatibility gap, not yet a native
hot-cache H2D solution: hw81 memfd PROT_READ registration returns107017 with
ACL_HOST_REGISTER_MAPPED; using the header's ACL_HOST_REGISTER_READONLY (8)
returns207000 on this910B2/CANN9.1 setup. Both bounded probes retired cleanly.
Writable MAP_SHARED remains the previously exact-qualified primitive. Do not
advertise DummyClient hot-cache pointers as DMA-ready or silently make an
immutable mapping writable. A writable registered staging arena or an explicit
native cache adapter with equivalent ownership protection is still needed.
Artifacts hw81-shared-dram-readonly* and hw81-shared-dram-readonly-flag8*.
An additional52 focused DMA/codec/NUMA/object/coordinator CPU tests pass.


The narrow existing write allocator is MooncakeHostMemAllocator.alloc/free,
not BufferPool(dummy). It uses ShmHelper shared memfd storage; register_buffer
maps it into the native service, and DummyClient put_from/get_into use the
shared address path. hw86-store-shared-stage4 qualified three16MiB patterned
CPU roundtrips with one reused registration and explicit unregister/free.

native_dram_staging.py now composes that allocator into a bounded reusable
slot ring (temporary staging only, no per-worker full LRU). Writer -> draining
-> sealed -> sending -> released; DMA and replica failures quarantine instead
of reuse. Background pointer PUT retains its slot through ACK. Close rejects
active leases and failed teardown cannot masquerade as success on retry.
Four CPU lifetime tests pass. hw86-store-shared-ring5 passes the real CPU ring.

hw86-store-shared-ring-npu6: on card0, NUMA6 first-touch, a32MiB arena with two
16MiB slots is registered once with Store and CANN. Three distinct patterns
go NPU D2H -> sealed staging -> background native Store PUT -> pointer GET into
staging -> NPU H2D, every byte exact; drain, unregister and free succeed and
all cards return idle. Store calls take about10-15ms per16MiB in this small
cold fixture, NOT a saturation benchmark. This establishes direct pointer
lifetime compatibility, not20GB/s, State-layout packing or live PD wiring.
Remaining integration: State lanes into preallocated frames without byte
serialization, separate rank staged/committed receipts, one node-wide DRAM
cache/replica readiness, and bounded load-side native pointer ingress.


native_state_frame.py is the next bounded packing primitive: only JSON metadata
is serialized; typed State payload views point directly into the existing
registered arena. Existing DMA descriptors scatter/gather physical rows into
that frame, avoiding the full-payload encode/decode assembly. CPU tests verify
mixed bool/BF16/FP32/I64, reordered rows, legacy wire decoder compatibility and
pre-H2D metadata rejection. hw86-store-state-frame7 qualifies three native
card0/Store roundtrips of the mixed-dtype498-byte frame, including unaligned
lane offsets; all selected State bytes match. This tiny test proves addressing,
NOT realistic State capacity, bandwidth, untargeted-row guards or live wiring.
Source is a prototype; borrowed frame views must not outlive their arena lease.


Native shared-cluster gate (native_store_node_probe.py / replica_probe.py):
one bounded master on hw86:55301 and one512MiB native DRAM segment per node,
TCP endpoints10.244.2.32:55307 and10.244.1.16:55307. No disk/hot-cache duplicate.
A D-side pointer PUT with replica_num=2/preferred_segments completes in24.36ms
for16MiB; descriptor inspection confirms COMPLETE memory replicas on both
specific nodes, and D/P readbacks are byte-exact (7.49/9.35ms). These small cold
calls are not saturated bandwidth. Both owned services were cleanly stopped.
Installed DummyClient get_replica_desc RPC fails “rpc function not registered”;
a separate ordinary native client queries metadata successfully. Do not confuse
the unavailable inspection RPC with failed data replication or assume the
reference fork's registered RPC list matches this wheel.

The prototype native_state_transport.py now composes frame DMA, bounded DRAM
slots and background pointer replication, with a SHA256 trailer written in the
background before PUT. Load verifies that trailer before any device write;
optional post-H2D audit catches DMA errors. Its namespace is native-frame-v1,
not the older HTTP object format. Eight staging/frame/transport CPU tests cover
local-ready before ACK, safe source overwrite, sparse identity hit, corruption
before H2D, sticky errors, capacity and quarantine. CacheWorker can emit separate
staged/committed rank receipts and does not remove a newly reused slot when an
old remote ACK arrives;46 worker/actions/incremental tests pass.

IMPORTANT next correctness hinge: the prototype transport deliberately declares
supports_staged_receipts=False, so the live worker rejects two-phase activation.
Existing-object presence and an earlier PUT ACK do not by themselves pin Store
objects until the whole multi-object transaction is staged/committed. A finite
staging ring may recycle earlier frames after ACK; cache eviction could then
invalidate the claimed full DRAM snapshot. Do not enable by flipping the flag.
Resolve native Store lease acquisition/renewal/expiry and deadline handling (or
retain a bounded whole-snapshot staging reservation) first. Reference
RealClient get_replica_desc calls Client::Query, but its Python result discards
the returned lease TTL; map that protocol carefully to the installed wheel.
No raw replica-address dereference, permanent hard-pinning of the full LRU, or
unbounded per-worker mirror is an acceptable shortcut.


The installed-wheel lease behavior is now independently checked by
native_store_lease_probe.py (native-store-lease1): with explicit2000ms master
TTL, both ordinary get_replica_desc and batch_get_replica_desc prevent
non-force removal (-706). A second query renews protection beyond the first
lease's expiry; after the renewed TTL elapses, remove returns0. Queries take
.24-.43ms locally. Reference Client::Query/BatchQuery are uncached master RPCs
and conservatively set expiry to query-start + returned TTL. Do not substitute
batch_is_exist for these lease-granting queries.

Next route to evaluate: reuse these bounded native read leases, scoped to the
existing Core checkpoint/page references (release on existing CacheActions
drop), rather than inventing another full cache or permanently hard-pinning
objects. On a successful background PUT, acquire a read lease before recycling
the staging slot; batch-query existing pages before skipping D2H. Renew active
references before expiry, enforce conservative deadlines, and fail closed on
a lost/expired lease. The configured master TTL must be explicit and verified;
Python descriptor results omit TTL, so guessing it is not acceptable. A paused
or failed renewal must not be silently “recovered” while claiming uninterrupted
ownership. This is a proposed integration, not yet implemented or qualified.


Lease-ledger implementation frontier: native_store_leases.py now scopes
renewable native leases to checkpoint object-reference sets, shares sealed
page references, acquires on publication before staging reuse, and stops
renewing after the last checkpoint drop. TTL is a required verified launch
input. Slow replies, expired protection intervals and missing held replicas
fail sticky; no silent reacquisition across a gap. The renewal thread uses
condition/deadline waits and must drain before closing its native query client.
Four CPU lease tests plus two integrated transport tests pass.
native-store-lease-ledger2 additionally holds an actual Store object past the
initial2s TTL using the renewal thread: non-force remove remains -706 after3.2s;
after checkpoint drop plus2.2s, remove succeeds. Both single/batch query gates
also pass again. This is a CPU/native Store lifecycle proof, not a fault-recovery
or model qualification.

CheckedReplicas now requires the same ledger as NativeStateTransport: a
successful pointer PUT acquires its eviction lease before the ring frees the
slot. The transport registers/reuses immutable checkpoint manifests, keeps
references until release(key), checks lease health before staged publication
and complete readiness before final success. The live supports_staged_receipts
flag remains FALSE pending a combined native State DMA/CRC/replica/lease gate
and actual online-launch integration. The older isolated frame NPU gate did
not exercise this newly composed ledger transport.


## Shared DRAM registration performance boundary — October2

The native two-phase backend now has a combined NPU/dual-Store/lease/checksum gate. native_state_gate.py uses the canonical83-lane TP2 fixture:116,572,172 payload bytes, one resident selection and16 kernel pages, three patterns. hw86 writes/restores; hw81 independently restores, including untouched rows. Both memory replicas must be COMPLETE; checksums precede H2D and post-H2D readback is exact. This qualifies explicit staged-receipt capability, not the online model wiring.

Evidence: runtime native-shared-cluster-D2/state.json and P2 counterpart. D local-ready20–31ms versus dual-replica commit279–294ms; P restore290–337ms with audit. A subsequent20-slot run against512GiB/node interleaved native DRAM (native-online-v1-store-D/frame-phases.json) remains exact: local-ready24–28ms, commit282–299ms. Native get81–83ms, each SHA256 scan77–79ms, device copies18–24ms. No disk/model workload; do not attribute all delay to network or remove integrity checks.

Matched one-card DMA probe: hw86 physical0, CPU/memory node6, same swap_blocks_batch/stream, preallocated buffers, eight copies/sample, four measured samples, full transferred-byte oracle. Runtime shared-dma-match.json / shared-dma-match2.log:
- torch pinned contiguous H2D25.56/D2H27.96GB/s;83-lane24.90/26.92.
- Mooncake writable memfd + aclrtHostRegister(MAPPED): contiguous6.53/6.51GB/s;83-lane6.31/6.31.
This isolates an allocation/registration-path distinction without Store traffic. Geometry/network cannot explain it; the driver mechanism is not identified.

Changed-hypothesis controls, not fixes:
- HostRegisterV2(PINNED) succeeds, contiguous6.56/6.16; fragmented varies6–14GB/s (shared-dma-v2-pinned.json). Not qualified.
- Local MADV_HUGEPAGE before first touch succeeds but smaps reports no ShmemPmdMapped pages. Actual memfd pages4KiB, all32768 pages on node6; DMA6.38/6.49 contiguous,6.38/6.41 fragmented (shared-dma-thp.json/.log). Advice is not proof of huge-page backing.
Pinned allocation maps through /dev/davinci_manager. No global hugepage setting changed; host has no reserved hugetlb pool. All comparisons exact. Preserve known-fast pinned memory as comparison; fewer CPU copies alone do not justify losing most device bandwidth.

Opt-in native_state_runtime.py initializes away from compute threads with explicit State NUMA policy. Native services currently use node-shared interleaved cache, not owner-local NUMA allocation. Full checkpoint manifests retain shared objects during sparse restores. Presence checks do not grant leases; fresh replica queries do. Master TTL must match the declared verified lower bound.


## Accepted correction: private rank-owned DRAM, not shared memory

Fletcher explicitly superseded the shared-memory cache route on October2. EP8 shares expert computation, not attention State ownership. Each session has one attention-group owner per side; TP2 ranks hold the corresponding State shards in their own NUMA-local pinned pools. P/D copies of the same shard are intentional; tiny replicated control fields do not imply a shared KV pool. Multiple P instances must not each keep a full replica. Common-prefix sharing, if later enabled across groups, needs explicit ownership rather than assuming all bytes are mathematically disjoint.

The scheduler must carry P-owner, D-owner and TP-shard placement. Existing D owner_for(session) is stable; existing acquire_p may move P on capacity pressure and p_affinity is only in memory. That was safe with node-shared Store lookup, but is NOT sufficient for rank-local caches. Keep a warm session on its P owner, or explicitly migrate/fetch its required State before rerouting. Never turn an object-not-found on another rank into a cache hit. EP collectives remain untouched.

Stopped the two shared native CPU services and removed the unqualified online factory wiring. Superseded factory/source is archived outside Git at runtime/native-shared-online-superseded-source.tgz and native_state_runtime_superseded.py. The tiny real factory startup/20-slot2.705GB teardown smoke passed, but is historical, not the selected route. Shared-frame/lease probes remain bounded evidence only. An already-running alias-D2D diagnostic failed with507001/SDMA; it was not repeated. Process exited and a fresh device0 context passed exact1024-element roundtrip. No device reset or global host setting change.

New rank_state_pool.py is a bounded private immutable-object prototype. Device D2H writes its final owned host allocation; local seal needs a real completion, not replica ACK. Checkpoint refs and active readers separately prevent eviction; failed DMA quarantines bytes. CPU tests cover16 distinct owner pools, shared-page checkpoint refs, reader pinning, capacity, concurrent writes and failed DMA. The current gate remains synthetic, not online scheduling or model qualification. Retain two-phase completion, writer/page fencing and TP quorum from the earlier work; do not retain shared mapping merely to reuse its Store facade.


### Rank-private continuation and sticky placement qualification

User confirmed: choose owners only for a new session; every subsequent turn is sticky on both P and D. The coordinator now defaults to sticky_owners=True, persists independent P/D group assignments in SQLite rank_placements, chooses a new D owner from eligible groups by remaining KV capacity (not session hash), and waits on an occupied old owner without blocking unrelated groups. P uses the same capacity-aware first-choice/sticky rule. Legacy nonsticky routing is an explicit diagnostic option only. A request larger than its assigned owner's total budget fails explicitly rather than migrating or waiting forever. The scheduler owns session placement; TP shards stay inside that group.

Private-pool evidence:
- rank-state-gate2.json: real RankStateTransport,83-lane116.6MB objects, three patterns, exact known-value/untouched-row oracle and post-H2D audit. Final owned DRAM is sealed before replica callback; warmed local-ready10.6ms. D2H15.7GB/s and H2D20.1–20.4GB/s in this one-shot State path; this is NOT the25–28GB/s repeated-copy microbenchmark and NOT simultaneous bidirectional qualification. Network callback in this gate is a stub.
- rank-transfer-v1-D/P/result.json: separate actual Mooncake TransferEngine CPU0.3.13.post1 clients, P2PHANDSHAKE/TCP, private torch pinned objects registered in their owning rank; no memfd or Store. D NPU→private DRAM→P private DRAM→P NPU, three complete83-lane samples, SHA256 before receiver commit and exact State/untouched-row checks. Transfer writes116.6MB in168/123/113ms (~0.69/0.95/1.03GB/s). This is cross-host TCP, not PCIe bandwidth. P H2D5.84–6.29ms. Ready/commit files are bounded external gate orchestration, NOT the online routing protocol. All registrations and processes retired after completion.

rank_state_transport.py composes final host allocations with existing worker two-phase completion and complete manifests. Audit uses a separate scratch buffer so it cannot overwrite an immutable cached source with concurrent readers. Failed enqueue retains source/destination lifetimes; CPU tests cover this plus early local-ready, sparse manifest refs and identity hits. Online rank-address discovery, routed replication/fetch, incoming-object ownership/drop and model qualification remain unfinished. Do not launch it merely because standalone gates pass. MTP remains parked by later user instruction.


### Per-rank receive transactions — follow-on to472e6ef

rank_replica_receiver.py owns bounded incoming object reservations independently per immutable key. An unfinished object does not prevent an unrelated session from preparing/committing another object on the same rank. The receiving pointer stays pinned/registered until a successful synchronous TE completion is followed by checksum verification and deregistration. Failure quarantines ownership rather than reusing an uncertain target. TP-shard and operation identity mismatches reject before publication; capacity-only refusal returns retryable busy without poisoning the rank. A sealed object's verified digest is cached with that object generation, avoiding a scan of the full historical prefix on each incremental handoff.

CPU tests cover concurrent independent transactions, corrupt receive, operation/shard fencing, shared immutable-page reuse without rehash, and capacity retry. Native rank-transfer-v2-D/P/result.json uses the real receiver prepare/commit path with three83-lane samples and distinct private pinned pools. All values and untouched rows are exact; D→P116.6MB writes112–117ms (~1.00–1.04GB/s), P H2D5.75–6.23ms. Source/receiver registrations retire cleanly. This still uses ready/commit files for bounded orchestration; it is not yet an online control-service or16-card model qualification.


### Real private control/TE backend gate

rank_peer_control.py and rank_replicator.py replace gate-file handshakes with bounded per-rank private control RPCs. Only metadata travels through HTTP; actual bytes use the existing native TransferEngine. Source registration/read ownership spans actual TE completion; receiver checksum verification precedes local publication. Peer group/TP identity is checked against explicit sticky placement. The control listener cannot close over incoming writers, and quiescing closes admission before joining bounded handler threads. Completed-operation receipts belong to their checkpoint lifetime, not an arbitrary recent-N window; backend release must use receiver.drop to retire both refs and receipts without reversing lock order.

rank-rpc-v1-D/P/result.json is a combined real RankStateTransport + RankReplicator + RankReplicaReceiver + control RPC + native TCP TE gate on hw86/hw81. Three83-lane samples are exact, including untouched rows and post-H2D audit. Local D ready15.9–18.6ms precedes committed309–342ms; P H2D5.82–6.22ms and audit D2H4.46–4.47ms. No shared memory, Store, disk, gate-file payload handshake, model forward or fabricated decode step. Lifecycle/control CPU tests subsequently cover checkpoint-scoped delayed ACK replay and release-hook cleanup. This still is NOT the16-card model/online-load qualification.

Next integration boundary: worker factory must construct private pools/listeners on the State NUMA thread and set an explicit per-checkpoint peer route via the TP-local control path before store. Use the receiver release hook; wire incoming checkpoint adoption/retirement with Core host metadata rather than silently dropping backend refs underneath it. The coordinator's tiny checkpoint manifests no longer need the node object service. Keep source/peer publication and both TP acknowledgements distinct from the earlier device-local staged quorum. Existing model/kernel capsules and pins remain unchanged until a fresh candidate is staged and qualified.

### Private online wiring candidate after c06eec1

Opt-in BETTERSCALE_PD_RANK_PRIVATE=1 selects rank_state_runtime.py rather than
ObjectStateTransport. Runtime construction, native TE and the receiving listener
start on an explicitly NUMA-bound State thread; per-rank buffers use private
torch pinned allocation. Physical placement must match P instance/D DP owner
and TP rank. Candidate is target-only, uncompressed, with unchanged model pins.

Core sets each checkpoint's explicit opposite-side owner through a TP-local
pd_rank_peer utility before store. CacheActions receives the two-phase flag:
local staged quorum releases device state independently of final peer quorum.
The coordinator accepts rank-private-v1 only with sticky placement and matching
P/D wire versions. Small checkpoint metadata lives in its SQLite directory;
private nodes do not start Store or expose legacy object endpoints.

After full source/peer completion, recipient Core adopts metadata. Once the new
session generation is published, obsolete boundary checkpoints are dropped
through both owning Cores and their rank receiver release hooks, then metadata
is retired. This bounds superseded versions and preserves shared page refs.
It is not yet pressure-driven eviction of the latest inactive session checkpoint
or recovery after process/host loss; neither is claimed as qualified LRU/HA.
CPU tests include two sticky turns (D then P-only), no legacy object service,
peer-index validation, publish-before-retirement, route cleanup and topology.
75 targeted CPU tests pass. The runtime factory and complete16-card model path
remain UNQUALIFIED until a frozen candidate hardware gate succeeds.

The first real16-card wiring gate (online-rank-v1,13bd8e3) used the default
8192-context/8GiB device budget, not the262144/24.25GiB production envelope.
All16 workers and private listeners started, but the first P→D replication
failed closed: receive deregistration returned failure and no checkpoint was
published. /proc/3712770/maps on hw86 showed the SYSTEM Ascend Mooncake engine,
not the qualified CPU0.3.13.post1 wheel. Its AscendDirectTransport reported
aclrtGetCurrentContext failure in fresh commit handler threads. This does not
invalidate the CPU-engine standalone gate; it exposes runtime import precedence.
Stop both services; preserve quarantined buffers until process teardown, never
retry that checkpoint. The correction pins the isolated CPU engine site before
model/plugin imports, carries it into spawned workers and verifies module path.
That site contains Mooncake/zstandard/pip only, no replacement Torch. No installed
runtime or donor pin changes. Rerun from fresh services/output/SQLite directory.

### First model-qualified private rank path — 0b6baec

After the import fix, online-rank-v2-source plus unchanged13bd8e3 kernel/core
capsules passed real P4 TP2 / D DP4TP2EP8 gates, all16 NPUs. Both nodes use
private per-rank NUMA pinned pools and the CPU TransferEngine; no shared Store.
This first qualification deliberately retains the short8192-context/8GiB
device budget. Audit remains ON, MTP OFF.
- online-rank-v2-gate1:4sessions,1K prompts,16 then8 outputs;4.021s complete.
- online-rank-v2-gate2:16sessions,4K prompts,128 then8 outputs;9.346s complete.
All four attention owners participate on each side, warm turns stay sticky,
post-H2D checksums pass and output counts/frontiers are correct. Every P and D
group reaches4simultaneous State transactions; all transfers drain. Only latest
checkpoints remain per session after Core-mediated retirement (the second gate
shares services with the first, hence5retained sessions/group, not a leak).
These are audit-on lifecycle diagnostics, NOT saturated throughput or long-
context production qualification. Byte readback is exact; greedy token equality
between differently segmented execution is not claimed.
Services stopped after these gates; next run explicitly sets
BETTERSCALE_PD_CONTEXT=262144 (24.25GiB device State/rank) with fresh services
and directory. Local evidence copies: online-rank-v2-hw86/hw81-evidence.tgz in
the migration backup. Source, runtime pins, model capsules and task goal persist.

### Full-budget private State qualification and first real replay

online-rank-v3 runs the same0b6baec frozen source/13bd8e3 cache capsules with
BETTERSCALE_PD_CONTEXT=262144,24.25GiB State/rank. Audit-on gate1 passes four
32K cold/warm sessions and one exact262144-token boundary in54.525s. All TP
post-H2D byte checks and terminal no-pending assertions pass. Only then was audit
disabled on all16 ranks (online-rank-v3-audit-disabled.json) for actual workload.

swe-prefix-reuse695dd8b, existing512-trajectory pool and unchanged0.3/300s
arrival plan: valid,90sessions launched,519requests,490in-window/29drained,
0failed/0missed due. Window output498.8733tokens/s total16chips
(31.17958/chip including P), TTFT P950.90915s, offeredTTFT P950.91034s,
wall385.134s. All90sessions remain incomplete trajectories at deadline, as
expected for this finite-window protocol; do not call them fully completed.
Prior concurrent-v1 matched plan produced284.2533tokens/s and10.4044s TTFT P95.
This is a matched offered-plan observation, not a saturated capacity ceiling,
controlled single-variable speedup, MTP result or device-step timing.
The new run has no worker/device cadence instrumentation. The next1.0/300s
point uses the same live services; pool residency includes earlier gate/replay
sessions with distinct salts. Retained artifacts: runtime online-rank-v3-* and
/workspace/swe-workloads/rank-v3-rate0.3 plus rank-v3-server.json.

### Overnight objective update

Fletcher explicitly re-enabled MTP qualification and set the optimization target:
8D chips total5600–6400+output tokens/s, step below50ms, active decode KV near80%
of each card's actual available KV budget. Investigate raising D budget above
24.25GiB using EP8's smaller weight shard and avoiding D prefill/mixed capture.
These are targets, not proven hardware limits. Preserve correctness, report
effective accepted outputs (not speculative proposals), and measure cadence/KV
occupancy in the same workload window. P budget and graph needs remain separate.

Prior MTP-v2 failure has now been localized from retained logs: D
host_metadata.prepare rejects 'pure verification belongs to its small graph,
not mixed'. integration.capacity classifies by computed>=prompt, whereas
MTPFrame roles use drafts>=0. A warm one-token tail can therefore require
investigation of disagreeing role/graph classification. This is a source-based
hypothesis, not yet a hardware-verified fix. D-only small graph policy is a
promising joint correctness/capacity seam; do not merely remove the assertion.

### Higher-offer pressure failure: unreachable pinned versions

The following rank-v3-rate1.0 run is INVALID, not a throughput result. At
15:33:40UTC, D worker3760075 (physical3, StateNUMA1) failed a128MiB host allocation:
aclrtMallocHostWithCfg207001 / halMemAlloc drvRetCode6, moduleId7,vaFlag1.
Peer P failed closed; no unknown checkpoint was published. Driver evidence is
/root/ascend/log/debug/plog/plog-3760075_20261002153340079.log. A later diagnostic
read of NUMA3 was NOT that rank's selected node; do not attribute its free-memory
figure to this failure. Host MemAvailable was~1.46TiB; cgroup current~694GiB.
These do not establish a pinning limit or allocator root cause. Both services
and frontend were stopped, all accelerator workers retired; restart fresh.

Source inspection did identify an owned retention bug: pool.drop removed
checkpoint refs but kept all unreachable resident/tail UUID versions until the
128GiB logical pool threshold. Such versions cannot be restored by any current
checkpoint and mostly cannot be reused by a later version. Long replay therefore
accumulated dead pinned snapshots. Pool now collects sealed zero-ref objects at
checkpoint drop or last reader close, never writing/quarantined/read-held objects.
Shared pages stay pinned by the successor manifest. A100-generation regression
keeps only its referenced base page. Runtime memory RPC reports private pool
counts/bytes plus Torch pinned-allocator and device memory statistics.

Installed ATen/core/CachingHostAllocator.h rounds host allocations to
PowerOf2Ceil(size); payload bytes are not reserved host bytes. Runtime metrics
must distinguish them. Do not claim GC alone has fixed all NUMA/driver pressure
until repeated real load and allocator stats agree. Next candidate keeps graph
policy unchanged for this qualification; D-only graph code is opt-in and still
CPU-only. No disabling pinning, shared-memory workaround, cache drops under active
DMA, or host-wide memory knobs were used.

The first GC candidate online-rank-v4 (546116f) did not reach workload: P physical5
failed binding its private control listener with EADDRINUSE after model capture.
No stale model NPU contexts were present before launch; subsequent netstat did
not show a persistent56405 listener. This does not prove which transient socket
held it. The old56400..56407 control range lies inside the host's32768..60999
ephemeral range, so the native engine or another task-local outgoing socket can
race those late listener binds. Move the explicitly paired control endpoints to
27400..27407; preflight all8 and reject ephemeral-range overlap BEFORE model
startup. Do not change host sysctls or kill an unidentified listener. The failed
candidate never qualifies GC under load; both model services were stopped.

Experimental MTP-prefix private wire is prepared separately from target-only,
with85rather than83State lanes and a distinct object namespace. It requires the
D-only graph opt-in and --mtp capsule overlay (state_address/draft_fia), retaining
P-side draft-prefix computation for a correct first baseline. This is not yet
hardware-qualified and is not the eventual P-no-drafter optimization. The next
GC pressure gate still runs target-only with mixed graph policy unchanged.

A bounded real pinned-allocator GC gate (rank-pinned-gc-gate.json/.py in runtime,
hw86 device0/NUMA6) cycles1000 unique96MiB resident versions beside one20MiB
referenced page. All resident allocations reuse one address; logical pool remains
20MiB, then zero after final drop. Allocator allocated_bytes.current remains
160MiB (power-of-two buckets), then zero after host_empty_cache. In this no-DMA
gate, active_bytes.current instead accumulates to134251282432 and does not reset
on empty_cache despite physical address reuse and two actual host allocations.
Treat that active counter as unreliable physical residency evidence for this
lifecycle; retain allocated_bytes/counts plus pool refs/address evidence. The
counter's exact runtime accounting cause is unresolved. This gate qualifies
bounded allocator reuse, NOT concurrent model/TE/DMA pressure or the prior OOM.

### GC hardware gate and bounded ingress frontier

online-rank-v5 stopped before worker/model startup because the new preflight
imported HostStateKey before the model capsule was on sys.path. d8cd599 makes
that import local to checkpoint decoding; an isolated subprocess import test
covers preflight without BetterScale/Torch. online-rank-v6 (d8cd599 source,
unchanged13bd8e3 target-only capsules, full262144/24.25GiB, original mixed graphs)
then passes the16-session4K/128+8 concurrent byte-audit gate in10.007s.

The fresh-service1.0/300s replay is still INVALID: one request received a
streaming error at239.496s;1063 others completed,240sessions launched before
client fail-fast, wall346.673s. Both nodes stayed healthy, no host allocation
failure, all admitted work drained. Reconstructed committed-turn intervals
reach exactly128 in flight immediately before rejection. The old submit guard
throws at128, while successful output responses can still retain their permit
for background backup. The client discarded the server's specific error text,
so the bound is strong source/trace evidence, not a retained exact error receipt.
Do not publish a valid throughput from this run.

After drain every pool has one checkpoint per sticky session (256 total per
side, including16 gate sessions), zero readers. D private payload9.64–12.09GiB/
rank and allocator allocated_bytes16.88–20.45GiB; P6.77–14.11GiB payload and
11.70–24.76GiB allocated. No unreachable-version growth like the prior run is
observed in this envelope; this is not an unlimited-duration/LRU proof.
Before load D freeHBM is24,518,684,672bytes/rank at24.25GiB State, evidence for
testing a larger D budget after small-graph/MTP qualification, not a certified
maximum. Runtime online-rank-v6-memory-* receipts retain all ranks.

Coordinator ingress now waits for a fixed128 transaction permit rather than
rejecting the next request solely because outputs have outrun backup commit.
It does not raise owner/device/State budgets. Permit survives output delivery
until final commit, cancellation-before-admission creates no State, and failure
is rechecked after waiting. Frontend traces bounded error type/text and in-flight
count to avoid losing the next cause.26 targeted CPU tests pass; hardware
qualification of the new ingress is pending with the next candidate.

### MTP private wire / D-only graph first hardware boundary

online-rank-v7 (48b25fd, --mtp overlays on online-v2 capsules) starts all16
workers. D captures12graphs (six capacities/two banks), reporting0.42GiB
rather than the prior mixed bank2.07GiB. Four1K cold sessions complete16 outputs
with85-lane State transfer; the next warm turn fails on P before publication:
draft_fia -> context_parallel.plan.schedule rejects 'Only Q1..3 decode/verification
is admitted'. Preserve this as FAILED warm MTP qualification, not a passed gate.

Pinned proposer redispatches its unpadded num_tokens independently of target
classification. A17-query warm delta uses target mixed32, but closest draft key
is24 (verification/CP kernel); draft step0 still consumes the17 real target rows.
The candidate keeps draft dispatch at least the already-classified/padded target
descriptor's capacity, retaining bank identity, exact live queries and native
numerics. Do not relax the Q1..3 assertion or swap the capsule's native kernel.
stage_online_candidate --mtp now overlays draft_banks alongside state_address
and draft_fia.17 focused CPU tests pass; the next hardware run must qualify warm
continuation before any MTP throughput claim.

The target-capacity fix3e93c77 passes online-rank-v8-gate1: four1K cold/warm
sessions,16+8outputs,85-lane audit,4.692s. The16-session gate then fails on D
before its first-turn quorum, not on P: 'prefill reached a verification-only
graph'. The saved scheduler dump identifies a new restored request with
prompt4097/computed4096 joining one running decode: both are scheduled Q3,
and the new row receives native [-1,-1] speculative placeholders. Native GDN
role metadata still labels that one-token-tail row prefill, despite graph-policy
proof that D owns no bulk prefill. Single-session gates missed this join case.

The next opt-in D-only adapter normalizes ONLY a negative GDN role whose
computed frontier is exactly prompt-1, after validating Q1..3 and D capacity.
It clones the tiny CPU role plane; native proposal IDs, scheduling, sampler
validity, device acceptance and recurrent candidate selection are untouched.
Unknown/bulk-prefill roles still fail. This lets the qualified verification
kernel process the actual tail plus native dummy proposal slots, retaining
candidate selection instead of executing an inappropriate bulk recurrence.
Ten focused CPU tests pass; simultaneous-arrival hardware qualification is
pending. Do not call v8 a successful concurrent or256K MTP qualification.

online-rank-v9 removes the graph assertion and all16 concurrent4K requests emit
128 outputs, but checkpoint commit stalls. D owner2's saved snapshot has no
pending DMA/readers, and the non-first resident frontiers have been invalidated;
only the first request's terminal checkpoint was stored. Source Frontier.advance
still assumes every prompt-phase draft slot invalidates identity: the same native
Q3/-1,-1 tail admission advances cursor by3 instead of the one accepted real tail.
The candidate adds one narrow frontier case: exactly one prompt token remains,
query minus draft slots equals1, and raw sampled output is exactly one token.
Only that selected token advances the frontier; bulk/unknown draft cases still
invalidate.26 resident/frontier and D-policy CPU tests pass. v9 was stopped after
the stable stall, before warm or256K phases; preserve it as FAILED qualification.
Hardware verification of the paired role/frontier interpretation remains pending.

### MTP v10 qualified envelope and ingress pressure (2026-10-03 CST)

beb9c43 source plus --mtp overlays on online-v2 capsules passes v10 gate1:
16 concurrent4K sessions128+8 outputs, TP byte audits and exact262144 context,
61.830s. A separate16-session cold/warm code-retrieval oracle reproduces all
expected five-digit codes; not a general numerical parity claim. D-only graph
capture reports0.42GiB/rank. After gates D freeHBM24.291–24.318GiB/rank;
P only0.954–0.998GiB. Larger D budgets must remain role-local.

rank-v10-rate1.0 is VALID: fixed300sessions/300s,1666 requests,1506 completed
within window and160 drained, zero failures/missed due requests; wall408.441s.
Window output1512.8533tokens/s total16chips (94.5533/chip including P).
Window-completed-request TTFTP95=11.5914s; all1666 including drain has a higher
tail and is a different denominator. All trajectories remain incomplete by
the finite-window dispatch policy, not failed requests.

The client connection queue P95 is2.0723s; individual worst TTFT requests
exceed20s even with zero connection wait. D actor-to-coordinator per-request
return P95 values have median11.86ms and95th-percentile19.27ms, not seconds.
Source acquires a D seat/page reservation BEFORE P admission/computation.
Existing traces cannot separate D admission from P admission/compute exactly.
Do not label this an ingress CPU bottleneck or a proved D-seat bottleneck yet.
A frontend-only diagnostic adds session-commit/global-permit, D/P admission,
and P-generation timestamps, with16 coordinator tests passing; v10 model and
State capacities remain unchanged for the replay.

Fletcher authorizes raising D resident seats and execution concurrency under
State budget and roughly50ms step, rather than treating E16/R20 as a ceiling.
Candidate role-local D budget and configurable R20..80 are source-only; E16
metadata/graph/native-kernel boundaries still require coherent extension.
Capacity work is temporarily deferred to isolate the TTFT cause first.

The diagnostic repeat rank-v10-ingress-rate1.0 is VALID:1653 requests,
1495 in-window/158 drained, zero failures/misses,1493.25total tokens/s,
window TTFTP95=11.6127s, wall411.681s. Across all admitted requests,
D pre-reservation wait P95=11.459s/max18.076s; P admission P95~1ms;
P generation P95=.362s, pre-generation State work P95=.772s;
global permit wait P95=1.224s, same-session commit wait P95=0.
These stage percentiles are not additive and include the drain population.
This isolates a dominant scheduling wait, not ingress CPU exhaustion.

Fletcher corrected the contract: P calculation must not acquire D device
resources. cf8397a separates route_d (sticky host-cache destination, no State
seat/page decrement) from acquire_d after P checkpoint publication and release.
The bounded128 in-flight transactions also bound queued handoffs; rank pools
retain independent byte budgets. Queue state is a committed host checkpoint,
not a D device seat. D resources are acquired just before load/generate.
18 coordinator tests include all-D-full routing without resource decrements
and P completing/releasing while D admission is blocked. The frontend-only
v11 uses cf8397a with unchanged beb9c43 worker capsules/E16/R20/24.25GiB.
16-way4K/128+8 concurrent cold/warm gate passes8.933s (audit remains off).
The same1.0/300 replay is running; no improved throughput claim yet.
Check D first-token gap and queue wait as well as TTFT: earlier P first-token
delivery alone does not prove higher service throughput or lower total latency.

The first v11 full replay is INVALID at152.5s:575completed/26failed,153sessions. D first failure is aclrtMallocHostWithCfg207001 in PID3995840, propagated from State cache completion; later Gloo disconnects are consequential. This reused the v10 worker processes after two300-session replays plus gates, retaining previous sessions. Do not attribute this to queue semantics or claim a throughput improvement. Clean-process v12 reruns cf8397a with identical v10 numerical capsules/E16/R20/24.25GiB and audit off, isolating accumulated cache residency. The latest-checkpoint pressure-LRU gap remains real; fresh-run isolation is not a production fix.

Fletcher's cache policy correction: keep rank-private NUMA-local pinned pools;
do NOT add an intermediate pageable-DRAM tier. Target aggregate pinned-cache
budget is80% of available machine DRAM, apportioned by actual NUMA capacity
with system/transfer reserve. Disk capacity is insufficient here, so disk
writeback is deferred. This is target policy, not a successful pin-capacity
qualification. Current runtime still caps each rank at128GiB logical payload,
and Torch pinned bucket reservation can exceed payload. LRU must preserve
readers/writers/DMA lifetimes; evicting the last cached checkpoint without disk
requires a real miss/re-prefill path, never a dangling manifest hit.
The v11 failure plog identifies a128MiB allocation, drvRetCode6, drvDevId0.
That alone does not establish total DRAM exhaustion or an immutable pin limit.

Clean-process v12 passes1.0/300s:1689requests,1541in-window/148drained,
zero failures/misses, wall411.116s;1565.3767total tokens/s and window TTFTP95
1.69047s. All1689 stage populations: D queue wait P9514.049s/max23.782s,
P admission P953ms, P generate P95.385s, pre-generation State P95.941s,
global permit P951.209s. TTFT improved markedly because P is no longer blocked
by D; throughput changes only modestly and queueing remains. Do not call this
full service-capacity resolution. Memory-before/after receipts in runtime
online-rank-v12-memory-*.json. Queue semantics are hardware-qualified within
this bounded envelope; latest-checkpoint LRU and physical pin budgeting remain.

### Wider D execution envelope (source/CPU candidate)

Fletcher prioritized execution concurrency over pinned-pool expansion after the
valid queue replay. D-only v13 supply control uses unchanged C16/R20/24.25GiB,
64 simultaneous4K requests with4096 outputs then warm8. All lifecycle checks
pass71.054s. Worker-dispatch receipts show1250-ish consecutive full16-row waves
per rank; intervals median32.8ms/P95~43ms including a12s native profile.
This is HOST dispatch cadence, not an unqualified device-only latency result.
Raw v13 D profile and timing are retained; no claim of SWE-equivalent throughput.

The candidate exposes D concurrency16/32/48/64/80, default unchanged16,
separate resident seats(default E+4, minimum20, at most96), and role-local
D State budget8..48GiB. Planned first hardware point is C32/R40/44GiB.
Wider admission requires D-only graphs and MTP2. Process-local capacity reaches
the scheduler, State allocation, fixed metadata/sentinels, target/draft FIA
frames, GDN State/device address publication row tiles, and disjoint graph keys.
Do not widen only EngineArgs or the Python guard: two Triton address-publication
kernels previously processed exactly16 rows. P stays C16 with its old budget.

stage_online_candidate preserves the capsule's DP/EP initializer and changes
only its execution-row guard; a whole source initializer replacement would
revert the private D capsule to DP1. Existing v10 plan.py actually has
MAX_QUERY=16; its old error text says Q1..3. Source/capsule plan diff before this
change is just that error message, NOT proof of a Q3 native ABI. No new native
library is introduced for row expansion; wider hardware correctness is pending.

174 targeted Qwen35/cache/State/CP/frontend tests pass after repairing the
native CPU fixture's missing cache_actions=None (object.__new__ fixture had
skipped this existing constructor field). Five isolated capacity cases cover
16..80 graph keys, complete per-row CP partitioning, padding and resident
defaults; these are not substitutes for full-row device/cold/warm gates.
