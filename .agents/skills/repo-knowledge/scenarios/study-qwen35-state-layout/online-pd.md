# Integrate incremental State with two-host online PD

Enter for the follow-on to `dual-host-pd.md`. Fletcher clarified on2026-10-02:
`codex/qwen35-incremental-cache` is the ready local cache implementation to
integrate, not an already completed cross-host connector. Do not wait for a
nonexistent newer branch or repeat naive whole-checkpoint profiling.

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
