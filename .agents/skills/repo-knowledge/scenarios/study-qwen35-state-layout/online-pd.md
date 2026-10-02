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
reported as device step timings. Sixteen-card profiling is still pending.
