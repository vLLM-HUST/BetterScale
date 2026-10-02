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

## Current integration candidate (not hardware-qualified yet)

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

## First bounded gates and artifacts

52 CPU tests passed for adopted lifecycle gates, control-only dispatch,
object bytes/geometry/failure quarantine and shared object publication. Additional
admission and byte-audit tests are in `test_online_coordinator.py` and
`test_qwen35_page_transport.py`. Exact post-H2D object readback is optional and
must be disabled for performance measurements; do not include audit copies in
claims about production transfer cost.

Runtime roots on both machines remain `/workspace/betterscale-pd-runtime`.
`online-v1-source` is frozen from69159a9; staged P/D packages overlay only the
listed cache seams on the already qualified task capsules. All16 NPUs were
checked idle before launch. hw81-online-v1 / hw86-online-v1 launchers use
context262144,24.25GiB State/rank and512GiB DRAM Store/node, with byte audit on.
The `hw86-online-v1-probe` gate is pending: four-owner cold/warm PD turns.
Do not promote startup, CPU passes or submitted work to hardware qualification.
