# Remaining Cube bubbles: supply versus command dispatch

Offline analysis of existing EARLY-DOWN receipts, 2026-09-16. No new NPU job.
Reproduce with cube_supply_audit.py on capsules115222 and115316; compact output
is cube-supply-result.json. The helper validates24-core generations and uses
last-core end→first-core start. This measures gaps between instrumented routines,
not internal MMAD utilization. Readiness is the coordinator-observed prerequisite
of the NEXT actual command, not an omniscient pending-work counterfactual.

## What the present scheduler already does

persistent_vector.cpp independently examines two live slots and prioritizes ready
down, otherwise ready up. Heterogeneous115316 contains6/9 up→up and6/9 down→down
transitions on servers0/1. It is not hardwired to alternate one pull and one
complete up/down batch. Resident kernels still use one CCMD and a24-core join,
so this is whole-command work conservation, not arbitrary tile work stealing.

## Largest measured gaps

Fully paired115222:

| Transition | Server0 median | Server1 median |
|---|---:|---:|
|up→down,24 transitions|8.16us|7.62us|
|down→up,23 transitions|253.98us|281.24us|

For the23 down→up transitions, only98.00/98.16us TOTAL occurs after the next
whole-command prerequisite is observed ready: about4.26/4.27us per transition.
The remainder precedes next pack readiness. Do not call it all physical data
absence: partially ready tiles are deliberately invisible to the current API.

Coordinator-time decomposition, adjacent fully paired waves (medians):

| Interval | Server0 | Server1 |
|---|---:|---:|
|down observed done→SEND observed done|44.50us|42.22us|
|SEND done→next FETCH issued|68.40us|109.00us|
|first FETCH issued→whole PACK done|134.88us|132.82us|
|PACK done→up command issued|0.28us|0.28us|

These component medians need not sum to the median gap. Individual FETCH commands
are about34us, PACK about41us; the134us aggregate also includes multiple source
fetches and intervening grouping/control. It is not a single134us DMA.

The burst harness pre-enqueues a finite graph of requests, but each source's
bank.body executes publish→collect→retire before its next request. There is one
outstanding frame/source, and the server consumes both sources in a paired wave.
After completing that wave it has no third independent source ready to hide the
return/next-publication turnaround. A pre-enqueued host graph is NOT a device
mailbox already containing24 independent jobs. Neither measured interval proves
CPU scheduling delay: this fixture has one host episode replay.

## Priorities

1. **Change data readiness granularity rather than invent a generic Cube queue.**
   Expert-major pack progress can start up before the entire41us pack ends.
   Current source-route-order pack cannot safely publish expert prefixes; use
   inverse routing from a frozen catalog. Preserve the one-fetch-per-token path
   initially; direct per-expert remote reads have a top-k traffic cost.
2. **Overlap partial down result return with remaining down.** FIX-ordered tile
   completion can reduce the exposed42–44us SEND tail. Return writes must cover
   disjoint source route/output slices, and final DONE still waits for all
   contributions. See DFC-MICROSCHEDULE.md for the pairing/visibility boundary.
   This does not promise the whole tail becomes free under HBM contention.
3. **Test saturated supply separately.** More genuinely independent attention
   lanes or bounded source frames can provide next work during current down.
   Do not submit dependent future layers before their input exists, or loosen
   source buffer reuse to manufacture saturation. A saturated stage benchmark
   and dependent two-source latency are different metrics.
4. **Then consider ahead-of-time Cube descriptors.** A bounded ping-pong command
   slot could remove some of the remaining4–9us routine-transition gap. Publish
   immutable descriptors before current work finishes; keep per-generation
   readiness and retirement, do not overwrite data another core still consumes.
   This cannot eliminate the250us supply turnaround on its own.

Avoid alternating tiny up/down invocations per expert merely to make the trace
look busier: that can restart tile resources, disrupt weight access and still
requires all top-k contributions before attention continues. Current broad-case
per-core skew is modest (GMM-SCHEDULING.md), and existing early-down tail waits
are about0.22us. A generic work-stealing scheduler has no measured justification
here. We have not ruled out intra-GEMM stalls; that requires finer hardware evidence.

## Interpretation of heterogeneous work

115316 still has up→up/down→down transitions and shorter down→up medians
147.04/155.26us, but wave composition differs. It shows the ability to consume
another slot, not a controlled speedup from arrival skew. No code policy changes
or claim of recovered service throughput follow from this offline audit.

For measured ready work INSIDE a wave, enter [EXPERT-READY.md](EXPERT-READY.md).
The opt-in row-completion observer confirms expert inputs are ready before the
full PACK boundary; it must not be interpreted as a new optimized scheduler.
