# Ready expert work inside a wave

2026-09-16 investigation of Fletcher's question: many experts are hit, so is
there enough work inside one wave to keep Cube busy? This is an observation
experiment, not a new scheduling implementation.

## Added evidence

Opt-in DEVICE_SERVICE_PACK_TIMING=1 records each packed destination row after
the existing Transfer::Copy has completed its MTE3_S wait. No new payload fence,
expert join, or tile drain is inserted. Each writer owns a64-byte observation
line; timestamps plus source/layer/route provenance are exported only after the
episode. There is one diagnostic16MiB allocation when enabled, zero when disabled.
The store/cache flush still costs time; it is not free or performance-neutral.

The observer requires internal timing and excludes resident/quantum movers.
Configuration index19 is the optional observer pointer. Existing slot ownership,
readiness, command dispatch and published Worker defaults do not change.
Capsules must pair the new config with the new frozen binaries.

Reconstruction is deliberately restricted to remote_dfc_control's deterministic
balanced burst. expert_ready_audit.py reconstructs the actual source generation,
layer and routes, verifies every destination index and contributor, then takes
the maximum row-completion timestamp for each expert. It validates full row
coverage, worker ownership and command intervals; it does not guess grouping
from adjacency in the timeline.

Down-input readiness uses existing fully completed activation segment receipts.
That is a conservative upper bound on when an individual expert became ready;
we did not add synchronization inside vector activation or treat an up tile call
return as completed FIX output. Waiting intervals end at the earliest core's
routine start, also conservative relative to actual per-expert consumption.

## Bounded result

Same devices2,3,5,7; expert servers5/7:
- control123806, pack observer disabled;
- observer123834, enabled.

Both complete48 checked outputs, changing inputs/layers over24 paired waves per
server. No NPU runtime/provider is modified. Only the prototype is instrumented.

Each server wave has256 routed rows over64 experts:384 up tiles with the current
M128/N256 geometry. There is considerable potential work *once inputs are ready*.

| Relative to earliest up routine start, median | Server0 | Server1 |
|---|---:|---:|
|First entire expert's packed inputs ready|25.97us earlier|32.34us earlier|
|Half the experts ready|21.63us earlier|25.89us earlier|
|All expert payload rows ready|14.36us earlier|17.41us earlier|
|PACK command duration|41.87us|48.41us|

This directly establishes some ready expert inputs wait behind the full PACK
boundary. The all-row timestamp does not include trailing scan/observation work,
worker retirement or command delivery; it is not equivalent to PACK's end.

The first activation segment is already complete while the remaining up command
continues for median129.73/114.70us. Those down jobs are available but Cube is
busy on up, not idle. Reordering them alone cannot create more compute capacity;
it becomes useful if it enables activation/return/next-input overlap.

Between measured Cube routines across the episode, union durations with proven
ready up work total604.00/733.68us; proven ready down adds173.84/200.88us.
These sums do not multiply time by expert count. Total routine-gap time is
5453.60/7950.66us, so the observed ready-work intervals do not explain all gaps.
They are neither an upper bound on all latent work nor a predicted speedup.
Routine envelopes include waits and are not MMAD activity measurements.

## Observer cost and limits

PACK control→observer medians41.44→41.87us and43.25→48.41us.
Pack→return564.92→558.47us and527.73→534.27us; episode durations also vary.
Thus the second server has material instrumentation/run sensitivity. Use the
ready-window numbers diagnostically, not as guaranteed uninstrumented savings.
No claim that a32us visible interval can all be hidden under resource contention.

The evidence favors expert/range-ready PACK→up as the next bounded prototype:
keep the batch and group offsets frozen, publish completed expert ranges, let
the same Cube tile traversal wait before reading each required range. Do not
reopen admission after those readers start. Current source-route-order writes
require joining all source/mover contributions; a per-source completion alone
does not make a shared expert ready. Expert-major packing is an implementation
option, not yet selected or benchmarked here.

We did not collect individual up-tile FIX completion or build a counterfactual
tile scheduler. DFC-style down→return remains another candidate. Adding more
independent sources should be a separate saturation experiment, not an explanation
that dismisses the ready work now observed inside a wave.

## Reproduce and view

Build build_persistent.sh into runs/attention-pack-ready-build. Use that absolute
PERSISTENT_BUILD with INTERNAL_PIPELINE=1, EARLY_DOWN=1, INTERNAL_TIMING=1,
PERSISTENT=1, PARALLEL=1, BURST=1 (all DEVICE_SERVICE_ prefixes); toggle
DEVICE_SERVICE_PACK_TIMING for observer/control. Run through run_remote_dfc.sh
with four admitted devices, then expert_ready_audit.py <capsule>.

Compressed visualization:
runs/remote-dfc-control-20260916T123834Z/analysis/expert-ready-relative.json.gz

Expert tracks show **ready-but-not-yet-started** intervals, not GEMM execution.
A separate track shows the measured Cube-team routine envelopes. Server clocks
are independently zeroed, not cross-device calibrated.
Compact results: expert-ready-result.json.
