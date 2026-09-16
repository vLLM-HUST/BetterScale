# One up invocation with internal prefix publication

This opt-in continuation tests the scheduling hypothesis without changing GEMM
tile math. `DEVICE_SERVICE_INTERNAL_PIPELINE=1` selects mode2 of the existing
persistent protocol. Published worker/default paths remain unchanged.

`actual_gmm.cpp::ActualGmmTypes` supplies the same external CATLASS Mmad type to
both controls. `streaming_gmm.hpp` owns only the group traversal and publication
hook; matrix loads, MMAD, FIXPIPE, buffers and swizzle remain external CATLASS.
The tile object survives the prefix boundary. A drain plus PIPE_ALL precedes
per-core generation publication; all24 prefix completion lines are joined before
AIV consumes those rows. Up continues without another CCMD or tile constructor.

Down remains one complete invocation after both activation segments finish.
This is deliberately an isolated up/activation dependency cut, not a claim of
full DFC dispatch/up/activation/down/combine pipelining. Input pack and final
return remain batch boundaries, and prefix publication still passes through the
persistent coordinator rather than DFC's cross-core hardware flag path.

Two cut policies are inherited: half of live rows at a whole-expert boundary,
or last N nonempty groups (`DEVICE_SERVICE_SEGMENT_TAIL_EXPERTS`). Empty segments
remain legal. Per-core prefix timestamps let the analyzer verify the actual
producer join, even when activation0 completes before the full up interval ends.
Timestamp instrumentation is optional in execution but required for that detailed
internal-pipeline causal audit.

Leaf `runs/persistent-control-20260916T072414Z` passes broad/hot/zero/skew/one-row,
independent random BF16 expert weights, source immutability, output guards and
bounded invalid/missing descriptor gates. Four-card gates and bounded timing controls are recorded below.

Build `build_persistent.sh`; run through the established idle-subset launcher.
The initial compile rejected ambiguous MatrixCoord integer constructors; explicit
uint32 coordinates fixed that compile-only issue before hardware execution.

## Fresh four-card controls

Same server cards5/7, 24 paired waves/server,32 rows/source, changing inputs and
alternating layers, identical per-core instrumentation. All48 outputs/run pass.

| Mode / run suffix | Server0 pack-to-return | Server1 pack-to-return | Math chain0 /1 |
|---|---:|---:|---:|
| Unsegmented /072514 |616.87us|574.07us|522.93 /484.55us|
| Internal half /072733 |592.90us|542.41us|509.13 /466.87us|
| Internal tail2 /072800 |626.79us|598.70us|540.80 /514.17us|

The half-cut reduces median math-chain time2.6%/3.6% and pack-to-return3.9%/5.5%
in this control. Up medians311.31/290.63 ->307.61/287.57us remain close; the result
supports a scheduling gain rather than a faster GEMM tile. It is not a stable
serving throughput improvement or DFC parity. Episode envelopes include client
startup/gaps and are not used as the optimization claim.

Same-wave activation/up work overlap is373.18/208.00us total for the half-cut.
Tail2 has ZERO such overlap. Measured from the last producer-prefix timestamp:

| Mode | Time until first AIV consumer0 /1 | Remaining up0 /1 |
|---|---:|---:|
| Half |23.16 /5.85us|150.14 /133.95us|
| Tail2 |14.46 /16.03us|3.60 /4.06us|

Thus the narrow tail window expires before this coordinator-mediated consumer
starts. Moving the publication into the group loop preserves tile resources,
but does not remove GM polling/coordinator/AIV dispatch latency. This gives a
concrete reason not to copy the DFC tail-two policy mechanically.

Retain opt-in status. Dispatch/pack-to-up and down-to-return are still coarse;
this experiment has not measured their available gains, nor established that
all of the remaining DFC difference comes from scheduling. No new DFC binary
measurement was run in this round.

Heterogeneous source32/1 rows with2ms delay also passes48 outputs and the full
prefix-generation causal audit in `runs/remote-dfc-control-20260916T072920Z`.
It exercises nonuniform naturally formed waves and slot reuse, not only paired
startup. Exact counts and all summaries are in `internal-pipeline-result.json`.
The run analysis directories contain compressed per-device-relative phase/core
views; no new independently aligned multi-device clock claim is made.
