# Yielding pull/pack at bounded AIV boundaries

2026-09-16, opt-in diagnostic atop the internal-prefix pipeline. Released worker
and default prototype behavior are unchanged. Enable
`DEVICE_SERVICE_INTERNAL_PIPELINE=1 DEVICE_SERVICE_MOVE_QUANTUM=128` (or256).
Zero retains the previous non-yielding path.

## Scoped implementation

The two slot records already form a bounded ready set for Cube: it may execute
another ready slot while AIV prepares data. This change does not add a generic
queue or split the16 movers into two teams. It removes immediate pull-to-pack /
next-chunk chaining in the experimental path. After each completed chunk,
return/activation readiness is reconsidered before pending pull/pack is resumed.
A currently executing DMA remains non-preemptible.

A quantum is128/256 route rows for pack and16/32 unique input rows for pull.
The cursor uses concatenated ACTUAL source lengths, not source-capacity padding.
Descriptor ownership, claimed source generations, and completed-copy cursors
survive suspension. Sources may join at the completed fetch boundary as before;
there is no intentional batch-wait period. Complete pack still gates up, and both
activation segments still gate whole down. Thus this is not the full per-expert
ready-work design or an autonomous Cube/AIV hardware-queue implementation.

The current experimental implementation rereads routing maps and goes through
coordinator command publication for each chunk. It intentionally measures whether
this cheap implementation is enough before building a new internal mover engine.

## Gates

Leaf `runs/persistent-control-20260916T073824Z`: independent random BF16 expert
weights, broad/hot/zero/one-row/512-route skew, immutable sources, output guard and
unowned poison checks, invalid descriptor and missing source watchdogs all pass.
Three four-card runs pass48 changing input/layer/route outputs each, with32/1-row
sources and2ms source1 delay. The extended analyzer validates all pull/pack chunks
precede dependent math and all producer-prefix/activation/down joins hold.

## Measured result: less blocking, no net win

Same cards2,3,5,7 (expert servers5,7), identical source work and instrumentation.
These independently arriving sources form different natural batches; do not claim
matched per-wave work or a precise causal episode speed ratio.

| Mode / run suffix | Waves0/1 (paired0/1) | Pack-to-return median0/1 | Server episode0/1 |
|---|---|---|---|
| No yield /073930 |41/40 (7/8)|556.62/517.66us|16.923/16.864ms|
| Quantum128 /073957 |38/40 (10/8)|614.46/544.36us|19.842/19.751ms|
| Quantum256 /074024 |38/40 (10/8)|591.62/512.91us|18.130/18.052ms|

We specifically intersect actual pull/pack core-routine intervals with the wait
from full up completion to first execution of the last activation segment. Total
intersections per server fall125.78/129.38us ->24.38/24.88us at quantum128, or
61.64/60.86us at256. This locates the observed AIV head-of-line overlap; it is not
a claim that every microsecond of that intersection is independently removable.
Most waves had no such overlap even in the control (only6/5 waves did).

**Decision:** do not enable chunking by default. Smaller chunks reduce this
blocking but do not improve the measured full episode. Extra command handoffs and
repeated map reads are plausible overhead sources; their separate costs are not
isolated. The results do not reject finer internal scheduling. They reject treating
many complete externally handed-off chunks as a free substitute for it.

A deeper design would retain mover descriptors/cursors inside its running team,
consume bounded ready work without returning through a whole command handoff,
and publish usable expert intervals early. That remains unimplemented; this
prototype does not claim to have eliminated the full-pack or full-down boundary.

`yielding-moves-result.json` preserves compact receipts. Each run's `analysis/`
contains compressed per-device-relative phase and core timelines. No independent
cross-device clock alignment or fresh DFC performance comparison was performed.
