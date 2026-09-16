# Continuous down with late activation readiness

Opt-in DEVICE_SERVICE_EARLY_DOWN=1 requires DEVICE_SERVICE_INTERNAL_PIPELINE=1.
No published Worker or default policy changes. This implements the first
prototype in DFC-MICROSCHEDULE.md, not per-expert pack or partial return.

## Contract

Coordinator can issue whole down after up finishes and activation prefix joins.
A single tile object traverses all experts. Before the first tail GM→L1 issue,
the scalar issuer polls a slot-specific, globally unique down command generation.
Prefix tile work need not be drained at this boundary. The coordinator publishes
tail readiness only after all16 AIV activation completions (ordinary or urgent).
Down preserves its ordinary complete24-core retirement and final SEND.

Two control lines85/86 fit the existing96-line allocation. downGen resets at
slot retirement; command generations never repeat within a bounded episode.
The flag cannot authorize a newer slot generation. Watchdog/STOP terminates
missing readiness. Empty segments are legal; an empty tail with no issue needs
no read guard. The fixed source catalog, count catalog and staging buffers stay
alive through complete SEND; this does not allow admission to grow mid-compute.

streaming_gmm.hpp shares the up/down traversal but keeps the external CATLASS
tile implementation untouched. Unlike restarting two separate down GEMMs, the
same tile buffers/pending parameters live across the readiness check.

## Fresh evidence

Single-card leaf115011 passes hot/broad/zero/skew/one, independent random BF16
weights, source immutability, output guards and existing invalid/missing-source
termination gates. This is not a separately injected missing-tail-flag test.

All four-card runs use devices2,3,5,7, expert servers5/7;48 outputs/run pass.
Inputs and layers change over24 requests/source. Compact receipts are in
early-down-result.json; capsules freeze source and binaries.

| Pair / policy | Run suffix | Pack→return median server0/1 |
|---|---|---|
| Uninstrumented control |115113|598.67/565.01us|
| Uninstrumented early |115140|553.73/513.16us|
| Instrumented early |115222|567.11/523.98us|
| Instrumented control |115250|587.14/559.93us|

Each has24 paired waves per server. First pair improves7.5%/9.2%; reverse-order
instrumented pair improves3.4%/6.4%. Do not compare the instrumented/noninstrumented
rows as if profiling overhead were identical.

In115222, down is issued29.28/17.89us (median) before tail activation completion.
All recorded tail readiness exits follow the corresponding producer join;
per-core tail-check interval median is0.22us, indicating this fixture generally
reaches the tail after activation is ready. These are scalar issue/wait intervals,
not hardware MMAD utilization. persistent_analyze.py validates these boundaries
while retaining the prefix-before-down requirement.

32/1-row heterogeneous arrivals with source1 delayed2ms:
- early115316:38/41 waves,10/7 paired; pack→return555.66/510.16us,
  episode17.223/17.198ms.
- control115404:40/37 waves,8/11 paired; pack→return562.43/542.04us,
  episode17.372/17.317ms.

No observed episode regression, but pairing changes and the gain is small;
this is not an identical-work causal throughput comparison or a guarantee
against monopolizing Cube when a future tail producer is badly delayed.
Keep opt-in pending that broader scheduling envelope. No fresh DFC parity or
real-model serving claim is made. Cards released after bounded runs.

## Reproduction and visual evidence

Build with OUTPUT_DIR=runs/attention-early-down-build and build_persistent.sh.
Set PERSISTENT_BUILD to that absolute directory. Use run_persistent.sh for leaf.
For run_remote_dfc.sh additionally set DEVICE_SERVICE_PERSISTENT=1,
DEVICE_SERVICE_PARALLEL=1, DEVICE_SERVICE_BURST=1 and the internal/early flags.
Use DEVICE_SERVICE_INTERNAL_TIMING=1 (not WORK_TIMES) for the core causal audit.
Disable EARLY_DOWN only for matched controls; keep the same compiled objects.

Run persistent_analyze.py on the capsule. It exports:
runs/remote-dfc-control-20260916T115222Z/analysis/persistent-work-relative.json.gz

This is actual internal per-core timestamp instrumentation, independently
zeroed per server, not a new cross-device clock fit or a native DFC instruction
timeline. Down envelopes include any scalar readiness wait and normal tile work.

For remaining command gaps versus absent/partially prepared input, use
[CUBE-SUPPLY.md](CUBE-SUPPLY.md) and cube_supply_audit.py before proposing
another Cube task queue. Existing heterogeneous traces already contain up→up
and down→down execution.
