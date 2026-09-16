# Separate batching efficiency from expert-kernel efficiency

This follows `ROUTE-SPREAD.md`. It is a diagnostic of the BF16 Qwen-sized expert
service, not a production batching policy or a serving-throughput result.

## Equal-work paired-source control

`DEVICE_SERVICE_PAIRED_CONTROL=1` makes the device selector wait for the next
published generation of both unfinished sources **before consuming either**.
Inputs, route IDs, weights, pack/scatter and NZ GMM are unchanged. The kernel's
parallel-mode bitfield reserves bit1 for this experiment. Every live cycle must
record two sources and matching generations; stale binaries/unpaired execution
must fail that receipt check. Each source still publishes/consumes its own frame.

The run `runs/remote-dfc-control-20260916T045757Z` uses physical2,3,5,7
(attention0/1, expert0/1), local910B2. Both expert servers select24 paired cycles
plus24 empty drain cycles; all48 client outputs remain bitwise correct.
The four-device profile maps each batch through device-recorded generations.
For broad32, each expert now receives4 routed rows rather than2:

- Both GMMs together:399–400us per expert server.
- Input pack:57us; scatter:16us.
- Server interval from pack start through DONE publisher end:497–498us.

Thus two sources do not double the approximately400us GMM cost. This directly
supports weight reuse by coalescing sources. It does **not** mean an individual
client's latency halves. `neural_prepare` includes waiting for the later source;
client graph duration includes that wait, and is deliberately not used as the
paired control's efficiency score. Neither the idle drain tail nor a sustained
arrival process is part of the reported active-server interval.

The fresh unpaired run050054Z on the same devices/build measures470–476us
per one-source active interval, versus497–498us per paired interval. For the
work of two sources this is about1.9x active-server processing efficiency, not
1.9x serving throughput. Both controls pass48 exact outputs. Results are retained in
`batch-organization-result.json`; compare its one-source interval twice with
one paired interval only as active-server work for two sources, not end-to-end
throughput. The unchanged DFC423us historical number has different capture and
measurement conditions; do not call the difference a precisely isolated tax.

## Local factor sweep: group catalog and padded compute also cost time

`gmm_batch_probe.py` pre-packs the exact broad32 fixture's rows for expert owner0,
using the same parity-scaled weights as DFC. No transport, route construction,
pack or reduction is timed. Three trials of32 prequeued FULL replays on one card.
All12 original sweep cases pass bitwise against per-expert BF16 arithmetic.
Run: `runs/gmm-batch-control-20260916T045701Z`, physical3.

| NZ chain layout | One source | Two sources |
| --- | ---: | ---: |
| 64 groups, only live rows | 385us | 375us |
| 64 groups, compute padding to512 rows | 411us | 390us |
| 128 layer/expert groups, padding to512 | 440us | 440us |

The small one/two-source timing difference is not a claim that extra work makes
GEMM intrinsically faster. The important result is that doubling live rows does
not double chain time. With two sources, changing128/padded to64/padded saves
about50us; removing the remaining padding saves about15us. These are empirical
changes in the whole configuration (catalog/weight shape and padding placement),
not proof that merely iterating64 empty counters takes50us. Native tiling and
memory behavior can change too. Do not add these local deltas mechanically to a
profile collected under different conditions.

This sweep also preserves the ND counterexample: at64 live groups, ND and NZ
are close and neither universally wins; NZ improves the current128/padded chain.
Prefer NZ where the qualified server configuration benefits, not as a blanket
claim about every GEMM shape.

### Capacity-only is not an adopted shortcut

`runs/gmm-batch-control-20260916T045842Z` keeps input capacity512 while letting the
group-list total cover only128/256 live rows. Its four cases happen to pass exact
live-output checks on this binary. **This is outside the installed operator's
documented contract.** `torch_npu/_op_plugin_docs.py` documents the A2 tensor
`group_type=0`, single-input/single-weight case as requiring group-list cumulative
end equal to input M. The diagnostic is therefore opt-in via GMM_BATCH_SHAPES;
it is not in the default sweep and is NOT enabled in the server.
A passing microbench does not establish safe behavior for empty groups, all-empty
cycles, changing layers, future tiling or library upgrades.

## What the DFC implementation actually contributes

The pinned BF16 DFC kernel uses device-produced per-expert `currentM` bounded by
`maxOutputSize`: capacity is a ceiling, not a mandate to compute512 routed rows.
It distributes each group's GEMM tiles across cores with a rotating start index
instead of resetting every small expert to the same first core. It bypasses L2
for small-M weight reads. AIV-produced group readiness gates AIC computation,
and AIC/AIV events connect GEMM, SwiGLU and combine inside the fused operator.

These are source mechanisms, not a measured per-feature ablation of the lab A2
binary. In particular, the pinned wrapper sets `epilogueGranularity` to
`expertPerRank - 2`: do not describe it as arbitrarily fine-grained expert-by-expert
SwiGLU overlap. Its source group waiting and return synchronization still assume
the synchronous EP topology; they cannot be pasted unchanged into independent
attention clients.

Two improvement fronts are now distinct:

1. **Scheduling:** form useful same-layer expert batches under sustained arrivals,
   with an explicit latency bound. The forced paired barrier only proves the
   compute benefit; it is not that scheduler.
2. **Compute backend:** reuse actual-count/tile scheduling from mature expert
   kernels without flattening all model layers and fake padding into the work.
   Preserve independent source generations and return ownership. This is more
   targeted than writing a new GEMM or assuming any standalone GMM chain equals DFC.

## Reproduce

```
OUTPUT_DIR=$PWD/runs/attention-device-paired-build \
  bash prototypes/attention-client/device-service/build.sh
DEVICE_SERVICE_SOURCE_BUILD=$PWD/runs/attention-device-paired-build \
DEVICE_SERVICE_PAIRED_CONTROL=1 DEVICE_SERVICE_PARALLEL=1 \
DEVICE_SERVICE_WEIGHT_FORMAT=NZ DEVICE_SERVICE_TIMING=1 DEVICE_SERVICE_PROFILE=1 \
  bash prototypes/attention-client/device-service/run_remote_dfc.sh 2,3,5,7
bash prototypes/attention-client/device-service/run_gmm_batch.sh 3
```

Use `profile_export.py` and `analyze_route_spread.py` as in ROUTE-SPREAD.md.
Timelines are compressed provider-clock exports; there is no independently
calibrated cross-device clock fit. Run045554Z was cancelled before admission to
switch away from occupied4/6; no foreign process was stopped.
