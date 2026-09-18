# Pipeline server SEND without changing the expert arithmetic

This prototype extends the fixed-order collection work upstream to the expert
server. New ordinary `build.py` closures enable it BY DEFAULT. Use
`--no-pipelined-export` for the serial SEND control. Route-ready builds retain
the old SEND backend by default; explicitly combining `--pipelined-export` and
`--route-ready` is rejected until separately qualified. Frozen existing binaries
are not silently rebuilt or overwritten. For matched isolated experiments:

```
python prototypes/attention-client/qwen38/build_server_export.py BASE CONTROL
python prototypes/attention-client/qwen38/build_server_export.py BASE CANDIDATE --pipeline
```

The latter copies a frozen closure and rebuilds only `persistent_vector.o` plus
the host launcher. Cube and client binaries stay identical. Both commands above
disable route-ready flags and advertise that fact in the channel ABI; do not
mix this with an online collector. Builders reject pipeline+route-ready until
that additional notification mode is qualified. The matched helper intentionally
keeps its explicit control/candidate choices, independently of build.py defaults.

## Consumer/producer ownership

`server_workers.hpp` uses `QuantExport` from `quant_export.hpp` only for target
INT8 SEND. MTP's BF16 copy and FETCH/REPACK/ACTIVATE remain unchanged.

Each worker still visits its own route map entries, determines the same expert,
and writes the same destination. Submit starts reading the next routed row,
channel scale and padded row scale before consuming the preceding row. Two
input/scale slots alternate; a shared FP32 vector performs the SAME
INT32->FP32 RINT Cast, row-scale Muls, channel-scale Mul, BF16 RINT Cast sequence.
Two separate BF16 output slots prevent writes from being overwritten prematurely.

- MTE2_V and MTE2_S protect vector data and scalar row-scale consumption.
- V_MTE2 releases a slot only after its channel scale and input are consumed.
- V_MTE3 licenses each output DMA; MTE3_V protects output-slot reuse.
- Finish consumes the final pending row and drains BOTH slots before worker DONE.
- Metadata map tiles own bytes[0,1024); inputs/scales/FP32/output/row-scale data
  live above that boundary inside the existing64KiB UB. There is no new HBM
  allocation, new owner, new generation or reordered expert sum.

The implementation retains a route-flag publication branch for an eventual
separate online gate, but the supported build interface deliberately disallows
that unqualified combination. Fixed-order mode publishes only existing all-owner
completion after every worker has drained.

## Same-host bounded evidence

hw0,910B2,CANN9.0.1,A1+E3,real target layer0,H2560/K10,synthetic shared weights,
1024 row channel capacity. Both arms use the already-qualified pipelined client,
compact routing maps and batched activation. Per-arm216 published calls complete.
The complete24 output tensors across1/7/32/127/512/1024rows and four changing-input
cases (including repeated expert0/empty owners and random probabilities/routes)
are BITWISE identical between server builds, not just sampled rows. Each client
also checks serial-vs-pipelined collection independently.

| Source rows | Old SEND, no flags (ms) | Pipelined SEND, no flags (ms) |
|---:|---:|---:|
|1|0.24193|0.23745|
|7|0.29633|0.28784|
|32|0.57643|0.56701|
|127|1.12843|1.11297|
|512|2.18341|2.08538|
|1024|3.42479|3.22769|

These are complete layer leaf medians of12 samples, not GEMM-only timings.
The two builds ran sequentially on the same devices; within each build, client
control/candidate replay order alternates. At1024rows the approximately5.8% gain
is accompanied by a lower convert/export coordinator interval:

| Server | Old / pipelined convert-export interval (us) |
|---|---:|
|0|376.34 /197.51|
|1|385.95 /207.41|
|2|370.19 /190.71|

Intervals include command handoff/join; do not call them pure vector instruction
times. The retained ring mixes the two client arms, not the server builds.
The preceding flags-on closure had a3.640ms1024-row leaf, versus3.425ms for the
new no-flags control. That is a useful direction but a separately timed comparison,
not an interleaved isolation of exactly215us of flag cost. Standard build.py already
defaults route-ready OFF: do not claim flag removal as a new universal default fix.

Raw samples and capsule/build identities: `export-pipeline-result.json`.
Use `probe_collect_pipeline.py` with `QWEN38_SAVE_COLLECT_OUTPUTS=1` to retain
CPU output tensors under its temporary run directory for cross-build comparison.
The wire supervisor preserves normal JSON receipts; `.pt` files are intentionally
not copied into Git or the compact receipt.

## Full-model gate

The candidate also passes A2TP1+E3 with real48 target layers plus MTP, FULL
prefill/decode,512 physical prefill rows,2 seats/source and4GiB State/source.
Both sources complete four SWE traces' first2turns (output capped8); all roles
exit0 and all three servers match2411/2461 completed source generations.
Twelve cold/continuation FULL-versus-eager shadows each have129/129exact State
values; prefix first-page checks pass. Cross-build equivalence comes from the
24 saved leaf tensors, not from the candidate-only eager/graph shadow.

The capped trace takes9.223/9.216s/source. This was a continuation/correctness
gate, not an interleaved whole-model performance control; it does not establish
an end-to-end throughput gain despite the isolated leaf improvement. No full
quality benchmark or uncapped trajectory claim is made. Owned hardware was
released after the gate.

## Remaining visible cost after this change

Reanalysis of the candidate's retained ring, selecting the stable broad1024-row
case rather than the two all-expert0 diagnostic calls, finds post-fetch/pre-pack
intervals429.04/437.07/429.98us (26 samples each). They still contain single-
coordinator admission/Group work, not a pure Group measurement. Source inspection
shows live-route histogram and map construction remain serial on that coordinator.
This is the strongest next preparation target; no claim that all430us is removable.

PACK remains224.81–235.66us. Each routed token copy and padded scale copy still
uses synchronous Transfer::Copy. Some duplication is intrinsic to expert-major
GEMM input, but batching DMA/scale transfers or pipelining them is a smaller next
change than redesigning the whole service. Up487–545us and down242–296us in this
broad-expert fixture are NOT automatically wasted time; wide weight coverage
and low per-expert M differ from the old hot-expert fixture.
