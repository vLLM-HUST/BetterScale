# Fixed-order pull/reduce pipeline

`client_reduce_pipeline.cpp` changes the consumer, not the mailbox protocol.
Within `QWEN38_FUSED_COLLECT=1`, ABI-marked new builds select the pipeline by
DEFAULT. `QWEN38_PIPELINED_COLLECT=0` retains the serial fused control; explicit
`=1` rejects older binaries without the new export. An unset pipeline option on
an old binary preserves its serial fused path. This does not enable fused mode
itself or the separate arrival-order online collector.
New `build.py` closures include both kernels. To preserve an already qualified
server exactly while rebuilding just the client:

```
python prototypes/attention-client/qwen38/build_client_pipeline.py BASE NEW_BUILD
```

The builder refuses to overwrite an existing output. It retains the server
binary/config geometry and marks the client capability in `abi.json`.

## What overlaps, and what deliberately does not

All owner DONE generations are still joined before pulling. The client still
owns each token's fixed-order FP32 `Cast -> Muls -> Add` sum. Routing probabilities,
rounding, output layout, layer identity and separate retirement are unchanged.

Two BF16 input slots alternate in64KiB UB. MTE2 may read route k+1 while Vector
reduces route k. MTE2_V licenses consuming a slot; V_MTE2 releases that slot
only after its Cast has consumed the BF16 values. Cast/product and sum occupy
separate FP32 regions. Output has another BF16 region, with V_MTE3 readiness
and MTE3_V reuse/drain. Metadata still uses the scalar-safe IO path. Idle AIVs
also balance their initialized events. There is no global route-expanded HBM
allocation and no change to the server's staging slots.

This borrows DFC's buffer/event ownership, NOT its whole push topology. The
serial fused collector remains a control. It is not the arrival-order online
collector and does not waive that collector's unresolved numerical gate.

## Bounded hardware evidence

hw0,910B2,CANN9.0.1; one attention/source device and three expert devices.
Real target layer0 routed weights, synthetic shared expert weights, H2560/K10,
16 collect AIVs. Both kernels live in the same binary/server closure. FULL graph
input changes cover1/7/32/127/512/1024rows, nonuniform BF16 probabilities,
random experts and repeated expert0 with empty other owners. All24 changed-input
comparisons are bitwise identical;216 published calls complete. Numeric outputs
are cloned immediately per arm, before another arm reuses the same bank.

| Rows | Ready serial / pipeline (us) | Complete leaf serial / pipeline (ms) |
|---:|---:|---:|
|1|18.74 /13.94|0.2575 /0.2574|
|7|24.42 /17.31|0.3342 /0.3262|
|32|46.80 /42.31|0.5883 /0.5876|
|127|140.12 /125.75|1.2120 /1.1934|
|512|527.36 /475.10|2.3628 /2.3200|
|1024|1029.23 /921.03|3.7445 /3.6404|

Each number is the median of12 alternating-order samples. Ready timing contains
16 collect-only replays/sample after all servers finish; it is warm and excludes
server computation/wait, NOT a cold bandwidth ceiling. Leaf timing includes
quant/pack/shared/server/collect/retire, one replay/sample. At1024rows the roughly
108us ready improvement accompanies roughly104us lower whole-leaf time. Small
row counts do not establish a useful complete-leaf gain. No SWE throughput gain
is implied. Full samples, topology and capsule identities are in
`collect-pipeline-result.json`. Reproduce with `run_wire.sh` selecting
`--client-probe probe_collect_pipeline.py` and the matching new build.

## Whole-model continuation gate

The same build also passes A2TP1+E3 with real48 target layers plus MTP layer,
FULL prefill/decode,K1,4GiB State/source,2 seats/source and512 physical prefill
rows. Four SWE traces, first2turns and output capped8 complete on both sources;
all five roles exit0. Each server observes the same2411/2461 source generations.
Prefix first-page checks pass. Both sources' cold/continuation FULL-vs-eager
shadows at lengths2/1/256 have129/129exact State tensors:12 checks total.

This shadow compares the NEW pipeline's eager and captured execution. The leaf
above supplies old-serial versus new-pipeline equivalence; do not substitute one
claim for the other. The capped trace takes9.168/9.062s/source in this run, but
there is no matched whole-model control, so no serving-throughput gain is claimed.
This is not an OpenCompass or uncapped full-trajectory quality qualification.

## Default promotion and remaining seams

New ABI-marked builds now choose the pipeline automatically inside fused mode;
`QWEN38_PIPELINED_COLLECT=0` selects the serial control. Older closures without
the export retain serial fused behavior when unset; explicitly requiring the
pipeline on such a closure fails rather than silently pretending it is enabled.
This routing change selects the kernel already qualified above; no additional
hardware measurement or speedup is claimed for changing the default.

Reanalysis of the SAME collector leaf's server receipts is retained in
`collect-server-seams.json`. At1024 source tokens and broad random routing,
owners process3392/3427/3421 routed rows. Their coordinator-observed intervals:

| Interval | Range across three owners |
|---|---:|
| fetch |155–169us|
| post-fetch / pre-pack gap |429–448us|
| pack |221–242us|
| up |487–553us|
| activate |149–162us|
| down |245–306us|
| convert/export |403–470us|

These intervals include command handoff and join. The retained ring mixes the
serial/pipelined client arms; medians do not form an additive wall-time breakdown.
Broad random routing activates many more experts than the previous fixed hot10
fixture: larger GEMM intervals here are NOT evidence of a GEMM regression.

Source-supported optimization leads, not measured recoverable time:

1. `server_workers.hpp` SEND still calls `RowExpert`, `QuantRow::Dequant` and
   `ToBf16` for each routed row. Dequant reloads H-wide channel scales per row,
   plus the padded row scale; ToBf16 waits for MTE3 before the next row. Server
   export can benefit from the same explicit buffer/event lifetime approach.
   Unlike ACTIVATE's expert-contiguous iteration, SEND currently walks route
   order through a forward map. Claiming expert scale reuse requires confronting
   that layout: two adjacent routes need not use the same expert. Do not simply
   reuse the previous scale without checking expert identity.
2. The inherited `persistent_vector.cpp::Group`, transformed by Qwen38 build.py,
   counts routes and builds row maps on one coordinator. Compact maps already
   remove capacity-wide work; live-route scans and serial publication remain.
   Post-fetch/pre-pack also includes admission/coordination. Parallelizing Group
   needs an ownership design, not just moving it across a FETCH that reads its
   descriptor buffer.
3. The measured closure has `route_ready=true`; its worker compile-time branch
   publishes a generation line after EVERY SEND row although this fixed-order
   collector only reads all-owner DONE. These publications are unnecessary for
   a fixed-order-only deployment. Standard `build.py` defaults route-ready OFF;
   this is an avoidable cost of this diagnostic closure, not a universal default
   bug. Removing flags from a closure also used by online collectors would break
   its protocol. Measure a matching no-route-ready closure rather than silently
   disabling publication behind a route-ready ABI.

First priority is server export, not a new collective or a different reduction
order. Copy/DMA gains and less server waiting remain distinct measurements.
