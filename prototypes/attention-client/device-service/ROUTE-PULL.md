# Publish route results; let the client pull and reduce tokens

Opt-in prototype: DEVICE_SERVICE_ROUTE_PULL=1 with persistent/internal pipeline,
fine-pack and early-return enabled. The published Worker is unchanged.

## Wire and ownership

The server still exports a route-major result area. After each ReturnStreaming
copy completes MTE3, its sole mover publishes that source generation in a separate
64-byte route flag line. Flags begin after the fixed256×2048 BF16 payload at byte
1048832 of each existing2MiB export.256 lines occupy16KiB; no new dense server
allocation or shared flag-line writers. Old generations need not be cleared.

The source's neural_collect_reduce assigns tokens to16 AIV workers, up to2 tokens
per worker at the existing32-token capacity. Each worker owns both FP32 accumulators
in UB. It scans its unfinished tokens and owners, pulls ready contributions exactly
once using an8-bit route mask, multiplies the BF16 routing probability in FP32 and
accumulates directly. Each complete token is rounded once to BF16 and written to
[tokens,H]. No client [tokens,K,H] expansion or subsequent native unpermute remains.
The two legacy client raw buffers become scalar placeholders in this mode only.

This is asynchronous contribution collection, not early attention execution.
The final client kernel drains both server DONE generations before retirement.
The next source generation is published only after collection and output complete;
that is the existing implicit consumption acknowledgement. A server can finish and
serve other accepted work without polling a client ACK, but must not overwrite an
export for that source absent its next generation. Host teardown separately waits
for client stop/unmap before freeing exports. Tokens and route flags never license
early input or export reuse.

Currently the producer releases outputs at the existing down prefix/tail completion
boundaries, then flags individual copied routes. This is not per-expert FIXPIPE
publication. Fine-grained client consumption does not remove the producer's bounded
prefix barrier. Arrival-order FP32 accumulation can differ from native reduction
order; keep numerical tolerance separate from bitwise equivalence.

Client config adds probability/output pointers, a bounded per-core timing buffer,
and optional polling delay. Timing reports kernel entry, first contribution read,
local token reduction completion, final all-server drain, and flag-block poll count.
The trace is per client clock, not a distributed time alignment. Server config23
enables route flags and requires early-return. Both sides must enable together.

## Small independent metadata simplification

Continuous mode2 now publishes the scalar split boundary in the existing aligned
last line of ptr[9], rather than constructing two unused128-entry segment catalogs.
The full ptr[6] catalog and mode1's external-segmentation catalogs are preserved.
Both control and candidate below include this cleanup. Preparation is still about
18us, versus roughly19us in the preceding study; no isolated hardware speedup is
claimed for this cleanup.

## Qualified hardware evidence

Host hw0, expert devices0/1, source devices2/3. Same copied CATLASS/early-return
build lineage as FAIR-DFC.md; new queue_service and persistent binaries are in
`runs/attention-route-pull-{client,persistent,timed,paced}-build` and copied to hw0.
Artifacts: `runs/route-pull-20260916/hw0/<mode>/`; compact results are retained beside
this note. All performance rows use32 tokens/source, two alternating weight address
sets,24 paired expert waves and48 checked client outputs.

| Mode | Source episode ms0/1 | Median server retirement interval us0/1 |
|---|---:|---:|
|First control|19.310/19.217|651.10/652.26|
|First candidate|21.424/21.178|577.50/582.48|
|Instrumented candidate|16.677/16.789|see receipt|
|5us idle-poll pacing|16.702/16.754|563.34/563.76|
|Unpaced candidate, reverse group|17.130/17.094|569.16/570.18|
|Following control|19.572/19.602|665.32/665.90|

Do not hide the first episode regression. Server1 has a4.527ms interval between
its FIRST and SECOND completions; no later interval has that large gap. Startup
phase/coordination is a plausible explanation, not a proven host-scheduling cause.
The steady interval gains reproduce in the reverse group, and later complete
source episodes improve about12.5–14.7% against its following control. This is a
bounded synthetic source burst, not whole-model serving throughput or DFC parity.

Pacing uses DEVICE_SERVICE_PULL_POLL_CYCLES=250 (5us at50 cycles/us) only after an
iteration with no new contribution. It lowers median flag-block polls from about
488–495 to145–146/core/wave; core read-to-reduce spans remain about15us, and the
post-reduction drain is roughly48–55us. That positive drain demonstrates the client
finishes useful collection/reduction before all-server DONE. Read-to-reduce includes
intervening waits; it is not pure DMA or vector instruction time. No-progress delay
is bounded to0..500cycles by the host adapter; default remains0 pending broader
policy qualification. Polling reduction alone is not a bus-transaction measurement.

Additional independent BF16 gates: delayed32/1 sources;32-token all-owner0 routing
(including an entirely empty expert server); single-row repeated same-expert routes;
and hot8 routes spanning BOTH owners per token. Balanced routes by themselves put
each token's8 experts on one owner, so they do not test cross-owner token reduction.
A separate hot8 random-probability gate checks weighted, not merely uniform, sums.
All gates retain source generations, alternating layer addresses and final drain.
No injected missing-route-flag fault or full48-layer neural serving gate is claimed.

Reproduce with the existing run_remote_dfc.sh subset launcher and separately
built client/server artifacts:

```
DEVICE_SERVICE_SOURCE_BUILD=$PWD/runs/attention-route-pull-paced-build \
PERSISTENT_BUILD=$PWD/runs/attention-route-pull-persistent-build \
DEVICE_SERVICE_PERSISTENT=1 DEVICE_SERVICE_PARALLEL=1 DEVICE_SERVICE_BURST=1 \
DEVICE_SERVICE_INTERNAL_PIPELINE=1 DEVICE_SERVICE_FINE_PACK=1 \
DEVICE_SERVICE_EARLY_DOWN=1 DEVICE_SERVICE_EARLY_RETURN=1 \
DEVICE_SERVICE_ROUTE_PULL=1 DEVICE_SERVICE_PULL_POLL_CYCLES=250 \
DEVICE_SERVICE_WEIGHT_FORMAT=NZ DEVICE_SERVICE_INTERNAL_TIMING=1 \
bash prototypes/attention-client/device-service/run_remote_dfc.sh 2,3,0,1
```

The example local launcher requires its documented donor environment; hw0 used
an isolated copied capsule and its Python runtime. For weighted correctness add
DEVICE_SERVICE_ROUTE_PATTERN=hot8 and DEVICE_SERVICE_RANDOM_PROBS=1. Uniform
performance does not add a per-job probability-copy node. Cases owner0/oneexpert
and DEVICE_SERVICE_BURST_ROWS=32,1 cover empty-owner/duplicate/delayed layouts.

collect_timeline.py exports the per-client markers to compressed
analysis/client-pull-relative.json.gz. persistent_analyze.py exports expert-side
markers. Neither exporter aligns client clocks to expert clocks. The paced run
retains both files; do not render the raw receipts directly in Codex Desktop.
