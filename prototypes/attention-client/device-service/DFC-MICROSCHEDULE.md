# Borrowing DFC matrix micro-scheduling

Source audit, 2026-09-16. No fresh accelerator run or performance claim.
Baseline: BetterScale f6213f4; pinned Ascend source9bf964cb4b87c8cd0d6852c41a55b3c29711fa95.

## Decision

Reuse the external matrix tile engine and DFC's producer/consumer boundaries;
do not replace them with a generic dynamic GEMM task queue. The present code
already has cyclic tile distribution, rotating expert start cores, actual-row
shapes, NZ weights, preload and L1/L0 buffering. The outstanding issue is not
established as a flat matrix engine. It is coarse *readiness and retirement*.

The optional DFC FIXPIPE adapter already passed an earlier leaf gate without
improving broad-expert latency (ACTUAL-COUNTS.md). Repeating that substitution
alone is not a new experiment. Identical tile dimensions do not establish equal
cache state, code generation, memory traffic or whole-chain latency.

## Concrete source mapping

Paths below are relative to `prototypes/attention-client/device-service` unless
prefixed with upstream. DFC files are under
`upstream/vllm-ascend/csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/`.
Packaged lab source inspected in DFC-TIMELINE.md has the same relevant structure,
but do not equate its binary with the pinned source solely from that similarity.

| Boundary | DFC | Current persistent implementation |
|---|---|---|
| Tile placement | GMM1/GMM2 rotate startCoreIdx across expert M/N tiles | actual_gmm.cpp/CATLASS and streaming_gmm.hpp already do this |
| Tile lifetime | One BlockMmad across group traversal; buffered pending tile parameters | Same across each up/down invocation, including internal up prefix |
| Pull→up | Per-group AIV readiness, waited before issuing relevant tiles | FETCH entire admitted source set, Group, REPACK entire set, READY_UP |
| Up→activation | Prefix completion published through FIX pipeline | Drain+PIPE_ALL,24 GM progress lines, coordinator join,16-worker command |
| Activation→down | AIC starts GMM2 after first activation signal, waits later at internal expert boundary | Mode2 requires actDone == parts before issuing whole down |
| Down→return | Each core's last tile for an expert carries syncLoopIdx; FIX completion releases matched AIV consumer | Whole down joins24 cores, then SEND all routed rows |

The tile implementation
`utils/block_mmad_preload_async_fixpipe_quant.hpp` retains the completion tag in
its pending L1TileMmadParams. L1TileMmad emits it after the corresponding final-K
output is issued through FIXPIPE. Thus a C++ tile-call return is NOT the correct
completion point. Its Finalize has both hardware cross-core and optional soft-flag
paths; the latter orders flag DMA after FIX via FIX_MTE3. Neither licenses an
ordinary scalar flag store before output completion.

DFC's CombineV2 repeats the Cube tile assignment and uses paired AIV subcores to
consume the corresponding output slices. The existing service instead launches
separate24-AIC and17-AIV kernels (one coordinator+16 movers), and assigns complete
rows to movers. Its hardware pairing and reduction/return ownership are different.
Do not transplant CrossCoreSetFlag IDs as if these launches had DFC's mixed-kernel
pairing, or wait only for one Cube when a row spans several N tiles.

## Smallest useful prototype next

First isolate **activation→down readiness**, not a whole new server:

1. Keep the existing frozen slot catalog and one continuous down tile object.
2. Publish the down command once up is finished and activation prefix is ready,
   rather than after both activation segments. At the boundary, wait for the
   second segment's completion from all producing movers before any GM→L1 read
   of that segment. The preload issue point is the dependency boundary.
3. Use slot-generation-tagged activation readiness independent of the mutable
   coordinator command. Never recycle slot data or flags until all readers and
   SEND have retired. Zero/empty segments must publish valid completion too.
4. Keep AIV progressing while Cube waits; poll STOP with a bounded watchdog.
   Preserve the current path as control: earlier down is not necessarily useful
   if it prevents the single Cube team from doing ready work on the other slot.
5. Measure up-end→down-start, down's internal readiness wait, activation overlap,
   and whole pack→return. Compare identical paired work and heterogeneous arrivals.
   Tiny gaps or a slower whole episode mean no adoption.

This is a design, not implemented behavior. It retains full-pack and whole-down
return barriers deliberately so the first experiment has one changed dependency.

## Following seams if that is worthwhile

**Down→return:** borrow completion tags that survive the tile preload pipeline.
For the current row-oriented movers, derive the set of Cube producers covering
each output row range and wait for those producers' completed FIX writes. Return
to distinct source route slots; the source may reduce only after both expert
servers publish final DONE. Partial sends never mean early source/slot reuse.
Do not put PIPE_ALL after every expert: that could destroy the overlap being
borrowed. A small mixed AIC/AIV leaf reproducing DFC's hardware pairing is an
alternative if software progress publication costs dominate, not an unproven
in-place flag substitution.

**Pack→up:** requires expert-major ready ranges. Current REPACK walks source route
order; scattering into expert-major destinations does not imply those destinations
become ready in expert order. Build inverse route indices from the already frozen
catalog, then pack/publish complete expert ranges with all contributing movers
joined. Retain the useful one-fetch-per-token staging initially; direct per-route
remote pulls duplicate traffic for multi-hit tokens. Freeze admission before any
such computation: extending the batch later shifts offsets and corrupts readers.

## Qualification boundaries

Use independent BF16 oracle, changing source generations/layers, all-zero local
hits, single hot expert, broad tiny groups, empty prefix/tail and delayed AIV.
Verify first consumption never precedes producer output completion, no flag ABA,
no partial DONE and no source reuse before both server receipts. Existing internal
work timestamps can show these boundaries; ordinary DFC Level1 envelopes cannot.
Nothing here establishes that all of the DFC latency gap is recoverable by
scheduling, or that the differing two-card/four-card topologies have equal costs.
