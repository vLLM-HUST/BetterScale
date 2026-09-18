# Route preparation: don't promote a moved interval as an end-to-end gain

Bounded negative/marginal experiments,2026-09-18. The production builder and
qualified defaults are UNCHANGED. These candidates are retained only through
`build_route_prepare.py BASE NEW --strategy ub_tiles|local_ids|local_counts`.
The helper freezes the existing ABI/Cube/client and rebuilds only Vector code.
It never overwrites an output directory and asserts all code-generation anchors.

## Three distinct hypotheses

- **ub_tiles:** bulk-write validated input IDs into the slot, then DMA256 IDs at
  a time into a separate1KiB UB tile for Group's two scans. Route tile starts at
 48KiB; a static assertion keeps the live route map below it. This avoids scalar
  GM accesses but adds DMA/UB access/synchronization. It REGRESSED. Do not assume
  scalar GM access is uncached or that replacing it with DMA must be faster.
- **local_ids:** validate global expert IDs unchanged, then store local expert
  group or-1 in slot-private scratch. Only Group reads that scratch. Count and
  map scans no longer repeat ownership division/remainder. Route order, padding,
  layer compatibility and worker descriptors remain unchanged. Group gets faster,
  but the whole leaf does not: work has also moved into acceptance.
- **local_counts:** additionally count per-source experts while validating each
  accepted source. Group sums the bounded per-source histograms instead of
  rescanning every route. Histograms belong to each staging Slot and source;
  `s.gen[c]` becomes live only after its count/ID frame is complete. Existing
  sources retain their counts when a compatible late source joins. This adds
  two Slots' source×GROUPS int32 private state (about6.9KiB at5×176), rather than
  a new wire field or allocation API. Stack/resource cost is not assumed free.

The counted case has a modest large-row signal, but not enough benefit across
shapes to justify promoting it or extending it to a full-model campaign now.
No new parallel coordinator, producer/consumer ownership or queue protocol was
introduced. `local_counts` was built against the already-expanded five-source
closure; it is not integrated into the production two-source expansion path.

## Same-host results

hw0,910B2,CANN9.0.1,A1+E3,real target layer0,H2560/K10,synthetic shared weights,
1024-row channel, pipelined client and SEND, route-ready disabled throughout.
Each arm completes216 calls and each numeric gate spans1/7/32/127/512/1024rows.
All three candidates match the first control's24 saved complete output tensors
BITWISE (random routes/probabilities and repeated-expert/empty-owner cases).
This is final routed+shared output equivalence, not an independent check of every
server intermediate or a full-model quality qualification.

| Arm |1024-row leaf (ms)|Post-fetch/pre-pack interval across owners (us)|
|---|---:|---:|
|Control before|3.27518|432–447|
|UB tiles|3.40585|551–553|
|Local IDs|3.28880|312–320|
|Local IDs + counts|3.22595|168–176|
|Same control after|3.28140|see receipt|

Twelve samples/leaf arm; medians, not best-of. Controls bracket the candidates
on the same devices. Metadata intervals come from26 retained steady broad-hit
waves/owner and include handoff/admission/Group; they are NOT isolated Group
instruction times. Ignore the two all-expert0 diagnostic waves when selecting
this population. Full samples for1024rows and all other medians are retained in
`route-prepare-result.json`; larger raw role receipts remain under runs.

Counts saves about1.5–1.7% at1024rows relative to both controls, but512rows is
essentially unchanged and127rows goes1.100/1.125->1.146ms. Small-row differences
include noise. Most importantly, the~260us shorter preparation interval does not
become~260us shorter complete service. Source inspection makes relocation into
Accept a credible contributor, but this experiment does not separately time
Accept, so it is not a quantitative attribution of the entire missing gain.

## Decision and useful next boundary

Do not change current serving defaults or add these flags to build.py. Preserve
this failed/marginal route so future work does not repeat it. A future metadata
redesign should cover acceptance/validation, count/prefix/map construction and
publication together, measuring complete service. Parallelizing only a named
Group interval or moving loops earlier is not sufficient evidence of success.
Existing PACK remains a separate, more local DMA-pipelining opportunity; these
experiments did not modify it. No full-model or MTP claim is made, and no NPU
job/watcher remains owned by this experiment.
