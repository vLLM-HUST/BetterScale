# Five-attention prefill timeline (2026-09-18)

The existing E3 SWE runner schedules each source independently. Any pending
prefill in that source takes priority over its decode work. With8 resident
requests and1024 input lanes it assigns each request at most128 tokens/wave.
Unused seats do not donate their128-row allowance to a longer request. This is
an experimental fixed-width scheduler, not a mature mixed scheduler.

`analyze_prefill_schedule.py` reconstructs effective prefill rows from monotone
encoded cursors and checks their sum against the receipt's actual prefill count.
For the completed `20260917T202430Z` E3 run:

-583 prefill waves across five sources,236,306 useful rows out of596,992 bucket
  rows:39.6% fill.391 waves advance only one request.
- The slowest source,attention2, has162 prefill waves;124 advance one request.
  Its174.9s cumulative prefill cost is distinct from42.1s decode-wave cost.
- These are tensor-row accounting figures, not a claim that every padded row
  costs exactly the same FLOPs as a valid row.

## Fresh profile

hw0 capsule `qwen38-model-20260918T063408Z`: same TP1x5+E3, real48 target+MTP K1,
State46GiB, e affine-head QSA. After normal warmup, all five attention roles
record their first two SWE prefill waves;1024 valid inputs/wave/source. The
receipt is labelled PROFILE and cannot be used as a completed throughput test.
Expert servers execute normally but are not profiled in this capture.

The two full waves show that padding is not the whole story. Attention0's
largest named task sum is `neural_collect`:98 calls,631.94ms total (~316ms/wave).
Attention2 records698.97ms. Collect includes remaining expert wait and result copy; weighted reduction
is a separate `MoeTokenUnpermute` task. **It is not expert GEMM time**. Other attention0 two-wave sums:
FIA103.45ms, QSA gather98.59ms, host PLE mailbox pull106.83ms. Do not sum
potentially overlapping task durations as an end-to-end decomposition, and do
not substitute instrumented timings for the unprofiled benchmark.

## View and reproduce

Local compressed files under
`/workspace/betterscale-confluence/runs/qwen38-prefill-profile-20260918/analysis/`:

- `qwen38-prefill-attention5-annotated.json.gz`: five-rank device view plus10
  recorded host wave bands carrying valid/bucket input counts (about4.9MiB).
- `attention2-prefill.json.gz`: individual attention2 TraceLoom detailed view.
- `operator-cost.json` and `profile-receipt.json`: query results and provenance.

Torch-NPU raw capture is analysed offline after releasing cards. TraceLoom
37323af preprocesses each provider DB, then its native distributed exporter
constructs event lanes. The view restores **same-host provider timestamps**;
we do not zero each rank's first event or invent collective endpoint matches
for point-to-point expert IPC. No additional independent clock calibration is
claimed. Host wave bands use the recorded PYTORCH_API timestamps in the same
provider domain. Each rank has2 markers and98 collect completions, matching
49 target/MTP layer calls per wave. The combined view has82,640 device slices.

Enable `QWEN38_PROFILE_STEPS=2` and absolute `QWEN38_PROFILE_DIR` before
`run_topology_case.sh tp1-e3 trace 46 <fresh-case-dir> affineheads`.
`device-service/profile_export.py <root> --roles attention0 attention1 attention2 attention3 attention4 --label qwen38-prefill-attention5` handles
preprocessing/merging; `annotate_prefill_profile.py <analysis-dir>` adds wave
bands. The remote exact launch/analyse scripts, raw profiles and SQLite files
are in `/workspace/betterscale-hw0/runs/qwen38-prefill-profile-20260918/`.
A relocated TraceLoom binary also needs its classification, structural-symbol
and reconciliation TSVs; pin the three TRACELOOM_*_RULES variables. Select the
provider's `ascend_pytorch_profiler*.db`, not its auxiliary `analysis.db`.

## Client/collect inspection

Source inspection of `client.py`, `wire.py`, `build.py`, and
`../qwen-next/client_kernel.cpp` establishes the current protocol:

- Submit, native shared expert, promotion, collect and retire are launched on
  the same current stream. Remote server execution can overlap shared expert;
  local submit copying cannot. Submit publishes READY only after all rows,
  scales, route IDs and the descriptor have been copied.
- Submit uses one AIV block and a row-by-row read/write loop. Target rows also
  copy their FP32 scale through a padded cache line. The IO helper fences each
  transfer and executes PIPE_ALL after each write. This is a correctness-first
  protocol, not a bandwidth-tuned prefill mover.
- Collect uses16 blocks. Each waits for every owner generation before reading
  any result, then copies route outputs individually into a materialized
  `[rows,10,2560]` BF16 buffer. At1024 rows this is50MiB per layer. Weighted
  reduction happens afterwards in native unpermute, not inside collect.
- `build.py` explicitly removes the old H2048 `neural_collect_reduce`: its
  fixed UB geometry/BF16 assumptions are not safe for this H2560 W8A8 port.
  Reusing token-ready collection requires adaptation, not flipping a flag.

In attention2's first captured layer, local submit runs647.82us and ends at
520999.48us; the first shared MatMul starts520999.49us. Collect subsequently
runs4028.82us, followed by a separate46.88us unpermute. Across98 layer calls,
submit totals62.105ms and collect698.971ms (two instrumented waves). Select
one representation of each task: the detailed TraceLoom export contains both
published-tree and device lanes; summing both doubles the measurements.

The current capture cannot separate server scheduling/compute wait from pull
cost. Next bounded diagnostic: timestamp owner completion versus collect
first/last transfer, with a results-already-ready copy control. Side-stream
submit should be tested separately, preserving input-ready, submit-ready,
shared-completion promotion and generation/retirement dependencies, including
FULL graph capture. Overlap does not remove the delayed READY publication or
prove a net speedup under shared memory-resource contention.

## Borrow the donor's shared-expert overlap, not just its stream count

Inspected the clean donor checkout at
`/workspace/strengthen-dsv4/upstream/vllm-ascend` (`9bf964c`).
`ops/fused_moe/fused_moe.py::_forward_shared_experts` switches to a dedicated
shared stream and inserts phase-specific event waits. `moe_comm_method.py`
records dispatch/combine boundaries; `moe_mlp.py` records the second-GMM
boundary. Shared quantization can overlap the router, shared gate/up overlaps
routing communication, activation overlaps routed down GEMM, and shared down
overlaps combine. The main stream joins shared only before consuming the result.
This inspected wrapper uses ordinary streams/events, not an explicit AIV quota;
that observation does not characterize every underlying collective's resources.

Retained native evidence, not a new hardware run:
`/workspace/strengthen-dsv4/runs/hw3-split-046/analysis/rank0.db`, TASK rows,
anchor `1789279550578849335`ns (first DequantSwigluQuant). Relative microseconds:

| Stream | Task | Start | Duration |
|---|---|---:|---:|
|10|QuantBatchMatmulV3|-1385.29|94.74|
|11|hcom_allGather_|-1356.05|160.64|
|12|GroupedMatmul|-16.54|250.83|
|10|DequantSwigluQuant|0|39.12|
|12|MoeTokenUnpermute|241.11|55.60|
|10|QuantBatchMatmulV3|249.27|64.66|

This confirms local task-envelope overlap matching the source's phase design;
it does not independently measure collective byte movement or saved latency.
Use native same-rank timestamps, not an inter-rank alignment fit, for this check.

For the separated design, start by forking shared computation from client
publication at the earliest immutable hidden-input boundary. Client input
quantization/packing may then overlap shared work; source READY still requires
all payload and route writes. Shared completion promotion must wait for BOTH
that generation's publication and shared completion: otherwise promotion can
read the previous generation. Collect/pull and shared can execute independently,
joining before output addition; retirement and bank reuse must preserve all
readers. Moving collect's busy-poll kernel earlier may contend with shared AIV
work, so initially isolate submit/shared overlap rather than changing collect
and publication simultaneously. Capture the fork and join into FULL graphs and
verify changed-input replay, not only an eager timing win. Unlike donor, routed
GEMM is remote here: do not copy its GMM2 wait onto the local shared path.

## First fork/join implementation and gate (September18)

`QWEN38_SHARED_OVERLAP=1` enables a warmed shared stream and bank-owned events.
The fork precedes input quantization/packing. Main still submits and then joins
shared before promotion/collect; the latter algorithms are unchanged. Default
remains serial. `probe_shared_overlap.py` is selectable through
`run_wire.sh ... --client-probe probe_shared_overlap.py`.

hw0 four-card gate: one source plus E3, real target layer0 routed weights and
**dummy BF16 shared gate/up/down weights** (not the complete model shared
expert). FULL serial/overlap outputs are exactly equal for1/32/1024rows,
priority0/1, three changed-input and rotated-routing replays/case. Servers drain
normally. This does not qualify MTP BF16 layer48 or the full48-layer service.
See `shared-overlap-result.json` for the complete receipt and10 alternating
serial/overlap event samples per case. At1024rows median serial/overlap is
4.657/4.676ms(priority0),4.648/4.679ms(priority1): **no net speedup**.

TraceLoom-preprocessed two-replay trace confirms the fork actually executes:
serial submit613.56us, then shared from613.58 to699.16us. Concurrent shared runs
from4902.44 to5006.80us while submit runs4936.12 to5562.46us. Shared's first
MatMul grows31.74 to49.48us in this one diagnostic pair (not a robust interference
estimate). Serial collect3804.48us versus concurrent3913.96us includes waiting;
these are instrumented samples, not the alternating benchmark medians.

Interpretation: moving shared under local submit removes its visible serial
placement, but routed server work already overlapped shared after publication.
With this short shared branch, the remote path still determines completion;
earlier collect can simply wait longer. Reducing time to READY and improving
collect remain distinct opportunities. Do not enable by default or claim a
serving win just because two streams visibly overlap. The hardware roles and
lease were released; profile analysis was CPU-only afterwards.

## Parallel input publication

`client_pack.cpp` adds a16-AIV `neural_pack` and a separate one-AIV
`neural_publish`. Pack assigns contiguous32KiB tiles across cores, including
route-ID tiles; it expands eight compact scales into eight padded wire entries
per read/write pair. Target INT8 and MTP BF16 retain the existing wire geometry.
A same-stream kernel boundary joins ALL packing blocks before publication.
There is no spinning cross-block barrier, and READY is never visible while a
packing block is still writing. The existing serial submit export stays intact.

Build new server/client closures normally through `build.py`, or use
`build_client_pack.py OLD_BUILD NEW_BUILD` to copy a frozen closure and rebuild
only its client object. This preserves the server binary and wire ABI, adding
an explicit `parallel_client_pack` capability bit. Do not mix a new client flag
with an old object: Session rejects that before opening channels.

`QWEN38_PARALLEL_PACK=1` selects this prototype; shared overlap remains separately
controlled. The four-arm gate uses `QWEN38_TEST_PACK=1` with
`--client-probe probe_shared_overlap.py`. It checks private staging bytes against
legacy submit for1/7/32/127/1024 rows in both input dtypes, then compares FULL
serial/shared/pack/pack+shared output under changed inputs and route order. The
warm, already-ready collect control publishes no new generation and leaves the
server output immutable; it is a copy-cost control, not a cold-link benchmark.

The four-arm hw0 E3 leaf passes all10 staging-byte cases and all6 FULL graph
shape/priority cases (three changed-input/rotated-route checks each). The first
run measured1024-row priority1 serial4.541ms, pack3.951ms, pack+shared3.946ms.
The follow-up added the ready-collect control (not another search for a win):
serial4.655ms, pack4.038ms, pack+shared4.053ms. Ten alternating observations per
arm; roughly13% net reduction, **not a whole-model serving speedup**. At1row,
extra launch overhead provides no win;32row improvements are much smaller.
Keep the flags opt-in, with the serial control and shared overlap independently
available. No source/server scheduling or expert computation was changed.

Profile of the follow-up: serial client625.90us; pack17.50us + publish1.42us.
Normal collect3810.52us in the pack arm; already-ready collect1075.32us in the
profile and1.102ms median across ten unprofiled event measurements. Do not take
their difference as an exact remote-GEMM time: normal collect includes remaining
scheduling/compute wait, while the ready control has warmed source/output data.
It does establish that both substantial pull cost and remote wait remain.
Results: `client-pack-result.json`; compressed four-arm native TraceLoom view:
`/workspace/betterscale-confluence/runs/parallel-pack-ready-20260918/client-pack-four-arms.json.gz`.
All four owned roles drained/exited and the hw0 lease was released. The new
client object was compiled with the frozen server objects unchanged. MTP wire
bytes pass, but full MTP computation is not covered by the layer0 loop.

## Fused collection, without changing completion semantics

`client_reduce.cpp::neural_collect_fused` retains the all-owner generation join.
Each AIV owns complete tokens, reads the corresponding10 remote BF16 expert
outputs, multiplies BF16 routing probabilities in FP32, accumulates in top-k
order, and rounds once to BF16. Only final `[rows,2560]` output is written to
local HBM. No atomics or inter-core reduction is needed. Publication, promotion,
retirement and remote output lifetime are unchanged. This is NOT early token
completion or overlapping collection with unfinished expert computation.

The fused bank omits `[rows*10,2560]` raw output (50MiB at1024 rows) and owns
5MiB of final output instead. Native unpermute is bypassed. Do not equate this
bank-level accounting with measured whole-model allocator peak reduction.
Routing/probability storage has small padded tails for bounded aligned reads;
only the valid10 values per token participate. UB holds the input, FP32
conversion and FP32 token accumulator in disjoint ranges of one64KiB buffer.

`build_client_reduce.py PACK_BUILD NEW_BUILD` retains the frozen server and adds
an explicitly advertised `fused_client_collect` export; regular `build.py`
includes it for future closures. `QWEN38_FUSED_COLLECT=1` selects the path and
rejects incompatible binaries. `QWEN38_COLLECT_BLOCKS` allows16/32/48 for the
bounded resource experiment. All changes remain prototype-only and opt-in.

`QWEN38_TEST_FUSED=1` extends the existing gate with fused and fused+shared
arms; it changes probabilities as well as hidden values and route order between
FULL replays. `QWEN38_SWEEP_COLLECT=1` adds32/48-block fused arms. Each arm uses
the same input/route state, original server computation, and alternating timing
order. This validates the real layer0 ten-expert fixture, not arbitrary routing,
full48 layers, full MTP computation or serving throughput.

Both fused gates passed exact FULL output equality for1/32/1024rows and both
priority classes, including changed hidden/probabilities and rotated routes.
In the first gate,1024-row priority1 pack-only3.937ms became fused3.850ms.
The follow-up resource sweep measured pack-only4.080ms, fused16=4.003ms,
fused32=3.994ms, fused48=4.111ms (ten alternating samples/arm). Original serial
in that same sweep was4.678ms. The added fusion saves about2% in this leaf;
packing+fusion together about14%. These are not full-model throughput claims.
No compelling32-block advantage supports complexity;48 blocks regress, and
small-row shapes do not show a consistent gain. Keep16 blocks and opt-in flags.

Warm results-ready fused collection measured1.067ms versus the prior roughly
1.10ms unfused pull (which still owes unpermute). Profile samples show
unfused1073.56us and fused1033.80us. Most of the remote output reads remain:
eliminating local intermediate storage does not eliminate50MiB of remote payload.
The all-owner join still makes collect wait for unfinished server work. Early
per-token collection needs a separate readiness/lifetime protocol and is not
implemented or claimed by this result. No hardware saturation cause is inferred
from the48-block regression alone.

Receipt: `client-collect-result.json`. Native TraceLoom compressed profile:
`/workspace/betterscale-confluence/runs/fused-collect-sweep-20260918/fused-collect-eight-arms.json.gz`.
All four roles exited and released the lease. Existing nine CPU tests, Python
compilation and diff checks pass; no release/default changes were made.

## Online contribution accumulation (route-ready protocol)

`build.py --route-ready` opts both binary roles into a version2 channel contract.
Each server output window appends one64-byte generation line per route, after
its fixed-capacity BF16 payload. The geometry is negotiated before IPC export;
version1 clients cannot silently pair with version2 producers. Old generations
need not be cleared. The32-byte publication occupies its own64-byte line.

In `server_workers.hpp`, SEND publishes a route generation only after that
route's ToBf16/Copy has waited for MTE3 completion. REPACK never publishes it.
The current Qwen38 producer still starts SEND at its existing down-completion
boundary: this exposes contributions during output conversion/export, NOT
per-expert down-GEMM/FIXPIPE completion. Do not attribute its gain to a finer
Cube schedule that has not been implemented.

`client_online.cpp` keeps two active tokens per AIV in bounded UB accumulators.
It scans only owners required by unfinished routes, reads each token/owner flag
block together, and immediately pulls, weights and adds every ready contribution.
A per-token bitmask prevents duplicates. It never waits for all contributions
before beginning accumulation. Finished tokens are written to the small final
buffer and the core moves to its next pair. This bounded pairing can still cause
head-of-line blocking; it is not an unbounded all-token task queue.

After all assigned contributions have been consumed, every client core drains
all owner DONE generations, including owners with no selected experts. Only
then does the existing retire authorize the next source publication. Server
output for a source remains immutable until that source's next generation;
serving other sources does not authorize overwriting it. No new per-token ACK,
slot recycling, or cross-layer early execution is introduced.

Select `QWEN38_FUSED_COLLECT=1 QWEN38_ONLINE_COLLECT=1` with a route-ready build;
parallel input pack and shared overlap stay independent. No-progress polling
uses the existing250-cycle pause (50cycles/us in this environment), and per-core
cycle receipts record entry, first contribution, last reduction, final drain,
poll count and generation. These are local clocks, not cross-rank alignment.
Arrival-order FP32 addition intentionally differs from fixed top-k order; retain
exactness results and bounded numeric errors separately. The leaf test still
requires exact output for all non-online arms; online additionally requires
relative L2<2e-4 and elementwise rtol=.016/atol=.0002. This is a numeric leaf
boundary, not a language-quality gate.

Single-source hw0 gate:1024 rows, priority1, all-owner fused3.904ms -> online
3.409ms (about13%);32 rows .809ms -> .771ms. Ten alternating samples per arm.
Both arms use the SAME route-ready producer, so this isolates consumption policy,
not the producer flag-publication cost versus the older build. Shared overlap
remains independent (online+shared3.417ms at1024rows). Keep these leaf results
separate from serving throughput. Non-online controls remain exact. Online's
largest observed relative L2 in the single-source gate is3.06e-6, consistent with
changed FP32 addition order but not proof of full-model language equivalence.

Two independent-source gates subsequently pass, each728 calls/source and matched
server completion counts. The final gate rotates from cross-owner routes to
repeated expert0 routes (two entirely empty owners), then restores cross-owner
routes, while changing inputs/probabilities. An isolated unpublished fake window
sets all owner DONE to generation2 but route flags to generation1: online collect
returns -101 under a bounded poll limit rather than consuming stale outputs.
Both clients pass that rejection and finish the real generation stream normally.
No server buffer is mutated by this fault fixture. Missing/stale generation
rejection is not a complete distributed fault-recovery qualification.

Receipt: `client-online-result.json`. Single-source compressed TraceLoom profile:
`/workspace/betterscale-confluence/runs/online-collect-single-20260918/online-contribution-collect.json.gz`.
All owned hardware roles exited, leases released. Ten CPU tests and syntax/diff
checks pass. Published Worker defaults and releases are untouched. No per-token
ACK/reuse or eager advance to the next attention layer has been introduced.

## Where the remaining collect span comes from

Reused the single-source capsule without a new NPU run. For1024rows priority1,
client per-core median entry-to-first-contribution is1975.87us;
first-contribution-to-last-reduction1242.65us; final drain1.86us. The middle
span includes waiting for subsequent contributions; it is NOT pure copy time.
These per-core medians are not a disjoint whole-system wall-time decomposition.

The server already retains a512-record coordinator event ring. Restore sequence
order before reading it. `analyze_server_phases.py` produces
`server-phase-result.json`, preserving source files, sampled routed-row counts,
cycle units and scope. Its retained tail mixes the different test arms; it is
not a per-arm latency attribution. No cross-device clock fit is needed for
same-device durations.

Heaviest owner2 (4096 routed rows,1024 source tokens, top-k10 split across E3):
fetch161.44us; post-fetch/pre-pack gap562.80us; pack349.46us;
up95.66us; activate/quant381.60us; down84.97us; convert/export563.61us.
These are coordinator command intervals, including dispatch/join overhead, not
pure instruction timestamps. Other owners handle3072 routed rows and show the
same qualitative imbalance. The two GEMM commands together are about181us:
most visible cost is not the matrix multiplication itself.

Source maps the post-fetch gap to another Accept check plus Group/descriptor
construction. Group runs on one coordinator, traverses route IDs for counts and
mapping, and clears/writes MAP entries even for inactive source slots (the binary
has5-source capacity although this run uses1). That is a concrete optimization
candidate, not proof that all562.8us belongs to one loop. Workers also read
per-route maps rowwise, and row-local quant/dequant repeatedly loads channel
scales and fences vector stages. Prioritize metadata construction and batched
vector processing before changing GEMM micro-scheduling. Preserve provenance:
this Qwen38 W8A8 whole-up/down path is not the earlier Next BF16 fine pipeline,
and these layer0 numbers are not an equal-work DFC comparison.

## Four-row ACTIVATE: reuse scales and batch UB work (September18)

`build.py --batch-activate` opts the target INT8 server into `quant_batch.hpp`.
This borrows upstream's row batching/scale reuse, **not** its whole mixed kernel
or its per-region Cube/Vector pipeline. The group-count GEMM, command protocol,
route-ready export, client retirement and BF16 MTP branch remain unchanged.

Each of16 movers receives a contiguous row range. Four-row batches stop at
expert boundaries; channel scales remain in UB until the expert changes. Cast,
SwiGLU stages and exports are batched, while rowwise maxabs/quantization retains
the previous operation order, rounding and padded8-float scale ABI. The existing
64KiB worker scratch fits all intermediates; no new GM workspace is allocated.
MTE3 completion fences protect reuse. Empty/tail workers still acknowledge the
original command. Binary and `abi.json` expose the compile-time option together.

`probe_activate_batch.py` loads `activate_probe.o` and compares both algorithms
inside one binary,16 AIV, FULL graphs,12 alternating event samples/mode.
`batch-activate-result.json` covers1/7/32/129/1024/4096/10003 rows, single/sparse/
wide expert distributions, changing inputs/scales/device group boundaries,
zero-valued rows and shrinking active M with sentinel tails. All compared
quantized output and padded scales are **bitwise identical**. These are dummy
INT32 intermediates, not language-model accuracy tests. Tiny shapes show no
reliable gain; absolute event times include replay/launch supply effects.

| Actual routed rows | Distribution | Old / batch median |
|---:|---|---:|
|1024|sparse|76.14 /58.75us|
|4096|sparse|222.27 /137.36us|
|4096|wide|317.83 /144.35us|
|10003|wide|731.30 /307.83us|

Fresh hw0 real layer0 A1+E3 runs compare the prior route-ready server with this
build, same five-source capacity/1024-row closure, same client and shared dummy
weights. They ran sequentially (not cross-run randomized), with alternating
client modes within each run. `batch-activate-server-result.json` preserves
capsules, both priorities, phase summaries and dual-source oracle results.

At1024 client tokens, online collection gives3.479/3.469ms ->3.194/3.194ms
(priority0/1): about8% shorter full leaf calls. Conventional fused collection
also improves4.050/4.041ms ->3.707/3.695ms. At32 tokens online is0.792/0.782ms
->0.770/0.760ms. Do not turn these leaf figures into a model throughput claim.

Coordinator ACTIVATE intervals fall257–258us ->126–128us on the3072-row owners,
and394.88us ->161.64us on the4096-row owner. They include worker dispatch/join;
the retained event ring mixes client arms, not an isolated kernel benchmark.
The improvement survives integration rather than merely moving work elsewhere.

A2+E3 independent true-weight oracle gate then passed:25 calls/source, matched
by every server, maximum sampled relativeL2 6.541e-6; eager/replay outputs exact.
Full48-layer/MTP and SWE serving qualification remain outside this gate. Public
Worker defaults and the default prototype build remain unchanged; pass
`--batch-activate --route-ready` when selecting this measured candidate.

Reproduce the leaf build using `device-service/build.sh`, with
`SOURCE=<qwen38>/activate_probe.cpp`, `OBJECT_NAME=activate_probe`,
`LAUNCH_SOURCE=<qwen38>/launch.cpp`, and a fresh `OUTPUT_DIR`; then run
`probe_activate_batch.py --build <OUTPUT_DIR> --output <receipt>` under single
card admission. The initial probe forgot the AIV kernel metadata section and
failed binary loading (no timing evidence); `activate_probe.cpp` now includes it.
All three admitted integration jobs exited0 and released their devices/leases.
