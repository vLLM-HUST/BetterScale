# Full-State and TP1/E3 topology campaign

Fletcher's September17 goal: compare useful context capacity and retained SWE
throughput across separated TP2×2+E4, TP1×4+E4, TP1×5+E3, and colocated
TP2×4/EP8 and TP1×8/EP8. All are equal-eight-card comparisons, not equal-State-
budget comparisons. Keep the repaired checkpoint, K1 and workload identity fixed.
No claim that these research controls are unmodified vLLM.

## Earlier C32 capacity evidence

`probe_capacity.py` physically fills all State and runs real48+MTP weights over
synthetic zero history: long-prefix prefill, capture and3 FULL decode replays.
It does not validate long-prefix language quality. It never clones the State.
Use sampled driver-free memory and allocator peaks separately; subtracting a
historical peak reserve from later driver-free memory is not a valid headroom
measurement. State budget excludes weights, graph workspace and communication.

hw0 capsules (under `/workspace/betterscale-hw0/repo/runs/`):

- `qwen38-model-20260917T160602Z`: TP2×2+E4,48GiB/rank PASS,32requests,
  3,379,712 logical history tokens/group,211,072-token exercised prefix/request.
- `qwen38-model-20260917T161154Z`:50GiB fails during warm-prefill QSA HCCL
  allreduce allocation. Not a State declaration failure.
- `qwen38-model-20260917T161622Z`:49GiB PASS on all4 attention ranks,
  3,455,616 logical history tokens/group,215,808-token prefix/request;
  minimum sampled driver-free across ranks561,909,760B. This is a fit edge,
  not automatically a safe production setting for arbitrary larger waves.
- `qwen38-model-20260917T160937Z`: colocated34GiB fails A2 MC2 tiling with
  `batchsize is invalid`, not OOM. The1024-row attention bucket gives512 unique
  rows/EP rank. A2 dispatch admits256; colocated_ep now splits fixed row chunks,
  invokes shared computation once, and preserves masked EP participation.
  Revised34GiB trial is separate; do not reuse this failure as a memory bound.

Do not multiply TP2 head-sharded capacities by two. Also distinguish physical
history pool capacity from usable C32 capacity under the262144/model context
limit and lookahead reservation.

## TP1 and server expansion gates

`prepare_tp1_overlay.py` builds a disconnected closure. It extends head ownership
and native paged gather/publication/FIA to2 KV heads and24 Q heads. Selection
remains shared: no duplicate indexer and no full-cache transpose. Never overwrite
an existing overlay or alter the TP2 control while a capsule is running.
`probe_tp1_qsa.py` is the one-card dummy eager/FULL-page test before full weights.
Launcher/model adapters accept `--tp-size 1`; all four expanded full-model gates
have now passed (see the checkpoint below).

Static K1 State geometry from the TP1 meta declaration:
27,456B/history token/rank and466,790,428B/fixed request/rank, compared with
TP2's14,144B and233,547,804B. TP1 is not free capacity: it removes TP duplication
of the indexer but brings both KV/GDN head slices onto each attention device.

Completed implementation sequence (historical order):
1. Finish current TP2 fit receipts; qualify TP1 page kernels and model construction.
2. One TP1+E4 full48+MTP shadow; native TP1×8/EP8 full-model control.
3. Expand external server source count (currently2), preserving generation,
   priority promotion, layer-isolated batching and retirement. Separate source
   count from the two staging slots; these are NOT the same dimension.
4. E3:512 routed experts do not divide3. Explicit contiguous ownership ranges
   and catalog padding must agree with route→owner/local mapping, NZ group ends,
   client pointers and binary ABI. Do not change only launch/configuration.
5. Qualify four/five-source traffic, then actual memory fits and matched SWE
   workloads. Retain failures and nonqualification; no estimated win as a result.

Current admission launchers on hw0 are bounded, use `/root/tp8.lock`, and stop
only owned children. See `/workspace/betterscale-hw0/runs/capacity-fit-20260917/`
and `/workspace/betterscale-hw0/runs/qwen38-tp1-leaf-run.sh` for live execution.

Updated receipts: `capacity-fit-result.json` records all-rank49GiB separated and
34GiB colocated passes. Revised native capsule161939Z has2,448,960 history
 tokens/group (9,795,840 allocated across4 groups); C32 exercised prefixes are
261,952/request, bounded by model context rather than pool capacity. Minimum
sampled driver-free is539,041,792B.34GiB is a passed fit, not an exhaustive upper
bound. No additional upward step is justified for these current pressure shapes
without preserving working runtime margin.

TP1 QSA leaf passed one/two-KV-head cases, rows1/4, changed inputs and FULL replay,
with exact page contents and exact one-selected-row GQA outputs. Remote artifact:
`runs/qwen38-tp1-qsa-leaf-20260917/run2/run.log`. The initial run failed before
Python import because admission changes cwd; all probe entry paths must be
absolute. The first full TP1+E4 real-weight gate was launched via
`/workspace/betterscale-hw0/runs/qwen38-tp1-model-run.sh`.

E3's Python catalog now has an explicit171/171/170 partition and zero-padded
last storage slot. CPU routing coverage passes for every expert ID. The matching ABI4/ABI5 binaries are now qualified; never pair this catalog with
the old E4 binary.

## Expanded protocol checkpoint

`topology-gates-result.json` now records full48+MTP FULL-shadow passes for one
TP1+E4, TP1×8/EP8 and TP1×4+E4. Early TP1 receipt topology text still saysTP2;
launcher inputs and this summary carry the correct topology, and new receipts
explicitly record TP width/source count. Do not alter historical raw receipts.

E3/two-source binary ABI4 passed its real-layer wire gate. ABI5 adds4/5 source
mailboxes independently of the two staging slots. Source/output pointer tables
are config27/28; the server config is29 words, client config remains17 words
(with an unused fourth owner pointer forE3). Traces are32 words: generations,
layers and rows each have `sources` entries, followed by live rows, slot,
priority, ticket and service rank. Completion counters cover every source.
`topology_codegen.py` changes only the disconnected generated closure. It does
not enable old segmented/prefix-pipeline flags.

Five sources+E3 and four sources+E4 passed real layer0, row counts1/4/32/1023/1024,
changed-input FULL replay and sampled independent arithmetic. Each source
completed25 calls. E3 observed5 coalesced calls/owner; E4 observed0 in this cold,
CPU-oracle-interleaved test. This is not evidence against batching or a throughput
comparison. Full48 TP1×5+E3 subsequently passed as recorded below.

The equal-card workload is now40 distinct SWE sessions so2/4/5/8 attention
sources receive equal integer seat counts without cloning trajectories. The
frozen40-session artifact contains1830 full turns; first comparison uses the
first2 complete turns/session, with no output cap. Full-trajectory results remain
separate. `run_topology_case.sh` owns a single admitted case and always preserves
its capsule/exit status; both capacity and trace use C40. Original C40 fit candidates were:
TP1×8/EP8 State30GiB, TP1×4+E4 State44GiB, TP2×2+E4 State48GiB, TP2×4/EP8 State34GiB.
Their completed outcomes are recorded below. Keep scope distinctions intact.

TP1×5+E3 full48+MTP also passed (`qwen38-model-20260917T164125Z`): all5
attention FULL shadows exact, all3 servers drain every source, clean exit.
All requested topology implementations now have full-model gates; capacity and
matched workload results remain outstanding. E3 C40 State44GiB was also tested.
The40-session two-turn workload is80 turns,14,362 output tokens,236,306 prefill
rows including40 continuation anchors, maximum selected horizon9,128 tokens.
This short matched pilot does not itself exercise the long-context capacity fit.


## C40 workspace diagnosis and common bounded-QSA path

Original-path capsules on hw0, September17:

| Capsule suffix | Layout | State GiB/rank | Outcome |
|---|---|---:|---|
|164426Z|TP2×4/EP8|34|PASS, 2,415,936 history tokens/group; 241,370 exercised prefix/request|
|164854Z|TP2×2+E4|48|PASS, 3,313,664 history tokens/group; 165,517 prefix/request|
|165339Z|TP1×8/EP8|30|OOM allocating gathered V, a second 2 GiB buffer|
|165642Z|TP1×4+E4|44|OOM allocating first 2 GiB FIA layout copy|
|165944Z|TP1×5+E3|44|Same FIA layout-copy OOM|

These TP1 failures are not irreducible State capacity limits. At1024 query
rows,2051 selected rows,2 KV heads,256 head dimension and BF16, each gathered
K/V is about2GiB. The original implementation additionally materializes both
transposed layouts for FIA. Even before those copies, K+V costs about4GiB.

The new disconnected overlay keeps full Q, indexer and top-k computation.
Only QSA consumption is split into at most128 query rows, preserving each
query's complete selected KV set and request-to-page mapping. Gather writes
physical B,H,S,D directly and exposes B,S,H,D as a view; FIA's transpose then
needs no contiguous copy. Apply this same path to TP1 and TP2 controls.
It is not a streaming-indexer or approximate-top-k optimization.

`probe_tp1_qsa.py` passed8 leaf cases on the b overlay: KV heads1/2, query
rows1/4/129/257, changed-cache generations and FULL replay. Independent page
indexing equals the fused gather exactly; nondegenerate selected33-row attention
is exactly equal to unchunked FIA. hw0 evidence:
`runs/qwen38-bounded-qsa-leaf-20260917/run3/run.log`. The c overlay adds only a
fail-closed row-map check before query slicing. A failed earlier compilation
(2D versus1D branch-local Triton SSA output name) is retained in run2; fixed by
using a distinct inactive-store variable, not by suppressing the check.

`run_topology_case.sh` defaults to the common `bounded128` c overlays; the
optional `original` argument preserves the historical route. Each case records
its exact overlay/build/configuration and freezes the Python sources in its
capsule. New capacity candidates51/36/48/48/33GiB (TP2-E4/TP2-EP8/TP1-E4/TP1-E3/
TP1-EP8) are being measured, **not yet passed capacities**. Their parent is
`/workspace/betterscale-hw0/runs/topology-bounded-20260917/`.
The original queued SWE cases were cancelled before launch so the comparison
will not mix memory implementations.


## Completed bounded fits and padding regression (supersedes pending above)

C-overlay C40 real-State passes: TP2-E4 51GiB (7,082,752 allocated history
 tokens/machine), TP2-EP8 36GiB (10,271,232), TP1-E4 48GiB (6,828,544),
TP1-E3 47GiB (8,510,080). Corresponding exercised prefixes/request:
176,909 / 256,602 / 170,522 / 212,544. E3 48GiB is NOT qualified: one source
failed warmup `aclnnGather` allocation while the other four completed. E3 47GiB
passed all sources (`174851Z`), but minimum sampled free was only145,285,120B;
use46GiB for the throughput pilot rather than presenting this as a safe reserve.

TP1-EP8 33GiB (`173632Z`) and31GiB (`181145Z`) were stopped during extremely
slow warmup, not certified OOM or collective deadlock. The stack at33GiB was
inside native MC2 dispatch, but at31GiB it moved to residual injection. A stack
snapshot is not causal attribution. These are incomplete fits, not capacity
limits. Old C-layout queued traces were cancelled before launch where marked.

The C head-major layout introduced a performance regression absent from its
small correctness gate. One instrumented real SWE prefill atState8GiB
(`181908Z`) takes38.72s: sum of layer device intervals35.67s, MoE0.232s, shared
0.026s. Each of12 QSA layers is about2.94s. Its inactive gather branch expresses
zero stores using vector division/remainder, losing the affine contiguous-store
form. Runtime falls as padding decreases. The d overlay restores contiguous
stores separately for each KV head; no Q, top-k, cache or attention semantics
change. Full2051-slot cases were added to the leaf, not merely small33-slot
cases. `qwen38-bounded-qsa-leaf-20260917/run4` passes all10 cases exactly,
including FULL replay and independent gather/FIA oracles. End-to-end timing is
being checked separately; do not call the d optimization successful from the
leaf's correctness alone. `QWEN38_TRACE_DIAGNOSTIC=1` creates one-wave stage
receipts labelled DIAGNOSTIC, never usable as completed workload throughput.

A preliminary CPU hypothesis was independently tested and was insufficient:
PLE lookup was not responsible for the seconds-long QSA intervals. Its separate
safe improvement (`ple_lookup.py`) retains native n-gram hashing, fetches only
the last needed history position and groups shard rows once rather than scanning
every ID for every shard. Six CPU synthetic-shard oracles pass exactly. A real
checkpoint1020-lane test is also exact: warm native25.59ms versus grouped-last
6.48ms. The first native call was53.64ms with71 major faults. These are CPU-only
observations, not an end-to-end speedup. Raw evidence lives under hw0
`/workspace/betterscale-hw0/runs/qwen38-real-ple-cpu-v2.log`; the reusable probe
is `probe_ple_lookup_real.py --trace-plan .../qwen38-swe40.json`.


The d full-model diagnostic is now complete (`182739Z`), same TP2-E4 State8GiB
C40 first wave: rank0 wall38.7219 ->1.12478s; summed layer device intervals
35,670.7 ->997.0ms. See `qsa-padding-result.json`. This is a regression repair,
not a new topology speedup. All five normal trace cases now select d, with PLE
lookup fixed identically, under `/workspace/betterscale-hw0/runs/topology-affine-20260917/`.
Native TP1-EP8 capacity is retried there before its trace, since the old c
warmup stops cannot establish its limit.


## Native MC2 mask contract correction

The d repair does not fix native TP1's layer0 warmup stop. The512-lane retry
(`183923Z`) explicitly logs only layer0 entry, then waits in DispatchV2; the
1024-lane d trial is `183043Z`. Neither is qualified as OOM. We stopped these
owned jobs rather than allowing unsupported inputs to run until a long timeout.

[CANN's DispatchV2 documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/aolapi/context/aclnnMoeDistributeDispatchV2.md)
requires a1D active mask to have all true entries before false entries. Our
request-padded prefill and finished decode seats can have holes. Therefore old
native masked workload passes are not sufficient qualification; retain them as
historical memory observations, not reliable legal-input performance controls.
This is separate from the QSA padding regression and from State capacity.

`colocated_ep.compact_prefix` now uses device prefix sums to place valid rows
first in each fixed MC2 chunk. It carries hidden/route IDs/probabilities through
the same permutation and restores combined results to original token order.
Shared expert still consumes the original local tokens exactly once. No host
route count or dynamic CPU allocation is introduced; the graph sees fixed sizes.
`probe_ep_prefix.py` checks16 CPU masks, including holes/all-inactive/all-active,
with exact stable ordering and inverse. Hardware qualification is pending.
The CLI now permits `--token-capacity` below the physical wire bound, recording
it in case parameters; the final intended comparison still uses1024/source.
The temporary512-lane diagnostic is not silently substituted into the matrix.

### A2 dispatch argument gate and long-run failures

The pinned vLLM-Ascend dispatcher only supplies TP communication arguments on
A3/A5. This A2 MC2 control now omits `group_tp`, `tp_world_size`, `tp_rank_id`
and `tp_send_counts`, as upstream does. The isolated real layer0 eight-card
`probe_native_ep.py` passes eager/FULL exact output, finite results and zero
inactive rows at32/256/1020 source rows, with broad and narrow routing and
non-prefix input masks compacted by the adapter. Receipt:
`hw0:repo/runs/qwen38-native-ep-leaf-20260917b/run/`.
The initial leaf failed before dispatch because NZ internal format was not
enabled; the corrected leaf explicitly enables it, like the model runner.

The legal TP2xDP4 full-model36GiB fit (`185923Z`) passes:10,271,232 allocated
history tokens/machine,256,602 exercised prefix/request, minimum sampled free
399,052,800B. It remains a zero-history memory gate, not language quality.

Two d-layout C40 two-turn traces failed with native SIGSEGV after many waves:
TP2-E4 State50GiB (`184816Z`) and TP2xDP4 State35GiB (`190147Z`). Their partial
wave timings are **not** completed workload throughput. A common cause is not
yet established; neither OOM nor expert transport is inferred from the signal.
The launch capsule now records role PIDs and runtime directory; trace diagnostics
sample allocator/driver memory every20 waves. Native stack capture is being used
on a matched reproduction. Avoid dumping every thread stack into model context.

The TP1-E4 d diagnostic (`190637Z`, with debugger attached) was stopped before
completion: prefill time increased sharply as active prefix length grew. The
active two-head gather stores still contain vector head division/remainder,
unlike TP2's constant single-head expression. The e overlay replaces them with
separate affine per-head loads/stores. This is a hypothesis until the paired
`probe_qsa_gather_cost.py` and full2051-slot correctness gate complete; do not
quote its expected benefit as measured throughput.

### Timed traceback crash and active-gather result

The TP2-E4 reproduction `191957Z` was observed under GDB. Attention1's SIGSEGV
is in CPython3.12 `dump_frame -> dump_traceback -> _Py_DumpTracebackThreads ->
faulthandler_thread`, triggered by our240-second `dump_traceback_later`, not in
an expert kernel. That scheduled traceback has been removed; external role
supervision remains bounded. This localizes the reproduced fault, not a claim
that all historical signals have been independently assigned. Full traces need
rerunning without the diagnostic timer.

The paired active gather probe validates the e hypothesis. For128 query rows,
2051 selected slots,2 KV heads and1024 valid selections/query, d takes
371.23–371.64ms versus e0.911–0.918ms. Head1 is unchanged (~0.73ms).
Both match the independent cache-row oracle exactly. See
`qsa-active-result.json` for all three samples, including an outlier41.68ms in
the e count128 case; no sample is deleted. This is a layout-lowering repair,
not an expert-server speedup or a completed end-to-end result. The e full
publication/gather/FIA graph oracle gates the next topology runs.

The e FULL graph publication/gather/FIA gate passes all10 cases (`qwen38-qsa-active-20260917b/exact`), including two heads,129/257 query rows, and2051 selections. The next matrix uses `affineheads` explicitly, not an overwritten old overlay.

The corrected TP1-EP8 State33 fit still stalls in HCCL initialization
(`194317Z`; native stack `WaitCommThread/CreateCommByAlg`). A one-time allocator
cache-reclaim retry (`200716Z`) does not change allocated/reserved memory and
also stalls. Crucially its native driver log explicitly reports OOM, including
failed211,812,352/421,527,552-byte allocations and persistent device-OOM status.
This is now an observed failed fit, not merely an inference from a Python stack.
The ineffective empty-cache experiment is not retained in the implementation;
its exact source is in the capsule. Lower31/32GiB fits are the next bounded
steps. Inspect the native plog when an apparent communication startup wait
has no Python exception; do not let it consume a whole20-minute timeout blindly.

Matched C40/two-turn e traces now complete for both TP2 layouts:
`192818Z` separated TP2x2+E4 uses702.869s (20.433 output tok/s), and
`195256Z` colocated TP2x4/EP8 uses724.010s (19.837 output tok/s).
Both perform236,306 prefill rows,14,362 outputs and193,043 retained-prefix
reused tokens. This single pair is only about3% apart, **not** a robust scaling
win. TTFT P95 is109.21/70.58s; output-interval P99 is43.11/54.38s. The prototype's
fixed-width prefill and phase scheduling dominate these tails. These remain
owned-runtime topology controls, not an unmodified vLLM performance claim.
