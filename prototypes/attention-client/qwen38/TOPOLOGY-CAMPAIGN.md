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
