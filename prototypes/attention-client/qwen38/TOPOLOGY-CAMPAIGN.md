# Full-State and TP1/E3 topology campaign

Fletcher's September17 goal: compare useful context capacity and retained SWE
throughput across separated TP2×2+E4, TP1×4+E4, TP1×5+E3, and colocated
TP2×4/EP8 and TP1×8/EP8. All are equal-eight-card comparisons, not equal-State-
budget comparisons. Keep the repaired checkpoint, K1 and workload identity fixed.
No claim that these research controls are unmodified vLLM.

## Current capacity evidence

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
Launcher/model adapters accept `--tp-size 1`; full-model qualification is pending.

Static K1 State geometry from the TP1 meta declaration:
27,456B/history token/rank and466,790,428B/fixed request/rank, compared with
TP2's14,144B and233,547,804B. TP1 is not free capacity: it removes TP duplication
of the indexer but brings both KV/GDN head slices onto each attention device.

Next gates, in dependency order:
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
absolute. Full TP1+E4 real-weight gate is now running via
`/workspace/betterscale-hw0/runs/qwen38-tp1-model-run.sh`.

E3's Python catalog now has an explicit171/171/170 partition and zero-padded
last storage slot. CPU routing coverage passes for every expert ID. The binary
still advertises E4/two sources: **E3 launch remains unavailable** until its
route mapping, descriptor and weight ABI are changed and verified together.

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
comparison. Full48 TP1×5+E3 is the next hardware gate.

The equal-card workload is now40 distinct SWE sessions so2/4/5/8 attention
sources receive equal integer seat counts without cloning trajectories. The
frozen40-session artifact contains1830 full turns; first comparison uses the
first2 complete turns/session, with no output cap. Full-trajectory results remain
separate. `run_topology_case.sh` owns a single admitted case and always preserves
its capsule/exit status; both capacity and trace use C40. Current queued fits:
TP1×8/EP8 State30GiB, TP1×4+E4 State44GiB, TP2×2+E4 State48GiB, TP2×4/EP8 State34GiB.
These are test candidates, not passed budgets. Keep scope distinctions intact.

TP1×5+E3 full48+MTP also passed (`qwen38-model-20260917T164125Z`): all5
attention FULL shadows exact, all3 servers drain every source, clean exit.
All requested topology implementations now have full-model gates; capacity and
matched workload results remain outstanding. E3 C40 State44GiB joined the queue.
The40-session two-turn workload is80 turns,14,362 output tokens,236,306 prefill
rows including40 continuation anchors, maximum selected horizon9,128 tokens.
This short matched pilot does not itself exercise the long-context capacity fit.
