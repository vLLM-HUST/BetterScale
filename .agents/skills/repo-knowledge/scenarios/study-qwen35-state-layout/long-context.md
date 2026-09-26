# Restore the live long-context serving envelope

Enter before extending or benchmarking the optional live root's context,
prefill, shared-page fit or exact-token streaming. This is the continuation of
`seat-scheduler.md`, not evidence that its original512-token gate covered256K.

## Accepted boundary (Fletcher, 2026-09-26)

Context capability must not regress from the native262144 envelope. Fixed
resident seats pay for their GDN/MTP/continuation State; the remaining physical
budget, after weights, execution calibration and explicit native allowance,
belongs to pooled attention pages. **Do not cap pages at R × context pages.**
Context length limits one sequence's addresses; it is not a pool-size policy.
Keep the root's existing calibration→fit→rebind→recapture transaction rather
than implementing another allocator or changing captured addresses after READY.

The requested final comparison uses the unchanged SWE-prefix-reuse workload,
full capacity and C8/C16. Do not truncate/filter it or substitute native results
for live results. The retained September24 matrix workload has360 turns, maximum
prompt140423 and maximum prompt+output141269;339 turns exceed4096. Its streaming
client requires actual output token IDs, prompt identity, usage and session salt.
A nonstreaming completion timestamp is not a substitute for token-arrival timing.

## Implementation seams and paid observations

- `bootstrap.py` no longer supplies a model-derived State byte ceiling. The
  backend charges exact resident declarations before its elastic page domain.
  Minimum calibration pages cover the widest declared execution footprint,
  not all possible request histories.
- The owned Ascend paged-attention resource reuses the qualified CANN planner
  and numerical launch, **without** installing its native runner patch.
  Prefill/decode graphs remain owned by the live root. Page-table metadata is
  O(context/128); no repeated-head, full-context KV gather is needed.
- Planning fixtures must hold **address/shape descriptors**, not borrowed
  calibration-State tensor objects. Keeping a tensor in a planner tuple would
  retain the old allocation during fitting. Rebind descriptors before final
  planning; fixed metadata/transport resources survive the transaction.
- GDN prefill consumes candidate zero in place. The first ordinary target
  step after a hot hit selects the previous accepted candidate and normalizes
  convolution history; only then may bulk prefill use candidate/history zero.
  Native conv FN ignores accepted-token metadata, whereas update mode reads
  history at accepted−1. Merely switching to FN on a hot MTP boundary is wrong.
- Chunked draft prefill pairs tokens at p..p+T−1 with target hidden at
  p−1..p+T−2. Retain only the current chunk and one boundary hidden, not context
  history. Prefill samples only the last target row; draft prefill needs no
  greedy head result. Its token budget is independent of execution-seat count.
- Whole-seat preemption must not duplicate streamed output. Persist committed
  output progress across recomputation, but never report recomputed prefix or
  unaccepted proposals. Cache salt is part of hot resident identity.
- Final one/two outputs need no speculative lookahead beyond max_tokens.
  Ordinary target steps at the tail preserve the entire context for committed
  tokens, rather than reserving two unusable positions globally.

Local capsules below are under workspace `runs/qwen35-state-lanes/`. Each is
frozen before hardware admission; use its source, not mutable current work.

1. `20260926-long-fia1`:16 actual graph-replay comparisons, Q8/KV1 and Q8/KV2,
   including256K context,1024 query rows and C16. All match native FIA exactly
   (observed maximum error0). Operator evidence, not whole-model qualification.
2. `20260926-paged-root1`: real0.8B, C4/R5,128 context,12 graphs, automatic
   calibration/fit/recapture. Fitted33570 shared pages while retaining5 seats;
   all4×8 output IDs match the earlier plain-attention small3 receipt and its
   new serial controls. Selected card0 returned IDLE. This closes the former
   seat/context-derived capacity cap, not a model256K or throughput gate.
3. `20260926-chunk-leaf1`: captured64/256/1024-token GDN chunks with QK8/V16
   and QK16/V16. Independent FP32 sequential recurrence; maximum output error
   1.6861e−5 and final-State error1.02092e−4, untouched candidate lanes exact.
   This qualifies the numerical leaf geometry, not a model output tolerance.
4. `20260926-chunk-root1`: real0.8B C2/R3,143/333-token inputs,256-token chunks,
  16 graphs. Both nine-token exact-copy outputs equal single-token target-only
   controls, and hot-prefix extension equals cold chunked recomputation.

The35B TP2 long-context/C16 gate and installed HTTP/leaderboard gates remain
separate; do not promote these operator/small-model observations into those
claims. The receipts below close model and HTTP lifetime gates; the measured C16
performance gate failed substantially.

## Fixed-first sizing correction (2026-09-26)

Fletcher rejected OOM-driven capacity search. After graph calibration, allocate
exact resident domains first, observe real driver-free bytes, reserve the
explicit free floor and allocator rounding allowance, and divide the remaining
budget by the complete attention-page cost. TP ranks agree on the minimum
**before** the one final elastic allocation. Failure is not retried at successively
smaller page counts. Do not credit inactive allocator cache as freely composable
capacity. The pinned Ascend bootstrap supplies20MiB per KV tensor as a conservative
large-segment rounding allowance (22 tensors for35B TP2,440MiB total); this is
separate from the1GiB free floor, not a seat/context quota or a measured waste.

`20260926-long-tp2-1` failed before READY in the old eight-attempt fitter.
`20260926-long-fit-diagnostic1` froze that same behavior and recorded per-rank
waterlines: about25310MiB free before full-State allocation, then only818–835MiB
free against the1024MiB floor. All eight candidates had identical allocated
(60147.65MiB) and reserved(60494MiB) waterlines despite decreasing page counts.
This proves the single-page decrement did not change physical allocator charge
in that window; it does not establish the exact composition of each fragment.
Do not call either capsule a long-context or model correctness result.
`20260926-long-tp2-2` passed both ranks with16981 pages (2173568 shared token
slots),30 graphs and20 seats. Raw12/warm8 IDs match independent native controls;
8193/32769/131073/262080-token retrieval returns the native exact code; actual C16
matches all16×12 native IDs. Both selected devices0/1 returned IDLE. The installed
HTTP/SSE and benchmark remain separate gates; see the compact source receipt in
`docs/evidence/qwen35-live-long-context.json`.

## Compare actual capacity, not just labels

At35B BF16 TP2 each logical token costs11264 bytes **per rank** in attention:
11 layers (10 target +1 MTP) × K/V ×1 local KV head ×256 dimensions ×2 bytes.
A128-token complete shared page costs1441792 bytes. TP ranks store shards of
these same logical positions; do not add their token counts together.

Declaration-derived R20 resident bytes are1912095760 (1.780777946GiB):
1887436800 recurrent candidate bytes (30 layers ×20 seats ×3 candidates ×16
heads ×128×128 FP32),24576000 conv bytes, and82960 continuation bytes. Increasing
R16 toR20 costs382419152 bytes (0.3562GiB), equivalent to33950.56 attention
tokens before page rounding, only about1.6% of the observed2.17M pool.

The same-model TP2 C16 native baseline in
`/workspace/my-ascend-workspace/runs/swe-frontier-sequential-20260924/native16/server.log`
explicitly allocated24.25GiB of hybrid State and printed2072810 tokens/7.91×256K.
It is a capacity observation, not a new correctness endorsement (its old C16
functional limitations remain). Its mixed pool has1107 logical blocks at
11×2138112 bytes each. The capacity formula charges128 attention blocks plus
3 GDN groups ×(2 alignment +2 speculative) blocks per262144-token request:
`int(1107 /140 *262144) =2072810`. It is not a literal count of pure KV slots.

The live16981-page observation pays22.801635742GiB for pure attention plus
1.780777946GiB for R20,24.582413688GiB State total, and has2173568 literal token
slots. This is4.86% above that native printed capacity with0.3324GiB more State
bytes; do not attribute the entire difference to layout. At the same24.25GiB
logical State budget, subtracting R20 then rounding to pages predicts16733
pages/2141824 tokens (+3.33% versus native's256K effective-capacity figure).
That last row is arithmetic, not a newly allocated native/live control pair.

Fletcher initially paused leaderboard work for the capacity audit, then resumed
**C16 only**, using the installed package and a full900-second window. C8 remains
paused. `20260926-long-http1` passed native-token/SSE/C16/salt and cancellation
checks but exited1 while Uvicorn restored a saved signal handler (value unknown).
The owned HTTP signal scope now normalizes an unset handler and returns normally
through distributed cleanup instead of re-raising graceful shutdown. Package2
passed52 installed CPU tests; the completed C16 service below exited0.

`20260926-capacity-ledger1` isolates capacity accounting without generation or
HTTP. The already model-qualified source reached READY with16982 pages; the
one-page difference from long-tp2-2 is an observed startup free-waterline change,
not a changed capacity policy. Rank0's actual byte ledger closes exactly to
60.95703125GiB visible to torch:

| Exclusive category | GiB |
| --- | ---: |
| Model-owned parameter/buffer storage | 33.292192 |
| R20 resident State | 1.780778 |
| Shared attention KV | 22.802979 |
| Other active allocator storage | 0.260428 |
| Default-pool inactive fragments | 0.176123 |
| Graph-private pool reservation | 0.162109 |
| Driver use outside torch allocator | 1.083626 |
| Actual driver-free memory | 1.398796 |

The last free row already contains the1GiB floor and surviving rounding allowance;
do not add the440MiB planning allowance again. Graph-private reserve is not
available KV; other active tensors and outside-allocator usage are measured
residuals, not claims of fully attributed graph/communication cost. Hardware
reports64GiB HBM, while torch exposes60.957GiB; keep that3.043GiB visibility
boundary separate from usage inside the torch-visible budget. Both TP ranks hold
shards of the same token pool, not two independent2.17M-token pools.

## Installed C16 performance gate — regression, not acceptance

Tested source `6bc65464f16ed1799f970e34d8184a39df1919e3`, installed wheel from
`20260926-long-package2`, SWE client `4b452cfcade95870a4904d71738bb539d5af5f44`.
`20260926-live-swe/c16-lease2` ran the unchanged workload/seed20260924 for900s
on local physical0/1, BF16 TP2 MTP2, E16/R20, context262144. Actual max batch16,
mean client inflight15.99925; all341 requests succeeded,325 completed in-window,
16 drained in227.24s. Service and guard exited0; both cards returned IDLE.

Primary control is the earlier **BetterScale small-fish C16**, not native or the
weaker20.25GiB run: `runs/betterscale-mtp-small-fish/20260924T160000Z-qualification/
swe/optimized35b-c16/c16` (run `fcdf3dd59edf4b4b89f3931f554297a4`).

| Metric | Prior BetterScale | Owned live |
| --- | ---: | ---: |
| Output tokens/s/chip | 368.9278 | 87.4556 |
| Decode tokens/s P90 | 55.3789 | 13.2454 |
| TTFT seconds P95 | 0.82347 | 6.04224 |
| Requests completed in900s | 1055 | 325 |

Throughput fell76.29%; this does **not** pass non-regression. No request was
preempted. The16913-page pool had at least13051 free pages in222 samples
(1670528 token slots); this run did not exhaust capacity. Actual C16 and almost
full client concurrency exclude merely failing to submit16 clients, not all
execution scheduling overhead. The bottleneck is not yet localized.

Same workload does not mean the closed-loop run visits the same turns: this
slower run reached maximum prompt40244 versus90095 in the control. Runtime,
prefill budget (1024 vs4096), graph portfolio and State policy differ. The
control uses24.25GiB hybrid State; this uses fixed seats plus automatic remaining
KV. Shared-host conditions and a single window preclude attributing the loss to
one mechanism. Do not publish this as a performance win or silently substitute a
weaker baseline. Full receipts and comparison live in the capsule and compact
`docs/evidence/qwen35-live-long-context.json`.

The preceding `c16-lease1` attempt is invalid: the terminal-associated guard
exited143 mid-window, while the child retained inherited device leases. Owned
workers were explicitly stopped, service exit0, cards released. The trigger is
not established. The retry detached the outer launcher from the terminal and
made the serving supervisor fail closed if its admission-guard parent vanished;
no serving source changed. Keep long-run supervision independent of a transient
terminal and preserve parent-liveness cleanup, not just inherited lock FDs.

## Diagnose the C16 regression before changing State again

`20260926-live-swe/c16-profile1` captures scheduler ticks500..579 on both ranks
of installed6bc6546 plus capsule-only profiler/CPU annotations. TraceLoomab8b513
recovers119 exact graph launches and197345 exact body members per rank. Rank0's
3334.26ms tick span contains2617.48ms of graph envelopes and707.44ms between
graphs (rank1:3341.72/2652.38/679.42ms). The median between-graph gap is5.41ms
(rank1 5.11ms). Thus do not blame only host gaps; most sampled time is in graph
bodies. Synchronize API durations mostly include waiting for those same bodies
and must not be added as independent overhead.

There is a concrete numerical-dispatch regression. For the same B8/3-token
**target shape**, old small-fish candidate3 has1266 exact members, versus3425
in the new root:80 AddRmsNormBias kernels become0, with131 Pows and131 ReduceMean
kernels absent from the old graph. Native Worker initializes
`register_ascend_customop`, which selects AscendRMSNorm, AscendGemmaRMSNorm and
other optimized leaves; the owned bootstrap only checks `enable_custom_op` and
constructs core models without that registration. Runtime logs explicitly warn
that RMSNorm/fused-add-RMSNorm have no selected priority and use native fallback.
This is lost optimized dispatch while replacing execution ownership, not proof
that exclusive State lanes or LiveModule intrinsically cost more. Restore the
intended numerical composition deliberately; do not blindly import the whole
native Worker or assume a global registration change preserves root semantics.

Other observed/source-supported seams remain distinct: C16 ready work splits
into serial binary buckets (e.g.14 target rows ->8+4+2); MTP proposals and
accepted-hidden repair use separate graph calls; `_run` synchronizes before and
after output clones; per-seat protocol uses scalar readbacks and small writes.
The profile has287 item readbacks and2697 copy_ calls on rank0. Nested host API
sums are not an additive cost ledger. MTP acceptance rate is not established by
this capture, nor is each seam's share of the900s regression.

Old candidate3 C8 decode/mixed timelines are structural references, not matched
C16/context controls. New target B8/3-token body median42.93ms across5 launches
cannot be promoted to a causal layout ratio. See
`docs/evidence/qwen35-live-c16-profile.json` and capsule `analysis/findings.json`;
raw profiles, two rank Perfetto timelines, exact SQLite and same-scale SVG stay
in the capsule. No production code changed during this diagnostic.

Capture boundary: both finite windows and exports completed, but synchronous
profiler export took~79s and exceeded the supervisor's60s health timeout. The
surrounding180s diagnostic client was interrupted; no successful SWE score is
claimed. Owned service exited0 and both cards released. The later tick679
sidecar was not written, so graph shapes are instead matched from official
profiler CPU labels to CANN execute connection IDs. Future probes must save
sidecars before export and move export offline or give bounded finalization its
own timeout rather than rerunning the same health-timeout failure.
