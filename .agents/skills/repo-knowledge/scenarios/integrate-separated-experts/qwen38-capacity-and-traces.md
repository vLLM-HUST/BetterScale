# Qwen38 capacity and retained-trace gates

Historical gates and current hazards. Use TOPOLOGY-RESULTS.md for current numbers;
read only the section needed for the proposed change.

## Multi-request capacity and batch gates

Qwen38 `--batch-size` now provides independent request State/PLE/block-table rows
within each TP2 source. The full-model wire still caps32rows/source; this is not
hardware capacity. `hw0-batch-scaling.json` qualifies per-source8/16/32 short-context
requests on two TP2 groups+E4:126.87/162.83/189.03tok/s, with increasing124/196/338ms
step medians. All same-State shadows are0 and within-run output IDs agree across
sources/ranks. Batching gains diminish; expert-only batching efficiency does not
prove linear end-to-end scaling. See README for inputs, cards and exclusions.

Use `qwen38/estimate_capacity.py` with the selected runtime overlay to recover
State payload geometry without NPU allocation. It needs plain
`LiveRuntime(device="meta")`, **not** AscendArchitectureBinding(meta). Current
TP2 payload is14,144B/history token/rank plus116,581,396B/request/rank (mostly GDN
active+scratch). Unused MTP QSA State is still declared and included. Never count
TP2 head slices as two independent copies of logical token capacity.

Large resident-budget testing must not blindly reuse the full-State cloning
shadow fixture: a40GiB State would be cloned to another40GiB. The current8GiB
batch tests avoid this confound and use only two64-token pages/request; static
40GiB estimates do not qualify graph fit or populated long contexts.

## Full-model MTP extension

Enter `qwen38/MTP.md` before using `--mtp-tokens`. Existing server ABI already
selects the BF16 layer48 alongside INT8 target layers; the integration reuses
owned draft/commit/reconciliation programs. K1 B2 dual passes FULL graph exact
output/count shadow and first16 sequential-target reference tokens. K1 B16 is
~161tok/s vs prior target-only~163: do not claim a throughput win. The32-row
source ceiling applies to B*(K+1), and K=0 State estimates exclude K>0 candidates.

Wider-query strict token reference can fail before any server fault. The K3 B8
first verification matches target-reference layers0/1 exactly; layer2 HC has
exact normalized input but5.58e-5relative difference in its stateless BF16 down
projection(width1 vs4), before attention/experts. See `mtp-shape-divergence.json`.
Later error amplifies. Preserve the failed reference gate; this is not language
quality qualification or proof that all later discrepancies are harmless. It
prevents repeating transport/state speculation without checking the first
numeric divergence. Larger-shape speed claims still require independent quality.

## SWE equal-card expansion

Enter `qwen38/SWE-COMPARISON.md` for the approved same-model topology control.
The pinned cohort is downloaded on hw0; its32 complete sessions have1,445turns
and new-prefill P95=4,732tokens. Do not reuse the old DSV4 tokenizer IDs.

`channel_layout.py` now negotiates binary capacity, source scales/payload and
output allocation. The1024-row build initially exceeded the AIV32KiB stack:
coordinator slot route IDs now live in explicit HBM scratch, and workers read
bounded256-route map tiles. Do not raise the stack limit or enlarge only a
Python buffer. ABI3 and source-row layout must match; ABI2 is valid only at32.
Warm only actual client row buckets before capture, avoiding quadratic banks.
`expanded-ep-gates.json` preserves the passed five-device real layer0 FULL wire
and native EP8 gates. Neither is a complete serving comparison.

For colocated TP2xDP4, the owned runtime facade originally equated TP with WORLD.
`prepare_colocated_overlay.py` makes a separate, explicit TP/DP-aware closure;
never edit the separated runtime in place. Model source tokens are striped once
across the TP pair at EP ingress. Use native MC2 counts/masks and NZ GMM, with
native BF16 down scales for BF16 GMM output. The sampled relative oracle error
is~0.0054, distinct from exact same-program eager/replay checks. All EP ranks
must execute the same layer/phase, including masked idle participants. TP-only
output agreement checks must name the TP subgroup, not default WORLD8.

The full-model gates subsequently pass:141139Z runs TP2xDP4/EP8 with real48+MTP
and exact same-State FULL decode shadows;141508Z runs both separated TP2 sources
with1024 actual prefill rows/source and the same MTP closure. See
`CAPACITY-AND-COLOCATED.md`. Full-model prefill remains eager. The short runner
still is not a multi-turn SWE reactor, and the two gates have unequal workloads;
do not quote their raw timings as the topology comparison. Catalog load inside
an NPU default-device context must explicitly enter CPU scope for safetensors
slices and ND assembly before explicit NZ upload.

## Enter real multi-turn traces

Use `qwen38/SWE-COMPARISON.md` and the prototype `trace_*` files. Do not treat
fixed-window gates as retained-session qualification. K1 State geometry is in
`capacity-k1.json`:14144B/history token/rank and233547804B/request/rank. The model
contract is262144 context tokens, not the old4096 short-gate configuration.
Two separated attention groups versus four colocated groups can reverse the
whole-machine capacity comparison despite lighter separated attention weights.

Two continuation traps were found statically: GDN prefill reads canonical row0,
whereas decode leaves an accepted candidate/convolution slice; and the native
bounded commit caps output count without reselecting pending/multi at that cap.
The scoped trace runner handles both; do not silently change the older numeric
receipts. `probe_trace_cpu.py` checks capped endpoint behavior. Retained hardware
qualification is still separate.

The PLE CPU response encoder's `bytes(tensor_row)` was measured at4.513s for
1024x1024 BF16 output versus5.773ms for byte-exact bulk copying. Use the scoped
`trace_ple.py` bulk codec for this experiment, not a larger device polling bound
as the first repair. It also acknowledges all-inactive waves; the old host
worker rejects empty requests, which cannot support idle native EP participants.

Qwen38's `prepare_qwen38_qsa_island` assumes TP is its parent WORLD. In the
colocated TP2/DP4 lane it instead creates different local pair member lists at
one WORLD8 group-creation position. A larger real trace warmup exposed HCCL
initialization error9. `colocated_model.bootstrap_groups` reuses each pre-created
TP2 coordinator in `_QSA_GROUP_CACHE`; QSA islands and TP pairs are identical
for this layout. Keep the post-catalog rendezvous and avoid a fresh island
communicator. The earlier short gates alone do not certify this startup order.

The first matched retained pilot is in `qwen38/swe-two-turn-result.json`:
4 distinct sessions,2 complete turns each,1408 outputs,26391 prefill rows,
19310 reused-prefix tokens. Separated/colocated maximum source durations are
40.5824/47.6982s; both pass first-page retention and final within-TP output IDs.
This is **not a vLLM result**: the native control currently uses global phase
voting instead of mixed target waves. Its decode stalls during other sources'
prefill. Do not turn the1.1753 pilot ratio into a mature DP+EP or expert-GEMM
claim. P99 is sample-sensitive (max gaps are9.09/10.11s despite very different
P99). No full-trajectory,262K fill,maximum-HBM,or full quality gate is implied.

For maximum-fit/TP1/E3 work, enter `qwen38/TOPOLOGY-CAMPAIGN.md`. Equal8GiB State
budgets do not compare physical topology limits. The49GiB TP2 separated gate
passes with about536MiB minimum sampled free memory;50GiB fails warmup HCCL
allocation. Native34GiB first failed A2 MC2's256-source-row bound, **not OOM**.
Do not confuse staging-slot count2 with source-count2 when expanding the server.
TP1 needs two-head QSA publication/gather/FIA as well as a parallel-plan change;
removing the TP2 guard alone silently leaves half of each KV row unpublished.


C40 TP1 full-prefill fit failures exposed a separate QSA workspace tax:1024
queries x2051 selected KV rows x2 heads gives about2GiB each for K and V,
then the old FIA layout adaptation copies both again. Enter the campaign's
bounded-QSA section before reducing State budgets. The disconnected overlay
now consumes <=128 queries at a time and gathers directly into head-major FIA
storage; Q/indexer/top-k remain whole-wave and exact. The8-case leaf gate covers
one/two heads and chunk boundaries, but does not by itself qualify full-model
capacity or timing. Apply the same overlay to TP2 controls for fair comparison.


The first head-major QSA version (c) passed leaf correctness and long-prefix
memory fits but made short-prefix prefill catastrophically slow. Layer events
isolated12 QSA intervals (~2.94s each) while all MoE took only0.232s. Inactive
padding zero stores must remain affine/contiguous per KV head; vector modulo
address expressions can trigger costly scatter lowering. The d overlay restores
that form. Always include full2051 selected slots with mostly inactive padding
in this gate. A long-prefix capacity gate is not a substitute for short-prefix
performance. Native MC2 stack snapshots during slow warmup were not proof of a
collective deadlock. CPU PLE lookup was also measured separately and could not
explain seconds-long waves; its exact last-position/grouped-shard improvement is
only millisecond scale. Enter the campaign for receipts and current timing status.


Native MC2 Dispatch/Combine1D masks require a true prefix. Arbitrary request
padding or finished-seat holes are illegal even when a short run happens to
complete. `colocated_ep.compact_prefix` adapts fixed chunks with device prefix
sums and inverts after combine; shared stays in original order. Enter the
campaign's MC2 section for the official reference, CPU oracle and latest hardware
qualification. Do not interpret the old native TP1 layer0 dispatch stops as
State OOM, and do not reuse pre-correction masked timings as a legal baseline.

A2 MC2 also differs from A3/A5 in optional TP arguments: follow the pinned
upstream dispatcher, omitting the TP group/count fields on A2. The isolated
real layer0 `probe_native_ep.py` gate now passes masked eager/FULL rows through
1020 after device compaction. Its NZ setup must explicitly enable internal
format before loading weights. See the campaign for the separate full-model
fit and unresolved long-run SIGSEGV receipts; passing this leaf does not certify
SWE completion. For head-major two-head gathers, inspect active stores too:
affine inactive zeroing alone does not remove active vector-scatter lowering.

The observed hw0 long-run SIGSEGV reproduction is inside CPython3.12's timed
`faulthandler_thread` walking frames, not an NPU operator. Do not reintroduce
`dump_traceback_later` into these clients as a routine progress diagnostic.
Use the external deadline and bounded role logs; capture native stacks only
when diagnosing a real failure. The e active per-head gather probe also
confirmed the vector-scatter tax (371ms ->0.915ms at2heads/count1024); retain
full-model and throughput qualification separately.

A TP1-EP8 maximum-State fit can present as `HcclImpl::WaitCommThread` while
native plogs already report device OOM. At33GiB State,211/421MB allocations
failed; Python did not promptly expose a normal allocator exception. Inspect
bounded native error slices rather than extending a communication timeout.
One-off `empty_cache()` after catalog loading did not reduce fragmentation or
repair this fit; do not retain it as a claimed capacity improvement.

For Qwen38 prefill cost or scheduler padding, enter `qwen38/PREFILL-PROFILE.md`.
The C40 E3 trace uses only39.6% of fixed prefill bucket rows; its first two fully
occupied waves also have substantial collect waits. The bounded profiler marks
receipts PROFILE, not throughput PASS. Use same-host provider timestamps for
combined attention lanes; never align independent sources by their first event.
Native source markers provide useful input/bucket bands. Relocated TraceLoom
needs all three rule TSVs, and offline parsing can create an auxiliary analysis.db
alongside the real ascend_pytorch_profiler DB; only ingest the latter.
