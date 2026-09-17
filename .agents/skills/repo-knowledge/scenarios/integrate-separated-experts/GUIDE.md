# Integrate attention and persistent expert roles

Enter here for the separated-serving prototype, especially moving the existing
Next BF16 server to Qwen3.8 mixed W8A8/BF16. This is distinct from public Worker
patches. Do not change their defaults or upgrade the pinned donor to obtain an
experimental model implementation.

## Owned sources and boundaries

- `prototypes/attention-client/qwen-next/README.md` owns the previously qualified
  80B Next A2/E4 baseline. Its priority subdirectory documents generation-tagged
  shared-completion promotion and bounded decode preference.
- `prototypes/attention-client/qwen38/README.md` owns the newer Qwen4Exp lane and
  compact passed receipts. Do not transfer the older model's full48 or
  fine-grained-prefix performance qualification onto this new ABI.
- Qwen38 Python source comes from owned LiveInfer branch
  `lumi/qwen38-flash-next-serving-plan`,820103bf. It requires TP2 minimum because
  QSA has two KV heads and adjacent-rank islands. The new client is one TP2 group
  with one publishing leader, not two independent TP1 sources.
- The exact IPC helper appeared later than820103bf; `qwen38/ipc_acl.py` is a
  disconnected, attributed copy. The old branch does not contain that import.
- Keep native binary/config ABI together. The Qwen38 client config has17 words,
  target input is INT8 plus FP32 per-token scales, MTP remains BF16, and the
  weight catalog has four pointers per layer. Never point old Next Python at it.

## Paid numeric and loading lessons

The shared Eco-Tech model has target routed W8A8_DYNAMIC, but fused BF16 MTP
experts. Sixty target QSA projections are quantized too. Attention-side loading
must intercept scales/offsets and preserve the existing island weight slices;
removing routed experts alone does not implement this checkpoint.

Custom BF16-to-INT8 input quantization produced a one-integer difference from
native DynamicQuant. Native input quantization is the accepted ingress; the
server's FP32 SwiGLU quantization has its own passed integer/float reference.
The selected INT8 GMM uses installed CANN CATLASS and actual device group ends,
including empty groups and capacity tails. Preserve CANN header order when
formatting; alphabetical include sorting broke this compile once.

Group full expert loading by shard **within each layer**. Per-tensor safe_open
reparses enormous headers and becomes pathological for this222866-tensor model.
Keep only layer-local ND temporaries while retaining the NZ catalog.

**Observed September17:** Eco-Tech stores the three PLE integer hash/index
buffers as BF16. Original shared `Qwen3.8-Flash-Next` has exact int64 values;
text configs match and casting each original buffer to BF16 reproduces the new
buffer exactly. Sampled unquantized router/PLE embedding rows also match.
`qwen38/ple_metadata.py` restores only these exact original buffers, validating
shape/dtype/cast identity. This is explicitly a **repaired-checkpoint lane**,
not proof that the untouched published quantized model behaves identically.
Never round corrupted BF16 multipliers back and treat them as exact hashes.
Original shard/content identity matters; do not synthesize hash constants.

## Run small gates before full checkpoint loading

`qwen38/run_wire.sh` runs one client plus E4 real layer0 math/graph gate, with
per-device admission and owned-child fail-stop. Passed at rows1/4/32 with changed
inputs, max relative L2 about3.27e-6. This is neither language quality nor full
serving throughput. `run_model.sh ... --construct-only` isolates TP2 root loading
before spending six devices on full-layer integration.

The selected wheel/native overlay is recorded in the capsule receipt. Activate
its OPP before torch_npu initialization, then register the extension after device
admission/binding. **Append** to CANN's PYTHONPATH; replacing it erased `tbe` and
failed compiler initialization. No global OPP installation or LD_LIBRARY patch.

Admission-helper snapshots siblings into PYTHONPATH. Copy only the admission
script into an isolated helper directory, and keep the experiment source in its
own capsule. Otherwise unrelated helper modules can shadow the selected ABI.
On errors, the launcher stops only its children; never kill a foreign NPU job.

## Persistent lifetime is not the process timeout

Inspect the actual launch attributes before blaming queues. The reused microbench
`device-service/launch.cpp` explicitly sets per-kernel timeout10,000,000µs. This
remained active despite a process-level1200s setter. Full-root cold work outlived
the resident server; downstream `neural_collect` then timed out. Qwen38 owns a
longer launch wrapper and rejects old-lifetime ABI receipts. The external role
supervisor remains bounded. Do not turn a short successful leaf into a claim of
indefinite persistent service. The full-model gate also exposed K=0 metadata
carrying a clamped selector into an ordinary GDN backend that requires `None`;
the narrow target-only binding adapter preserves K>0 candidate semantics.

For single-card leaves, require a physical-device argument to agree with
`ASCEND_RT_VISIBLE_DEVICES` **before** set_device(0). Admission alone does not
bind a process. A missing binding was caught and the owned run stopped; that
capsule is rejected. Multi-role launchers bind each child explicitly.

The repaired full48 target lane subsequently passed prefill plus three decode
calls,192generations per E4 owner, with diagnostic per-layer sync disabled; see
`qwen38/full-target-result.json`. PLE Conv1d needed leaf-scoped ACLNN dispatch for
capture (`ple-conv-result.json`). Retain the entire graph input frame, not only
three convenient tensor fields, through graph reset. Full graph qualification
is distinct from these preceding gates; consult the current prototype receipt.

Final September17 gate: `qwen38/full-graph-result.json` passes all48 real target
layers on one TP2 attention group plus E4. Same-State eager/decode-graph hidden
relative L2 is0 on both ranks; three changed-input replays preserve the eager
output sequence. Each E4 owner drains240 calls, all six processes exit0.
This excludes full-model MTP, large prefill, independent-source batching,
language quality and throughput gains. Do not repeat the full model merely to
reconfirm those passed contracts; rerun for an affected change or a new risk.

## Two-source extension and admission boundary

Qwen38 `--sources 2` uses two independent TP2 groups plus E4 (eight cards),
not TP4. Each leader discovers all four output windows before exporting its
input to all server PIDs. Servers must send windows before waiting for input
registration; source IDs cannot be inferred from accept order. Shutdown waits
for both EOF generations before releasing either server's source mappings.

The coordinator's `tasks` field is bounded to32 and also sizes its rolling trace.
Do not enlarge it just to retain more trace records. Full-run co-batch count is
`sum(completed_counts) - waves`; sampled layer pairs are only a ring-tail check.
The two-source CPU integration passed syntax and receipt-analysis checks, but
September17 local attempts063838Z and064359Z were interrupted by foreign NPU
occupancy after admission. The latter loaded all E4 target weights and began
four attention ranks' loading; it did not reach model forward qualification.
No dual-source throughput/correctness claim follows. The launcher now handles
SIGTERM through its fail-stop/finally block to preserve role logs.

For the authorized hw0 migration, enter `qwen38/HW0.md`. Models download directly
on hw0; `/model` is read-only, so use `/workspace/betterscale-hw0/models` and the
explicit QWEN38 model/reference/Python environment overrides. Only two original
Qwen shards are needed for the three PLE integer fields. Do not copy or download
the entire original model to restore those buffers. A dummy INT8 FULL-graph gate
passed on hw0; full-model admission waits behind the direct download receipts.


hw0 subsequently passed full48 two-source graph service. A cold four-wave test
observed0 co-batches because independent cold/capture work shifted the sources;
do not use that as evidence that the coordinator cannot batch. A once-only
post-capture rendezvous plus63 replays/source produced689–694 paired waves per
server (calls[3168,3168]); no per-layer barriers were imposed. Same-State errors
were0 and complete output IDs matched single-source controls. Read the qwen38
README/result before quoting scaling: both single-source controls had a400–493ms
pause at wave52, making the raw >2x ratio unsuitable as a clean server gain.
At that point the cause was unproven; subsequent GC evidence is below.
Typical source step medians were42–43ms alone and46ms together. These are6-card
vs8-card, identical-prompt, target-only short windows, not equal-card or online
workload evaluation. Ring-tail records can contain no pairs despite nonzero
full-run pairing; never label a vacuous sample check as observed layer parity.


The pause is now causally localized:120404Z records a379.9ms generation2 GC
inside the425ms wave52, collecting0objects. Deferring cyclic GC only for the
bounded steady window removes it (wave52 becomes42.1ms). Keep default production
GC unchanged. `--observe-pauses` exposes timestamps; `--defer-steady-gc` is an
explicit <=96-step measurement control with collection before and after.
Both controls yield23.24tok/s(one TP2+E4,6cards) vs44.38tok/s(two TP2+E4,8cards),
zero shadow errors, identical token IDs. In this controlled dual run pairing is
only48waves/owner, not the earlier689–694: concurrent-source capacity gain is
not evidence that co-batching alone caused the scaling. See GC result and
comparison receipts. Do not discard the original pauses or report raw >2x.

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
