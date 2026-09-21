# Qualify and compare Qwen MTP FULL graphs

Enter here for speculative candidate-state lifetime, MTP-count sweeps, mixed
FULL composition and target/draft timelines. This separate entry avoids rereading
the long non-speculative optimization history for every MTP experiment.

The bounded model is Qwen3.8-27B/qwen3_5_text,64layers (48GDN/16FA),hidden5120,
TP2 qk8/v24/K=V128, BF16. Native donors are vLLM752a3a504485790a2e8491cacbb35c137339ad34
and Ascend9bf964cb4b87c8cd0d6852c41a55b3c29711fa95. Remote capsules and copied
runtime-source/model live under hw3 `/workspace/my-ascend-workspace/runs/qwen27-partition-serving`;
Python is `/workspace/my-ascend-workspace/runs/liveinfer-online/20260907-donor-dspark-runtime/env/bin/python`.
Use the probe-npu skill and capsule subset-admission helpers; these paths grant
no right to bypass occupancy or change installed donors. Prototypes are in
`prototypes/qwen38-serving/mtp`; normal production admission remains unchanged.
For changes to the underlying non-speculative runtime, enter the separate
[hybrid-serving scenario](../optimize-qwen-hybrid-serving/GUIDE.md).

## Reference semantics, not an Ascend qualification claim

Read-only LiveInference commit `05ac1541` supplies the existing protocol:

- `src/livemodule/llm/dsv4/device_dspark.py`: target context length counts
  processed target rows (anchor + accepted drafts), not the correction/bonus
  emitted token. DSpark lowering prepares draft context/query masks and advances
  query positions on device without accepted-count D2H.
- `src/livemodule/serve/qwen35/mtp/root.py::_commit_decode_cabin` publishes
  `accepted_count + 1` into state-selection metadata. Output count can be clipped
  by EOS/budget independently; terminal lanes must not continue.
- `src/livemodule/llm/qwen35/gdn.py` retains K+1 state columns even when the
  current query has length one: its initial state may be previous column K.
- `src/livemodule/arch/cuda/llm/qwen35_gdn_state.py` migrates accepted state only
  at an APC anchor boundary, then resets selection to column zero (count one).
  This is not a per-step full-state gather/scatter requirement.

Qwen target GDN updates candidates directly in its physical state pool. Current
BetterScale pool is K-V; native Ascend speculative recurrence is V-K and is not
an interchangeable consumer. Metadata ping-pong banks are unrelated to candidate
state columns. Requests' candidate slot rows must be disjoint until retired.

## Ascend convolution is a different representation of the same prefix

Pinned Ascend `9bf964c`,
`csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h::ProcessUpdateTasks` selects
convolution history at temporal offset `accepted_count_including_anchor - 1`.
For width four / MTP2, five temporal rows retain two old input rows plus up to
three current inputs. `WriteBackStateSpec` performs this update in the request's
single convolution slot. GDN instead writes one matrix-state slot per token.
Do not manufacture three complete convolution state banks to mimic GDN.

## Initial integration boundaries (resolved below unless explicitly retained)

- Owned metadata currently assumes non-speculative CPU lengths; accepting MTP
  cannot simply copy this publication slab over device-owned acceptance feedback.
- Native runner already corrects optimistic computed lengths on device, but also
  waits for hybrid accepted-count D2H and performs APC preprocess on CPU. Preserve
  correctness before attempting to eliminate those dependencies.
- Per-request prefill/verification roles must not be inferred from total tokens.
  Mixed chunk recurrence and candidate recurrence must write disjoint live rows.
- FIA's native shared planner consumes CPU sequence lengths. The pinned runner
  already sets `_needs_seq_lens_cpu_sync` for AscendAttentionBackend and corrects
  optimistic lengths from the preceding side-stream valid-count receipt before
  metadata construction. This is not a new seq_lens D2H. Confirm that corrected
  `_seq_lens_cpu` reaches the owned planner; public `seq_lens_cpu` is still None
  during async speculation. Removing the existing receipt fence is a separate
  protocol change, not required merely to capture MTP.
- MC2 policy currently distinguishes `.decode`; pure multi-token verification
  must not accidentally become prefill merely because its query length is >1.

## Bounded experiment entry

`prototypes/qwen38-serving/mtp/candidate_state_probe.py` uses actual TP-local
qk8/v24/K=V128, width-four convolution, K-V float32 state, two ACL graphs and
changing device metadata. CPU recurrence checks all intermediate candidate
states as well as current outputs and convolution history over successive waves.
It covers zero/partial/full draft acceptance, T=1 after previous acceptance3,
request row permutation, empty lanes, physical slot zero and strided slot rows.
No model weights; not end-to-end serving evidence.

Initial capsule `mtp-candidates-20260920` and diagnostic `-v2` on hw3 found correct
outputs/final candidates but incorrect intermediate candidate states in the
formerly dormant Triton multi-token loop. This is why output-only verification
is insufficient. A speculative-only `tl.debug_barrier` attempt (`-v3`) hit the
Ascend compiler's `BlockPtrAnalysis.cpp:496` unsupported AddPtrOp abort. These
are observations, not a proven root cause. Static unrolling at V tile64 exceeded UB (2631680 bits required vs1572864
available, `-v4`). With MTP2 statically unrolled and V tile32, `-v5` passed both
FULL banks across24 successive waves: max state error8.3819e-9,
max output error3.8147e-6, exact convolution history, exit0 and release checked.
This narrows the dynamic-loop issue but does not prove its compiler root cause.
Single-token decode retains V tile64. Candidate input rows are now explicitly
restricted to width3; wider speculation is not qualified.

`mtp-mixed-20260920` then passed six changing role/length layouts (capacity64,
up to4 active requests, warm/cold prefill, verification widths1/2/3) in two FULL
graphs. Independent CPU recurrence: max output7.6294e-6, max state7.3040e-5
(chunk BF16 arithmetic), exact convolution history. State never leaves the pool.
The first composition uses activation index_select/cat, not state gathering;
`mtp-mixed-20260920-v2` folded these activation copies into preprocessing/output
stores and passed the same six cases with identical reported maxima.
This is not model-level or end-to-end performance qualification.


## Recurrence control, not service speedup

`mtp-recurrence-20260920-v2` compares captured kernels on the same hw3 device5,
BF16 normalized Q/K, float32 pool, actual qk8/v24 geometry. MTP2 width3 owned vs
native means were about26.59/26.99us (C1),50.06/50.59us (C4),79.68/73.01us (C8).
Thus correctness does not imply a recurrence speedup: the C8 kernel costs about
6.7us more per GDN layer. Non-speculative width1 controls passed and retained
lower measured cost. This is a short ABBA operator control, not an end-to-end claim.

A V-tile-parallel grid (`mtp-recurrence-20260920-v3`) passed native comparisons
but worsened width3 to33.13/63.71/104.18us at C1/C4/C8; rejected and reverted.
Do not rerun that launch geometry without a new hypothesis.

The first recurrence control incorrectly asked native to select previous
candidate3 with current query width1. Its kernel explicitly checks accepted <=
current seqLen and skips that task; it is not an oracle for our wider persistent
candidate-row contract. The CPU oracle covers that valid LiveInference case.
No claim here that the native service invokes this unsupported operator ABI.


## Device feedback and progress (bounded)

`mtp-feedback-20260920-v3` captures count remapping, conv/GDN execution, and a
synthetic device verifier/continuation update in the same two alternating FULL
graphs. After bootstrap the host supplies token evidence and request ordering,
not accepted counts. Twenty-four successive waves passed, including inactive
lanes preserving count/cursor/anchor. Device cursor advances by accepted target
context; correction/bonus is stored as the next anchor. This is a protocol
stand-in: service integration must reuse native valid-count output, not add a
second rejection sampler.

The feedback probe's changed acceptance history exposed a one-ULP BF16 SiLU
rounding difference between CPU and native convolution (max3.05176e-5), not a
candidate-state-selection error. The final stagewise oracle checks convolution
separately, then feeds its observed BF16 outputs into independent CPU recurrence;
max state error9.31323e-9, convolution history exact. Do not quote that state error
as whole-model numerical parity. `-v1` lacked inactive-lane suppression and is
superseded; `-v2` incorrectly attributed propagated convolution rounding to the
strict1e-7 state threshold. Preserve this distinction when interpreting receipts.

The ongoing service prototype uses `service_adapter.py` / `service_metadata.py`
in the same prototype directory. It patches existing builder/publication/model
leaf hooks in an isolated capsule, not another Worker, and is not enabled by
product admission. `mtp-service-20260920` is the first TP2/APC/FULL integration
attempt; read its receipt before assuming it started or passed.


### Service bootstrap findings

The first service attempt loaded weights but correctly failed native admission:
MTP2 + APC align selected a1536-token Mamba block, so the diagnostic512-token
scheduler budget was invalid (`block_size (1536) must be <=
max_num_batched_tokens (512)`). Use the normal2048-token budget, not repeated
small-budget model reloads. Current `mtp-service-20260920-v2` uses2048 budget,
4096 context,2GiB diagnostic KV budget and26 banked target captures; these are
correctness settings, not production/performance qualification.

Do not bootstrap this adapter with unrestricted `sitecustomize`: importing vLLM
there adds log text to architecture-inspection subprocess stdout and breaks its
JSON protocol. The isolated v2 capsule bootstraps at the existing
`betterscale.worker` import instead; the tracked public Worker is unchanged.

Before the second model load, `mtp-publication-20260920` passed six mixed cases
using the real publication slab, fences and device accepted-count mapping;
`mtp-publication-20260920-pure` passed four pure-verification cases, including
query1 with previous acceptance3. Both banks are checked against CPU recurrence
and exact convolution-history updates.


### Native MTP + APC postprocess ABI gap

`mtp-service-20260920-v2` completed all26 target FULL captures (500s cold,
2.25GiB reported graph memory) and started HTTP. Its first request failed in
native Mamba postprocess before a terminal response: `dynamic_func() got multiple
values for argument 'block_size'`. The vLLM V1 caller sends18 positional arguments
(DS-row metadata and optional V2 index mapping added), while Ascend installs the
older15-positional SD/V1 kernel. This is not a GDN output/capture failure.

`src/betterscale/patches/qwen_gdn/mamba_abi.py` projects only the qualified SD/V1
call onto that native kernel. It rejects DS layout, nonempty request mapping and
unknown launch keywords, and verifies the pinned legacy kernel signature before
installation. It is not installed by normal product admission yet.
`mtp-apc-abi-20260920` calls the real V1 method through this bridge under FULL
replay: cross-block candidate selection, convolution temporal shift, in-place
accepted-count reset, no-op self-copy and no-boundary cases match independent
CPU pool/count oracles exactly. Two CPU contract tests also cover argument
identity/order and fail-closed rejection of unsupported semantics.

Current model retry is `mtp-service-20260920-v3`, with complete v2 content-addressed
Triton cache seeded to avoid repeating500s of cold operator compilation. It also
uses runner device acceptance feedback when no draft dictionary is passed to the
builder (capture still initializes neutral count1); query1 must not forget a
previous accepted column merely because this step schedules no draft tokens.
Read its final receipt; the v2 service is explicitly FAILED, not qualified.


### First model smoke passed, APC hit remained unproved

`mtp-service-20260920-v3` passed seven HTTP completions, including new requests
inserted during decode, repeated1537-token text equality and clean exit. Target
capture reused Triton cache:37s/26 graphs,2.26GiB. Native MTP counters observed
41 accepted drafts /60 proposed. The first request and first mixed cohort still
paid cold runtime kernel compilation (~11s); these are not timing results.

Crucially, its observed prefix hits were0. The switch was on but the hit path was
NOT qualified. `_mamba_block_aligned_split` reserves a full trailing block in
Eagle/MTP mode: `last_cache_position = floor(num_tokens/block_size)*block_size
- block_size`. With1536-token blocks, even2049-token requests alone do not force
a retained1536 checkpoint. The initial hypothesis that a3073-token primer alone would fix2049-token
replay was insufficient: v4 completed all cohorts but still recorded0hits
(115291queried tokens). Lookup also drops the last complete attention block;
a2049-token lookup has only one1536-token block and therefore returns zero
even with a longer primer. The replay itself must cross two complete blocks.
Native3073-token replay now confirms the lookup boundary: two extra requests
produced3072 aggregate hit tokens (1536 each). The candidate still needs its
corrected long-replay qualification; this is not yet candidate APC acceptance.

The in-flight paired `mtp-service-20260920-v4` / `mtp-native-20260920` capsules
use an explicitly recorded3073-token companion primer. Its
`checkpoint-primer.json` must show `before_benchmarks=true` before accepting
measurements. The tracked harness now primes serially before smoke instead, so
future runs do not require this one-off companion. Both controls use6GiB KV,
TP2/APC/MTP2/async,2048 budget and4096 context. Native gets ONLY the required
SD/V1 ABI bridge, not owned GDN/FIA/MC2 or convolution-weight packing; label that
compatibility fix rather than pretending an unmodified crashing native was the
control. The paired synthetic prompt cohorts are not SWE trace performance.

The native control received an extra long-prefix diagnostic while serving;
its cohort timing is consequently diagnostic, not a clean paired benchmark.

### Capture input contract (v5 admission failure)

Adding a query<=3 assertion caught native general dummy partitioning12 tokens
into eight requests as seven1s plus5. Earlier captures tolerated this invalid
verification input; their operator continuation evidence still used valid rows,
but model captures must be repeated. A uniform dummy attempt (v6) failed because native initializes4 active query
ends into the general FULL descriptor's fixed8-row slice. The final approach
(v7) balances the existing mutable scheduled-token vector in the dummy-only
dispatch hook:12 becomes [2,2,2,2,1,1,1,1]. Total tokens, request count and
graph keys are unchanged. Runtime request schedules are untouched. Two CPU
contracts cover all verification capacities and invalid over-capacity inputs.
Do not remove the query<=3 assertion merely to make capture pass.

## Qualified bounded service result (v7, 2026-09-20)

Candidate `mtp-service-20260920-v7` and control `mtp-native-20260920-v2`
both PASS, server exit0 and lease release verified. Local evidence is under
`runs/qwen-<capsule>/evidence/{receipt.json,server.log}`. Candidate CPU suite:
105 tests PASS (`runs/qwen-mtp-service-20260920-v7/cpu-tests.log`). The final
capture uses legal verification dummy inputs,26 target graphs/38s/2.26GiB.
Native draft remains its own ACLGraphWrapper; this is not one monolithic
captured target+sampler+draft transaction or a replacement DSpark scheduler.

Conditions: same hw3 physical6/7, model/source identities above, TP2/MTP2/APC
align/async,6GiB KV,8 request seats,2048 scheduled-token budget,4096 context.
Primer/replay3073tokens, cohort prompts3073+8*i,64 output tokens, one full
warm cohort and three measured cohorts at each C. Median complete-cohort
output tokens/s (includes prefill, not decode-only):

| C | Native + ABI bridge | Owned mixed FULL | Ratio |
|---|---:|---:|---:|
| 1 |38.78|42.30|1.091x|
| 4 |71.35|80.05|1.122x|
| 8 |93.90|104.93|1.117x|

Both arms:170587 prefix query tokens,82944 hit tokens. Native accepted2073/
2910 proposed drafts; candidate2071/2926. Repeat text equality true in both.
The smoke also inserts new requests after another request begins decoding.
No claim of strict cross-runtime deterministic equivalence, SWE performance,
MTP-vs-no-MTP speedup, or cancellation/preemption qualification. Native's old
receipt field `decode_tokens_per_second` means cohort output throughput; the
tracked harness now uses the accurate `output_tokens_per_second` field.
The earlier native run with an extra diagnostic is not this timing control.

This completes the bounded MTP/full-mixed experiment, not product admission.
The experimental lifecycle bootstrap remains in `prototypes/qwen38-serving/mtp`;
normal packaged Qwen admission/default behavior is unchanged. Product promotion
must replace that bootstrap with normal leaf lifecycle wiring and revalidate
that actual entrypoint; do not silently present the prototype as shipped.

## MTP-count sweep (in progress, 2026-09-20)

`count_policy.py` admits experimental K1..4 (candidate-state widths2..5).
Verification capacity keys3/6/12/24/40 deliberately remain disjoint from mixed
prefill capacities16/32/64/...; report padded descriptor capacities in the
comparison. This is capacity bucketing, not request-partition enumeration.
Pure capture input is clamped to legal GDN widths only during dummy execution;
FIA retains its padded envelope, and runtime request lengths are not clamped.
The pinned runner supplies its separate unpadded `gdn_query_start_loc` to GDN.

Naively expanding the recurrence to4 tokens left one `bishengir-compile`
process at100% CPU for more than14minutes; `mtp-counts-20260920` was explicitly
cancelled, not labeled numerically failed. K1/2 had passed. The queued mixed
probe was also cancelled before launch. Do not silently wait unboundedly or
weaken the state oracle to support larger counts.

The replacement uses at most3 statically expanded tokens per kernel. K3/4
record a second kernel in the SAME graph, reading candidate column2 and writing
columns3/4 directly in the same pool. No state gather/scatter, host acceptance
branch or replay-time host loop. Query<=3 makes the second kernel's rows empty.
`mtp-counts-20260920-v2`: all K1/2/3/4 pass24waves,2banks,device acceptance/cursor/
anchor feedback and full-pool CPU oracles; max state errors7.45e-9/1.12e-8/
1.12e-8/1.12e-8. `mtp-count-mixed-20260920-v2` also passes all four counts' six
mixed and four pure publication/CPU-output/state cases. Both admissions exit0.
These qualify operators/publication, not yet every model-serving count.

Service sweep uses K0..4, native and candidate, same hw3 pair6/7 and prior
3073+8*i prompt /64 output synthetic workload. Each C1/4/8 gets one warm and
three measured cohorts. Profiles are separate: six C4 decode steps after eight
skipped iterations, plus six steps while joining33/97/257-token requests behind
one65-token request. `timeline_probe.py` hooks the existing Worker, not a new
subclass. `profile_counts.py` preserves native raw PROF export and uses exact
FIA-member counts to distinguish target(16) from merged draft(1..4); after-target
interval includes draft/sampling/preparation, NOT pure idle.

Important confound across COUNTS: K0 retains165888 cached tokens in this fixture,
while the previous MTP2 control retained82944. Native Eagle drops an extra1536
block at lookup; same APC flag is not equal cached work. Report this real service
policy effect separately from pure target/draft timeline costs. Native/candidate
within one K must still agree on hits. No SWE or universal optimum claim follows
from the short synthetic workload.

### Count sweep retained results and qualification failure

`runs/qwen-mtp-count-comparison-20260920/{summary.json,README.md,throughput.csv}`
retain the fresh count campaign (not the older v7 numbers). Native K0..4 and
candidate K0..2 passed. Complete-cohort output tokens/s, C1/C4/C8 medians:
native K0 32.38/83.39/141.55, K1 32.03/66.05/91.42,
K2 39.34/71.86/94.48, K3 38.71/72.51/50.51, K4 39.77/70.48/81.64;
candidate K0 35.38/124.72/221.13, K1 34.11/76.89/101.72,
K2 41.07/79.30/104.38. Native K3 C8 is an observed nonmonotonic drop, not a
localized cause. Exact C4 profiled target periods native/candidate:
K0 37.00/34.04ms, K1 53.67/49.07ms, K2 55.77/49.83ms;
native K3 58.31ms and K4 60.36ms. Increasing draft graph cost is near2ms/token.
Six-step profiles are not unprofiled throughput or proof of recoverable idle.
All eight qualified cases have both ranks' decode and mixed TraceLoom Perfetto
exports,32 total, archived in that comparison directory.

Candidate K3 `mtp-count-service-20260920-v2/candidate-k3` failed after C1, entering
C4 warmup: rank1 AICore MTE address error, rank0 later HCCL SDMA failure. Partial
C1 numbers are excluded. K4 candidate was not launched across this unresolved
boundary. `mtp-k3-diagnostic-20260920` added only host metadata journaling and
CANN logging; it reproduced earlier, joining97/33/257 requests behind query4.
Fault PC0x12c0a3ff4208 maps via runtime LaunchKernel registration to
`FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_5000000000010200203`.
This identifies the faulting FIA kernel, NOT its root cause or a GDN failure.
Host accepted counts were1 and columns/slots nonnegative at the last journal.
Read the capsule's `fia-pc-evidence.txt`, receipt, logs and metadata journal;
large CANN logs stay remote. A follow-up `mtp-k3-fia-20260920` records FIA query,
KV and block-table metadata to distinguish padding/planning/lifetime hypotheses.
Read its receipt before inferring an outcome; no numerical fix yet.

Independent `mtp-count-apc-20260920` additionally passed exact matrix/conv/count
CPU oracles for all K1..4 and every acceptance bias, alongside the state and
mixed/publication probes above. Passing these does not qualify failing FIA
service execution. Summarizer explicitly excludes failed/missing receipts rather
than publishing successful partial cohorts from an unqualified configuration.

FIA diagnostic follow-up also FAILED, now at the second short-request verification
(capacity6,q offsets[4,6],KV[38,0],same FIA fault PC). It completed only the long
primer. This is not a reproducible C4-only or mixed-only failure. Capsule
`mtp-fia-padding-20260920` then isolated capacity6, query lengths[1,5]/[4,2],
KV[38,0]/[38,1]/[38,2],2banks,24replays: active outputs exactly matched eager
native FIA, exit0/release verified. Thus zero-KV padding alone did NOT reproduce
the service error; do not claim padding sanitization or GDN changes fix it.
Further work must distinguish service capture/pointer/stream lifetimes or earlier
corruption. Faulting-kernel identification is not root-cause attribution. This
comparison ends with candidate K3 failed/K4 not run; neither is qualified.

## Remove MTP's whole-block APC retreat (bounded, 2026-09-20)

Enter here before attributing the count sweep's C8 loss to decode computation.
The pinned Ascend proposer `AscendSpecDecodeBaseProposer.set_inputs_first_pass`
shifts token IDs left but leaves target hidden states/positions unchanged. Draft
KV at position i depends on hidden[i] AND token[i+1]. An ordinary block hash ending
at B does not cover token[B]; simply removing EAGLE's last-block drop is unsafe
when two requests diverge immediately after B. This is more precise than saying
all MTP cache hits require regenerating target hidden states.

Prototype `apc_protocol.py` hashes the normal block plus a tagged lookahead token,
using native chained hashing and extra keys (including cache salt). Hash publication
waits until that token is known. This slightly stricter identity is applied to all
cache groups, avoiding a separate hidden-state store or mutable shared draft tail.
Identical continuations can reuse the GDN checkpoint at B; differing lookahead
retreats to the preceding checkpoint. This is NOT arbitrary-token GDN rollback.
`apc_boundary.py` disables only this legacy cache retreat and Mamba prefill split
retreat; ordinary speculative scheduling remains enabled. It also explicitly
feeds the known next prompt token to draft at incomplete-prefill chunk boundaries,
instead of relying on native padded proposer backup selection. No new Worker.
A tiny explicit experimental scheduler subclass provides engine-process entry;
this is not installed product admission or a production leaf-lifecycle promotion.

Important effective-source lesson: Ascend replaces the upstream coordinator.
`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py` owns runtime
`AscendHybridKVCacheCoordinator`, with THREE-element attention-group tuples and
`eagle_attn_group_indices`; upstream source's four-field SpecGroup is not the
live ABI. Both read and write flags must be adjusted together. Never reload a
model merely to discover this Python ABI: inspect the installed patch first.
First capsule `mtp-apc-20260920` failed `_replace`; v2 failed four-value unpacking.
V3 restored3072 hits and passed initial repeat/mixed smoke, then harness404:
`/reset_prefix_cache` requires `VLLM_SERVER_DEV_MODE=1`. Bind diagnostic services
only to127.0.0.1. The reset API returns200 without reporting actual reset success;
validate subsequent cold hit0 from the scheduler journal, not status200 alone.

`mtp-apc-20260920-v4/candidate-k2` PASS, server exit0 and admission release verified.
Same hw3 physical6/7, Qwen TP2/MTP2/APC align/6GiB KV/2048 budget/4096 context.
Both1536 and3072 boundaries passed cold/warm generated-text equality and changed
lookahead warm/cold equality. Observed hit sequences (cold, warm, divergent-warm,
divergent-cold) were [0,1536,0,0] and [0,3072,1536,0]. All52 cohort requests then
hit3072. This is bounded model continuation and isolation evidence, not exhaustive
state/logit equality or cancellation/preemption qualification.

One warm cohort +3measured cohorts each C1/4/8, same synthetic3073+8*i inputs,
64outputs, no profiler:58.51/193.36/305.92 output tokens/s. Prior morning MTP2
candidate41.07/79.30/104.38; ratios1.425x/2.438x/2.931x. C8 TTFT median2511.82→269.56ms.
Controls are same host/config/workload but NOT interleaved; no SWE, pure-decode,
or broad production speedup claim. Summaries under local
`runs/qwen-mtp-apc-20260920/result`; frozen v4 receipts/hit journal under
`runs/qwen-mtp-apc-20260920-v4/candidate-k2`.109CPU tests passed, including delayed
hash, divergent lookahead, incremental chain and salt isolation. New harness
entry is `MTP_APC_BOUNDARY=1` (K2 candidate only); `apc_verification.py` performs
cold/branch checks separately from timings. K3 FIA failure remains unresolved.

## APC-fixed C8 scheduling seams (2026-09-21)

`mtp-apc-profile-20260921-v2/candidate-k2` PASS on hw3 physical6/7, server exit0,
release verified. Frozen v4 numerical implementation; six C8 decode steps after
8 skipped iterations, all24tokens/[3]*8/FULL; mixed includes390tokens
[3,33,97,257] atcapacity512/FULL. Both APC boundary/branch checks pass again.
Local `runs/qwen-mtp-apc-profile-20260921-v2/candidate-k2/README.md` and four
TraceLoom exports retain evidence. No fresh throughput/native-control claim.

Rank0 target42.90ms,draft4.29ms,target→draft2.09ms,draft→nexttarget6.05ms;
5 target periods55.34ms. Rank1 agrees (~6.20ms postdraft). Target→draft is
kernel-covered (vocabulary MatMul/allGather/sampling/postprocess), not idle.
Postdraft TASK union~0.41ms. Offline PyTorch CPU parsing (exact CANN timestamps
matched) finds626–666 nested API records/seam on rank0, union2.55–2.83ms.
Per-request CPU tensor construction in `Core.prepare_service` is the source
candidate; persistent NumPy host views, already used by non-MTP publication,
are the next cheap controlled alternative. Not yet method-level causality or
unprofiled recoverable-time evidence. Profiler amplifies small-call overhead.
Six steps retain30 FIA calls/24update pairs, but target wave-FIA bypass remains;
native draft still owns separate updates. ExecuteAsync averages~14.6us, notms.
`gap_ledger.py` separates kernel/memcpy coverage from TASK wait records.

The cancelled preparation v1 exposed lifecycle care: admission's process-group
cleanup does not reach a service launched with its own `start_new_session`.
Explicitly stopped that owned server and verified idle before v2. Future service
harnesses need SIGTERM cleanup of their server group; do not assume admission
exit alone proves release. Capsule preparation and launch must be fail-closed.

## Host packing versus full N+2 ownership (2026-09-21, ongoing)

Fletcher explicitly corrected the optimization target: do not stop at faster CPU
metadata construction; MTP should inherit full N+2 device continuation, aiming
at about1ms exposed handoff rather than accepting the6ms gap. This is a target,
not measured performance. Native target/draft captures currently do NOT supply
that complete protocol. A representative previous C8 rank0 trace enqueues draft
at target-relative9.03ms, then the main host thread waits15.07→44.72ms before
next-target preparation. This wait overlaps target work;29.65ms is not additive
idle. Source `_prepare_inputs` waits accepted-count D2H, then reads/remaps counts
and copies them back; corrected CPU seq_lens and CPU `preprocess_mamba` remain
consumers. Device raw valid count, APC-reset state-selection count and emitted
count must remain distinct. Native V2 fused preprocess/precopy offers prior art,
not an already-qualified replacement for this pinned Ascend V1 ABI.

`host_metadata.py` adds persistent NumPy views for pure/mixed packing.112CPU tests
PASS; local `runs/qwen-mtp-metadata-20260921/check_equivalence.py` compares all
fields with frozen old Torch methods across1200changing cases/K1..4. NPU
`mtp-numpy-publication-20260921-{pure,mixed}` both PASS existing two-bank output,
whole-state and exact convolution-history oracles. These do NOT qualify service.

Control `mtp-metadata-20260921/before` failed an existing3072 cold/warm text check
(hits correct; continuation diverged), before new code ran. A fresh performance-
only control `mtp-metadata-20260921-v2/before` retained text checks as observations,
kept hit checks strict, and passed all text comparisons this time; C1/4/8 output
medians59.03/192.37/296.02. Its first profiled target had a109.54ms startup outlier;
remaining targets~43.05ms. Do not average the startup spike into steady periods.
New `.../after` failed the first3072 cold prefill (1536scheduled tokens) with FIA
MTE out-of-range, PC0x12c0a3ff4208; no valid after performance result. Same PC as
older K3 is only kernel identification, not proven same root cause. Do not
attribute this to array values or declare async lifetimes broken without proof.
`mtp-numpy-fence-20260921` adds a DIAGNOSTIC device drain after each sample call:
all service smoke/boundary checks and C8 warm complete, but server shutdown was
killed at20s (exit-9), devices released. This suggests ordering/lifetime sensitivity,
not a production repair. Harness now retains90s normal shutdown,20s only on
cancellation. Next `mtp-numpy-reset-drain-20260921` drains only before diagnostic
prefix-cache resets, leaving ordinary execution/timings async: FAIL at the same
first3072 cold prefill/FIA fault. Reset-only draining is not sufficient.

FIA prerequisite: `mtp-fia-envelope-20260921` plans4096 once and varies device KV
lengths down across3072/2048/128 boundaries; three cases exactly native, PASS.
`mtp-fia-feedback-20260921-v2` then records actual device length updates INSIDE
both banks, uses three streams/two outstanding waves, async H2D tags and D2H
receipts, and passes48waves (24pureC8 +24mixed390/512) with exact native outputs.
Acceptance is synthetic; not real service N+2 qualification. V1 probe mistakenly
launched FIA on a pre-capture default stream (outside graph); corrected to
`torch.npu.current_stream()` inside capture, not a numerical-kernel change.
Local receipts: `runs/qwen-<capsule>/`. All operator leases released.


### Draft bank resource repair and matched service evidence

The shared dispatcher already gives native draft graphs bank-qualified keys,
but donor `_draft_graph_params` and `_draft_graph_prefill_params` are global,
capacity-only registries. `draft_banks.py` scopes BOTH planes to the same bank
around proposer dummy capture and `_propose`, restoring globals on exit. It
changes leaf resource ownership, not Worker classes or installed donor source.
114CPU tests PASS, including bank independence and exceptional restoration.

Frozen `mtp-banked-draft-20260921` combines this fix with NumPy packing, no
per-step or cache-reset diagnostic drain. Strict1536/3072 APC cold/warm and
branch gates PASS, server exit0, devices released. Compared with the same-day
`mtp-metadata-20260921-v2/before`, C1/4/8 output-token/s medians are
58.97/191.76/309.42 versus59.03/192.37/296.02. C8~4.53%, C1/4 essentially flat;
three sequential repeats, not interleaved, synthetic not SWE, includes prefill.
Do not attribute the whole change to either packing or bank repair alone.

Six decode steps now have12 task-update pairs and18 FIA calls versus24/30;
draft graph itself has two FIA members. This directly confirms removal of the
other bank's updates. Matched rank0 postdraft seam mean8.197→7.071ms, only~0.408ms
kernel/memcpy coverage in the new seam. The older6.05ms profile is NOT this
matched control. New target~43–44ms, draft~4.3ms. CPU API records fell to358–369
per seam versus older626–666, but cross-profile CPU union times are not a clean
matched causal comparison. Four TraceLoom exports live under local
`runs/qwen-mtp-banked-draft-20260921/traceloom/`.

This is a proven resource-indexing defect and successful bounded service repair,
not exhaustive proof that every historical FIA fault (notably K3) has that cause.
Full N+2 service is still unimplemented: corrected CPU sequence lengths, native
APC preprocessing/staging and accepted-count feedback still gate preparation.

### Device-only APC slot/migration prerequisite

`mtp-device-precopy-20260921` tried upstream V2 generic Triton preprocess/precopy;
Ascend compilation aborts in `SmallVector<PtrOffsetInfo::AxisInfo>::operator[]`,
`idx < size()`. Do not repeat that exact compile assuming upstream CUDA coverage
qualifies this donor. `...-v2` instead reuses the pinned SD postprocess copy
primitive with a synthetic boundary encoding to express an exact precopy:
`running=(destination+1)*B-(accepted-1)`. Four disjoint permuted block-table rows,
block16, width3, real conv/GDN pool shapes,24waves cross multiple boundaries;
full pools, conv history, positions, selected columns and reset acceptance are
EXACT against an independent CPU oracle. PASS, exit0, devices released.
This v2 test synchronizes per-wave for observation; it is not async service.
The encoding is a probe technique, not a published production interface.
Raw accepted count and APC-reset state-selection count remain distinct; copying
state must never reset the logical token-progress increment to1. Receipts are
under local `runs/qwen-mtp-device-precopy-20260921-v2/`.

`mtp-device-precopy-20260921-v3` removes v2's per-wave compute synchronize:
two outstanding bank-qualified graph waves, canonical state shared in compute
order, independent egress stream, per-bank captured diagnostic snapshots fenced
until D2H retirement.24waves exact against the same independent full-pool/conv/
position/slot oracle; PASS, exit0 and selected device released. Snapshot copies
are deliberately debug-only and cannot be used as performance evidence. No
acceptance, computed length or selected state column is read by host to author
any following wave. This proves the bounded APC continuation/copy mechanism,
not full model N+2, EOS/generation reuse, preemption or mixed service integration.
Local receipt: `runs/qwen-mtp-device-precopy-20260921-v3/`.

## Real-service device APC integration (2026-09-21, N+2 still in progress)

`device_apc.py` separates device raw progress from APC-reset selection and keeps
stable request seats across batch reorder/temporarily unscheduled requests.
Finished, preempted and resumed IDs invalidate seats, including empty waves.
Fresh requests bootstrap source columns from their admitted prefix progress.
The qualified SD copy primitive handles direct-pool migration, then actual GPU
computed/scheduled positions are staged for postprocess (never synthetic ones).
Postprocess saves reset selection directly to device seats, without accepted-count
D2H. Its native `global_stream()` producer is fenced with a DEVICE event before
subsequent shared block-table/count writes; never replace it with an absent wait.

`device_apc_runner.patch` is a bounded experimental source fork with explicit
caller seams, NOT a promoted product monkey patch or new Worker. It removes the
CPU accepted-count round trip, CPU APC preprocess/staging, and BOTH native
`do_mamba_copy_block` sites. The old donor calls copy inside preprocess and again
just before forward. Deferred CPU bookkeeping stays after model enqueue. This
first stage deliberately RETAINS corrected CPU FIA lengths to isolate APC.
`stage_device_service.py` creates a NEW frozen capsule from the banked-draft seed,
validates original source pins, applies exact no-fuzz hunks, updates only copied
experiment pins, and preflights fixture/modules/removed local-variable references.
No installed donor or normal distribution pins are changed.

`mtp-device-apc-service-20260921-v4` PASS, exit0, hw3 physical6/7 released. Same
TP2/MTP2/APC6GiB/2048budget/4096context/8seats/3073+8*i inputs/64outputs.
Both1536/3072 boundary hit sequences and cold/warm/branch text checks PASS;
17smoke/boundary completions plus all cohorts and short decode/mixed profiles.
C1/4/8 three-repeat output-token/s medians56.40/188.75/303.84; no speedup claim
from removing only APC feedback while FIA still waits. Local frozen evidence:
`/root/my-ascend-workspace/runs/qwen-mtp-device-service-20260921/apc-only/`.

Preparation failures are NOT protocol evidence: v1 rejected the changed runner
source pin before model load; v2 finished startup but lacked prompt.json; v3
hit an unremoved `preprocess_bufs` use at the second copy site on its first
request. All exited/released. The staging helper and pre-start fixture read now
prevent these expensive rediscoveries. Do not bypass source pin checks or
repeatedly load a model to diagnose capsule/Python wiring.

### Removing the remaining FIA length consumer

`device_metadata.py` constructs actual GDN slot addresses from GPU sequence
lengths and current GPU block tables, while host still owns query partition,
roles and resource assignments. The initial version uses ordinary tensor ops;
no claim of metadata-kernel fusion or complete transaction capture.
`draft_fia.py` wraps the native merged runnable OUTSIDE ACLGraphWrapper, reuses
wave-FIA planning/publication for both draft positions and banks, publishes GPU
actual lengths after each host tiling slab, and suppresses obsolete native task
updates. It also reuses caller-owned replay ordering/startup priming. FULL-only
runtime admission fails closed instead of silently using optimistic CPU lengths
in an eager draft path. Native engine async queue depth is already2 (pinned
`VllmConfig.max_concurrent_batches`); the missing piece is device ownership, not
another external scheduler process or an extra Worker class.

Native merged draft's second phase pads request rows to TOKEN capacity (up to
2048) despite max8 live requests. The wave planner admits9 rows. `compact_padding`
preserves8 live rows and folds only all-zero-KV tail queries into one padding
row. `mtp-draft-padding-20260921` validates capacities24/2048 with two banks,
24device-feedback waves each, comparing active outputs against native FIA's
ORIGINAL many-row representation:48waves exact, PASS/exit0/device5 released.
It completed while the TP2 service was still starting, not during its timings.
Local `/root/my-ascend-workspace/runs/qwen-mtp-draft-padding-20260921/`.
Combined trial `mtp-device-length-service-20260921` failed during startup:
native dummy warmups were NONE without draft metadata, so the first FULL call
had no FIA frame. Warm the exact envelope via the raw runnable only in an empty,
disposable startup capture bank; restore context.attn_metadata afterward because
the merged loop mutates it. Never double-execute a live request to create a frame.

`mtp-device-length-service-20260921-v2` completed startup but FAILED1536 cold/warm
text equality (branch equality passed). Exit1 from qualification, server0 and
cards released. Do not dismiss this as nondeterminism: pinned
AscendAttentionMetadataBuilder.build puts the CPU mirror in AscendMetadata.seq_lens.
CommonAttentionMetadata.seq_lens is the actual DEVICE plane. The initial FIA
prototype copied the misleading backend field and republished optimistic lengths.
`bind_device_lengths` now preserves the Common device tensor explicitly as
_mtp_device_seq_lens and rejects CPU sources; existing CPU fields are only the
planning envelope. Padding compaction must also slice the explicit device plane.

The next version also replaces fresh blocking device tensor construction for
APC seat/fresh flags and GDN membership with persistent pinned buffers / banked
metadata slab publication. Preserve native synchronize_input_prep: it protects
pinned-source reuse, not accepted-result feedback. CPU suite123PASS. Combined
`mtp-device-length-service-20260921-v3` PASS:17smoke/boundary completions,
1536/3072 cache hit sequences and cold/warm/branch equality passed, all C1/4/8
cohorts completed, server0, cards released. Same three-repeat whole-cohort
output-token/s medians62.89/203.44/310.79 versus APC-only56.40/188.75/303.84 and
prior banked-draft58.97/191.76/309.42. Sequential bounded synthetic workloads,
not ABBA/SWE evidence. Short decode/mixed profiles captured; gap attribution
still pending. Local receipt:
`/root/my-ascend-workspace/runs/qwen-mtp-device-service-20260921/device-length-v3/`.
Target and merged draft remain separate FULL graphs; sampler/logits and metadata
construction are not claimed as one captured transaction. Product admission
remains unchanged.

TraceLoom's `/workspace/traceloom-continuous-replay/build/release/traceloom` path
is LOCAL, not hw3. A copied binary also needs its three native/data/default_*rules.yaml
files: set TRACELOOM_CLASSIFICATION_RULES, TRACELOOM_SYMBOL_RULES and
TRACELOOM_EVENT_RECONCILIATION_RULES to the copied files. `--version` alone does
not preflight its analysis assets. The frozen experiment stage holds this copy;
installed tooling is unchanged.


### Qualified device-length service seam (2026-09-21)

TraceLoom exact_direct target(16FIA)/merged-draft(2FIA) graphs,6steps per rank;
5post-draft seams. APC-onlyv4 → combined device-lengthv3, same synthetic C8
short-profile harness and hw3TP2. Rank0 mean decode seam8.947→1.042ms,
mixed8.972→1.121ms; rank1 decode8.583→1.036ms, mixed8.745→1.123ms.
This is NOT the native baseline comparison or the older6.05ms profile.
Rank0 target-start period decode59.133→51.116ms, mixed77.414→70.045ms;
target/draft graph duration approximately unchanged. Thus this did not merely
move the original post-draft hole into either graph. Remaining seams have
~1.03/1.11ms kernel/memcpy union and effectively zero uncovered TASK time;
TASK coverage is still not proof every microsecond is essential work.
Native CaptureTaskUpdateBegin/End12pairs→0 in each6-step profile. FIA planning
APIs remain for wave publication; do not claim zero host FIA calls.

The achieved boundary is device-authoritative continuation through the existing
native depth2 queue with separate target/draft FULL replay, no accepted-count
or corrected-length CPU feedback gating the next target. Normal pinned-source
reuse and device dependency events remain necessary. This is NOT a monolithic
capture including sampler/logits/metadata, general K3/K4 qualification, broad
SWE validation, or production release. CPU123tests and strict service boundary
gates passed. Finished NPU processes released6/7. Both ranks' processed Perfetto
files, summaries and gap ledgers are under local apc-only/traceloom and
device-length-v3/traceloom in the artifact root above. Full augmented DBs remain
under each frozen remote capsule's traceloom directory.

## MTP2 whole-session SWE acceptance (September21, PASS)

`mtp/swe_acceptance.py` reuses the existing HTTP request and closed-loop replay
implementation, derives service argv from the qualified frozen receipt, removes
profiling and raises context from4096 to8192 for the ORIGINAL complete fixture.
No trace truncation or resampling. Uses `apc-swe-scaling1/trace.json`:8sessions,
78requests/20,648outputs per C, max prompt7874/combined8120. Every startup warms
complete first turns; every C clears cache and asserts actual reuse plus at least
one zero-hit first turn. It retains request identity, output budget, TTFT, latency,
cache hits and MTP metrics; server termination is process-group-owned and guarded
by finally. `summarize_swe.py` now records/validates MTP count instead of emitting
its formerly hardcoded no-MTP label.

Frozen remote root `mtp-swe-acceptance-20260921`: same hw3 physical6/7 ABBA,
C1/2/4/8 then reversed in repeat1, TP2/BF16/MTP2/APC align/AIV,6GiB KV,
2048scheduled tokens,8seats,8192context,TASK_QUEUE_ENABLE=0 both arms.
Baseline uses `mtp-count-service-20260920-v2/native-k2` (ONLY SD/V1 ABI bridge);
candidate uses `mtp-device-length-service-20260921-v3` (explicit runner source
patch plus all owned leaves and lookahead APC). This compares the entire stack,
not isolated async scheduling. Code/source capsules are reused unchanged;
content-addressed runtime compile caches may grow with the new8192 envelope.
Per-arm admission is bounded5400s with fresh occupancy/release evidence. Retain
comparison.json and four receipts before declaring the ABBA run passed.

Initial baseline/candidate repeat0 both PASS: output tok/s52.21/69.63(C1),
79.49/111.55(C2),104.82/170.65(C4),99.56/140.53(C8). Do not publish these as final
pooled ABBA values. C1/2/4 baseline147456 versus candidate247296 cached prompt
tokens; C8 drops to39936/67584. Both flags on does NOT mean equal cached work.
There were no preemption warning lines in baseline repeat0; cache-hit reduction
alone is not proof of preemption or a localized capacity mechanism. Website
publication is pending final evidence and exact-copy approval.

Fletcher's scope decision (September21): record the C8 hypothesis that requests
were evicted and rescheduling did not restore KV before reuse, but do NOT expand
this acceptance/publication task into eviction/recovery optimization. Observed:
cache hits and throughput drop at C8. Hypothesis: eviction/reschedule/cache restore
or recomputation explains it. Not established by these metrics; baseline repeat0
had no preemption warning lines. Retain the distinction in public methods and
avoid presenting the hypothesis as a diagnosed defect.


Final ABBA acceptance PASS: all4services exit0/admission0,6/7 released after
each;16cohorts ×78requests =1248 timed completions with exact recorded budgets.
Pooled output tok/s native→candidate:51.295→69.608(C1),76.604→111.492(C2),
107.807→170.596(C4),107.422→140.638(C8); gains35.70/45.54/58.24/30.92%.
Mean TTFT ms1076.97→502.31,1365.14→612.69,1453.98→691.34,2374.87→1786.52.
Native C8 repeats99.56/116.63 vary; candidate140.53/140.74. Final pooled native
C8 is nearly flat versusC4, not proof of a stable C8 throughput regression.
C8 cached totals over2repeats84480/136704 vs C1/2/4 totals294912/494592.
Two observations, no confidence/population claim, no MTP-off comparator.
Local frozen receipts, summary, plan and release evidence:
`/root/my-ascend-workspace/runs/qwen-mtp-swe-acceptance-20260921/`.
Website exact-copy review uses branchfeat/qwen-mtp-async-swe and distinct
MTP2 E2E/profile snapshot, leaving current released product admission unchanged.

## Fresh candidate timeline with current TraceLoom main (September21)

`mtp-candidate-profile-20260921` PASS, server/admission0, hw3physical6/7 released.
Reuses frozen device-lengthv3 implementation,8192context like SWE acceptance;
only warmup and6decode/6mixed-window steps, no timed cohorts or baseline rerun.
Decode [3]*8 at24tokens; mixed window [3],[3],[3,33,257,97],[3]*4×3 (only ONE
actual390-token mixed step, padded512). Both ranks retained. Local artifact root:
`/root/my-ascend-workspace/runs/qwen-mtp-candidate-profile-20260921/`.

TraceLoom mainc1328e98d09c95df2b6cce144583a64c4896bf44 fetched/built clean and
checked again before analysis. Reanalyzed full native PROF inputs, not merely
re-exported old AugDBs. This main fixes temporal replay-envelope suppression of
ordinary work between constituent graph launches. Every selected seam MatMul
and AICPU Index now appears exactly once in the primary Perfetto event plane.
Old missing display slices were not evidence of actual device idle.

Largest explicit composition hotspot: `mixed_core.py::restore_kernel` writes
prefill/verify activations back to original token order (NOT state gather/scatter).
The390-token mixed target has48restore_kernel_3 calls, allstream77: rank0sum20.3445ms,
median423.889us/min422.049/max425.108; rank1sum20.3275ms,median423.449us.
Target duration170.596/166.810ms. Source launches(capacity,3),BLOCK1024—1536small
programs atcapacity512—with strided head-major prefill reads. Local source equals
frozen source by direct comparison. Better tiling or producer-side output routing
is a supported next experiment, not an already established20ms speedup. These are
48layer observations within one mixed step perrank, not independent service repeats.

Post-draft decode has3AICPU Index calls/seam: rank0mean0.4134ms,rank1mean0.3918ms
across5seams; rank0seammean1.066ms. Native TASK→CANN connection and offline Torch
op timestamps match aclnnIndex/aten::index. Repeated arange/unsqueeze/add/index/
slice/copy sequence matches `device_metadata.py::publish_slots` multi-dimensional
block-table gather. No Python stacks were collected: code attribution is by
operation signature plus source, not captured Python call-stack evidence.
This is device-side AICPU, not host accepted-count feedback. Slot mapping adds
~0.122ms/seam and synthetic-boundary APC precopy~0.085ms. Fuse addressing only
while preserving device-authoritative counts, bank fences and APC selection.

Target→draft decode~2.05ms is mostly kernel/copy-covered, including vocabulary
MatMul~1.04ms and allGather~0.315ms. Rank1firstdecode target149.237ms is an outlier,
subsequent~43ms; retain/exclude explicitly from steady summaries. Rank1mixedfirst
two post-draft gaps2.803/4.989ms have1.004/1.187ms kernel/copy coverage, versus
rank0gaps0.985/1.199ms. Transition/rank-local skew remains undiagnosed. Do not
flatten that into a universal1ms gap or silently drop anomalies. No eviction/
recovery investigation or numerical code modification was made in this task.

## Whole mixed GDN layout fusion and next-wave addressing (September21)

Before changing only the restore tail, read [mixed-fusion.md](mixed-fusion.md).
It maps the entire inter-projection path, records the layout-only complete-core
and device-slot fusion probes, and distinguishes remaining fusion hypotheses
from measured improvements. The direct state pool and device continuation
protocol are unchanged; real-service qualification is recorded there separately.
