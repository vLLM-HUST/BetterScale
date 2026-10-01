# Explore PD storage and head-major transfer

## Resume here: accepted decisions and next PD work

Fletcher confirmed on 2026-10-01 that the accumulated experience should prevent
reopening the same numerical inquiry. The chronological evidence below includes
historical failures and earlier blocking judgments; **this section and the
[accepted envelope](#accepted-numerical-envelope-and-non-strict-hccl-assessment)
take precedence over those old judgments.**

- Accept the measured GDN segmentation/continuation envelope for current PD
  development. Do not start another GDN precision rewrite or repeat the same
  full-model probes merely because warm/cold logprobs differ.
- Non-strict HCCL variability is consistent with BF16 summation-order rounding.
  Strict is an explicit repeatability/diagnostic control, not a mandatory
  production accuracy fix. Do not equate a changed greedy token with corruption.
- Exact transfer bytes, immutable pinned generations, single active writer,
  owner affinity and publication-after-ack remain non-negotiable. Numerical
  acceptance does not relax those storage/lifetime contracts.
- Reopen numerical work only for new evidence: changed arithmetic/runtime,
  materially different geometry, transfer/State identity failure, departure
  from the measured envelope or representative task-quality regression.
- Native AsyncLLM/DPLB targeted-owner coordination, retired async export,
  hot eviction and worker-direct dense AND target-checkpoint transport now pass
  the P2/D6 matrix. See the direct-checkpoint/latency evidence below. Preserve import
  quarantine until both TP workers drain; never release on an unknown RPC outcome.
  Strict remains an isolation control, not a production default.
- Fletcher accepts remaining handoff/control latency as a current development
  boundary; do not make eliminating it a new blocker. The shorter DP finish
  cadence is an opt-in experiment, not a production default or throughput claim.
- hw86 is the active host; hw180 is retired and must not be required. Dedicated
  task artifacts live under /workspace/betterscale-pd-runtime, implementation in
  /workspace/BetterScale. Runtime/release pins are unchanged.

## Historical investigation and retained evidence

Initial source-only research, 2026-10-01. BetterScale baseline
852c10663e703f853c81435d6fd89a6c8affdef3; Mooncake reference
0d1a8040faebb7c127c8901840a38c2ff57e80c5, inspected at
/workspace/Mooncake-reference on hw180. No runtime install or device test.

## Fletcher's intended boundary

- TP2 P instances, D pool initially DP8EP8; one active writer per request.
- Dense target KV streams incrementally; target GDN checkpoint at turn end,
  not each decode step. MTP persistence is not required; bootstrap remains open.
- DRAM/SSD cache, not all-session HBM mirrors. Do not replicate a session across
  every P instance. D has a definite attention-group owner.
- Explore per-head token chunks to avoid TP2/TP1 rearrangement. Extra bandwidth
  can be preferable to packing cost. This is a direction, not a measured result.

## Observed seams

BetterScale AttentionState declares [page_tokens, local_kv_heads, head_dim].
patches/qwen_fia/host_metadata.cpp describes paged K/V as
[pages,128,kvheads*256]. TP2 local_kv_heads=1 masks the distinction between
token-major and head-major. TP1 heads=2 does not. A reshape alone cannot change
the interpretation. Kernel reads AND cache writes need qualification before
claiming direct common-layout consumption or unchanged attention performance.

Mooncake source:
- mooncake-store/include/real_client.h: batch_put_from_multi_buffers;
  get_into_ranges with source/destination offsets, registered destination memory;
  snapshot reads fail on expired lease rather than silently requerying.
- mooncake-store/include/replica.h: replica_num=1, preferred_segments,
  hard/soft pins, optional group_ids. Groups describe lifecycle, not atomic
  multi-object publication or attention-group ownership.
- allocation_strategy.h: preferred allocation can fall back to other segments;
  preferred_segments is not a strict owner-placement guarantee.
- python/mooncake/async_store.py wraps synchronous methods in run_in_executor.
  This is host nonblocking, not evidence of device-stream overlap.
- Ascend Direct local_copy_engine.cpp has H2D/D2H batch-copy paths and async
  fallback for unsupported batch copy. Hardware/driver/registration compatibility
  and completion/event semantics remain untested on hw180.
- Store has standalone-service mode and DRAM/SSD support. LICENSE-APACHE.
  Preserve upstream license/notices if borrowing.
- real_client.cpp is 8615 lines and store_py.cpp 3475 at this pin: broad forks
  incur meaningful comprehension cost; inspect only narrow seams initially.

## Current inference and next discriminator

Prefer a BetterScale session/State adapter over Store + Transfer Engine before
forking Store. Use immutable sealed token/head chunks and a turn manifest;
do not make a whole growing request one repeatedly rewritten object. Tail
chunk policy and group eviction still need design. Byte objects do not establish
valid target State or execution ownership.

First determine whether the pinned attention read/write ABI can directly consume
paged head-major data. Then qualify range/multi-buffer transfers, lease/eviction
races and snapshot publication on CPU; accelerator overlap is a separate admitted
probe. Do not silently upgrade BetterScale's pinned CANN/donor for Mooncake.

## First CPU prototype

Read [the layout probe](../../../../../prototypes/pd-kv-layout/README.md) for
the 22 passing byte-layout tests, CPU timing limits and the installed Mooncake
import/shutdown failure. Native Store, NPU overlap and kernel head-major support
remain unqualified; do not repeat model setup before resolving that preflight.

Fletcher subsequently accepts sub-second CPU conversion as a first-cut tradeoff;
do not make zero-reorder head-major kernels a prerequisite for the connector.
The isolated generic Mooncake0.3.13.post1 wheel imports libcuda.so.1 and is not
a CPU-only escape from the installed NPU wheel failure; see the prototype README.

CPU Store blocker subsequently resolved using the mooncake-hust documented
mooncake-transfer-engine-non-cuda0.3.13.post1 in an isolated venv. The real
multi-buffer/range/lease roundtrip and clean shutdown pass; see the prototype
README and store-receipt.json. This does not fix the old NPU package or qualify
PCIe/SSD. Fork HEAD is c992ba75; the tested published wheel is not a fork build.
Configure lease TTL explicitly: upstream tests and installed defaults differ.


## Current handoff boundary (2026-10-01)

Mainline is now DRAM by Fletcher's decision; disk-native ranges are deferred
under https://github.com/vLLM-HUST/mooncake-hust/issues/3 (published in Chinese
with explicit approval). Do not let SSD block PD implementation.

The prototype README now owns session publication/fencing, selected GDN/conv
TP-layout oracles, real Store P-D-P-D receipts, and admitted NPU pinned-staging
results. Read its final sections rather than interpreting the historical initial
blocker above as current status. SQLite and opaque Store checkpoint strings are
NOT production distributed ownership or model resume evidence. Actual target
State CPU binding has a separate test. NPU0 20/80/320 MiB staged roundtrips passed;
no compute overlap or model qualification is established by those timings.

Today's new hw180 machine is explicitly assigned entirely by Fletcher, without
the former host's lease service. This dated authority does not waive future
shared-host admission. Preserve installed runtimes and qualified donor pins.

### MTP discard is a cache-validity problem, not just an omitted payload

Observed at pinned core752a3a504 and Ascend9bf964cb (both submodules now checked
out exactly, no shared runtime changed):
- vllm/model_executor/models/qwen3_5_mtp.py constructs an independent
  full_attention decoder layer and consumes normalized target hidden states
  concatenated with token embeddings.
- vllm_ascend/spec_decode/llm_base_proposer.py: propose -> set_inputs_first_pass
  consumes target hidden states and common attention metadata; the draft pass
  copies common seq_lens and slot_mapping into its own metadata banks.
- Owned apc_boundary.install_draft_boundary corrects the lookahead token at a
  prefill boundary; it does not create absent draft-prefix KV.

Inference: dropping MTP storage remains sound as a design choice, but resuming
with absent prefix KV and unchanged full-context draft lengths is unsafe. Need
an explicit invalid/cold draft state. A target-only fallback is the correctness
reference; a separately bounded suffix-only draft context is a candidate that
requires address/mask/position qualification and acceptance-rate measurement.
Neither path exists in the currently qualified MTP2-only serving entry. Target
KV alone does not contain the historical hidden-state inputs needed to recreate
identical full-prefix draft KV cheaply. Do not claim warm draft equivalence.

### Model gate environment is not yet restored

Current container has CANN9.1.0, torch_npu2.10.0.post4 and differently pinned
installed donors. Qualified BetterScale needs CANN9.0.1 / torch_npu post2 plus
its explicit vLLM/Ascend source and native payload pins. Source submodules alone
are not a built runtime. Fletcher subsequently explicitly chose adapting BetterScale to the new CANN /
torch_npu environment while retaining BOTH original donor source pins. This
authorizes a new qualification track, not guard bypass or reuse of old results. P-only/no-MTP and Qwen35 DP/EP are new qualification
surfaces even after the old environment is restored.

### Pinned upstream connector seams worth borrowing

Read narrow seams rather than forking the 2059-line Ascend hybrid connector:
`upstream/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`.
Its SupportsHMA path handles MambaSpec address groups, a P-side N-1 truncation
for D last-token recomputation, and delayed freeing until transfer completion.
These are useful frontier/lifetime precedents. Its save_kv_layer/wait_for_save
are no-ops and request_finished handoff is P->D, not our bidirectional durable
session manifest/DRAM lifecycle. Owned resident GDN candidates are not ordinary
native Mamba blocks; do not register all candidate storage as the selected state.

`kv_pool/ascend_store/backend/mooncake_backend.py` wraps multi-buffer Store puts
and gets, but `put` logs errors and does not return an acknowledged result to
its caller. Our manifest publication must require positive completion; borrowing
this wrapper unchanged would erase that correctness boundary. Use lower-level
Store results or an explicitly failing wrapper. This is a local integration
assessment, not a claim about the correctness of the upstream system's cache-
miss policy. Our prototype already fails closed on Store put errors.


### Active new-runtime qualification track

Task-owned environment/artifacts: `/workspace/betterscale-pd-runtime` on hw180.
The system installation is untouched. Its venv inherits system torch2.10.0+cpu /
torch_npu2.10.0.post4 and installs transformers5.14.1 locally. Pinned core
752a3a504 built in empty-device mode, VLLM_VERSION_OVERRIDE=0.25.1 (shallow clone
otherwise generated an incorrect development version), then installed locally;
import resolves to this venv and reports0.25.1. Optional Rust frontend is absent;
not needed for the intended Python serving route. Build needed semantic_version
for setuptools-rust even with the Rust frontend optional.

Pinned Ascend9bf964cb native build runs CPU-only, SOC_VERSION=ascend910b2,
MAX_JOBS=16, no dependencies/build isolation upgrade. Exact catlass41bf90da was
cloned locally from the installed source's matching submodule. Build's global
safe.directory writes are redirected using GIT_CONFIG_GLOBAL to a task-owned
file; source donor is not patched. Logs: logs/ascend-build.log. A successful wheel
still requires the owned four-file runtime.patch and source validation.

CANN9.1 FIA CP materialization initially failed on an outer else formatting
change; builder now accepts the two observed spellings. Its alignment header
was renamed fused_alignment.hpp, FLOAT_PER_BLOCK needs an explicit 32-byte/sizeof(float)
constant (the AscendC name is vector-pass-only), and FAInfer gained leading K/V layout template parameters. The adapter
uses the vendor's ColumnMajor/RowMajor defaults. Existing numerical rewrite
assertions remain. Current changed prepare.py is DEVELOPMENT ONLY, not qualified.
Compiled artifact: fia-cp-build-5/libbs_fia_cp.so (logs/fia-cp-build-5.log). Do not refresh
native.json until numerical/graph gates validate the new artifact.

Owned GDN compiled at gdn-build with BS_GDN_OWNED_INIT=ON and
exact catlass, -j2; logs/gdn-build.log. GDN host adapter also compiled after
adding the CANN include path required by torch_npu post4 ACL headers; regular FIA
host library compiled. Model startup and numerical gates remain unqualified.
Qwen35 worker does not use HC patch; HC packaging is not the immediate gate.
Continue from evidence, not by replacing shared installations or relaxing all
version/source checks. Only the runtime versions explicitly adapted should
change; vLLM and vLLM-Ascend source hashes remain authoritative.


New FIA CP gate caught an adaptation bug: substring matching an outer else also
matched the indented inner empty-slice else, placing padding initialization in
the wrong branch. Fixed with line-anchored matching and an explicit nested-else
regression test (8 CP CPU tests pass). No native manifest was changed. Build5
passes edges, C16-short and nonuniform split cases with two-bank16-replay checks;
moderate/large zero-KV padding comparison failed. Current hypothesis is undefined
native padding reference (mismatches observed in padding indices), not proven;
probe now records live vs padding errors and a small output/reference artifact.
Do not treat a passing compile or the first three cases as complete qualification.


The padding hypothesis was confirmed by fia-cp-moderate-diagnostic/failure.json:
live max0.000244, candidate padding0, native padding max3.328125. Gate now checks
live rows against native and independently requires candidate padding0 on both
banks. Full leaf3 then passed edges/short/nonuniform/moderate but failed extreme
replay7 on LIVE rows (max0.1755). Not dismissed as padding. An independent CPU
oracle for short rows was added to distinguish native vs candidate; fresh run
fia-cp-oracle is in progress. Keep CP artifact unqualified, do not update its
native.json. Native-attention diagnostic mode is an existing potential fallback
for progressing model PD work, not an excuse to claim CP compatibility.

Fletcher redirected native FIA bug reporting/investigation to a future fork;
PD proceeds with owned FIA. The independent CPU short-row oracle in the mixed
harness showed candidate error0.000586 vs native0.156696 for request11. However,
standalone native fixed/dynamic controls and metadata-planner-only control all
passed; do NOT claim the isolated root cause is established. The prototype
README lists exact evidence capsules. Mainline now runs the full owned FIA gate
against independent CPU attention (`--reference cpu`), not the unreliable mixed
harness native comparator. Runtime native manifests remain unchanged pending
that result. Pinned native page-lifetime CPU fixture passes in the new core venv.

The independent full CPU FIA gate remains incomplete: five cases passed, but
long16 replay0 atKV3 exceeded the unchanged tolerance in one live element
(error0.00321957). See the CP probe README and
`/workspace/betterscale-pd-runtime/fia-cp-cpu-oracle-2/failure.json`. Do not
confuse the parked native-reference bug with owned FIA qualification, or
promote the rebuilt binary's native pins yet.

Owned TP2 GDN recurrence now passes24 NPU graph waves against independent CPU
recurrence, including accepted-candidate1..3 selection and a wave12 host export /
new-slot restore / continuation. See `prototypes/pd-kv-layout/gdn_resume_probe.py`
and its README. This is actual recurrent target State, but not yet conv/Store
or model handoff; do not generalize the leaf result to whole-model PD.

The same one-layer GDN numerical continuation also passed through two real
DRAM Store clients (`store_smoke.py --gdn-resume`), not just host RAM. Exact
checkpoint bytes are required before restore; all24 post/replay state checks
use an independent CPU recurrence. See `gdn-store-resume/receipt.json` under
the task runtime root. Conv, full model and distributed attention owners remain
unqualified. Pinned Ascend full native build reached3341 object files before
the initial30-minute timeout; no compiler error was emitted. It is continuing
incrementally with a longer bound and verbose log, without changing donor pins.

Owned CP FIA's strict-FP32 outlier is now explained by its documented BF16
exponential rounding before PV: the independently emulated single-tile output
is bit-exact for every request in the failing wave. The source-aware independent
CPU gate passes the complete device-length matrix, without increasing tolerance
or using native FIA as reference. The earlier pure-FP32 failure remains a distinct
precision diagnostic. See the CP README for exact scope and source; serving
qualification and native artifact admission are not implied by this leaf gate.

Both owned CP gates now pass: device-authoritative lengths and exact-host
full/split plan transitions. Pinned donor build recovery: initial timeout left
eight task-owned `kernel_meta_Compressor*/kernel_meta.lock` files with dead PIDs;
the first incremental attempt failed only on these locks. After verifying every
PID absent, locks were renamed to `kernel_meta.lock.timeout-backup`, preserving
artifacts. `ascend-build-resumed.log` is the current incremental build log. Do not
delete a compiler lock without checking its owner or conflate this timeout
cleanup with a source/CANN compilation defect. Model14 safetensors shards are
present (71,903,878,016 bytes), not yet loaded or tensor-validated.

Pinned Ascend build and isolated source-digest validation succeeded. Canonical
conv history and the owned prefill GDN composition both pass independent CPU
oracles including host checkpoint/new-slot continuation; see PD prototype README.
Next full-model qualification uses a separate candidate package with explicit
post4 and exact rebuilt artifact pins. It must not change repository release
qualification merely to permit the experiment; donor source hashes remain fixed.

### Next integrated topology: one host P2 + D6 (Fletcher, 2026-10-01)

Use all eight hw180 cards for the first real PD cooperation experiment: two P
cards and six D cards. Prefer TP2 attention groups (one P group, three D groups)
while retaining a definite D owner per session. This is a same-host functional
step toward the production layout, not a demand to finish cross-host networking
first. EP6 and EP8 should share protocol logic, but kernel compatibility still
needs evidence. Qwen35 has256 experts; pinned core `expert_map_manager.py`
supports uneven EP partitions43/43/43/43/42/42 with EPLB off. EPLB requires even
division. Do not declare EP6 impossible just because256 is not divisible by6,
or assume the Ascend MC2/fused-MoE routes already support that uneven mapping.

The first TP2 model candidate startup passed source/artifact admission but failed
before weights on missing `libatb.so`. ATB9.1 is installed separately at
`/usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh`; sourcing it selects the torch
C++ ABI and `_register_atb_extensions()` then passes. The second startup uses
that environment, loaded weights and is capturing the unchanged FULL graph
catalog. Logs: `model-server.log` (loader failure) and `model-server-atb.log`
(current). Candidate package is isolated under the task runtime root, not a
published distribution. Context8192/State8GiB per rank is a bounded smoke, not
256K capacity qualification.

The first full-model smoke returned32 deterministic greedy tokens, but long
requests exposed legacy FIA serving admission: single-request decode uses a
native FD plan, rejected by the old fixed non-FD guard. Owned CP attention now
uses only native geometry metadata (not native launch identity), supports the
capacity3 target route and selected MTP phases, and explicitly initializes large
draft padding. Both new integration leaf probes pass; see CP README. Full model
continuation is still pending. Current branch source requires the rebuilt
`plan_discard_latest` host ABI; release manifests remain deliberately unchanged.

P2+D6 topology progress: real NPU cards2..7 passed TP2-group reductions and8
unequal-split EP6 HCCL dispatch/combine roundtrips using the pinned core's exact
43/43/43/43/42/42 expert map. This settles the basic six-rank transport/mapping
question, not fused-MoE or model integration. See PD prototype README/receipt.

### Current model boundary (2026-10-01, supersedes pending gates above)

Use `prototypes/pd-kv-layout/README.md` for the reproducer/artifact details.
Full target-only TP2 continuation now passes exact warm/cold/uninterrupted
64-token comparison. Complete target export/drop/import also passes after moving
resident0→1 and permuting FA page ownership, with4475 cached tokens. Real-request
MTP forwarding/proposals are disabled; startup still loads/captures the drafter.
The original speculative warm-continuation mismatch remains unresolved and is
not excused as a near-tie; do not silently re-enable it in PD acceptance.

Native DP3/TP2/EP6 real40-layer retrieval passes after the two uneven-expert map
corrections **and** the normal Ascend non-SP backend marker. An explicit Worker
skips Ascend platform's worker_cls=auto-only `all2all_backend` fixup, otherwise
Qwen chunks each TP input in half while Ascend's non-SP AllGather expects full
replicas. Trace weights match source exactly; the pre-fix first-layer input
halves and post-fix six correct retrievals make this a concrete integration gap.

The isolated `candidate-package-7-ep6-state` additionally passes real owned State
DP3/TP2/EP6 skew/idle-rank warm-versus-cold continuations (cursors4475/267/39).
Its runtime idle participants force eager/no-attention and temporarily suppress
drafter dummy work: no resident-state writes and no extra MTP EP collectives
absent on active peers. Startup/capture remains unchanged. This does not qualify
FULL idle replay, real-request MTP, or production ownership/availability.

Safe untyped native utility RPC does not reconstruct nested tensors; use the
prototype's explicit shape/dtype/bytes envelope rather than enabling insecure
serialization. The target-to-Store adapter uses40 head-major dense streams,
turn-end GDN/conv blobs and the existing local CAS/immutable-manifest protocol.
Full-geometry synthetic DRAM P→D→P and incremental manifests pass; actual-model
P2/D6 Store handoff is the next live gate, not yet a claimed result here.

A first P→D0 actual-model Store import acknowledged both TP workers, but a request
sent to only that offline SyncMPClient stalled because its other EP peers were
not awakened. The experiment controller must explicitly start idle Core waves.
For production, prefer existing native async DP coordination: DPAsyncMPClient
sends FIRST_REQ, DPLBAsyncMPClient accepts request.data_parallel_rank, and the
completion frontend reads X-data-parallel-rank. This is a frontend/coordinator
integration boundary, not evidence against the qualified EP6 numerical route.

Actual-model P2/D6 Store handoff is now PASS:
`/workspace/betterscale-pd-runtime/p2d6-model-store-4/complete.json`. Three fixed
owners each complete P→D→P→D→P; second-turn16-token D continuation equals cold,
with warm frontiers4440/280/60 and cold0. Native utility Futures bridge response
arrival to actual State retirement before export. Twelve synchronous handoffs,
1.41–4.63s including full CPU snapshot/RPC/Store, are correctness evidence only.
NPU→CPU incremental export, compute overlap, production async frontend attachment,
cache eviction/recovery, and removal of startup-only draft memory remain open.

The same full model matrix now passes incremental dense D2H:
`p2d6-model-incremental/complete.json`. Only new target token intervals leave the
NPU; Store reconstructs full history before import. Across12 handoffs dense bytes
are98,877,440 rather than390,103,040 full-snapshot bytes; selected GDN/conv remains
a full turn-end checkpoint. This is still quiescent RPC export, not compute
fallback/overlap or async page-lifetime qualification. Receiver H2D is full.

Model-page two-slot D2H→Store streaming now passes the same P2/D6 matrix:
`p2d6-model-streamed/complete.json`,12 handoffs and all three exact continuations.
Dense bytes bypass model/Core RPC; each TP rank uses two20MiB pinned slots with
D2H-event-before-Store and Store-ack-before-reuse fences. Core remains blocked in
the retired idle export, so this proves transfer-stage pipelining, **not compute
overlap**. Whole2.55–4.16s handoffs still include GDN/conv RPC and full receiver
H2D. The next async boundary needs native page refcounts (block_pool.touch /
free_blocks), not merely retained tensor objects; see the prototype README.

The next retired async-export cut is **unqualified**. Native FA page pins now
outlive explicit hot eviction and release only after both worker drain acks;
CPU failure/refcount tests pass. The actual concurrent-activity gate failed D1's
second warm/cold comparison at280 tokens (`p2d6-model-async-retired`). A control
waiting for copies but preserving pin/eviction/request order passes all owners
(`p2d6-model-async-serialized`). Root cause is not established. The prototype
README names the exact-byte/logprob discriminator; do not silently enable the
async route. Native DPLB public utility calls broadcast to all owners, unlike
its targeted `_call_utility_async` seam—important for later frontend integration.


Hardware migration admission: the restored hw86 task runtime passes the35 CPU
checks, owned FIA CPU-oracle Q-transition replay gate and synthetic exact-byte
NPU→DRAM Store gate on driver26.0.rc1 (same CANN9.1/post4). See the prototype
README for receipts and the model-path override. This is not yet a new-host
full-model or concurrent-async qualification; the old async discrepancy remains
open. Full historical artifacts are preserved locally outside the source repo;
large trace directories are not required in the new runtime bootstrap.


New hw86 evidence supersedes an async-copy-only explanation: blocking, exact-byte
async-oracle and plain retired-async offline matrices each pass12 handoffs, but
switching to native AsyncLLM/DPLB coordination reproduces the short-session token
bifurcation even with blocking copies. A **D6-only control with no P, Store or
checkpoint** repeats the identical281-token input12 times with fresh salts and
cached0: two16-token sequences,9/3 split. One alternate wins by0.25 logprob at
position4; not all alternates are ties. See prototype README's pure-D section and
`hw86-native-cold-only/numerical-summary.json`. This localizes the acceptance
problem beyond a required PD transfer trigger, not to a proven component. Do not
promote numerical qualification or silently relax exact-token gates; preserve the
byte/lifetime and model-numerical evidence separately. The native frontend actor
uses pinned targeted utilities (never DPLB broadcast) and native FIRST_REQ;
its routing works but numerical acceptance is not stable.

## Cold-request numerical repeatability investigation (hw86, 2026-10-01)

Keep this distinct from the parked native-FIA reference issue: these probes use
our owned FIA. User requested black/white-box verification, not a relaxed token
gate. Experimental diagnostic sources are in `prototypes/pd-kv-layout/`.

**Black box:** `hw86-cold-fixed-seat` repeats the same281-token prompt12 times,
D6 only, no P/Store/checkpoint, cached0 and fresh salts. Diagnostic Scheduler
forces seat0 using the ordinary cold victim path; retirement Future records the
actual completed resident. All12 use seat0/FA page1 (epochs1,3,...23). Two output
sequences remain (11/1). Logprobs also vary after the first request. Neither a
changing seat/page nor PD transfer is necessary for the observed bifurcation.

**White box:** `hw86-whitebox-python-graph` and
`hw86-whitebox-python-graph-trace` capture layers0/1/3/7/15/39 on EP ranks2/3.
FULL ACL graphs remain enabled, but every rank sets ForwardContext.skip_compiled
during the wrapper call so original Python forward/hooks participate in capture.
Device-copy taps, a post-prefill synchronization and CPU snapshots perturb timing.
This is an explicitly instrumented arm, not unchanged compiled baseline evidence.
The281 live input IDs/positions are identical across12 prefills; output still has
two sequences. Compared with repeat0 on both TP ranks (22 comparisons):
layer0 residual is exact in all22; layer0 hidden/MLP output differs in all22,
max abs0.000244140625. Layer1/3/7/15/39 hidden maxima rise to
0.01123046875/0.0205078125/0.04150390625/0.547119140625/8.21875.
Layer0 pair0→1 has1021 differing elements out of575488, max0.0001220703125.
Both TP ranks have identical comparison statistics. These observations locate
the first *sampled* divergence within post-attention norm/MLP; they do not yet
identify a kernel or establish acceptable numerical error.

Reproduction uses `BETTERSCALE_MODEL_PATH=/workspace/models/Qwen3.5-35B-A3B`,
`BETTERSCALE_NUMERICAL_TRACE=<external trace dir>`,
`run_native_cold_probe.sh <new output> --residency fixed-seat --repeats 12
--prompt-receipt <hw86-p2d6-native-async/pd-session-1-cold2.json>`.
Use `analyze_numerical_trace.py <trace dir>` with CPU torch.
`BETTERSCALE_NUMERICAL_MOE_TRACE=1` adds first-MoE taps (experimental, pending
qualification at this note's creation); gathered active DP1 rows are512:793.

Observer lessons: target names include `language_model.model.layers.N`.
Compiled forward did not execute ordinary Python hooks even with compile cache
disabled; sentinel failures in `hw86-whitebox-fixed*` are observer failures,
not model-error evidence. Do not repeat those expensive startup arms. The
successful capsule contains the exact source snapshot and protocol metadata.
No change here qualifies PD production numerics or identifies HCCL as the cause.


First-MoE taps now succeeded (`hw86-whitebox-moe{,-trace,-source}`):
on EP2/3, MLP input, gathered active input, router logits, local routed output and
shared output are exact across all22 repeat comparisons. Final routed output
(after DP reduce-scatter) differs in all22, max0.000244140625. Other four ranks'
local outputs were not tapped; this alone would not isolate the collective.

The independent `reduce_scatter_repeat_probe.py` closes that leaf uncertainty:
six devices, parity DP groups[0,2,4]/[1,3,5], fixed seeded BF16 input1536×2048,
output512×2048,32 eager and32 FULL-graph calls/rank. Under AIV/default HCCL
determinism,219/384 outputs differ from their mode's first call, max0.00048828125.
Setting **only HCCL_DETERMINISTIC=strict** in a fresh process yields0/384 differing
outputs (`hw86-reduce-scatter-{repeat,strict}`). Inputs remain exact. Saved final
outputs fit an explicit CPU BF16 three-input summation order exactly; strict uses
(0+2)+1 across allsix ranks. Its max difference from FP32-sum-then-BF16 remains
0.00048828125: determinism is not a claim of FP32 accumulation or better accuracy.
The model-level strict control is a separate gate, pending at this note's creation.

Exact-runtime donor `vllm_ascend/batch_invariant.py:override_envs_for_invariance`
also selects strict HCCL, but the diagnostic does NOT enable its other numerical
rewrites. [HCCL's official environment reference](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclenvref_07_0010.html)
documents determinism priority over AIV, including possible algorithm changes.
Keep strict as an explicit experimental setting, not a hidden production default;
performance and the complete PD matrix need separate evidence.


Model-level strict control now passes: `hw86-cold-fixed-strict` uses the original
compiled/FULL route (no white-box trace or skip_compiled), same fixed seat0 and
281-token prompt,12 fresh cold requests. All16 generated tokens and every returned
top5 logprob are exactly identical. This is strong causal evidence for reduction
nondeterminism in the original cold-repeat failure, not an accuracy waiver.

`hw86-native-pd-strict` uses native AsyncLLM, strict HCCL, blocking incremental
streamed DRAM handoffs and two extra cold controls per owner. All12 P→D→P
handoffs and allthree16-token warm/cold comparisons pass. Both cold controls have
exact tokens and top5 logprobs for every owner. However warm/cold logprobs are
**not exact**: maximum common-candidate absolute deltas for D0/1/2 are
0.6871938705/0.8673906326/0.4368438721, with5/1/5 changed top5 candidate sets.
All generated prefixes match so these comparisons are like-for-like. The fixed
warm/cold difference is separate from repeated-cold nondeterminism; do not call
numerical acceptance complete just because the sampled argmax stays the same.
A same-owner/no-transfer split-prefill control is the next discriminator.
No production default or release pin was changed; no strict async-export
compute-overlap qualification is implied by this blocking handoff result.


The same-owner split-prefill control `hw86-native-warm-local-strict` reproduces
a fixed warm/cold difference **without any transfer**. D1 prefills the first280
tokens, then continues the same281-token prompt from cached280; cold recomputes
all281. Three pairs all return the same16 tokens. Each path is internally exact
across repeats, but warm/cold top5 logprob max delta is0.3743658066 each time.
This demonstrates a local continuation-versus-prefill numerical difference in
addition to the now-controlled reduction nondeterminism. It does not explain
all of the larger P/D-path difference or prove either path is acceptably accurate.


White-box continuation follow-up:
`hw86-whitebox-warm` and `hw86-whitebox-gdn` preserve strict HCCL and use the
explicit Python-forward/FULL observer arm. The matched warm first token has
**capacity16, prefill with initial State**, whereas cold has capacity512; this
comparison is not simply recurrent-decode versus prefill. Later generated tokens
use capacity3. Actual input ID and position match. Layer0 post-attention residual
already differs. Embedding/input norm/gate and BA projection are exact across
six rank/repeat pairs; QKVZ has only a tiny4.70e-38 difference, while GDN core
output reaches0.00390625 max difference. Do not attribute the larger discrepancy
to the subnormal projection difference without following the inputs.

`hw86-whitebox-recurrence` captures full first-layer preprocessed q/k/v/g/beta
for the280-token prefix, the1-token continuation and the281-token cold prefill.
These five arrays concatenate **exactly** to the cold arrays, for both TP ranks
and both repeats. An independent CPU gated-delta recurrence (formula reused
from `gdn_resume_probe.py`, with FP64 and FP32 controls) therefore yields exactly
the same warm/cold result. CPU FP32 versus FP64 maximum error is6.17e-7.
NPU GDN output versus FP64 has whole-prefix relative L2 error0.181–0.192%;
last-token relative L2 is0.071–0.178%, depending on rank/path. The NPU warm/cold
last-token difference is0.001953125 on rank2 and0.00390625 on rank3.
These are bounded real-model-layer measurements, **not accepted tolerances**.
They locate a chunk/initial-State continuation numerical seam after preprocessing.
An explicit State-before/after tap is still needed to separate State carry from
chunk arithmetic; do not label this a proven copy bug or harmless rounding.

`analyze_gdn_recurrence.py` runs CPU-only against captured activations. Its
independent recurrence has hand-derived scalar-embedding and zero-beta tests;
it does not validate convolution or whole-model accuracy. Keep these capsules
outside Git; source/protocol and compact findings belong here.


Final State discriminator in this investigation:
`hw86-whitebox-state` records the selected first-layer State before/after each
prefill core call. The prefix's published State is **bitwise identical** to the
continuation's input State on both ranks, both repeats. Preprocessed inputs still
concatenate exactly to cold input. Thus this observed seam is not a local State
carry corruption. The independent reference finds prefix-State relative L2 error
0.2066%/0.2373% (rank2/3). Even when seeded with the *actual* carried State, the
one-token chunk update differs from FP64 by0.1806%/0.1662% relative L2 in final
State (max abs0.05649/0.02810); warm/cold final-State difference is
0.2044%/0.1720%. This separates carry identity from arithmetic error without
claiming that a particular fused operator has been identified or setting a
whole-model acceptance tolerance.

**Disposition updated by Fletcher, 2026-10-01:** the measured fixed GDN
continuation/cold difference is accepted for the current PD prototype and no
longer blocks PD development. Preserve its measured envelope below; do not turn
it into an unlimited tolerance or a new kernel-rewrite task. Exact activation,
State-carry and transfer-byte identities remain separate contracts. Strict HCCL
is a repeatability control, not a demonstrated accuracy requirement for
production. Broader service/production qualification is not implied.

To reproduce the final trace, combine the native warm-prefix control with
`HCCL_DETERMINISTIC=strict`, `BETTERSCALE_NUMERICAL_TRACE=<external dir>`,
`BETTERSCALE_NUMERICAL_LAST_TOKEN_TRACE=1`,
`BETTERSCALE_NUMERICAL_GDN_TRACE=1`, and
`BETTERSCALE_NUMERICAL_RECURRENCE_TRACE=1`. Use owner1, the same281-token prompt,
natural residency and2 repeats. The recurrence trace keeps full real-token GDN
inputs (not only last-token taps); `analyze_gdn_recurrence.py <trace dir>` checks
them without NPU access. The large State buffers are diagnostic-only, not a serving
checkpoint protocol. FULL graph observation still explicitly bypasses compiled
Python and perturbs timing.


### Accepted numerical envelope and non-strict HCCL assessment

Fletcher explicitly accepts the currently observed GDN segmentation difference:
we cannot eliminate it within the present work and should record it rather than
block PD progress. This is a bounded engineering acceptance, not proof that every
future shape, context length or kernel change is safe.

Accepted observation anchors (BF16 Qwen35, TP2, hw86, layer0,280+1 versus281):
- Whole-prefix GDN output relative L2 versus FP64:0.181–0.192%.
- Prefix recurrent State relative L2 versus FP64:0.2066–0.2373%.
- Warm/cold last-token GDN output max absolute difference:0.001953125–0.00390625.
- Warm/cold final-State relative L2:0.1720–0.2044%.
- Local no-transfer warm/cold common-candidate logprob max difference:0.37437;
  the broader P/D fixture reaches0.86739. These downstream observations are not
  universal logprob tolerances; the measured fixture's output tokens agree.

Retain exact State-carry/transport-byte checks. For comparable numerical
regressions, report movement from this envelope; do not blindly reuse an
elementwise atol for differently scaled tensors. Historical statements above
that call this an unaccepted blocker are superseded by this explicit decision.

Non-strict HCCL: the saved final output on each of six leaf-probe ranks is
bitwise equal to a legal three-input BF16 addition order. No element lies outside
that order envelope. A follow-up CPU FP64-sum comparison of those saved outputs
(not every one of384 transient replays) gives:
- default/non-strict relative L2:0.23189–0.23277%;
- strict relative L2:0.23164–0.23277%;
- both arms max absolute error versus FP64:0.00042724609375;
- all inspected elements satisfy the standard two-addition bound
  abs(error) <= gamma2 * sum(abs(inputs)), u=1/256,
  gamma2=2*u/(1-2*u). This is an absolute conditioning-aware bound, not a claim
  of a fixed small ULP distance near cancellation.

The0.00048828125 reported earlier is the *difference between repeated NPU
outputs*, not the error against FP64. Strict fixes the reduction order; the
measured accuracy is essentially unchanged. Evidence supports ordinary BF16
rounding-order variability, not a demonstrated numerical failure requiring
strict. Model amplification and greedy-token bifurcation are real but do not,
by themselves, establish degraded task quality. Use strict for deterministic
diagnosis/token regression isolation; do not silently make it a production
accuracy mandate. Non-strict end-to-end quality and strict's performance cost
have not been measured in a representative workload.
CPU assessment receipt: runtime root hccl-rounding-assessment.json, copied into
the local migration backup. No NPU rerun or runtime change was needed.


### Native async export and import failure ownership (2026-10-01)

`hw86-native-async-export-oracle` passes12 handoffs/24 exact-byte TP shard
oracles/allthree16-token comparisons through native AsyncLLM/DPLB. Source hot
residents are evicted while their exported native pages stay pinned; another
request runs before export finish.15/24 host transfer-job intervals intersect the
host request intervals. This is host overlap evidence, not measured device
compute/DMA overlap or PCIe saturation. Strict HCCL isolates the transport gate;
it is not a new production default.

Source inspection exposed an independent destination-lifetime defect: restore
freed its pages on any RPC error without proof that worker H2D had drained.
The prototype now binds imports to a unique ticket, validates rank/seat/epoch
and both worker drain acknowledgements, and quarantines the resident/native pages
if completion is unknown. An explicit matching abort can drain and release;
a second import cannot bypass that quarantine. A reported copy failure with both
drain acks releases safely but never publishes the failed resident. Worker copy
exceptions return as data after the device fence rather than bypassing it.

`hw86-native-import-recovery` injects a rank1 failure after actual H2D enqueue,
confirms drain-before-release and no failed publication, then retries successfully
and completes the entire12-handoff/24-byte-oracle native async matrix (exit0).
The affected CPU suite passes55 tests, including missing/duplicate/stale/not-drained
acks, lost-RPC quarantine, explicit abort, clean retry and copy/drain failures.
RPC connection loss itself remains a CPU fault-injection qualification; the NPU
fault is after-enqueue copy failure. No process-death/HA claim is implied.

At that checkpoint the receiver still staged complete dense bytes through the
controller/Core RPC. The worker-direct ingress below supersedes that data path;
full H2D remains necessary for a fresh GPU resident. Large capsules stay outside Git.


### Worker-direct Store ingress (2026-10-01)

The opt-in native actor path now passes only the immutable dense chunk descriptor
through Core RPC. Each TP worker reads its 20 local head planes directly from
Mooncake into two registered 20 MiB pinned slots, then scatters H2D into its
reserved native pages. A slot cannot be refilled before its prior H2D event.
Chunk boundaries may cross physical pages; Store history remains incremental.
GDN/conv remain complete turn-end checkpoints through RPC; MTP is not transferred.

Import owns the Store client, registrations, pinned buffers, stream and page
reservation through the final drain acknowledgement. Cleanup failures retain
these resources for explicit retry; missing drain acknowledgements quarantine
the seat/pages. Manifest validation is not a lease on dense objects: eviction
between planning and read must fail without publishing a partial resident.

The real CPU Store SDK check validates multi-pointer get_into_ranges reads from
one registered buffer. The affected CPU suite passes 64 tests, including plan
validation, missing objects, stale tickets and cleanup failure/retry.
Hardware evidence: hw86-native-stream-ingress, target-only Qwen35 BF16,
P TP2 + D DP3/TP2/EP6, owned FIA, strict HCCL, native AsyncLLM, async retired
export, streamed ingress, transfer oracle and post-enqueue failure injection.
It passes 12 handoffs, 24 export shard byte oracles, 24 receiver byte oracles,
three exact 16-token comparisons and drain-before-release/no-failed-publication/
clean-retry assertions. Source capsule: hw86-native-stream-ingress-source.

The matrix exports 98,877,440 dense bytes and imports 390,103,040: **receiver H2D
is full-history for each fresh resident**, not incremental H2D. This is correct
for the tested eviction policy, not a promise of optimal reuse. Whole handoffs
span 3.568–8.079 s with diagnostics and fault injection; these are not PCIe
bandwidth measurements. No device compute/DMA overlap or saturation claim.

Remaining production boundaries: one-host DRAM placement, SQLite prototype
coordination, per-import client setup, GDN/conv RPC staging, and HTTP/scheduler
integration. Do not claim physical per-side single-copy LRU placement, HA or
representative workload performance from this matrix. The actor remains
fail-fast on ordinary unexpected utility exceptions; the injected retry is a
bounded Core recovery probe, not general service recovery.

The separate hw86-native-stream-ingress-plain arm (same source capsule, no byte
oracle or injected failure) also passes all 12 handoffs and three token comparisons.
Receiver pipeline intervals are 0.041–0.157 s; complete handoffs are 3.515–8.073 s.
These intervals include per-import setup and are not an isolated PCIe benchmark.
The oracle therefore is not required to make this bounded matrix pass. A Python
resource-tracker shutdown warning reports four shared-memory objects; after exit,
/dev/shm is empty and all eight NPUs and task Store processes are released.
Next measure whole-handoff phase costs before choosing an optimization: receiver
copy time alone does not explain multi-second checkpoint/RPC/control latency.


### Handoff cost discrimination and connection lifetime (2026-10-01)

Do not interpret the earlier whole-handoff timer as pure PD latency: the harness
deliberately evicts the source hot resident and runs an unrelated eight-token
request before joining the export. In the plain matrix that injected request
takes 1.17–1.25 s on P and 2.89–3.20 s on D. Even subtraction is not a no-activity
benchmark because the injected work overlaps the export.

The unchanged-path phase run hw86-native-stream-ingress-phases passes all 12
handoffs. Median seconds (P-source / D-source):
- export begin 0.025 / 0.597;
- hot drop 0.001 / 1.322;
- injected request 1.167 / 2.963;
- finish-export RPC 0.317 / 0.686;
- publish 0.129 / 0.274;
- restore plan 0.095 / 0.067;
- import RPC (to D / to P) 1.791 / 1.582.
Full receipts/source capsule are under the task runtime, not Git. This locates
cost; it does not prove each phase's internal cause.

A bounded CPU SDK probe store-control-cost isolates three setup/size-query/close
cycles: setup 14.5–21.1 ms, two size queries 0.22–0.25 ms, close 1.006–1.031 s.
get_size returns the exact object byte count and -704 for a missing object in
this DRAM fixture. A separate store-close-gil experiment has a 5 ms Python ticker:
close lasts 0.978 s, max ticker gap 0.983 s, only one ticker sample inside close.
That is strong evidence of Python-thread starvation during close, consistent
with GIL retention; it is not a source-level proof or universal runtime claim.
Do not blame all D control latency on DP synchronization before removing this
worker-side teardown stall. The pinned donor also checks global DP completion
only every 32 steps, but no DP protocol change has been adopted.

Source inspection also finds Turn construction reading the entire prior dense
history and GDN/conv checkpoint merely to validate the base manifest. New code
uses manifest validation plus get_size dependency checks: missing/truncated
objects still fail, historical payloads are not materialized. This remains a
point-in-time check, not a pin/lease; receiver reads still fail closed if Store
evicts a dependency afterwards. Metadata work still scales with manifest size.

Worker-owned connections now outlive individual transfers. Endpoint-specific
busy entries retain DMA operands until drain and unregister; an unreleased
transfer cannot be replaced or closed. Normal actor stop closes clients through
targeted owner utilities after all jobs finish. Per-transfer pinned-buffer
registration and exact-byte/failure gates remain intact. CPU tests cover reuse,
busy-close rejection, setup failure, close retry, and metadata-only publication
with missing/truncated base objects (76 affected tests, followed by three more
malformed-manifest cases; the final session subset passes all 24 tests).


hw86-native-persistent-store-oracle passes 12 handoffs, 48 byte oracles and
post-H2D failure/drain/retry. hw86-native-persistent-store-plain passes the same
handoff/token matrix without diagnostics. Both exit 0; no actor failure files,
all NPUs/task Store processes released, /dev/shm empty after shutdown.

Compared with the phase baseline, plain median seconds:
- P-source: import RPC to D 1.791 -> 0.855; injected request 1.167 -> 0.151;
  entire instrumented handoff 3.515 -> 1.615.
- D-source: hot-drop 1.322 -> 0.293; import RPC to P 1.582 -> 0.599;
  injected request remains 2.963 -> 2.927; entire handoff 7.514 -> 5.302.
- Subtracting the injected interval gives 2.356 -> 1.466 (P-source) and
  4.551 -> 2.375 (D-source), but this is NOT a separate no-activity measurement.
The large P injected-request improvement is consistent with removing background
client-close Python starvation. D wave/execution overhead remains unresolved.
These are one bounded before/after matrix, not distributions or throughput SLAs.
Comparison receipt: persistent-store-comparison.json. Oracle/plain source
capsules differ only by a later malformed-manifest validation guard/tests;
ordinary valid handoff behavior is unchanged.

Next remove target checkpoint payloads from control RPC. GDN/conv still take
approximately 64 MB per handoff through worker/Core/controller/Core/worker,
including serialization and TP broadcast; dense ingress alone does not remove
that data-plane detour. No direct-checkpoint implementation or qualification is
claimed yet.


### Worker-direct target checkpoint and true handoff timer (2026-10-01)

The opt-in --direct-checkpoint path keeps GDN/conv payloads off controller/Core
RPC too. Source workers snapshot the retired target State before hot eviction,
then serialize/upload one approximately 32.2 MB shard per TP rank as part of the
background export. Only acknowledged descriptors return through control RPC.
Manifest schema2/format tp2-target-shards-v1 names both immutable checkpoint
shards; schema1 remains the legacy GDN/conv format. Destination workers fetch
and validate identity/cursor/rank/census/geometry before installing their local
target State. Dense transfer, fencing and publication-after-both-drains stay
unchanged. Checkpoint blobs still use CPU msgpack/byte copies locally: this is
not a zero-copy or pinned checkpoint pipeline claim.

CPU tests pass 87 cases. The real DRAM SDK checkpoint roundtrip passes with a
32,198,674-byte synthetic target shard. hw86-native-direct-checkpoint-oracle
passes 12 handoffs, three exact 16-token comparisons, and 96 shard byte oracles:
24 each for dense Store export, dense H2D, checkpoint Store export and checkpoint
H2D. Post-H2D rank1 failure still drains before release, never publishes the
failed resident and retries successfully. No actor failures or leaked NPU/Store
processes remain after exit.

The same-source hw86-native-direct-checkpoint-latency arm removes byte oracles,
fault injection AND the unrelated export-activity request (--no-export-activity).
It preserves hot eviction/real handoffs and passes all 12 handoffs/token gates.
This is a real no-activity measurement, unlike subtracting the old injected
interval. Six handoffs per direction, context frontiers 32–4456:
- P->D: median 0.727 s, range 0.646–1.624 s.
- D->P: median 1.462 s, range 1.363–1.597 s.
- Manifest publication medians 7.5 / 15.3 ms; restore-plan medians 0.7 / 0.7 ms.
- Destination import RPC medians 0.591 s (D) / 0.109 s (P).
- D source export begin/drop/finish medians 0.629 / 0.411 / 0.285 s.

D-side control scheduling is now a visible cost beyond data transfer. The
pinned donor checks global DP finish only every 32 steps and executes dummy
batches while its wave remains active. This is a candidate explanation for
small-utility delays, not yet a causal qualification or permission to alter
production cadence. A shorter cadence must be applied consistently to every
D core and preserve pause-consensus semantics; qualify it separately and keep
the donor/default behavior unchanged unless deliberately adopted.

All evidence/source capsules live under the task runtime; direct-checkpoint
oracle and latency arms share the oracle source capsule. Neither test qualifies
cross-host physical placement, busy-engine request admission, HA, representative
throughput or a production latency SLA. Receiver dense H2D is still full-history
for a fresh GPU resident; MTP is still not transferred.


### DP finish cadence: bounded experiment and accepted remaining latency

Fletcher accepts the remaining RPC/control delay on 2026-10-01 and does not want
PD development blocked on removing it. His event-driven control-thread idea is
an exploration, not authorization to invent a second DP coordinator or mutate
State/page ownership from an IO thread.

The already-running hw86-native-direct-checkpoint-sync1 arm completes after that
steering: same direct-checkpoint/no-activity matrix, --dp-finish-sync 1, strict
HCCL, default model/runtime pins unchanged. All 12 handoffs and three token
comparisons pass. Source receipts prove D0/D1/D2 all use interval1 and report
engines_running=false / step_counter=0 at their export snapshots.
Compared with interval32, median whole handoffs are:
- P->D 0.727 -> 0.436 s (sync1 range 0.306–1.648 s);
- D->P 1.462 -> 0.654 s (sync1 range 0.575–0.794 s).
D export begin/drop/finish medians become 0.314 / 0.0012 / 0.170 s; destination
import medians become 0.271 s to D and 0.152 s to P. First/large handoff remains
a tail: do not claim universal subsecond latency or a production SLA.

model_dp_control.py retains the native step counter and pause-consensus state
transitions and only changes the collective finish-check interval, consistently
across D cores. --dp-finish-sync defaults to32 and leaves the donor implementation
untouched; 1/4/8 are explicit experiment choices. Seven CPU tests cover cadence,
sync inputs, pause consensus, unchanged default and invalid intervals; the
affected control/checkpoint/owner-utility subset passes22 tests. No installed
runtime is patched. The sync1 arm did not repeat the byte/fault oracle: those
96 oracle checks qualify the unchanged direct data path, while this arm qualifies
the bounded control-cadence/token experiment. Representative throughput and
busy-pool/multi-request scheduling costs remain unmeasured. Do not promote
interval1 merely from its better quiescent handoff numbers.

The pinned vLLM core already has background input/output socket threads
(core.py process_input_sockets / input_queue.put_nowait); the main Core loop
consumes the queue. The 32-step rule is global DP finish detection, not socket
polling cadence. A locally empty attention owner may still need to participate
as an EP expert, so local queue emptiness cannot independently stop its rank.
Reuse the native coordinator/FIRST_REQ wave protocol. A useful future connector
boundary is notification -> background transfer -> ready event -> scheduler
publication at a safe boundary; receiving a notification is not authority to
mutate resident/page ownership or reorder HCCL collectives from an IO thread.
No such busy-engine admission integration has been implemented by this experiment.

All resources were released after the sync1 arm. Its source capsule and receipts
are under the task runtime. Long probes should use a detached bounded wrapper,
a prelaunch source capsule and an exit-status sentinel. A quiet long SSH monitor
has disconnected twice despite a surviving job: inspect the recorded PID/status
instead of relaunching. A 45-second monitor heartbeat avoided that transport
idle failure; it is not hardware polling or a substitute for ownership checks.
