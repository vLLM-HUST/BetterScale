# Explore PD storage and head-major transfer

Source-only research, 2026-10-01. BetterScale baseline
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
