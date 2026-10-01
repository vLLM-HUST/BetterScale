# PD KV layout and target-state handoff prototypes

Bounded research prototypes, not production qualification. The first section
records the CPU-only layout experiment; later sections add real DRAM Store,
NPU/model State restoration and isolated P2/D6 integration gates. Release pins
and shared installed runtimes remain unchanged.

## Current-version boundary (2026-10-02)

Keep `idle(core)` for Core admission/mutation in this version. Numerical,
exact-byte and ownership/lifetime correctness are the acceptance boundary;
online load/unload and fine-grained frontend/network pipelining are deferred.
This conservative gate does not imply that online block transfers inherently
require global DP idle. Integrating generation/fence-aware online operations
is separate work, not a reason to relax the current gate ad hoc.

The related `codex/qwen35-incremental-cache` branch at `2dac92c` contains
incremental host backup and sparse device restore work; it was located, not
merged or qualified with this PD path. See the repo-knowledge
[resume entry](../../.agents/skills/repo-knowledge/scenarios/study-qwen35-state-layout/pd-storage.md)
for exact source pointers and evidence boundaries.

## Question and scope

Compare a TP2 producer with (1) TP2 decode, (2) native token-major TP1 decode
with CPU layout conversion, and (3) head-major TP1 decode storage.
The third arm validates bytes only: no attention kernel is changed or qualified.

The wire unit is one target FA layer, K/V plane, global KV head, token interval,
with contiguous [tokens,256] uint16 payload. uint16 carries opaque BF16 bits;
there is no arithmetic or lossy conversion. Ten target FA layers, two heads,
K+V cost 20 KiB/token in aggregate across both producer ranks. MTP is excluded.
The earlier 22 KiB/token census included draft FA and is not this experiment.

## Run

Use an existing NumPy/pytest CPU environment, not a donor upgrade:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q test_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 probe.py \
  --output /path/outside-repo/cpu-layout.json
```

The paged arm handles permuted physical pages and incremental intervals. It
includes Python per-page traversal and validation. Export also includes output
allocation. The bulk arm uses one NumPy copy for the entire ten-layer payload,
contiguous pages and preallocated destinations. It isolates a simpler byte
movement route; it does not implement a real fragmented allocator or connector.
Both arms warm up twice and rotate arm order across seven measured repetitions.
Times are host wall time. Payload GB/s counts useful bytes, not read+write traffic.
No CPU affinity or NUMA binding was imposed. These are same-process repeated
buffers on a shared host, not production throughput or isolated DRAM bandwidth.

## Observed 2026-10-01, hw180 new container

Host CPU Kunpeng-920/aarch64, NumPy1.26.4, Python3.12.13.
192 CPUs visible, eight NUMA nodes; cpu.max=17600000/100000.
22 tests pass, including asymmetric physical maps, incremental boundaries,
guard pages, untouched tails, reverse export, malformed maps and snapshot copies.
Every benchmark arm also validates every layer/plane/head against its input.

Bulk CPU medians (milliseconds, both P ranks' payload combined):

| Tokens | Target KV MiB | TP2 copy | TP1 native transpose/copy | TP1 head-major copy |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2.5 | 0.292 | 0.559 | 0.303 |
| 1024 | 20 | 2.623 | 8.812 | 2.405 |
| 4096 | 80 | 11.919 | 41.435 | 12.069 |
| 16384 | 320 | 46.653 | 166.379 | 47.350 |

Small incremental updates (1/16 tokens) in the paged Python arm cost roughly
1.6ms for all three routes: host loop/validation dominates. This is an argument
for coalescing/batching descriptors, not evidence that actual DMA takes1.6ms.
Paged4096 timing varied between runs (~23–39ms TP2 and ~40–58ms native TP1);
do not treat a single run as a stable hardware bound. See retained samples.

Inference: TP2 is the lowest-integration-risk first physical path. Head-major
could remove native TP1 interleave cost if both writer and attention reader
support it. CPU conversion remains a valid fallback; this single-thread NumPy
implementation is not an optimized C++/multithreaded lower bound, and does not
settle its cost relative to TP collectives or EP. Do not change decode topology
solely from these host-copy numbers.

## Mooncake native preflight: blocked before client creation

Installed mooncake-transfer-engine-npu0.3.11.post1 exposes
batch_put_from_multi_buffers and get_into_ranges. However, a fresh subprocess
with only the Store import prints its success marker and aborts during exit:

```text
corrupted size vs. prev_size
exit134
```

Reproduced with minimal import and with torch imported first. Removing loader
overrides instead produces missing libascendcl.so; that is not a fix.
Normal loader resolves CANN through ascend-toolkit/latest -> cann -> cann-9.1.0.
BetterScale's qualified baseline is CANN9.0.1. Neither this version difference
nor the native failure establishes the cause of heap corruption.
No os._exit workaround, client/server test, upgrade or shared-library edit was
used. Native Store roundtrip, SSD behavior and accelerator overlap remain OPEN.
The installed wheel is not asserted identical to the independently inspected
Mooncake source0d1a804. Its API presence alone is not compatibility evidence.

Exact reproducer (disable core files):
```sh
ulimit -c 0
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c \
  'from mooncake.store import MooncakeDistributedStore; print("STORE_IMPORT_OK", flush=True)'
```

## Boundaries for the next prototype

1. A clean-exit Mooncake runtime is needed before accepting a real Store smoke.
   Use an isolated, version-identified runtime; preserve the current installation.
2. Start real Store roundtrip with registered CPU buffers and immutable chunks;
   validate multi-buffer writes/ranged reads before device transfer.
3. TP2 can use current target page ordering. TP1 head-major requires separate
   cache-writer and attention-reader qualification, not just a view change.
4. No GDN checkpoint, MTP bootstrap, owner fencing, manifest atomicity, eviction,
   retries or asynchronous buffer lifetime is implemented by this byte-layout
   probe. Do not present passing tests as a production PD connector.

Full JSON receipts are retained outside Git at /workspace/pd-kv-layout-results/
on hw180. The compact receipt alongside this README records key results.

## Subsequent user decision and isolated runtime check

Fletcher accepts sub-second layout conversion as a workable first prototype
tradeoff. Prefer CPU conversion with existing target layouts before paying for
head-major attention changes or changing decode TP solely for byte layout.
This tolerance does not establish an end-to-end latency SLO or cover unmeasured
full-history restores/queueing. Dense copies should remain incremental.

An isolated venv at /workspace/pd-kv-layout-results/store-venv was used to test
an alternate public wheel without modifying the installed NPU package:
mooncake-transfer-engine0.3.13.post1, cp312 manylinux_2_28_aarch64 (74.4MB).
With loader overrides removed, Store import fails on missing libcuda.so.1.
Thus the generic wheel is not a working CPU-only substitute in this container.
No fake CUDA library, forced process exit, or driver/runtime installation was
attempted. Both package failures are environment receipts, not Store API tests.

## Resolved CPU Store route via mooncake-hust

Inspected vLLM-HUST/mooncake-hust c992ba75a86a944cbb5440241545ec2153bc4532.
HUST_FORK.md describes a thin upstream-aligned fork, not a separate core fix.
Its quick-start names mooncake-transfer-engine-non-cuda. The earlier generic
CUDA wheel was the wrong choice for CPU-only testing.

The non-CUDA0.3.13.post1 cp312/aarch64 wheel is installed in a fresh isolated
/workspace/pd-kv-layout-results/store-cpu-venv. Import and normal interpreter
exit pass with loader overrides removed. This is the published wheel recommended
by fork documentation, not a build of fork HEAD. Original NPU/donor installs
are untouched.

store_smoke.py passes with two clients in one process, a32MiB segment and16MiB
local buffers per client:
- multi-buffer put of two64KiB head chunks;
- cross-client full-object exact bytes;
- ranged scatter into reversed head destinations,32-byte guards intact;
- partial token17..33 range, untouched prefix/suffix;
- immediate normal removal returns OBJECT_HAS_LEASE(-706);
- normal removal succeeds after the explicit2000ms read lease;
- clients close, master exits0, Python exits0.

An initial expiry test assumed5000ms from upstream tests, but the installed
master defaults to10000ms. The final test configures its TTL explicitly.
Earlier test-owned cleanup used force=True; the final gate does NOT.

TCP is configured, but same-host memcpy is auto-enabled. This qualifies object
and range APIs, not cross-machine bandwidth, PCIe, SSD or device overlap.
Transfer Engine can choose dynamic internal service ports; all test clients and
master were closed. No accelerator context was started.

Run using the isolated non-CUDA environment:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH /path/to/cpu-venv/bin/python store_smoke.py --base-port 55241 --output-dir /path/outside-repo/store-smoke

Choose an unused local six-port range; recent ports may remain in TIME_WAIT.
Full final receipts: /workspace/pd-kv-layout-results/store-smoke-final on hw180.
Compact receipt: store-receipt.json. NPU import corruption remains unresolved.

## Session/checkpoint and DRAM mainline (2026-10-01)

`session.py` is a bounded protocol oracle, not a production coordinator:
SQLite CAS epochs fence publication, Store contains immutable chunks/manifests,
and only a completed dense frontier plus target GDN/conv checkpoint transfers
ownership. The single-thread Store executor has two-batch / 64 MiB backpressure.
Failed transfers cannot publish; incomplete restore fails closed before live
State installation. Distributed consensus, orphan GC, scheduler/device fences,
chunk compaction and production placement/eviction policy remain open.

`gdn_checkpoint.py` captures selected target recurrent and normalized conv state,
converts TP2/TP1 shard geometry with Q/K/V channel boundaries intact, and validates
all leaves before installation. `test_declared_checkpoint.py` additionally uses
real BaselineStateRoot CPU declarations/binding. Other seats and MTP buffers stay
untouched. This is a byte/state-address oracle, not resumed-model parity.

The real Store smoke now exercises P0 -> D0 -> P0 -> D0 (129/145/257 tokens),
with two incremental writes per turn and deliberate missing-object rejection.
GDN/conv payloads in this Store test are opaque synthetic bytes, not model state.
It also runs an independent reader process. Reserve SIX explicit local ports;
Transfer Engine may allocate additional internal ports. Ordinary object removal
checks lease expiry; deliberate missing-object injection uses force only on a
probe-owned key. `session-final/receipt.json` outside Git preserves the full run.

Fletcher chose DRAM as the current backend; disk is not a prerequisite. Disk-only
full reads succeeded, while native `get_into_ranges` failed with
`Invalid string value: too few bytes` / `RPC_FAIL`. Full bounded chunk read plus
CPU slice passed. Root cause is unknown, no corruption claim. Published report:
https://github.com/vLLM-HUST/mooncake-hust/issues/3

The isolated NPU wheel 0.3.13.post1 imports and exits cleanly; this does not repair
or explain the system NPU 0.3.11.post1 shutdown corruption. Keep installs isolated.

## Admitted NPU host-staging probe

Fletcher explicitly assigned the whole new hw180 container/machine for today's
work (2026-10-01); the old machine's lease service is not applicable. Do not
propagate this one-day authorization to future shared-host work. Preflight found
all eight NPUs healthy, idle, and without processes. Probe uses only NPU0.

`npu_staging_probe.py`: torch2.10.0+cpu, torch_npu2.10.0.post4, CANN9.1.0,
910B2. Nonblocking copies on a dedicated stream, explicit completion events,
pinned host buffers, exact roundtrip bytes. One warmup + three measured repeats,
wall-clock median including submission/event wait, no affinity tuning:

| MiB | H2D ms | D2H ms |
|---:|---:|---:|
| 20 | 1.451 | 1.368 |
| 80 | 4.917 | 4.971 |
| 320 | 16.356 | 17.133 |

These are basic same-device staging measurements, not Store end-to-end,
compute-overlap, cross-host, model correctness, or BetterScale baseline runtime
qualification (baseline CANN9.0.1). Receipt: npu-staging.json outside Git.

`store_smoke.py --npu-staging` now passes the actual NPU0 -> pinned two-slot
ring -> immutable DRAM objects -> pinned host -> NPU0 path: eight 2.5 MiB
chunks (20 MiB), distinct byte values, exact restoration, clean normal exit.
A Store thread waits each device event; the producer waits the slot's Store
future before reusing it. Failure cleanup drains both executor and stream.
First-run publish time was 91 ms including cold fill/submission, NOT a throughput
benchmark. No artificial compute-overlap result is claimed. Full receipt:
/workspace/pd-kv-layout-results/npu-store-staging-loader/receipt.json.

Environment: isolated `store-staging-venv` with system-site-packages for torch
and its own non-CUDA0.3.13.post1 wheel. Keep normal CANN loader paths in the
parent (torch_npu needs libhccl.so); the CPU master/reader subprocesses use the
harness's scrubbed loader environment. Removing LD_LIBRARY_PATH from the NPU
parent failed at import before allocation; restoring existing loader paths
resolved it, without changing system libraries. All 51 CPU protocol/layout/
State-binding tests pass together. Direct Ascend transport remains untested.

## Owned GDN host-checkpoint continuation, CANN9.1

`gdn_resume_probe.py` passed24 captured waves on NPU0 using the exact owned
qk8/value16/d128 TP2 recurrence and an independent FP32 CPU recurrence. Three
active rows cycle accepted candidates1..3, plus a retired row. At wave12, after
writer completion, the selected1MiB target recurrent State goes through CPU,
restores byte-exactly into a different candidate-slot group and continues with
selector1. All candidate states, outputs, untouched slots and zero retired-row
output are checked each wave. The CPU reference does not adopt device State
after restore. This validates one-layer recurrence/host export/resume, NOT
conv history, Store transport, full-model correctness or inter-rank execution.

Run in the isolated pinned-core venv with one authorized idle device exposed:
`ASCEND_RT_VISIBLE_DEVICES=0 TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
VLLM_PLUGINS='' python gdn_resume_probe.py --output /new/receipt`.
The new-container receipt is
`/workspace/betterscale-pd-runtime/gdn-resume/complete.json`; all NPUs were idle
after normal exit.

`store_smoke.py --gdn-resume` composes that same numerical gate with two real
Store clients: the wave12 selected recurrent checkpoint is put into DRAM and
read by the other client before new-slot restoration. This passed with the same
output/state errors, clean client/master exit and all devices idle afterwards.
Receipt: `/workspace/betterscale-pd-runtime/gdn-store-resume/receipt.json`.
Use the clean `store-staging-venv` and put the exact core source checkout on
PYTHONPATH (the probe imports its Triton utilities). This is one1MiB blocking
checkpoint, not asynchronous dense transport, conv restore or full model PD.

## Pinned native runtime and composed prefill, CANN9.1

The exact Ascend0.25.1rc1 source built successfully against the current toolkit.
Wheel and isolated adapted donor are under `/workspace/betterscale-pd-runtime/`;
`runtime.prepare` validated every adapted source digest and native payload.
Correct extension import is `vllm_ascend.vllm_ascend_C`, not `_C`.

`conv_resume_probe.py` passes eight captured TP2 prefill-conv waves with an
independent CPU depthwise-conv/SiLU oracle, canonical3-token history and new-slot
restore. Histories are byte-exact; max output error0.00024414.
`prefill_resume_probe.py` then passes the actual owned MixedCore prefill path:
conv, gates/normalization, layout-fused WY, owned AscendC chunk/H/O, and output
restore. Eight variable-length waves cross64/128 tile boundaries and transfer
both conv/GDN host checkpoints at wave4. CPU recurrence remains independent
after migration. Max output error2.143e-5, State error2.620e-4; conv histories
exact. This is prefill-only one-layer composition, not speculative mixed rows,
Store-backed composed state or full-model inference.

Standalone MixedCore callers must initialize pinned donor Triton device properties
with `init_device_properties_triton()` after device selection, as the real Worker
does. The first attempt omitted this lifecycle call and failed before numerical
execution; the corrected receipt is `prefill-resume-initialized/complete.json`.
Use the isolated owned donor and repository source on PYTHONPATH,
MTP_GDN_LAYOUT_FUSION=1, BETTERSCALE_GDN_SMALL_COPIES=1, and explicit rebuilt
BETTERSCALE_GDN_LIBRARY / BETTERSCALE_GDN_HOST_LIBRARY. Hardware admission still
belongs to the caller. Product native/version pins have not yet been promoted.

## Six-rank same-host D transport

`ep6_transport_probe.py` passed on physical cards2..7: three TP2 groups perform
exact reductions; pinned core expert mapping assigns256 experts as
43/43/43/43/42/42 with exactly one owner each. Eight unequal-split HCCL all-to-all
dispatch/combine roundtrips preserve exact rank/expert identities. Receipt:
`/workspace/betterscale-pd-runtime/ep6-transport.json`. This is communication and
core mapping evidence, NOT qualification of Ascend fused-MoE, DP3 scheduling,
model EP6 or PD serving. Each initialized process group is explicitly retired.

The isolated venv inherits torch but has no `bin/torchrun`; use its Python with
`-m torch.distributed.run --nnodes 1 --nproc-per-node 6 --master-addr 127.0.0.1
--master-port 29661`. Expose exactly six
authorized devices. The probe is bounded by120s HCCL watchdog and an outer
240s process timeout; it does not scan ports or hosts.

### EP6 native MoE numerical leaf (2026-10-01)

`ep6_moe_probe.py` passed four six-rank eager waves on physical2..7. It uses
pinned core linear43/43/43/43/42/42 ownership and actual CANN BF16 routing,
grouped matmul, SwiGLU, second grouped matmul and unpermute, with HCCL input
all-gather/output reduction. Independent CPU per-expert products reproduce
BF16 stage boundaries; max absolute error0.000227. Cases cover all256 experts,
changing ownership, a skew with five entirely empty expert ranks, and zero
contribution/padding rows. Receipt: task runtime `ep6-moe.json`. This is a small
hidden64/intermediate64 eager fixture, not model-size/FULL-graph qualification.

Pinned A2's `_select_a2_moe_comm_method` selects ALLGATHER for EP6, not MC2.
Two Python integration gaps precede an actual model launch: AscendMoERunner's
local-capacity check assumes floor(256/6)=42 on every rank; ALLGATHER dispatcher
uses rank*local_count, giving incorrect rank4/5 starts168/210 instead of172/214.
Its non-EPLB placement helper already returns the correct core map. ALLTOALL
has additional equal-count assumptions but is not this A2 default route.
Correct capacity/range arithmetic under an explicit uneven-linear/no-EPLB
contract; do not change the donor source pins or infer MC2 support from this
all-gather result. Current BetterScale TP2/DP1 admission is still unchanged.

### New-runtime model continuation gate remains open

`model_continuation_probe.py` compares warm resume, cold reconstruction and an
uninterrupted greedy generation, recording tokens/logprobs and exact cache hits.
CANN9.1/post4 candidate4 (owned FIA Q1..16) serves4412-token prefill and4476-token
warm continuation with4475 hits, but warm diverged at continuation token9.
Cold reconstruction and uninterrupted generation match exactly across64 tokens.
A fresh diagnostic repeats the mismatch; at the first difference warm favored
`key` by3.25 logits, cold favored`gate` by8, not a near-tie explanation.
Receipts: task runtime `model-smoke-q16/`, especially `diagnostics/`.

A length diagnostic had matching32-token warm continuations for output lengths
1..8,15..17,31..33, but differences for63..65 (at continuation30/29/28 in that
run). Do not generalize one failure index to every seat/history. Appending1 or8
newline tokens matched cold; appending3 differed immediately. The appended16
case failed closed on a draft-frame query exceeding Q16. Shape admission and
numerical continuation are separate unresolved boundaries.

To distinguish kernel transitions from scheduler/terminal ownership,
`gdn_transition_probe.py` uses actual owned prefill/verify compositions with
5-token extended conv storage, independent CPU recurrence, selected candidates
1..3 and verify→prefill→verify transitions. All24 waves pass (max output error
0.00002447). Receipt:`gdn-transition-4/complete.json`. Earlier fixture attempt3
incorrectly reused the recurrent selector as the conv selector after prefill;
production intentionally keeps these separate. This passing leaf does not clear
the model-level failure. The two required native library envs are
`BETTERSCALE_GDN_LIBRARY` and `BETTERSCALE_GDN_HOST_LIBRARY`.

`target_only_diagnostic.py` suppresses only the scheduler's next proposal list,
retaining draft computation and lookahead buffers. It is an isolation experiment,
not production P-only execution or a claim that cold MTP State is safe.

### Target-only full-model isolation

Both `model-target-only-continuation/receipt.json` (draft computed but proposals
not scheduled) and `model-no-draft-continuation/receipt.json` (no real-request
draft forward) pass exact64-token warm/cold/uninterrupted equality on the4412
prompt. Warm hits4475 tokens; cold hits0. This localizes the observed earlier
failure to the speculative path/composition, but does not establish its cause.
The second mode keeps native asynchronous sampled-count/next-token publication;
returning no draft is not permission to skip that feedback. Startup still loads
and captures the baseline draft model: memory/initialization reduction is open.
Three CPU tests protect proposal suppression, feedback, and rejection of actual
speculative work in this profile.

### Integrated EP6 source boundary and current evidence

`stage_ep6_runtime.py` now requires a pristine pinned model runner, stages only
the Ascend package (not an entire site-packages directory), and applies the two
explicit uneven-linear EP6 Python corrections. The native diagnostic has its own
candidate pin manifest using the original four donor files, not the State/MTP
adapted ones. An earlier attempt accidentally used the owned runner, hit the
State APC hook with a native no-APC configuration, and failed; it is not native
MoE evidence. Preserve `ep6-dummy-4` under the task runtime root.

`ep6-dummy-5/complete.json` passes three simultaneous native TP2 clients/EP6,
four dummy layers, eager, two balanced/skewed request phases and all clean exits.
The full40-layer real-weight `ep6-real-2` executes both phases and exits cleanly,
but all six strict code-retrieval checks FAIL (repeats prompt filler instead).
The same64-token MARBLE input returns the correct answer through our TP2
no-draft path (`ep6-quality-control.json`). The original-native TP2/no-EP control now also passes both retrievals
(`ep6-native-tp2-control/complete.json`), narrowing the failure to the EP6
integration rather than a generic native new-runtime failure. This does not yet
distinguish MoE routing/weights/reductions from DP-dependent attention behavior.
No six-rank real-model correctness qualification is claimed.

Launcher pitfalls are fixed in `run_ep6_model_probe.sh`: source CANN+ATB without
nounset (vendor scripts reference unset shell variables), then append CANN's
Python path so `acl` remains importable. The pinned config parser first calls an
HF override on a model-type-only placeholder. Transformers5.14 chat templates
return BatchEncoding by default: use `return_dict=False` and assert an integer
list before passing prompt_token_ids; malformed input led to client shutdown
waiting on DP peers, not an accelerator deadlock established by evidence.

### First complete target checkpoint vertical

`model_checkpoint.py` is an idle-engine-only TP2 prototype, not a production
connector. Core owns native FA page references and resident epochs; workers
export only target dense KV, selected GDN and canonical conv history. Import
validates every tensor before writes, installs into fresh page/seat ownership,
normalizes selectors, updates the worker epoch mirror so ingress does not erase
the imported State, and publishes the hot resident only after both worker acks.
Failure releases pages and invalidates the unpublished epoch. MTP is excluded.
Four CPU tests use the actual ResidentLeases implementation to cover successful
publish/export/drop, incomplete acks, worker failure and active-engine rejection.

`checkpoint_model_probe.py` / `checkpoint_entry.py` select the existing no-draft
candidate plus idle checkpoint utility methods; all other baseline gates remain.
The model gate exports, drops the original hot cache, restores at different
physical storage, then requires cached continuation to match cold exactly.
CPU tensors are the initial transport; Store integration, incremental overlap,
concurrent import and cross-owner execution are subsequent gates, not claims of
this first vertical. Model result is pending.

The first checkpoint model attempt failed before requests because the offline
entry imported the Worker/model graph before native general-plugin patches;
`UnquantizedFusedMoEMethod.is_monolithic` was missing. The launcher now loads
general plugins first, matching normal CLI bootstrap. Preserve
`logs/model-checkpoint-roundtrip.log`; this is not checkpoint numerical evidence.

The failing EP6 first-layer trace (`ep6-trace`, six external `.pt` files) shows
all local expert weights exactly match their source model slices, but TP peers
receive different half-sequences: nonzero input differences occupy rows0..31
of the64-token request, with the other half zero padded. The source explains
this: core's `use_sequence_parallel_moe` enables model-side chunking under
DP+TP+EP; Ascend `platform.py:615` switches `all2all_backend` to
`flashinfer_all2allv` to disable that when its non-SP route is active, **but only
for worker_cls=auto**. Our explicit Worker skips that platform fixup. The native
probe now sets the same backend marker explicitly and guards against model-side
SP. This marker does not select CUDA execution: Ascend still chooses its native
AllGather implementation. The repaired real-model gate must pass before this
source/trace diagnosis is considered sufficient runtime qualification.

Checkpoint attempt2 exported target State and retired the original cache, but
import rejected tensor geometry. Source inspection found the safe utility RPC
encodes tensors into untyped triples and does not reconstruct nested tensors
without insecure serialization. The prototype now uses an explicit BF16/FP32
shape/dtype/bytes envelope, validated before decoding, rather than enabling
pickle. Its msgpack roundtrip and malformed-length CPU test pass; model rerun is
required. Keep `model-checkpoint-roundtrip-2` as a failed integration receipt.

**Post-fix result:** `ep6-real-nosp/complete.json` passes all six strict real-model
retrievals (DP lengths64/65/65, then834/65/65), with clean exits on all clients.
The first-layer captured expert weights remain exactly equal to the source
slices. CPU-local MoE sampled-row errors are small (largest0.0001135) after the
non-SP correction (`ep6-trace-nosp/analysis.json`). This qualifies this native
eager no-MTP gate, not owned State, FULL DP replay, or P/D handoff.

**Complete target restore result:** `model-checkpoint-roundtrip-3/receipt.json`
passes exact64-token restored/cold continuation equality. Cursor4475 is fully
cached after restoring from seat0 to seat1 and physical FA block order[1,2,3] to
[3,2,1], after dropping the original resident. Both workers acknowledge the new
epoch and no draft state is transferred. One observed quiescent utility-RPC
export/import costs1.006/1.117seconds including host copies/serialization, not
an incremental PCIe benchmark. The explicit wire path fixes attempt2 without
enabling insecure utility serialization. This is still one TP2 engine, not
cross-group or Mooncake model transfer.


### Owned State on six decode cards

`stage_ep6_state_candidate.py` copies the qualified no-draft candidate and changes
only two topology admission checks in that isolated artifact, not the released
source. `ep6_state_entry.py` keeps the same Scheduler/State ownership and adds an
explicit post-startup idle-rank route: eager with no attention metadata, and no
draft dummy forward. This is necessary because idle ranks must participate in
target MoE collectives without modifying hot sessions or issuing unmatched MTP
collectives. Native startup/capture is unchanged; two CPU tests check suppression,
restoration on errors and unchanged startup behavior.

`ep6-state/complete.json` PASSES real-model DP3/TP2/EP6 target-only State with
uneven prompts4412/252/32 and output lengths64/16/8. All three32-token warm/cold
continuations match exactly; warm hits4475/267/39, cold hits0. Early-finished
ranks retain their hot sessions while peers continue. All clients exit0 and all
six NPUs are subsequently free. This is not qualification of FULL idle replay.

`model_store.py` bridges explicit target wire tensors to40 head-major streams,
selected GDN and canonical conv blobs through the existing session protocol.
Only newly committed dense ranges become new Store objects; previous immutable
chunks are retained. The current producer still exports complete snapshots at
quiescent boundaries; no incremental D2H or overlap claim. Two CPU tests cover
bidirectional model geometry, append-only manifests and missing dependencies;
`model-store-cpu-2/receipt.json` repeats full-geometry synthetic bytes against real
Mooncake DRAM. `dram_store_fixture.py` owns a task-local loopback CPU master,
one1GiB DRAM segment and two clients, with bounded startup/cleanup.

`pd_model_probe.py` launches P on0/1 and three D TP2 owners on2..7 concurrently,
then attempts P→D→P→D→P per session through that Store adapter. It compares second
D continuation with cold computation and checks exact cache frontiers. Result
is pending; it is a synchronous turn-boundary first vertical, not the desired
compute/PCIe/Store incremental pipeline or a distributed durable coordinator.

The first actual P2/D6 run booted all four engines and transferred4412-token
real target State P→D0 through DRAM with both import acknowledgements (3.91s
whole quiescent handoff, not PCIe timing), then D0's first request waited on idle
EP peers. The offline `SyncMPClient` instances have no DP coordinator and do not
broadcast FIRST_REQ; prior three-client probes submitted work on every rank.
The explicit test controller now wakes the other idle Core loops when dispatching
to one owner, preserving each Core's wave counter and rejecting composition with
a native coordinator. A CPU contract test covers that boundary.

Do not promote that experiment-only wake helper into a second production DP
protocol. Pinned `DPAsyncMPClient` already sends FIRST_REQ to its coordinator;
`DPLBAsyncMPClient.get_core_engine_for_request` honors `request.data_parallel_rank`,
and the completion frontend reads `X-data-parallel-rank`. Production session
routing can reuse those existing owners/waves instead of recreating them.

The stuck first run was intentionally interrupted, not counted as a successful
D continuation. Its native signal path left the task's CPU master alive; that
exact master was identified by its loopback55401 command and terminated, and all
NPUs were verified free. Attempt2 failed only the occupied-port preflight before
NPU launch. The fixture now records its master PID and reinstalls Python SIGINT
unwinding after client initialization, with master cleanup in a finally block.
Retain `logs/p2d6-model-store*.log`; corrected model attempt3 is pending.

**Actual P2/D6 result:** `p2d6-model-store-4/complete.json` PASSES all three fixed
D owners. Each real-model session completes P→D→P→D→P via Mooncake DRAM (12
handoffs total), appends a second user question, and matches16 restored D tokens
exactly against cold recomputation. Warm frontiers4440/280/60 are fully cached;
all cold controls hit0. MTP is never transferred. Process exit0, all8 NPUs free,
and no task master left listening. Whole synchronous handoffs range1.41–4.63s;
these include complete snapshot export, Python RPC/serialization and Store, not
just PCIe or layout conversion. Dense Store objects append incrementally, but
NPU→CPU is still full-snapshot and no compute overlap is claimed.

Attempt3 already passed the long D0 conversation, then D1 correctly rejected an
export while async terminal work still owned State. `pd_export_retired` now uses
native utility Futures: service the pending export only after the normal engine
step drains requests, queued work and pending-hot fences. It never sleeps for an
assumed fence delay or weakens the idle guard. A CPU test checks both batch-queue
and pending-hot barriers. The fourth run above qualifies that path end to end.
A native C SIGINT handler can make signal.getsignal return None; the fixture uses
Python's default interrupt handler as the restorable fallback rather than trying
to register None. Attempt3's cleanup did terminate its master despite that error.

### Actual incremental NPU→CPU dense export

`pd_model_probe.py --incremental` now passes the same three-owner/two-turn matrix
through real DRAM (`p2d6-model-incremental/complete.json`). The producer selects
only intersecting logical pages and copies only `[Store frontier, target cursor)`
to CPU; import refuses a partial payload unless Store first reconstructs all
required history. Permuted-page/partial-tail CPU checks and a delta reconstruction
test cover the new boundary. The retained State-retirement Future is unchanged.

All12 handoffs still pass exact restored/cold16-token continuations at4440/280/60.
Each D turn exports only16 new dense tokens (327,680 aggregate bytes); each second
P turn exports12 (245,760 bytes), alongside the full selected GDN/conv checkpoint.
Total dense D2H payload is98,877,440 bytes versus390,103,040 for full snapshots of
those same frontiers. This is an exact payload count, not a throughput claim.
We also removed one redundant checkpoint repack in the controller; elapsed-time
changes must not be attributed solely to reduced PCIe bytes. The pipeline remains
quiescent and synchronous at the RPC boundary, and receiver H2D remains full.
No compute overlap, async page-reference lifetime, DRAM eviction policy, native
async frontend, or production recovery qualification is implied.

### Model-page D2H → DRAM double buffering

`pd_model_probe.py --streamed` passes the complete same P2/D6 matrix at
`/workspace/betterscale-pd-runtime/p2d6-model-streamed/complete.json`:12 handoffs,
three exact warm/cold second D continuations,98,877,440 incremental dense bytes.
Exit0, all8 NPUs free, task Store master gone. Each TP worker directly stores its
20 head-major planes through two20MiB pinned slots. D2H events gate Store reads;
acknowledged Store futures gate slot reuse, with gathered device tensors retained
until completion. All jobs and the copy stream drain before unregister/close,
including failure. Dense bytes no longer pass through Core/controller RPC.

Core still executes a synchronous **retired idle** export: its existing page
ownership lasts until both workers finish. This is transfer-stage pipelining,
not overlap with model execution or qualification of active-page lifetime.
The long initial P export enqueued both later chunks while an earlier Store
future was pending on each TP rank; that is software concurrency evidence, not
hardware profiling. Producer timing includes per-call Store setup, not teardown.
Whole handoffs2.55–4.16s include full GDN/conv CPU/RPC and full receiver H2D; no
latency improvement or physical network throughput is claimed. Same-host CPU
Store may use memcpy. Per-call clients are intentionally not yet amortized.

`model_stream_probe.py` first qualified NPU0 synthetic model-page geometry,
permuted physical IDs, four chunks/two slots and a partial-tail increment against
exact Store bytes (`model-stream-oracle/complete.json`). The internal
`Turn.append_acknowledged` seam accepts only producer-acknowledged, unique keys
under the current session/epoch namespace. Both TP layouts/frontiers/stream sets
are checked before the existing final manifest CAS. Acknowledgement does not
prevent later eviction: restore still checks every dependency and fails closed.
CPU tests cover missing acknowledgement, mismatched spans, foreign keys, duplicate
keys, revoked epochs and dependency loss. Checkpoint/Store/session suite:27 pass.

Next lifetime boundary: do not simply release the idle RPC and leave the same
copy jobs running. Pin native block-pool references before any asynchronous
export, preserve immutable token intervals (not just Tensor object lifetimes),
and release pins only after both workers' DMA/Store completion, including abort.
GDN/conv must remain a selected, matching retired frontier. Production still
needs native async frontend integration, eviction/recovery and draft validity.

### Retired asynchronous export — experimental, NOT qualified

`--async-export` starts at the same retired frontier, snapshots selected GDN/conv
before returning, pins native FA block references, and runs dense transfer in a
worker thread. The probe evicts the source hot resident and executes an unrelated
8-token request before collecting the transfer. Pins release only after both TP
workers explicitly acknowledge drained work; RPC/unknown-drain failures quarantine
pins rather than pretending DMA finished. Completed worker receipts survive a
Core-side retry. The slot/stream resources are retained if device drain fails.
The one-export-per-engine bound is intentionally narrow, not a scheduling policy.

**Do not promote this route:** `p2d6-model-async-retired/failure.json` passes the
long D0 session, then D1's280-token second continuation diverges at output index4.
Interestingly its restored sequence matches all three earlier qualified runs;
the new cold control instead emits `**Final Answer:**`. This is an observation,
not proof that transfer or cold compute is at fault, nor a near-tie dismissal.
All8 NPUs and its Store master were subsequently verified released.

The changed-hypothesis control `--serialize-export` waits for each worker job
before the unrelated request, **without releasing native pins** or changing the
hot eviction/request sequence. `p2d6-model-async-serialized/complete.json` passes
all12 handoffs and all three exact continuations; exit0 and full cleanup verified.
This narrows the investigation to concurrency/timing-sensitive behavior, but does
not establish the faulty component. Worker host-job intervals include Store client
teardown; their intersection with a request interval is NOT hardware DMA/compute
profiling. A next `--verify-transfer` diagnostic compares every streamed Store
chunk byte-for-byte against a retired main-thread CPU snapshot and records top5
output logprobs. Its initial live run is `p2d6-model-async-oracle` (pending).

Eight new CPU tests cover retained page references after eviction, single-export
admission, wrong IDs, missing/failed drain acknowledgements, retry, Store failure,
worker completion fencing and the serialization control's unchanged pin lifetime.
The existing blocking `--streamed` route remains the last qualified transfer mode.

Native async frontend follow-up: at pinned core752a3a504, DPLBAsyncMPClient's public
`call_utility_async` **broadcasts to every Core and returns only the first result**.
Do not use that method for an owner-specific session export/import/drop. Its
`_call_utility_async(..., engine=core_engines[owner])` is the inspected targeted
seam; any eventual adapter must preserve the pinned source boundary. Request
routing/FIRST_REQ already exists as described above; don't build a second DP
coordinator just to work around the offline experiment.

**Hardware migration stop (2026-10-01):** Fletcher requested backup/sync because
hw180 is being reclaimed. The `p2d6-model-async-oracle` run was interrupted during
startup, before any new numerical/transfer result; its KeyboardInterrupt receipt
is not a failed oracle or qualification. Resume that discriminator on the newly
assigned hardware, after checking its environment and resource authority. The
last qualified route remains blocking `--streamed`; serialized async lifecycle
control passes, concurrent async numerical gate remains unresolved. No native
release pins or shared system installation were changed.


### hw86 migration admission (2026-10-01)

The PD source and both exact donor commits are restored. The private runtime
archive passed its transfer SHA-256 check; inherited system CANN9.1.0,
Torch2.10.0+cpu and torch_npu2.10.0.post4 match the previous host. Driver changed
from25.2.1 to26.0.rc1. The isolated venv retains vLLM0.25.1,
vLLM-Ascend0.25.1rc1 and transformers5.14.1; no system package was replaced.

The35 checkpoint/Store/session/async CPU tests pass. New one-card receipts under
`/workspace/betterscale-pd-runtime/`:

- `hw86-fia-query-transitions/complete.json`: owned FIA against the CPU oracle,
  two captured banks,16 replays, changing Q lengths, guards/input immutability
  and zero-padding checks pass. This does not use the parked native FIA oracle.
- `hw86-model-stream-oracle/complete.json`: permuted synthetic model pages and
  partial-tail increments roundtrip through two-slot pinned D2H/DRAM Store with
  exact byte comparisons. No model inference or compute overlap is claimed.

All NPUs and the task Store master were released after these gates. Full model
qualification waits for the replacement weight download; old weights were not
part of the migration archive. Override the old shared-path default with
`BETTERSCALE_MODEL_PATH=/workspace/models/Qwen3.5-35B-A3B` when launching
`run_pd_model_probe.sh`. Do not silently substitute another installed model.
The last qualified real-model transfer remains the old-host blocking route.


### Restored model gates and native async frontend boundary

On hw86, `hw86-p2d6-streamed/complete.json` passes all12 real-model handoffs and
all three exact16-token warm/cold comparisons. The second-decode sequences also
match the old-host blocking baseline. Whole handoff2.43–3.75s remains an end-to-end
correctness measurement, not PCIe bandwidth. `hw86-p2d6-async-oracle` also passes
all12 handoffs,24 TP shard exact-byte oracles and all three model comparisons;
`hw86-p2d6-async-plain` passes without that extra snapshot or logprob collection
(exit0). These bounded passes do NOT establish that the old discrepancy was fixed.

`--native-async` replaces the four offline SyncMPClient actors with one P AsyncLLM
and one D DPLBAsyncMPClient pool. Common model options are unchanged. The native
FIRST_REQ/coordinator handles EP peer activation; the controller issues no manual
wake. Generation explicitly supplies `data_parallel_rank`; session utilities use
only the pinned targeted `_call_utility_async(..., engine=core_engines[owner])`
seam. Eight CPU tests reject invalid owners/manual wake and protect against using
the public broadcast utility; the affected suite totals43 passing tests. This is
an in-process frontend integration experiment, not an HTTP service or production
connector. Source gates/candidate runtimes remain unchanged.

**The first native frontend numerical gate failed**, even with blocking
`--streamed`: `hw86-p2d6-native-async/failure.json`, D1 second continuation,
frontier280, first output difference at index4. Its warm output now takes the
`**Final Answer:**` branch previously seen in the old async run's cold output.
Its cold output matches the earlier baseline. Input token IDs and all preceding
P/D outputs are identical to the restored blocking baseline. Therefore concurrent
KV export is not a necessary trigger; this does not prove every transfer is sound.

The changed diagnostic `--logprobs --cold-controls 2` records repeated all-cold
requests after each warm/cold pair, without enabling the transfer snapshot.
`hw86-native-cold-controls/complete.json` passes12 handoffs/all model comparisons,
but D1's index4 top-two logprob margins vary0.25(warm),0.125(cold),0.375/0.375(two
fresh cold controls). This measures variability, not an accepted error tolerance.

### Pure-D discriminator: output variation without PD

`native_cold_probe.py` starts **only D6**, with the same owned target-only runtime
and native AsyncLLM coordinator. It never creates P, Store, or invokes checkpoint
utilities. Supply `--prompt-receipt` from the preceding D1 cold receipt,
`--output`, and optionally `--repeats`(2–16, default12)/`--owner`(default1). Use the
same task-local CANN/ATB/native-library environment as `run_pd_model_probe.sh`,
then run this script instead of `pd_model_probe.py`. Its `completed` status means
the experiment completed, not that numerical determinism passed.

`hw86-native-cold-only/complete.json` and `numerical-summary.json`: same281-token
input, same D1 owner,12 unique salts, every cached count0,16 greedy output tokens.
There are **two token sequences (9 versus3)**, exactly the two earlier branches.
At the first divergent output position4, two alternate outcomes have a tie, but
one has the alternate token332 ahead of22365 by0.25. Do not dismiss the whole
phenomenon as an argmax tie. Cold runs can use different resident seats/pages and
DP wave timing; this is not proof of a particular kernel, storage, allocator or
coordinator defect. All NPUs and task Store masters were verified released.

Historical decision boundary (superseded by the accepted numerical envelope
below): exact warm/cold token equality is confounded by a
reproduced pure-D baseline variation. Do not weaken numerical acceptance or
promote production correctness implicitly. Transfer byte/lifetime evidence still
stands within its envelope; model numerical qualification/root-cause work is a
separate unresolved gate. No change to the parked native FIA-reference bug work.

### Numerical repeatability diagnostics (hw86)

The fixed-seat cold control still bifurcates without P/Store/checkpoint. Bounded
white-box taps locate first-layer variability after DP MoE reduction, and a
fixed-input BF16 reduce-scatter leaf reproduces nondeterminism. Strict HCCL
eliminates variability in that leaf and in12 original-compiled cold repeats
(tokens and top5 logprobs exact). The strict native P2/D6 matrix passes12
handoffs, but fixed warm/cold logprob differences remain. Same-owner no-transfer
controls and independent FP64 recurrence isolate a GDN chunk/State
continuation accuracy seam. Fletcher accepts the measured envelope for current
PD development; it is no longer a blocker. Strict HCCL is a repeatability
control, not a demonstrated accuracy requirement: saved strict/non-strict leaf
outputs both have about0.232% relative L2 error against FP64 summation. This
does not qualify every production workload.
See the scenario's [evidence and reproduction envelope](../../.agents/skills/repo-knowledge/scenarios/study-qwen35-state-layout/pd-storage.md#cold-request-numerical-repeatability-investigation-hw86-2026-10-01).
The numerical_entry/trace helpers are opt-in diagnostics; do not make the
fixed-seat eviction or Python-forward instrumentation a serving default.


### Worker-direct streamed ingress

On the qualified hw86 task runtime, use a fresh output path:

```bash
HCCL_DETERMINISTIC=strict BETTERSCALE_MODEL_PATH=/workspace/models/Qwen3.5-35B-A3B bash prototypes/pd-kv-layout/run_pd_model_probe.sh /path/to/fresh-output --native-async --async-export --stream-import --verify-transfer --import-failure-probe
```

--stream-import bypasses controller/Core RPC for dense payloads: workers read
immutable Store chunks into a two-slot pinned ring and scatter H2D into reserved
native pages. Target GDN/conv still travel through RPC. Both workers must drain
before publication or page release; unknown completion quarantines the import.
The exact-transfer oracle plus rank1 post-enqueue failure/retry matrix passes
12 handoffs and 48 shard byte oracles. See the repo knowledge ingress evidence
for the accepted envelope. Remove the two diagnostic flags for the plain arm.
Fresh/evicted GPU residents still need full-history H2D; incremental Store
publication does not imply incremental receiver H2D. This remains a single-host
prototype, not a production placement/HA or bandwidth qualification.


Worker Store connections are now retained across transfers and explicitly closed
at normal actor shutdown. Buffer registrations and DMA lifetime remain per
transfer. Incremental publication validates prior object sizes rather than
reading and discarding all prior payloads. Both optimized oracle and plain
P2/D6 matrices pass; see repo knowledge for before/after phase costs and the
important injected-request timing boundary. GDN/conv still use control RPC.


Add --direct-checkpoint to --native-async --async-export --stream-import to
move target GDN/conv shards directly between workers and Store. The manifest
uses schema2 with explicit tp2-target-shards-v1 format; control RPC carries
descriptors, not model-state payloads. The oracle/failure matrix passes 96 shard
byte checks. For actual quiescent handoff latency, remove --verify-transfer and
--import-failure-probe and add --no-export-activity: the ordinary overlap harness
intentionally includes another request and is not a pure latency benchmark.
The qualified no-activity medians are 0.727 s P->D and 1.462 s D->P, not an SLA.


--dp-finish-sync defaults to32 (unchanged donor). The explicit native-actor
experiment --dp-finish-sync 1 applies finish checks consistently to every D core;
it passes the no-activity matrix with median 0.436 s P->D / 0.654 s D->P.
This does not qualify throughput, busy-pool admission or universal subsecond
latency. The remaining delay is accepted for current PD work; do not make
removing it a blocker or independently put locally idle EP ranks to sleep.

## D-only efficiency observation

Use `bash run_d_cluster_probe.sh /absolute/new/output` on the prepared hw86
task runtime. This runs balanced B1/B8/B16 target-only cohorts and a bounded native
CANN/msprof B8 capture; no P engine is started. Read the
[D6 evidence and interpretation boundaries](../../.agents/skills/repo-knowledge/scenarios/study-qwen35-state-layout/d-cluster-efficiency.md)
before interpreting timings or increasing concurrency. The configured B16 width
now passes after fixing positive-KV native padding at the owned target metadata
boundary; the16-live-request guard remains intact. Use --batches 8 16
--profile-batch 0 for timing without a capture. D uses the isolated
candidate-package-8-fia-padding; its two-file staging recipe is retained in the
knowledge entry.
