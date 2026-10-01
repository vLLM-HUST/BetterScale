# PD KV layout: first CPU prototype

Research prototype only. No serving hooks, NPU initialization, model loading,
Store service, new runtime installation or production topology change.

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
