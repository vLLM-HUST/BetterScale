# Resume P4 / D4 two-host PD

Enter for the hw81/hw86 naive dual-host coordinator, native DP4TP2EP8 admission,
or the subsequent 256K qualification. These observations do not qualify MTP,
asynchronous State pipelines, HA, or production throughput.

## Accepted target and current boundary

Fletcher assigned both eight-card machines exclusively, without the old lease
service. hw81 hosts four independent TP2 P instances; hw86 hosts native
DP4/TP2/EP8 D. Preserve idle(core) mutation admission. One session has one active
writer and a deterministic D attention owner. Both sides retain immutable DRAM
checkpoints, not one copy per P instance. No MTP State crosses machines.

The final production-debug envelope must support the model's **262144 context**.
The first wiring experiment deliberately uses 8192, 8GiB State/rank and target-only
generation; passing it is NOT completion of the long-context goal. The current
whole-checkpoint HTTP ceiling (512MiB), msgpack token-array bound, Core/worker
8192 guards, native engine options and request admission all need coherent
long-context qualification, not just a CLI length change. Read long-context.md
for earlier native/owned long-context evidence; it is not a two-host result.

## Hosts and pinned restoration

hw86 container IP10.244.2.32; hw81 IP10.244.1.16. Only these explicit peers were
tested. hw81 was empty and restored from hw86: repo bundle, exact donor source
archives, task-local Python runtime/candidate packages and all14 model shards.
Transfer manifest hashes verified every archive/model file. ModelScope immutable
snapshot identity is not established for these downloaded weights.

Donors remain vLLM752a3a504485790a2e8491cacbb35c137339ad34 and
Ascend9bf964cb4b87c8cd0d6852c41a55b3c29711fa95; CANN9.1.0,
torch2.10.0+cpu / torch_npu2.10.0.post4, Python3.12.13. hw81 donor directories
are exact source snapshots, not initialized Git submodules. Do not substitute
the container's default0.23 source.

The source bundle is shallow: plain clone fails for absent parent5097df.
Restore into a new repo with git init, copy the source .git/shallow, fetch the
bundle branch and check it out. Do not repeatedly retry plain bundle clone.
Bootstrap artifacts/manifest are in each host's betterscale-pd-runtime:
hw86/hw81-bootstrap and hw81/bootstrap. Temporary hw86 HTTP55486 was stopped.

## Bounded network evidence

probe_peer_tcp.py tested hw86 client to hw81 listener55481,64 RTTs and128MiB
each direction, with exact patterned payload checks. Median RTT0.147942ms;
hw86→hw81 0.54637GiB/s, reverse0.80591GiB/s. These Python/TCP observations are
not saturation or physical-link capacity. Listener closed afterward.

Both containers expose eth0 MTU1450, reported10Gb/s. RDMA sysfs reports an active
25Gb/s mlx5 bond, but hw81 has no /dev/infiniband; no usable RDMA claim.
Receipts: hw86 runtime/hw81-network/tcp-client.json and hw81 runtime/network/
tcp-server.json.

## Native D8 admission

candidate-package-10-ep8-state changes only the two EP6 topology admission guards
in qwen35/__init__.py and small_fish_runtime.py to DP4. The first candidate9
failed because the draft-startup guard was still DP3: keep that failure, do not
interpret it as numerical or transport failure. stage_ep8_state_candidate.py
reproduces both changes from qualified candidate-package-8-fia-padding.

ep6-runtime's two narrow uneven-expert donor fixes fall back to native behavior
for EP8. MTP is still allocated/captured but target-only execution remains active.
Do not call this MTP qualification.

hw86-d8-admission2 exited0,1K prompt and128 output tokens/request:
C4/B1 median13.894901ms; C32/B8 18.546955ms; C64/B16 22.2723005ms.
B16 is16 per TP2 owner,64 total, not64 per rank. All requested lengths passed.
Host dispatch periods are not device busy times. Fixed analyzer scope now says
generic native DP/TP2/EP rather than falsely hardcoding EP6 for eight-rank data.
Original failed run: hw86-d8-admission; successful source/receipts/run scripts
are frozen beside hw86-d8-admission2. Devices idle before node services launched.

## Coordinator and failure boundaries

naive_pool_node.py owns local Mooncake DRAM and model actors. Network is
msgpack/HTTP with explicit peer IP allowlists; pickle exists only in trusted
local multiprocessing pipes. This is an isolated experimental network, not
authenticated/TLS public serving. P uses55581; D55586. Local Store master55401.
Do not expose these endpoints publicly.

naive_pd_coordinator.py uses the existing SQLite Directory CAS:
P claims→generates first token→exports→drops→copies checkpoint→publishes to D;
D claims/imports a bounded wave→generates→drains native DP→exports/drops→copies
back→publishes to P. D owner is crc32(session)%4, at most16 requests/owner.
Four independent P workers pull from one queue. Same-session overlap is rejected.
Caller cancellation keeps admitted work/session fenced. An uncertain RPC fails
the whole controller closed, settles all callers and leaves active leases fenced;
there is no speculative revoke, automatic replay or distributed recovery.

The initial real request failed safely: an85,394,559-byte full checkpoint exceeded
the Store client's64MiB local staging buffer (put=-600). The16GiB global DRAM
segment was not exhausted. Cache now writes8MiB content-addressed chunks and
publishes a manifest only after all chunk acknowledgements. Missing/evicted
chunks are cache misses, corruption is failure. Unreferenced chunks may remain
until eviction. An actual CPU Mooncake85MiB repeated-put/get gate passed
(runtime/naive-cache-chunk-gate/receipt.json). This is blocking chunked storage,
not asynchronous device transport.

Initial services hw81-p4-node / hw86-d8-node stopped with exit0; native shutdown
reported4 shared-memory objects and all cards returned idle. Failed request
directory/source/log are hw86-dual-pd*. Updated services use *-node2-source,
preserving the first source and failure rather than editing a live capsule.
The second full two-host integration result is still pending at this entry.

CPU tests test_naive_pd_coordinator.py and test_naive_pool_cache.py cover
single-writer leases, warm/cold fallback, caller cancellation, all-caller failure
settlement, atomic manifest publication and missing chunks. Native owner utility
tests retain exact owner targeting. Full-model token identity across non-strict
HCCL/batch geometries is not the acceptance gate; preserve the accepted numerical
envelope in pd-storage.md and verify exact transfer/lifetime separately.

### Staggered warm admission is a separate target-only boundary

hw86-dual-pd2 passed State import and byte-exact post-H2D re-export before
generation, then failed closed on a native scheduler path not exercised by the
balanced cold D8 capacity run. The pinned Scheduler.schedule waiting branch
pads a new one-token hot-hit request to1+num_spec_tokens when another decode
request is already running. It directly creates scheduled_spec_decode_tokens
[-1,-1], bypassing the old diagnostic's post-schedule clearing of request
proposals. The target-only worker correctly rejected this; subsequent worker
IndexError/peer-close errors are not evidence of checkpoint corruption.

pool_state_entry.Scheduler now declares num_spec_tokens=0 after native startup,
leaving lookahead allocation and drafter startup/buffers intact. Do not simply
remove the worker's speculative-work guard or trim an already-accounted output.
test_pool_state_entry.py preserves this admission/capacity distinction.
Frozen D node3 changes only this scheduler initialization from node2; the joint
probe3 is the next hardware gate. P node2 remains valid and running.

### Preparatory 256K wire/config work, not model qualification

pd_limits.py gives explicit8192 and262144 experiment envelopes. The latter
uses24.25GiB logical State/rank and a6GiB total checkpoint ceiling. Core/worker
imports require cursor<context_limit; request prompt+generation must fit.
msgpack token-array limits and node/controller context handshakes must agree.

A CPU codec gate actually encoded and decoded a5,368,951,007-byte representative
40-plane payload with both msgpack and the native RPC's msgspec codec:
13.48s and9.47s respectively, peakRSS10,641,352KiB for this isolated gate.
Thus >4GiB aggregate payload is supported by these codecs when each plane is
<4GiB. This says nothing about native utility RPC, network throughput, long
model State, or peak concurrent production memory. The naive whole-payload
path still copies/serializes large objects; do not label it O(network time).
Full long-context gates remain required before claiming262144 support.

### Qualified short-context two-host roundtrip (node3 / probe3)

hw86-dual-pd3 exited0 in145.30s with hw81-p4-node2 and hw86-d8-node3.
Eight sessions cover all four P instances and four fixed D owners,16 first-turn
outputs and8 continuation outputs; a ninth session exercises P-only1-token
completion. All8 warm P resumes and16 D resumes report prefix hits.
The24 post-H2D full target State re-export comparisons are byte-exact, and each
final directory key has identical complete checkpoint bytes on P and D.
Every directory row ends owner=P, active=0; the8 two-turn sessions reach epoch4.
D waves actually include1 and7 requests, not only simultaneous balanced batches.

Observed first-turn request latency21.82–64.49s and second-turn24.16–69.96s
includes naive full copies, hashes, diagnostic State re-export/readback and
global-idle wave gating. It is emphatically not a production latency score or
the D step period. Single-token path2.06s. No MTP or task-quality claim.

Both services subsequently stopped with exit0 and8/8 NPUs idle on each host.
Raw source/run/log/directory/summary remain beside hw86-dual-pd3. The active
follow-on is the262144 model gate with24.25GiB/rank State; it is still unqualified.
The EP8 staging helper's two changed Python files compare exactly against
candidate-package-10-ep8-state (ep8-staging-reproduction2).
Fifty affected CPU tests pass at this checkpoint.

## 256K follow-on and next admission frontier

The first full-context configuration uses the same24.25GiB State/rank on both
roles. This is a bounded qualification budget, **not a measurement of maximal
D8 HBM utilization**. Both native pools report2,140,160 pooled token slots per
TP2 attention owner, or8.16×262144 before allocator reservations.
Sources: hw81-p4-256k-source and hw86-d8-256k-source, frozen from af91af4.
The512GiB DRAM segment/host is within the inspected~2TB container memory limit.

With context262144, the1K four-owner two-turn gate passed in103.65s.
The32768-token four-owner two-turn gate passed in566.71s, including12 exact
post-H2D State comparisons and final copies on both sides. This long delay is a
naive full-payload/readback diagnostic cost, not device step timing. Its ordinary
checkpoint is735,908,281 bytes. The262080-token gate and a subsequent exact
262144-total-token warm continuation remain active and unqualified here.
Receipts: runtime/hw86-long-pd-{1024,32768,262080}; the final single-session edge
uses exact-context-boundary.py and hw86-exact-context-boundary.*.

The next source change (not yet loaded into these frozen services) introduces
an idle Core capacity receipt and admission by both execution seats and complete
request KV-page footprint, including retained lookahead allowance. Sixteen seats
must not imply that16×256K fits a24.25GiB owner. CPU fixtures show8 such requests
per1044 free2048-token blocks, while16×100K fits. Deferred requests are packed
into subsequent waves, not serialized accidentally one at a time.

It also adds a loopback-only token-ID frontend, per-operation transport receipts,
a same-directory controller lock, refusal to restart over unfinished ownership,
and idempotent content-addressed Store puts. These changes need a live startup/
frontend gate before promotion.55 affected CPU tests pass; an actual85MiB Store
repeat-put/get gate passed at~0.3s per operation (naive-cache-idempotent-gate).
Do not infer that this isolates the long-service bottleneck. A future optimized
path should reuse the existing worker-direct streamed State protocols rather than
treating repeated multi-GiB Python serialization as the desired architecture.
