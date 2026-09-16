# Device-driven neural expert service (bounded prototype)

This is the next gate after [the host-controlled four-card reference](../joint/README.md).
It replaces CPU route extraction, host expert batch selection, and pipe completion
messages with device publication, cross-source selection, and completion. It does
**not** replace the published BetterScale worker or modify the installed donor.

For the newer opt-in coalescing/live-row path and its performance limits, see
[Actual counts](ACTUAL-COUNTS.md). The original native path below remains default.

## The small route: reuse the mature GEMM

The pinned donor's
`csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_kernel.hpp`
shows that device-produced expert counts can drive AIC GEMM (AIV publishes counts;
AIC waits on a cross-core event and uses `currentM`). We do not copy that licensed
implementation or write another matrix multiply. Instead, the existing
`torch_npu.npu_grouped_matmul` reads a changing **device int64 group_list** during
replay. That is sufficient for a bounded service graph:

```
queue prepare / compact pack / group_list
    → native BF16 grouped matmul → native SwiGLU → native grouped matmul
    → scatter expert slots / publish DONE
```

One captured graph contains a finite sequence of these cycles. A single host
replay grants the entire episode. Which source is ready, its layer, its expert IDs,
and its row counts remain device decisions. `(layer, local expert)` becomes the
GMM group index, so layers need no host dispatch either. The queue kernel exits
before GEMM executes: no permanently resident AIV poller competes with the server's
native operators. Attention clients wait on different devices.

**This is not an infinite persistent service.** The fixture admits a known task
count and captures enough cycles for the worst case of one source per cycle.
Inactive tail cycles still run padded native math. Graph size scales with this
bounded budget. Production replenishment, termination, cancellation, fairness and
an unbounded queue are not qualified here. The original integer server's decode
priority policy is not implemented by this neural adapter.

## Ownership and execution

Each attention client owns one source frame and one device generation counter.
It publishes hidden rows and native top-k IDs, then READY. Both expert servers
pull independently. A server accepts either source or both, packs by expert, and
runs its half of the expert weights. Each server writes its own return allocation:
`[rows, topk, hidden]`, zero in slots it does not own. It publishes DONE only after
all selected payloads are written. The client waits for **both exact generations**,
pulls both contributions, and performs the weighted token reduction using captured
native tensor operators. Only then can its frame be reused.

Wire flags/descriptors occupy separate 32-byte regions. Transfer completion and
publication use the already exercised ACL IPC / Ascend C ordering pattern from
workspace `pull-expert-server` commit `3532418`; that server is unchanged. IPC keys
are bootstrap capabilities and are neither logged nor checked in. The source
allocation, both result allocations, weights and graphs outlive the episode and
are unmapped only after both clients and servers drain.

- `kernel.cpp`: bounded queue/select/pack/scatter/client kernels; no custom GEMM.
- `probe.py`: three-card primitive, immutable plans consumed by persistent device
  clients; one bounded server replay. Independent eager expert oracle.
- `device_joint.py`: real neural adapter, device gate/top-k and completion banks,
  native GMM service graph. Host still submits attention-layer work and polls a
  local event to resume its coroutine; this is **not** a deviceized whole scheduler.
- `device_worker.py`: thin native worker entry (import after platform initialization).
- `../joint/probe.py`: shared four-card oracle, optional backend selected only by
  the experimental launcher. The original host backend remains its default.

The server packs at most `2 × 32 × 8 = 512` routes. Unused capacity is zeroed and
assigned to the last expert group. This keeps native GMM extents static, at the
cost of padding compute. One AIV core currently performs pack/scatter. Neither
choice is offered as the final performance implementation.

## Qualified result (2026-09-16, local 910B2)

See [compact receipt](result.json). Raw runs remain in the recorded workspace paths.

1. **Three cards:** two episodes, 16 tasks per source per episode, **64 tasks**.
   Inputs, counts (1–8), layer and expert IDs change without recapture, including
   the last expert in both layers. All outputs equal the independent eager BF16
   oracle bitwise; unused output padding and guards remain intact. Device traces
   show cross-source coalescing in both episodes. Runtime submits one graph per
   rank per episode, not a host loop of service replays.
2. **Four cards, Attention2 + Expert2:** full Qwen3-30B-A3B layer dimensions
   (H2048, M768, E128, K8, Q32/KV4/head128), **two dummy BF16 layers**, not the entire
   real-weight model. Each client completes 12 forwards at rows1/16/32 and24 layer
   replays. Outputs and KV match the native model **exactly**, including six later
   forwards after sealing the six attention banks. The shadow restores pre-forward
   KV before candidate execution; reference KV writes cannot mask missing writes.
3. Each expert server completes48 source-layer jobs in one48-cycle graph replay,
   sees generations1–24 from each source exactly once, and observes one combined
   cross-source wave in this run. This is **not** evidence of high batching
   efficiency; the host native shadow/capture fixture leaves long source gaps.

No latency or end-to-end throughput improvement is claimed. Dummy routing need
not represent real trained routing. Dedicated malformed-descriptor/cancellation
coverage belongs to the prior integer protocol, not this new BF16 integration.

## Reproduce in the current lab runtime

```
bash prototypes/attention-client/device-service/build.sh
bash prototypes/attention-client/device-service/run.sh 0,1,2
bash prototypes/attention-client/device-service/run_joint.sh 0,1,2,3
python3 prototypes/attention-client/device-service/summarize.py \
  <three-card-run>/run/measurements <four-card-run>/run/measurements
```

Launchers freeze the Python sources and binaries, then use the existing subset
lease/occupancy supervisor. No eight-card window or real weights are required.
They rely on the workspace's pinned donor environment, IPC helper, CANN9.0.1 and
Qwen model configuration; this is a lab acceptance capsule, not an installable
public service. Queue waits, kernel execution and the outer process supervisor
are bounded. A timeout fails the episode; it is not a retry or recovery protocol.

## Four-device timeline

Set `DEVICE_SERVICE_PROFILE=1` for `run_joint.sh`. Collection stops after the
IPC drain; parse/export offline using the pinned runtime Python:

```
python prototypes/attention-client/device-service/profile_export.py <capsule>
python prototypes/attention-client/device-service/profile_window.py <capsule>
```

The 20260916T032439Z capsule passes the same exact output/KV acceptance under
profiling. `analysis/attention2-expert2-sealed-window.json.gz` is the compact
161.532ms generations13–24 view; `attention2-expert2-provider-clock.json.gz` keeps
the full recording. Both use TraceLoom37323af's exported event identities and
source timestamps. The original first-event-normalized export is retained too.
Roles0/1 are attention clients; roles2/3 are expert shards. Provider rank IDs are
not distributed service roles: each native attention process is TP1 rank0.

There are no matching HCCL collectives in this point-to-point protocol. Do not
fabricate a collective-end clock fit. The provider-clock view restores native
profiler timestamps rather than independently translating each device's first
event to zero. Profiler realtime-minus-monotonic mappings differ by5.44us in this
run. This is **not** a measured bound on device-to-device clock error. Original
source timestamps and mapping receipts remain available; avoid microsecond-scale
causal claims. Window cropping marks truncated intervals explicitly.

In the sealed window each server's `neural_prepare` occupies about154.17ms,
**including device polling for source work**, whereas its native GMM slices sum
to about1.62ms and `neural_complete` to3.04ms. Do not call that154ms packing cost.
Attention-side GroupedMatmul belongs to the independent native correctness oracle;
clones, comparisons and host coroutine polling also remain. Each client's12
`neural_client` invocations total about4ms, including wait and transfer. These
profiled diagnostic sums are not latency/speedup claims. Use a separate
oracle-free load fixture before judging service throughput or batching policy.

For the first native FULL-graph versus separated-expert stage comparison, read
[TIMING.md](TIMING.md). The current remote path is slower, especially at16/32 rows;
scalar return scatter/zero fill, rather than GMM alone, is a measured bottleneck.
Do not reuse the eager oracle's host gaps to claim a graph-to-graph speedup.


The next opt-in candidate is documented in [DFC-ADAPTER.md](DFC-ADAPTER.md):
parallel owned-route movement, native unpermute, no zero return slots, and linear
routing metadata. Original serial results above remain the oracle, not the current
candidate's performance. Neither mode changes the released worker default.

Fletcher explicitly authorized idle-subset use on2026-09-16 without waiting for
the global machine lease. These prototype launchers now use `admit_subset.py`:
ordered per-device locks plus fresh health/occupancy admission and foreign-owner
monitoring. They never terminate another task. A rejection test on occupiedcard0
confirmed the workload launch marker was not created. The old queued global-lease
job was cancelled before any model launched; no duplicate watcher remains.

For the EP2 fused-DFC target rather than TP1 local MLP, read
[DFC-COMPARISON.md](DFC-COMPARISON.md). Matching synthetic route/input controls show
conditional parity: concentrated routing favors the remote stage; broad128-expert
routing at16/32 tokens is still46–52% slower. The910B DFC provider is a lab A2 port
with a different BF16 ABI, not the stock pinned donor operator package.
