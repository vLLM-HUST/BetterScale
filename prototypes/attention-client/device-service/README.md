# Device-driven neural expert service (bounded prototype)

This is the next gate after [the host-controlled four-card reference](../joint/README.md).
It replaces CPU route extraction, host expert batch selection, and pipe completion
messages with device publication, cross-source selection, and completion. It does
**not** replace the published BetterScale worker or modify the installed donor.

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
