# Reuse DFC data movement, change the service topology

Source inspected: pinned vllm-ascend
`csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_kernel.hpp`.

The original `DispatchAndCombine` invokes mature routing, gathers peer expert
counts, derives expert-major prefixes, and pulls grouped rows from peer memory.
`CombineV2` distributes expert output tiles over cores/subcores; completion is
followed by token unpermute/weighted reduction. Data movement is not one scalar
core writing every expanded route. On the other hand, its cross-rank counts,
shared-memory layout, cross-core AIV→AIC flags, and all-EP-wave synchronization
are real contracts. It is not enough to rename the ranks: independent attention
sources must not be required to enter a common EP wave.

This adapter keeps native grouped GEMM and now calls **native
npu_moe_token_unpermute** for final weighted combine. Narrow IPC movement kernels
adapt the row-ownership and completion pattern to source-specific generations.
No donor source is copied into a differently licensed custom GEMM.

With `DEVICE_SERVICE_PARALLEL=1`:

1. The existing device selector only chooses source descriptors and creates
   expert-major offsets/group counts. Count → prefix → assign is O(E+N*K),
   not a rescan of all routes for every expert. This small control plane stays single-core.
2. `neural_pack`:16 AIV blocks pull disjoint live route rows according to those
   offsets. The following graph node starts GMM only after the pack kernel ends.
3. `neural_scatter`:16 blocks copy only owned output routes. No placeholder zeros
   are written. `neural_complete` then publishes per-source DONE and trace rows.
4. Client publication stays a small single-core task. `neural_collect` waits for
   both exact generations and divides live routes across16 blocks. Each route's
   top-k expert selects its one return owner; it reads no other server's slot.
5. A separate graph node retires the source generation after all pulls complete;
   native token unpermute weights/reduces the canonical token-major route view.

The fixed-layout return allocations still exist, but **unowned slots are neither
written nor transferred**. At32 tokens the client pulls1MiB total rather than2MiB.
Keeping the canonical K-slot order avoids adding a new distributed floating-point
reduction contract. Parallel-mode allocations are poisoned with BF16 NaNs during
bootstrap so an unowned initial read is observable in the full-model shadow.

Graph-node ordering supplies joins between parallel movers and publication;
there is no unsafe core0 DONE before other cores finish. The serial mode remains
the initial oracle. GMM's fixed512-route padding and the single-core selector remain
explicit limitations. This is not claimed to be the complete fused DFC pipeline:
expert-by-expert AIV/AIC overlap, native routing integration, and unbounded service
scheduling are separate changes. Sixteen polling AIV blocks may compete with
future same-device concurrent attention; this prototype does not qualify that
resource partition yet.


## Qualified evolution (same local host, September16)

| Actual rows | Original serial client | Parallel mover + native combine | Plus linear routing map |
| --- | ---: | ---: | ---: |
| 1 | 211.1us | 184.4us | 174.9us |
| 16 | 665.4us | 483.1us | 246.9us |
| 32 | 1089.8us | 463.2us | 292.9us |

Receipts: `timing-result.json`, `parallel-timing-result.json`, and
`parallel-prefix-timing-result.json`. These are unprofiled external client-graph
event brackets, not complete request latency. Final ranges are165.8–177.3us
(16 decode observations),244.0–253.4us and290.4–296.7us (four observations each).
The intermediate16-row range327.8–636.9us is retained; do not report its median as
stable throughput. All24 forward shadows and KV snapshots remain bitwise exact
at both parallel revisions, including poison-initialized unowned return slots.

The intermediate profile `runs/attention-device-joint-20260916T034718Z` exposes
32-row server scatter14.8–15.1us plus completion8.9–9.2us, versus the original
serial completion441.8–443.9us. Parallel pack is34.0–34.5us. These are profiled
kernel medians; queue prepare still includes idle waiting and cannot be charged
entirely to routing. This profile precedes the linear-map change; do not label it
the final kernel revision.

Native local FULL MLP remains119.5/136.8/155.7us for1/16/32 rows. The new separated
path is therefore not yet a local-latency win, and multi-request overlap/throughput
is unmeasured. Original serial used physical0–3, the new candidate uses2–5 while
foreign work may use0–1. No selected-card collision was observed, but these are
not isolated whole-machine or equal-resource performance controls. The exact
ratios are diagnostic observations, not portable acceleration guarantees.

Run the candidate with:

```
DEVICE_SERVICE_PARALLEL=1 DEVICE_SERVICE_TIMING=1 \
  bash prototypes/attention-client/device-service/run_joint.sh 2,3,4,5
```

A failed profiling startup at034559Z was an EADDRINUSE31552 rendezvous collision,
not an arithmetic failure. Independent TP1 engines now receive base ports spaced
by100 (`ATTENTION_JOINT_PORT_BASE`, default41500). Do not force unrelated engines
onto adjacent rendezvous ports when admitting independent device subsets.

Final-revision profile: `runs/attention-device-joint-20260916T034959Z`.
Its `analysis/attention2-expert2-sealed-window.json.gz` is the four-device view;
`analysis/expert-stage-comparison.json` records per-source-generation stage costs.
The final profile again passes all exact forward/KV checks
(`parallel-profile-result.json`).32-row pack is34.1–34.4us; scatter plus completion
is24.1–24.3us. No profiler timings are substituted for the unprofiled table above.
