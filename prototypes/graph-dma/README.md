# Captured DMA as a composable graph stage

Prototype only: no BetterScale worker changes and no new serving claim.
Tests local H2D, D2H and same-device D2D via `aclrtMemcpyAsync`, not HCCL,
peer HCCS or network RDMA. A memcpy API call does not by itself prove which
physical engine executed it; inspect profiler task types before naming SDMA.

## Questions and evidence gates

1. Capture H2D → local D2D → D2H, then change source contents after drain and
   replay the same graph. Verify all output bytes for three generations.
2. Separate optional `--update` process marks H2D as an ACL capture task group,
   then changes its source address and byte count at quiescence. Check the copied
   half and untouched half independently. This is not concurrent-update safety.
3. Compare BF16 and NZ INT8 GEMM proxies with16/64MiB transfers, alone, serial,
   and event-fork/join overlap. Record both constituent spans and joint completion.
   Three alternating-order repeats,16 repeated operations per captured block;
   event timings are outside capture because this runtime cannot read captured
   timing events reliably. Results are sustained-block costs, not layer latency.

Use `aclrtMallocHost` host backing; keep all source/destination addresses alive
until graphs are destroyed. Drain before host overwrites and before freeing
host storage. Use independent compute/transfer buffers so the contention trial
has no real data dependency; production consumers must wait on actual completion.
Output bytes / same-input GEMM reference are checked outside timed regions.
No shape or value dependence on model weights, no collective initialization.

Launch only through a selected-card health/occupancy admission and home lease.
The current local capsule `runs/graph-dma-local-20260914` waits for that lease,
then device0 only; it never treats an idle member of a leased8-card job as spare.
It waits at most30min for the lease and30min for admission, with10min owned
execution. `TASK_QUEUE_ENABLE=0` avoids mixing raw ACL calls with a Python
background submission queue. That is a fixture choice, not a production setting.
Raw receipts remain outside Git. Existing communication/GEMM evidence lives in
`../kv-prefetch-overlap/README.md`; do not attribute its HCCL measurements to DMA.

## Primary API evidence

Huawei's [capture guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/appdevg/acldevg/aclcppdevg_000519.html)
shows asynchronous H2D/D2H inside capture and requires ACL-allocated pinned host
memory. It describes task-group updates, but the example updates an operator,
not a memcpy: memcpy update remains an experimental question here. The8.5 page
also carries a trial-use/non-commercial warning; do not infer commercial support
for our installed9.0.1 from that older guide. Local9.0.1 `acl/acl_rt.h` exposes
capture task-group/update functions; hardware/parameter support needs execution.

A normal graph replay has no new pointer/length arguments. Stable addresses with
new contents and explicitly updating a captured task are distinct contracts.
Successful quiescent update does not authorize changing an in-flight transfer,
changing topology/direction, or freeing old backing before all users retire.

Next stages if warranted: native task-type timeline; a same-graph stream fork/join
with compute consuming the transferred input; a multi-layer double-buffer
correctness oracle; then admitted two-card peer-copy characterization. Network
RDMA completion/visibility is a separate transport protocol, not established by
local memcpy capture.

## First local observation (September14)

`runs/graph-dma-local-20260914/first/` exits0; 24 cases,288 timed records all pass
output checks. The captured H2D → local D2D → D2H chain passes three different
host-input generations. `local-result.json` retains every case including the
small BF16/16MiB D2H regression (~9%); do not claim universal improvement.

For INT8 Q-B-like M4096/K1024/N32768,16MiB H2D serial1.617ms→overlap0.905ms,
compute slowdown~0.0%;64MiB3.738→2.857ms, compute slowdown~0.3%.
D2H16MiB1.693→0.906ms,64MiB4.059→3.173ms. These are independent-buffer,
sustained16-operation blocks, not per-layer application speedups. There is no
NUMA pinning or multi-device host-bandwidth characterization in this fixture.
Local D2D64MiB slows that compute proxy~8.1%, and the smaller INT8 M512 compute
~32.3%; separate transfer scheduling does not eliminate HBM contention.

The tested full-block host transfers are about22–24GB/s. This contrasts with
prior HCCL/AIV proxy interference, but it is not an equal-bytes backend A/B or
proof of a particular hardware execution engine. Source/destination locality,
physical engines and NUMA behavior require separate characterization.

## Task-group update boundary on this installed runtime

Three independent processes tested H2D, D2H and local D2D as the marked task.
All reject `aclrtMemcpyAsync` *during initial capture inside the task group*,
with507009 / `task not supported`; capture end subsequently reports507903
(capture invalidated). No update/replay was reached. Ordinary capture of those
same directions passed above. Consequently this experiment does NOT support
using this task-group API to make memcpy source/destination/length dynamic on
910B2/CANN9.0.1. It does not prove every newer runtime or alternative API lacks
that capability. Artifacts: `runs/graph-dma-local-20260914/update/` (H2D) and
`update-directions/{d2h,d2d}.log`. The directions wrapper exits0 after recording
both child exit1 statuses; that wrapper status is NOT an update success.

For the currently demonstrated route, retain fixed pinned-host/device buffers,
change their contents only after retirement, and use separately captured fixed
slots/buckets if different addresses or sizes are needed. Consumer events and
reuse credits remain necessary. A graph has no pointer-argument interface merely
because its payload can change. No in-flight mutation or direction/topology
update is qualified. All local experiment processes have exited and the leases
were released. No8-card or hw3 job was launched by this prototype.

For the four-way historical TP/DP target prefill/decode window census, read
[TIMELINE-WINDOWS.zh-CN.md](TIMELINE-WINDOWS.zh-CN.md). `timeline_windows.py`
subtracts existing task overlaps and preserves rank/phase scope; the22GB/s
conversion is a planning estimate, not measured free8-card bandwidth.
