# Separated-expert profiles and hw0 evacuation (2026-09-24)

Enter before resuming the expert-sharded EP candidate or interpreting the A4/A6
MTP profiles. Workspace artifact root:
`runs/qwen35-expert-mtp/20260924-profiles`; read its `README.md` first.

Both captures use the qualified layer-placement installed MOD (aac0b7a), not
new EP. A4v2 has four ranks, A6v3 six; fixed120s trigger/8s capture under real
MTP2/SWE C64. TraceLoom bf6fb491's ten AugDB/Perfetto exports and sealed raw
archives were copied locally and byte-verified. `summarize_local.py` reproduces
remote costs from the local databases without a device or hw0 dependency.

Observed collect unions occupy53–63% of A4 windows versus69–76% of A6;
collect includes remote queue/compute/transport and local copy. This is not an
isolated expert-math measurement or matched per-call causal comparison.
Dynamic inputs lack host/sqlite/stream_info.db; recovered graph count0 means
insufficient graph evidence, NOT disabled FULL decode. No calibrated cross-rank
clock. Retain these limits when interpreting the short mixed-workload windows.

hw0 NPU services exited and all task-scoped analysis/transfer processes finished.
A6 capture was sealed before its remaining workload was intentionally cancelled
for administrator shutdown. EP queues were cancelled before worker launch.
Do not resume hw0 work. Old expert placement means disjoint expert IDs (EP),
not matrix TP. The current EP2/EP4 candidate has CPU contracts and clean CANN
builds, but NO hardware qualification or performance result. Source changes
remain experimental in codex/expert-mtp-serving, not a published0.5.1 feature.

Operational lessons: npu-smi process rows include vertical-bar separators;
ownership parsers must accept them. Avoid coupling large exported-profile
transfers to the NPU campaign: per-rank compressed sealed raw data and final
reports transferred successfully. For AugDB cost queries, load canonical TASK
kernel metadata into an event-ID dictionary instead of an unindexed global
anchor/event join (the latter hit the60s query budget).

Further CPU audit: see `BOTTLENECK-NOTES.md` and `server-admission-audit.json`
in that capsule. All six owner lifetime receipts prove one source/wave; this was
an intentional V2Lite optimization, not newly proven Qwen35 root cause. The
512-command rings are shutdown tails (3-token frames), not the8s profile window;
never subtract their timings from profiled client collect. One shared Cube
command stream is non-preemptive despite two-slot movement/compute overlap.

The subsequent MOD batching migration is recorded in
[expert-implementation-lineage.md](expert-implementation-lineage.md); the installed5
profile and cap1 audit remain immutable historical evidence.
