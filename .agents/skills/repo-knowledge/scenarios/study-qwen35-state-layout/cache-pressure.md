# Automatic State cache under session rotation

Enter before tuning backup admission or host capacity, or repeating C16/C32 rotation pressure. This is a bounded serving observation, not a universal speedup claim. The accepted mechanism remains in [cache-policy.md](cache-policy.md).

## Exact envelope — 2026-09-27

- BetterScale `5d1dbfc8849c1002a40d1e23d81f93e52e4608e7`; official `swe-prefix-reuse` client `8bb99ebac120f8907623fc1b5a04946894424d31`, seed20260924. No runtime/kernel change for these timings.
- Qualified hw3 CANN9.0.1 / torch-npu2.10post2 / vLLM0.25.1 / Ascend0.25.1rc1, Qwen3.5-35B-A3B BF16, TP2/MTP2/FULL/native async. Original qualified native libraries, NOT the separate hw180 rebuild.
- All arms: E16/R20, context262144, query4096, State26,038,239,232 bytes/rank. Resident declaration1,912,095,920 bytes; remainder funds16,720 shared128-token FA pages. All16 rank-budget receipts agree.
- Policy ON: host8GiB/rank, watermark0.7. OFF: no automatic host cache. Same State layout and numerical path.
- D means C×D independent session slots, round-robin within each client lane, one in-flight request per lane, fresh salt per play. C32 is OFFERED client concurrency, NOT E32 execution.
- Two colocated TP2 groups, C16D1/D2 then C32D1/D2. Other four cards carried foreign work. Selected-card leases,30s admission and continuing foreign-owner guard; two48-CPU affinities. Cross-group or cross-host causal comparisons are not qualified.
- Fresh server per arm; C2D1/60s qualification, then900s fixed window with drain separate. Group0/3 OFF→ON, group1/2 ON→OFF. One predeclared run/arm, no selection by throughput, no profiler, no public leaderboard publication.

## Observed matched results

Output throughput is tokens/s/chip. TTFT P95 is seconds. Cache is cached prompt tokens / total prompt tokens, including drained requests.

| Offered C / D | OFF → ON throughput | Change | OFF → ON TTFT P95 | OFF → ON prompt-cache fraction |
|---|---:|---:|---:|---:|
| C16 / D1 | 417.85 → 416.74 | -0.27% | 0.591 → 0.628 | 97.55% → 97.54% |
| C16 / D2 | 206.58 → 310.96 | +50.53% | 4.606 → 3.137 | 15.99% → 72.94% |
| C32 / D1 | 177.88 → 406.70 | +128.64% | 37.492 → 27.370 | 0.00% → 93.76% |
| C32 / D2 | 251.56 → 275.40 | +9.48% | 41.037 → 44.549 | 0.00% → 34.43% |

All eight official summaries are valid, zero failed requests, all session slots revisit a continuation. No native running-request preemptions. Sampled shared-page usage peaks at50.38% across the campaign, so these misses are not evidence of exhausted FA capacity.

| Policy arm | Store / load / drop completions | Median host entries while ≥90% full |
|---|---:|---:|
| C16 D1 |765 /0 /756 |17 |
| C16 D2 |957 /587 /941 |21 |
| C32 D1 |1128 /978 /1112 |19 |
| C32 D2 |813 /408 /792 |28 |

Counters are scalar samples across deployment lifetime, including qualification/drain, not exactly the900s window. Host entries are checkpoints, NOT distinct sessions. Sampled host byte charge stays within8GiB/rank; simultaneous pending restore/drop actions can exceed one even though background backup issuance is serialized.

## Interpretation and counterexamples

- C16D1 already retains every continuation on device. It performs765 backups, about398.9GB payload/rank, with zero host loads; throughput is nearly unchanged in this one sample, but TTFT P95 rises6.24%. This demonstrates redundant backup work, NOT a proof that arbitrary extra copying is free.
- C16D2 and C32D1 show material reuse recovery: actual host loads coincide with higher cache fractions and reduced prefill recomputation. Do not describe the gain as faster kernels or expanded page capacity.
- C32D2 is an important retained counterexample: throughput+9.48%, but TTFT P95+8.56%. Its64-session working set still has many misses and host-LRU turnover. The observed host occupancy is consistent with insufficient retained host coverage; a larger-host matched run has NOT established the cause or a fix for tail latency.
- Async stress has protocol/liveness and exact-token transport acceptance. It does not compare every long restored generation numerically against cold recomputation; the separate native policy gate owns its bounded token-equality claim.

## Evidence, teardown and replay

Controller capsule: `runs/qwen35-state-lanes/20260927-hw3-cache-pressure/`; remote same relative path beneath `/workspace/my-ascend-workspace/`. Local `evidence/attempt1/comparison.json`, eight official summaries/requests, rank budgets, scalar telemetry and `evidence/admission1/` retain the raw basis. `protocol.json`, `run_group.py`, `run_matrix.py`, `pressure_observer.py`, `env.sh`, `launch.sh` and `analyze.py` preserve the bounded harness. Re-enter current host admission; historical device choices are not future allocations.

All eight server exit codes and campaign exit are0; selected cards return to baseline and leases are released. Every server logs a shared-memory resource_tracker cleanup warning; four log output-handler EngineDeadError AFTER requested shutdown. Preserve these teardown warnings separately from zero request failures; do not claim warning-free teardown.

The abandoned hw180 attempt failed before model execution because a telemetry Scheduler subclass violated the strict canonical scheduler-class admission. The corrected harness leaves that admission intact and wraps scalar observation through the native `vllm.general_plugins` lifecycle. Never weaken source/configuration gates just to observe a run.

Fletcher redirected the campaign to the qualified hw3 environment while the hw180 CPU source build continued. Its queued automatic NPU qualification was explicitly cancelled. These pressure results do not qualify or complete that separate rebuild.
