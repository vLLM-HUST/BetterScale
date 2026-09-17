# Qwen3.8 topology experiment: TP1 and E3 pay off in this prototype

**2026-09-17, hw0, eight Ascend910B2 cards per configuration.** All five
configurations completed the same retained-prefix workload. This is an
owned-runtime topology experiment, **not an unmodified vLLM benchmark**.

## Result

TP1x5 attention + E3 is the fastest tested configuration: **66.13 output tok/s**,
versus55.92 for TP1x4+E4 and32.86 for colocated TP1x8/EP8. Its throughput is
18.2% above TP1x4+E4 and2.01x the colocated TP1 control in this single campaign.
E3 also increases tested history capacity by24.6% versus TP1x4+E4. It does **not**
beat either colocated topology's whole-machine history capacity.

| Eight-card topology | Elapsed s | Output tok/s | TTFT P95 s | Output interval P99 s | Maximum interval s |
|---|---:|---:|---:|---:|---:|
| TP2x2 + E4 | 702.87 | 20.43 | 109.21 | 43.11 | 96.27 |
| TP2x4 / EP8 | 724.01 | 19.84 | 70.58 | 54.38 | 62.02 |
| TP1x4 + E4 | 256.82 | 55.92 | 51.14 | 12.16 | 46.68 |
| **TP1x5 + E3** | **217.19** | **66.13** | **40.98** | **9.52** | **36.11** |
| TP1x8 / EP8 | 437.09 | 32.86 | 42.16 | 31.69 | 36.45 |

Elapsed is maximum source duration after a common warmup rendezvous, not a
precisely timestamped global makespan. A single campaign does not establish
repeatability/confidence intervals. These tails are still poor serving quality;
a topology win is not production readiness.

### Exactly what was compared

- Real48 target layers + BF16 MTP layer, K1, repaired Eco-Tech W8A8 checkpoint.
  See [HW0.md](HW0.md), [MTP.md](MTP.md) and the checkpoint boundary below.
-40 distinct Open-SWE-Traces sessions, seed20260917, first **two complete turns**
  each:80 turns,236,306 new-prefill rows including continuation anchors,
  14,362 committed outputs,193,043 reused-prefix tokens. No copied sessions,
  output caps, or synthetic cache hits. Own generated history is retained;
  tool observations are replayed, not executed. The harness checks per-session
  work accounting matches across all five cases.
- This is **not** all1,830 turns in the selected40 full trajectories. Selected
  two-turn maximum encoded horizon is9,128, so this does not saturate the
  multi-million-token capacity measured separately.
- FULL target/MTP decode; prefill is eager. All use the affine-head e QSA
  implementation, bounded128-query workspace and the same PLE lookup fix.
- Each source has a1024-row prefill bucket. Therefore aggregate available
  prefill rows differ:2048/4096/4096/5120/8192 in table order. Request seats
  per source are20/10/10/8/5. These are whole-topology results, not a matched
  per-kernel benchmark.
- Service State budgets/rank are50/35/47/46/31GiB respectively: one GiB below
  each highest passed fit below. No model-load/capture time is charged to the
  steady trace interval.

The colocated control uses native A2 MC2 Dispatch/Combine + NZ grouped GEMMs,
with valid-prefix compaction, but our prototype globally coordinates prefill
versus decode phases instead of using a mature mixed scheduler. The separated
clients can progress through different layers/phases independently. Consequently
**the2.01x result must not be advertised as2x faster than stock vLLM, or as an
expert-GEMM-only gain**.

Cross-source coalescing is modest, not the entire explanation. Whole-process
expert-server calls/wave are about1.001 for TP2x2+E4,1.022 for TP1x4+E4 and1.026
for TP1x5+E3. These counters include warmup. More independent attention sources,
prefill allocation and asynchronous phase progress all change with the topology.

## Capacity is a separate result

Full real model + MTP, all State touched, long-prefix prefill and FULL decode
completed using **zero synthetic historical KV**, not a language-quality test
or a replay of that much actual history. All fits use40 resident requests.

| Topology | Highest passed State GiB / attention rank | Allocated history tokens / machine | Exercised prefix / request | Minimum sampled free MiB |
|---|---:|---:|---:|---:|
| TP2x2 + E4 | 51 | 7,082,752 | 176,909 | 290 |
| TP2x4 / EP8 | 36 | 10,271,232 | 256,602 | 380 |
| TP1x4 + E4 | 48 | 6,828,544 | 170,522 | 394 |
| TP1x5 + E3 | 47 | 8,510,080 | 212,544 | 138 |
| TP1x8 / EP8 | 32 | 9,331,200 | 233,012 | 187 |

These are tested fits, not exhaustive global optima or production-safe budgets.
The configured single-request context limit is262,144; the C40 allocation and
pressure prefix are different quantities. Do not count TP head shards as
independent logical token capacity. Independent expert-card spare HBM is not
implicitly usable by attention State. This is why removing routed weights from
attention alone does not guarantee the largest whole-machine KV capacity.

Capacity receipts retain their exact c/d/e overlay identities. Their tensor
allocation geometry is unchanged by the subsequent affine-store lowering
repair; the table does not silently relabel those earlier extreme-fit gates as
new e production tests. All service results above use e with the explicit
one-GiB margin. TP1-EP8 State33 fails with native device OOM;31 and32 pass.

## Implemented and qualified

- TP1 ownership of both KV heads and24 Q heads, including publication,
  head-major gather and FIA; FULL graph independent oracles pass10 cases.
- E3 contiguous171/171/170 expert ownership; one padded dummy expert slot in
  the last owner, never a valid routed expert. Multi-source ABI supports four
  and five attention clients independently of the two staging slots.
- Target W8A8 and BF16 MTP share the server with layer/generation/reclamation
  contracts intact. Short full-model shadow gates and all five retained traces
  pass their explicit checks.
- Native MC2 masks are compacted to the documented true-prefix form on device
  and inverted after combine. A2 omits A3/A5-only TP arguments. The isolated
  real expert eager/FULL gate covers up to1020 rows/source.
- Bounded QSA gather no longer uses expensive vector-scatter stores for either
  padded tiles or active TP1 heads. The active two-head leaf drops from~371ms
  to~0.915ms; this is a regression repair, not an expert-server claim.

The Eco-Tech PLE integer metadata repair uses exact matching original integer
buffers; the untouched checkpoint is not qualified by these results. Numeric
shadow/transport checks do not replace independent end-to-end language quality.

## Evidence and reproduction

- [Machine-readable service matrix](topology-swe-result.json)
- [Machine-readable capacity matrix](topology-capacity-result.json)
- [Gates and chronological failure evidence](TOPOLOGY-CAMPAIGN.md)
- [Active gather paired samples](qsa-active-result.json)
- `run_topology_case.sh` selects layout/mode/State/bucket/immutable overlay.
  `analyze_topology_matrix.py` rejects failed cases or unequal per-session work.
- hw0: `/workspace/betterscale-hw0/runs/topology-stable-20260917/`.
- Local source/role receipt archive:
  `/workspace/betterscale-confluence/runs/qwen38-hw0-results-20260917/topology-receipts.tar.gz`.

Failed runs were retained, not discarded into a clean-only story: earlier QSA
layout regressions, native input-contract gaps, the CPython timed traceback
thread SIGSEGV and State33 device OOM are recorded in the campaign. The timed
traceback callback was removed; all five replacement traces complete. Published
Worker defaults and package releases are unchanged.
