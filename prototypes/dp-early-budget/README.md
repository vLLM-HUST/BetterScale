# Early DP budget agreement — experimental, not shipped

The retained optimized run120 trace still has a 9.480ms rank0 compute/comm
coverage gap. Its small CPU metadata all-reduce waits for rank1 arriving about
9.37ms later. A metadata graph and double input banks do not by themselves move
this rank rendezvous off the critical path.

This prototype retains **each native DP scheduler**, the same scheduled requests,
K5 budgets and native device-derived progress. There is no central scheduler,
acceptance prediction, speculative extra invocation or new device graph.

## Exact seam

Use native `MultiprocExecutor` so an EngineCore can prepare its next scheduled
packet while its worker is still submitting the previous eager draft. At
`EarlyExecutor.execute_model(SchedulerOutput)`, before the worker RPC, agree a
small CPU packet among the existing independent EngineCores. Send the completed
immutable decision **with that same SchedulerOutput** through native RPC.

`EarlyWorker` consumes this decision at the original
`_sync_metadata_across_dp` call after native preparation. It checks the actual
local bucket and mode before using the agreed maximum padding. Draft continues
to use its original coordination. A disagreement fails closed; a rank must not
silently enter native all-reduce after its peers have skipped it.

The agreement deliberately waits in EngineCore before publishing the RPC.
That wait may overlap PREVIOUS device work because the worker is independent;
this is not a claim of an asynchronous EngineCore event loop. Merely changing
`async_op=True` at the original late site would not establish this overlap.
Measure actual readiness/launch lead and the device timeline before retaining
this design. A parent whose next schedule is itself late may still expose a gap.

## Bounded admission

- Single machine, TP1/DP8/EP8, existing qualified K5 two-seat configuration.
- Local target proposals use only captured FULL buckets6/12 queried after READY.
- Same request set, six queries/request, five scheduled draft slots, no new,
  resumed, finished, preempted, encoder or structured-output request.
- Any rank declining, including an empty/dummy rank, makes the WHOLE group keep
  its original worker coordination. Prefill and turnover remain native.
- Zero-token execute RPC is NOT an actual forward in this non-external-launcher
  configuration. It must not consume a coordination ordinal; the later native
  dummy forward does participate. Repeated/mismatched ordinals fail closed.
- No skipping active/empty ranks, early KV release, new cancellation semantics,
  numerical acceptance host reads, dynamic graph capture or graph-size inflation.

This adds one early CPU exchange on native-fallback waves. That cost and the
additional native MP process topology must be included in the final decision.

## Experiment entry and comparison

Put this directory on the test runtime's Python path. The CANN and pinned donor
runtime remain the existing environment, not something this prototype installs.
Candidate adds native options:

```
--distributed-executor-backend early_executor.EarlyExecutor
--worker-cls early_worker.EarlyWorker --async-scheduling
```

The **same MP** control uses `--distributed-executor-backend mp` and the same
package's `strengthen_dsv4.worker.Worker`. It is NOT a stock-donor comparison,
and an older UniProc run is not a causal control for this patch.

`EARLY_BUDGET_PORT` names one explicit loopback TCPStore port for this bounded
single-host experiment, separate from all donor process groups; donor config's
port sequence is untouched. `EARLY_BUDGET_OUTPUT` is an existing artifact
directory. Its optional per-wave JSONL receipts use host epoch nanoseconds and
record parent agreement, worker entry, consume and host forward bounds. These
receipts are diagnostic overhead, NOT device execution timestamps. Disable
receipt writes for final throughput timings. The environment knobs and custom
executor are prototype-only, not a change to the delivered worker-only entry.

## Verification and current boundary

- Thirteen CPU protocol tests pass (stdlib `unittest` discovery here).
- `cpu_transport_probe.py --port PORT --output NEW_DIRECTORY` passes the actual
  executor's two-rank native Gloo exchange, heterogeneous6/12 padding, dummy,
  request changes, recovery and RPC handoff. It imports the pinned donor runtime
  but uses no NPU/model. Completed capsule:
  `/workspace/strengthen-dsv4/runs/tp-continuation-20260914/early-budget-cpu-transport`.
- The imported native executor/worker classes resolve in the pinned runtime.
- Local142's FINAL lease/admission found foreign occupancy and stopped before
  any model process launched. Its `admission.txt` is preserved. The earlier idle
  sample was not permission to bypass the changed state.
- Full-model/graph/state correctness, actual earlier device submission and
  same-host MP throughput/quality are **not yet qualified**. Do not enable this
  in the shipped Worker, claim the bubble removed, or merge it into defaults.

### Hardware pause, September14 ~05:22 UTC

hw3143/144/145 were also rejected at final admission (rank1 >4GiB), before
model launch. Fletcher confirmed another task starts/probes intermittently,
then stopped retries after it started. Ordinary SSH and the complete probe
environment each also produced healthy idle samples between those transitions;
no environment-specific npu-smi discrepancy was established. No owned model
process remains and the admission leases were released. Resume only on a fresh
window; use a NEW capsule name, starting with the short full-model gate, then
same-host MP control/profile. Scripts and immutable candidate closure are on
hw3 under `runs/tp-continuation-20260914/early-budget-v1` and
`run-hw3-early-budget.sh`; do not treat any of143–145 as NPU qualification.

### Local resume, September14 ~06:53 UTC

Fletcher reported the local eight-card window available. New capsule
`146-local-dp8-early-budget-smoke` passed the home-lock and final eight-card
health/process/HBM admission, then began HTTP server initialization. Before
model qualification, foreign processes appeared on devices0/2/4/6; the existing
supervisor stopped only its owned process group and released the lease.
`admission.txt`, `foreign.txt`, `release.txt` and `146-launch.log` preserve the
transition under the local `runs/tp-continuation-20260914/` root. No numerical,
throughput or early-budget device result was obtained;146 must not be counted
as a passing hardware gate. This was post-admission contention, unlike142's
pre-launch rejection. Do not relaunch while those foreign workers remain.

## Two-card Qwen route (September14, in progress)

Fletcher redirected scarce eight-card work to the existing shared
`/data/shared_models/Qwen3-30B-A3B` BF16 model, TP1/DP2/EP2. All16 safetensors
shards and tokenizer are present. Tensor headers give54GiB expert weights plus
2.8705GiB replicated weights: ideal EP2 weight payload29.8705GiB/rank, excluding
runtime, format conversion, KV and graph workspace. No eight-card launch is
needed for this route; admit only the selected pair and retain same-pair controls.

`qwen_worker.QwenControlWorker` uses the native Ascend Worker with the pinned
runtime check; it installs NONE of the DeepSeek execution patches.
`QwenEarlyWorker` adds the same early-budget consumer. The EngineCore executor
now derives query width from the admitted native configuration: plain decode1,
existing DSparkK5 decode6. The default pure protocol API keeps its old K5 scope.
Known FULL bucket capacities, not arbitrary incoming token counts, gate admission.
A one-token prefill tail must have native cached-request output progress before
plain decode admission; unknown/mixed/turnover/dummy waves remain native globally.

`EARLY_BUDGET_ORACLE=1` checks every admitted early decision against the original
late native exchange before the unchanged model forward. It adds synchronization
and is diagnostic only, never timing evidence. Host per-wave JSONL receipt writes
must also be disabled for uninstrumented throughput controls/candidates.

Local capsules and source closures:
`/workspace/strengthen-dsv4/runs/qwen-dp2-early-budget-20260914/`.
The selected-subset launcher now polls under the existing home lease (2s, bounded
30min), then launches immediately; a CPU rejection exercise confirmed no Popen
on busy cards.001 was cancelled while waiting for occupied pair1,3;002 resumes
on pair0,2. No result is qualified merely by model selection or startup.

This isolates the distributed scheduling protocol. It does not qualify
DeepSeek acceptance/shadow metadata, prove removal of an eight-rank tail, or
promise that plain-decode benefits match a speculative workload.

### Qwen native wrapper and budget oracle

Run003 failed before model execution because native `WorkerWrapperBase` owns
`execute_model(SchedulerOutput)` with a fixed signature: a keyword added to the
underlying Worker cannot pass through it. `early_rpc.execute_with_budget` uses
native MP's callable RPC seam, preserves the wrapper's `_apply_mm_cache` before
calling the Worker, and passes the same immutable decision. No installed donor
file is edited. The CPU seam test asserts cache-before-model and object identity.

Run004 (same local pair0,2, real BF16 weights) completed all three synthetic
HTTP cohorts. Every admitted wave matched the unchanged native coordination
oracle:656 checks on each rank. Host receipts show median0.783ms agreement,
12.407ms lead to worker entry and17.437ms lead to host forward. These are NOT
device bubble savings, throughput qualification or target/KV numerical checks.
The oracle deliberately executes the extra native exchange; do not time it as
an optimization. All18 protocol/wrapper CPU tests pass.

Optional `EARLY_BUDGET_PROFILE=EXISTING_ARTIFACT_DIRECTORY` installs the same
post-warmup observer in both Qwen workers:8 waiting forwards,1 warmup,16 active.
It records native CPU/NPU data without worker-side parsing, then stops; later
forward wrappers stay intact. Profiled cohorts are diagnostic, not throughput
controls. Parse offline with the existing `full-mixed/profile_tools` helpers.

### Two-rank timelines and current performance boundary

Runs005(native MP) /006(early budget) captured the same local pair0,2, two
requests/rank, ordinary FULL decode, real Qwen weights. Native profiler output
was parsed offline in separate processes and analyzed with TraceLoom37323af.
Both compressed aligned exports are under each capsule's
`engine/analysis/qwen-dp2-*-aligned.json.gz` (about4.8MB each).
The short pure-FULL window had fewer than20 eager-only markers;005's failed
attempt is retained in `005-align.log`. Unique all-provider collective
identities pass the unchanged50us holdout gate: P95 native0.420us,
candidate0.826us. These are candidate display alignments, not physical clocks.

`inspect_profile.py` uses first/final captured norm identities on the model
stream and discards incomplete cycles. Do NOT use the first observed norm:
profiling can begin halfway through the preceding replay, producing a false
~12ms body gap. Here capture task IDs18 and1778 bound complete bodies on both
ranks. Fifteen complete same-rank cycles give:

| Profiled median | Native rank0 / rank1 | Candidate rank0 / rank1 |
|---|---:|---:|
| Transformer body |17.263 /17.258ms|17.246 /17.170ms|
| Start-to-start cadence |18.731 /18.798ms|19.104 /19.053ms|
| Between transformer bodies |1.461 /1.477ms|1.820 /1.828ms|
| No observed compute/comm coverage between bodies |0.720 /0.742ms|1.089 /1.079ms|

The body gap includes logits/sampling/metadata, not just idle time. The native
Qwen ordinary-decode workload does NOT reproduce DeepSeek's ~10ms gap. This
profile shows no improvement; candidate host receipts also add diagnostic cost.
Do not infer eight-rank speculative behavior from the small model.

Run007 attempted three-repeat, receipt/oracle/profiler-disabled candidate
throughput. Foreign occupancy appeared on a selected device after admission;
the supervisor stopped only its owned process group. `foreign.txt` and
`release.txt` retain the evidence.007 is rejected, not a performance result.
No subsequent baseline was launched into that occupied window. Uninstrumented
paired throughput and DeepSeek speculative validation remain unqualified.
