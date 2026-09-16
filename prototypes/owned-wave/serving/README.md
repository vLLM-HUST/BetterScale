> Current candidate: `OWNED_HOST_FIA=1` reuses the installed native FIA host
> planner once per wave and shares banked GM tiling across layers (including FD).
> See [host attention metadata](../fia-plan/HOST-METADATA.zh-CN.md).
> `OWNED_HOST_FIA=0` selects the older frozen non-FD control described below;
> `OWNED_STATIC_FIA=0` retains the original per-layer update rollback.

# Concurrent original-trace replay with native APC and owned N+2

This is the successor to the fixed32-token gate. It admits multiple resident
requests, replays variable-length prompts without truncating their actual tokens,
batches decode, reuses prefix blocks, and defers block release across in-flight
old generations. It is a direct execution/scheduling benchmark, not an HTTP
service or SWE-bench task-solving evaluation.

**Historical frozen non-FD caveat:** that static-FIA candidate regressed in the
full SWE run; [matched TraceLoom diagnosis](../fia-plan/REGRESSION-30B.zh-CN.md)
locates slower attention and a missing FD plan path. Integration is not a30B
performance-release claim.

The candidate now defaults to owned static FIA (`OWNED_STATIC_FIA=1`) and
candidate-only execution (`BENCH_ARM=owned`). It retains native numerical kernels
but removes per-layer host attention task updates. See
[`../fia-plan/README.md`](../fia-plan/README.md) for the pinned ABI and safeguards.
`OWNED_STATIC_FIA=0` is an explicit startup rollback to the historical path;
unsupported static-plan variants fail initialization, never silently fall back
mid-capture or claim host-free execution while using task updates. This is the
owned candidate, not the separate published DSV4 Worker/PyPI entry.

## Workload and comparison contract

prepare_trace.py reuses the locally pinned NVIDIA Open-SWE-Traces snapshot
fb0c0dccc7a5cce79b3f6de891848acdede36685 (CC-BY-4.0). It selects WHOLE recorded
trajectories, renders the original messages/tool definitions with the local
Qwen3-30B-A3B chat template, and records rejection reasons. The resulting v2
fixture contains four sessions,44 calls and12,111 output tokens. No task/tool
code is executed. The first v1 preparation rejection is retained; independently
BPE-tokenizing the template suffix avoids assuming prefix tokenization is
invariant under appending text.

The fixture has no arrival timestamps. Initially one request per session is
ready; its next recorded turn becomes ready at completion. Tool delay is zero.
Generated replies do NOT replace the original recorded history. Output lengths
are the Qwen-tokenized complete assistant serialization suffix; ignore_eos
makes the numerical work budget equal in both arms. This is not a model-quality
score, reconstructed production timing, or a representative population claim.

Both arms use the real48-layer BF16 Qwen, TP2/EP2,4 resident requests, native
FULL attention and the same6GiB KV arena per rank. APC is enabled in both.
Native async-scheduling defaults remain unchanged. Prefill has a1024-query
maximum, not a32-token substitute workload. The owned portfolio decomposes
remaining prompt work into maximum chunks plus a ceiling-bucket tail. Actual
query lengths remain exact; padded rows do not write KV or advance requests.
Only the older fixed-plan control decomposes tails into exact powers of two.

Keep cold-start-cache and subsequent retained-cache rounds separate. Keep
profile collection out of timed runs. Run both native-first and owned-first
orders before interpreting performance. The owned arena cannot be handed back
to a native engine carrying stale APC hashes: reverse-order runs start with a
fresh native cache, while native-first runs never use native inference after
handoff. A completed root relinquishes its State lease, not native cache metadata.

**Measurement boundary:** native TTFT is observed at LLMEngine output delivery;
owned TTFT is observed at all-rank worker receipt retirement. Both have closed-
loop sessions, but candidate inputs are preloaded in the worker and it lacks the
native client/engine IPC path. Report the execution-plane results with that
boundary, not as identical-frontend latency or a deployed HTTP speedup. The
reported PyTorch allocated/reserved peaks are not total driver HBM peaks.

## Ownership and safety

- cache.py reuses vLLM KVCacheManager, Request hashing, refcounts and eviction.
  It reserves the finite request horizon but publishes only receipt-confirmed
  computed blocks. A terminal generation with references remains pinned.
- scheduler.py owns admission, session readiness, prefill/decode arbitration,
  projected authorization and complete-TP-quorum commit. CPU token receipts
  update native cache hashes; they never seed the next numerical graph input.
- root.py uses actual LiveModule State/MetaTensor/activation/replay. Compute
  alone publishes resident tables and continuation, including a new generation.
  Prefill executes one request's ceiling bucket with exact valid lengths; decode
  batches authorized residents.
- reactor.py retains separate input-reader and output-copy fences, owned pinned
  sources/destinations, metadata carriers and LiveInvocations. It queues two
  waves and admits N+2 only after N's full quorum; old drains remain in order.
- Native model/attention/MoE/sampler numerical kernels are retained. Attention
  dispatch is owned; graph-resident State supplies GM lengths. Runner execution methods
  are fenced. FULL mode uses FIA for decode too, unlike the earlier PA fixture.
  FIA needs its causal mask in decode metadata; serving-dummy1 preserves the
  missing-mask rejection, fixed in serving-dummy2.

This is one DP owner (TP2/EP2), not yet asymmetric DP or multi-node execution.
No speculation, grammar, sampled EOS, stochastic policy or preemption is claimed.
Resource exhaustion must wait for owned retirements or fail; it cannot silently
reuse an outstanding generation's blocks.

## Reproduce

Use the existing donor environment; do not install into it. The launcher freezes
source, trace and LiveInference code and uses selected-device admission/watchdog.

```sh
BENCH_ROUNDS=3 PROBE_DEVICES=4,5 \
  PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/owned-wave/UNIQUE \
  bash /workspace/strengthen-dsv4/prototypes/owned-wave/serving/run.sh
```

For a fresh two-arm comparison only, explicitly set `BENCH_ARM=both`; set
BENCH_ORDER=owned-first for reverse order. Do not rerun a native baseline when
Fletcher requested candidate-only measurement. Compare to retained native receipts:

```sh
python3 /workspace/strengthen-dsv4/prototypes/owned-wave/serving/compare_candidate.py CAPSULE \
  --baseline /workspace/strengthen-dsv4/runs/owned-wave/swe-perf-native-first2 \
  --baseline /workspace/strengthen-dsv4/runs/owned-wave/swe-perf-owned-first1
```

The comparator rejects configuration/trace drift, excluded/profiled baselines,
changed work or TP disagreement. It labels the baseline as historical, never a
fresh contemporaneous A/B. No generated-token identity gate is reinstated.

 QWEN_DUMMY=1 uses a clearly
separate two-layer synthetic257/389-token gate, not SWE performance. Use
PROFILE_STEPS=64 and BENCH_ROUNDS=1 for a separate native CANN/msprof collection.
Each arm's first64 waves are diagnostic windows, NOT identical-work intervals.
Never infer an end-to-end speedup from their summed kernel durations.

```sh
python3 /workspace/strengthen-dsv4/prototypes/owned-wave/serving/report.py CAPSULE
# Donor Python + CANN environment; CPU-only, AFTER hardware release:
python /workspace/strengthen-dsv4/prototypes/owned-wave/serving/parse_profiles.py CAPSULE
```

The offline parser uses a fresh official parser process per rank, exports DB and
Chrome traces, validates SQLite/rank identity and compresses the native timeline.
Raw PROF directories and native backend DBs stay beside the PyTorch export.
Workers collect without analysis: daemon-process parsing is not safe here.

## Host identity and evidence cautions

On2026-09-16 the local environment and ssh hw2 reported the SAME kernel boot ID:
they are not independent benchmarking hosts. Profile startup was cancelled when
this was discovered; swe-real1-hw2 is explicitly timing-excluded and serves as
correctness intake only. hw0 reported idle910B2 cards but did not have this model
or pinned environment at the known paths; do not assume it is launch-ready.

## Numerical acceptance versus performance

Fletcher explicitly chose performance comparison without an exact cross-arm token
gate on2026-09-16. The driver still requires equal calls, prompt/output counts,
TP token agreement and successful owned lifecycle checks. `token_comparison`
records divergences; PASS completion is not a model-quality/equivalence claim.
The excluded real intake completed44calls/12,111tokens in each of two rounds;
only2/44 calls matched across arms, while native cold/retained matched10/44.
The cause of cross-arm divergence is unresolved; native variability does not
prove that all candidate differences are harmless. Do not restart that separate
inquiry as a prerequisite for the authorized performance comparison.

The completed two-order real-weight measurements and native profiler observations
are in [PERFORMANCE.zh-CN.md](PERFORMANCE.zh-CN.md). Rebuild the bounded provider
summary with `summarize_profiles.py CAPSULE` after the official offline export.

For readable TraceLoom Perfetto timelines, run `export_traceloom.py CAPSULE`
after `parse_profiles.py`. It reuses frozen TraceLoom37323af, creates four separate
rank-local analysis DBs/timelines under `CAPSULE/traceloom`, indexes only derived
child edges for the established exporter workaround, and validates complete JSON
before publishing filenames. It does not alter original profiles or claim
cross-rank clock alignment. The completed swe-profile1 exports are indexed in
`runs/owned-wave/swe-profile1/traceloom/exports.json`.

Default host-planned prefill now chooses the smallest **ceiling** bucket and carries
the actual query count separately;70 tokens replay128 once, not64+4+2. Padding
has no KV/cursor/sampler authority. See
[the planner boundary and qualification](../fia-plan/CEILING-PREFILL.zh-CN.md).
The old frozen-plan switches retain exact-size tail splitting.
