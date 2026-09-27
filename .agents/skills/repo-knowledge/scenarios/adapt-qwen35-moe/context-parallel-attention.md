# Landing quota attention in Qwen35 serving

Enter before changing pure decode/verification FIA scheduling or reusing the
single-decode research kernel in a model graph. Current work is opt-in; see
`src/betterscale/patches/qwen_fia/context_parallel/README.md` for code-owned
contract/build selection. Do not resume the closed same-instance AIV regression
hunt merely because this note mentions its source.

## Source and serving boundaries

Research donor `ascend-op-prof-single-decode@ea92f61` has Q28/KV4/D128. Its
split-capable partial-outlined producer is the intended candidate; the full-only
FF diagnostic cannot implement context parallelism. Qwen35 TP2 changes geometry
to Q8/KV1/D256, with nonuniform real query rows1..3 and optional zero-KV padding.
One request is one head-group task;24 groups partition its context without
replicating the head group. Only split requests use partial O/logsumexp reduction.

Main `c238443` already contains the packaged Qwen35 Worker and its own four-file
pinned Ascend adaptation. The old local worktree at0e20558 and frozen
`moe-request-sampling1` capsule are not current main. Do not silently substitute
an old experiment for a current-main integration or import its top-level module
names into the packaged runtime. Preserve other tasks' dirty worktrees; the
attention landing uses `research/betterscale-context-parallel`.

## Three paid integration traps

1. **CPU lengths are envelopes.** `draft_fia.Frame.prepare` copies actual device
   KV lengths after host-slab publication. Static context endpoints derived only
   from optimistic CPU lengths can omit actual tiles. Encoded endpoint
   denominators let the device scale both ends using its actual tile count;
   static partial slot/reducer ownership stays unchanged. Empty scaled pieces
   publish O0/LSE-inf by vector DMA; do not infer direct-output ownership from
   the scaled span or use scalar cached writes into adjacent LSE slots.
2. **Dummy capture is not real verification.** Native mixed partitioning uses
   `[2]*15+[10]` at capacity40 and `[1]*15+[9]` at24. Existing GDN capture clamps
   its own query view but does not change FIA's. Only in disposable startup with
   an empty request pool, copy/clamp target FIA metadata and append zero-KV
   padding; leave shared metadata/device source untouched. Never weaken live
   Q1..3 admission to make dummy capture pass. The first old-capsule full-model
   attempt exposed this before readiness; no numerical/model result came from it.
3. **Preserve CANN's PYTHONPATH.** Prefix task-owned source/runtime paths, do not
   overwrite the environment sourced by CANN. Overwriting it caused missing
   `acl` before worker initialization. This was environment preparation, not an
   attention or device failure.

Native/draft frames keep2528-byte tiling. Opt-in target verification capacity
keys6/12/24/40/48 alone have4096-byte banked tiling. Capacity3/nativeB1,
prefill/mixed and draft remain original. Preserve existing upload/consume fences,
device-feedback copy ordering, invocation-local capture scratch and all native
workspace invariants. No new graph key or per-request host synchronization.

## Bounded leaf evidence, September27

Artifact root: workspace `runs/attention-response/20260927-betterscale-c16`;
hw3 mirror `/home/jingyuan/ascend-probes/betterscale-c16-20260927`.
`build-device-lengths/libbs_fia_cp.so` identity
`e925f5e57132dd824e1269ca1482d44a03e7c535f07032fcca7916387773ea6d`.
The native FIA host adapter is byte-identical between the leaf reference and
current-main packaged payload. No installed CANN/runtime was edited.

`leaf2` passed six inputs: tile/page edges, C16 short, nonuniform Q1..3,
14×32K+2×256K,15×1K+1×256K and16×256K. Two captured banks,8 changing waves each;
fixed upper-length schedules with device actual lengths down to3/511/512/513
and back to full;13-query zero-KV tail, immutable inputs and output/workspace
guards. Maximum native BF16 difference0.0009765625; edge requests also passed
independent FP32 CPU reference. These observations do not certify arbitrary
Q shapes, mixed execution, whole-model correctness or throughput.

The E2E protocol must keep source/runtime/State policy/KV budget identical and
change only attention. Current main's frozen SWE-prefix-reuse C16/900s protocol
uses natural MTP2, seed20260924, exact output-token continuation and unchanged
prepared workload, not AgentX's forced-AL protocol. Retain actual occupancy,
cache mix, failures, tails and repeated-arm variability with throughput.

## Current-main functional and one-pair E2E evidence

September27, base `c23844344b5a0c8711f66e875ad5f464c27f912a`, hw3 physical0/1.
`functional-main1` passed24 cold/warm/concurrent code retrievals through262080
input tokens with natural MTP drafted98/accepted98. Those short answers did not
exercise a real split; the separate overlap gate warmed32769/261000 prefixes,
then overlapped256-token continuations. Both arms retrieved the code correctly,
and candidate logs prove real split producers in both ranks and banks. Forced
post-EOS continuation is execution stress, not semantic quality evidence.

One **AB pair**, as Fletcher requested to conserve compute: separate C2/60s
qualification then C16/900s per arm, SWE-prefix-reuse client
`29136f1f481ebab8566014a05b7ed53bcf79dc84`, frozen seed20260924 workload,
exact-token continuation, natural MTP2. Same current-main source, pinned runtime,
BF16 TP2, native aligned State, query4096,16 requests,262144 configured context,
26,038,239,232 KV bytes/rank. Only attention selection differs. This does not
qualify resident State or forced-acceptance AgentX. Both C16 protocol gates PASS.

| Metric | Native attention | Quota attention | Change |
| --- | ---: | ---: | ---: |
| Output tokens/s/chip |369.8144|391.3000|+5.81%|
| TPOT median (ms) |20.5244|19.3595|−5.68%|
| TPOT P95 (ms) |34.2535|34.0589|−0.57%|
| TPOT P99 (ms) |76.3561|79.7673|+4.47%|
| TTFT P95 (s) |0.91629|0.93675|+2.23%|
| TTFT P99 (s) |1.39731|1.34847|−3.50%|

Zero failed requests and sampled preemptions in both arms; in-window completions
1058→1119,16 drained each; full-concurrency fraction99.550→99.530%.
Sampled natural mean acceptance length2.91664→2.90809 (includes drain), so the
candidate did not obtain higher throughput through increased acceptance.
Cache fraction including drain93.734→93.884%; uncached tokens1,924,899→2,020,763.
Fixed-time closed-loop runs reached different workload prefixes; this is one
observation, not a repeat-based confidence interval or statistical significance.

The longest SWE contexts were90095/92944, **not a C16 256K-skew throughput test**.
Descriptive context bins show32–64K TPOT P95 23.419→21.881ms and >64K
23.703→21.562ms. Conversely ≤8K P95 77.694→81.396ms. Both arms' twelve slowest
TPOT rows were initial turn0 requests starting in the opening burst. Preserve
these rows in headline tails; this stratification does not uniquely identify a
kernel cause, nor justify calling all long-tail latency improved. The separate
warm32K/261K overlap gate took16.866/16.726s native versus14.371/14.590s quota
for256 output tokens each; that is one overlapped-service observation, not
isolated kernel timing. Keep the feature explicit opt-in, native by default.

Artifacts: `comparison-ab.json`, `tail-cohort.json`, `swe-{1-baseline,2-candidate}-c16`,
`model-gate-{1-baseline,2-candidate}` and `campaign1` under the capsule above.
The compact colocated `context-parallel-ab.json` preserves metrics/limits.
`ab-amendment.json` records AB_COMPLETE_REMAINDER_CANCELLED: the original driver
had an ABBA loop; it was intentionally stopped after candidate server exit0,
before a third arm. Its raw FAIL/exit1 refers to cancellation of the unused
remainder, not either completed measurement. Do not relabel raw receipts.
Both servers exited0, owned workers/tunnel stopped and cards0/1 released with
no foreign-owner event. No additional NPU repetitions were made.

Final main-tree source matches the executed snapshot semantically: wave.py is
byte-identical; other Python ASTs differ only in docstrings, and tiling.hpp only
adds its vendor license notice. Seven CPU planner/adapter tests, six FIA tests,
and six unittest-discovered Qwen35 checks passed; this is not the full pytest
suite. The leaf and model gates above cover the changed device path.

## Resident State plus balanced attention: one combination point

Fletcher then requested only the combined configuration, no new control/repeats.
Source `283e06d`, hw3 physical0/1, same pinned SWE client/workload/seed and
C16/900s protocol, natural MTP2, E16/R20, query4096,262144 maximum context.
Unlike the AB above, select `using_live_runtime=true` and `LiveStateScheduler`
**together with** `BETTERSCALE_CONTEXT_PARALLEL=1`. Both earlier resident commits
`a8abd05` and `d1ca3ec` are ancestors of283e06d: the AB omitted activation, not
source integration. Never treat latest main alone as proof every optional path
is enabled. Total State+FA budget remains26,038,239,232 bytes/rank; this funds
resident State1,912,095,920 bytes plus16,720 shared128-token pages, not24.25GiB
of FA alone. Startup reports2,140,160 FA token positions.

The independent gate first retains32769/261000-token prompts after three forced
output tokens, then appends the actual generated IDs and a new-user delta.
Both hot requests hit exactly32771/261002 cached tokens and retrieve the code;
concurrent256-token forced continuations observe split producers in both ranks
and both banks. Post-EOS text is not quality evidence. Repeating the original
prompt is NOT this gate: resident-only State cannot roll back to arbitrary old
prefixes. C2/60s qualification then the single C16/900s point both pass.

- Output **442.9806 tokens/s/chip**, decode P90 **62.9063 tokens/s/user**.
- TPOT median/P95/P99 **17.3895/22.8271/30.0165ms**;
  TTFT P95/P99 **0.58866/0.81222s**.
- 1280 in-window completions,16 drained, zero failed requests and sampled
  preemptions; full-concurrency99.427%. Longest observed prompt106308 tokens.
- Prompt-token cache fraction97.653%,956827 uncached tokens including drain;
  sampled natural mean acceptance length2.90104 (sampling includes drain).

Relative to historical resident/native-attention414.8011, throughput is6.79%
higher and decode P90 is5.76% higher, **but TTFT P95 is27.35% worse** (0.46225s
historical). Relative to earlier hw3 native-State/balanced391.3, throughput is
13.21% higher. These are descriptive historical comparisons, not an isolated
attention gain:414.8 ran on the local host, this point on hw3, without a new
paired control. Do not infer statistical significance or additive gains.

`runs/attention-response/20260927-resident-balanced-c16` owns frozen source,
launch/client commands, original requests, `result.json`, long gate and release
receipts; colocated `resident-balanced-c16.json` retains compact metrics.
Only one combination arm launched. Server/controller/admission exited0; no
foreign-owner event. Selected0/1 were returned without disturbing other cards.
