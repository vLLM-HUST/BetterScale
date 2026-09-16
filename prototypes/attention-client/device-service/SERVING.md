# Native generation through persistent remote experts

This is the model-side integration checkpoint, separate from GEMM optimization.
The experiment uses the frozen unsegmented server from commit7206096; ongoing
operator changes in the other worktree are not dependencies of these results.
Working branch: `lumi/expert-serving`.

## What runs end to end

Two independent native vLLM instances execute a two-layer dummy Qwen with the full
Qwen3-30B-A3B layer dimensions. Each instance retains native token scheduling,
embedding, gate/top-k, attention KV, final norm, LM head and greedy sampling.
Only routed expert math crosses to the two persistent expert-server devices.
Each server holds its half of both layers' experts and handles both sources.

Flow: `llm.generate` -> native worker/model runner -> per-layer FULL attention
bank -> device gate/publication -> remote expert owners -> device weighted
retirement -> next attention layer -> native LM head/sampling -> next decode.
Attention continuation is still host-driven, with a cooperative event-query loop;
it is not a full-model continuous device scheduler or an HTTP deployment.

`ATTENTION_JOINT_SHADOW=1` preserves original-forward plus whole-KV/output checks.
`ATTENTION_JOINT_SHADOW=0` removes original forward, KV snapshots, per-step shadow
comparison and its global synchronize. It also replaces local expert.forward
with a raising guard after remote attachment. Thus successful generation cannot
quietly execute the native local FFN. Weights still exist locally for bootstrap:
this is a compute-ownership witness, NOT a memory-disaggregation claim.

The attention catalog is populated by two initial generation requests, then
sealed before the measured requests. New shapes after seal fail explicitly.
This is bounded warmup, not production general shape admission. Qwen3-30B-A3B
has no shared expert branch; no shared-branch overlap claim is made.

## Bounded budget, not an implicit 24-task EOF

The existing wire admits one outstanding frame per source. The fixture now passes
an explicit `DEVICE_SERVICE_TASKS` budget (1–32 for unsegmented; segmented at most24
because of its trace capacity). Each source must publish exactly that many layer
jobs; both owner completions precede reuse. The model probe checks its expected
budget before generation: four requests × requested output tokens × two layers.
`ATTENTION_JOINT_OUTPUT_TOKENS=4` therefore exercises32 jobs/source rather than24.

This removes the hard-coded24 assumption from the persistent consumer, but does
NOT implement arbitrary EOS, unequal source lifetimes, online replenishment or
unlimited service. All requests use ignore_eos=True to make the budget exact.
A real service needs an explicit drain/EOF/replenishment protocol, not a larger
magic count. No such production qualification is claimed here.

## Qualified gates

Run capsules under `/workspace/betterscale-expert-serving/runs/`:

* `attention-device-joint-20260916T072614Z`: persistent backend, original shadow,
  12 forwards/client, outputs and entire ordinary Qwen KV tensors exactly equal.
* `attention-device-joint-20260916T072712Z`: shadow disabled from attachment,
  12 forwards/client, zero reference calls, local expert guard never reached;
  native sampling generates three tokens/request. Includes four-device profiling.
* `attention-device-joint-20260916T072932Z`: shadow disabled, four tokens/request,
  16 forwards and32 layer jobs/client; both servers retire all64 source-layer jobs
  without duplicate/lost generations. This tests longer repeated decode and the
  explicit budget, not a larger model or long-context workload.

* `attention-device-joint-20260916T073202Z`: matched32-job shadow control passes
  every output/KV check; both clients' measured generated token sequences match
  the independent no-shadow32-job run exactly. This closes the specific risk
  that reference forward was secretly preparing state needed by generation.

All are BF16 dummy, two layers, TP1 attention roles, at most32 input rows and
256-token context. Numerical equality belongs to the shadow gates; merely
producing tokens in no-shadow mode is not an independent accuracy benchmark.
CPU continuation/contract regressions:11 tests pass.

`serving_receipt.py RUN` validates model forwarding, sealed catalog, device route
and completion flags, per-source exact generation sequences on both servers and
zero reference calls in no-shadow mode. See `serving-result.json` for compact
receipts. Historical primitive/host-server summarizers are not this ABI's checker.

## Reproduce

Use a frozen qualified `PERSISTENT_BUILD` and `DEVICE_SERVICE_SOURCE_BUILD`.
Do not consume another task's mutable build directory while it is compiling.
From this worktree, on four admitted idle cards:

```bash
ATTENTION_JOINT_SHADOW=0 ATTENTION_JOINT_OUTPUT_TOKENS=4 \
DEVICE_SERVICE_TASKS=32 DEVICE_SERVICE_PARALLEL=1 DEVICE_SERVICE_PERSISTENT=1 \
DEVICE_SERVICE_SEGMENTED=0 DEVICE_SERVICE_TIMING=1 \
  bash prototypes/attention-client/device-service/run_joint.sh 0,1,2,3
```

Set `DEVICE_SERVICE_PROFILE=1` to capture; parse with `profile_export.py RUN`
after releasing devices. The exported `attention2-expert2-provider-clock.json.gz`
preserves provider timestamps, not an independently fitted cross-device clock.
Do not compare these host-shadow/profiler/bounded costs to DFC throughput.

## Next integration boundary: full-model ownership

Current bootstrap allocates a complete expert-weight staging area on each
attention device, loads local native experts, then copies them to servers.
For48 layers, just that staging area is54GiB; local BF16 expert weights add
another54GiB before attention, KV and graphs. Full-model expansion would OOM for
structural reasons on64GiB cards, not because the remote communication is broken.
The two expert servers would each own about27GiB of routed BF16 weights.

The full-model route must instead load by role, retain only native gate/attention
and other dense weights on clients, and populate expert shards directly on the
servers. The current two-layer kernel catalog also needs a stable per-layer
weight-address interface rather than silently extending128 flattened groups.
Then add real EOS/drain and finite startup bank admission, before measuring
full-model TTFT/TPOT or claiming production serving. These are explicit remaining
engineering tasks; this checkpoint does not substitute tiny-model throughput for
them. Published BetterScale Worker defaults remain unchanged.
