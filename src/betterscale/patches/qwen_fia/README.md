# Wave-shared FIA for Qwen27 mixed FULL

This is the head256 adaptation of `prototypes/owned-wave/fia-plan`, enabled only
by the owned Qwen route of `betterscale.worker.Worker`. No attention arithmetic is replaced.
Native NONE/eager attention remains the numerical oracle. DSV4 and native Qwen MTP2 compositions do not install this patch.

## Pinned boundary

CANN9.0.1 / Ascend910B, BF16, TP-local Q12/KV2 heads, head256, causal TND,
paged128, no sinks or sliding window, up to eight real requests plus a padding
row and2048 query tokens. The admitted native function,24-block non-FD grid,
2528-byte tiling and descriptor relocation layout are checked, not guessed.
An unexpected variant fails closed. The donor/runtime pins still apply.

Each wave runs the native host planner **once**, suppressing its numerical
launch, and publishes one packed pinned slab: native tiling, query endpoints,
KV lengths and block table. Every full-attention layer reads this bank's device
metadata inside its captured kernel. No per-layer task-group update, FIA host
launch, or GE parameter injection is needed during replay. A padding request
may have positive query capacity but zero KV length; it must not be rejected or
extend a real request. Native host planning is retained, not replaced with an
invented FD/tiling policy.

Frames are per `(capacity, bank)`. Upload completion protects pinned-host reuse;
consumption events protect device overwrite. The existing ingress stream
publishes and compute waits on device. Layer scratch is invocation-local Torch
allocation from the existing graph pool, not duplicated persistent bank scratch.
A single independent Q/output planning fixture is retained per runner; KV fixture
references borrow the model's existing persistent pool. The planner uses a
transient128MiB workspace allocation; its intercepted launch never reads it.
This is a serial single-runner protocol, not concurrent multistream replay.

## Build and deploy

Source the pinned CANN environment, expose this package, then CPU-build:

```bash
python -m betterscale.patches.qwen_fia.build /absolute/path/libbs_fia.so
export BETTERSCALE_FIA_LIBRARY=/absolute/path/libbs_fia.so
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
export TASK_QUEUE_ENABLE=0
```

Set preload **before starting Python**. `qwen_gdn/serve.sh` does this from
`BETTERSCALE_FIA_LIBRARY`. The native digest and process-global symbol identity
must match `native.json`; a late dlopen is not sufficient. Do not install the
interceptor into CANN or globally preload unrelated services. Native binaries
are not in Git. A different toolchain binary requires requalification, not
removal of the gate. Run `test_native.cpp` against CANN headers with `-ldl` for
CPU ABI rejection and bounded persistent-plan refresh checks.

The captured FIA numerical launch and ordinary graph replay are still necessary.
This does not capture sampling or replace the native request scheduler.

## Replay ordering and startup

The pinned ACLGraph wrapper's `enable_enpu` field is scoped only around owned
FULL calls to select its existing caller-ordered replay branch. This does NOT
enable the runner's ENPU path. It removes the CPU stream barrier that existed
for native attention task updates; serial compute ordering and both metadata
reuse fences remain. The wrapper flag is restored even on exceptions; NONE,
other workers and other graph modes keep their original behavior.

Each newly captured bank is replayed and drained once during startup, before
requests are admitted. The gate requires `_owned_capture_bank` and an empty
request pool; it is not live-request state rollback or speculative execution.
Only disposable startup state is touched. Ordinary cold-request initialization
still owns real state/cache rows. This follows LiveInference's principle of
priming unpublished graphs: measured first-use runtime setup is paid at startup,
not hidden in a later request's first encounter with that capacity/bank.

`NPUGraph.replay()` / `aclmdlRIExecuteAsync` remains the actual graph submission.
The cold call is not a steady per-step7ms cost. The retained observations and
qualification capsules are indexed in the repository's Qwen-serving knowledge
scenario; profiler spans must not be presented as fresh HTTP throughput gains.
