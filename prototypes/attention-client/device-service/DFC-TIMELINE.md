# Native DFC timeline: outer execution versus internal pipeline

## Reproduce

`DEVICE_SERVICE_PROFILE=1 bash run_dfc.sh 5,7` captures eight warm broad-hit
32-row/source FULL graph replays, outside oracle/capture and timing trials.
Run the pinned runtime's Python on
`profile_export.py <capsule> --dfc` after devices release. Default four-role
export remains unchanged. This is the existing **lab A2 BF16 DFC provider**, not
the stock donor package and not quantized GroupedMatmulSwigluQuant.

Observed capsule: `runs/dfc-ep2-control-20260916T080421Z`.
Compressed TraceLoom output: `analysis/dfc2-provider-clock.json.gz`.
Both ranks pass all six oracle cases. Unprofiled broad32 medians are
419.61/422.68us. In the profile, the seven subsequent DFCs are approximately
408–413us. The first rank1 DFC is3544us: its start precedes rank0 by3134us,
consistent with waiting for the peer's independently started profiler. Do not
average that startup into the steady comparison. Preserve it in the raw trace.
Inter-kernel gaps are about5–6us; these are NOT internal up/down gaps.
Provider timestamps are retained; no independently fitted cross-device clock.
Compact measurements: `dfc-timeline-result.json`.

## What this does and does not establish

Native Level1 emits one fused DFC envelope per invocation. It does not reveal
individual Cube MMAD, MTE, or AIV wait intervals. Our instrumented persistent
kernel's internal timeline cannot be compared as if this DFC envelope proves
uninterrupted computation. No per-core utilization claim follows from it.

The source shipped alongside the loaded provider was inspected at:
`/workspace/my-ascend-workspace/stateharbor/build/lib.linux-aarch64-cpython-312/livemodule/arch/ascend/_native/opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/dispatch_ffn_combine_bf16/dispatch_ffn_combine_bf16_kernel.hpp`.
This is packaged-source evidence, not a disassembly proof of binary identity.

- AIC entry214–218 executes GMM1, waits for activation, then GMM2. It does
  **not** interleave arbitrary up/down tasks on those same Cube cores.
- GMM1 creates one BlockMmad before traversing experts. Its per-expert readiness
  waits permit computation after that expert's pull, before all experts arrive.
- The prefix boundary around499 drains/publishes finished up while subsequent
  up proceeds. The persistent prototype already explores this middle overlap.
- AIV publishes pull progress per group; CombineV2 around1040 consumes staged
  down completion flags. Our full-pack admission and whole-down return remain
  coarser boundaries. These are concrete next candidates, not measured causes
  of the entire performance difference.

Thus “flat versus pipelined” is useful only after naming a dependency boundary:
input arrival→up, up→activation, down→return, or intra-GEMM memory→compute.
Matching tile geometry alone does not establish equal memory traffic/utilization.

## Internal profiling route and limitation

Official msOpProf documentation describes MC2 communication/computation timelines
using a rebuilt operator with `-DASCENDC_TIME_STAMP_ON -g`. Generic TimelineDetail
excludes communication-compute fusion; its instruction view is simulated, not a
free real-board per-core trace. Do not use independent kernel replay on two
communicating ranks or claim this ordinary torch profile supplies those details.
No provider was rebuilt/reinstalled in this experiment. Internal DFC timing
remains unmeasured; a separate isolated instrumented provider is needed.

Sources:
- https://raw.githubusercontent.com/Ascend/msopprof/master/docs/en/user_guide/msopprof_usage.md
- https://raw.githubusercontent.com/Ascend/msopprof/master/docs/en/user_guide/msopprof_user_guide.md
