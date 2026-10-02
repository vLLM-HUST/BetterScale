# Native Qwen MoE shared-expert overlap (hw112, 2026-10-03 CST)

Bounded two-card experiment and opt-in adapter, **not** a 16-card engine throughput
qualification. Do not touch the parent PD campaign's hw86/hw81 to reproduce it.

## What moves, and what does not

Pinned Ascend already implements a two-stage shared expert on a separate stream.
The default `multistream_overlap_shared_expert` is false. Its BF16 first stage
waits for `before_dispatch`, which the AllGather implementation records **after**
DP input/router gathering. AllGather's token dispatcher is local routing; real
network work is in prepare/finalize. Thus merely enabling the donor flag misses
an early communication window.

`src/betterscale/models/qwen35/moe_overlap.py` records local input readiness before
native `no_shared_forward_impl` and substitutes that event for the first shared
stage. It preserves the donor's second-stage `before_combine` dependency, stream
join, sigmoid gate, expert arithmetic, collective order and final TP reduction.
The patch does not parallelize the inherently dependent dispatch → expert GEMMs
→ combine chain. No donor forward function or installed runtime is copied/edited.

Qwen35's `before_init` accepts two explicit additional-config flags:

```json
{"multistream_overlap_shared_expert": true,
 "betterscale_shared_expert_overlap": true}
```

Default remains unchanged. Install is process-local/idempotent. The adapter
rejects non-BF16, quantization, non-AllGather, SP, PCP and shared-expert-DP paths.
An EP8/full-model deployment still requires its own graph/correctness gate.

## Evidence envelope

- BetterScale base `3e93c777509bf6b78fcb7b6108d47c008b9adaa9`.
- Core vLLM `752a3a504485790a2e8491cacbb35c137339ad34` / 0.25.1;
  Ascend `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` / 0.25.1rc1.
- hw112: two available Ascend 910B2 (physical 1/6, logical 0/1), CANN9.1.0,
  driver26.0.rc1, Torch2.10.0+cpu / torch_npu2.10.0.post4.
- Private overlay `/workspace/ep-overlap-runtime/pinned`; system vLLM0.23 remains
  unchanged. `run-hw112.sh` is deliberately task-local, not a portable launcher.
- Native FusedMoE/AscendMoERunner, random BF16 weights, Qwen geometry:
  hidden2048, intermediate512, shared512, experts256, top8. No model weights or
  model-quality claim. TP2 uses shared TP shards and a replicated scalar gate;
  DP shared weights are replicas in the full-forward runs.

Each graph case compares four changing-input/routing replays to its own eager
output exactly; `compare.py` additionally requires bitwise serial/candidate
identity on both ranks. It uses native full MoE forward (including final TP
reduction) when `--full-forward` is set. Initial wiring-only receipts do not
qualify that final reduction. `ctx.moe_layer_index` resets once per model wave,
matching native forward-context lifetime rather than accumulating leaf calls.

Mean synchronized wall time per graph replay (200 iterations, no profiler in
this timing window; not a latency distribution):

| Native full-forward envelope | Serial | Early-event overlap |
| --- | ---: | ---: |
| DP2 TP1 EP2, 48 tokens/rank | 0.783 ms | 0.717 ms |
| DP1 TP2 EP2, 48 tokens | 0.654 ms | 0.606 ms |
| DP2 TP1 EP2, 16 vs 1 tokens | 0.492 ms | 0.463 ms |

The earlier 30-replay, inner-forward three-arm control was 0.779ms serial,
0.724ms donor multistream, 0.723ms early-event. **Most measured benefit is enabling
the existing donor stream; incremental benefit of the earlier fence is small**.
Eager was host-launch-bound and slower with multistream (~3.3ms vs ~3.8ms);
do not present this as an eager optimization or extrapolate single-layer gains
to whole-engine token throughput.

`early-fullgraph2` profiles five replays on each rank. Shared ordinary Matmul
(routed experts use GroupedMatmul) intersects AllGather for approximately
37–41us total and ReduceScatter for 77–80us total across those five waves.
Serial intersections are zero. Donor-only first-stage AllGather intersections
are zero. These are **device kernel time intersections**, not Python annotation
or mere stream existence. No claim of full communication hiding.

Receipts/logs/timelines: `hw112:/workspace/ep-overlap-runtime/`:
`{serial,donor,early}-{eager1,graph1}`, `{serial,early}-fullgraph2`,
`{serial,early}-tp2graph1`, `{serial,early}-skewgraph1`. Profile files are beneath
`profile-rank{0,1}/*/ASCEND_PROFILER_OUTPUT/trace_view.json` and
`kernel_details.csv`. Keep bulky traces outside Git.

## Reproduce without repeating bring-up failures

Run only on the user-assigned pair, after checking both devices are idle:

```sh
bash prototypes/ep-shared-overlap/run-hw112.sh serial unique-serial --graph --full-forward --profile --iterations 200
bash prototypes/ep-shared-overlap/run-hw112.sh early unique-early --graph --full-forward --profile --iterations 200
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python prototypes/ep-shared-overlap/compare.py /workspace/ep-overlap-runtime/unique-serial /workspace/ep-overlap-runtime/unique-early
```

Use `--tp 2` for the separate TP reduction gate, or `--tokens 16 --peer-tokens 1`
for DP padding/skew. Launches have a timeout and refuse existing output paths.
Do not run multiple NPU fixtures concurrently. `tests/test_qwen35_moe_overlap.py`
checks event ordering, untouched combine fence/output, scope and config opt-in.

Leaf setup needs native Ascend distributed groups and MC2 token capacity even
when selecting AllGather, because native setup registers every communicator.
Import `vllm_ascend.vllm_ascend_C` to register custom native operators. The actual
ModelConfig is necessary for the native MoE backend selector. Use
`VLLM_PLUGINS=ascend` here to avoid the unrelated general-model registration
plugin's missing HunYuanVLProcessor in this container; this workaround does
**not** qualify full-engine imports or authorize a Transformers/pin upgrade.

## DeepSeek secondary inspection

Official source checkouts are isolated under `hw112:/workspace/deepseek-ascend-play`:

- [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend),
  `8491bbb4b8c02a094a2318965f50c70438a3e73c`.
- [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend),
  `3b25377d04b24fc6154698ded78a2bcb2c59afff`.
- [DeepJIT](https://github.com/deepseek-ai/DeepJIT),
  `3732a3b9a0a2e5126396908a362ba5abdf043b2f`.
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA),
  `2e5429fc5653bab6e081f09477126f731882a6a9`.

The September30 release targets Ascend950/CANN9.2, not this 910B/CANN9.1 pair.
DeepEP specifically requires A5 UBMEM/UBC_CTP/URMA and says graph capture is
unsupported. It is not a drop-in replacement for this captured decode path.
DeepJIT is header-only, not a standalone pip runtime; its host preflight fails
here because GCC11 lacks `<format>` (`host-cxx20-preflight.log`). No global
compiler/toolkit/runtime upgrades, no unsupported kernel launch, and **no claim
that these libraries were installed or performance-tested**. Study deferred
EP epilogues/pipelining as prior art without importing A5 bandwidth claims into
910B qualification. Hardware-compatible installation remains outside this host's
verified capability.

A local evidence backup is retained outside Git at
`/Users/fletcher-tian/Developer/BetterScale-ep-overlap-artifacts/overlap-evidence-20261003.tgz`.
