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

## Full-service continuation on CANN9.1 (qualification in progress)

The subsequent `overlap-swe` experiment branches from public main852c106 rather
than importing the parent PD development branch. Historical E16 source is
`eee35fd6b50fca73385ad8f8af714fcd7e5c2684`; E36 is
`96cd03a362b18d68ea1a1b1f7ada5633d9c5e60c`. Task-local launch/qualification
helpers live here; raw capsules remain `/workspace/overlap-swe` on hw112.
These are explicit experiments, not new packaged EP admission or leaderboard
points. `ep_service_entry.py` changes only validation views for TP2/DP1/EP2;
actual runner configuration remains EP-enabled. `target_only.py` is the parent
PD diagnostic cut: it retains sampler feedback while skipping draft work, not
an MTP memory optimization or MTP qualification.

Do not combine old FIA Python planning with new native libraries. The accepted
migration `candidate-package-4` supplies `qwen_fia/wave.py` and
`context_parallel/{adapter,plan,prepare}.py` together with its native payload.
`plan.py`'s Q16 bound changes derived workspace offsets; `prepare.py` has the
CANN9.1 vendor syntax adaptation and disjoint Q16 zero/LSE scratch. The staging
input `native/patches` includes these four files, not just `.so`/native.json.
The old main Python rejected real target-prefill4097 at the native plan guard;
restoring wave/adapter passed257/4097/8193 but a16-request wave then hit the old
Q3 planning bound. Preserve failed `ep-serial2`, `ep-target-serial1/2` logs.
`draft_fia.prepare` also wraps target attention: its stack name does not prove
that a failure is in draft execution. No numerical guard is weakened here.

The GDN host wrapper is rebuilt from historical E36 source with37-row admission;
its9.1 build needs CANN include in `CPLUS_INCLUDE_PATH`. Pinned donor sources are
unchanged. Only task-private Transformers5.14.1 and torch-npu post4 admission
replace the historical runtime metadata; system packages stay untouched.

The final target publication additionally restores **parent3e93c777's**
`wave.py` and `context_parallel/adapter.py` (rather than candidate4's older two
files). Native C16 appends a KV1 padding row; the parent's `target_metadata`
proves the runner's live-row/token frontier before translating only padding to
owned KV0. The Q16 plan/build/native files remain candidate4. Attempt
`ep-target-serial3` failed on the missing translation, not a real17th request.
The main tree's numerical attention source is **not** changed by this overlap
branch; these explicit runtime migration changes stay in the experiment overlay.

### Real-model TP2/EP2 target-only result

`ep-target-serial4` versus `ep-target-early1` passed the same requests on the
same frozen overlay/model/assigned pair. Three257/4097/8193-input,64-output
requests have identical token IDs **and every selected-token logprob**. The
16×256-token profiling batch and three16×512-token measured batches also have
exactly matching output IDs for all64 request rows. This is bounded cross-arm
parity, not independent full-model quality, long-context PD or MTP qualification.

Unprofiled output tokens/s (whole two-card server):

| Arm | Three fixed short-prompt C16 batches | Median |
| --- | --- | ---: |
| Serial |683.905,693.472,684.399|684.399|
| Overlap |669.973,680.028,669.798|669.973|

Candidate median is **2.11% lower** in this sequential pair. Do not assert
statistical significance. Each batch is8192 output tokens, no synthetic MTP
acceptance; draft forward is disabled explicitly. This is **not** a SWE point.
Therefore no C1–C32 rerun, website update or unconditional default flip follows.
The useful two-card leaf gain did not become an end-to-end gain here.

Offline native profiler analysis is necessary: worker daemon processes collect
but cannot export their traces. Run `torch_npu.profiler.profiler.analyse` in a
separate non-daemon process after collection. The five captured waves include
prefill/admission; do not average them as steady decode. `compare_service.py`
selects complete40-layer/24-token graph occurrences by unique TP2 shared-matmul
shapes, requiring120 shared matmuls and80 routed GMMs. Serial graph3 device span
is20.386/20.484ms (ranks0/1); candidate20.772/20.745ms. Shared/routed-GMM time
intersections are zero in both (both consume Cube resources); separate streams
alone do not prove useful overlap. This DP1 topology has no DP gather/scatter
window. Earlier DP2/TP1 leaf communication overlap remains valid evidence, not
proof of a TP2 multi-DP server speedup.

`service-comparison.json` binds all numerical checks, timings and profile spans.
For the real target topology the next gate needs at least **four assigned cards,
DP2/TP2/EP4**. hw112 exposes only two; parent hw86/hw81 are not authorized here.
Keep the adapter opt-in pending that deployment gate. The CANN9.1 full-MTP
composition and historical pure-TP2 SWE reproduction remain unqualified; do not
silently relabel this target-only run as MTP2 or discard the original website
measurements. The current SWE client has a different commit/runtime metadata,
but canonicalizing only prepared tokenizer path/version reproduces the exact
historical prepared-workload SHA8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85.

All owned services exited and both assigned devices returned idle. Local backup:
`/Users/fletcher-tian/Developer/BetterScale-ep-overlap-artifacts/overlap-swe-evidence-20261003.tgz`
contains the final executed overlay/native payload, raw/parsed profiles, actual
responses, manifests, scripts and failed-run logs (no model weights/venv/cache).
Two existing adapter/config CPU regression tests pass on the staged pinned
runtime; new comparator has been exercised on both real retained capsules.
