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

A final **MTP-enabled serial control**, `ep-mtp-serial3`, uses the same repaired
target overlay with only the target-only cut removed.257+64 passes;4097 fails
specifically in `llm_base_proposer._propose → run_draft → draft_fia.Runnable`,
native FIA `fd=1 blocks=9 cap=3 q=[1,3] kv=[4097,0]`. Unlike the earlier ambiguous
wrapper stack, this is direct evidence of the remaining historical draft-route
migration gap. Overlap is disabled. Do not weaken the FD/grid guard or attribute
this failure to shared experts; a full-MTP comparison requires a separately
qualified draft baseline. The owned server exited and both devices are idle.
The small `overlap-mtp-baseline-failure-20261003.tgz` companion local archive
preserves the launch, actual response and complete failure stack.

### Owned-draft control: execution restored, correctness still not qualified

The next isolated overlay additionally replaces only
`models/qwen35/draft_fia.py` with parent3e93c777's file. This is an explicit
post-staging copy, **not** a change to the packaged mainline draft path or the
previous target-only measurement. `ep-mtp-owned-{serial,early}1` both complete
all three cold requests and the C16 profiling batch; actual serial counters are
1448 draft steps,2896 proposals,2831 accepted tokens. Both capture32 FULL graphs.

The exact cross-arm gate fails:4097-input case first differs at output24;
serial ties comma/` string`, candidate prefers ` string` by0.25logit. C16 profile
outputs16/16 match; the three timed batches match16/16,14/16,14/16. Critically,
**serial versus itself** (identical prompts/fresh salts) also matches only14/16
between batches0/1; early's batches0/2 match14/16. This proves baseline output
variability exists; it does not prove that every cross-arm difference is harmless
or exclude an additional overlap effect. The failed gate remains failed.
`compare_service.py` now writes diagnostic evidence before rejecting parity;
its existing target-only comparison still passes. MTP timings in the companion
JSON are diagnostic only (median1445.76/1446.71 outputtokens/s), not a score.

`continuation_service.py` is the parent's exact warm/cold/uninterrupted probe,
with only the served model name and default port changed. Both MTP arms have
identical IDs and selected-token logprobs on all four requests, but both fail
the continuity invariant: warm=cold, while both differ from uninterrupted at
suffix token12 (global76). Prefix64 matches; warm cached4475, cold cached0.
There is no EOS among the first64 tokens. The warm branch prefers ` red` by6
logits; uninterrupted ties ` text`/` red`. Do not label the full discrepancy
proven benign rounding.

A final same-overlay **target-only serial** control, `ep-target-continuity1`,
has exactly the same IDs and selected-token logprobs on all four requests and
fails the identical continuity check. Thus this observation is **not isolated
to speculative execution or overlap**. It also does not invalidate the earlier
cross-arm target-only parity: that was a different, explicitly bounded gate.
The parent repository's passing no-EP target continuation cannot be silently
substituted for this TP2/EP2/historical-source combination. Stop performance
publication here until the baseline's continuation behavior is understood or
an accepted baseline is supplied; no numerical guard/tolerance was weakened.

For the future DP2/TP2 gate, preserve the native non-SP backend marker
`all2all_backend=flashinfer_all2allv`. An explicit custom Worker skips the
platform's auto-Worker fixup. The core `use_sequence_parallel_moe` predicate
requires DP>1, so the absent marker does **not** explain this DP1 failure; do
not "fix" or rerun DP1 merely on that hypothesis. It matters at the four-card
boundary. No hw86/hw81 operation occurred.

`overlap-mtp-controls-20261003.tgz` in the same local artifact directory preserves
both MTP arms' raw/parsed profiles, all outputs and failures, the final target-only
control and its exact overlay. These three services exited and the pair is idle.

### Pure-TP2 secondary gate and current stopping boundary

`tp-mtp-owned-serial1` selects the historical public TP2 Worker with EP disabled,
keeping the same owned-draft overlay/runtime and real MTP. All three cold HTTP
checks pass. Unlike EP2, its full warm/cold/uninterrupted4412+64 continuation
**passes**, with cached4475 versus0. Prefix/warm/cold IDs match the EP control;
only EP uninterrupted differs at global76. This narrows the observation to the
experimental EP combination, not a universal MTP or migrated-TP2 failure, but
neither identifies its numerical cause nor certifies general EP serving.

The requested SWE path was then actually attempted: client695dd8b, canonical
prepared workload, C2/60s, seed20260924, real MTP2,2 chips,262144 context.
`tp-serial-swe-c2-valid` aborts after~13.9s in **draft** owned-FIA planning because
a real query exceeds the compiled Q16 envelope (`plan.py` retains the stale
error text "Only Q1..3"). This is not the cold-request/continuity gate and is
not a measured leaderboard point. Do not filter the failing turn, disable MTP,
change the prepared corpus, or enlarge only the Python guard to manufacture a
score. The first client invocation omitted `/v1/completions` from its URL and
failed before requests; its log is retained separately, not counted as a model
failure. No900s point was launched after the real protocol failure.

A SWE rerun now needs a separately qualified wider draft-attention baseline;
that is more than turning on this expert-overlap patch. The primary multi-DP
route still needs four assigned cards. Stop the optional campaign here rather
than silently expanding it into a numerical-attention port. Four-card allocation
was requested; no new allocation has arrived. Default remains off and website
unchanged. `overlap-tp-swe-control-20261003.tgz` in the local artifact directory
retains the full TP control, both client logs and protocol failure. The API server
has exited and both assigned cards are idle.
