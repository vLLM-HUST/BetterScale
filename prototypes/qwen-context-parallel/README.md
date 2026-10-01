# Qwen35 quota-FIA leaf qualification

`probe.py` compares the built Q8/KV1/D256 kernel with native FIA, two captured
banks, nonuniform Q1..3, contexts through16×256K and guarded output/workspace.
It also checks immutable inputs and an independent CPU reference on short edges.

Build with the main module's `build.py`, then run this probe only inside a fresh,
supervised selected-device lease. It imports torch-npu and initializes logical
NPU0; the launcher must expose exactly one admitted physical card. The native
FIA host adapter must be preloaded before Python startup.

```sh
LD_PRELOAD=/path/to/qualified/libbs_fia.so python probe.py \
  --output /path/to/new-receipt \
  --source ../../src/betterscale/patches/qwen_fia/context_parallel \
  --kernel /path/to/build/libbs_fia_cp.so \
  --native /path/to/qualified/libbs_fia.so --device-lengths
```

`--device-lengths` fixes the host plan while device lengths change across tile
boundaries and down to tiny contexts, including empty partials and13-query
zero-KV padding. Omitting it exercises exact host/device lengths and full/split
plan transitions. This is a correctness gate, not a serving performance score.
The real serving integration is now in main's `wave.py`; no external capsule
stager or alternate Worker is part of the feature.

For a bounded failure investigation, `--case <name>` selects one existing case;
that subset does not qualify the full gate. A numerical mismatch writes a small
failure.json / failure.pt capsule with live/padding errors and independent CPU
reference errors for rows with KV <=4096. Full KV tensors are not dumped.

CANN9.1 observation: native FIA produced nonzero zero-KV padding in the moderate
case, while the candidate produced zero and live-row max error was0.000244.
The probe therefore compares real rows to native and independently demands
exact zero padding in BOTH candidate banks (including initial execution).
It does not relax live-row tolerance. This separates two distinct contracts.
Subsequent extreme-case replay had a real live-row mismatch; CANN9.1 candidate
is not qualified merely by fixing the padding reference.

### CANN9.1 investigation boundary, 2026-10-01

Fletcher chose to fork the native-reference issue investigation and keep the PD
mainline on owned FIA. Do not continue the native bug hunt in the PD task.
Use `--reference cpu` for the independent full correctness gate: every valid
row is compared to FP32 CPU causal attention, including long contexts, with
reference caching only for identical immutable KV / query-sign / length inputs.
Padding remains exact zero, both banks and all guard checks remain enabled.

Evidence lives at /workspace/betterscale-pd-runtime on hw180:
- fia-cp-oracle/failure.json: extreme replay5, request11 CPU max error
  candidate0.000586 vs native0.156696; other short rows were close. This localizes
  the wrong reference value, NOT its root cause.
- native-fia-reference.json and native-fia-reference-dynamic.json:
  standalone native-only control passed16 repeats each with/without zero-KV
  padding, both fixed and varying lengths.
- native-fia-with-planner.json: native plus two metadata-only planner calls also
  passed. No owned CP kernel runs in these controls. Therefore a generic
  standalone CANN bug is NOT yet established; the full mixed harness interaction
  remains relevant. The frozen failure output and original logs are retained.

`native_reference_probe.py` is the bounded control for that fork, not a model
qualification test. Keep it dormant in PD work unless new evidence requires it.

Independent CPU gate (`fia-cp-cpu-oracle-2`) passed edges, c16-short,
nonuniform, moderate and extreme, including all16 replays per case. long16
passed both initial banks at256K, then failed replay0 (KV3) on1/98304 live
elements: absolute error0.00321957, at the unchanged rtol0.02/atol0.003
criterion. Padding stayed zero. This is a strict gate failure, not yet a
diagnosed owned-kernel defect; native-reference tolerances are not automatically
a complete FP32 mathematical error budget. Do not mark this candidate qualified
or update native artifact pins from these partial results. PD recurrence and
checkpoint work can proceed independently.

### Independent numerical contract, 2026-10-01

The single-element strict FP32 discrepancy is explained for the captured failing
wave, not erased: the vendor arithmetic used by our kernel declares ElementP as
BF16 and `DownCastP` rounds the **unnormalized exponentials** with CAST_RINT
before PV; row sums remain FP32. In
`fia-cp-cpu-rounding/failure.json`, independently emulating that single-tile
arithmetic gives **exactly zero difference for all16 request outputs**, including
the previously failing element. No native attention operator is used.

`--reference cpu-kernel` therefore uses that source-derived CPU arithmetic for
KV<=128 (one tile), and the original FP32 mathematical reference for longer
contexts. It leaves rtol0.02/atol0.003 unchanged and retains the separate
`--reference cpu` strict mathematical diagnostic. The complete device-length
gate passes all six cases, two banks,16 replays each, fixed upper-envelope
schedules through16x256K and exact zero padding, workspace/output guards and
input immutability. Largest reference error is0.001476. Receipt:
`/workspace/betterscale-pd-runtime/fia-cp-cpu-kernel/complete.json`. This qualifies
the bounded owned CP leaf on the new runtime, not full serving or the native FIA
reference. No inference about the parked native issue's root cause is needed.

Source for the precision boundary is CANN9.1's
`flash_attention_interface.cpp` (ElementP=InputDtypeQ) and
`attn_infra/epilogue/block/online_softmax/`
`fused_block_epilogue_online_softmax_softmax.inc.hpp` (`DownCastP`).

The exact-host-plan variant (without `--device-lengths`) also passed all six
cases and all16 replays, exercising full/split plan transitions. Receipt:
`/workspace/betterscale-pd-runtime/fia-cp-cpu-host-plan/complete.json`.

### Serving integration: metadata is not native launch identity

The first new-runtime TP2 model produced32 tokens, but the long-request test
failed the old non-FD/24-block guard. A bounded planner-only observation found
Q3/KV4410 -> FD/9 blocks and padded Q1/Q3 -> FD/10 blocks, while Q4096 prefill
remained non-FD/24. This is a serving admission gap, separate from the parked
native numerical-reference anomaly.

Owned attention now borrows only the validated2528-byte geometry. It releases
the newest metadata-only plan without retaining a native function/grid identity
or accumulating plan tombstones. Native-launch paths retain their strict guard.
The owned launch allocates the admitted128MiB scratch bound, stable across
metadata-plan changes. Capacity3 target attention and small MTP frames use the
owned route; later one-token MTP steps may use larger capacities. Large draft
frames explicitly zero output before launch: the Q1..3 kernel only initializes
one small padding tile and must not be mistaken for an arbitrary-padding fill.

`--wave-planner --case single` passes two banks/16 replays with actual serving
Planner metadata changing between KV3 and256K, comparing to independent CPU.
`--wave-planner --initialize-padding --device-lengths --padding-tokens 4095
--case draft-padding` passes one live query plus4095 padding rows. The naked
kernel test failed padding, as expected from the discovered small-tile limit;
the explicit zero+kernel composition passes guards, live output and exact zeros.
Receipts are `fia-wave-single-2` and `fia-wave-draft-padding` under the task
runtime root. The new `plan_discard_latest` host ABI needs the rebuilt adapter;
old native manifests are NOT promoted by these tests.18 affected CPU tests and
the native C++1000-wave lifecycle check pass. Model validation remains pending.

A subsequent full-model request exposed a separate metadata normalization gap:
MTP may put zero-KV padding inside the first16 rows, while the old compactor
kept those rows and folded only rows17 onward. `compact_padding` now preserves
the exact1..16 live prefix and collapses the entire zero suffix, rejecting holes
and seventeen live requests. This is validated for capacities through4096;26
affected CPU tests pass. The owning CP planner still rejects noncanonical input.
The model-level run before this fix failed closed, not a successful continuation.

### Short-prefill owned route on CANN9.1

The model's4412-token long prefill plus64-token generation passes, but the
warm4476-token continuation selected a16-token short-prefill bucket and hit
native FD/10 admission. Extend owned query geometry toQ1..16 and capacity16,
not just a Q1..3 bypass. Empty-partial output uses128KiB UB at Q16, so its LSE
fill moves to128KiB to avoid overlap (AtlasA2 UB192KiB).

`fia-wave-q16/complete.json` passes mixed Q1/4/8/16 through256K context with
real serving metadata, two banks and16 replays, independent CPU-kernel oracle,
guards and input immutability; max error0.001377. The six original cases also
pass with dynamic host metadata in `fia-wave-q16-regression/complete.json`,
max error0.001476. Both use the new `fia-cp-q16-build/libbs_fia_cp.so` under
`/workspace/betterscale-pd-runtime`;26 affected CPU tests pass. These are leaf
and metadata-composition gates, not yet successful model warm continuation.
