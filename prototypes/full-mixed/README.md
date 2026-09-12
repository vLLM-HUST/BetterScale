# DSV4 FULL mixed prototype

**Experimental worker extension; not a production-enabled plugin.** It imports
installed donor packages matching the pinned Ascend source. No installed package
or upstream submodule is edited. Startup/worker code is copied into each run
capsule before launch.

Current accepted bounded result: `runs/full-mixed-012-valid-rows` (2026-09-12),
TP2 physical cards 0,2; dummy four-layer DSV4 `[SWA,SWA,C4,C128]`, real attention
sizes, eight routed experts, W8A8, shared-expert overlap retained, no speculation.
FULL buckets 8 and 256, request capacity4. Thirteen requests across four cohorts
completed. Each rank checked39 replays (9 large,30 small): valid output max
absolute difference0, complete KV max absolute difference0 against eager from
identical state. All generated token sequences match the independent NONE-mode
run `runs/full-mixed-008-eager`. Not a production-model or TP8 acceptance.

## Why the patch is more than an ALWAYS flag

- DSACP prefill RoPE was transient; target build now updates persistent buffers.
  Explicit drafting metadata remains on its original private-buffer route.
- Runner's generic FIA padding helper changed request dimensions under a graph.
  Capture had4 requests; replay supplied2 and left captured tail offsets stale.
  The native fault was Compressor. DSACP instead retains descriptor request
  capacity and fills inactive query-offset tails with the last actual offset.
- Target metadata allocation/scalar bounds use captured token/request capacity;
  device query lengths still encode actual work. Padded state rows are cleared
  through the builder's existing real-request handling before capacity binding.
- The original replay synchronization and shared-expert dependencies remain.

The shadow checker restores the same KV state for eager/replay and then restores
replay state for serving. Only valid output rows are semantic outputs: FlashComm
returns TP-local token rows and inactive rows are unspecified. KV is checked in
full. Output and state use rtol=.01, atol=1e-6, with actual observed max diff0.
Early coarse `.01` absolute checks were insufficient for small dummy activations;
run012 supersedes those, rather than reporting run010 as tight acceptance.

`wo_a` dummy layout repair reproduces `ops/linear.py`'s checkpoint loader reshape;
it is separate from FULL mixed behavior and only runs for dummy loading.

## Running

Uses the existing local donor runtime and CANN9.0.1. No package installation.
The launcher takes `/root/tp8.lock`, admits only selected healthy idle cards,
monitors selected-card ownership, and reclaims only its own process group.
Other cards may be busy. Do not interpret subset timings as exclusive TP8 data.

```sh
FULL_MIXED_SHADOW=1 PROBE_DEVICES=0,2 \
 PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/CHOOSE-A-FRESH-CAPSULE \
 bash prototypes/full-mixed/run.sh --tp 2
```

Use `--mode NONE` for eager control; `--spec` and `--budget` are exploratory
extensions, not accepted simply because the arguments exist. Result status is
intentionally conservative; inspect shadow receipts and scope before claiming
correctness. Shadow memory/timing includes full cache clones and comparisons:
it is NOT a production memory/performance result. Next gates: longer prefixes,
K5, larger budget, TP8, then realistic model/service validation and packaging.

CPU test (runtime Python has NumPy):
```
/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env/bin/python \
 prototypes/full-mixed/test_contract.py
```

The initial supervisor classified two exiting, already-owned workers as foreign
when they lost parent/PID visibility. It was repaired to retain proven host PID
identity and include live owned session members. Native kernel failures in runs
005/009 remain failures, not evidence of genuine foreign interference. No foreign
process was killed. Generated CANN exception dumps are ignored, not published.

## K5 fixture and bucket trap

Run015 (K5, TP2) completed 17 requests and checked29 small replays/rank with
zero valid output, KV and pre-HC MTP-buffer differences. **Not mixed-prefill
acceptance:** only the24-token bucket survived upstream sizing. With maximum256,
rounding256 up to a multiple of K+1=6 produces258 and silently drops it. Use a
large budget divisible by both TP and K+1 (e.g.288 for TP2/TP8 K5), and require
receipts to show the large bucket captured **and replayed**.

`fixture.py` shrinks the draft config separately; dictionary target overrides
are not propagated by upstream. Target callable overrides conflict with Ascend
quantization, so this repair composes the draft-only config transform. It is not
part of the proposed serving patch. Run013's missing aux-state tuple and run014's
quantization validation error were fixture failures, not kernel correctness data.

## Larger-bucket correctness

Run018 passed TP2 K5 with FULL24/4128:19 requests,44 checked replays/rank,
including3 mixed waves and up to4112 valid input tokens in one wave. All valid
outputs, complete KV and valid MTP side-buffer rows had max difference0. See
`k5-4k-tp2-result.json`. Run017's OOM was in the oracle's whole-cache FP32
conversion, not capture; chunked comparison retains whole-state coverage.
Before shadow, native logs report0.98GiB graph memory/rank and13.39GiB allocated
/14.08GiB reserved after warmup. These are FOUR-LAYER dummy figures, not a full
model capacity estimate. The later~58GiB shadow peak is not serving memory.

For hw3, override PROBE_RUNTIME, PROBE_MODEL and PROBE_HELPERS; the launcher uses
that execution user's home `tp8.lock`. Transfer the helper dependencies together
(`probe_host_npus.py`, `supervise.py`, `idle_gate.py`). Dummy model config can be
copied without any weights. Key installed DSACP/runner/ACLGraph files on hw3 were
byte-compared to the pinned source before the TP8 experiment.

## Timing pilot and current TP8 boundary

`timing-tp2-pilot.json` compares run019 FULL with run021's original
FULL_DECODE_ONLY target path (`FULL_MIXED_PATCH=0`), same dummy K5 workload.
After one warm round, two measured repetitions:6/7 cohorts improved, one worsened;
generated tokens all matched. The4145+19-input cohort was~0.574s vs~0.458s.
This measures completion of these small cohorts (eight outputs/request), NOT
isolated prefill latency or a production-model speedup. NONE-mode run020 is a
secondary reference, not the donor's best decode baseline.

TP8 run001 stopped at native MRV1 alignment: max(6,8) is not their common
multiple. The opt-in extension now uses LCM24 for sizing while retaining K5.
Runs002/003 captured both buckets but failed strict shadow checks: small BF16
output differences and larger KV differences including NaN mismatches. **TP8
is not accepted.** Run004 compares eager against eager from the same restored
state to separate an oracle/state-reproducibility issue from graph-specific
failure. Do not relax tolerance or relabel NaNs as rounding. First-failure files
are retained without overwrite by subsequent queued work.

### TP8 diagnosis resolved; bounded FULL result

With `HCCL_DETERMINISTIC=strict`, run006 eager/eager passed66 steps/rank;
run007 actual FULL/eager passed65/rank, including6 mixed waves, all differences0.
This is TP8 dummy K5 at24/288, not yet a real-weight performance claim.

The NaN reports above came from a flawed whole-pool typed comparison: allocator
`shared_by` maps BF16 SWA and FP32 compressor-state views to the same backing.
A page belonging to FP32 state must not be interpreted as BF16 KV. The current
oracle snapshots unique untyped storages once and checks exact bytes under the
deterministic control. This is stronger state coverage, not a relaxed tolerance.
Do not enable deterministic HCCL in production merely to make tests look good.

Run008 was stopped by its own supervisor after15 zero-difference checks because
byte comparisons unnecessarily converted full chunks to FP32 and accumulated
several reductions. Run009 uses direct equality for byte chunks. Stop/relaunch
changed only the oracle's cost, not the candidate graph or pass predicate.

### TP8 4K and full-weight execution

Run009 passed TP8 K5 FULL24/4128:63 checks/rank,9 mixed waves, up to4112 valid
input tokens. All unique KV pools matched byte-for-byte; valid output/MTP rows
also had max difference0. See `k5-4k-tp8-result.json`.

Run010 then used the complete real checkpoint on hw3, without shadow or strict
HCCL:38 requests completed across two rounds, and rank0 recorded18 large-bucket
and60 small-bucket replays. The extension did not apply dummy layout/config
repairs (`--real`). FULL_DECODE_ONLY control is run011. Do not confuse dummy
same-state proof with output-quality evaluation on real agent workloads.

`--real` removes all model-size overrides and lets donor size KV automatically
at85% memory utilization. Supply PROBE_MODEL explicitly on the other host.
The opt-in worker extension remains experimental; installed donor and release
submodules are unchanged. FULL here captures target forward, not the scheduler,
sampling, or the eager DSpark drafter.
