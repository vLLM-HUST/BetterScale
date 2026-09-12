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
