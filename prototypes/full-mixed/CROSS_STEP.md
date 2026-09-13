# Cross-step authorization experiment

Scope: fixed active-set K5 verification, at most four requests, DSV4 DSACP,
TP8 with DCP size one. Builds on the qualified target/draft FULL prototype.
No installed runtime or release pin is changed. Correctness gates and two unprofiled comparisons passed. Fully prewarmed
cohort timing is reported separately below.

## The actual dependency cut

The donor already maintains authoritative device progress:
`update_num_computed_tokens_for_batch_change` adds the previous valid sampled
count to prior device progress, then device operations construct positions,
sequence lengths and slot mappings. Its async scheduler already authorizes work
optimistically. This is not a reason to replace the scheduler.

Two CPU consumers still force the host to catch up before target submission:

1. `_correct_optimistic_seq_lens_cpu` waits for valid-count D2H before producing
   CPU lengths. In the admitted DSACP decode path these lengths determine
   SAS/QLI **maximum tiling lengths**; actual attention lengths stay on device.
2. `execute_model` invokes deferred CPU state correction early for compressed
   attention, then fills `_dsa_positions_cpu_buf`. DSACP derives RoPE and
   compressor inputs from device positions and does not consume that CPU buffer.
   Other backends are not admitted merely because they also compress KV.

`cross_step.py` leaves optimistic CPU lengths as conservative tiling bounds,
and defers CPU bookkeeping until AFTER enqueuing the target. The original
callback still executes once in the same model invocation, before sampling and
before another request-state update. No output receipt or scheduler correction
is dropped, fabricated or applied to a later request generation.

## Ownership, not just fewer synchronizations

The native `synchronize_input_prep` event still protects reused pinned host
buffers. The late callback ALSO waits for the current preparation event before
mutating CPU state: some state tensors were asynchronous H2D sources. A device
wait cannot authorize the host to overwrite those sources.

The ordering is:

```
retire previous input DMA -> prepare authorized bounds and enqueue device metadata
                         -> enqueue target FULL
                         -> retire current input DMA -> apply previous CPU receipt
                         -> enqueue sampler and draft FULL
```

This moves the receipt wait behind model submission; it does not remove every
host wait. Device metadata writes and replay remain on the same stream, so no
new unsynchronized multi-stream access to graph inputs is introduced. Existing
draft/copy streams and scheduler retirement stay under donor ownership.

This is a bounded dependency transformation toward continuous submission, NOT
an implementation of arbitrary N+2 admission, new request-generation slots,
separate H2D metadata banks, or a new serving engine. Those changes require an
observed residual need and independent ownership checks.

## Numerical reference

Merely replaying the graph twice with the SAME new metadata would not qualify
this change. In shadow mode the reference instead:

- restores the original synchronized exact CPU lengths;
- asserts these equal the device lengths and are no greater than our bounds;
- rebuilds attention metadata with the original builder;
- replays the unchanged native graph from the identical KV/MTP state;
- compares valid outputs/MTP and all unique KV pool bytes;
- restores the candidate bounds/metadata and its post-execution state.

Draft's separate initial-capture and subsequent replay checks remain enabled.
All synchronization and state cloning for these oracles are excluded from
performance runs. Bounds may differ from exact lengths; metadata bytes need not
match when tiling changes, but numerical outputs and owned state must pass.

## Admission and fallback

Require unchanged request order, all K+1 query counts, five scheduled draft
slots, no new requests, prompt already consumed, and an available prior device
count/draft. Length bounds must be positive and within the configured model cap.
Unsupported shapes and transitions retain ordinary correction. DCP and other
attention builders are rejected at configuration. CPU tests exercise admission,
bounds, reference restoration and DMA-before-bookkeeping ordering.

Use `--cross-step-bounds` with the existing ordered/CPUQLI/draft flags. For an
alternating same-engine control use `--cross-step-study --rounds 4` instead of
`--decode-study`; both arms retain target/draft FULL and differ only in this
cross-step transformation. Native target control remains FULL_DECODE_ONLY;
prefill/mixed is deliberately outside the optimized admission.

## State qualification

[Compact receipts](cross-step-shadow.json) identify all eight ranks:

- Run027: bounds-only dummy control, 65 exact-metadata shadows/rank.
- Run028: bounds plus late CPU callback, 65 exact-metadata shadows/rank,
  65 late commits. Including fallback verification, 67 target checks/rank.
- Run029: full real weights, 17 exact-metadata shadows and 17 late commits/rank,
  21 total target checks/rank. Every observed output/MTP difference was zero;
  all compared unique KV pools were byte-identical. CPU upper bounds exceeded
  exact lengths by 0–5 tokens, so this is not only an all-accepted dummy case.
  Real draft banks 1/2/3/4 each passed the initial invocation and respectively
  2/1/3/8 subsequent graph/eager checks per rank.

All use strict HCCL, the unchanged native target graph as reference, and restored
identical state. Real shadow reserves 3 GiB KV/rank for comparison snapshots;
that is not a production capacity setting. These observations qualify the
bounded paths, not arbitrary long-running slot reuse or service throughput.

## Run030: isolate the dependency cut

Real weights, ordinary HCCL, one loaded engine, alternating OFF/ON/OFF/ON.
Both arms already use target/draft FULL, ordered target replay and CPU QLI.
The four-request K5 selection contains 21 intervals/rank/phase.

| Cross-step | Cycle ms | Draft-end to next target ms | Cohort seconds | Positive waves |
| --- | ---: | ---: | ---: | ---: |
| OFF | 51.503 | 6.053 | 3.382 | 43 |
| ON | 46.055 | 1.569 | 4.154 | 44 |
| OFF | 51.504 | 5.847 | 3.518 | 45 |
| ON | 46.152 | 1.296 | 3.303 | 46 |

Rank-0 medians; [all-rank data](cross-step-030.csv), [protocol and outlier](cross-step-030.json).
All-rank median cycles cluster at 51.36–51.61 ms OFF, 46.06–46.23 ms ON.
Matched-wave latency falls about 10.4–10.6%; the main change is the intended
inter-forward interval rather than faster neural operators. These intervals
still include actual metadata/copy work, not only hardware idle time.

The first ON cohort contains one two-request draft call of 901.98 ms host time
(900.92 ms device-event span); the next target-to-target cycle is 945.38 ms.
This is consistent with first bank capture, not proved by a capture-specific
timestamp. The second ON cohort has no such spike. Thus acceptance variability
alone does not explain these cohort timings. Run031 explicitly warms all four
banks and checks their existence on all ranks BEFORE any measured cohort.

The separate run030 profile confirms eight late callbacks per rank. On rank0,
all eight late-receipt host scopes total 1.20 ms; input-DMA retirement scopes
still total 190.32 ms. Those waits occur AFTER enqueueing target and overlap
queued device work; adding them to device cycle latency would be incorrect.
Native eight-rank TraceLoom export is under
`/workspace/strengthen-dsv4/runs/hw3-cross-step-030/analysis/`.
Its prefill/mixed beginning deliberately remains native FULL_DECODE_ONLY;
this is not a claim that the whole trace is FULL replay.

## Run031: prewarm the whole bounded bank set

`--warm-draft-banks` performs counts 1–4 with 32 output tokens each, then
asserts all four graph entries exist on EVERY rank before starting observations.
Warmup cost was 12.49 seconds, excluded from measured cohorts. This distinguishes
warmed throughput from cold shape-admission latency rather than hiding that cost.
All six measured phases have maximum draft host calls below 40 ms; no 901 ms
spike remains. No graph bank beyond the admitted four counts can be created.

| Cross-step | Cycle ms | Draft-end to target ms | Cohort seconds | Positive waves |
| --- | ---: | ---: | ---: | ---: |
| OFF | 51.810 | 6.452 | 3.757 | 50 |
| ON | 46.522 | 1.574 | 3.142 | 42 |
| OFF | 52.322 | 6.963 | 2.950 | 34 |
| ON | 46.231 | 1.298 | 2.765 | 34 |
| OFF | 51.672 | 6.386 | 3.403 | 43 |
| ON | 46.145 | 1.288 | 3.378 | 48 |

Rank-0 matched-wave medians, 21 selected intervals/rank/phase. Cycle reduction
is 10.2–11.6%, consistent with run030. Each paired cohort finishes faster, but
by 0.7–16.4%, with different speculative/batch trajectories. The equal-wave-count
pair is 2.950 -> 2.765 seconds (6.3% shorter). These small, fixed-order short
cohorts do not establish production service throughput or a high-concurrency SLA.
[All-rank CSV](cross-step-031.csv), [protocol/warmup/cohort evidence](cross-step-031.json).

All NPU runs released their leases and devices. CPU contracts total 17 tests.
The deliverable is an opt-in, reversible dependency cut, not a wholesale
continuous-serving default. Prefill/mixed, DCP, more than four seats and general
N+2 request-generation/retirement protocols retain their previous boundaries.

For run030's distributed view use the eager-marker candidate fit. The initial
all-provider marker set had a rank2 holdout P95 of 113 ms due to inconsistent
replay-associated endpoints; it is not used for the final aligned export.
Native-task membership (not residual-based pruning) gives 0.86–1.51 us P95
with eager markers. Same-rank unprofiled cycle measurements do not depend on
this display alignment. See the profile helper note for retained failed evidence.
