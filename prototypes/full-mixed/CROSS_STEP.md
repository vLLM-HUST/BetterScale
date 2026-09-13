# Cross-step authorization experiment

Scope: fixed active-set K5 verification, at most four requests, DSV4 DSACP,
TP8 with DCP size one. Builds on the qualified target/draft FULL prototype.
No installed runtime or release pin is changed. Results below are pending.

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
