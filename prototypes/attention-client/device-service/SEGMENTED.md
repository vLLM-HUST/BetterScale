# Intra-wave segmented expert pipeline (opt-in prototype)

2026-09-16. Start from `PERSISTENT.md` and `GMM-SCHEDULING.md`.
This adds a two-segment publication protocol to the existing persistent AIV/AIC
service. It does not replace GEMM, add host per-wave scheduling, or alter the
published BetterScale worker. `DEVICE_SERVICE_SEGMENTED=1` enables it; default
remains unsegmented.

## What overlaps, and what does not

After grouping, freeze two cumulative-count catalogs. Each segment uses original
expert weight IDs, but its counts and input/output pointer are relative to its
own packed-row slice. No activation copy or new large matrix workspace is needed.
The catalogs and four small GEMM descriptors add 2240 bytes per slot.

Default cut: the first whole-expert boundary reaching half the routed rows.
Optional `DEVICE_SERVICE_SEGMENT_TAIL_EXPERTS=2` leaves the last two nonempty
(layer, expert) groups to the tail, closer to the pinned DFC epilogue policy.
Do not split a hot expert's rows just to balance segments. Empty segments are
legal, including all-zero work and a single expert holding all 512 routes.

AIC and AIV each still execute one command at a time. After all 24 AIC cores
finish up[0], the coordinator may start up[1] and publish activation[0] to the
16 AIV movers. Activation[1] can overlap down[0]. Every down segment waits for
all of its activation producers. The return/DONE stage waits for both down
segments; slot counters and catalogs cannot be reused before all return writers
finish. Source frame ownership and cross-server generation retirement are unchanged.

The compute lifetime uses independent up/activation/down completion counters,
not a single mutable stage shared by two overlapping engines. The existing UP
stage now covers that lifetime; PULL/PACK and RETURN remain exclusive. The
unsegmented control runs the same scheduler with one segment.

This is **not yet DFC's continuous internal tile pipeline**: each segment calls
the existing CATLASS adapter separately, with a coordinator-mediated all-core
join. This avoids copying matrix implementations but adds initialization and
publication costs. Faster-looking overlap alone does not prove a net win.

## Gates and evidence

* Half-cut leaf: `runs/persistent-control-20260916T070229Z`.
* Tail-two leaf: `runs/persistent-control-20260916T070856Z`.
* Both pass independent random BF16 weight oracles for broad/hot/zero/skew/one-row
  routes, immutable source payloads, unowned output poison and end guards.
  Invalid descriptors and absent sources terminate with bounded error status.
* Four-card changing-layer/input/route episodes use two sources and two servers,
  24 jobs/source, the existing synthetic oracle, and fresh idle-subset admission.
  `persistent_analyze.py` joins events by engine generation and validates
  pack -> up[p] -> activation[p] -> down[p] -> return for every segment.
* Heterogeneous 32/1-row sources with a 2 ms source delay:
  `runs/remote-dfc-control-20260916T070743Z`, 48 outputs pass; servers process
  42/43 waves with 6/5 naturally paired waves. Same-wave activation/up routine
  overlap is 133.58/98.04 us total. Preparation/math overlap also remains present.
  This is a correctness/overlap case, not matched performance against a baseline.

Internal timings use per-core device intervals, exclude mailbox wait, and have
independent per-device zero origins. The profiler's separate provider-clock view
is not independent cross-device calibration. Never send raw timeline JSON.

## Reproduce

Build with `build_persistent.sh` and select its output with `PERSISTENT_BUILD`.
Python config layout and both engine binaries must come from the same source
closure; launchers freeze them into the run capsule.

For leaf, use `run_persistent.sh <idle-card>`. For four-card, use the existing
`run_remote_dfc.sh` recipe in `PERSISTENT.md`, plus the opt-in variable above.
`DEVICE_SERVICE_INTERNAL_TIMING=1` records per-core intervals. Work-time storage
now admits 512 generations per engine to cover both segments for bounded episodes;
this allocation exists only when instrumentation is enabled.

`persistent_analyze.py RUN` exports compressed phase and core-routine timelines,
checks causal ordering instead of requiring serial phase completion, and reports
same-wave overlap separately from overlap between different request slots.

## Matched performance: overlap passed, net speedup did not

Same physical server cards 5/7, common prequeued startup, both sources32 rows,
24 paired waves on both servers, identical arithmetic and instrumentation:

| Path / run suffix | Server0 pack-to-return median | Server1 median | Server0/1 active episode |
|---|---:|---:|---:|
| Unsegmented / 070649 | 603.15 us | 546.85 us | 18.598 / 18.347 ms |
| Half split / 070716 | 609.73 us | 565.89 us | 18.945 / 18.797 ms |
| Tail2 / 070940 | 625.27 us | 591.31 us | 19.268 / 19.615 ms |

All48 outputs pass per run; maximum relative L2 stays below0.0003.
Half-split same-wave activation/up overlap is184.18/176.30us total; tail2
increases this to299.04/275.06us. Nevertheless both are slower in these controls.
The measured tail up/down commands cost roughly34/28us, illustrating the cost
of creating tiny separately dispatched segments. Resource initialization,
coordinator joins and scalar scheduling are plausible contributors; their
individual costs have not been isolated. These are bounded observations,
not statistically established universal regression percentages.

The independent-arrival pair070456/070523 improves client episode25.129 ->
18.736ms, BUT changes naturally paired waves from0 to24. It is not evidence
that segmentation itself provides that speedup. Common-start client event spans
also include startup waits; use the matched server intervals above for this claim.

**Decision:** retain the opt-in prototype and current unsegmented default. Do not
ship a latency-win claim. A future optimization would need to publish progress
inside one persistent CATLASS group loop and keep its resources alive, rather
than merely issuing more complete GEMM calls. That is a deeper change and has
not been implemented or justified by a measured net benefit here.

Full compact receipts: `segmented-result.json`. Heterogeneous profile artifacts:
`runs/remote-dfc-control-20260916T070743Z/analysis/` contains
`persistent-work-relative.json.gz`, `persistent-phases-relative.json.gz`, and
`attention2-expert2-provider-clock.json.gz`.
