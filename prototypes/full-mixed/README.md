# DSV4 FULL prefill/mixed target graph

**Opt-in experimental worker extension, not a default-enabled production plugin.**
Pinned vLLM0.25.1 + vLLM-Ascend0.25.1rc1. No installed package or upstream submodule
is modified; source is copied into each run capsule before launch. Shared-expert
overlap and original replay synchronization are retained. FULL captures target
forward, not scheduler, sampling, or the eager DSpark drafter.

## Evidence and current boundary

- **TP2 target-only:** run012, four-layer SWA/SWA/C4/C128 dummy, FULL8/256,
  13 requests,39 checks/rank, valid output and whole KV-view differences0.
  Independent NONE-mode generated tokens matched. `target-tp2-result.json`.
- **TP2 K5:** run016 FULL24/288 and run018 FULL24/4128 passed same-state checks
  including mixed traffic and MTP side outputs. `k5-tp2-result.json` and
  `k5-4k-tp2-result.json`.
- **TP8 K5:** hw3 run007 FULL24/288 passed65 checks/rank with6 mixed waves.
  Run009 FULL24/4128 passed63/rank with9 mixed waves and up to4112 valid input
  tokens: valid output/MTP differences0 and **every unique KV backing pool
  byte-identical**. These use `HCCL_DETERMINISTIC=strict` as a correctness control.
  `k5-4k-tp8-result.json`. The mixed counts also satisfy valid_tokens >6×decodes,
  so inactive padded rows are not the sole evidence of prefill work.
- **Full real checkpoint:** hw3 run010 completed38 requests in two rounds under
  normal HCCL, with18 large and60 small replays on rank0. Run011 original donor
  FULL_DECODE_ONLY also completed. Real-weight same-state shadow run012 passed42 steps/rank, including2 actual
  mixed waves and4112 valid tokens: output/MTP differences0 and all KV pools
  byte-identical. See `real-tp8-shadow.json` for strict-HCCL/3GiB-KV scope and
  the separately corrected supervisor teardown false-positive.

Scope tested: TP2/TP8, four request seats, C4/C128, K5, shared-expert overlap,
small and~4K buckets, changed lengths, request counts and chunked prefixes.
Not yet agent-workload/SLO acceptance, long-context capacity, arbitrary graph
buckets/backends/parallel configurations, or a universal performance claim.

### Normal-performance pilot

`timing-tp2-pilot.json` compares matched four-layer dummy modes (one warm round,
two measured rounds). `real-tp8-timing-pilot.json` compares full real weights on
hw3 (one warm round, ONE measured round):6/7 cohorts were faster, one slower.
The4145+19-input cohort completed in~1.062s donor vs~0.632s FULL; the sum across
seven measured cohorts was~6.333s vs~6.023s. These are small-cohort completion
times with eight output tokens/request, not isolated prefill times or stable
throughput estimates. The two sparse buckets24/4128 can overpad shorter work.

Real greedy streams differ both across arms and when the unchanged donor repeats
itself; speculative acceptance work can therefore differ. Do not use the pilot
as numeric-equivalence evidence or present a favorable cohort as overall speedup.
Shadow runs include snapshots/comparisons: their timings and peaks are NOT
serving performance or serving memory.

## What changes

1. Opt into DSACP FULL coverage within the explicitly selected worker extension.
2. Target prefill RoPE updates persistent buffers instead of ephemeral tensors.
   The explicit drafting builder retains its private-buffer route.
3. Keep descriptor request capacity. The native generic FIA helper shrank replay
   metadata despite capture retaining four rows, causing a Compressor fault.
   Repeat the final actual query offset through inactive rows: zero-length seats,
   not an invented FIA dummy request. Stable pointers alone are insufficient.
4. Bind target token/request allocation bounds to the captured capacities;
   device query lengths still encode real work. Existing real-request handling
   clears inactive state-map rows before capacity binding.
5. Use LCM(K+1,TP) for capture alignment: K5/TP8 needs24, not max(6,8).
   This changes bucket sizing only, never speculative length.

Inspect `extension.py`. Dummy-only `wo_a` conversion reproduces the real checkpoint
loader's layout reshape and does not run for real weights. `fixture.py` shrinks
DSpark's config separately because upstream dictionary target overrides are not
inherited by draft. Neither fixture change belongs in an engine patch.

## The state oracle

Replay and eager start from the same restored KV pool and MTP side state.
After checking, restore replay state before serving continues. Only valid output
rows are semantic; FlashComm1 returns TP-local rows. MTP uses its valid global
prefix. Numerical output checks use rtol=.01, atol=1e-6; observed passing maximum
is0. Do not reuse the early coarse absolute .01 tolerance for tiny dummy outputs.

Donor's `shared_by` allocator exposes BF16 SWA and FP32 compressor views over the
same storage. Comparing the whole pool through every typed view interprets other
groups' state bytes as that view's dtype, producing apparent NaNs/huge numbers.
Current snapshots deduplicate untyped storages and compare exact bytes, avoiding
both reinterpretation and redundant whole-pool copies. A tolerance-based cache
oracle would need group-owned pages and their true dtype; this code does not
pretend otherwise.

Non-deterministic TP8 eager/eager itself exceeded tight tolerance. With strict
HCCL, eager/eager run006 passed66 checks/rank with all differences0, then actual
FULL/eager run007 passed. Strict HCCL is a correctness control, not a production
default. `FULL_MIXED_ORACLE=eager` is an oracle diagnostic, NEVER graph acceptance.

Historical failures retained in capsules: stale request-capacity Compressor fault;
missing draft aux IDs; target callable override conflicting with quant config;
256 rounded to258 and discarded under K5, leaving only24; whole-cache FP32 shadow
OOM; expensive byte diagnostics. Run008 on hw3 was deliberately stopped after15
passing checks to replace costly byte-to-FP32 diagnostics with direct equality.
No candidate graph change was made for that last optimization.

## Run

Uses an existing donor runtime + CANN9.0.1. No build/install. The launcher takes
the execution user's home `tp8.lock`, admits only selected healthy idle cards,
monitors ownership and reclaims only its own process group. Other cards may be
busy; subset timings are not exclusive TP8 measurements.

```sh
FULL_MIXED_SHADOW=1 PROBE_DEVICES=0,2 \
 PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/CHOOSE-FRESH-NAME \
 bash prototypes/full-mixed/run.sh --tp 2 --spec --budget 288
```

For hw3, set PROBE_RUNTIME, PROBE_MODEL and PROBE_HELPERS. The borrowed CPU helper
set is `probe_host_npus.py`, `supervise.py`, `idle_gate.py`. Exact DSACP/runner/
ACLGraph installed sources there were compared to the pins; no whole-environment
identity is claimed. Dummy model config may be copied without weights.

- `--real` removes model-size and block-count overrides, with85% donor auto KV
  sizing. Set PROBE_MODEL to the real checkpoint.
- `--kv-gib 3` explicitly bounds KV for real-weight shadow snapshots; not a
  suggested production capacity setting.
- `FULL_MIXED_PATCH=0 --mode FULL_DECODE_ONLY` restores the original target
  graph/metadata path, retaining fixture support and the K5/TP8 alignment repair.
- `--mode NONE` is a secondary eager reference, not the best donor baseline.
- `--rounds N` repeats the seven cohorts. Protocol/config/receipts accompany data.
- FULL success requires the large bucket both captured and replayed. Requested
  config alone is not proof: budget must survive joint alignment and max limits.

CPU contracts (runtime Python supplies NumPy/Torch):
```sh
/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env/bin/python \
 prototypes/full-mixed/test_contract.py
```

## Decode and draft

See [DECODE.md](DECODE.md) for the opt-in private-bank DSpark FULL draft,
stream-ordered replay, exact initial-call checks and corrected same-engine timings.
The native proposer remains eager unless the experimental body wrapper is enabled.
