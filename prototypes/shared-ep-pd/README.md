# Shared EP pool with rank-local P/D work

Bounded feasibility probe, not a new connector, KV migration, or isolated PD
serving product. Real `/data/shared_models/Qwen3-30B-A3B`, BF16, DP2/TP1/EP2,
local physical6/7. Both workers report the same EP world size2 and ranks0/1.
No donor source, dispatch, collective, or graph implementation is patched.

## Workload and result

Native synchronous offline engines, maxseq1, KV2GiB/rank, maxlen8192,
FULL buckets1/16/256/1024/4096, no speculation/APC. Rank0 starts a16-token
prompt and generates24 tokens. Rank1 starts a16-token seed producing2 tokens;
at client step4, after rank0 has entered decode, rank1 receives a256/1024/4096
prompt producing2 tokens. Idle ranks participate in native dummy EP work.
A no-injection control and a separate4096 profile window complete too.

All five windows complete with exactly the requested output counts. Rank0's24
greedy output IDs match the no-injection control in every arm (including profile).
This is a bounded ID consistency check, NOT a full quality or raw KV oracle.

| Injected rank1 prefill | Rank0 real decode queries | Both graph buckets | Rank0 forward | Rank1 forward |
|---|---:|---:|---:|---:|
|none|1|1|~12.6ms steady|dummy EP partner|
|256|1|256|42.96ms|43.78ms|
|1024|1|1024|90.68ms|91.34ms|
|4096|1|4096|298.77ms|299.49ms|

All observed mixed waves select FULL on both ranks. Subsequent waves return to
bucket1. Single occurrence per size: feasibility/shape evidence, not stable
throughput comparison. External events surround native `_model_forward`;
no synchronize is inserted inside each forward. Client barriers and synchronous
scheduling deliberately control injection and are NOT production host latency.

Native `_sync_metadata_across_dp` pads ranks to the maximum token count for
FULL, then re-dispatches. Thus a real1-query decode selects4096 on rank0 when
rank1 prefills4096. Profile corroborates actual rank0 QKV `[4096,2048]` and O
projection `[4096,4096]` GEMMs,48 of each, matching rank1's large shapes. Both
ranks execute expert GMM with local weight shape `[64,2048,1536]` / `[64,768,2048]`
and large routed input shape65536 rows. This is allocated/routed shape evidence,
not a claim every padded row is meaningful. EP collective identity markers match
across ranks. Decode does not advance independently while P uses the same pool.

## Fixture correction and observation caveat

Trial1 completed decode-control, then failed at asymmetric injection: default
async scheduling's in-flight batch queue conflicted with the fixture's strict
per-step client barrier. Rank1 hit BrokenBarrierError, its exit closed rank0's
Gloo peer. Trial2 explicitly sets `async_scheduling=False`, keeps synchronous
native DP coordination, and completes. This does not prove async donor cannot
support the workload; don't reintroduce client lockstep around an async queue.

Worker observation stores the last scheduler dictionary; native `_dummy_run`
bypasses `execute_model`, so idle-rank rows can contain stale request IDs and
computed positions. Do NOT count them as live decode. Client completion receipts
identify idle periods; only the rank1 injected wave4 and subsequent wave5 are
live work after seed completion. The mixed-wave comparisons use live rows only.

## Artifacts

- `driver.py`, `worker.py`: runnable fixture/observation. Frozen versions and
  lease/admission launcher in `runs/shared-ep-pd-20260915/v2` and `run-v2.sh`.
- Raw capsule: `runs/shared-ep-pd-20260915/trial2/measurements`.
- `results.json`: control/mixed live-wave receipts and output-ID comparisons.
- `profile-shapes.json`: native large projection shapes.
- `p4096-profile/analysis/qwen-dma-dp2-end-aligned.json.gz`: two-rank TraceLoom
  export (historical helper filename),7,187,669bytes,217,727events. Provider
  rank-device maps0→6 and1→7. Unique collective endpoint fit passes50us gate
  (reported holdout700ns, drift0); display alignment only, not physical clocks.
- Unprofiled times above are separate from profile. Selected cards released;
  preserved other local work. No hw3 used.

Conclusion: rank-local prefill and decode can share a real expert pool and retain
FULL in native donor. It is a synchronized heterogeneous wave, not latency-isolated
PD. Removing decode-side large padding and deciding EP scheduling granularity
are distinct future design questions; no production patch is adopted here.
