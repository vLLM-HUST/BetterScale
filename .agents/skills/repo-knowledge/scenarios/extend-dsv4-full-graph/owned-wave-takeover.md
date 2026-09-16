# Model donation with actual LiveInference execution (2026-09-16)

Use this when extending runner-independent execution, not the earlier
fixed-snapshot execute/sample wrapper. Implementation and rerun recipe:
`prototypes/owned-wave/README.md`. Capsules freeze source; do not rerun merely
to recover these observations.

For the subsequent FULL-prefill/N+2 implementation, jump to
[the continuation gate](#full-prefill-and-n2-continuation-gate). The first sections
below describe the earlier, eager-prefill takeover evidence.

## Qualified fresh evidence

- BetterScale base b0e6ad9, native vLLM pin
  752a3a504485790a2e8491cacbb35c137339ad34 and Ascend pin
  9bf964cb4b87c8cd0d6852c41a55b3c29711fa95.
- Actual, unmodified LiveInference Python runtime at
  05ac15419c0e73650e687ceb9daffeb7874865f0 is imported from capsule source.
  Installed donor runtime stays unchanged; key native files compare equal.
- `runs/owned-wave/takeover1`: preliminary TP2/EP2, two dummy layers, PASS.
- `runs/owned-wave/takeover2`: lifecycle gate, two dummy layers,
  TP2 × DP2 / EP4, all four ranks PASS.
- `runs/owned-wave/takeover-real1`: real Qwen3-30B-A3B, all 48 layers BF16,
  TP2 × DP2 / EP4, ALLGATHER MoE, all four ranks PASS. Two different 32-token
  prompts, greedy, ignore_eos, seven output tokens. Both client exits0 and
  supervisor exit0; release receipt exists.
- Each rank: two startup graphs; four decode-entry Python executions
  (warmup/capture only), 18 metadata shadow/construction actions; no fenced
  runner calls. Owned eager prefill and each of six decode waves exactly match
  independent native tokens and ALL deduplicated KV storage bytes.
- Second request generation: at most two outstanding invocations; all six
  tokens and final KV exact. Admission while invocations are live is refused;
  two terminal drains yield count0/token-1 without cursor/anchor/KV mutation.
  Activation restores donated KV after warmup/capture.
- Analyze with `prototypes/owned-wave/analyze.py CAPSULE`. The latest analyzer
  requires lifecycle fields absent from preliminary takeover1.

These are correctness/ownership observations, not a throughput measurement or
general serving compatibility. DP0 reference itself repeats tokens3270/11;
the takeover matches it, not an independently asserted language-quality result.
DP1 autonomous IDs are [220,16,11,220,17,11].

## Architecture facts and boundaries

OwnedRoot has no runner reference. Direct native model + compute_logits +
sampler calls are inside the decode graph with device cursor, token feedback,
remaining budget and receipts. Runner execute/sample/metadata/capture methods
are forbidden throughout candidate execution. Native engine startup and RPC
still host the harness, and native reference is collected BEFORE transfer.

LiveModule owns graph activation, State, MetaTensor construction, shadow replay
and invocation retirement. AdoptedStateBackend lends existing raw allocations
into declared State; WaveSchema declares capture_state_blocks so activation
really snapshots/restores them. Do not treat State registration without that
capture declaration as equivalent preservation. Original native typed KV views
still alias the donated allocation; exclusive ownership is a harness contract.

**Shadow is metadata/ingress projection**, not numerical oracle execution.
The independent native oracle is a separate earlier episode. Every submit uses
invoke → shadow_replay → native PA task update → replay → event → retire.
Never replay the PA graph without publishing its ExternalEvents.

PA still needs CPU context lengths and host task updates. PALength is a real
MetaTensor, but it deliberately constructs the carrier the installed operator
accepts. Device cursor does not by itself change this ABI. Private banked PA
registries are swapped under serialized host execution; no thread-safe API claim.
Known balanced non-speculative geometry supplies host lengths; sampled IDs
never travel through host to seed the next wave.

This is not yet a replacement general resource allocator/scheduler. It adopts
the original allocated block table and a finite grant, uses serial admissions,
balanced one-request DP, eager prefill and fixed greedy policy. EOS, arbitrary
DP imbalance, speculative acceptance, dynamic batching/preemption and robust
serving failure recovery are not implemented. Failures abort the bounded process.

## Avoid repeating the wrong attention experiment

`runs/owned-wave/attention{1,2,3}` retain bounded rejected stock SFA leaf probes.
The installed signature differs from the stale docstring (keyword-only
sparse_block_size); missing rope fails; a 64+64 Q/K split does not establish
ordinary GQA support. LiveInference native SFA tiling requires attentionMode2,
KVheads1, QK/V512 and rope64. Read its
`native/ascend/csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp`
before trying to transplant that DSV4 operator into Qwen.
Do not pad/transform Qwen semantics just to force an unrelated kernel to pass;
these failures do not prove device-length Qwen attention is fundamentally impossible.


## FULL prefill and N+2 continuation gate

The current `prototypes/owned-wave/` supersedes the earlier eager-prefill and
same-compute-stream-copy harness; old capsules above retain their old scope.

Fresh September16 capsules, unchanged donor/native pins and actual LiveInference
runtime commit above:

- `full-n2-dummy1`: two layers TP2/DP2/EP4. All four numerical receipts and
  both client result files passed. Foreign occupancy appeared on cards0–3 during
  teardown; the supervisor aborted before complete.json. Keep this INTERRUPTED,
  not an end-to-end PASS. No foreign process was killed.
- `full-n2-real1`: real 48-layer BF16 Qwen, TP2/DP2/EP4 on freshly admitted
  cards4–7. Four-rank PASS, both client exits0, supervisor exit0 and release
  receipt. Each request emits7 tokens; continuous trace has16 waves:
  terminal6 -> old drain7 -> new prefill8.
- `full-n2-real2`: same real envelope, six-output continuous requests:
  terminal5 -> old drain6 -> new prefill7,14 waves. This deliberately exercises
  BOTH prefill banks on replay, rather than only capturing bank1. Four-rank PASS,
  both client exits0, supervisor exit0, release receipt.
- Each run separately checks FULL prefill and six decode steps against the
  original native episode's tokens and all deduplicated KV bytes, followed by
  two no-write drains. Its continuous two-generation episode compares every
  emitted token and final whole KV. No KV restoration occurs between those
  continuous requests; restoration happens only before the episode.
- Four startup graphs: prefill/decode ×2 banks. Eight numerical Python entry
  executions during warmup/capture, ZERO afterward. Shadow action counts33
  (real1) and31 (real2). Runner execution calls remain empty.
- Seven CPU fence/scheduler tests cover complete quorum, stale sequence or
  generation, failed rank, rejected asymmetric termination, insufficient grant,
  terminal-prefill drain and odd-parity turnover. Analyzer validates every rank,
  ordered trace and TP token agreement; it also supports the older capsules.

### Protocol now owned

`schedule.py` is a small balanced resident ledger, NOT the entire DSV4 policy.
It requires a complete EP CPU-communicator quorum for N before issuing N+2.
N+1 remains outstanding. The device authors cursor, anchor, remaining, DONE and
generation. Decode is submitted before the first prefill receipt is retired;
host never installs its sampled token.

`executor.py` adapts the actual LiveInference AscendDSV4WaveExecutor reactor:
banked pinned authorization/prompt ingress, graph-reader fence before input
overwrite, separate ordered compute, output-copy fence before bank reuse, and
independent egress with invocation-private pinned destinations. Metadata carriers,
inputs and LiveInvocation remain live until copy completion/retirement.
Native attention additionally has its task-update stream; three-stream reactor
does not mean three total backend streams.

A terminal quorum authorizes the next generation's captured prefill while an
old-generation drain is outstanding. New generation initialization occurs IN
the prefill graph, behind the old drain on compute—not via an early host State
write or an all-invocations-empty admission requirement.

### Deliberate limits

FULL prefill is qualified at exact32-token width, all48 layers, including the
native sampler and device continuation initialization. This does not establish
arbitrary prompt lengths, mixed/partial/chunked batches or long-context behavior.
Each DP owner repeats its own same prompt across the two generations; prompts
differ across DP. This proves generation/bank turnover, not an arbitrary
changed-payload request workload.

The PA/FIA host task-update seam remains; no claim of device-only attention
length ABI. Balanced greedy/length-only requests, one resident per DP owner,
fixed borrowed physical grant, no EOS/speculation/preemption/general serving
allocator. Asymmetric termination is rejected, not silently coerced.

The host trace demonstrates N+2 submission, complete-rank authority and lifetime
order. It does NOT measure physical compute/DMA overlap or throughput improvement.
Do not call already-issued N+1 necessarily still executing merely because its
receipt remains unretired.

## Concurrent original SWE trace with APC (2026-09-16)

For variable-length multi-session execution, enter
`prototypes/owned-wave/serving/README.md` and its frozen run capsules. This is a
new TP2/EP2, DP1 envelope, not a generalization of the earlier balanced DP2 gate.
Native KVCacheManager owns hashes/refcounts/eviction; the owned scheduler reserves
finite horizons, publishes only receipt-confirmed blocks and defers terminal
leases until every in-flight generation reference retires. Exact-power-of-two
prefill and four-row decode graphs share the actual LiveModule State arena.
FULL decode uses FIA and needs a causal mask: serving-dummy1 rejected its absence;
serving-dummy2 passes both rounds and TP peers, including all8 synthetic outputs.

The pinned NVIDIA Open-SWE-Traces fixture swe-trace-v2 has four whole sessions,
44calls and12,111 generated-token budget, original history/tool definitions and
no truncation. It is zero-tool-delay closed-loop input replay, NOT SWE task solving
or a recorded arrival log. Native timings end at client-engine output; owned
at worker quorum retirement, so do not advertise identical-frontend HTTP latency.

swe-real1-hw2 completed both arms/two rounds with owned lifecycle and TP agreement,
but only2/44 whole outputs matched across arms; native cold/retained matched10/44.
Cause is unresolved. Fletcher explicitly removed cross-arm token identity as a
performance gate; retain differences as observations, do not imply equivalence.
That intake is timing-excluded: local and ssh hw2 are the SAME kernel/host, and
a briefly overlapping profile startup was cancelled. Never use these aliases as
two independent machines. Collect timing and profile serially; parse native CANN
profiles offline, outside daemon workers, with the serving parser.

Fresh unprofiled native-first2/owned-first1 swe-perf capsules on cards4,5 both
complete: cold mean94.413s native/101.073s owned, retained87.561s/89.809s.
No speedup: owned is6.6–7.5% slower cold,2.2–3.0% slower retained. Reuse
`serving/PERFORMANCE.zh-CN.md` for the full measurement boundary and source anchors.
Separate swe-profile1 exports four validated native profiles: ArgMax is outside
native graphs but inside owned graphs; both retain48 FIA host task updates/wave.
The two64-wave windows have unequal neural work; do not attribute kernel-speed
changes from their totals. Same-process reverse-order memory peaks inherit owned
resources, so they are not independent-deployment capacity measurements.

For removing per-layer attention task updates, enter
`prototypes/owned-wave/serving/DEVICE-ATTENTION.md`. The installed TorchAir device-
length FIA passes three changing paged-decode cases exactly, but GE model execution
cannot nest in ACL capture (same-stream ge3 runtime rejection). That GE failure does NOT require replacing attention or migrating to GE. The
subsequent native launch adapter in `prototypes/owned-wave/fia-plan/README.md`
retains the original kernel/opaque plan and replaces only two length relocations
with persistent GM tensors. Two-bank decode through8192 and128/256-token prefill
with changing pages/prefixes pass max-error0 against native FIA. Do not repeat
GE nesting or assume fixed3008-byte launch blobs (prefill row counts change size).
Whole-model/performance receipts remain distinct from these leaf gates.

The native-wrapper whole-model result is in `fia-plan/PERFORMANCE.zh-CN.md`:
real Qwen3-0.6B TP1 (not30B/distributed), two-order retained means2.9772s native,
1.7593s static owned; legacy owned2.6789s. Separate64-wave TraceLoom audit
confirms1792 FIA graph tasks and ZERO host FIA/task-update calls. Reuse those
capsules; no need to repeat GE probes. Keep frontend boundaries, cold APC hit
mismatch, full-model token divergence and startup/capture exclusion explicit.


### Integrated30B candidate: static-plan regression, not a win

Enter `prototypes/owned-wave/fia-plan/REGRESSION-30B.zh-CN.md` before extending
this wrapper. Candidate-only full SWE retained95.827s vs historical native87.561s,
old-owned89.809s. The matched four-step TraceLoom window isolates slower device
attention (~20.5%), with host updates still0; dense matmul/communication do not
explain that window's regression. A bounded installed native-launch diagnostic
chooses FD5100000000010200203/23blocks at~4.5K for Q16/KV2/batch4, whereas startup
non-FD500… has only8 useful tasks. The wrapper deliberately admits only non-FD.
Do not infer lack of adaptive dispatch from graph capture labels; preserve FD's
split/core/reduction metadata at wave granularity before claiming30B optimization.
Baseline was reused, not rerun; no FD repair has yet been qualified. The1K short
profile is insufficient for attributing the full-trace regression. Reuse the
matched four-step capsules and dispatch snapshots instead of profiling full runs.
