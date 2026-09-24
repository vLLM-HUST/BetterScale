# Expert transport campaign result

## Final transport campaign acceptance (2026-09-24 23:40 UTC)

Qualified MOD source **397efcf**, exact `mod-fused-build2`, keeps threshold0/pull
as defaults. Explicit native1024/push is qualified only for whole-layer/cap1.
After independent vector ID validation, Group freezes the native route map before
FETCH. Movers read each eight-token peer chunk once into ping-pong UB and fan out
to expert destinations, eliminating owner-local token staging plus REPACK. FETCH
still completes before READY_UP; this is **not intra-frame GEMM/communication
overlap**. Small/default/quantum paths and final DONE/drain ownership stay intact.

Evidence root: workspace `runs/expert-transport/20260924/`.
`ffn-mod-fused67` passed28 real target0/draft40 cases, rows3/96/1023/1024/4096/
4097/8193, hot8/broad and changed inputs,4520 exact generations, max L2 .004907.
Prototype `ffn-fused-random67` additionally passed8 odd-size/random routing cases.
All144 CPU contracts pass. Same local0/1 sequential vector control versus fanout:
layer40/4096 hot3122→2942us, broad4127→3912us. Do not use cross-card tiny-call
timing drift as a causal speedup estimate.

`serving-fused/analyze.py` checks exact leaf build identity, actual mode, all41
native target/draft shadows per attention rank, generation drain and clean release.
Both hw3 points passed with zero request errors, all8 server exits0. Source/build
objects agree with the leaf gate; build2 only adds comments versus build1.
Same prepared SWE protocol744670516b02796518372dd10826e4f278ab5854, seed20260924,
C64,900s measurement,8chips,query4096,maxseq32,KV32GiB/chip,262144 context,
real MTP2/native FULL decode/no host KV offload as earlier four points.
Canonical result: `serving-fused/swe-comparison.json`.

| Arm | Original0/pull tok/s/chip | First native plan+push | Final vector+fanout | vs original | TTFTp95 original→final |
|---|---:|---:|---:|---:|---:|
|A4E4|142.369|160.759|163.913|+15.1%|4.444→3.633s|
|A6E2|133.905|167.219|173.576|+29.6%|2.567→1.996s|

Final incremental gains over first plan+push are2.0%/3.8%; single sequential
closed-loop samples do not establish repeatability or attribute these small
increments independently. Faster arms reach different context/turn mixes.
Lifecycle MTP acceptance lengths2.899/2.873 include startup/drain. These are not
semantic-quality or tuned-peak claims. Both jobs/controllers finished and released
hw3; no public/frontier update or superproject pin change.

### Readiness experiments: correct, not promoted

Research branch7135935 retains builders and private runtime capsules; do not
blindly reapply patches expecting pre-fanout MODf8d8847 to current MOD.
`ffn-input-overlap01` uses M64/per-route64B generation flags:16 cases pass, but
4096 broad4891us. `ffn-compact-ready01` replaces flags with immutable tile input
dependencies plus conservative source-prefix readiness:16 cases pass, hot3165us,
broad4323us. Same-code no-early-admission `ffn-compact-serial01` passes16 cases:
hot3970us,broad4021us. Both readiness variants lose to plain fanout2942/3912us.
CPU prefix tests cover non128-aligned tails; these device gates do not establish
odd-tail/random/EP/full-serving acceptance. FETCH/UP command envelopes overlap,
but UP includes waits: this is not msprof proof of physical MMAD overlap.

Inference from command ownership, not a measured cause: early UP may occupy the
single whole-Cube command slot while waiting for input, blocking another ready
frame. A future overlap design must test that head-of-line cost and readiness
traffic, not just add event flags or copy dense MatmulAllReduce tiling. Existing
cross-frame overlap and reduced-output DMA pipelining are distinct mechanisms.
Do not enable old PackGate/early-return paths based on these observations.

### Matched native semantic-work reference, not universal superiority

Research `native_dfc.py --sentinel-source` assigns N-1 inputs to rank0 and one
zero-valued final input to rank1. IDs128..255 place all experts on rank1; totalN
and route counts match `ffn.py --patterns remote128-sentinel` (allN from client0).
Both use real BF16 target0/draft40,H2048/M512,E256,K8, FULL16-call trials×5,
local6/7. Native has one fewer remote token, material for tinyN. Seeded payloads
were not byte-compared across runtime versions. This matches semantic work and
one compute owner, not transport implementation or raw network bandwidth.
Frozen compatible provider provenance is in `dfc-provider/provenance.json`; it
is not proof binary identity with inspected donor9bf964cb.

`native-dfc-sentinel-full`20 rank cases and `ffn-remote-reference67`10 cases pass
independent changed-input oracles (max L2 .004154/.004595), exact drain/release.
`analyze-matched-native.py` / `matched-native-summary.json` reduce each native
trial by slower rank, then median. Layer40 complete FFN medians native→ours(us):
3:470.5→199.0;96:652.4→804.0;512:1878.9→1205.0;
1024:3511.6→1358.4;4096:13396.3→3330.4.
**96 rows remains23.2% slower**, despite4096 being75.1% faster. Do not claim all
shapes graduated, a bandwidth ceiling, or serving parity from this fixture.
Earlier zero-input second-rank attempt `native-dfc-one-gate` failed507015/VEC
illegal configurations before graph timing; preserve failure, do not retry an
unchanged hypothesis or cite it as a timing/upstream-general defect.
