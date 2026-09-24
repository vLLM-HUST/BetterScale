# Expert IPC transport diagnostics

Run only under the workspace's admitted selected-device lease and external
supervisor. Two independent processes/devices; no model weights, FFN or host
per-generation RPC. Do not use this echo server in serving.

`transport.cpp` includes the actual packaged client translation unit unchanged.
It adds a resident echo server, one-waiter/split-copy controls, and a DFC-style
ping-pong copy candidate. Echo retains READY/descriptor/DONE/retire generations,
but omits route grouping, metadata consumption, local expert fanout/reduction,
queueing and the real persistent coordinator. Thus its latency is a diagnostic
lower bound, not a measurement of production network-only cost.

`run.py` scans graph-captured calls (64 unrolled calls by default), device-event
intervals, changing BF16-exact payloads and exact final client/server generations.
The first zero-row case measures control-only traffic, not a serving shape.
Collect modes: production16 blocks, production1 block, separate1-block wait plus
16-block copy, and fused16-block pipelined copy. The single-block production
variant changes BOTH waiting and copy parallelism; the split control helps
separate these effects but adds a graph task.

Copy-only diagnostics keep the server idle without advancing READY and compare
local versus imported peer memory using the same kernel, bytes and block count.
The remote output retains the last4096-row echo. No new payload is published
while source staging is repurposed for the local-copy comparison.

Build with packaged `build-vector.sh`, setting SOURCE to this directory's
`transport.cpp`, LAUNCH_SOURCE to packaged `native/launch.cpp`, and OUTPUT_DIR to
an explicit new artifact directory. Run under the qualified torch-npu Python
with this worktree's `src` prepended to sourced CANN PYTHONPATH:

```
python run.py --devices 4,5 --build /absolute/build --output /absolute/result.json
```

The process parent bounds execution to600s, fails if either role dies, and
cleans only its children. The caller still owns hardware admission, foreign
process guard and leases. IPC capability keys stay in private pipes.
All streams drain before mappings are closed, peers acknowledge unmapping before
allocation release, and no inactive graph is replayed after resource teardown.

Artifact root: workspace `runs/expert-transport/20260924/`. Separate from the
concurrently developed `expert-event-client` worktree; do not modify that fork.

## Existing two-slot overlap discriminator

`multisource.py` is an A2E1 real-weight layer0 leaf with a96-row/64-call
decode-shaped graph and a4096-row/16-call prefill-shaped graph. Both source
processes register with the same persistent owner, then execute isolated-source0,
isolated-source1 and simultaneous phases. CPU barriers only bracket complete
graph episodes; no per-request host RPC is added. Five samples after a warmup,
full independent BF16 output and changed-input gates, exact generation drain.
It does not include attention, scheduler priority or a serving workload, and its
per-call average is not a decode-tail-latency estimate.

Use the package-owned MOD branch and exact build for candidate/control, under
three selected-device leases and a continuing ownership guard. First attempt
was interrupted by foreign admission on card4 after our launch; its measurements
are unqualified. The three-card5/6/7 candidate/control/push gates pass; see the scenario note
for their bounded measurements. Do not infer cross-slot efficiency merely from the `pulls_during_cube` counter.

## Real-geometry native reference

native_dfc.py uses the existing native EP2 operator with real Qwen35 target/draft
weights and an independent unfused oracle. Requires a task-owned qualified BF16
A2 provider (DFC_EXTENSION/ASCEND_CUSTOM_OPP_PATH), HCCL_BUFFSIZE>=512, and two
admitted cards. Default rows3/96/512/4096, layers0/40; maximum capacity admits all
routes on one owner. Initial and changed input FULL-graph checks pass in
native-dfc2. See the scenario for provider identity and topology limitations;
do not compare its two-compute-rank global workload as if it were an A1E1 call.

## Remaining admission scan

build_accept.py emits two coordinator-observed variants from the source-owned
MOD: engine3/kind1 records successful admission; engine3/kind2 records Group.
Use the same observer in both arms. --skip-planned-ids retains every original
range check but skips the scalar-GM ID cache for native-planned frames, since
their Group consumes counts/maps instead. Legacy small frames still write the
cache. test_accept.py executes the actual emitted loops on CPU, including
invalid IDs at first/middle/last positions and both threshold sides. This is an
unqualified diagnostic until its own real-weight/lifecycle gate completes.

The scalar skip-cache diagnostic passed real target/draft output/lifecycle gates
but is **not promoted**: small-call regressions and hot8 variability remain.
Its4096-broad admission envelope falls699->409us (same observer); this identifies
remaining scalar validation as useful work rather than proving overall speedup.

`route_validate.hpp` replaces only the planned-frame range scan with existing
AscendC Cast/ReduceMin/ReduceMax. A2 reduction supports FP32; conversion preserves
membership in[0,256) for every int32 value. Scratch fits the existing196608-byte
UB, disjoint from32768 incoming IDs. `validate_kernel.cpp`/`validate_probe.py`
exercise eager and captured changing inputs, invalid first/middle/last IDs,
INT_MIN/INT_MAX and both boundaries across1..4096 rows. Artifact `validate2`
passed108 cases on local6;4096-row standalone graph call median7.70us. This is
not a whole-admission/FFN measurement. `build_accept.py --vector-validate`
keeps the original small-frame loop outside the fast branch and copies this
header into emitted source. Real FFN integration remains its separate gate.

The vector admission prototype subsequently passed `ffn-accept-vector1`, with
4096-broad Accept15.32us versus699/719us bracketing controls and wholecall4.07ms
versus4.82–4.88ms. Tiny control calls drift194->244us across the same bracket;
do not attribute every small-call difference to codegen. The optional noinline
variant builds but **fails persistent_vector binary loading**, before readiness
(`ffn-accept-vector2`); it has no performance result and is not promoted.
The accepted inline helper is now MODf8d8847; `mod-vector-build1` passes28 expanded
real FFN cases/4520 generations. Historical build_accept/test_accept helpers target
the pre-vector MOD6a3b810; do not patch the already-migrated admission twice.

`build_fused_pack.py` is the next narrow discriminator atop MODf8d8847, cap1 and
layer-only. It freezes the single admission, builds native maps before FETCH,
then pulls each8-token chunk once into UB and fans out locally. It does not
multiply peer payload by top8, enable PackGate, start GEMM early, or publish
partial results. It requires ordinary full-command movers (no urgent/quantum/
fine-pack). This first step measures the transport layout before adding tile
readiness; its device gate is separate from vector-admission qualification.

### Input-ready prototypes (not MOD defaults)

build_input_overlap.py uses a private MOD runtime capsule, row64B generation
flags after completed MTE3, interleaved8-token peer chunks and M64 tile-local
PackGate instead of whole-expert readiness. ffn-input-overlap01 passes16 real target/draft changed-input cases but is slower:
4096 broad4.89ms versus direct-fanout3.91ms. FETCH/UP command envelopes overlap;
that alone is not a profiler proof of the amount of actual MMAD hidden. UP's
3.01ms envelope includes readiness waiting and flag-poll cost, not pure GEMM.

build_compact_ready.py replaces per-route publication with per-mover chunk
progress plus immutable per-tile source-token dependencies. Coordinator Max
reduces16 FP32-exact dependency catalogs; a conservative source-prefix watermark
releases a tile only when all its rows are visible. Reuses existing readiness
storage; a DMA ID cache supports dependency construction without restoring the
old scalar-GM loop. test_compact_ready.py checks ownership/prefix safety including
non128-aligned tails; actual device graph/oracle remains a separate gate.
ffn-compact-ready01 passes16 cases,4096 broad4.32ms: better than per-route flags,
still worse than the3.91ms no-overlap fused input. Do not promote on attractive
command overlap alone. A same-code serial-input control isolates gating cost.

ffn.py now optionally accepts --patterns random to exercise nonuniform expert
counts and tile boundaries with unique random top8 IDs. Defaults remain hot8/
broad, preserving existing controls. Neither prototype changes weighted combine,
DONE/retire, output ordering or source lifetime. Only layer/cap1 is admitted.

### Matching the native reference's useful global work

The original EP2 reference has two full input ranks; its timing is not a direct
A1E1 ratio. native_dfc.py --one-source attempted N+0 inputs with all active
experts128..255 on rank1. The retained native-dfc-one-gate failed at zero-row rank1
with507015/VEC illegal configuration before graph timing; cleanup/release passed.
Do not repeat that unsupported provider shape as a throughput point.

--sentinel-source uses N-1 inputs at rank0 and one zero-valued final token at
rank1, all routed to128..255. It preserves exactly N global tokens and expert
counts without a padded extra row (which could cross a GEMM tile boundary).
ffn.py --patterns remote128-sentinel uses the same seeded global BF16 fixture,
int32 IDs, probabilities and original target/draft weights, but sends all N rows
from the one attention source. The native baseline therefore saves one token's
remote traffic; disclose this difference, especially for tiny N. The only active
expert compute owner is rank1 in both cases. Native still has synchronous EP2
collective semantics and its compatible provider/runtime; this is a stronger
matched-work reference, not full-serving or bitwise-math parity. A3-token gate
passed; larger/changed-input measurements have their own artifacts.
