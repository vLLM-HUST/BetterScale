# Native client and persistent-server gates

Historical bounded evidence; read the relevant section, not as current defaults.
Paths beginning `prototypes/` are repository-relative.

## Split Qwen attention around an asynchronous expert boundary

Enter `prototypes/attention-client/README.md` before building per-lane attention
replays. The reduced native Qwen oracle retains original output/KV checks, not
just a normalized-path reference. Native2 qualifies the layer cut; graph3 qualifies
fixed-context FULL attention segments; metadata7 qualifies native attention-task
updates across later same-shape invocations with changed positions/lengths.
These are TP1 dummy small-model gates, not independent multi-client serving.

Three hidden singleton contracts matter: ForwardContext.moe_layer_index selects
native MLPs, acl_graph._graph_params owns task handles/events/workspaces, and
attention_v1._ATTN_KEYS_BUFFER caches metadata layer order. Separate layer/bank
registries must scope ALL three (and restore them), or a later layer updates the
first layer's key. Current scoped registry swaps require serialized host access.
Native eager startup does not create runner.update_stream; the bank owns one.
PrefillNoCache provides None block_table, rejected by native FULL weak references.
The prototype uses a scoped paged ChunkedPrefill metadata view and checks against
ORIGINAL eager arithmetic; never globally rewrite scheduler metadata to hide this.
Bank IO and metadata tensors are private; graph capture-on-miss in these oracles
is not an acceptable published online warmup policy.

lanes12 additionally qualifies two prebuilt attention graph lanes with disjoint
ordinary Qwen K/V, metadata and IO. Suppressing prefill's first expert completion
does not prevent decode finishing both layers. Outputs and private KV match native
snapshots exactly; native MLP is still a local reference on a separate stream.
Raw-byte-backed typed K/V views cannot be Python-deepcopied on this runtime: clone
each tensor while retaining the tuple shape, and scope the implementation's cached
K/V references as well as attn.kv_cache at capture. This is not a general strategy
for compressed pools with semantic aliases. For the delivered external server,
use attention-client/server_contract.py and its explicit INT32-only boundary.
`attention-client/ipc/` qualifies an external packet producer against the unchanged
3532418 server binary on two cards (ipc3). Both captured graphs replay changed
INT32 plans, exact top-k CPU retirement; this is separate from neural inference.
ACL binary loading requires the named AIV `.ascend.meta` section, not only a
successfully linked ELF. Real BF16 route production/reduction and expert GEMM
remain the joint integration boundary; do not overstate these two separate gates.

For the subsequent real BF16 **Attention2 + Expert2** reference, enter
`attention-client/joint/README.md`. Full Qwen3-30B-A3B layer dimensions/two dummy
layers pass exact native output/KV checks on four cards. This consumer is
host-driven (IPC data, CPU descriptors/completions), not an extension of the
persistent INT32 server. It qualifies true remote GEMM and two-owner retirement,
not cross-source neural batching or production performance. Its KV shadow restores
the pre-forward state before candidate execution; reference writes left in place
can hide a missing KV update. Import the worker via native initialization, not
attention_v1 before donor device initialization, to avoid its circular import.
Align only episode startup after both clients prewarm: otherwise one client may
finish the bounded fixture before the second finishes loading, yielding no useful
concurrent-ownership coverage despite a four-card run.

For the next **device-driven BF16** gate, enter
`attention-client/device-service/README.md`. Device-authored int64 GMM group counts
and `(layer, local expert)` groups allow unchanged native grouped GEMM/SwiGLU/GMM
inside a finite, pre-unrolled service graph. Queue/pack and scatter/DONE are AIV
kernels; one server replay consumes a bounded episode without host route extraction
or per-batch decisions. The three-card primitive passes64 exact outputs; the
four-card two-layer Qwen dummy oracle passes24 exact forwards/KV across two clients.
This is NOT an infinite persistent neural kernel or deviceized whole scheduler:
host attention continuation, fixed padding compute and scalar packing remain.
Both expert DONE generations must be consumed before reusing a source frame;
publish all payloads before DONE (the publication helper reuses its UB). Keep
native Worker imports out of the expert-only process to avoid the donor's platform
initialization cycle. New neural policy does not inherit the integer server's
priority/cancellation qualification merely by using its publication pattern.


The device-service continuation in `DFC-ADAPTER.md` replaces serial payload movers
with16 AIV blocks and reuses native token unpermute. Only the device descriptor
selector remains one-core; count/prefix/assign avoids O(experts*routes) scans.
Never publish DONE before all mover blocks finish: graph-node boundaries supply
those joins. Owner-directed reads allow unowned return slots to remain poisoned,
not zero-filled. Full dummy output/KV remain exact; final32-row client cost is
about293us versus the initial1090us, still slower than local FULL MLP156us.
Those are stage diagnostics, not a serving throughput claim. Preserve mode and
card-set provenance in the timing receipts.
Fletcher's2026-09-16 override permits idle subsets without the global tp8 lease;
this prototype's per-device admission still rejects foreign occupancy. Independent
TP1 engines need separated rendezvous port ranges, not adjacent base ports.

EP2 fused comparison is in `attention-client/device-service/DFC-COMPARISON.md`.
Pinned donor A2 OPP lacks BF16 DFC; stateharbor's lab A2 build supplies it but adds
xActiveMask to its ACLNN ABI. Never load that OPP with the older donor Torch
binding (observed host SIGSEGV); load its matching extension in an isolated probe.
The qualified synthetic controls exclude gate/top-k and use identical routes:
hot8 favors remote latency, broad128-expert cases16/32 still trail DFC46–52%.
Different2-card versus4-card topologies and synchronous versus independent batches
make this a stage-cost diagnostic, not a serving throughput win.

For broad-expert regression attribution, enter device-service/ROUTE-SPREAD.md.
Join server cycles through recorded source generations, not timestamp proximity.
The32-row profile puts roughly400us of the spread penalty in the two native GMMs;
pack/scatter barely change. All warm sampled cycles were single-source, so do not
claim cross-source batching efficiency from queue capability alone. Opt-in NZ
passes48 synthetic outputs and cuts broad client latency9–11%, not DFC parity.
A separate server_ready after conversion/capture is necessary: weights_loaded
only protects bootstrap ownership, and bounded device polls must not wait for
host graph preparation. The local GMM-chain NZ control is slower; keep its scope
separate rather than claiming NZ universally wins or extrapolating the first GEMM.

For the same-device paired-source control and group/padding factor sweep, enter
`prototypes/attention-client/device-service/BATCH-ORGANIZATION.md`. Paired broad32
keeps GMM near400us while doubling live rows; active pack-to-DONE work is497us
for two sources versus470–476us for one. This is not an online-throughput or
latency guarantee. The native128-group/padded chain also carries measured costs
relative to64 live groups. Capacity-only group sums passed this binary but violate
the installed GMM group-sum == input-M contract; never silently enable them.
The paired selector is an explicit experimental gate, not a deployed scheduler.


### Independent-source actual-count expert service (2026-09-16)

See `prototypes/attention-client/device-service/ACTUAL-COUNTS.md` for bounded
coalescing, live-row CATLASS/DFC adapters, common-burst and late-source gates.
Native GMM final group end must equal input M; do not use short group ends as
a production capacity trick. Explicit custom capacity/live semantics passed
leaf canaries and joint dummy checks, but broad tiny-expert GEMM remains slower
than native NZ. Common startup produced24/24 paired cycles; unaligned source
arrivals produced none. Do not infer online batching or DFC parity from support.

For the work-conserving successor, enter device-service/PERSISTENT.md.
It runs one persistent AIV team and one persistent AIC team with generation-tagged
commands, per-core joins and two staging slots; it does not wait to grow batches.
Pull completion is a mailbox observation boundary before expert grouping.
Real BF16 four-card and per-core-timestamp gates pass, but equal-work burst
latency improvement varies2.3–23.6% with natural arrival phase; broad expert
latency still trails DFC. Do not confuse persistent residency with guaranteed
batching, overlap, DFC parity or unlimited service. Capture raw ACL launches on
torch.npu.current_stream() INSIDE the graph context: an enclosing stream can
differ from the graph's internal stream and silently produce an empty graph.


### Small-row expert GEMM cache and phase scheduling (2026-09-16)

For persistent expert service scheduling, read
`prototypes/attention-client/device-service/GMM-SCHEDULING.md` and its runnable
`gmm_schedule_probe.py`. Same-card broad64-expert down changes from61us repeated
alone to149us with alternating weight catalogs; whole math is455–462us for2–8
rows/expert. Do not compare isolated warm down to full service or call this an
HBM roofline. Existing CATLASS already stripes tiles and preloads/double-buffers;
DFC additionally segments up/SwiGLU readiness. Two request slots alone do not
provide that intra-wave pipeline. Under interleaved slots, join command generation
to stage events; odd/even command IDs are not a valid up/down classifier.


For intra-wave expert segmentation, enter device-service/SEGMENTED.md. Two frozen
relative count catalogs and disjoint row slices pass leaf and four-card changing
route/generation gates, with real same-wave AIV activation/AIC up overlap. Yet
matched24-paired-wave controls regress: half-cut pack-to-return medians603/547us
become610/566us; tail-two625/591us. Keep opt-in, not the default. Splitting complete
CATLASS invocations is not DFC's continuous internal tile pipeline. Never attribute
the independent-arrival25.1->18.7ms episode to segmentation: pairing changed0->24.
Use the causal segment audit and same-wave timing in persistent_analyze.py rather
than requiring serial completion order or mistaking cross-slot overlap for this gain.


The subsequent device-service/INTERNAL-PIPELINE.md keeps one up tile object alive
and publishes a prefix from inside group traversal (mode2), retaining whole down.
Leaf and four-card paired/heterogeneous gates pass. Matched half-cut pack-to-return
617/574 ->593/542us is a bounded positive result, not DFC parity or stable serving
throughput. Tail-two still regresses: producer join leaves only3.6–4.1us up work,
while first AIV consumption follows14–16us later. Reuse the prefix timestamp audit
in persistent_analyze.py to separate lost notification windows from GEMM cost.
The underlying external CATLASS tile type is shared; no matrix kernel is rewritten.


For AIV pull/pack head-of-line experiments, read device-service/YIELDING-MOVES.md.
Opt-in quantum128/256 yields at DMA chunk boundaries while preserving slot/source
ownership; leaf and48-output four-card cases pass. Quantum128 cuts measured
pull/pack overlap with final-activation waiting126/129 ->24/25us per episode,
but total episode is slower. Complete-command handoffs and repeated map parsing
are not free internal queues. Keep disabled; full pack readiness and whole down
remain batch barriers. Use the analyzer's actual core-overlap metric rather than
attributing every up/down hole to DMA or comparing unlike naturally paired waves.


The resident continuation is in device-service/RESIDENT-MOVES.md. A separate urgent
mailbox allows AIV activation inside one ongoing FETCH/REPACK command, retaining
maps/cursors. Actual interruptions and four-card numerical/causal gates pass.
Two A/B orders reduce measured mover blocking, but net episode gain is not stable;
keep opt-in. Timing engine2 is urgent work on the SAME AIV cores, not a new team:
subtract its intervals from suspended move envelopes and assert physical-lane
exclusivity (implemented in persistent_analyze.py). Source retirement and producer
joins remain mandatory; this is not arbitrary DMA preemption or full DFC fusion.

### Native DFC timeline boundary (2026-09-16)

For broad-hit expert scheduling comparisons, read
`prototypes/attention-client/device-service/DFC-TIMELINE.md` before interpreting
a fused DFC envelope as Cube utilization. The two-card capture/export entry is
recorded there; internal MC2 timing requires separate instrumentation.

### Fine-grained expert input readiness (2026-09-16)

For the latest persistent expert dependency cuts, enter
`prototypes/attention-client/device-service/FINE-PACK.md` directly.
It qualifies row-generation publication before up tile issue, composing with
EARLY-DOWN.md; both remain opt-in. Use expert_ready_audit.py for fine-pack
receipts, not cube_supply_audit.py's historical whole-command readiness model.
The per-expert causal trace is diagnostic, not an MMAD-utilization measurement.

For a fairer persistent-service versus DFC comparison, enter device-service/FAIR-DFC.md.
It records same-host one/two physical weight-catalog controls, fetch-inclusive
intervals and client/retirement scope. Address reuse affects DFC materially but
does not establish remote parity; the remaining input-preparation seam has
descriptor ownership and late-source-coalescing constraints.

For server-published contributions and client-owned token reduction, enter
device-service/ROUTE-PULL.md. It records the generation/retirement contract,
mode2 catalog cleanup, same-work controls, paced polling and zero-owner/weighted
gates. Early token reduction is not permission to reuse exports or advance
attention before full-wave drain; independent client clocks are not aligned traces.

For native model generation with the persistent expert backend, enter
`prototypes/attention-client/device-service/SERVING.md` and use serving_receipt.py,
not the older unrolled-service summarizer. Four-card two-layer dummy shadow gates
match native outputs/KV; no-shadow generation uses zero reference calls and a
raising local-expert guard. A matched32-job shadow run reproduces generated tokens.
The explicit1–32-job unsegmented budget is still bounded/exact, not online EOF or
unlimited service. Full48-layer BF16 needs role-specific weight loading: current
client staging alone would be54GiB plus54GiB local experts. Do not scale the dummy
bootstrap and blame its OOM on the expert transport. This is separate from the
operator worker's evolving GEMM implementation and changes no released defaults.


For independent expert roles and native forward without shadow, enter
`prototypes/attention-client/roles/README.md`. Four-card two-layer dummy sources
retire unequal 44/66 jobs using device EOF, not a predeclared job budget. Clients
allocate no routed weights; servers independently own shards. Initialize remote
Session at load_model completion: native memory profiling precedes warmup. Share
MoE graph scratch pools, but keep returned outputs outside that pool for native
residual lifetimes. Otherwise many tiny private pools can exhaust virtual address
reservations, not physical expert storage. This gate uses eager native attention
and captured MoE, not full-model graph composition or real-weight qualification.
Host teardown remains session-wide, with failure recovery unsupported. Keep
persistent binaries and the extended configuration ABI from the same source.

For scaling independent roles to DSV4, read roles/DSV4-PLAN.md and use
roles/weight_census.py before choosing ranks. Local W8A8 target routed payload
alone is258.67GiB: E4 on64GiB cannot fit. The0731 checkpoint adds THREE DSpark
expert layers; older W8A8 adds one. Config expert_dtype=fp4 is misleading for these
INT8 checkpoint tensors. E6 needs uneven43/42 ownership, not native integer-floor
EP placement. Capacity evidence is checkpoint headers, not a runtime peak gate.

For a smaller shared-expert separation candidate, read roles/QWEN-NEXT-PLAN.md.
Local Qwen3-Next80B is BF16:144GiB target routed weights plus3GiB MTP. E2 does
NOT fit64GiB devices; E3 or E4 does on weight accounting. Its gated shared expert
is one same-width MLP versus top-k10 routed MLPs; overlap opportunity is real but
coverage is unmeasured. Preserve native36GDN/12full-attention state management.

Qwen3-Next A2/E4 implementation lives in `attention-client/qwen-next/README.md`.
Four-layer dummy native hybrid generation and independent MoE oracle pass in
131201 (30/54 calls, relL2<=0.000255). A2 inputs retain gated shared compute
between submit/collect; this is not a measured overlap speedup. Compile QWEN_NEXT
for top-k10/512 experts and single-layer slot catalogs; legacy shape remains the
default. Layerwise weight pointer lookup preserves two reusable workspaces, not
48 full group catalogs. Set persistent server device execution timeout explicitly
for cold compilation; retain a bounded supervisor. An arm with a daemon device-snapshot thread
had an unclean teardown; the observer was removed rather than accepting that arm. Never report audit
weight reconstruction memory as ordinary client peak. Full-model evidence remains
separate from these fixture gates.

Full48 BF16 real-weight A2/E4 passed132952; see `qwen-next/real-result.json` beside
the fixture evidence. All six roles exited0 and matched242/1010 calls; oracle
layer0/3 relL2<=0.000158. Prepare reference weights/results BEFORE registering IPC,
not while persistent service is live. Earlier real arms generated successfully
but failed during post-generation reference construction; causality is unresolved.
Clients retain no routed weights; E4 each holds36GiB. Five short requests and1252
waves for1252 calls do not demonstrate batching/throughput or full quality. The
1GiB fixture KV budget avoids the observed128MiB smoke-test preemption. Report
Torch allocated peak separately from device residency and diagnostic allocations.

Qwen Next six-role profiling: `NEXT_PROFILE=1`, then the shared profile_export.py
with explicit six `--roles` and `--label attention2-expert4`; see the Next README.
134603 passes all role exits. IPC has no HCCL clock-fit markers: provider-clock
translation is not independent calibration. Persistent E bars include waits and
do not expose inner GEMM phases; never count their entire duration as useful work.

Next FULL decode gate141742: see qwen-next README/export_attention.py. Use mode0
plus native runner.use_aclgraph metadata initialization; Dynamo cannot trace the
ctypes kernel calls. Inline remote nodes into outer capture, not nested replay.
17 native RI replays contain816 submit/collect calls in one graph model. Prefill
remains eager (GDN UNIFORM_BATCH). Export one native attention hierarchy, not only
flattened distributed lanes; server persistent bars are uninformative internally.
Collect214us under FULL versus eager7.7us does not prove slower experts: eager host
supply changes how much wait remains at collect. Retain that attribution boundary.

### A2/E4 efficient-server confluence and long-lifetime completion race

The qwen-next tree combines layer-addressed open service with continuous GEMM,
fine pack, early return and token-owned route pull/reduce. Enter its README's
confluence section before reusing builds: server config slots24–26 now own open
service/weight table/layer count; earlier serving slots16–18 collided with the
fine scheduler. Launcher requires one fresh `abi.json`-qualified binary closure.
Open service disables bounded generation-indexed work observers; cross-layer
weights override both whole and continuous GEMM addresses. K10/E4 geometry differs
from the older K8/E2 leaf; do not copy raw export offsets or move quanta.

Observed in failed `qwen-next-20260916T153945Z`: server3 slot1 full down completion
2322 with prefix readiness stuck at2318; preceding full48 run153434 hung during
third-request decode. Inference supported by source ordering: completion can occur
between the prefix and full readiness polls, and clearing the active Cube slot
then loses the prefix publication forever. The full-completion branch must also
publish prefix readiness. General invariant: a stronger completion observation
must satisfy all still-required weaker notifications before its owner retires.
Use `NEXT_WIRE_STRESS=1 NEXT_WIRE_ONLY=1` for a cheap unequal-source/EOF/long-generation
gate; optional control snapshots are diagnostic, not valid performance evidence.

The fixed confluence passed wire154246 (290/1058 calls), full48 real154330 and
same-host old-path control155142. Source1 FULL graph median20.07124→18.91032ms
(17 steps each; same output IDs). See qwen-next/confluence-result.json, not older
A2/E2 DFC numbers, for this specific scope. Concurrency155704 showed zero paired
waves even for nominally simultaneous same-layer sources. Prepublished-source
leaf160001 did pair:797.67us same-layer versus1112.53us different-layer median.
This distinction is preserved in qwen-next/CONCURRENCY.md: admission can batch,
but independent-client arrival/early-slot assignment can miss the opportunity;
no waiting-for-batch policy was added or uniquely blamed by that observation.

### Bulk expert-source frames (2026-09-16)

For prefill batching rather than repeated 32-row decode fixtures, enter
`prototypes/attention-client/qwen-next/bulk-prefill/README.md`. Frozen 1024-row
source variant passed three-device run163223: dual128→dual1024 grows input 8x,
coordinator span0.966→2.550ms; solo1024 vs dual512: 1.569/1.582ms. Both sources
are ready before admission; only one quarter-expert owner, no full serving
throughput claim. Published serving ABI stays32. Same-layer coalescing works
under backlog; continuous arrival opportunity remains separately unproven.

Avoid a costly harness trap: `device-service/admit_subset.py` copies sibling
Python files into its snapshot and prepends it to PYTHONPATH. That shadowed the
expanded runtime with32-row allocations while loading its widened binary,
causing507011/memory corruption. Isolate the admission helper directory and
assert the runtime module path matches the binary/geometry closure before NPU
initialization. Frozen coordinator IDs now use int16 UB storage and workers
retain strided maps; increasing the scalar stack to128KiB is not supported by
the current compiler. Do not infer algorithm limits from mismatched-ABI failures.

### Slack-aware expert priority (2026-09-16)

Enter `qwen-next/priority/README.md` under `prototypes/attention-client` for
decode/prefill scheduling. Original class, dynamic urgency and layer compatibility
are separate contracts. Prefill client publishes a generation-tagged promotion
after shared, before collect; source+16 is not the descriptor class at+11.
Check both unclaimed and already-staged tasks. A staged prefill must remain in
burst-fairness accounting or the other slot can recycle decode indefinitely.
Current scope is two homogeneous source frames, not arbitrary mixed rows.
Role ABI4/client16 words rejects stale clients; graph decode omits no-op promotion.
Final four-layer native gate165554 and bulk priority165658 pass. Diagnostic165349
observed one in-slot promotion but predates decode no-op removal; do not report
it as a final performance A/B. The linked model-readiness audit separates the
new Qwen4Exp W8A8/PLE requirements from the existing BF16 Next path.

The new Qwen3.8 snapshot is fully downloaded under shared_models; see priority's
model-readiness.md/model-snapshot.json for222,866-tensor coverage. Do not mistake
filename W8A8 for uniform quantization: target experts are per-expert I8, MTP
experts fused BF16, with95.43GiB host-capable PLE. Reuse LiveInfer's owned branch
`lumi/qwen38-flash-next-serving-plan` at820103bf (not main); its host-PLE/full-root
qualification does not qualify this quantized loader. The current static root
requires TP2/4/8 and EP==TP; remote-expert injection must preserve QSA/shared
ownership while removing local routed allocation. Explicit-revision ModelScope
inventory matters: README and .gitattributes differed from the initial listing,
although all weight hashes matched. Final corrected inventory/census passed.
