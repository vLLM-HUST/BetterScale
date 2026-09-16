# Extend DSV4 prefill/mixed FULL graph

For the newer bounded **DP8 device preparation/metadata continuation**, enter
[the fresh evidence and ownership boundary](liveinfer-continuation.md#fresh-bounded-dp8-continuation-september13)
and `src/betterscale/patches/async_decode/README.md`. Do not confuse it with
the rejected all-mode worker-retirement extension or expand the TP8 claims.

## DP startup preparation (0.3.1)

`async_decode/_warmup.py` prepares the finite auxiliary graph catalog before READY;
DP two-seat K5 has4 producer banks and6 metadata entries. Runtime misses fall back,
never grow the catalog. See the module README and `docs/evidence/release-0.3.1.json`
for the reused qualified implementation and release provenance. This DP-only fix
does not adopt the experimental TP/N+2/early-budget extensions. Reuse the measured
run155 rather than repeating an expensive eight-card load for package identity.

## Maintained serving entry

For deployment and reporting, start from `docs/RUNBOOK.zh-CN.md` and
`patches/README.md`, not historical launchers. Install the package into the user's
existing donor environment; the only public entry is native
`vllm serve ... --worker-cls betterscale.worker.Worker`. The worker installs
the kept hooks before construction and after native warmup. No custom CLI,
CANN/LD_LIBRARY_PATH/HCCL/allocator setup, private profile, required artifact
folder or post-start activation RPC remains. Receipts in old capsules are
historical; the shipped worker now logs READY and performs no file writes.

The launcher removal does not expand the qualified configuration envelope or
prove fresh NPU quality/performance. `config.py` validates native options without
mutating them; the four-full-context KV-capacity gate is removed, not the graph
shape/parallelism guards. `docs/acceptance.json` is the OLD CLI's HTTP acceptance,
not the new entry's hardware result. CPU lifecycle tests verify hook order,
native configuration/environment preservation, and no artifact-directory writes.
Build/install the wheel in a disposable environment; do not upgrade donor deps.
Check wheel contents, not only the source tree: setuptools reused a stale
`build/lib/strengthen_dsv4/cli.py` after source deletion. A clean build (move old
build output aside recoverably) removes it. Assert no console entry or cli.py in
the wheel and resolve the installed worker class without depending on checkout
PYTHONPATH. Native-base stubs cover packaging only, not accelerator execution.

Maintain readable package code with `python -m black src/betterscale`
(default style, py312 target in pyproject.toml). Do not pack assignments or
control-flow statements onto one line. Run the formatter in a development
environment, not by adding dependencies to the donor runtime.

Write module READMEs for a reader who has not inspected donor source: lead with
the supported workload and problem, place the patch in the original execution
flow, explain the mechanism with concrete input examples, then map to hooks and
bounded evidence. A list of private API names is an inventory, not an explanation.
Use target_full/README.md as the local example; do not assume the reader shares
Lumi's experimental context.

The TP combination uses six closed patch directories: compat_lcm,
target_full, ordered_replay, qli_cpu, split_draft, cross_step. Each owns install;
worker alone composes them. DP adds the closed async_decode module and native
DSA branch of target_full; it does not install the TP-only split_draft module.
split_draft owns _graph.py; its metadata normalizer
lives inline beside DraftGraphRunner in __init__.py. Old flat
file paths in historical experiments refer to their original capsules, not
current maintained code. Importing qli_cpu no longer mutates a donor method;
ordered_replay owns its wrapper hook rather than borrowing target_full's installer.
Both install after warmup (the previous ordered wrapper was inactive/fallback
during warmup). `tests/test_patch_installation.py` covers import inertness, isolated
hook ownership and repeat installation against native CPU doubles. This is not
new NPU qualification or a promise of arbitrary subset performance.

Ascend preserves explicit worker_cls (only `auto` is substituted). Removing this
worker also removes its K5/TP8 LCM repair, so a native rollback must use a known
working donor command, not promise an identical-config unpatched A/B. Historical
prototype controls remain separate from the sole packaged integration entry.

## Historical investigation and experiments

Enter here when interpreting donor host gaps, deciding graph coverage, or
implementing a fixed-budget DSV4 prefill/mixed graph. Read
[the investigation and gap map](investigation.md) before changing graph modes.
That file records the initial source audit. For the subsequent bounded TP2
implementation and exact shadow checks, enter
[`prototypes/full-mixed/README.md`](../../../../../prototypes/full-mixed/README.md).
Do not confuse this target-only result with K5, TP8 or production acceptance.

Use the pinned submodules. The installed runtime that produced the September 12
trace is separate: three key Ascend files (DSACP attention, ACLGraph wrapper,
fused MoE) were byte-compared and matched the pins. No blanket whole-environment
identity is claimed.

Do not mistake a dense device task track for graph replay. Use native API links,
not only the collective-end-aligned export. The reproducible CPU-only gap tool
is at `evidence/inspect_launch_gaps.py` in the repository root; it consumes the
native provider DBs referenced in `evidence/donor-launch-gaps.json`.

The long wait_event is a one-off, not a general root cause. Empty compute/comm
coverage is not proof all engines are idle or a removable speedup estimate.
The native profile lacks event handles/producer links and CPU scheduling data;
do not infer a particular MoE event from temporal proximity.

Preserve existing decode FULL and the upstream shared-expert overlap. Start
with a fixed-shape model-forward probe before integrating scheduler dispatch.
Use probe-npu and lease/admission rules before any accelerator execution.

## Fixed-capacity lesson from the first passing probe

Native Compressor faults were traced to request-capacity mismatch: generic FIA
padding changed replay metadata dimensions despite capture retaining four rows.
Keep descriptor request capacity and repeat the final actual query offset through
inactive rows. Stable addresses alone are insufficient; tensor dimensions and
captured scalar bounds are part of the protocol. The CPU regression exercises
the exact padding function used by the probe.

Use shadow checks on valid output rows and the entire KV state. FlashComm1
outputs are TP-local; padded output rows have no semantic contract. The accepted
run012 has zero output/KV difference at rtol=.01, atol=1e-6. Coarse absolute .01
is unsuitable for this dummy model's small outputs. Shadow timings and memory
include copies/comparisons and must never be used as serving performance.

DSpark dummy fixtures need the draft's auxiliary target-layer IDs to follow the
shrunk target. Upstream dictionary hf_overrides deliberately do not propagate to
draft config; applying a callable to the target instead conflicts with Ascend's
quantization config requirement. Keep fixture repair separate from engine patches.

## TP8 oracle: heterogeneous pool aliases and determinism

On hw3, run006's eager/eager control and run007's graph/eager control both
passed with HCCL_DETERMINISTIC=strict (66 and65 checked steps/rank respectively,
all differences0; run007 includes6 mixed waves). Non-deterministic eager/eager
itself exceeded the tight tolerance. Do not attribute that failure to graph.

The allocator maps multiple `kv_cache_tensor.shared_by` layers onto ONE raw
allocation. Runtime observation found SWA BF16 views and compressor FP32 views
with the same data pointer. Each group's block table selects its owned pages.
Comparing the entire pool through every BF16 view interprets other groups' FP32
state bytes as BF16—including apparent NaNs and huge values. This is not evidence
that the corresponding active attention KV contains NaNs.

For exact deterministic state verification, snapshot each unique untyped storage
once and compare raw bytes. This both preserves the entire backing and avoids
many redundant whole-pool clones. The CPU alias/restore test covers different
dtypes and nonzero view offsets. For tolerance-based verification instead, one
would need group-owned pages and their true dtype; do not invent a tolerance for
aliased whole-pool views. Keep deterministic correctness controls separate from
normal high-performance serving measurements.

Native MRV1's max(K+1,TP) alignment rejects K5/TP8 although LCM24 works. The
probe's alignment repair changes capture bucket sizing only, not speculation
length. Both large and decode buckets must remain divisible by the joint
alignment and survive max-capture filtering.

## Decode/draft continuation

Enter [`prototypes/full-mixed/DECODE.md`](../../../../../prototypes/full-mixed/DECODE.md)
for the scoped stream-ordered target replay, CPU QLI maxima and private-bank
DSpark body graph. Native DSpark explicitly disables graph in its constructor;
its config flag alone is not evidence of graph coverage. The donor already has
async DSpark scheduling: reuse it rather than transplanting another scheduler.

A runtime capture is initialization, not a committed invocation. Run021's first
capture returned matching dummy token IDs but different KV bytes. Explicit first
replay fixed that boundary in run022; later replay checks alone had missed it.
Check the INITIAL invocation as well as later replays. The CPU emulation test
protects this contract even on a backend whose capture does not execute writes.

Match the reference to the owned change without erasing failures: native
FULL_DECODE_ONLY graph/eager whole-pool comparison failed in run023 after output
checks passed. Its cause is not established. `native_graph` compares our replay
policy against the unchanged native graph with exact backing-byte checks; it is
NOT a graph/eager equivalence claim. Keep this distinction in results.

Compare equal-work intervals (four real requests each scheduling six target
queries) and report the actual wave counts. Greedy/speculative nondeterminism
changed a fixed-output cohort from20 to32 waves in run015; cohort completion
time alone would have dramatically overstated the small fence-only benefit.
Device-event spans include queued work/waits; the inter-forward interval includes
metadata/copies/sampling and is not synonymous with idle hardware.

For repeated eight-rank profiles use the tested
[`profile_tools`](../../../../../prototypes/full-mixed/profile_tools/README.md).
Worker-side torch-npu export cannot parse in a daemon; collect first, parse
offline. Reuse TraceLoom's fitter/exporter, retain candidate-only clock receipts,
and link compressed timelines rather than putting raw JSON into the conversation.

## Cross-step authorization and receipt placement

For the next dependency cut enter
[`CROSS_STEP.md`](../../../../../prototypes/full-mixed/CROSS_STEP.md).
DSV4 device progress correction already exists. CPU sequence lengths and exact
positions are different contracts: in the admitted DSACP decode path CPU lengths
only set conservative tiling maxima, while device tensors drive actual addresses
and attention. Do not transfer that conclusion to DCP or other builders.

There are TWO pre-forward receipt consumers: CPU length correction and the
compressed-model early deferred-state callback. Removing only the former still
leaves the latter. The bounded prototype submits target first, retires current
input DMA, then applies the original callback once. Native input-prep fences
remain necessary: the callback can mutate pinned CPU H2D source tensors.

A valid oracle must rebuild ORIGINAL exact-length metadata before reference
execution, not run both sides with the candidate metadata. Run028 passed 65
such checks per rank with dummy weights; run029 passed 17 with full real weights
and 0–5-token CPU bound slack. Run030/031 independently measured the dependency
cut at about 10–12% shorter matched K5 cycles. Keep the scope in the linked note.

Warm all bounded draft bank counts and verify coverage on every rank before
claiming warm cohort throughput. Eight-token warmup did not guarantee that;
run030 had a 902 ms two-seat draft spike consistent with first capture. Run031
explicitly prewarmed all four shapes and eliminated the spike. Preserve cold
shape-admission cost separately rather than blaming every outlier on acceptance.

Do not trust a near-identity clock fit without its holdout residual. Run030's
replay-associated provider names gave rank2 a 113 ms P95 despite a near-zero
median offset. The helper now gates export at 50 us; eager-only native-task
membership gives consistent candidate markers. This is a semantic restriction,
not deletion of inconvenient residuals. Preserve the rejected fit and keep
same-rank timing conclusions independent of distributed display alignment.

## All-mode N+2 continuation

Before reopening continuous decode, read [LiveInfer's continuation protocol](liveinfer-continuation.md).
It distinguishes host authorization from device-derived execution parameters,
explains DONE/generation drain across N+2, and separates the producer/metadata
speedups from the later worker cut's lack of incremental benefit. Do not infer
that acceptance is an unavoidable CPU barrier or that the whole transplant was
qualified from the target replay's submission lead alone.

Enter [`N2.md`](../../../../../prototypes/full-mixed/N2.md) for the native
two-wave queue, deferred-free fence, mixed/turnover bounds and bounded all-mode
draft banks. Run036 qualifies the dummy envelope, not real-weight service.
Prefill padding exposed signed-zero differences; the oracle exception is
confined to addressed BF16 rows, never entire heterogeneous pools. Exact
prefill/decode counts can change while the pinned draft forward consumes only
the prefill boolean; retain the actual tensor offsets and original reference
metadata instead of generating a graph for every equivalent composition.

Runs039/040 additionally qualify real weights at budgets288/4128 against the
explicit **padded** draft program, with byte-identical graph/eager KV and IDs.
Run038 preserves why that is NOT original unpadded-draft equivalence: a few
BF16 differences amplify downstream and change proposals. Keep target/state
correctness separate from proposal quality; the unchanged greedy rejection
kernel verifies every published token, while acceptance changes still belong
in the work/throughput account. Do not silently switch oracle meaning.

## Prefer separate context ingestion to oversized draft banks

Enter [`SPLIT_DRAFT.md`](../../../../../prototypes/full-mixed/SPLIT_DRAFT.md)
for the smaller post-N2 route. DSpark's native merged runnable contains TWO
workloads: target-hidden context KV ingestion and small candidate-query execution.
Do not mistake large context counts for large candidate-query counts, or pad all
context to the target wave budget merely to graph the query body. The split
prototype retains fused small decode and admits actual-length context ingestion
before query-only capture; no new scheduler is required.

The041/042 total-time difference includes different output trajectories and
wave counts, including first-token differences within a repeated run. Their
cause is not established. Fletcher stopped the separate divergence inquiry;
do not restart it implicitly or relabel it proven harmless. Quality acceptance
uses an explicit OpenCompass gate; matched-step timing is a separate claim.

For a bounded quality gate use [`QUALITY.md`](../../../../../prototypes/full-mixed/QUALITY.md)
and its native runner/scorer instead of rediscovering the workspace's historical
OpenCompass deployments. The retained32 original-input retrieval questions are
9.9–15K tokens:3GiB KV only holds1.19 max-length requests and is NOT a four-seat
quality envelope. Native hybrid-aware capacity gives4.77 at12GiB. Runs048/049
score32/32 for both the prior optimized control and split candidate. This is a
LongBench retrieval subset pass, never a full OpenCompass-suite claim.

## Investigate KV prefetch hidden behind matrix work

Enter `prototypes/kv-prefetch-overlap/README.md` before repeating communication /
GEMM interference experiments or proposing PCP integration. It maps existing TP
output-weight gathers and DP/MoE multistream consumers, retains native rank-local
matmul shape/block inventories, and separates net block-time savings from compute
slowdown. Neither a Matmul name nor timeline-envelope overlap proves spare Vector
resources or free communication. The HCCL payload microbench omits owner State pack
and existing EP traffic; it is not a serving or whole-layer qualification.

This runtime rejects elapsed_time for timing events captured inside a graph
(`event recorder null`,507000). Use external events around fixed repeated-operation
graph blocks, preserve that throughput-block scope, and record actual NZ format.
The probe explicitly enables internal format; otherwise format_cast can warn and
silently leave ND. Keep selected-device admission and source/destination roles in
the capsule, including rejected windows. Do not compare different hosts as if
only the communication policy changed.

For **captured local DMA rather than HCCL**, enter
`prototypes/graph-dma/README.md`. The single-card ACL H2D/D2H/local-D2D fixture
uses explicitly pinned host storage and distinguishes fixed-address content
updates from capture task-group parameter updates. Its sustained GEMM overlap
results differ materially from HCCL/AIV; do not transfer backend conclusions.
The first24-case run passes; task-update support and multi-layer consumer/slot
lifetimes are separate gates, not implied by that throughput microbenchmark.

For whole-model rather than GEMM-only overlap, read
`prototypes/graph-dma/qwen-dp2/README.md`: real Qwen DP2EP2 FULL prefill plus
independent graph-external bulk DMA. Native Qwen attention consumes/resets
ExternalEvents, so replaying its bare graph handle without parameter-update
publication stalls even with unchanged metadata. Reuse native `_model_forward`
when freezing this invocation. Bulk host DMA and sustained local D2D have very
different observed interference; one fast4GiB local copy does not characterize
continuous traffic. The fixture preserves full KV-byte and output checks.

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
