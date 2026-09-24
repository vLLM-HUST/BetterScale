# Dissect the separated-expert transport before changing it

Enter for the post-cap7 communication investigation (2026-09-24), not the
concurrent event-client/dual-attention-graph fork. Source base a4588cf in
`codex/expert-transport`; task artifacts at workspace
`runs/expert-transport/20260924/`. Do not edit the other fork's source or runtime.

## Protocol and interpretation boundary

The d1d67fa cap1/cap7 SWE comparison is **whole-layer** placement, not EP.
Per-layer there is one remote owner. Source publication and output completion
are generation-tagged IPC words; host sockets bootstrap/drain, not per-layer
RPC. Server fetches hidden once, fans top8 out locally, combines locally and
returns one reduced hidden. Principal peer payload is8192 bytes/token/layer
(H2048 BF16), plus route/probability/control traffic. EP's repeated hidden
fetch and route-valued return have a different byte budget.

The production reduced collector starts16 blocks, each polling the same remote
DONE before copying its partition. It includes queue/compute/return wait: its
profile envelope is NOT network-only time. Cap7 actual lifetime frames/wave
were only about1.015 A4 /1.118 A6; small throughput gains do not rule out compute.

## Echo and matched copy observations (not complete FFN cost)

Prototype `prototypes/expert-transport/` uses the unchanged production
pack/publish/collect/retire functions plus a minimal dedicated echo server.
It omits actual routing/grouping/compute/reduction, so neither its roundtrip nor
its server movement is an exact isolated production stage. Five device-event
samples each,64 unrolled calls per FULL graph, BF16-exact changing payloads,
exact final generations. Local selected-device admission, guards, exits/release
retained. Other cards had foreign serving workloads; no isolated host-ceiling
claim. No msprof overhead in these timings.

`echo1` on4/5:6930 generations, PASS. Roundtrip medians with16-block production
collect: zero-row10.92us,3rows13.30us,96rows50.01us,4096rows1659.13us.
One-block collect:9.71/12.65/57.98/2013.24us. Split one-block wait +16-block copy:
11.13/13.42/50.51/1646.94us. Single collector also changes copy parallelism.
Thus redundant polling is not established as the dominant delay in this idle
single-source echo; this says nothing conclusive about busy multi-source servers.

`echo2`/`echo3` on6/7: PASS. Same production roundtrip about1659us/4096rows;
fused ping-pong return copy about1658us, not a useful whole-roundtrip improvement.
16MiB standalone local copy21us, double-buffered14us; remote pull about815us
(~20.6GB/s useful payload). Four versus16 blocks and double buffering change
little once the remote path saturates; single nonpipelined block is slower.
Runtime memcpy pull about817us; remote pushes about864us for AIV and883us for
runtime memcpy. This strongly motivates testing useful overlap/topology rather
than assuming another copy implementation fixes the bulk path. It is NOT proof
of link hardware maximum, full-model network dominance, or DFC parity.

`npu-smi info -t topo` reports all pair relations HCCS. IPC import flag1 is already
ENABLE_PEER_ACCESS, not a missing peer-enable bug. Official API:
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_1936.html

Native DFC's `CopyGMToGM` (pinned vllm-ascend
`csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_kernel.hpp`)
also uses two32KiB UB slots with per-slot MTE2/MTE3 ownership. It pipelines group
readiness into compute. Our production server FETCH already uses this style;
client pack/reduced-return use the older draining IO helper. A pipelined client
copy is measurably better locally but not yet a qualified MOD improvement.
Do not replace source-controlled code based on local-copy-only gains.

## Next discriminator

`ffn.py` provides the complete real-weight A1E1 two-layer call (0/40), hot8 versus
broad routing,3/96/4096rows, FULL replay and changed-input independent oracle.
A1E1 is a leaf exception, not enough memory for a whole-model E1 deployment.
Compare transport lower bound against this complete cost before attributing the
serving limitation to bytes. Any optimization must retain exact generations,
no early source/slot reuse, and complete output correctness. The mature DFC
comparison additionally needs explicit geometry/catalog/topology matching;
old EP2 H2048/M768/E128 results do not qualify Qwen35 M512/E256.

## Serial routing preparation first (Fletcher's goal update)

`ffn-pull1` on6/7 passed real layers0/40,3/96/4096 rows, hot8/broad patterns,
changed hidden/IDs/nonuniform probabilities and exact generations. Complete
median calls:3rows~195us;96rows hot8~284us/broad~1449us;4096rows hot8~5595us /
broad~6915us. Echo13/50/1659us is not a matched subtraction of these costs.
The server's final512-command ring covers the final4096 broad case (including
changed-input gate). FETCH median~830us, FETCH-end→REPACK-start~1533us, REPACK~177us,
UP~1321us, DOWN~789us, SEND~699us. Commands overlap: component medians must not be
summed as an additive decomposition. No calibrated cross-device clock.

`build_group.py` makes a temporary build, not a default MOD change. Candidate2
DMA-writes accepted IDs and DMA-reads them into UB for Group count/map creation;
it preserves source/expert ordering, maps, counts and segmented boundary. Candidate1
only changed readers and is NOT qualified: scalar-GM cache producer visibility
would need separate treatment. Never run that incomplete build. CPU execution
of actual old/new Group agrees across layer/EP2/EP4,1/2/7 sources,1..4096rows,
empty owners/hot/broad IDs and segmented modes/tail cuts; bounds checks cover the
196608-byte UB allocation. Device gate remains separate.

Existing-primitives probe `planner1` on local5 passed counts, complete permutation
and hidden-row/expert-bucket mapping in FULL graphs. Native
`npu_moe_init_routing_v2(...quant_mode=-1,row_idx_type=0)` produces a row-major
source→destination inverse map in this installed runtime. Five32-call samples:
4096 broad native routing with32-wide dummy hidden~143us, fullH2048~208us; a
composed FP32-unique-key sort + scatter inverse + native prefix operator~601us.
At512 broad, fullH2048~54us beats dummyH32~92us, so narrower hidden is NOT a
universal faster tiling.3rows native~11us,96rows~26–28us. This is primitive cost,
not source-plan IPC or multi-client coalescing acceptance. The prior Qwen38 plan
is read-only foreign work; this probe uses native APIs without importing it.

Next routing option if simple DMA staging remains expensive: prepare native
per-source inverse maps/counts before READY, let the owner combine per-expert
counts, and distribute map-offset adjustment to movers. Preserve no-wait batching,
source lifetime, expert-local ownership and canonical final weighted route order.
This is a design, not yet an implemented ABI or accepted performance improvement.

After metadata, study DFC plus AllGatherMatmul (input-ready→GEMM) and
MatmulAllReduce (GEMM-ready→return), as explicitly requested. Official tutorial:
https://asc.gitcode.com/guide/operator_practice/simd_operator_impl/fusion_operator_programming/general_fusion/operator_impl.html
Its dense collective/local-work overlap cannot be transplanted unchanged to
asynchronous expert-only owners with no local attention input. Borrow dependency
and lifetime rules, not collective synchronization or hardware flag IDs blindly.

### Native source-plan prototype acceptance

`ffn-group3` (local4/5, group-dma-build2) passes correctness but does **not** fix
serial preparation: final large-frame gap~1674us vs original~1533us. Do not
promote this merely because the memory access looks cleaner.

`build_plan.py` / `client_plan.py` introduce an explicitly cap1/layer-only wire
prototype. Native routing inverse/counts are prepared on the client and copied
by pack into existing aligned IPC padding beyond the maximum hidden payload.
READY still follows complete pack; mappings stay immutable until DONE/retire.
Server validates original IDs and plan counts, consumes native maps, and retains
the original segmented boundary and fixed-order weighted combine. Extra wire
payload is4*n*8+2048 bytes per planned source frame; principal hidden traffic is
unchanged. No server per-frame host launch or communication RPC is introduced.

`ffn-plan1` passed real-weight/changed-input gates on4/5. Its final Group-containing
gap is~22.66us (not1533us);4096 complete calls~4.27ms hot8 /5.56ms broad. Always
planning regresses tiny frames. `plan-build2` gates planning at512 rows and shares
legacy behavior below the threshold; ffn-plan2 passed the expanded3/96/256/512/
1024/4096 sweep, layers0/40, compared with fresh `ffn-control45` on the same pair.
The512-row complete-call gain is only~0–1%;1024/4096 improve materially. Small
frames still differ across runs despite taking the old algorithm, so do NOT
claim decode neutrality yet. Control timings themselves drifted relative to the
first6/7 run. Shared-host samples are not a repeatability/isolated-peak estimate.

The next `plan-build3` uses one common Group body (not duplicated legacy/native
functions) and a conservative1024 threshold. It needs its own device gate and
bracketing control. Prototype remains outside default MOD source; batching/EP,
full serving and integration gates are still open. CPU execution compares actual
emitted Group maps/counts/boundary with the original across legal cap1 shapes,
routes, segmented modes and empty sources; this does not prove device visibility.

## Source-owned MOD integration branch

Implementation is being qualified independently in `codex/expert-transport-mod`,
worktree `../expert-transport-mod`; keep `codex/expert-transport` as the immutable
research/prototype lineage. Neither branch modifies `expert-event-client`.
The MOD owns native planner invocation, wire constants, build flag and CPU ABI
admission; no prototype imports. `--route-plan-min-rows 1024` is opt-in, default0
unchanged, and rejects cap>1/EP until those have a compatible plan-composition
implementation and hardware evidence. Original IDs remain independently checked
on the server. Source window extents remain unchanged; new metadata fits padding.

Prototype plan-build3 (one shared Group body, threshold1024) passed the same
expanded4/5 gate. The small-frame slowdown observed with duplicated Group bodies
was absent in this run; compiler-layout sensitivity and shared-host drift are
not isolated from one another. Do not infer an exact crossover or default-on
policy. MOD build/CPU tests and fresh complete-call gate precede any hw3 model
load. Full multi-attention-rank MTP/shadow/drain and performance remain open.

MOD `mod-plan-build1` / `ffn-mod1` passed the same real-weight24-case timing and
changed-input gates on4/5 using ONLY package-owned producer/consumer code (no
client-plan monkeypatch).41-layer build, layers0/40 exercised, exact generation
drain and admission exit0.141 CPU tests pass.4096 complete medians~4.31ms hot8 /
5.54ms broad; small paths~original leaf range. This is still a two-layer leaf,
not whole-model or cap7/EP qualification. Subsequent source tightens per-expert
plan-count bounds (<=frame routes) before whole-model acceptance; no normal-input
arithmetic changes. New build and exact-build shadow receipts remain required.
