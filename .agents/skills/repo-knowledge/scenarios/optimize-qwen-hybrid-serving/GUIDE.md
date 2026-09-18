# Optimize Qwen hybrid TP2 serving

Enter here before profiling or extending FULL coverage for Qwen3.8-27B HTTP
serving. This is not the DSV4 Worker or the ordinary Qwen3-30B-A3B owned reactor.

## Identity and reproduction boundary

Observed local model `/models/vllm-ascend-models/Qwen3.8-27B` is qwen3_5_text inside
Qwen3_5ForConditionalGeneration: 64 layers,48 GDN/16 full attention,hidden5120,
head256,Q24/KV4, BF16~51.75GiB. One MTP layer's15 tensor keys actually exist.
Do not substitute the512-expert Flash-Next model or Qwen3-30B-A3B.

Qualified donor Python:
`/workspace/my-ascend-workspace/runs/rp-legacy/20260903T155041Z-layout/rp-upstream-0.25.1/.venv/bin/python`.
Native vLLM752a3a504485790a2e8491cacbb35c137339ad34,
Ascend9bf964cb4b87c8cd0d6852c41a55b3c29711fa95,
Torch2.10.0+cpu/torch-npu2.10.0.post2/Transformers5.14.1.
Use baseline `/usr/local/Ascend/ascend-toolkit/set_env.sh`, append rather than
replace CANN's PYTHONPATH (otherwise `acl` disappears). Never modify this runtime.
TP2 probes use two idle leased cards, not all eight; obey current admission skill.

Harnesses live in `prototypes/qwen38-serving`. Frozen source per capsule plus
source-commit.txt is essential: no live-source experiment masquerading as a release.
Runtime evidence root `/workspace/strengthen-dsv4/runs/qwen38-tp2-serving`.

## FULL can be hidden by overlap — paid counterexample

Do not infer FULL is useless from the older2048-token result. Historical native
and fixed-FULL profiles had rank0 model bodies448.8/446.3ms, but host entered
its final event wait413.7/5.7ms after model start. CANN calls in that body dropped
~22369 to235. Native submission already overlapped a long device workload.
A long FULL receipt wait is not device idle or a new graph overhead.
Evidence: `full-prefill-boundaries.json` in the evidence root; historical paired
DBs under `/root/my-ascend-workspace/runs/qwen38-27b-tp2-{baseline,fullgraph}`.

The discriminating experiment `fixed-full512-1` uses the same loaded model,
metadata adapter, async HTTP frontend,1seat,APCoff,512input/1output. Idle RPC
switches only prefill FULL versus ungraphed compiled NONE (not uncompiled eager).
Six measured requests/mode in ABBA order: mean unprofiled TTFT359.645→178.366ms.
Eight queued requests, two cohorts/mode:2.83990→1.41675s. Output one-token texts
match; this is not broad numerical parity, model accuracy, or general GDN support.

Four profiled prefills/mode after timings, both ranks:
- rank0 body405.864→162.522ms; compute union118.355→117.797ms;
  communication39.637→40.050ms; uncovered247.872→4.675ms.
- rank1 NONE communication272.390ms, uncovered15.469ms;
  FULL communication40.871ms, uncovered4.726ms.
This supports host-submission starvation plus peer communication waiting, not
faster matrix kernels. Profile overhead is NOT the HTTP performance result.
Uncovered compute/communication may contain memory/control; don't label pure idle.

## TraceLoom protocol pitfalls

Torch integrated export can omit CaptureStreamInfo even when four ACL graphs
really replayed. Feed native `msprof --export=on --type=db` from a copied raw PROF
directory to preserve graph evidence; never recapture hardware just to fix export.
Directory input's TraceLoom metadata source_path is its resolved `msprof_*.db`
child, not the input directory. Validate that relationship, not false equality.
`native_graph_export.py` and `export_timeline.py` implement the paid route with
frozen TraceLoom37323af. Separate per-rank timelines are not cross-rank aligned.
`compare_full_prefill.py` guards16 FIA and128 communication intervals/body;
512 uses304 MatMulV2 before final norm,2048 used256 MatMulV3. Async host traces
do not satisfy the old single synchronous-event-wait invariant; disable that
specific inference instead of choosing a convenient event.

## Current extension boundary

GDN chunk prefill includes host cu_seqlens/chunk_indices lists consumed by
AscendC fwd_h/fwd_o. Stable tensor addresses alone do not make a graph captured
for512 safe at513. Exact shape contracts, correct state writeback, and live
request membership must be proved before padding/general mixed replay.
`bucket_full_worker.py` is a bounded prototype: per-shape stable metadata,
single-prefill exact buckets, native mixed/other-length fallback,8live seats.
Do not call it production-qualified without the actual capsule receipt.

### Padded GDN investigation: current hidden is not a state oracle

Experimental identity padding masks q/k/v/g/beta outside the real endpoint and
runs chunk recurrence against a static padded cu_seqlens. Native conv retains
its real tensor endpoint. `padded-full2` reduced2051 TTFT~812→549ms, but513 FULL
produced corrupted continuation; reject this candidate pending state validation.
One-shot `padded-shadow3` compares FULL against restored-before-state NONE:
valid hidden[:513] was exactly equal on both ranks while112 cache checks differed.
Metadata revealed GDN qloc[0,513,513], two prefills and stateindices[3,0]: FIA
padding inserted a zero-length virtual request, bypassing single-prefill stable
buffer binding. Correct present hidden does NOT validate future decode state.

The pinned AscendCommonAttentionMetadata.unpadded override intentionally retains
block_table_tensor/slot_mapping unsliced for FIA, unlike core vLLM's helper.
For a GDN-only real-request view, explicitly slice block-table rows too; merely
calling unpadded left state-index shape2 and triggered the stable shape assertion
in `padded-shadow4`. Do not alter shared FIA metadata while repairing GDN.

Full6GiB-cache before/after snapshots plus13 graphs exceeded59GiB in the shadow.
Use the diagnostic1GiB KV budget and only512/1024 plus decode graphs for513;
this is correctness evidence with a changed memory budget, not a timing control.

The GDN-only real-row fix passed `padded-shadow5`:129 checks/rank, max_abs0 on
both ranks. `padded-full3` then passed all36 C1 continuations against the unpadded
oracle at512/513/1024/1536/2048/2051 tokens, plus C4/C8 mixed fallback transport.
FULL mean TTFT at513/2051 was270.67/552.86ms; same-run padded NONE407.81/857.93ms.
This qualifies the bounded non-speculative single-prefill mechanism only. MTP
composition and general mixed/full/prefix-cache coverage require their own evidence.

## MTP composition: distinguish padding from capture and prompt sensitivity

Native async/MTP2 service frontiers use64outputs,C1/C4/C8,two cohorts each,APCoff,
6GiBKV on the same local TP2 host (separate runs, not interleaved causal A/B).
Native sync22.03/59.67/91.53tok/s; async26.53/68.24/101.88; MTP2
40.76/81.56/116.02. Exact native sync→async decode graph bodies stay~34.7ms,
intergraph gaps~8.8→1.4ms. MTP2 profile originally underfilled:20outputs exhausted
before8warmup+4active model iterations. Use64output budget, still only4active steps;
empty exact-graph export from that old capsule is not evidence of absent graphs.

`full-mtp2-3` apparent55.96/92.63/117.81tok/s is **not a qualified combined speedup**:
its repeated-English synthetic512prompt produced a repetitive continuation, changing
speculative acceptance. `spec-shadow1` proves target512prefill plus first3-token
verification FULL/NONE validhidden/allcache max_abs0 under identical paddedmetadata.
`spec-policy1` then shows512→513 padding changes the output even with FULL disabled;
exact513 does not. Thus this discrepancy is not specific to graph replay.

`padding-kernel2` isolates actualTP2 q/k8,v24,K/V128. Nativeconv validoutput/cache
are exact for512 vs513/576/1024 with initialstate false/true. Chunk validoutputs
are exact; finalFP32state has small finite delta~.0032/.0020, equal across padded
sizes. Do not assert a special513partialchunk bug or repair it with untested alignment.
`logical-state1` wholemodel cold512 comparison shows differences already inlayer0
conv cache before recurrence, amplified acrosslayers; FULL and paddedNONE stats
agree. A physicalshape-dependent arithmetic perturbation is plausible, not proved
as the only cause. `spec-policy2`192-alignment(576/1152) yields coherent but different
continuations in both paddedNONE/FULL, not exact native recovery.

`chat-policy1` uses4authored nonrepetitive chat prompts84/94/99/101tokens,128outputs,
native/FULL/FULL/native, MTP2 with native3-token bucket alignment. All16requests
complete and inspected continuations are coherent; some texts differ, no observed
synthetic repeated-prompt loop. This bounded smoke evidence is not a quality
benchmark and does not establish population-wide MTP speedup. Do not demand
bitwise equality where native shape/batching is already numerical, but don't count
changed speculative acceptance as a graph kernel improvement either.

## Independent package integration

`betterscale.qwen_worker.Worker` is the bounded **nonSpec/APC-off** opt-in entry,
separate from DSV4 and from the older owned-Qwen reactor. Patch README owns its
native command and admission. It strips prototype RPC/profiling/source-path imports;
MTP composition remains experimental. NPU acceptance uses a frozen package snapshot
through `package_probe.py`, then a fresh wheel/sdist build verifies package delivery.
Do not call the source addition a PyPI release.

Editable vLLM source may not live beneath distribution.locate_file(). Qwen's own
pin check validates find_spec(package).origin roots, actual imported files and
versions. Initial `package1` correctly failed closed before model load on missing
legacy distribution-relative files; `package2` passes the corrected independent gate,12 C1 continuations matching
`padded-full3` plus C4/C8 mixed cohorts. It is actual entry acceptance, not a
fresh claim against an independently measured native baseline.
Do not silently weaken content pins or modify the DSV4 Worker to fix this route.

The existing CPU suite needs the pinned Ascend submodule populated. A new parent
worktree initially has empty submodule directories; use a detached worktree at the
pinned donor commit in its upstream/vllm-ascend slot. Mere directory existence does
not mean the fixture exists.69tests pass after supplying that unmodified fixture.

### Immutable convolution layout: measured small fish

Ascend patches its methods onto donor `QwenGatedDeltaNetAttention`; instances are
NOT `AscendGatedDeltaNetAttention`. `conv-mtp2-1` selectedzero and failed before
measurements. Corrected `conv-mtp2-2` pre-packs48target weights before compile:
logicalshape/values/Parameter identity stay fixed; view(C,4).T becomes contiguous.
70CPU tests include value/identity/stride and repeated-packing checks.

TraceLoom's native raw export gives4exact targetgraphs and4combined-draftgraphs
perrank. Repairing only the previously underfilled nativeMTP2 short profile gives:
rank0 target36.253→35.367ms; rank1 36.268→35.445ms.48weightTranspose[5120,4;2]
per targetbody (~.848/.879ms summed) disappear on bothranks. This supports a real
~.8-.9ms/targetstep saving, not an assertion that all E2E variance is explained.
Separate-run C1/C4/C8 rates40.78/81.57/116.02→42.36/90.10/121.16tok/s; C1texts
match. Do not advertise the C4~10% as a controlled causal result. Helpers
summarize_graphs.py and root conv-layout-profile-comparison.json retain exact
operator/shape identity; each rank timeline has its own clock.

### Native AOT cache can mix multimodal and text-only signatures

`package-mtp1` passes pins/config/load but fails at draft startup with
`AttributeError: 'NoneType' object has no attribute 'size'` in cached AOT call_size.
Native MTP experiments had multimodal support enabled with text requests; package
entry explicitly sets image/video limits0. Both use the same backend hash
0e19c40fdc; failing log directly loads eagle AOT model5732557447b9913f... compiled
for the earlier tensor-valued inputs_embeds, while text-only dummy passesNone.
This supports a signature/cache-key collision, not a weight-layout arithmetic bug.
Do NOT clear shared compiler caches or alter installed donor files. `package-mtp2`
uses a fresh capsule-local VLLM_CACHE_ROOT to test that explanation. Fresh-cache `package-mtp2` PASSES C1/C4/C8(two cohorts each), C1texts match
`conv-mtp2-2`; this supports the cache-collision diagnosis. Retain the failed
capsule, document dedicated text-only cache use, and do not claim an upstream fix.
Package C1/C4/C8 rates41.98/84.19/114.21 vary versus the earlier prototype; this
reinforces reporting exact graph cost removal rather than a stable concurrency%.

## Direct no-MTP native comparison (September17)

`no-mtp-abba2`, source3d138c2, uses local4/5 in one admitted window, separate
native/candidate/candidate/native servers. Both noMTP, async, text-only, APCoff,
6GiBKV,8seats,2048token budget, same synthetic English input IDs and64outputs.
Native entry is unmodified NPUWorker/default graph policy; candidate is the shipped
Qwen Worker/FULL policy. Startup, compilation and warmup are excluded. Four measured
cohorts/case/arm; C1lengths512/1024/2048, C4/C8 mixed512/2048/1024/1536.
Initial0/1 watcher `no-mtp-abba1` was cancelled without launch due foreign occupancy;
no historical0/1 timings were mixed into this comparison.

Pooled output tok/s native→candidate:
- C1/512:26.64→29.25 (+9.80%), TTFT369.18→184.34ms.
- C1/1024:26.71→28.01 (+4.87%), TTFT373.85→278.03ms.
- C1/2048:25.37→25.70 (+1.30%), TTFT486.01→471.74ms.
- C4mixed:66.56→66.30 (-.39%); native roundmeans69.14/64.17, candidate66.48/66.13.
- C8mixed:101.09→99.98 (-1.10%); native101.00/101.19,candidate99.59/100.38.
All C1 continuation text sets match. No confidence intervals or population-wide
accuracy claim; C4 is within observed run spread, not a proven throughput regression.
Short TTFT improvement is NOT a50%end-to-end throughput gain. Summarizer and complete
round receipts preserve TPOT=(HTTPcompletion-firstcontent)/63 and workload identity.

Fletcher asks whether concurrent differences are scheduling. Six-step C4 profile
probes record actual scheduled tokens/request IDs/computed counts and native graph
mode; don'tinfer from requested configuration alone. Observation hook must attach
in load_model AFTER NPUWorker.init_device constructs the runner, not Worker.__init__.
`no-mtp-concurrent-candidate1` failed beforeload on that diagnostic-only mistake;
its queued native counterpart was cancelled. Corrected native2/candidate2 results follow below.
No serving implementation change belongs to these diagnostic probes.

The corrected `no-mtp-concurrent-native2` / `candidate2` both PASS. Six-step C4
observations have different first-arrival order; do not compare whole profile
makespans as a matched timing A/B. Actual dispatch:
- Native:512single NONE;1decode FULL;[1,2047] NONE;[1,1,1024,1022] NONE;
  [1,1,1,514] NONE;4decode FULL.
- Candidate:2048single FULL;[1,512] NONE;[1,1,1024,1022] NONE;
  [1,1,1,514] NONE;4decode FULL;4decode FULL.
Thus native mixed is also NONE, not a PIECEWISE path uniquely lost by candidate.
The main missed opportunity is FULL coverage of real scheduler-produced mixed
batches. In this candidate trace the short512prefill joins a decode token and
misses the large standalone FULL benefit; the only captured prefill is the long
2048case where device compute already masks submission overhead.

Both TraceLoom raw exports retain TASK/compute/comm/API evidence. Exact graph
reconstruction can be empty with onlytwo same-shape decode repeats; don't recapture
just for that. Generalized compare_full_prefill.inspect checks6GemmaRmsNorm starts,
6ArgMaxV2 samples,304totalMatMulV2/V3 +16FIA +128comm per modelbody. Initial norm
inputshapes match all six scheduled token counts; API timestamps fall within the
recorded host profile window. It bounds bodies at finalAddRmsNormBias, before sampling.

Candidate mixed513:body451.33ms;rank1compute119.99ms,comm41.09ms,uncovered290.25ms;
~23k CANNcalls begin during the body. Mixed517 native/candidate body447.50/456.84ms;
rank1uncovered285.14/293.28ms,~23.6k CANNcalls. Rank0communication268.97/284.89ms
is consistent with waiting on the starved peer. Uncovered is not pureidle (may
include memory/control); API profiling overhead is not an E2E speedup measurement.
Both still exhibit the host-submission bottleneck in short mixed batches.
Same-width4decode FULL bodies: native36.78/36.80ms(rank0/1), candidate~37.15ms;
only1vs2samples, not enough to attribute the whole C8-1.1% result causally.

Engineering direction: extend the FULL metadata/capture contract to real mixed
prefill/decode batches rather than first altering queue policy or forcibly splitting
continuous batches into single requests. This is a coverage diagnosis, NOT shipped
mixedFULL support or proof that the native scheduler policy is inefficient.

## Exact mixed FULL capture: correctness and causal small-shape timing

The next pilot uses the native GDN mixed split (decode prefix, prefill tail),
stable metadata tensors and invariant host chunk lists per exact partition.
It does not change queue policy, pad tokens, enable MTP/APC, or modify the package.
Dummy capture mutates the shared scheduled-length array before query offsets are
built. Its native max_query_len is conservative even for all-one dummy batches;
classify decode from actual lengths, not that scalar. `mixed-full1` captured513
but failed on a later small decode graph for this diagnostic mistake; corrected
source d678670 passed `mixed-full2` on local4/5.

`mixed-full2`: two actual [1,512] mixed steps, both ranks,129checks/step (valid
hidden plus128cache tensors), every max_abs0. `mixed-full4req1`, source4fa80d1:
same proof for [1,1,1,514], first fresh prefill then a2559prompt whose2045chunk
leaves514tokens. All129checks/rank/step max_abs0. Has-initial-state changes from
[true,true,true,false] to alltrue and live state slots change, so replay is not
merely comparing the capture's dummy inputs or one fixed state assignment.

`mixed-full4req-timing1`, source1a0ccdb, same local4/5,1GiBKV, no shadows:
warmup both modes then NONE/FULL/FULL/NONE twice. Three ongoing512prompt/96output
requests are already decoding when514prompt/4output joins. Exactly one matching
mixed dispatch per cohort verified on each rank. Four unprofiled TTFTs/mode:
NONE466.457/443.506/460.933/460.479ms; FULL246.580/248.021/248.582/248.919ms.
Means457.844→248.026ms (-45.83%). This is joining-request latency under this staged
load, not a45.83%general C4 throughput improvement or the earlier6GiB service A/B.
Full/NONE use identical metadata and unmodified native scheduling in one process.

Separate3step profiles/mode, exported with native_graph_export.py --label:
both ranks have one517mixed body then two4decode bodies. TraceLoom inspection
guards304MatMulV2/V3,16FIA,128communications/body. Mixedbody464.18→168.28ms;
compute union stays~121.5→123.3ms. NONE rank0uncovered302.79ms/rank1comm341.34ms
becomes FULL~4.4msuncovered/~40.5mscomm. APIs started inside mixedbody fall from
23440/22827 to458/298. This supports removing host submission starvation/peer
waiting, not faster compute kernels. Profiling overhead is excluded from timings;
uncovered still is not a proof of pureidle. Subsequent decode bodies remain~37ms.
comparison.json, per-rank step-costs and readable Perfetto timelines remain in
that capsule under native-graph-{none,full}/traceloom.

`mixed-full4req-two-prefill2`, source3aa1494, proves [1,1,1024,1022] FULL as well:
two steps/rank,129checks/step, max_abs0 throughout. Native scheduling then emits
[1,1,1,514] and decode continuations. Earlier two-prefill1 completed HTTP requests
but never hit the exact partition (remaining shadow budget2); its SHADOW_MISMATCH
label means missing coverage, not a numerical failure. Simultaneous HTTP client
threads do not guarantee same-step admission. The corrected correctness harness
briefly holds worker RPC500ms, queues1024then1536prompts, and lets native scheduling
resume. This explicit arrival-staging hold is NOT used in the successful timing
experiment. Runtime shape counts now make missing-witness diagnostics explicit.

Integration boundary: this pilot captures only one exact partition/process.
Native FIA GraphParams keys by total token count; GDN host chunk lists depend on
the full partition. A2048single and [1,1,1024,1022] must not share handles/buffers.
Supporting both needs deliberate graph-key ownership and bounded capture memory,
or a proved padded partition contract; simply removing the mixed fallback is unsafe.

### Multiple exact partitions now coexist (September17,06:46UTC)

`ARM=partition-full` enables the separate MIXED_COEXIST pilot. PartitionDescriptor
extends native BatchDescriptor with the complete scheduled-length tuple. Capture
initialization installs those keys, and _warmup_and_capture carries the chosen
partition into dummy metadata. GDN stable buffers/scalar contracts are keyed by
the same tuple. Native decode remains on its original descriptors/resources.

FIA's native GraphParams remains internally indexed by token total, but each
partition owns a distinct bank. A scoped bank selection surrounds the entire
host _model_forward, including attention parameter updates and native ACL replay;
finally restores the original bank. It adds no device synchronization. This
adapts the pinned serial model-submission worker only: global native-bank swapping
is NOT qualified for concurrent model threads/multiple runners, MTP, DP, PCP,
LoRA or general graph eviction. Do not silently reuse it in those contexts.

`partition-full1`, source8c198c4, local4/5, TP2/noMTP/APCoff/1GiBKV, passes six
alternations across [2048], [1,1,1024,1022], [1,512], [1,1,1,514]. Followup
`partition-full2`, source960bc93, adds [1,1,1022,1024] to distinguish partitions
with identical total AND request count. Eight alternating actual HTTP shadows,
both ranks:2064total checks (16x129), every max_abs0. All five banks remain
distinct and each retains exactly16FIA handles/16events across reuse. Native
startup reports12s capture/.76GiB total graph memory (five partition graphs plus
four native decode graphs); not incremental memory versus a controlled baseline.
Short decode continuations complete after each tested prefill/mixed batch.

The HTTP staging uses the earlier correctness-only500ms worker hold for joins;
there is no new timing/profile claim in these capsules. Three CPU tests cover
same-total/same-request-count key and bank separation, exception restoration,
and unchanged native decode resource selection. This closes bounded coexistence,
not arbitrary-length/padded partition support or package integration. Production
Qwen/DSV4 paths and installed donor sources remain unchanged.

### End-to-end partition candidate on hw3 (September17)

Fletcher reassigned local cards to an eight-card experiment during this task;
all subsequent TP2 work moved to hw3. Local `partition-abba1` is FAIL, not a
performance result: natural concurrency exposed a one-token prefill tail with
the same length tuple as a decode-prefix graph. GDN correctly classified it as
prefill and the pilot asserted. Dispatch now additionally checks computed>=prompt
for every supposed decode-prefix request; role mismatches use native NONE and
do not bind the fixed mixed metadata. Lengths alone do not identify request role.

hw3 root `/workspace/my-ascend-workspace/runs/qwen27-partition-serving`:
- `model/` is the same local27B checkpoint, copied because the discovered hw3
  Qwen3.8 model was Flash-Next, NOT the requested27B.18safetensor shards; rsync
  final size/mtime dry comparison empty. Do not repeat broad content grep over
  old accuracy JSON: those single-line files contain huge token arrays.
- `runtime-source/` contains copied local pinned vllm and vllm_ascend import
  trees, not edits to an installed hw3 donor. Python is the existing
  `/workspace/my-ascend-workspace/runs/liveinfer-online/20260907-donor-dspark-runtime/env/bin/python`.
  Preflight confirms actual import roots and torch2.10.0+cpu,torch-npu2.10.0.post2,
  transformers5.14.1. `partition-abba2/launch.sh` preserves CANN PYTHONPATH and
  owns explicit localhost ports32181/32182,HCCL29664–29727,devices4/5 admission.
  Copied admission helpers require idle_gate.py as well as supervise/probe_host_npus.
- Local downloaded evidence is under the usual root with `hw3-` capsule prefixes.

`partition-abba2`, source3cd25d1, PASS: baseline/candidate/candidate/baseline in
one admitted hw3 pair4/5 window. Same noMTP,async,text-only,APCoff,6GiBKV,8seats,
2048budget,64output workload as the earlier comparison; no staging holds or
shadows. Candidate is the prototype mixed Worker, not the packaged Qwen Worker.
Its coexistence set now also includes exact single512/1024/1536, restoring those
ordinary prefill graphs alongside2048 and four mixed partitions (eight total).
Unknown lengths/partitions and short-prefill role mismatches retain native fallback.

Pooled output tok/s native→candidate (four cohorts/case/arm):
- C1/512:26.31→29.25 (+11.19%); TTFT423.30→181.56ms.
- C1/1024:26.52→28.07 (+5.84%); TTFT396.75→266.98ms.
- C1/2048:25.69→25.66 (-.12%); TTFT468.87→468.05ms.
- C4mixed:65.08→69.84 (+7.30%); TTFT1135.90→934.61ms.
- C8mixed:96.01→105.20 (+9.57%); TTFT1882.64→1568.62ms.
C4round rates64.16/66.03 versus72.83/67.08; C8round rates95.79/96.24 versus
104.64/105.78. C1text sets match. Limited synthetic cohorts, not population-wide
or SWE accuracy evidence; don't subtract old local-host results to isolate a
causal gain attributable solely to mixed capture. No new package/PyPI shipment.

Short profiles are separate from all timings: `partition-profile-baseline` and
`partition-profile-candidate2` PASS. Initial candidate profile inherited two
sample step hooks and stopped after three forwards; fixed observer2f440fd advances
exactly once whether the base Worker already owns a hook or not. Keep that initial
artifact but use candidate2 for the requested six-step comparison.
Actual candidate2 schedule:512single FULL;[1,2047] NONE;[1,1,1536,510] NONE;
[1,1,1,514] FULL;4decode FULL twice. Baseline:1024single NONE;[1,512] NONE;
[1,1,2046] NONE;[1,1,2,1536] NONE;4decode FULL twice. Arrival order differs;
whole-profile makespans are not a matched performance A/B.

TraceLoom raw exports pass6starts/6samples,304MatMulV2/V3,16FIA,128comm perbody
on both ranks; initial norm shapes exactly match dispatch token totals. Candidate
512body162.53ms and517mixedbody168.11/168.09ms,~4.4msuncovered,~40mscomm,
~400–460APIs started/body. Its uncovered2048mixed partitions still run NONE with
~23kAPIs and486/490msbodies. Both arms'4decode bodies remain~37ms. This supports
effective FULL coverage where implemented, not full coverage of natural batching.

Four readable Perfetto timelines plus schedules, step-costs and unprofiled summary
are packaged locally as `hw3-partition-timelines.tar.gz` in the evidence root
(~9.4MB). Directory `hw3-partition-timelines/` contains baseline/candidate-rank0/1
files. Per-rank clocks remain independent. All hw3 probes reclaimed/port32181
closed; no local NPU job was launched after Fletcher's reassignment.

## Dynamic packed GDN operator feasibility (September17)

Fletcher chose operator-first feasibility, NOT further exact-partition enumeration.
`dynamic_gdn_probe.py`, source863dc20, `hw3/dynamic-gdn3` PASS on one admitted
910B2/card6 with TP2-local qk8/v24/KVdim128. No model load or service modification.
One graph captured at capacity512tokens/4requests plus an empty sentinel replays
[512], [1,511], [1,1,256,254], [129,63,1], [64,64,64,64], [1,1,1,1], then[512]
with changed state slots. Stable device cu_seqlens/chunk-index/chunk-offset buffers
control existing Triton cumsum/KKT/solve/WY/H/O stages. Actual request count and
total tokens change without recapture; unused chunk tasks target the empty sentinel.
State gather/masked cold start/writeback are inside the graph. Metadata preparation
is host-side before replay, not a claim of device-generated scheduling or H2D overlap.

Native chunk pipeline (AscendC H/O) is the oracle on exact active shapes. All28
output/full-state-bank/second-pass output/state checks pass atol=.01,rtol=.01;
max output error .0004883, max state error .003380, all finite. Cold cases reset
on both passes; continuing cases consume the first final state. This is bounded
random-input recurrence evidence, not long model accuracy, mixed cold/continuing
flags in one batch, convolution/FIA integration, or arbitrary capacity support.
Do not call it bitwise parity or full-service mixed support.

20-replay event timing: dynamic .807–.929ms versus fixed native chunk graph
.564–.744ms; +.186–.268ms per invocation. Dynamic includes state gather/writeback,
native control excludes that glue. Sequential short timings, no service speedup
claim; all-one control is chunk, NOT optimized recurrent decode. Dynamic ability
is demonstrated but replacement compute is not faster. Card6 reclaimed.
Local receipt: evidence-root/hw3-dynamic-gdn3/receipt.json; remote full capsule
under the existing qwen27-partition-serving root. dynamic-gdn1 cancelled before
launch; dynamic-gdn2 captured successfully but its oracle control illegally copied
CPU cu to NPU inside capture (107030). Fixed by moving that copy outside capture.

Native boundary insight: csrc/moe/chunk_gated_delta_rule_fwd_h/op_host/op_api
converts aclIntArray metadata into tensors before the AICore launch. Tiling derives
request capacity from tensor shape, not boundary values. arch22 block scheduler
reads device cu, removes empty rows, builds chunk counts at runtime. BUT it also
uses actual totalTokens/totalChunks as head strides; fixed physical capacity with
shorter logical total cannot blindly reuse a tensor-only wrapper. Empty internal
rows also imply compact state indexing. Inspect arch20/current device implementation
before applying this observation. A tensor ABI plus explicit capacity/stride contract
is a promising native-kernel route, not yet tested. Existing recurrent operator's
MAX_MTP=16 blocks simply feeding long prefills to that unchanged implementation.

AscendC followup source audit: `prototypes/qwen38-serving/ASCENDC-GDN.md`
records the narrow H/O fork contract and performance hypotheses. Important:
910B selects arch22; arch20 is __CCE_AICORE__==200 compatibility. H arch22 AIV
initial-state loop traverses all heads/requests without core partitioning;
changing it requires preserving producer/consumer readiness, not a naive striped
copy with existing per-pair signals. O already uses physical token stride but
logical chunk stride. No compiled fork or measured optimization follows from this
audit. Avoid attributing the Triton-vs-native whole-pipeline delta to one kernel.

### Owned AscendC fork qualified at operator scope

`prototypes/qwen38-serving/ascendc_gdn/README.md` now owns the build/replay recipe
and bounded observations. Only pinned H/O arch22 plus common Catlass block helper
are copied; Catlass41bf90da is an external pinned dependency. Independent CANN
raw kernel library avoids native schema/GE/installed-runtime modifications.
Fresh Release build directory is important: mixing empty/Release configuration
caused duplicate host objects in CANN's legacy linker.

hw3 `ascendc-gdn-fixed1` passes unchanged owned kernel output/state/H/Vnew max0.
`ascendc-gdn-dynamic1` (build2/2d6cb4b) and `ascendc-gdn-owned-init1`
(build3/9e49edc) each pass all28 changing-partition/full-state/continuation checks
max0. The latter restricts initial-state copies to their consuming core pair,
retaining both AIV subblock signals/copies; no global barrier or algorithm change.
Host CPU mapping test passes1–64requests, hardware evidence remains4requestcapacity.

Matched H/O stage ABBA probe d8bae04 reports H .109/.134/.160ms native versus
.078/.086/.065ms owned at [512]/[1,511]/[1,1,256,254]; O .074–.083 versus
.060–.062ms. Graph-stage savings include wrapper differences, not purekernel
or end-to-end savings. Whole dynamic pipeline still costs .871–.934ms versus
fixed native .587–.773ms with unequal capacity/state glue. Runtime is a ctypes
prototype requiring TASK_QUEUE_ENABLE=0, stable resource lifetimes and packed
positive request prefix/empty suffix; do not install it into asynchronous Torch
submission unchanged. No fullmodel/mixed-role/MTP/PCP/service qualification.

### K-V pool and actual-role non-regression gate

Fletcher rejected keeping V-K compatibility as a fixed constraint. Current owned
prototype uses one K-V FP32 pool for chunk H and a K-V-aware single-token Triton
decode adapted from pinned vLLM FLA. Direct H state mode requires owned-pair
initialization ON. Native V-K decode MUST NOT consume this pool. Temporal copy
specs are whole-row opaque; convolution cache is separate. Shape equality128x128
is not layout compatibility. Production allocation/routing remains unchanged.

`ascendc_gdn/README.md` owns the precise contract and evidence. hw3 pool4/7a01f9c
with build4/f6f1bed passes50 checks (32native-chunk/16actual-role/2decode-policy),
including poisoned-NaN cold slots and continuation. Native-chunk max0; mixed
native recurrent comparisons differ at most .000244 output/.003468 state.
Complete core-GDN ABBA graph timings include capacity, transformations and state
handling:512prefill .761→.692ms; [1,1,256,254] mixed .853→.674ms; four cold64
prefills .944→.652ms; [1,127,63,321] mixed .997→.692ms; four decode .03046→.02996ms.
All eight tested rows are no slower. DecodeC1 independently passes, owned~15us
versus steady native~17us (native first timing~59us retained as an outlier).
Common normalization, convolution/projections, host metadata publication and
full-service effects are outside these timings. Metadata is one536-byte pinned
slab perwave; asynchronous slab reuse/overlap is not proved.

Avoid paid dead ends: build5/pool2's V-K UB conversion failed numerical checks
and was removed, not shipped.128-wide recurrent blocks repeatedly exceeded192KiB
UB; accepted decode uses two64-wide V tiles in a single head program. Earlier
address-only/one-program-per-Vtile variants were correct but slower than native.
Do not substitute those or claim an end-to-end service win from the microprobe.


### Owned K-V mixed FULL service (September17 integration)

Source ddc8a3d adds opt-in `betterscale.qwen_worker.MixedWorker`, separate from old
Qwen/DSV4 Workers. Package `patches/qwen_gdn/README.md` owns its deployment contract:
TP2, eight seats,2048 budget, no MTP/APC, queue0 raw ACL launch, fresh K-V state
pool, token-capacity graphs, same owned core for uncaptured execution. Native FIA
updates remain; metadata uses blocking pageable copies, not a new overlap claim.
Engine scratch/POD construction must occur before capture, not on first core call.
Conv-weight packing validates the explicit owned consumer, not the native function.

Critical paid failure: build4 was not safe beyond its four-request evidence.
Five/eight mixed warm/cold rows reuse ping/pong H UB while a previous MTE3 store
is still reading it. MTE3_MTE2 does not fence vector Duplicate. In core4, erroneous
initial H head positions exactly match the ownership-loop reuse order. Source
4e21bb1 adds MTE3_V before the cold fill; no global barrier or bank copy. Separate
initial-bank snapshot (core3) still failed and was rejected. Core5/build6 passes
12 graph/NONE full-core cases plus an independent warm-seed/cold-zero initial-H
invariant. Do not trust FULL/NONE parity alone: both modes can share this bug.
Native convolution isolation passed. Never deploy old build4 to eight-seat service.

Service9 (hw3 cards6/7,1GiB diagnostic KV,32 outputs) passes ten single-request
lengths1/7/17/129/512/513/1024/1536/2048/2051 and C4/C8 cohorts.44 rank-step shadows,
5,676 valid-hidden/full-cache checks all max0. Actual mixed rows include
[1,1,512,513,17] and [1,1,1,1,18]. Eight active rows are covered by core5, not inferred
from submitted C8 concurrency. Service5/6/8 failed with the old library; service7
was a diagnostic wrapper keyword error. Service9 uses production execution without
layer-debug capture copies. No language-quality or unlimited-context claim.
Artifacts under established remote qwen27-partition-serving root and local
qwen38-tp2-serving/hw3-elastic-* mirrors. Binary manifest in package pins build6
(kernel4e21bb1); the separately built binary is not bundled in Git or PyPI0.4.2.


Whole-service AB `elastic-ab1`, same hw3 cards6/7,6GiB KV,64 outputs, two warmed
cohorts/case, native queue1 versus owned queue0: pooled tok/s native→owned
C1/51226.655→28.783; C1/102426.600→27.411; C1/204825.686→25.179;
C464.513→70.546; C896.149→104.167. C4/C8 TTFT1104→895/1889→1590ms.
Single-request output gaps worsen about0.7–1ms, so NOT universal non-regression.
Do not confuse this new elastic service with partition-abba2's exact enumeration.
No new profile or statistical confidence claimed; all round metrics retained in
`docs/evidence/qwen-mixed-full.json`. Cards released. CPU73 tests and fresh
sdist-to-wheel archive contents pass; no PyPI publication/version change.

### SWE whole-session crossover and remaining metadata gap (September17)

Enter `prototypes/qwen38-serving/SWE-TRACE-COMPARISON.zh-CN.md` for the bounded
real-trace HTTP comparison and paid profile findings. `swe-elastic2`/b05739f uses
hw3 pairs2/3 and6/7 simultaneously, then swaps arms. Eight whole <=8K
mini-SWE-agent/Qwen3.8 trajectories (78 calls,20,648 outputs/cohort) selected from
15,525 rows, not a representative full-dataset claim. APC/MTP off. C4 andC8 both
replay all8 sessions with at most4/8 active; two repeats perarm/case. Native→owned
pooled output70.316→71.196 (+1.25%) and97.556→99.860 (+2.36%). Mean TTFT
1363→1239/1526→1395ms. This supersedes using synthetic short-prompt gains for SWE.

Fixture `swe-qwen27-v5/trace.json` is retained in the evidence root. Reuse it rather
than rescanning. Old OpenHands trajectories exceed8K; whole-history rejection is
not grounds to truncate or silently relax Worker admission. Qwen template needs
JSON tool arguments as mappings; isolated CPU preparation needs pyarrow/jinja2.
All tool content is inert. Fixed recorded suffix-token budgets, original history,
zero tool delay; not task solving or model quality. Summarizer checks every call.

Both short profiles have identical6-step schedule:decode1 twice, [1,1472]mixed,
decode2 three times. Native mixed NONE body507ms→candidateFULL346ms; body APIs
~23.6k→310. TraceLoom/native export passes per-body compute/communication guards.
Three decode2 bodies/rank reconstruct exact_direct; unique mixed not reconstructed
is not absent FULL. Per-rank clocks remain independent. Four exported timelines
and schedules/analysis are in `qwen-swe-traceloom-timelines.tar.gz` (~4.2MB).

Priority observation: candidate steady model-to-model gaps4.74–4.96ms vs native
1.52–1.63ms. Three blocking metadata H2Ds (~.14ms actual API sum) plus serialized
post-drain derivations; each metadata group repeats cu cast/copies and computed
Sub/Sub/Greater. First wait spans previous model: don't count its full duration
as new overhead. Profile magnifies host costs; no promise of3.2ms unprofiled gain.
Second hypothesis: solve16 takes10.52ms/48layers vs native3.01ms;1536capacity's
large-chunk table has9 tasks, only3 active,6 empty. Donor masks memory but still
executes fixed recurrence for emptyT. Investigate device-side skip, NOT partition
keys; native mixed also uses separate recurrent-prefix arithmetic, so timings are
not an isolated empty-task experiment. Production code was not changed here.

`swe-elastic1` abandoned after0/1 crossover startup free-memory rejection and
observed unlisted device1 HBM/compute activity. Its first wave is not headline.
Retain strict idle gates between server reloads, not just initial lease acquisition;
new wait_reclaimed CPU rejection smoke passed. Do not reduce memory guard to get
past uncertain occupancy. Final campaign released its cards. Fletcher now permits
local and hw3 cards5/6/7 for follow-up TP2 work; prefer those, still require fresh
subset admission and preserve foreign work. This permission does not reserve them.

### Alternating metadata banks (qualified, 2026-09-17)

`qwen_gdn/publication.py` gives each capacity two descriptor keys and separate
native FIA task-resource banks; partitions remain device metadata, never key
enumeration. All GDN groups share one pinned packed slab/H2D per wave. CPU
block-table column0 is authoritative only for `mamba_cache_mode=none`; initial
flags must reproduce **CPU seq_lens - real query lengths**, including synthetic
capture lengths, not blindly reuse request counters. No speculative decoding.
The uploaded event protects pinned-source reuse; consumed protects device-bank
overwrite. Compute waits uploaded on device; neither requires a per-wave host
compute synchronization. Initial allocation gets a one-time ingress wait on the
allocating stream. Native model inputs, sampling, D2H and KV retirement remain
native-owned: this is not a full LiveInference reactor transplant. Compare the
already earned protocol in `prototypes/owned-wave/fia-plan/ASYNC-TRANSPORT.zh-CN.md`.

hw3 `dualbank-core1`: triangle active rows match donor exactly on7 fixtures;
12 core graph/NONE cases retain exact output/conv/whole-state and independent
initial-H checks. `dualbank-solve2`: captured20-solve graphs, ABBA30replays each,
[1,1472] mean378.85→253.86us; all7 active-output comparisons exact. This is the
whole solve including merge, not an isolated16x16 kernel or serving speedup.
Eager-loop timing was host-limited/noisy and is not the performance claim.

`dualbank-service2`:26 captures,5.25GiB graph memory/rank;10single prompt lengths
1..2051 and C4/C8 mixed cohorts;22shadow steps/rank,5676checks max_abs0. Both
banks observed, every GDN host/device field independently compared during
shadow. Diagnostic1GiB KV, not performance. `dualbank-service1` had captured
successfully but probe RPC returned404: diagnostic `/collective_rpc` needs
`VLLM_SERVER_DEV_MODE=1`; elastic_probe now sets it only for shadow subprocesses.
Do not treat that harness failure as a model or graph failure.

`dualbank-swe1`, source7115858, same hw3 cards6/7 sequential native/candidate/
candidate/native, same full8-session78-call fixture and6GiB KV: pooled C4
70.021→75.092tok/s (+7.24%), C8 96.774→104.409 (+7.89%). Two repeats, no
population/quality claim. Native queue1 versus candidate queue0 remains part of
service configuration. Both arms'6-step profiles have identical schedules.
TraceLoom rank0 steady gaps: candidate1.496/1.487/1.478ms, native1.478/1.535/
1.554ms; historical candidate4.955/4.793/4.742ms. Mixed solve16 kernel sum
10.522→4.604ms against historical candidate; don't attribute the entire native
NONE→FULL343ms mixed body change to this incremental patch. Async stream40 has
6copies, subsequent5 wholly inside previous body envelope,2.24–2.76us each;
metadata-stream ownership is source-informed inference, not provider buffer IDs.
`DUALBANK-RESULTS.zh-CN.md` and `docs/evidence/qwen-dualbank.json` retain results.
Archive `runs/qwen38-tp2-serving/qwen-dualbank-traceloom-timelines.tar.gz` contains
four pristine TraceLoom exports. All owned jobs exited0/cards reclaimed.

`swe_compare.py` supports one physical pair as sequential ABBA as well as the
original two-pair crossover. Keep the phase files cleared between sequential
arms, and require fresh idle before every reload. One-pair whole campaign uses
about50minutes including4model loads; admit with sufficient bounded runtime.
`elastic_probe.py` owns enabling localhost diagnostic RPC only in shadow mode.


### Framework-owned GDN scratch and continuous replay analysis

Double graph count is not double persistent state. The5.25GiB capture delta above
included27 engines per bank (9mixed capacities ×3GDN groups), each eagerly
retaining64MiB workspace plus H/V/output. Manual cross-bank sharing was only a
rejected prototype (`shared-scratch-service1`, interrupted before qualification).
Fletcher chose native operator allocation through the capture pool instead.

`qwen_gdn/host.cpp` wraps the unchanged build6 H/O launches. Workspace size is
computed in C++ (~22MiB including the reserved16MiB prefix); temporary H/V/output
and workspace use invocation-local framework allocations, captured by the native
shared graph pool. The double-bank metadata and persistent K-V state stay owned
as before. The final mod requires both qualified library paths; no silent
state-pool fallback to resident scratch. Build helper is colocated with mod source.

`graph-scratch-core2`:12 exact graph/NONE plus independent initial-H cases.
`graph-scratch-service1`:26graphs,22shadowsteps/rank,5676comparisons max_abs0;
10single lengths and C4/C8, exit0/reclaimed. Same1GiB diagnostic KV capture delta
**5.25→0.88GiB/rank**, not an isolated graph-descriptor statistic. Core1's H oracle
read the wrong invocation after eager replaced its handle; retain captured H
explicitly before checking graph initial state. Output/state were already exact,
but core1 is not a passed qualification. Core harness now fails its process on a
failed receipt, instead of printing a large failed result with exit0.

For candidate-only SWE regressions, `SWE_CANDIDATE_ONLY=1 SWE_PROFILE=0` runs two
same-pair candidate cohorts with the existing immutable trace and no new profiler
or native server. Preserve retained controls as historical controls, not fresh
paired evidence. Keep fresh-idle admission before each model reload.

TraceLoom f1ccc85 (and older37323af) incorrectly grouped the6 dual-bank invocations
into one521.6ms legacy overlap envelope because no launch period repeats3times.
TraceLoom recovery e42dc84 (rebased on0d65fea), with lookup fixc2a6920, recovers6 exact single-graph replays /16,177 bodymembers per
candidate rank from the same full profiles, using explicit completion/capture
identity, not a model-step guess. Existing periodic compositions retain priority.
Native controls still have3 exact decode replays; don't invent missing coverage.
`qwen-dualbank-traceloom-continuous-timelines.tar.gz` contains four regenerated
Perfetto timelines and compact reports. NewderivedAugDBs live in
`dualbank-swe1/traceloom-continuous-qualified`; original captures remain unchanged.


`graph-scratch-swe1` (runtime e5460c0): two candidate-only same-pair hw3 6/7
cohorts,78calls/20,648 outputs each at C4/C8; no new profiler. Pooled75.049/104.440
tok/s versus retained candidate75.092/104.409 (−0.058%/+0.029%). Against retained
native70.021/96.774: +7.18%/+7.92%; not a fresh paired comparison. No speedup from
scratch reclamation is established; capacity gain without observed throughput
loss is the result. Both servers/admission exit0 and devices are reclaimed.
`summarize_swe.py` supports candidate-only receipts without manufacturing a
baseline; prior paired metrics remain byte-for-byte equal as Python values.
`docs/evidence/qwen-graph-pool.json` owns compact qualification and round metrics.
Latest TraceLoom export also composes upstream's common-plane graph internals;
all16,177 member geometries and9,790 replay repeat windows per candidate rank
are verified. Missing identity lookup indexes were fixed in the writers, so
new AugDBs need no manual index. Four exported files are in the archive above.

### GDN inter-projection fusion audit (September18; investigation, not shipped)

Use the existing `dualbank-swe1/traceloom-continuous-qualified/candidate-rank{0,1}.db`
exact launch/member surface, not a new NPU capture. `graph-scratch-swe1` did not
recapture a profile; the audit uses the older, arithmetically unchanged GDN path.
Raw aggregates are in evidence-root `gdn-fusion-audit/kernel-costs.json`.

`rearrange_mixed_qkv` in pinned core `qwen_gdn_linear_attn.py` splits convolution
output then flattens/cats Q/K/V into one contiguous buffer. Its comment expects a
compiled single-copy kernel, but the custom attention core actually launches
ConcatD. This is layout packing, NOT concatenation of requests/decode+prefill.
At T>1 each split's flatten can also materialize a copy. CPU checks at T1/2/17/1536
confirm view-only Q/K/V have identical values/shared input storage. For T>1 their
token stride is5120, versus packed1024/1024/3072. Existing norm/recurrence kernels
assume packed addressing: deleting the cat without adapting consumers is unsafe.

Observed kernel-duration sums across48GDN layers (NOT projected HTTP savings):
- Decode1: concat .788/.855ms, QKnorm .173/.172ms, gating .642/.700ms (rank0/1).
  Conv-start to output-projection-start spans sum2.896/3.051ms.
- Decode2: concat .754/.845ms, gating .623/.676ms.
- Mixed[1,1472] at capacity1536:480transposes sum10.424/10.355ms; concat .903/.974ms.
  KKT and WY each repack beta+cumulative-g, then H/O repacks g again. With B=1,
  [H,B,T] and [B,H,T] have identical contiguous storage order. Three duplicate
  gate-layout conversions/layer account for2.234/2.170ms by source/sequence mapping.
  The other transposes include Q/K/W/U packing and O's output conversion.

Promising first bounded prototype: packed-convolution-input-aware QK normalization
plus V packing (optionally gating), replacing cat+two norm launches while keeping
normalized Q/K and beta BF16 rounding boundaries. Decode can then consume packed
input/gating directly in its owned recurrence; don't merely flip existing inline
norm: it uses sqrt division/FP32 intermediates, unlike current rsqrt→BF16 storage.
Maintain exclusive state slots, empty sentinel and two-bank publication unchanged.
For mixed, first share head-major beta/g across KKT/WY/H/O; later let producers
emit consumers' layouts to remove W/U and output transposes. Output norm already
fuses its z gate, so don't count those as two unimplemented-fusion opportunities.
No fused kernel, NPU correctness or speedup has been qualified by this audit.

### GDN fusion implementation (September18)

`preprocess.py` now reads packed convolution output directly, emits BF16 normalized
Q/K, copies V, and computes FP32 g/BF16 beta in one Triton launch. It preserves
rsqrt and BF16 rounding rather than fusing normalization into recurrent FP32 math.
`chunk_wy.py` shares one head-major beta/g conversion across pinned KKT, WY and
owned H/O; neither recurrence, AscendC binary, workspace allocator nor publication
fences change. Only MixedWorker uses this path.

Paid lowering lesson: flattening token/head rows and using `%24` produced scalar
GM gathers in the generated `preprocess_kernel.ttadapter` (including 32 individual
gate loads), despite mathematically contiguous access. `gdn-fusion-preprocess2`
was exact but ~847us at T2 versus ~22us control: reject it. A runtime-branch/pointer
selection experiment also exited -11 during compilation; its root cause was not
isolated. The branch-free explicit token/head axes avoid that path. One token per
program recovered decode performance but regressed T2048 (151us versus93us).
Four-token tiles above T16 recover large-capacity efficiency; do not undo this
layout on the strength of source-level operation counts alone.

`gdn-fusion-preprocess6`: T1/2/4/8/16/129/512/1024/1536/2048, three changed-input
replays each, all five intermediates exactly match pinned donor. Captured ABBA
micro timings: T1 18.4->3.81us, T8 23.0->4.18us, T1536 72.6->33.0us,
T2048 85.9->41.6us. Not HTTP savings. `gdn-fusion-core1`: 12 mixed cases
(changing partitions/slots, NaN padding, independent initial H) and12 decode cases
(changing input/slots/inactive lanes), output and entire conv/GDN pools maxabs0
against the pre-fusion core frozen in the capsule. `gdn-shared-gates-core2` isolates
only shared layout: ~60-67us/layer (~5%) less core time at capacity1536.

The donor gating kernel assumes contiguous a/b. Its caller makes them contiguous;
passing raw stride48 views directly is an invalid oracle, not a fusion failure.
`gdn-fusion-strides1` records that rejected oracle. `gdn-fusion-strides2` compares
stride-aware fusion with donor fed `.contiguous()` at T1/8/16/129/1536/2048:
all changed-input intermediates exact. These receipts and frozen sources live in
the evidence root; keep oracle input contracts explicit when extending fusion.

`gdn-fusion-service1`: same26-graph TP2 full-model shadow envelope passes all5,676
comparisons (22steps/rank); server/admission exit0. Diagnostic capture delta is
0.84GiB/rank at1GiB KV. CPU77tests pass. See `docs/evidence/qwen-gdn-fusion.json`.

`gdn-fusion-swe1`, runtime d803579: two warmed same-pair candidate-only SWE rounds,
78calls/20,648 outputs each at C4/C8. Pooled78.527/108.429tok/s versus retained
scratch candidate75.049/104.440: +4.63%/+3.82%. Retained native70.021/96.774 gives
+12.15%/+12.04%; neither control rerun. Mean TPOT42.93/55.48ms (old45.00/57.95).
Services/admission exit0, cards6/7 reclaimed. Profile separate and only6steps/rank.
Native export of copied raw PROF inputs plus TraceLoom c2a6920 recovers6exact
bodies/14,593members per rank. Same dispatch sequence as dualbank-swe1:
[1],[1],[1,1472],[1,1],[1,1],[1,1], capacities1/1/1536/2/2/2, banks1/0/1/0/1/0.
Every replay contains48preprocess kernels and zero ConcatD/l2norm/gating kernels;
mixed transposes480->336. Rank0 GDN conv-to-outproj sums (selected same-shape
launches) decode1 2.89->1.45ms, decode2 3.66->2.01ms, mixed68.34->63.38ms.
Model gaps remain~1.3-1.5ms; don't relabel in-graph work reduction as host-gap work.
`gdn-fusion-swe1/traceloom` holds AugDBs, costs.py/costs.json, Perfetto and exact
member/structure verification. Use MatMulV2 **or V3** for the out-projection
boundary; mixed capacity1536 uses V3. Compact qualification and all HTTP metrics
are in `docs/evidence/qwen-gdn-fusion.json`.

### TP2 allreduce synchronization audit (September18; no runtime change)

Use `evidence-root/allreduce-audit/{analyze.py,summary.json,collectives.json,
one-layer-witness.json}` against the existing fused AugDBs before another profile.
Pinned NPUCommunicator inherits DeviceCommunicatorBase.all_reduce -> c10d
ProcessGroupHCCL; its PyHcclCommunicator helper is NOT the service path. Both
local/hw3 torch-npu version files identify2.10.0.post2/git8751b36d5d6959e499e6bf6530c1928060ced030.
Downloaded exact-source ProcessGroupHCCL: syncStreams255 (producer stream event
-> private HCCL stream), collective3821/3987 (entry fence/end event), Work
synchronizeInternal901 (consumer stream waits HCCL end). Normal Work.wait is not
necessarily a host device-wide synchronize; the barrier branch is separate.

Both ranks, graph-launch-1:129 allreduces (embedding+2/layer),5120 BF16 elements
=10KiB. Every op has4 SDMA +2WriteValue +2NotifyWait observations. Internal
COMMUNICATION_TASK_INFO distinguishes the two4-byte Reduce_Inline endpoint tasks
from two10KiB payload tasks; don't count four full payload transfers. Rank0 shows
local-copy/notify/wait/local-copy/notify/wait, rank1 wait/remote-inline-reduce/
notify/wait/remote-copy/notify. Exact notify IDs pair the two ranks independently
of clock alignment. This establishes protocol stages, not a proof any peer
handshake is redundant or an exact private executor class identity. Provider alg
label is MESH-RING-NHR. Mixed capacity1536 uses15MiB payload and14 inner tasks.

Stable decode1 rank0/rank1: entry-gap median11.90/11.78us, exit12.02/11.98us;
entry+exit sums3.223/3.207ms per body. Inner allreduce spans sum2.611/2.647ms.
Decode2 gaps sum~3.28ms; mixed~3.98/4.02ms but inner spans~113.75/113.36ms.
These are disjoint nearest-main-compute/first-last-member boundaries per rank,
not pure idle or guaranteed recoverable time. Raw TASK has CAPTURE_RECORD/WAIT
inside the gaps; one wait task may have multiple context observations, so don't
count each row as a distinct fence. No cross-rank clock offset was calibrated.
The symmetric~12us boundary floor supports investigating graph stream handoffs,
not attributing every gap to peer arrival skew. No host per-layer submission is
implied by these already-captured device controls.

First discriminating prototype should keep HCCL's algorithm fixed and submit its
AllReduce directly on the producer/consumer compute stream, comparing changed
input/generation correctness plus captured producer->collective->consumer time.
This could remove PG stream round trips by FIFO ordering, NOT by deleting peer
readiness/completion fences. Do not ship before graph/alternating-bank lifetime
qualification. AIV is a separate experiment afterward: workspace's relocated
`notes/archive/communication-stack-2026-07/hccl-rework2/aiv-a2-actual-path-results.md`
and `prototypes/hccl-aiv-a2-actual-path` already prove CANN9.0.0/A2 AIV graph
capability, not performance on this9.0.1 BF16 workload. Reuse that evidence rather
than claiming AIV is A3-only or equating an environment flag with actual dispatch.

### Same-stream allreduce isolation result (September18; not shipped)

`evidence-root/allreduce-stream1` retains source/probe.py, launch/admission,
receipt-rank{0,1}.json, analyze_profiles.py, analyze_gaps.py and six TraceLoom
AugDB/Perfetto pairs. hw3 cards6/7, CANN9.0.1, TASK_QUEUE_ENABLE=0, HCCL_BUFFSIZE256,
no HCCL_ALGO/OP_EXPANSION override. Three arms: ordinary PG, raw HCCL with explicit
compute/comm ready+done events, and raw HCCL on compute stream. Both raw arms use
ONE communicator, same in-place BF16 allreduce and producer/consumer Add kernels.
Each graph contains16 serial chains; two independent graph banks/arm/shape.
8 changed-input generations alternate banks; final y/z full-tensor checks pass
72/rank,144 total. This checks the final chain, not separately saved outputs from
all16 chains, and is not whole-model/service qualification.

Non-profiled NPU event time, us/chain (includes both Adds and replay amortization),
mean of two block medians; each block30 alternating-bank samples after4warmups;
order PG/split/same/same/split/PG:

| BF16 payload | PG rank0/1 | raw split rank0/1 | raw same rank0/1 |
| --- | --- | --- | --- |
| 10KiB | 30.177/30.659 | 29.833/30.354 | 26.714/26.942 |
| 20KiB | 33.099/33.501 | 32.708/33.248 | 29.073/29.526 |
| 15MiB | 913.773/914.380 | 913.099/913.648 | 909.683/910.035 |

Separate short10KiB profiles: two exact graph replays and32 collective ops per
arm/rank, all MESH-RING-NHR/count5120. Every collective still has4SDMA+2WriteValue+
2NotifyWait. Same-stream moves communication onto the compute stream and removes
all CAPTURE_RECORD/WAIT rows inside graph bodies; PG/split retain them. Peer
handshakes remain. Median profiled pre/post gaps: PG~22/12us, split~22/12us,
same~1.4/1.7us, inner span~19us for all. **Do not equate these profiled gap savings
with production savings:** second replay member span/16 on rank0 is55.59/58.24/
25.82us (PG/split/same), whereas non-profiled timing is~30/30/27us. First profiled
replays have additional startup outliers. Observation supports strong measurement
sensitivity of the cross-stream path; does not identify the profiler mechanism.

Accepted bounded result: same raw communicator saves~3.1-3.7us/chain at10/20KiB,
not the~24us previously visible in service profiles. Multiplying by129 suggests
~0.4-0.5ms/step, NOT a measured service win. No service path or communicator owner
changed. Before integration, qualify real model ordering/lifetimes and measure
unprofiled service; before claiming large idle recovery, control profiler effects.

Protocol prior art (September18; research, not a new hardware qualification):
- vLLM main csrc/custom_all_reduce.cuh `cross_device_reduce_1stage` directly
  reads registered peer input pointers and reduces locally; TP2 selects one-stage.
  custom_collective_common.cuh still implements both start and final peer barriers.
  Borrow peer-pull structure, not a claim that upstream already removes end waits.
- FlashInfer main include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh implements
  Lamport payload polling and three rotating buffers (LamportBufferLayout,
  LamportFlags); dirty-buffer clearing and per-CTA arrival bookkeeping remain.
  Sentinel checking/sanitization is representation-sensitive. This CUDA/NVLink
  implementation is protocol prior art, not an Ascend drop-in or permission to
  port volatile loads as sufficient NPU ordering. Downloaded research snapshot:
  evidence-root/allreduce-protocol-audit/trtllm_mnnvl_allreduce.cuh (unpinned main).
- CANN SHMEM official gitcode.com/cann/shmem documents device RMA, MTE/xDMA and
  A2/A3 build support. Exact installed CANN9.0.1 API/graph compatibility untested.
- Existing local A2 AIV paid evidence above is the cheapest native comparison;
  do not fork private HCOMM ABI before testing its public AIV selection at10/20KiB.

Design inference: retirement need not gate result consumption. Publish a per-slot
read-complete epoch asynchronously and check it before the next overwrite, or
prove a later collective's arrival causally implies the prior reads completed.
Such proof must include same-graph allocator alias/reuse, not only next replay;
separate output is required if peers still read the input. Whole-model graph
banks do not automatically provide per-collective peer-stable storage. Model
phase readiness may absorb retirement, but readiness to pull current data remains.

### Native AIV small-message switch qualification (September18)

Fletcher authorized trying native switches before building a custom TP2 protocol.
**Found an effective existing path:** `HCCL_OP_EXPANSION_MODE=AIV`, set before
worker/HCCL initialization. No installed/runtime/service edits. Same hw3 cards6/7,
CANN9.0.1, BF16 shapes and three-arm microprobe as allreduce-stream1. Capsule
`allreduce-aiv1` changes only the expansion mode;144 final-chain output checks
pass. PG10KiB ~14.1/14.6us versus retained30.2/30.7; raw same~9.9us versus~26.8.

Adjacent strengthened controls `allreduce-host2` then `allreduce-aiv2` retain
ALL16 chain consumer outputs, use signed varying per-element exact integer BF16
values,24generations/arm/shape with alternating banks and deliberate3ms rank
skew. Each run passes216checks/rank (432total), each checks16 full consumer
outputs and final allreduce y. All admissions exit0 and reclaim6/7. These are
serialized replay validations, not unbounded asynchronous service qualification.
Unprofiled mean of two block medians, rank0/rank1 us per producer+AR+consumer:

| Payload | default PG | AIV PG | default raw same | AIV raw same |
| --- | --- | --- | --- | --- |
| 10KiB | 30.211/30.113 | 15.168/14.394 | 26.538/26.585 | 11.339/10.476 |
| 20KiB | 32.831/33.019 | 18.236/17.384 | 28.827/29.030 | 13.608/13.275 |
| 15MiB | 927.339/927.945 | 909.354/908.522 | 925.006/924.945 | 906.319/905.607 |

`allreduce-aiv1/traceloom`: six two-replay profiles exported from raw nativePROF
and analyzed byc2a6920. Each graph contains16 hcom_allReduce_ KERNEL_AIVEC members,
no old inner SDMA/WriteValue/NotifyWait members. Provider COMMUNICATION_OP.algType
still reports MESH-RING-NHR: **that label alone does not identify AIV vs SDMA**.
Same-stream collective kernel median4.54–4.71us; PG6.02–6.50us. This proves AIV
execution, not the absence of synchronization inside its kernel. Cross-stream
profile gap inflation persists. Only10KiB profiled;15MiB dispatch not established.
`profile-summary.json`, `paired-summary.json` and README.md retain compact results.

Conclusion: native switch roughly halves small-message PG chain cost, much larger
than moving the old SDMA path to the compute stream.129*~15us suggests~2ms/step,
NOT an HTTP measured gain. Existing public switch is preferred over a private
HCOMM fork/custom peer protocol until real-model qualification shows another gap.
Do not enable globally for unrelated workers: process-level expansion mode may
change other collectives; actual Qwen service/bank lifetime/performance remains
unqualified. Same-stream communicator ownership is a separate optional change.

Fletcher then explicitly chose enabling the existing switch in the mod. The
packaged qwen_gdn/serve.sh now exports HCCL_OP_EXPANSION_MODE=AIV before exec;
README documents the same pre-start requirement for direct MixedWorker launches.
No import-time process-wide mutation, communicator replacement, DSV4 change or
native-Qwen Worker change. Shell syntax plus a fake-Python exec environment probe
verify AIV overrides an inherited HOST value and preserves MixedWorker/TASK_QUEUE.
This deployment choice does not convert microprobe results into service metrics.
