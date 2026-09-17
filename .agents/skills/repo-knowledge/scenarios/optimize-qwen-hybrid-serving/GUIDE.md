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
