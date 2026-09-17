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
its queued native counterpart was cancelled. Corrected native2/candidate2 pending.
No serving implementation change belongs to these diagnostic probes.
