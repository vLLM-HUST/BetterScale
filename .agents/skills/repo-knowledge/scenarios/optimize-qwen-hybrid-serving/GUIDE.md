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
