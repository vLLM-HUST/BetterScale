# Qwen3.8-Flash-Next W8A8+MTP: September16 readiness audit

Requested repository: [Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp](https://modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp).
Pinned download revision `c76aa96a5e730de35211d1b0b3d05cc3c67efae5`, destination
`/data/shared_models/Qwen3.8-Flash-Next-w8a8-mtp`. The remote75-file API inventory
sums to245,484,021,010 bytes. Approximately228.625GiB,61 safetensors shards.
The shared volume had719GiB free; no old profile database was removed.

Config declares Qwen4ExpForConditionalGeneration,48 text layers, hidden2560,
expert intermediate640,512 experts/top-k10, HC count4,12 full-attention/QSA and
36 linear-attention layers, PLE n-gram state, and one MTP layer. It is not the
current H2048/M512 Qwen3-Next BF16 geometry. Quantization excludes shared expert,
PLE, linear attention and HC, among others: an INT8 filename is not proof every
weight is INT8. Current expert server is BF16 NZ, not an INT8 quantized backend.

## Native support is not yet a qualified local starting point

- [vLLM #53896](https://github.com/vllm-project/vllm/pull/53896) introduces the
  model; current upstream qwen4_exp package resolves CUDA/ROCm implementations.
- [Ascend #15040](https://github.com/vllm-project/vllm-ascend/pull/15040) remains
  draft in the inspected page. Its author reports CPU/lint checks, explicitly
  not NPU/end-to-end validation. Git's current PR head is
  `aaa96c54ddfe774cd7d4b614ae608106b273b291` (the web page showed an older force-push).
  Captured patch covers QSA, HyperConnection, PLE placement and MRV2 model state.
  Main observed at`52ce43b15172fce79297ae4f20612fec3a476f3f` does not have these
  two proposed model files at their PR paths. Local pinned donor registries
  likewise contain no Qwen4Exp registration.
- The PR explicitly does **not** support VLLM_PLE_CPU_OFFLOAD: upstream uses
  CUDA Driver IPC/semaphores. The remote index assigns26 entire files exclusively
  to PLE, totaling95.429GiB including headers. A TP1 attention owner cannot simply
  retain this whole region in64GiB HBM. This must be solved before selecting A/E
  topology; moving routed experts away does not by itself solve PLE residency.
- Index metadata.total_size=142,916,142,560 is smaller than the actual remote
  file inventory. Do not use that metadata alone for download/state budgeting.
  Full tensor-header census after download is authoritative for placement.

No donor environment/pin or installed model runtime was modified. The optional
new-model split should begin with native support/PLE and quantized expert ABI
qualification, not relabeling the existing BF16 Next gate as this model.

## Active artifact handles

Download runner, pinned API inventory and logs:
`runs/qwen38-download/`. Uses a separate `/tmp/betterscale-model-download-env`,
ModelScope snapshot_download with exact revision and local_dir; SDK receives
remote SHA256 values and verifies downloads. Worker count increased4→12 after
observing low aggregate transfer rate; incomplete files are resumed, not deleted.
The final receipt must be named using the complete model basename (dots in
Qwen3.8 are not filename extensions).

Remote header probing hit a connection close; no successful all-shard census is
claimed. It did not alter downloaded weights. Finish with local header and index
coverage checks after the SDK completes. Model loading and full A/E serving are
not yet qualified.

## LiveInfer route found after Fletcher's steering

The upstream gap is **not** absence of an owned implementation. Read-only lookup
found LiveInference branch`lumi/qwen38-flash-next-serving-plan`, commit
`820103bf7912f6cebbfd73602c1002a6fbb1bfdc`, at
`/root/my-ascend-workspace/stateharbor-qwen38-plan`. Its local Skill scenario
`complete-qwen38-flash-next-serving` owns the detailed gates. Main30738958 does
not contain this model; inspecting only main misses the existing work.

That branch already has host-mmap PLE tables, selected-TP semantic head slices,
a generation-checked mapped-host publication before layer0 and pull before
layer1 inside one ACLGraph, QSA two-rank islands, native GDN, HC, scheme-A MTP,
and a complete text root. BF16 real TP8 load/graph and a290-case target-only
retrieval spot-check are documented (not full-suite or new W8A8 qualification).
Thus PLE offload is an available owned adapter, **not a new invention required
by the chosen LiveInfer route**. The upstream-only obstacle above remains true
only for the unadapted donor path.

Current reuse boundaries: existing root admits TP2/4/8, with a minimum two-rank
QSA island; existing expert loader expects fused BF16 expert tensors. New W8A8
checkpoint uses per-expert tensors and quantization auxiliaries. Reuse the model
and host-PLE protocols, but explicitly bridge weight loading, quantized expert
math and the remote routed-expert seam. Do not silently treat two TP2 attention
groups as two TP1 attention ranks, or report BF16 dequantized execution as native
W8A8 inference. Candidate A4/E4 (two TP2 attention groups, four expert servers)
is an engineering hypothesis pending the adapter/weight census, not a placement
already qualified.

### Downloaded shard spot-check (not a complete checkpoint census)

The first completed shard has actual I8 expert weights: gate/up`[640,2560]`,
down`[2560,640]`, with F32 per-output-channel scale/offset`[out,1]`.
Non-quantized GDN and HC entries in that shard are **F32 on disk**, not BF16;
placement and conversion must follow the runtime contract, not the filename.
The existing host PLE table accepts floating checkpoint tensors and preserves
storage dtype during lookup, so it does not inherently require BF16 backing.
The mapped response protocol must still be checked against the new table dtype.

Source seam verified at the pinned LiveInfer commit:
`arch/ascend/llm/qwen38/moe.py::AscendQwen38MoE` owns the router, separate gated
shared expert and routed FusedMoE. Its current local expert count requires
EP==TP; replacing the routed branch must not accidentally keep that local
allocation or change shared/QSA ownership. The model's
`causal_lm.py::checkpoint_tensor_slice/load_weights` explicitly recognizes
fused`experts.(gate_up_proj|down_proj)` tensors, not this checkpoint's per-expert
triples. Passing quant_config through constructors is not sufficient evidence
that the new checkpoint is loadable. No new-model runtime gate is claimed yet.

### Pinned snapshot inventory correction

At final-download preparation, the initial75-file listing proved to describe
newer repository documentation rather than the exact pinned snapshot. A fresh
explicit-revision listing has76 files and245,484,022,806 bytes: its README is
1389 rather than1972 bytes and it includes a2379-byte`.gitattributes`. All tensor
shard sizes and SHA256 identities match the initial listing. The SDK already
uses the intended revision; no weight redownload is needed. Final validation
uses`runs/qwen38-download/pinned-remote-files.json`, not the initial listing.
