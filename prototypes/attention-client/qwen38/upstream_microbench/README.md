# Borrowing Qwen3.8-Flash-Next Ascend kernels

**September18 decision:** keep current QSA selection, expansion and gather+FIA.
Retain **SGLang HC norm/mix** as a useful, explicitly selected microbench
candidate. No installed runtime, published Worker, server/client protocol or
whole-model default was changed. All41 accepted matched cases completed on hw0
910B2/CANN9.0.1, with no weights required.

This is Qwen4Exp/Flash-Next, not Qwen3-Next-80B or Qwen3.8-27B. Our existing
indexer already calls an absorbed LightningIndexer; upstream is not introducing
fused selection for the first time.

## What was actually compared

`fetch_sources.py` downloads exact upstream commits plus licenses, rather than
installing either engine or maintaining another fork:

- [SGLang NPU #807](https://github.com/sgl-project/sgl-kernel-npu/pull/807),
  `7ede76da2f3e12fa16f7cad3161b199b24aeb6e1`: paged MQA, sparse attention,
  expansion and HC leaves.
- [vLLM-Ascend #15162](https://github.com/vllm-project/vllm-ascend/pull/15162),
  `9082742026e64e0311b4f4913850817dbc752875`: paged QSA attention and expansion;
  the LightningIndexer adapter supplies the native-TND calling convention.
- Current control: qualified hw0 `qwen38-bounded-qsa-runtime-20260917e/overlay`.

The sole source transformation for the Ascend Triton module replaces
`vllm.triton_utils` imports with installed Triton imports. Kernel equations,
launch tuning and wrappers remain unchanged. The TND arm uses the native
`torch.ops.npu.npu_lightning_indexer`; the control uses the existing
`torch.ops._C_ascend` operator. Qwen geometry: indexer4heads×128,
compression4/topk512groups; attention12Qheads/KVhead×256 with1or2KVheads.

## Results

Graph timings are milliseconds, medians of7 alternating-order rounds with3
replays per sample. Each candidate has a same-process, same-card control.
Comparisons across separate capsules are not controlled before/after trials.

### HC: a useful narrow borrowing

Hidden2560, HC4, lowrank320, BF16 dummy weights; norm, gate projections, mix and
injection are included. `hc_adapter.py` borrows only norm/mix, leaving existing
injection preparation and consumption in their original locations around the
attention/expert call.

| Rows | Current HC | Entire upstream HC | Borrowed norm/mix |
|---:|---:|---:|---:|
|1|0.0703|0.0621|0.0689|
|4|0.0817|0.0728|0.0773|
|32|0.1698|0.1044|0.1075|

The useful32-row gain is36.7%, or62.3us for this HC fragment. It is **not** a
whole-layer, token or service-throughput gain. The adapter preserves the prior
pre-block injection-gate schedule; simply replacing the whole upstream HC
transaction could move projection work onto the post-expert critical path.

Twelve changing-input graph generations per shape compare both outputs across
both candidate arms:144 comparisons. Rows1and4arebitwiseequal. At32rows,
max relativeL2 is7.61e-5 (11of24 comparisons non-bitwise per arm); no top-k
relaxation is involved. Whole-model hidden/State/MTP and language-quality
qualification are not supplied here. Large rows retain current backend behavior.

### Sparse attention: less workspace, worse latency on this machine

Both upstream sparse consumers were tested at1/4/32/128/512rows,1/2KVheads,
8K/64Kcontext, and2051selected slots; an additional128-selected case covers
short active sparse sets. A2 timings below include metadata adaptation:

| Consumer,128rows /2KVheads /64Kcontext | Graph ms | Capture allocated peak increment MiB |
|---|---:|---:|
|Current gather+FIA (Ascend comparison)|1.626|545.1|
|vLLM-Ascend paged sparse / matrix dots|7.251|198.6|
|Current gather+FIA (SGLang comparison)|1.624|545.1|
|SGLang direct sparse / vector reductions|28.317|27.2|

Ready-metadata arms remain slower too:7.207and26.747ms. We do not blame only
adapter Torch indexing or promote a kernel merely because it saves memory.
These are peak **allocated** increments during isolated capture, not reserved
pool size or whole-model peak. Direct sparse may be worth revisiting under a
real memory constraint, not as the current latency default.

Each attention case uses scrambled page tables, two request identities,
negative selected slots, a masked-out row when rows>1, and two changed Q/V
replays. An independent FP32 CPU attention oracle checks up to3rows including
the masked row; all-mask output must beexactzero. This is not exhaustive full
output/state validation or an end-to-end quality gate.

### Indexer and expansion: do not replace

Native TND and current BSND are close for independent query rows. SGLang's
MQA+topk reaches4.224ms at128rows/64Kcontext versus current0.291ms. Final top-k
sets match in the sampled independent-row cases; scoring alone is not the
measured comparison.

Causal prefill includes query/metadata reshaping, not only the native kernel:

| New rows / total context | Current residue lanes | Upstream row-TND |
|---|---:|---:|
|128/8192|0.132|0.114|
|512/8192|0.291|0.257|
|2048/8192|0.930|0.764|
|4096/65536|5.227|7.730|

One of two512-row generations differs in1selected set;4096-row generations
differ in2and1sets. Changing TND weights from scaledFP32 to the control's
unitBF16 does not remove those differences. Their exact cause is not proven;
no candidate is promoted and these are not dismissed as harmless perturbation.

Expansion over valid completed groups is exact but current is faster:
128rows current0.054ms, SGLang1.040ms, Ascend1.135ms. The first expansion fixture
incorrectly allowed selection of an incomplete compressed group, outside the
current indexer's contract; its failed capsule is excluded. Corrected sampling
chooses only completed groups. Invalid-group compaction is broader upstream
functionality, not a demonstrated bug in the current valid-input path.

## Reproduce and inspect

`result.json` is the compact41-case receipt. Full logs and per-run source
snapshots are in local `runs/qwen38-upstream-microbench-20260918`; remote artifacts
are `/workspace/betterscale-hw0/runs/qwen38-upstream-microbench-20260918`.
Accepted subdirectories: `indexer`, `attention`, `ascend-attention`,
`prefill-final`, `hc-final`, `expansion-final`. Earlier probes remain separate.

Use a private qualified environment and the existing subset admission helper:

```bash
python fetch_sources.py /absolute/fresh/upstream-directory
export QWEN38_PYTHON=/absolute/private/env/bin/python
export QWEN38_OVERLAY=/absolute/qualified/overlay
export QWEN38_UPSTREAM=/absolute/fresh/upstream-directory
export QWEN38_ADMISSION=/absolute/isolated-helper/admit_subset.py
export PROBE_HELPERS=/absolute/host-admission-helpers
bash run.sh 0 hc /absolute/fresh/hc-capsule
# Other kinds: indexer, prefill, expansion, attention.
# Ascend sparse attention: append --family ascend.
```

The helper owns bounded admission/foreign-occupancy monitoring/reclamation.
Only one selected device is needed per probe. Fetching and CPU compilation do
not require NPUs. Numerical checks precede reported timings. The graph results
retain their input frame through reset; production remains unchanged.
