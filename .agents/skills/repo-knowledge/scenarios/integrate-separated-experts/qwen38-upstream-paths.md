# Choosing an upstream Qwen3.8-Flash-Next attention path

Observed 2026-09-18; this is a source/author-evidence survey, not a local NPU
qualification. Recheck PR status before adoption. Do not confuse this Qwen4Exp
model with Qwen3-Next-80B or Qwen3.8-27B. Our current Qwen38 client still comes
from the disconnected LiveInfer root documented in qwen38-loading-and-wire.md.

## First candidate: vLLM-Ascend QSA Lightning Indexer

- https://github.com/vllm-project/vllm-ascend/pull/15162 was open, targeting
  releases/v0.26.0rc and depending on #15072. It includes packed-cache backing
  reuse, six contiguous state slabs, graph-safe/address-stable PLE, QSA cache
  updates and sparse key load layout fixes, and MTP metadata fixes.
- https://github.com/vllm-project/vllm-ascend/pull/16367 was an open documentation
  PR. Its A3 recipe uses our Eco-Tech W8A8-MTP model, TP8/EP, FULL_DECODE_ONLY,
  eager MTP3, and experimental prefix caching with mamba-cache-mode=align.
  It supplies temporary A3/A5 images, not an A2/910B qualification.
- Read `vllm_ascend/models/qwen4_exp/lightning_indexer.py` in #15162:
  BF16 Q, head width128, <=64 query heads, compressed cache [pages,192,1,128],
  compression4, token topk2048. Every query row becomes a TND sequence;
  metadata supplies per-row causal visible compressed groups and page tables.
  Native `npu_lightning_indexer` selects512 groups; expansion appends the
  incomplete tail into a2051-wide output. Optional E3 is a custom compiled op.
  This is not a drop-in cache-layout replacement. A2 availability and numerical
  contracts need a bounded operator gate before model integration.
- Flags in the recipe are VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1 and
  VLLM_ASCEND_ENABLE_QSA_E3V=1. Do not merely set these in our existing runtime.
- #15040 is a draft adaptation without local NPU/E2E validation in its body;
  #15574 is an enablement scaffold. Neither alone proves a mature deployment.

## Second candidate: SGLang NPU kernels

https://github.com/sgl-project/sglang/pull/37570 and
https://github.com/sgl-project/sgl-kernel-npu/pull/807 were both open.
Recorded tested pair: SGLang e2f90ac5f4616cd7d521ef7bc19124dd86ed3b95,
kernel 7ede76da2f3e12fa16f7cad3161b199b24aeb6e1. Author evidence is A3/BF16,
graph replay+NEXTN, not our A2 W8A8 quantized lane. Kernel modules were not yet
in the CI dependency2026.9.0 according to the integration description.

Four small kernels under sgl_kernel_npu/qwen3_8_flash_next: sparse_attention,
mqa, expansion, hc. Most useful independent reference is paged MQA:
loads physical keys from page tables, sums ReLU(dot(Q_head,K)) across heads,
avoids gathered-key/per-head-logit materialization. **Still writes a final
FP32 [query_rows,width] score tensor; it does not fuse away topk/logits.**
Its capability gate is <=128 rows and <=8 heads. HC fastpath is <=32 rows;
large sparse attention batches use bounded launches. Do not assume the small
row MQA kernel is an efficient large-prefill implementation.

## vLLM main: borrow shape organization, not NVIDIA speed claims

https://github.com/vllm-project/vllm/pull/54513 merged Sep2: separate paged
uniform decode/verify and prefill indexer kernels; per-row visible blocks in
metadata. Prefill workspace has a configured cap (default512MiB there), decode
not covered by that cap. Heuristics and microbenchmarks are GB300-specific.
https://github.com/vllm-project/vllm/issues/55922 tracks PLE fusion (#54517),
QSA cache work and other pending optimizations; it is not an Ascend support
matrix. #57105 addresses QSA logits-workspace fragmentation (open at lookup).

Decision: keep server/client fixed; first compare native Ascend QSA/indexer
and state contracts with our current attention, then borrow individual kernels
or use a native model adapter if it actually removes owned code. Do not migrate
engines, upgrade release pins or run a full-model campaign just from this survey.

## A2 microbench outcome (later September18)

Before another adoption attempt, read
`prototypes/attention-client/qwen38/upstream_microbench/README.md` and its
41-case receipt. The current LiveInfer selector **already uses absorbed
LightningIndexer**, including causal prefill residue lanes. Native row-TND is
not a universal upgrade: some shorter shapes improve,4096/64Kregresses and
selected top-k sets differ in a few rows even with identical unitBF16 weights.
Do not waive those differences or relabel them accepted rounding noise.

Both upstream direct sparse attention implementations save allocated workspace
but lose latency to the current gather+FIA on910B2. SGLang vector-MQA+topk and
broader expansion kernels also do not justify replacement. No runtime default
or release pin changed.

The useful candidate is HC norm/mix borrowing at<=32rows:32-row complete HC
fragment169.8→107.5us,12changing-inputgraphgenerations, maxrelativeL2 7.61e-5.
`hc_adapter.py` deliberately preserves existing injection-gate preparation
before attention/expert computation and final injection afterward. Full-model
validation remains distinct; the entire upstream HC call would move some
precomputed work across that dependency boundary. This is a leaf candidate,
not a promoted whole-model backend or a measured serving gain.
