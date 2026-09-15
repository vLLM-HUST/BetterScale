# Static follow-up: shrink the request, not the numerical program

September15, BetterScale0.4/ec01754 and pinned Ascend9bf964c. No NPU run,
operator rebuild, public patch or allocator change in this follow-up.

## 1. HC-pre has a literal208MiB workspace floor

`upstream/vllm-ascend/csrc/torch_binding.cpp:1421` routes `npu_hc_pre_v2`
to fused `aclnnHcPre`, not the composite FP32 materialization path above it.
`csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp:332–347` calculates three buffers:

- per-cube double-buffered FP32 cast tiles;
- K-partition matmul partials;
- K-partition squared-sum partials.

It then adds16MiB and takes `max(requiredSize, 16MiB + 192MiB)`.
The non-arch35 kernel uses `GetUserWorkspace`, and its
`hc_pre_m_k_split_core.h:32–60` derives the three offsets from the same tiling
fields. No fixed192MiB addressing was found in this kernel path. The208MiB
floor is strong evidence of over-reservation, not permission to omit the runtime
workspace prefix or violate any tiled buffer bounds.

Illustration using24 AIC cores, hc_mult4, hidden4096, the source's256-row M tile
and1024-column conversion tile:

| HC input rows | Three user buffers MiB | Including existing16MiB MiB | Current floor MiB |
|---:|---:|---:|---:|
|24|6.301|22.301|208|
|516|35.273|51.273|208|
|4128|50.268|66.268|208|

These are **source-formula projections**, not extracted runtime tiling receipts.
The exact AIC core count and HC input shape must accompany an operator probe.
The proposed change is to retain the calculated bound + runtime provision,
removing only the unexplained floor; preserve tiling, precision, and arithmetic.
It may save about142–186MiB of this operator's request in those examples, NOT
necessarily that much graph reserved footprint: MoE becomes a competing peak.
No per-layer multiplication is legitimate. Rebuilding the native op is required;
a Python workspace setting cannot override this host tiling contract.

## 2. Clear dense KV backing once, not every strided view

BetterScale `src/betterscale/patches/auto_kv/_draft.py:65` recursively calls
`zero_()` on each exposed view. Its key `(data_ptr,numel,dtype)` identifies views,
not complete underlying storage and does not include stride.
Pinned installed `vllm_ascend/worker/model_runner_v1.py:3836` allocates compressed
KV as contiguous INT8 backing, shared by listed layers. `_adjust_kv_layout:3950`
then constructs typed `as_strided` views with common-page stride; logical payload
can therefore have gaps between pages. The post-warmup clear is operating on
those views, not on the original contiguous allocation.

A native strided-zero temporary is plausible but NOT proven without its backend
implementation or a small isolated probe. What is proven is the unnecessary
view-wise organization and the222MiB observed high-water increase at final clear.
Prefer a startup-only clear of each **verified KV-exclusive backing** exactly once,
including padding, after all warmup work completes and before admission. Preserve
all graph-bound addresses. Do not blindly zero any reachable storage, model/graph
pool, old trial backing or `_base` chain. Native byte allocations provide the
ownership proof; dtype views need not retain a useful Python `_base` chain.
A fallback is page-chunked strided clearing, bounding any materialization without
changing contents. Do not simply delete cleanup: warmup writes scratch State.

This is an easier Python-side patch candidate than changing neural arithmetic.
Whole-pool clearing is a safe-scope candidate, not a proved speedup; scratch-page-
only clearing requires a complete producer-write census and is not the first fix.

## 3. Residual clones are candidates; in-place HC-post is a different change

`vllm_ascend/models/deepseek_v4.py:992,998` clones hidden before each HC-pre.
The v2 binding takes const input, allocates new y/post/comb outputs, and the kernel
reads x while writing separate outputs/workspaces. HC-post also constructs a
fresh output (`torch_binding.cpp:1270`). This supports replacing a defensive
clone with an alias, subject to the whole compiled forward mutation contract.
Do NOT confuse that with making HC-post overwrite residual in-place: cross-tile
read/write safety would need another kernel-level argument.

For BF16 [rows,4,4096], one image costs rows*32768 bytes:16.125MiB at516 rows,
129MiB at4128 rows. Actual rank-local shape and compiler clone elimination must
be checked before claiming those bytes are paid or saved. Residuals and graph
scratch reuse across layers;43 layers do not mean43 simultaneously live copies.
Retain the forward tuple interface and account for existing draft/aux consumers.

## 4. Lower priority

HC-head still casts its input to FP32, materializes square/product tensors and
reduces. Chunked or fused row-local processing is plausible, but HC-head did not
set the measured target apex; preserve FP32 arithmetic and do not assume a local
saving changes graph reservation. MoE already uses fused gate/up/quant and GMM2;
its shoulders are not evidence for wholesale replacement.

Recommended bounded follow-up: isolated dense-versus-strided State clear, and a
single HC-pre workspace-size/oracle/FULL replay probe. Only after these answer
the real gap should another full-model reserved-memory comparison be purchased.
