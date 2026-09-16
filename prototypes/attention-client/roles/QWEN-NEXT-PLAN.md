# Qwen3-Next-80B-A3B experimental suitability

2026-09-16: local checkpoint-header and pinned donor source investigation only;
no new accelerator test and no end-to-end support/performance claim.

## Capacity and topology

Local checkpoint is BF16, not INT8. It contains48 target layers plus one MTP
layer. Routed matrices occupy144GiB target-only and3GiB MTP. Shared experts
occupy294MiB including MTP; all other tensors plus shared gate occupy4.192252GiB.
Total non-routed payload is4.479361GiB, a checkpoint floor, not runtime peak.
See qwen-next-weight-census.json; weight_census.py reads headers without weights.

E2 cannot hold the target:72GiB/rank before workspace. E3 ownership171/171/170
fits approximately48.094GiB maximum target-only,49.096GiB including MTP. E4 evenly
places128experts/server:36GiB target-only or36.75GiB with MTP. Thus A2+E3 is a
five-card candidate with uneven expert placement; A2+E4 is an easier six-card
initial integration with the existing two-source protocol. A4+E4 on eight cards
requires expanding source count and is useful for testing supply/merging scale.
Do not promise A2+E2 or A3+E1 with this full BF16 checkpoint. INT8 would alter
capacity but requires a separately qualified quantized artifact and kernel path.

## What makes it useful (and what it does not guarantee)

Hidden2048 matches the earlier Qwen prototype. Each layer has512 routed experts,
intermediate512,top-k10,normalized routing; shared expert intermediate512 and
sigmoid scalar gate. Native Qwen3NextSparseMoeBlock passes shared_expert and gate
to FusedMoE. Qwen2MoeMLP implements sigmoid(shared_expert_gate(x))*MLP(x).
Retain that gate and exactly one output addition. Shared computation depends on
the same input as routed computation, not its result: submit remote work, execute
local shared MLP+gate, collect routed output, add. A blocking call followed by
shared MLP is not overlap. Avoid persistent polling starving shared compute on A.

One shared MLP has roughly one tenth the GEMM FLOPs of ten same-width routed MLPs
per token; this is NOT a measured latency ratio. It supplies useful work but
cannot be assumed to hide all routed service or all communication. Existing
Ascend FusedMoE has multistream shared-expert overlap, so compare against a
properly configured native overlapping baseline, not an artificially serial one.

This is a hybrid model:36 GatedDeltaNet layers,12 full-attention layers at
interval4. Donor contains AscendGatedDeltaNetAttention and registration, but the
native recurrent state, prefill/decode behavior and full-graph coverage require
actual gates. It is not a substitute for measuring long-context MLA-heavy V4.
512 small experts/top-k10 also stress broad-expert batching: expect routing and
small-GEMM efficiency to matter; do not sell this as an easy favorable workload.

## Recommended experiment

First target-only, no MTP: retain native attention/recurrent state and scheduler,
replace only routed backend, leave shared expert local. Extend layer/expert/row
catalogs, explicit owner mapping and role loading; no client routed weights.
Start with a representative native layer and same-state output oracle, then
full model A2+E4 or A2+E3 when uneven placement is supported. Compare serial versus
overlapped local-shared execution with identical routes and arrivals. Measure A
waiting, shared duration/overlap, expert batch rows, server idle and end-to-end
throughput/latency. Only then vary number of A sources to study cross-source
batching. No remote GEMM worker changes are made by this investigation.

Source anchors: pinned runtime vllm/model_executor/models/qwen3_next.py,
qwen2_moe.py; vllm_ascend/ops/fused_moe/fused_moe.py and ops/gdn.py.
