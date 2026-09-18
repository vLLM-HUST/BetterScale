# Integrate separated attention and expert serving

Use this scenario for persistent expert kernels, client publication/retirement,
or the Qwen Next/Qwen38 serving prototypes. This is **not** the public Worker
patch path. Read this guide, then only the entry matching the actual question.
Paths beginning `prototypes/` are repository-relative.

## Start with the current task, not the historical sequence

| Current question | First useful entry |
|---|---|
| Current throughput/topology/capacity claims? | `prototypes/attention-client/qwen38/TOPOLOGY-RESULTS.md` |
| Client/shared overlap, input pack or expensive collect? | `prototypes/attention-client/qwen38/PREFILL-PROFILE.md`, `client-pack-result.json` beside it |
| Choose an upstream Qwen38 Ascend attention/QSA implementation? | [upstream paths and adoption boundaries](qwen38-upstream-paths.md) |
| Qwen38 checkpoint, PLE repair, INT8 math or bootstrap? | [loading and wire gates](qwen38-loading-and-wire.md#paid-numeric-and-loading-lessons), then `qwen38/README.md` under `prototypes/attention-client` |
| Multi-source EOF, layer identity or IPC lifetime? | [two-source contract](qwen38-loading-and-wire.md#two-source-extension-and-admission-boundary); [Next completion race](native-client-and-server-gates.md#a2e4-efficient-server-confluence-and-long-lifetime-completion-race) |
| MTP, TP1/E3, State fit, retained SWE or QSA peaks? | [capacity and traces](qwen38-capacity-and-traces.md), then `qwen38/TOPOLOGY-CAMPAIGN.md` |
| Fused W8A8 GEMM scheduler, C/V pipeline or ACTIVATE optimization? | [upstream SwiGLU scheduling](upstream-swiglu-scheduling.md) |
| DFC return/weighted reduction versus fused client collect? | [return and reduction audit](dfc-return-and-reduction.md) |
| Fine-grained GEMM/input readiness or fair DFC comparison? | `device-service/FINE-PACK.md`, `FAIR-DFC.md`, `ROUTE-PULL.md` under `prototypes/attention-client`; [bounded gates](native-client-and-server-gates.md#fine-grained-expert-input-readiness-2026-09-16) |
| Priority/promotion/fairness? | `prototypes/attention-client/qwen-next/priority/README.md` |
| Native attention layer cut and FULL graph metadata? | [native-client gates](native-client-and-server-gates.md#split-qwen-attention-around-an-asynchronous-expert-boundary) |
| Hardware/environment on hw0? | `prototypes/attention-client/qwen38/HW0.md`; use probe-npu before launching |

## Contracts to carry into every change

- Keep binary, config ABI and buffer geometry together. Qwen38 uses17 client
  words, target INT8+FP32 scales, BF16 MTP and four weight pointers/layer.
  Old Next fine-grained collect cannot simply be enabled for H2560/W8A8.
- READY follows all payload writes. Shared-completion promotion belongs to
  the published generation; retirement follows every required reader/producer.
  A stronger completion must satisfy pending weaker notifications before reuse.
- Preserve layer compatibility during batching. Queue capability or persistent
  residency does not prove sources actually co-batched or that latency improved.
- Capture raw launches on the actual current capture stream; prewarm banks,
  keep their complete input frame alive and test changing inputs/generations.
- The checkpoint lane repairs three corrupted PLE integer tensors from an
  independently checked original. This is not untouched-checkpoint validation.
- Isolate admission-helper snapshots from experiment modules; a shadowed Python
  module has previously paired32-row allocations with a widened binary.

## Evidence boundaries, not current defaults

The hw0 C40/K1 topology matrix covers two turns of40 traces, not full trajectories.
Its native control uses global phase voting: it is **not stock vLLM**. Synthetic
capacity gates and realistic populated-context throughput are different tests.
The client pack leaf reduces1024-row stage time about13%; it is not yet a
whole-model gain. Shared overlap works but alone did not reduce that leaf time.

Retained evidence is split by the question it answers, not by date:
[checkpoint/transport](qwen38-loading-and-wire.md),
[capacity/workload](qwen38-capacity-and-traces.md), and
[native integration/server scheduling](native-client-and-server-gates.md).
Read the relevant section; do not concatenate these notes on entry.
