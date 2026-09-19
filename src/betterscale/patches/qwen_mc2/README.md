# Qwen owned-wave row projections

Installed by the Qwen owned model composition after weight loading.

Native Qwen sends each attention output projection and MLP down projection through
`RowParallelLinear.forward`: local GEMM, then TP AllReduce, then residual consumers.
The TP2 Qwen27 model has64 of each (local K3072 and8704, output5120).

This leaf binds those128 model instances after weight loading. It preserves the
original forward for pure decode and metadata-free initialization; owned GDN
metadata marks prefill/mixed waves, including the smallest prefill capacity16.
These use stock `torch_npu.npu_mm_all_reduce_base` on the separately initialized
native MC2 communicator. No weight conversion, new Worker, persistent scratch,
or per-step communicator initialization is introduced.

The branch lives **inside an opaque Torch custom op**. Dynamo otherwise specializes
a Python size/metadata branch while building the shared1–2048 compiled graph;
checking the Python source is insufficient to prove decode takes its old route.
During ACLGraph capture each bank records its wave's selected device operations.
Replay invokes that graph directly: it does not execute a host branch per layer.

Scope is the pinned, BF16, bias-free, unquantized TP2 owned route. Native MTP and
DSV4 are not modified. Qualification must check small/large prefill, mixed and
pure decode, both graph banks, and actual timeline routing—not just HTTP success.
