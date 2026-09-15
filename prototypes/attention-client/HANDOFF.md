# Attention client prototype — September15

## Qualified

- Native two-layer reduced dummy Qwen:13 original-output and whole-KV exact checks,
  including prefill, mixed and decode shapes.
- Bank-private FULL attention-half graphs, native task metadata updates:10 reused
  layer stages with later positions/lengths. Native MLP is the local reference.
- Two prebuilt graph lanes with private KV and IO: withheld prefill reply does not
  stop decode completing both layers. Both native outputs/private KV exact.
- External INT32 packet producer ↔ unchanged workspace3532418 server, two cards:
  two same-graph episodes,16 packets/eight lane retirements, exact returned values
  and weighted top-k plus shared stand-in. IPC imports/exports drained/released.
- 11 CPU tests: route ownership, stale/duplicate replies, cancellation drain,
  out-of-order reduction, capacity, scheduler progress and packet-generation ABI.

## Where to read

`README.md` explains native insertion points and each evidence boundary.
`metadata_graph.py` owns graph banks; `scheduler.py` owns layer continuation.
`independent_lanes.py` is the two-lane native snapshot oracle.
`server_contract.py` plus `ipc/` own the delivered integer transport adapter.
The other task's server source was not changed or copied into the release plugin.

Frozen experiment root: `runs/attention-client-20260915/`.
Passing model runs: native2, graph3, metadata7, scheduler8, lanes12.
Passing joint IPC run: ipc3 (binary/source capsule ipc-v3).
Compact receipts live beside the corresponding prototype code.

## Not claimed

No real remote expert GEMM, BF16 communication ABI, real-weight quality,
throughput improvement, live HTTP scheduler integration, TP attention group,
or native DP2EP2 replacement. The model graph oracles capture on unseen shapes;
only the two-lane episode uses prebuilt banks throughout. A production worker
still needs a bounded startup catalog and resource admission policy.

The real joint path requires server-side expert compute and an agreed tensor
ABI; client-side device route packing and weighted reduction must then preserve
the original model router/quantization contract. Do not replace them with this
integer packet fixture or present the two separate passing tests as neural E2E.
Qwen3-30B-A3B has no shared expert: local shared hiding is not measured here.

Release defaults and installed donor runtime were not modified. The prototype
uses bounded local single/two-card leases and leaves foreign tasks untouched.
