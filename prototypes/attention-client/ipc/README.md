# External packet producer for the delivered integer server

This is the attention/client-owned adapter to workspace commit `3532418`.
It does not modify or rebuild that server for integration: the test uses the
server binary from its accepted `20260915T181647Z` capsule.

The original server's test client manufactures its own inputs inside a kernel.
This client instead reads an **external immutable packet plan**. It publishes
payload and descriptor before READY, polls exact-generation DONE, pulls output,
and only then permits slot reuse. One persistent AIV invocation is captured in a
FULL graph. A second episode changes plan contents at the same graph addresses.
The CPU `OraclePackets` adapter owns route identity and weighted top-k retirement;
its output oracle also adds a local shared-branch stand-in once.

This connects the real IPC envelope, not a neural FFN: payload/result remain
INT32 width64, at most8 rows per packet. There is no BF16 reinterpretation,
quantization claim, distributed neural output check or throughput claim.
Attention graph qualification and this transport qualification are separate tests.
A real model still needs the server's real expert execution and row ABI, plus a
device route producer/reducer in place of this CPU packet-plan oracle.

`build.sh` builds only the owned producer and its ACL loader. The AIV ELF needs
its `.ascend.meta.pull_expert_client` kernel-type section; plain linking without
that section passes compilation but ACL binary loading rejects it with107000.
Keep the loader/ELF matched. The kernel IO primitives are reused from the delivered
server implementation; do not replace them with unordered flag writes.

`probe.py --clients 1|2 --client-build ... --server-build ... --out ...` runs a
bounded episode under the caller's NPU admission wrapper. It uses one server plus
one/two clients, not a model process per client. Export capabilities stay inside
private bootstrap pipes and never enter logs. Drain all kernels, close every
peer import, then release exported owner backing. On failure the episode is
poisoned and owned subprocesses are stopped; no live recovery claim is made.
