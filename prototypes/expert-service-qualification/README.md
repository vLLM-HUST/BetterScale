# MTP expert execution qualification

Run `multi.py` with the pinned runtime and installed MOD package, under the host's
admission/lease/foreign-owner guard. It calls packaged `Deployment`; only request
generation and assertions live here. Default C64/128 requests exercise257..32769
input lengths, 4096-frame boundaries, mixed target/draft work, APC reuse and128
forced output tokens. Sampling uses real MTP acceptance, never synthetic rates.

This is an execution/liveness check, **not** a semantic-quality score, AgentX
Frontier point or isolated throughput comparison. The separate native-engine
retrieval inquiry owns the observed repeated-7 failure. Keep it visible without
requiring a generated-text oracle for communication/generation correctness.

Acceptance requires41 native routed shadows per attention rank;164 total bytes
of local routed metadata and no expert payload; successful native MTP acceptance;
zero output-length/request errors; two directly launched persistent kernels per
owner; exact client/owner generation equality; clean all-role exits and the
external supervisor's device-release observation. Server graphs are not admitted
as a substitute for the direct resident launch contract.

`synthetic_probe.py` separately qualifies the explicit benchmark-only sampler at
TP1/TP8 against an independent CPU oracle and cross-rank agreement. Fletcher stopped
the AgentX comparison before completion and switched to `vLLM-HUST/swe-prefix-reuse`.
Do not resume the old matrix or relabel its partial data.

`benchmark.py` uses installed Deployment for candidates, and a tiny native
observation Worker for DP8EP8/TP8EP8 controls. Controls prove all41 target/draft
layers have32 local experts and a complete nonoverlapping256-expert union;
eight independent replicas are not an admissible baseline. Both paths include
the qualified async feedback ownership repair and use **real** MTP2 acceptance,
query4096, FULL decode and native262144 context. The SWE client uses unchanged
prepared input deltas and actual generated token IDs, not synthetic acceptance.
This is fixed-shape serving, not correctness of generated tool calls or SWE solving.

The session relay retains `agentx_router.py` as its historical filename but now
routes by SWE `cache_salt` when no correlation header exists. New session plays
round-robin, later turns stay on their rank; request bodies/SSE bytes are unchanged.
Both native DP and separated groups use the same loopback relay. `test_router.py`
checks salt affinity and exact body/stream preservation. Keep server-side APC
metrics alongside client occupancy/turn coverage; correct IDs alone do not prove
cache hits. Each900s C64 run owns a fresh admitted eight-device deployment.
