# Owned wave: native model, LiveInference execution

This prototype transfers a constructed Qwen3-MoE model and fixed KV allocation
into the **actual LiveInference** Python runtime. It does not rename a wrapper
around runner.execute_model. The numerical root has no runner reference.

For the newer multi-session/APC candidate, enter [serving](serving/README.md).
Its static-FIA boundary is integrated, but the30B regression and missing FD plan
are explicitly recorded in [the diagnosis](fia-plan/REGRESSION-30B.zh-CN.md).
The fixed-wave protocol below is historical and retains its original PA updates.

## Ownership

- Native startup owns weight loading, distributed groups, original KV allocation,
  and compiled model construction. The harness first runs a separate native
  reference episode; those tokens are never installed into candidate feedback.
- A ModelBundle hands over the unwrapped model, sampler, frozen greedy policy,
  configuration, borrowed KV backing and its already allocated block table.
- LiveInference owns State declarations, activation snapshot/restore, four startup
  graph captures (prefill/decode × two banks), MetaTensor construction/shadow replay and invocation lifetime.
- OwnedRoot directly builds attention metadata and calls model + compute_logits +
  sampler. Decode captures these together with device cursor/token/remaining
  updates and receipt publication. FULL prefill captures model, sampling,
  continuation initialization and receipt publication too; there is no eager
  candidate prefill.
- The host issues a bounded, balanced wave program, updates PA/FIA tiling, and retires
  receipts. Two outstanding invocations are allowed; terminal waves still enter
  the distributed computation but publish count=0 and do not write KV.
- During candidate execution the runner's execute/sample/metadata/capture entry
  points throw if called. This is an executable fence, not a claim that every
  native object is physically independent of all startup-global state.

LiveInference source is copied unmodified into each ignored run capsule; its
commit is recorded. No runtime package or donor source is edited. Follow its
upstream license for any distribution; this prototype does not vendor that tree
into tracked BetterScale source. The reactor protocol is adapted from its
Apache-2.0 `src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py`
and scheduling authority from `src/livemodule/serve/dsv4/scheduler.py`, at
05ac15419c0e73650e687ceb9daffeb7874865f0. DSV4 packet/model/resource classes are
not transplanted into Qwen; LiveModule lifecycle code is reused directly.

## The remaining attention seam

The installed native paged-attention backend uses a CPU context-length carrier.
PALength is a genuine MetaTensor recipe, evaluated by LiveInference's construction
and shadow protocols; task update then publishes native PA/FIA ExternalEvents before
graph replay. It is **not device-only attention metadata**.

The private native PA registry is scoped per transport bank. This requires
serialized host calls; it is not a multithreaded runner API. An invocation retains
its CPU carrier through completion. Shadow replay projects metadata construction;
it does not run another numerical model or mean the independent native oracle.

Device cursor and sampled anchor feed the next graph directly. Host PA lengths
are projected from a known, non-speculative fixed wave program; this is not yet
the protocol for variable acceptance, arbitrary request churn or asymmetric DP.

## N+2 is an authority and lifetime protocol

`schedule.py` keeps at most two outstanding waves. All EP ranks return the
receipt for N through the CPU communicator; only a complete, sequence/generation-
qualified quorum permits N+2. N+1 need not be retired. CPU receipt tokens are
collected for verification/output only, never used as the next model input.

`executor.py` has separate ingress, compute and egress streams:

1. Ingress waits for this bank's previous graph reader, copies owned pinned
   prompt/authorization payloads, runs shadow, then publishes ready.
2. Compute waits ready and this bank's previous output-copy completion, publishes
   native attention task updates, and replays the entire graph.
3. Egress waits graph completion, copies the banked result into an invocation-
   private pinned destination, and records copy completion.
4. Retirement waits only the oldest copy event and retires its LiveInvocation.
   Pinned inputs, metadata carriers and output storage live through this point.

Native attention also owns an update stream; “three-stream reactor” does not mean
exactly three total device streams including native internals.

Numerical continuation has ONE copy, ordered on compute. Host ingress never
overwrites it. A terminal receipt authorizes the next request's prefill behind
the already-issued old-generation drain, not before it. New-generation State
initialization itself runs in that ordered prefill graph.

The current oracle uses two same-prompt generations per DP owner (different
prompts across DP). Its six-output limit deliberately makes the second prefill
land on the OTHER bank: terminal wave5 -> pending drain6 -> prefill7. Both
prefill banks therefore replay real inputs, not just startup exemplars.
Host sequence traces prove ordering/window/quorum, not physical overlap duration
or a speedup. Asymmetric DP termination is explicitly rejected in this portfolio.

## Run and evidence

Use the fixed donor runtime and the shared-host subset admission supervisor:

```sh
QWEN_TP=2 QWEN_DP=2 QWEN_PROMPTS=language PROBE_DEVICES=0,1,2,3 \
  PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/owned-wave/UNIQUE_NAME \
  bash /workspace/strengthen-dsv4/prototypes/owned-wave/run.sh
```

Set QWEN_DUMMY=1 for a two-layer startup/lifecycle gate. Omit it for all 48
BF16 layers of /data/shared_models/Qwen3-30B-A3B. Each DP rank gets one 32-token
prompt; language mode uses different prompts on the two ranks. EP uses ALLGATHER.

The strict oracle checks FULL prefill plus six decode steps against independent
native tokens and every deduplicated KV backing byte, followed by two no-write
terminal drains. Then the continuous N+2 episode runs two generations without
resetting KV between them, checks all tokens and final full KV, and records every
submission/quorum transition. Activation must restore adopted backing after
capture. Candidate numerical Python forward runs eight times during startup
warmup/capture and zero times during all subsequent invocations or shadow.

```sh
python3 -m unittest discover -s /workspace/strengthen-dsv4/prototypes/owned-wave -p 'test_*.py'
python3 /workspace/strengthen-dsv4/prototypes/owned-wave/analyze.py CAPSULE
```

This is an isolated ownership/correctness gate, not a throughput result or a
finished serving/plugin API. The harness still uses the native engine's startup
and worker RPC, and adopts its allocated block addresses. It does not implement
a replacement general KV allocator, dynamic batching, EOS, grammar, stochastic
policy evolution, preemption, speculative acceptance, or failure recovery.
Requests are finite length, greedy, balanced across DP, one resident per owner.
FULL here qualifies the complete graph at an exact 32-token prompt width, not
arbitrary-length, chunked, padded or mixed prefill/decode batches.
A failed invocation aborts the bounded process; there is no unsafe in-process retry.

## Recorded gates (2026-09-16)

- Earlier takeover2 / takeover-real1 qualify eager-prefill ownership, not FULL
  prefill or the new three-stream N+2 protocol.
- full-n2-dummy1: four ranks completed numerical checks; a foreign job appeared
  on cards0–3 during teardown, and admission supervision aborted the process.
  Preserve as interrupted, NOT an end-to-end PASS.
- full-n2-real1: real 48-layer BF16 TP2 × DP2 / EP4, cards4–7, all ranks PASS,
  both clients and supervisor exit0. Seven-output requests, 16 continuous waves,
  terminal6/drain7/prefill8.
- full-n2-real2: same real topology, six-output requests, 14 continuous waves,
  terminal5/drain6/prefill7. Both prefill banks replay; all four ranks PASS,
  both clients and supervisor exit0, release receipt present. Current source
  implements this odd-turnover gate.
- CPU capability-fence + scheduler tests: 7 passed. Missing quorum, stale/error
  receipt, asymmetry, resource grant, terminal-prefill drain, and odd/even
  prefill bank selection are covered.

## Device-length alternative: rejected experiment, not a hardware verdict

attention_probe.py / leaf.sh retain three bounded failures under
runs/owned-wave/attention{1,2,3}. Stock sparse_flash_attention did not accept the
ordinary Qwen GQA shape. The LiveInference native SFA donor is specialized for
attentionMode=2, single KV head and 512+64/512 dimensions. Its working DSV4 path
is not evidence that the same ABI directly supports Qwen. Do not distort Qwen
dimensions or numerical semantics merely to force that operator to pass.
