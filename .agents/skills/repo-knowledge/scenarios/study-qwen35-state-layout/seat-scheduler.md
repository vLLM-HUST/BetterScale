# Live seat scheduling and shared-page pressure

Enter when extending the owned Qwen scheduler, changing admission/preemption,
or interpreting C16/R20 evidence. This is distinct from the already-qualified
E1 end-to-end execution in `owned-execution.md`.

## Accepted semantics (Fletcher, 2026-09-26)

- E bounds concurrently executing requests; R bounds resident GDN/continuation
  seats. Attention pages are one shared pool, allocated as computation advances.
- Do NOT reserve prompt + maximum requested output at admission. That was a
  rejected draft, not an invariant of this runtime. Ordinary pooled KV pressure
  is resolved by request preemption, not by immobilizing each active horizon.
- Without offload/checkpoints, preemption invalidates the **whole seat** after
  its current wave drains: request lease, represented prefix, all target/draft
  KV pages, GDN candidates/conv and continuation. Discarding only attention KV
  while retaining the final GDN boundary is not a resumable request state.
- The CPU request keeps original input and committed output IDs. Re-admission
  recomputes that history in an independently leased seat; no output duplication
  and no speculative/unaccepted IDs in the saved prefix.
- Normal completion is NOT preemption: it retains a hot seat and trims only
  uncommitted lookahead pages. Empty-seat-first / idle-LRU remains the policy.
- A wave's write/read lifetime is not a whole request's lifetime. Protecting a
  live invocation does not forbid preempting the request at its next boundary.

## Implementation and qualification

`generation_steps` is the single greedy/MTP protocol used by synchronous calls
and `Scheduler`; it yields page demands and bounded numerical steps. Scheduler
consumes completed replies before choosing whole-seat victims. Newest requests
lose first; following pressure the surviving cohort drains before re-admission,
so a victim is not immediately reintroduced into the same eviction loop. This
is a bounded initial policy, not an optimal fairness/throughput claim.

Wave selection serves one numerical family at a time, oldest-ready first (then
larger groups). Unselected steps remain pending without advancing their protocol.
Advancing all target/draft groups in lockstep can preserve arrival phase skew:
`20260926-scheduler-http1` reached16 active requests but only batch8, despite
correct16×12 raw outputs and16×9 long-chat outputs, eight TP2 pressure
preemptions, clean shutdown and released cards. Its strict batching assertion
failed before the disconnect case. This is a real grouping issue, not a token
failure or permission to label active16 as numerical batch16. The CPU regression
introduces a request one target step late and requires coalescence while both
are still in prefill, rather than waiting for accidental decode alignment.

Equal-width numerical steps use real batches in power-of-two graph buckets;
odd counts are decomposed, not padded with invalid GDN rows. The candidate
kernel skips negative-state rows without initializing its output, so dummy-row
padding is not safe merely because the convolution has a pad-slot sentinel.

20260926-scheduler-small1 captured its portfolio but failed on first replay:
metadata construction's replay context is None, not its capture-time `name`.
A stable per-graph construction action must retain the bank name itself. The
fix binds that name in a persistent action and adds actual CPU metadata replay
coverage; capture-only tests did not expose this boundary. This failed run is
not numerical-equivalence evidence.

20260926-scheduler-small2 stopped during automatic-fit calibration retirement:
the Qwen graph schemas had not declared `graph_pool_key`. The common backend
correctly refused to hand off an unpooled capture. The Qwen portfolio now
explicitly owns one serial scratch pool; all replay uses one stream, and only
retained State/MetaTensor/output copies cross calls. Do not bypass this guard or
replace the root's fit/rebind/recapture lifecycle with a second allocator.

Artifacts live under `runs/qwen35-state-lanes/`; each capsule freezes its source
before admission. CPU protocol coverage includes C16/R20, shared-page pressure,
recompute without duplicate output, stale leases, cancellation and mirrored
async ingress. `20260926-scheduler-small3` passes real C4/R5, all4×8 tokens equal to serial
target-only controls,12 graphs, and5 fitted shared pages. The complete root fit,
State rebind and recapture path passed, not merely a capacity calculation.
`20260926-scheduler-pressure1` passes4×125-token inputs plus8 outputs in only4
shared pages: two preemptions, one with committed output, exact recomputation,
stale-lease rejection, and real cancellation clearing all GDN candidates/conv
and invalidating continuation. Selected device2 returned IDLE.
`20260926-scheduler-tp2-1` passes35B BF16 TP2 C16/R20 with20 graphs, actual batch16,
all16×8 outputs equal to serial target-only controls, and both ranks fitting40
pages for256-token context. Its native anchor matches the prior independent
native receipt. Extra work uses seat16; the15-token hot prefix resumes seat0.
Selected0/1 returned IDLE. This is not throughput or maximum-context evidence.

`20260926-scheduler-http2` passes the installed public CLI/HTTP entry on35B
TP2, C16/R20,512-token context and only16 shared pages. It reaches actual batch16,
returns all16×12 raw native-reference tokens and all16×9 long-chat tokens exactly
(the latter with189-token inputs), performs eight whole-seat preemptions,
and handles an actual client disconnect (cancelled1, active0, waiting0, free
pages0→2). Service exit0; selected0/1 returned IDLE. The earlier E1 chat/EOS,
warm-prefix and unsupported-input cases also pass through this async entry.

The final source differs from that NPU-tested install only by presenting a
recognized client disconnect as handled HTTP499 instead of leaking CancelledError
into the ASGI error log, plus model documentation. A CPU ASGI regression proves
the waiting execution task is still cancelled. Do not reload72GB of weights to
requalify this error-presentation-only change. Final artifacts/tests and bounded
scope are in `docs/evidence/qwen35-live-scheduler.json`; no throughput or C32
execution result is implied. Pressure-cohort draining is deliberately conservative,
not a claim that admission/fairness/throughput tuning is finished.
