# Read LiveInfer's continuation protocol before reopening donor cross-step work

Source audit, September 13. This is an explanation and comparison, not renewed
authorization to deploy the rejected ping-pong candidate or a new speedup claim.

## Source identities and entry points

Retained LiveInfer checkout:
`/root/my-ascend-workspace/livemodule-main-integration`, commit
`e3acf0c1ded6227ecaeef4b8d4c813f27d65c2a1`. This is the inspected snapshot,
not a claim about today's latest upstream main.

- `src/livemodule/serve/dsv4/wave.py`: immutable host authorization packets.
- `src/livemodule/serve/dsv4/scheduler.py`: `replenish`, `_step_requirements`,
  `_plan_device_wave`, `_commit`; two-wave ledger and resource reuse.
- `src/livemodule/serve/dsv4/continuation.py`: `qualify`,
  `lower_mixed_target_seats`, `commit_continuation`; device truth and DONE drain.
- `src/livemodule/arch/ascend/request_parallel/dsv4/speculative.py`:
  `_construct_wave_routes`, `forward_device_wave`; graph scope and placement of
  state-dependent metadata computation.
- `src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py`:
  `submit`, `receive_oldest`; stream/event and invocation ownership.
- `tests/test_dsv4_wave_scheduler.py`:
  `test_terminal_reuse_keeps_already_submitted_old_generation_draining`,
  `test_every_rank_is_required_even_if_that_owner_is_empty`. These are protocol
  tests, not evidence of NPU speed or independently rerun here.

Donor experimental reference: `/workspace/strengthen-dsv4-pingpong`, commit
`b13f776`, particularly `prototypes/full-mixed/decode_shadow.py`,
`worker_submission.py`, `PINGPONG.md`, and the scenario's `worker-submission.md`.

## The actual dependency cut

Host metadata authorizes identity/generation, resident-to-seat mapping, prompt
payload, block grants, output limits and cancellation. It does NOT reconstruct
the next decode from accepted tokens received on CPU. Device continuation State
owns committed cursor, generated count, anchor token, next draft, phase and
generation. That numerical State is SINGLE-copy across the two input banks.

`_construct_wave_routes` explicitly limits shadow construction to host-known
owner geometry. Exact lengths, positions, validity and addresses are computed
inside `forward_device_wave`, in ordered compute execution AFTER the preceding
wave updates State. Reading that State early on ingress would break the protocol.

The device program qualifies grants -> constructs target metadata -> target ->
verifier/result selection -> draft metadata/context ingestion/query/proposal ->
commits continuation -> publishes banked output. Target and draft are not just
two independently captured bodies with host-mediated numerical handoff.

## Two waves, not two versions of reality

`replenish` fills a maximum of two outstanding submissions. After the complete
rank quorum for N is received, the host constructs N+2 while N+1 can execute.
Sequence parity chooses metadata/output bank. For K5, resource lookahead in
`_step_requirements` is `2*(K+1)+K = 17` tokens beyond the receipt-derived base,
capped by the request horizon. This covers both authorized waves and draft;
it is a physical grant bound, not a prediction of accepted tokens.

`submit` queues:

1. Ingress waits for this bank's previous graph reader; pinned inputs are copied,
   shadow is replayed, and ready is recorded.
2. Compute waits ready and this bank's previous output-copy completion; replays
   the whole wave and records graph done.
3. Egress waits graph done, copies banked results and records copy done.

`submit` does not synchronously finish those waits. It retains invocation, pinned
inputs and output destinations. `receive_oldest` waits its rank's copy completion,
retires that invocation and decodes its receipt. Input overwrite and output reuse
have separate gates. A rank-0 event never substitutes for all-rank completion.

EOS/length termination in N sets DONE on device. Already-issued N+1 for the old
generation is valid but drains with zero output and no logical continuation
writes. The host can authorize a replacement generation in N+2; same-stream
ordering keeps that replacement behind the old drain. Old ledger records remain
while outstanding waves reference them. Cancellation affects a future authorized
wave, not work already irreversibly submitted. No arbitrary cancellation rollback
or zero-cost physical empty-row execution is promised.

## What this means for DP and for donor

LiveInfer's inspected program has a host-planned fixed owner-major seat plane
and a fixed collective envelope, including empty owners. Local acceptance can
change valid lengths without forcing a new CPU rendezvous to discover each
rank's exact graph shape. This does NOT eliminate EP data communication, padding
costs, host planning or the requirement to receive the complete rank quorum.
Native donor's independently scheduling DP engines and per-step CPU token/mode
exchange are not this same contract; a TP-only worker patch does not replace it.

The donor experiment already implemented real pieces of this cut: owned pinned
ingress, device feedback-derived preparation, captured target metadata, and later
split-draft plus deferred count-receipt retirement. Do not describe it as only
two target graphs. But its stable-K5 adapter leaves prefill/turnover native and
retains separate native execute/sample/draft calls; the final worker cut is TP
only. It is not a complete transplant of LiveInfer's whole-wave generation and
distributed scheduling protocol.

## Do not collapse the experimental conclusions

- Runs098/101: explicit producer + captured metadata shortened matched cycles
  relative to native (TP ~68.5 ->57ms; DP ~65 ->51ms). Cohort throughput did not
  consistently improve; no universal service-throughput claim follows.
- Run109: moving receipt retirement later, on top of composed target/draft,
  did not give a stable additional cycle win.
- Run112: deduplicating draft metadata traversal reduced host work but did not
  yield a stable additional cycle win. Target replays already had substantial
  submission lead in the profiled stable window. That says nothing by itself
  about whether the entire preparation/sampling/draft chain was ready without
  gaps; a target-only launch marker is not whole-wave coverage.
- The existing non-adoption decision stands. These results neither prove all
  cross-step optimization failed nor prove remaining intervals are removable.

Next useful investigation is a dependency audit of a matched *current* window:
for preparation, target, verifier, draft preparation and draft, identify the
producer of each blocking value, launch lead and actual device coverage. Separate
already-queued device work from host-late issuance. Do not blindly reinstall N+2
or count every draft-to-target millisecond as idle. Gap-free steady execution
also requires host quorum/retirement/planning/ingress to finish within the next
wave's overlap window; the abstraction alone cannot guarantee that inequality.
