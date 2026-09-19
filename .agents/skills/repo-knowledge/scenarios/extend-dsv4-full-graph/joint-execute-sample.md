# Joint execute_model + sample_tokens capture gap audit

2026-09-16, source-only audit at BetterScale `b0e6ad9`, vLLM
`752a3a504485790a2e8491cacbb35c137339ad34`, vLLM-Ascend
`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`. No runtime or NPU qualification.

Fletcher's objective is to capture execute_model and sample_tokens together,
not accept their present separation as an architectural limit. Python bookkeeping
inside those methods is a migration gap, not proof that their numerical program
cannot be jointly captured. A proposed bounded greedy pilot is an experiment,
not a decision to exclude other sampling policies from the eventual objective.

## Observed dependency map

Paths below are relative to the pinned upstream checkouts.

- `vllm/v1/engine/core.py:step, step_with_batch_queue`: execute is submitted
  before grammar is obtained; sample is a separate call. The batch-queue path
  can defer sample pending prior structured-output tokens. A worker-only joint
  replay therefore needs an explicit launch point: without grammar it could
  launch during execute and let sample retire; with grammar it could prepare
  during execute and launch the joint graph during sample. These are proposed
  adaptations, not implemented behavior. Preserve one replay and one retirement
  per invocation across both synchronous and queued engine paths.
- `vllm_ascend/worker/model_runner_v1.py:execute_model`: `_model_forward`,
  hidden-state selection, and `compute_logits` precede an `ExecuteModelState`
  Python tuple. There is no unconditional logits D2H between target and sampler.
  Capture currently ends too early to include the logits/sampling tail.
- Same file, `_sample` calls `input_batch.update_async_output_token_ids`.
  `vllm/v1/worker/gpu_input_batch.py` shows this waits on a copy event and calls
  `.tolist()` only when CPU sampling histories need unresolved placeholders.
  It is not an unconditional greedy barrier. Policies consuming history need
  device history or independently prepared history inputs, not silent omission.
- `sample_tokens` applies grammar via logits CPU transfer in this pin.
  Grammar is an external input dependency; applying a ready bitmask on device is
  a separate implementation gap from generating that bitmask on time.
- `_bookkeeping_sync` mixes CPU request/placeholder updates, optional token
  readback, prompt-logprob work and request-output snapshots. Async mode already
  avoids the ordinary token readback; its Python mutations must still happen
  exactly once per invocation outside replay or be explicitly deviceized.
- `sample/rejection_sampler.py:rejection_sample` already produces fixed
  `[batch, max_spec_len+1]` token storage with invalid entries marked -1.
  Greedy verification and `spec_decode/llm_base_proposer.py:prepare_inputs_padded`
  consume device acceptance without requiring an acceptance scalar on CPU.
  CPU `.item()` on query-start metadata is host-known shape work, not device
  synchronization. DEBUG rejection diagnostics *do* read device scalars; do not
  mistake those gated reads for a mandatory numerical dependency.
- `prepare_next_token_ids_padded` mixes device selection/counting with CPU
  backup-token lookup and H2D. Move host-known backup/discard facts into stable
  ingress, retaining their semantics for incomplete prefill/discarded requests.
- Runner `_copy_valid_sampled_token_count` combines asynchronous D2H with
  publication of device feedback references. Separating copies must preserve
  feedback publication. `_copy_draft_token_ids_to_cpu` already skips copies in
  eligible async paths; do not claim all drafts always round-trip through CPU.
- Proposer `_propose` calls `_sync_metadata_across_dp` between sampling and
  draft. Depending on configuration this is a CPU-group all-reduce; DP1 and
  explicitly eligible skip paths differ. A joint distributed graph needs its
  target/draft collective envelope agreed before launch, not erased rendezvous.

## Capture mechanics and ownership gaps

- `compilation/acl_graph.py:ACLGraphWrapper` owns model-only capture/replay.
  Runner attention task updates also surround that boundary. Joint capture
  must own the larger recording, execute original child numerical programs
  during recording, and retain required attention task/event updates. Changing
  runtime mode to NONE indiscriminately can select different attention behavior.
- BetterScale `split_draft.DraftGraphRunner.__call__` explicitly asserts
  `not ctx.capturing`; its private child graph policy cannot simply be nested
  inside a new parent. Reuse its original numerical callable and qualified
  context/query distinctions under a new capture owner, rather than assume
  existing graph replay can be recorded as a transparent child operation.
- Python tuple assignment, counters, object allocation and hooks run at capture,
  not every replay. Persistent device tensors and per-call host bookkeeping
  require distinct ownership even if the user-facing methods stay unchanged.
- Captured output allocations become fixed addresses. Async output objects
  holding references alone cannot stop a later replay overwriting pending D2H.
  Use explicit bank/credit fences or an ordered private snapshot. The number
  of physical graph executables follows address/backend requirements, not a
  blanket rule that every time step needs another graph.

## Cheapest next discriminating experiment (not yet executed)

Use one admitted fixed-shape greedy wave, retaining original target, logits,
rejection and DSpark numerical functions, and record their device work together.
Separate host ingress and retirement without redefining the numerical algorithm.
Compare first and subsequent replay token/count/draft outputs and full KV against
the original reference; vary device acceptance and input contents at fixed shape.
Delay egress deliberately to expose fixed-output overwrite. Then broaden shape,
turnover and policy coverage; validate two-step feedback separately from single
wave capture. Hardware work must enter probe-npu admission first. No speedup or
all-mode support follows from this source audit.
