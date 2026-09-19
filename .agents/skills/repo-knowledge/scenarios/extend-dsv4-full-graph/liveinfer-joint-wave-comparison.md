# Compare the qualified LiveInference wave with BetterScale's seams

Source/evidence audit, 2026-09-16. Grammar is explicitly outside this inquiry.
This is not a new hardware run or an implementation decision. Follow-up to
[the donor gap audit](joint-execute-sample.md).

## Reference selected by actual acceptance, not latest-source appearance

Read LiveInference commit `4e3323ef5a964cabb94811041ec2102e0a66fda2` using
`git show` in `/root/my-ascend-workspace/LiveInference`. The current checkout
`05ac1541` has later resident-slot, ragged-wave and pipeline changes; do not
attach the historical receipt to those unexamined differences.

The local capsule exists at
`/workspace/my-ascend-workspace/runs/dsv4-main-acceptance/20260904T155850Z-4e3323ef`.
Read `source-receipt.json`, `qualification.json`, `trie-acceptance.json`, and
`opencompass-acceptance.json` directly; their source identity agrees.
The real-weight TP8/EP8 K5 route used 43 target + 3 draft layers:

- Trie fixture: 24 completed turns, 768 output tokens, 250 waves including five
  mixed waves; eight zero worker exits and all 118 blocks/rank returned.
- Retrieval: 32/32 full-score rows, **31/32** exact reference token rows,
  239 waves, eight zero exits. The remaining answer adds a phrase around the
  correct paragraph number. This is not a full OpenCompass-suite result.
- Profile receipt reports near-contiguous steady decode (rank-0 median gap
  0.0002 ms). Trie prefill and mixed gap medians are about 26.25 and 27.09 ms;
  do not call the entire serving path gap-free. These are historical direct
  probe observations, not an HTTP or current-plugin speed comparison.

BetterScale comparison is `b0e6ad9` plus the source pins in the donor gap audit.
Its TP split-draft and DP async-decode branches are separate qualified paths;
their mechanisms must not be described as one already-qualified composition.

## Proven program and exact cursor convention

Historical `src/livemodule/arch/ascend/request_parallel/dsv4/speculative.py`,
`forward_device_wave`, owns the following ordinary tensor-call sequence:

```
qualify identity/generation against continuation
  -> lower target inputs/metadata from device State
  -> target -> owner-hidden restore -> global top tokens (or logits/argmax)
  -> greedy verification -> EOS/length/cancel selection
  -> derive draft metadata -> publish accepted auxiliary context KV
  -> DSpark -> proposal sampling
  -> commit continuation -> publish banked egress
```

This route uses `DSV4GreedyVerifier.verify_tokens` and device result selection,
not an unmodified generic vLLM `sample_tokens` method or full-policy sampler.
It proves joint numerical composition, not capture of every native sampler
feature. Preserve native donor numerical functions where possible rather than
silently substituting this verifier for arbitrary donor policies.

`serve/dsv4/continuation.py:lower_mixed_target_seats` constructs
`[anchor, draft_1, ..., draft_K]` and starts at `committed_cursor - 1`.
The cursor counts committed sequence tokens **including the emitted anchor**,
whose target KV has not yet been computed. For a nonterminal wave accepting
`a` draft tokens, output width is `a+1`, cursor advances by `a+1`, and the last
correction/bonus becomes the new anchor. For K5, cursor 101 and acceptance 2
mean target starts at position 100; accepted input context covers 100..102,
new anchor is position 103, new cursor is 104, next target starts at 103.
Do not map this cursor directly to donor `num_computed_tokens` without the
pending-token convention. Terminal clipping is separately owned by result.py.

## Seam-by-seam differences and what transfers

| Seam | Qualified LiveInference implementation | BetterScale/pinned donor difference and implication |
| --- | --- | --- |
| Dynamic inputs | `continuation.lower_mixed_target_seats/metadata` reads device anchor, draft and cursor inside the wave. Host supplies grants/identity/prompt. | DP `_producer.Slot.replay` already copies native device counts/anchor/draft and replays derivation, but still takes donor CPU budget/carriers. Reuse that arithmetic; do not characterize it as CPU reconstructing every token. |
| Metadata | `_construct_wave_routes` handles host-known owner geometry only. Exact positions, masks, sparse indices and native descriptors are computed in the wave. | DP captures target metadata separately; draft builders remain host-orchestrated. Move the changing device derivation into the joint recording while preparing only host-known facts outside. |
| Target to sampling | One root calls target, owner-hidden restore, `compute_top_tokens` or logits/argmax, then verifier directly. | Donor graph ends around model body; logits and sampling are later runner calls. Widen capture; preserve donor SP/TP output layout rather than importing LiveInference's owner-restore collective. |
| Verification to draft | `result.py` clips outputs, selects anchor and proposal mask; `llm/dsv4/device_dspark.py` uses accepted input-context width to derive draft start/length/slots without scalar readback. | Padded donor helpers already consume device acceptance, but mix backup-token H2D, CPU DTO construction and D2H publication. Separate these side effects from tensor work. Fixed tensors need stable per-replay values, not frozen capture-time Python lists. |
| Context versus draft queries | Context plane is K+1 rows masked to accepted input prefix; query plane is K rows starting at correction/bonus. Context KV write stays in the correct owner-local mask scope. | Existing TP split-draft has a qualified small fused path and actual-length context ingestion for larger paths. Reuse its numerical distinction; do not pad all context to a huge target bucket just to obtain one graph. |
| Draft to next target | `commit_continuation` writes cursor, generated count, anchor, draft, phase and reason into one persistent State. Next ordered replay reads it directly. | Feedback is distributed among runner attributes; `_copy_valid_sampled_token_count` also publishes device references. Removing its copy must not remove publication. Joint recording should write stable feedback destinations. |
| Capture ownership | Prototype `model.py` registers root `device_wave_0/1` before activation. Common graph backend invokes the root entry inside `torch.npu.graph`; children are ordinary numerical calls. | Target ACLGraphWrapper and TP DraftGraphRunner own separate recordings. The latter asserts `not ctx.capturing`. A joint owner must enter original numerical bodies and preserve attention capture/update semantics, not assume replay of existing child graphs is transparent nested capture. |
| Output transport | `resources.DSV4WaveEgress` owns two fixed banks. Executor waits prior graph reader before ingress overwrite and prior D2H before output overwrite. | Native uncaptured sampler allocates invocation-private outputs. Capturing it changes address ownership; retain explicit copy-complete fences or ordered snapshots. |
| Completion/turnover | `qualify` accepts old DONE generations as zero-output drains. Scheduler retires using complete rank receipts; already-issued old wave runs before replacement. | Stable-decode patches deliberately fall back on turnover. This lifecycle must be separately implemented/qualified before extending a joint wave beyond stable cohorts. |
| Distributed shape | Fixed owner-major target K+1 and draft K envelopes retain empty-owner collective participation. | Native DP independently coordinates shape/mode; `_propose` can do CPU-group synchronization mid-chain. Pre-agree a joint envelope; do not transplant request-owned collectives or assume TP and DP share this gap. |

## The actual host/device overlap law

Historical `wave_executor.py:submit` queues ingress -> ready -> whole-wave
compute -> done -> egress-copy -> copied, retaining invocation and pinned inputs.
`receive_oldest` waits copied and retires. Host waits exist; they are outside the
numerical target/sampler/draft chain, overlapped with a later admitted wave.
Input and output banks are separate from the single-copy numerical State.

`scheduler.py` reserves at most two outstanding device waves and sufficient KV
lookahead: `2*(K+1)+K` tokens beyond its receipt-derived base (17 for K5), capped
by the request horizon. It does not predict acceptance or reserve a whole output
lifetime. `test_terminal_reuse_keeps_already_submitted_old_generation_draining`
and `test_every_rank_is_required_even_if_that_owner_is_empty` specify CPU protocol
contracts, not fresh hardware evidence.

Do not copy the early full-Python shadow traversal: the scenario records an
observed ~5-second/wave host preparation failure. Explicit route-construction
actions fixed that path; shadow prepares host metadata, not the numerical model
again. The qualified route still spends host preparation time; overlap, not its
absence, is the achievement.

## Consequence for the next experiment

Two questions need distinct receipts: (1) can the donor execute+sample numerical
work, including draft, inhabit one graph; (2) can consecutive such graphs advance
without receipt-derived numerical preparation? Start by reusing native bodies
and giving them one capture owner, stable feedback/egress, and explicit host
ingress/retirement. Then test the cursor/acceptance law, terminal drain and
resource authorization. The reference supplies a concrete protocol, not a reason
to import its model implementations, request-owned parallelism or whole runtime.
