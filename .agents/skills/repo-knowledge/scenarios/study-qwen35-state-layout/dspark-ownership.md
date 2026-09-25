# Existing DSpark answer: local State owners, composed lifetime

2026-09-25 source inspection, LiveInference `05ac15419c0e73650e687ceb9daffeb7874865f0`.
Fletcher corrected DFlash to DSpark; no different checkout is needed. This
supersedes the search blockage, not any numerical qualification boundary.

## Ownership already exists, but distinguish two kinds of candidates

- `llm/dsv4/dspark.py:DeepseekV4DSparkModel(LiveModule)` owns its draft decoder
  children, built with `is_draft_layer=True`. The causal-LM wrapper is also a
  LiveModule. In the Ascend execution tree the speculative tail registers the
  complete draft as a child; target decoder layers are not descendants of that
  tail. This is structural ownership, not a global cache name registry.
- `arch/ascend/llm/dsv4/dsa.py:DSAAttention(LiveModule)` registers the exact
  `swa` StateTensor in each draft attention leaf. The admitted DSpark metadata
  path requires SWA-only State and rejects unexpected other persistent lanes
  (`metadata/dspark_invocation.py:compile_live_dspark_invocation`).
- Proposal **token IDs retained for the next verification**, however, live in
  `serve/dsv4/continuation.py:DSV4DeviceContinuation(LiveModule)`, not in DSpark's
  attention module. `draft_token_ids` has block_shape `(speculative_width,)`,
  int64 dtype and the resident slot domain. That same owner declares cursor,
  anchor, slot generation, phase, output count and granted block table.

Thus "DSpark owns its state" is right for draft numerical State, but not a
reason to put every proposal-related object under DSpark. The continuation
owner owns the cross-step continuation truth.

## Capacity and allocation are not a second ownership tree

`DSV4SIMDSWAState` selects the bounded geometry's slot domain and explicit
pages-per-slot. Distinct layers still own exclusive State lanes. Continuation
selects ExactStateCapacity(resident count), or ScaledStateCapacity over a shared
StateCapacityUnit when fitting. Wave sequence and poison latch use a separate
ExactStateCapacity(1) domain.

The containing root discovers declarations and the backend realizes/binds a
complete generation. A child owning State does not privately allocate/release
HBM on every call; semantic ownership, capacity coupling, physical backing and
invocation lifetime are separate responsibilities already supported by the
existing system. Do not invent another manager for this distinction.

## Content lifetime and execution order

`device_tail.py` verifies the previous proposal and selects admissible context,
anchor and proposal mask. It then lowers DSpark metadata, publishes projected
target context into the draft's own SWA State (`dspark.py` context publication),
runs the draft, and masks invalid next-proposal rows to -1.

The draft attention's visible domain includes its fixed-width noncausal query
block plus retained context. Its persistent SWA lane is not a collection of
Qwen-style recurrent candidate checkpoints. Do not map it one-to-one onto GDN
convolution-window selection or accepted recurrent-row selection.

Continuation `reset` initializes generation/phase/cursors, fills proposals and
block tables with -1, and clears wave sequence/poison. Both
`_initialize_live_generation` and `_rebind_live_state` call reset before final
use/capture. Prefill commit writes the new admitted slot generation, proposal,
anchor and block table under a mask. Continuing commit writes accepted progress
and the next proposal under a qualified resident mapping, then sets DONE or
CONTINUE. Terminal is a semantic phase transition, not physical State release.

Root close rejects active invocations, retires graph users, runs generation
release hooks, unbinds State, and releases the realization. These are existing
root lifecycle laws, not bespoke DSpark destruction. Request turnover should
not be described as allocation destruction, nor as necessarily zeroing the
entire numerical SWA lane. Visibility/addressing and authoritative generation
must prevent stale contents from being consumed.

## Consequence for BetterScale Qwen35 research

Correct the previous emphasis: begin with module-local declarations and the
existing composed root lifecycle, not a new centralized three-domain manager.

- GDN numerical layers own convolution/recurrent State declarations.
- The MTP/continuation owner owns cross-step selection, anchor and progress;
  exact placement follows the real consumer/writer, not the word "candidate".
- Retained GDN checkpoints need explicit semantic ownership and a retention/
  restore protocol separate from live request seats. DSpark's SWA/proposal
  example does not itself supply that GDN checkpoint policy.
- Reuse existing domains, capacity units and backend admission to couple costs
  without merging address spaces. No need to recreate activation/close.

What remains Qwen-specific is candidate selection, checkpoint materialization,
+1-token prefix identity and restore-to-writable-state ordering. What is already
answered by LiveInference is how module-owned State joins allocation and root
lifetime without the donor's global union-storage owner.

Read-only investigation; no source/runtime edits or test execution in the
reference repository, and no device experiment. This comparison is not a
claim of a qualified Qwen35 migration.
